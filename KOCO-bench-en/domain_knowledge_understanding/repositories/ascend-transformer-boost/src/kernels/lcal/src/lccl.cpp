/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "lccl.h"
#include "lcal_internal.h"

#include <chrono>
#include <mutex>
#include <thread>
#include <acl/acl.h>

#include <mki/utils/log/log.h>
#include <mki/utils/env/env.h>
#include <mki/utils/dl/dl.h>

#include "profiling/report_timing.h"

using namespace std;
using namespace chrono;
using namespace Mki;

namespace Lcal {
using AclrtGetResInCurrentThreadFunc = int(*)(int, uint32_t *);

int GetAclResInCurThread(int type, uint32_t &resource)
{
    static std::once_flag onceFlag;
    static std::atomic<int> initFlag{LCAL_ERROR_NOT_INITIALIZED};  // -1
    static std::unique_ptr<Mki::Dl> mkiDl; // 持久保存，避免库被卸载
    static AclrtGetResInCurrentThreadFunc aclFn = nullptr;

    std::call_once(onceFlag, []() {
        std::string p;
        const char *c = Mki::GetEnv("ASCEND_HOME_PATH");
        if (c) {
            p = std::string(c) + "/runtime/lib64/libascendcl.so";
        } else {
            p = "libascendcl.so";
        }
        auto dl = std::make_unique<Mki::Dl>(p, false);
        if (!dl->IsValid()) {
            MKI_LOG(ERROR) << "Try load libascendcl.so failed: " << p;
            initFlag.store(LCAL_ERROR_NOT_FOUND, std::memory_order_release);
            return;
        }
        auto sym = dl->GetSymbol("aclrtGetResInCurrentThread");
        if (sym == nullptr) {
            MKI_LOG(WARN) << "Symbol aclrtGetResInCurrentThread not found in: " << p;
            initFlag.store(LCAL_ERROR_NOT_FOUND, std::memory_order_release);
            return;
        }
        mkiDl = std::move(dl); // 保留句柄，防止卸载
        aclFn = reinterpret_cast<AclrtGetResInCurrentThreadFunc>(sym);
        initFlag.store(LCAL_SUCCESS, std::memory_order_release);
        MKI_LOG(DEBUG) << "Loaded libascendcl.so and resolved aclrtGetResInCurrentThread from: " << p;
    });

    // 初始化结果判定
    int rc = initFlag.load(std::memory_order_acquire);
    if (rc != LCAL_SUCCESS) {
        return rc;
    }

    if (type != ACL_RT_DEV_RES_CUBE_CORE && type != ACL_RT_DEV_RES_VECTOR_CORE) {
        MKI_LOG(ERROR) << "aclrtGetResInCurrentThread not support resource type: " << type;
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }

    // 调用前检查函数指针有效性
    if (aclFn == nullptr) {
        MKI_LOG(ERROR) << "aclrtGetResInCurrentThread function pointer is null.";
        return LCAL_ERROR_INTERNAL;
    }

    // 调用底层函数
    const int ret = aclFn(type, &resource);
    if (ret != ACL_SUCCESS) {
        MKI_LOG(ERROR) << "aclrtGetResInCurrentThread failed. type: " << type << " err: " << ret;
        return LCAL_ERROR_INTERNAL;
    }

    MKI_LOG(DEBUG) << "Got resource in current thread. type: " << type << " resource: " << resource;
    return LCAL_SUCCESS;
}

uint32_t GetLocalReduceBlockDum(int64_t dataSize)
{
    constexpr int oneDataSize = 190 * 1024;
    constexpr int maxBlockDim = 8;
    int blockDim = dataSize / oneDataSize + 1;
    return blockDim <= maxBlockDim ? blockDim : maxBlockDim;
}

bool GetParallel()
{
    static int parallel = -1;
    if (parallel == -1) {
        static const char *ENV = Mki::GetEnv("LCCL_PARALLEL");
        parallel = (ENV && (string(ENV) == "1" || string(ENV) == "true")) ? 1 : 0;
        MKI_LOG(INFO) << "LCCL_PARALLEL is " << parallel;
    }
    return static_cast<bool>(parallel);
}

uint32_t GetAllReduceDetermBlockNum(uint32_t rankSize, int64_t dataSize, uint32_t extraFlag)
{
    constexpr uint32_t quickOneshotRankSize = 2;
    constexpr uint32_t twoBlockNum = 2;
    constexpr uint32_t threeStepNum = 3;
    constexpr uint32_t rankSize910a3 = 16;
    constexpr uint32_t dbRingBlockNum = 34;
    constexpr int64_t smallDataSize = 1 * 1024 * 1024;
    constexpr int32_t smallDataSize910a3 = 32 * 1024 * 1024;
    if ((extraFlag & ExtraFlag::TOPO_910_93) != 0) {
        constexpr uint32_t maxAivNum = 40;
        const bool isAivNumSupport = ((extraFlag & ExtraFlag::IS_GREATER_THAN_40_AIV) != 0 ||
            rankSize * threeStepNum <= maxAivNum);
        if (rankSize % quickOneshotRankSize == 1 || rankSize == quickOneshotRankSize ||
            (rankSize <= rankSize910a3 && dataSize <= smallDataSize910a3 && isAivNumSupport)) {
            return rankSize * threeStepNum;
        } else {
            return dbRingBlockNum;
        }
    }
    if (dataSize < smallDataSize) {
        return rankSize * twoBlockNum;
    }
    return rankSize * threeStepNum;
}

uint32_t GetAllReduceBlockNum(uint32_t rankSize, int64_t dataSize, uint32_t extraFlag)
{
    constexpr uint32_t twoBlockNum = 2;
    constexpr uint32_t threeStepNum = 3;
    constexpr uint32_t dbRingBlockNum = 34;
    constexpr int64_t smallDataSize = 1 * 1024 * 1024;
    constexpr uint32_t smallRankSize = 8;
    constexpr uint32_t cceSmallDataSize = 2 * 1024 * 1024;
    constexpr uint32_t quickOneshotRankSize = 2;
    const int64_t quantSmallDataSize = ((extraFlag & ExtraFlag::QUANT_FP16) != 0) ? (smallDataSize / 2) : smallDataSize;
    constexpr int32_t smallDataSize910a3 = 32 * 1024 * 1024;

    if ((extraFlag & ExtraFlag::TOPO_PCIE) != 0) {
        return rankSize * twoBlockNum;
    } else if ((extraFlag & ExtraFlag::QUANT_FP16) != 0) {
        return dataSize <= quantSmallDataSize ? rankSize : rankSize * twoBlockNum;
    } else if ((extraFlag & ExtraFlag::TOPO_910B2C) != 0 && rankSize > smallRankSize) {
        return dataSize < cceSmallDataSize ? rankSize : (rankSize / twoBlockNum * threeStepNum + twoBlockNum);
    } else if ((extraFlag & ExtraFlag::DETERMINISTIC) != 0) {
        return GetAllReduceDetermBlockNum(rankSize, dataSize, extraFlag);
    }

    if (GetParallel()) {
        return rankSize;
    }

    if ((extraFlag & ExtraFlag::TOPO_910_93) != 0 && dataSize > smallDataSize910a3 &&
        (rankSize != quickOneshotRankSize)) {
        return rankSize % quickOneshotRankSize == 0 ? dbRingBlockNum : rankSize * threeStepNum;
    }
    return (rankSize == quickOneshotRankSize || dataSize >= cceSmallDataSize) ? rankSize * twoBlockNum : rankSize;
}

uint32_t GetReduceScatterBlockNum(uint32_t rankSize, int64_t dataSize, uint32_t extraFlag)
{
    constexpr uint32_t twoBlockNum = 2;
    constexpr int64_t smallDataSize = 1 * 1024 * 1024;
    constexpr uint32_t quickOneshotRankSize = 2;
    constexpr int64_t cceSmallDataSize = 2 * 1024 * 1024;
    constexpr int64_t a3BigDataSize = 32 * 1024 * 1024;
    constexpr uint32_t fourStepBlockNum = 34;
    constexpr uint32_t a3SupportRankSize = 4;
    constexpr uint32_t smallRankSize = 8;
    constexpr uint32_t dbRingBlockNum = 36;

    const bool isDbRing = (rankSize == a3SupportRankSize || rankSize == smallRankSize) &&
        (dataSize * smallRankSize > cceSmallDataSize && dataSize * smallRankSize <= a3BigDataSize);

    if ((extraFlag & ExtraFlag::TOPO_910_93) != 0 && (rankSize > smallRankSize || isDbRing)) {
        if (isDbRing) {
            return dbRingBlockNum;
        } else {
            return dataSize <= smallDataSize ? rankSize : fourStepBlockNum;
        }
    } else {
        return (rankSize == quickOneshotRankSize || dataSize >= cceSmallDataSize) ? rankSize * twoBlockNum : rankSize;
    }
}

uint32_t GetAll2AllBlockNum(uint32_t rankSize, int64_t dataSize, uint32_t extraFlag)
{
    constexpr uint32_t twoStepBlockNum = 16;
    constexpr uint32_t twoBlockNum = 2;
    constexpr int64_t smallDataSize = 1 * 1024 * 1024;
    constexpr uint32_t smallRankSize = 8;

    if ((extraFlag & ExtraFlag::TOPO_910_93) != 0) {
        if (rankSize <= smallRankSize && dataSize > smallDataSize &&
            dataSize % (smallRankSize * smallRankSize * rankSize) == 0) {
            return twoStepBlockNum * twoBlockNum;
        } else {
            return rankSize <= twoStepBlockNum ? rankSize * twoBlockNum : twoStepBlockNum * twoBlockNum;
        }
    }
    return rankSize * twoBlockNum;
}


uint32_t GetAllGatherBlockNum(uint32_t rankSize, int64_t dataSize, uint32_t extraFlag)
{
    constexpr uint32_t axRankSize = 16;
    constexpr uint32_t twoBlockNum = 2;
    constexpr uint32_t quickOneshotRankSize = 2;
    constexpr uint32_t allGatherHDBRingBlockNum = 32;
    constexpr uint32_t cceSmallDataSize = 2 * 1024 * 1024;
    constexpr int64_t smallDataSize910a3 = 32 * 1024 * 1024;
    constexpr uint32_t smallRankSize = 8;

    if ((extraFlag & ExtraFlag::TOPO_910B2C) != 0 && (rankSize == axRankSize)) {
        constexpr uint32_t axBlockNum = 10;
        return axBlockNum;
    } else if ((extraFlag & ExtraFlag::TOPO_PCIE) != 0) {
        return rankSize * twoBlockNum;
    }

    if (GetParallel()) {
        return rankSize;
    }

    if ((extraFlag & ExtraFlag::TOPO_910_93) != 0 &&
        (dataSize > smallDataSize910a3 || rankSize > smallRankSize) &&
        rankSize > quickOneshotRankSize && rankSize % quickOneshotRankSize == 0) {
        return allGatherHDBRingBlockNum;
    }
    return (rankSize == quickOneshotRankSize || dataSize >= cceSmallDataSize) ? rankSize * twoBlockNum : rankSize;
}

uint32_t GetKernelBlockNum(LcalType cclType, uint32_t rankSize, int64_t dataSize, int localRankSize, uint32_t extraFlag)
{
    constexpr uint32_t twoStepBlockNum = 16;
    constexpr uint32_t twoBlockNum = 2;
    constexpr int64_t smallDataSize = 1 * 1024 * 1024;
    constexpr uint32_t gatherDefaultBlockNum = 4;
    const uint32_t rankSizeLocal = static_cast<uint32_t>(localRankSize);

    if (cclType == LcalType::LOCAL_REDUCE) {
        return GetLocalReduceBlockDum(dataSize);
    }

    if (cclType == LcalType::BROADCAST) {
        return rankSize;
    }

    if (cclType == LcalType::ALL2ALL_V_C) {
        return twoStepBlockNum * twoBlockNum;
    }
    if (cclType == LcalType::ALL2ALL) {
        return GetAll2AllBlockNum(rankSize, dataSize, extraFlag);
    }
    if (cclType == LcalType::BANDWIDTH) {
        return twoStepBlockNum * twoBlockNum;
    }
    if (cclType == LcalType::ALL_REDUCE) {
        return GetAllReduceBlockNum(rankSize, dataSize, extraFlag);
    }
    if (cclType == LcalType::REDUCE_SCATTER) {
        return GetReduceScatterBlockNum(rankSize, dataSize, extraFlag);
    }
    if (cclType == LcalType::ALL_GATHER) {
        return GetAllGatherBlockNum(rankSize, dataSize, extraFlag);
    }
    if (cclType == LcalType::GATHER) {
        return gatherDefaultBlockNum;
    }
    bool sendOrRecv = cclType == LcalType::RECV || cclType == LcalType::SEND;
    if (sendOrRecv) {
        return dataSize <= smallDataSize ? rankSizeLocal : rankSizeLocal * twoBlockNum;
    }
    return twoStepBlockNum;
}

uint32_t Lccl::GetBlockNum(LcalType cclType, uint32_t rankSize, int64_t dataSize,
                           int localRankSize, uint32_t extraFlag) const
{
    if (comm_ == nullptr) {
        MKI_LOG(ERROR) << "comm is nullptr " << __LINE__;
        return 0;
    }
    uint32_t limitVal = 0;
    aclrtDevResLimitType limitType = aclrtDevResLimitType::ACL_RT_DEV_RES_VECTOR_CORE;
    uint32_t blockNum = GetKernelBlockNum(cclType, rankSize, dataSize, localRankSize, extraFlag);
    if (comm_->isEnableMix_) {
        constexpr uint32_t aivNumPerAic = 2;
        if (blockNum % aivNumPerAic == 1) {
            MKI_LOG(ERROR) << "Lccl not support odd block number at msprof op enabled!";
            return 0;
        }
        blockNum = blockNum / aivNumPerAic;
        limitType = aclrtDevResLimitType::ACL_RT_DEV_RES_CUBE_CORE;
    }

    int res = GetAclResInCurThread(static_cast<int>(limitType), limitVal);
    if (res == LCAL_SUCCESS) {
        MKI_LOG(DEBUG) << "Required blockNum(" << blockNum <<
            ") limit:(limitVal=" << limitVal << ", limitType=" << static_cast<int>(limitType) << ")";
        if (blockNum > limitVal) {
            MKI_LOG(ERROR) << "Insufficient blockDim: Required blockNum(" << blockNum <<
                ") exceeds limit (limitVal=" << limitVal << ", limitType=" << static_cast<int>(limitType) << ")";
            return 0;
        }
    }
    return blockNum;
}

int Lccl::LoopBack(const void *sendBuff, void *recvBuff, int64_t count, HcclDataType dataType, aclrtStream stream) const
{
    if (sendBuff != recvBuff) {
        auto ret = aclrtMemcpyAsync(recvBuff, Count2Size(count, dataType), sendBuff, Count2Size(count, dataType),
                                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        if (ret != 0) {
            MKI_LOG(ERROR) << "LoopBack failed!";
            return LCAL_ERROR_INTERNAL;
        }
    }
    return LCAL_SUCCESS;
}

int Lccl::AllReduce(void *sendBuff, void *recvBuff, int64_t count, HcclDataType dataType, HcclReduceOp op,
    aclrtStream stream, HcclDataType outputDataType, const void *scale, int64_t scaleCount, const void *offset) const
{
    if (!CheckBuff(sendBuff, recvBuff)) {
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    if (!CheckDataType(dataType) || op == HCCL_REDUCE_PROD ||
        (outputDataType != HCCL_DATA_TYPE_RESERVED && !CheckDataType(outputDataType))) {
        MKI_LOG(ERROR) << "Lccl not support.";
        return LCAL_ERROR_NOT_INITIALIZED;
    }
    std::unique_ptr<ReportTiming> report;
    if (comm_->isEnableMsprofOp_) {
        report = std::make_unique<ReportTiming>("LcclAllReduce", comm_->rank_, true,
            comm_->commArgs_.dumpAddr, stream);
    } else {
        report = std::make_unique<ReportTiming>("LcclAllReduce", comm_->commDomain_, count, dataType);
    }
    if ((dataType == HCCL_DATA_TYPE_INT8 && outputDataType == HCCL_DATA_TYPE_FP16) !=
        static_cast<bool>(comm_->commArgs_.extraFlag & ExtraFlag::QUANT_FP16)) {
        if (dataType == HCCL_DATA_TYPE_INT8 && outputDataType == HCCL_DATA_TYPE_FP16) {
            comm_->commArgs_.extraFlag |= ExtraFlag::QUANT_FP16;
        } else {
            comm_->commArgs_.extraFlag &= ~ExtraFlag::QUANT_FP16;
        }

        auto ret = aclrtMemcpyAsync(comm_->commArgsPtr_, sizeof(CommArgs), &(comm_->commArgs_), sizeof(CommArgs),
            ACL_MEMCPY_HOST_TO_DEVICE, stream);
        if (ret != ACL_SUCCESS) {
            MKI_LOG(ERROR) << "aclrtMemcpy err " << __LINE__ << " " << ret;
            return LCAL_ERROR_INTERNAL;
        }
    }

    if ((comm_->commArgs_.extraFlag & ExtraFlag::QUANT_FP16) != 0 &&
        (comm_->commArgs_.extraFlag & (ExtraFlag::QUANT_DELAY | ExtraFlag::QUANT_CURRENT)) == 0) {
        uint32_t blockDim = GetBlockNum(LcalType::ALL_REDUCE, rankSize_, Count2Size(count, dataType),
                                        comm_->localRankSize_, comm_->commArgs_.extraFlag);
        AscendCCLKernelArgs ascendArgs = {sendBuff, recvBuff, comm_->commArgsPtr_, count, comm_->magic_, op, 0, 0,
            scale, scaleCount, offset};
        comm_->magic_++;
        return LoadMTE(LcalType::ALL_REDUCE, ascendArgs, blockDim, dataType, stream);
    }

    if (rankSize_ <= 1) {
        return LoopBack(sendBuff, recvBuff, count, dataType, stream);
    }

    if ((comm_->commArgs_.extraFlag & (ExtraFlag::QUANT_DELAY | ExtraFlag::QUANT_CURRENT)) != 0) {
        uint32_t blockDim = GetBlockNum(LcalType::ALL_REDUCE, rankSize_, Count2Size(count, dataType),
                                        comm_->localRankSize_, comm_->commArgs_.extraFlag);
        AscendCCLKernelArgs args = { sendBuff, recvBuff, comm_->commArgsPtr_, count, comm_->magic_, op, 0, 0, scale,
            scaleCount};
        comm_->magic_++;
        return LoadMTE(LcalType::ALL_REDUCE, args, blockDim, dataType, stream);
    }

    uint32_t blockDim = GetBlockNum(LcalType::ALL_REDUCE, rankSize_, Count2Size(count, dataType),
                                    comm_->localRankSize_, comm_->commArgs_.extraFlag);
    AscendCCLKernelArgs args = {sendBuff, recvBuff, comm_->commArgsPtr_, count, comm_->magic_, op, 0};
    comm_->magic_++;
    return LoadMTE(LcalType::ALL_REDUCE, args, blockDim, dataType, stream);
}

bool Lccl::CheckDataType(const HcclDataType &dataType) const
{
    return (dataType == HCCL_DATA_TYPE_INT32 or dataType == HCCL_DATA_TYPE_FP16 or dataType == HCCL_DATA_TYPE_FP32 or
            dataType == HCCL_DATA_TYPE_INT8 or dataType == HCCL_DATA_TYPE_INT16 or dataType == HCCL_DATA_TYPE_BFP16 or
            dataType == HCCL_DATA_TYPE_INT64);
}

bool Lccl::CheckBuff(const void *sendBuff, const void *recvBuff) const
{
    bool res = true;
    if (sendBuff == nullptr) {
        MKI_LOG(ERROR) << "Lccl sendBuff is nullptr";
        res = false;
    } else if (recvBuff == nullptr) {
        MKI_LOG(ERROR) << "Lccl recvBuff is nullptr";
        res = false;
    } else if (comm_ == nullptr) {
        MKI_LOG(ERROR) << "comm is nullptr" << __LINE__;
        res = false;
    }
    return res;
}

int Lccl::ReduceScatter(void *sendBuff, void *recvBuff, int64_t count, HcclDataType dataType, HcclReduceOp op,
    aclrtStream stream) const
{
    if (!CheckBuff(sendBuff, recvBuff)) {
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    if (rankSize_ <= 1) {
        return LoopBack(sendBuff, recvBuff, count, dataType, stream);
    }
    std::unique_ptr<ReportTiming> report;
    if (comm_->isEnableMsprofOp_) {
        report = std::make_unique<ReportTiming>("LcclReduceScatter", comm_->rank_, true,
            comm_->commArgs_.dumpAddr, stream);
    } else {
        report = std::make_unique<ReportTiming>("LcclReduceScatter", comm_->commDomain_, count, dataType);
    }
    if (CheckDataType(dataType) and op != HCCL_REDUCE_PROD) {
        uint32_t blockDim = GetBlockNum(LcalType::REDUCE_SCATTER, rankSize_, Count2Size(count, dataType),
                                        comm_->localRankSize_, comm_->commArgs_.extraFlag);
        AscendCCLKernelArgs args = { sendBuff, recvBuff, comm_->commArgsPtr_, count, comm_->magic_, op, 0 };
        comm_->magic_++;
        return LoadMTE(LcalType::REDUCE_SCATTER, args, blockDim, dataType, stream);
    }
    MKI_LOG(ERROR) << "Lccl not support.";
    return LCAL_ERROR_NOT_INITIALIZED;
}

int Lccl::AllGather(void *sendBuff, void *recvBuff, int64_t count, HcclDataType dataType, aclrtStream stream) const
{
    if (!CheckBuff(sendBuff, recvBuff)) {
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    if (rankSize_ <= 1) {
        return LoopBack(sendBuff, recvBuff, count, dataType, stream);
    }
    std::unique_ptr<ReportTiming> report;
    if (comm_->isEnableMsprofOp_) {
        report = std::make_unique<ReportTiming>("LcclAllGather", comm_->rank_, true,
            comm_->commArgs_.dumpAddr, stream);
    } else {
        report = std::make_unique<ReportTiming>("LcclAllGather", comm_->commDomain_, count, dataType);
    }
    AscendCCLKernelArgs args = { sendBuff, recvBuff, comm_->commArgsPtr_, count, comm_->magic_, 0, 0 };
    comm_->magic_++;
    uint32_t blockDim = GetBlockNum(LcalType::ALL_GATHER, rankSize_, Count2Size(count, dataType),
                                    comm_->localRankSize_, comm_->commArgs_.extraFlag);
    return LoadMTE(LcalType::ALL_GATHER, args, blockDim, dataType, stream);
}

int Lccl::All2All(void *sendBuff, void *recvBuff, int64_t count, HcclDataType dataType, aclrtStream stream) const
{
    constexpr int32_t supportRankNum = 2;
    if (!CheckBuff(sendBuff, recvBuff) || (rankSize_ > 1 && rankSize_ % supportRankNum != 0)) {
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    if (rankSize_ <= 1) {
        return LoopBack(sendBuff, recvBuff, count, dataType, stream);
    }
    ReportTiming report("LcclAll2All", comm_->commDomain_, count, dataType);
    AscendCCLKernelArgs args = { sendBuff, recvBuff, comm_->commArgsPtr_, count, comm_->magic_, 0, 0, 0 };
    comm_->magic_++;
    uint32_t blockDim = GetBlockNum(LcalType::ALL2ALL, rankSize_, Count2Size(count, dataType),
                                    comm_->localRankSize_, comm_->commArgs_.extraFlag);
    return LoadMTE(LcalType::ALL2ALL, args, blockDim, dataType, stream);
}

int Lccl::All2All(void *sendBuff, void *recvBuff, int64_t count, int32_t burstLen,
    int32_t stride, HcclDataType dataType, aclrtStream stream) const
{
    if (!CheckBuff(sendBuff, recvBuff)) {
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    if (rankSize_ <= 1) {
        return LoopBack(sendBuff, recvBuff, count, dataType, stream);
    }
    ReportTiming report("LcclAll2AllTranspose", comm_->commDomain_, count, dataType);

    AscendCCLKernelArgs args = { sendBuff, recvBuff, comm_->commArgsPtr_, count, comm_->magic_, burstLen, stride};
    comm_->magic_++;
    uint32_t blockDim = GetBlockNum(LcalType::ALL2ALL, rankSize_, Count2Size(count, dataType),
                                    comm_->localRankSize_, comm_->commArgs_.extraFlag);
    return LoadMTE(LcalType::ALL2ALL, args, blockDim, dataType, stream);
}

int64_t GetSizeByHcclDataType(const HcclDataType &dataType)
{
    int64_t dataSize = sizeof(int);
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8:
            dataSize = sizeof(int8_t);
            break;
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_BFP16:
        case HCCL_DATA_TYPE_UINT16:
            dataSize = sizeof(int16_t);
            break;
        case HCCL_DATA_TYPE_FP32:
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_UINT32:
            dataSize = sizeof(int32_t);
            break;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_UINT64:
            dataSize = sizeof(int64_t);
            break;
        default:
            MKI_LOG(ERROR) << "unknown datatype";
    }
    return dataSize;
}

int Lccl::Broadcast(void *buff, int64_t count, HcclDataType dataType, int32_t root, aclrtStream stream) const
{
    constexpr int supportRankSize = 8;
    if (rankSize_ <= 1) {
        return LCAL_SUCCESS;
    }
    if (rankSize_ > supportRankSize) {
        MKI_LOG(ERROR) << "Broadcast does not support ranksize over 8";
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    if (!CheckBuff(buff, buff)) {
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    ReportTiming report("LcclBroadcast", comm_->commDomain_, count, dataType);
    AscendCCLKernelArgs args = { buff, buff, comm_->commArgsPtr_, count, comm_->magic_, 0, root };
    comm_->magic_++;
    uint32_t blockDim = GetBlockNum(LcalType::BROADCAST, rankSize_, Count2Size(count, dataType),
                                    comm_->localRankSize_, comm_->commArgs_.extraFlag);
    return LoadMTE(LcalType::BROADCAST, args, blockDim, dataType, stream);
}

Lccl::~Lccl()
{
    if (rankSize_ == -1 and comm_ != nullptr) {
        delete comm_;
    }
}

Lccl::Lccl(LcalComm *comm) : comm_(comm)
{
    if (comm != nullptr) {
        rank_ = comm->rank_;
        rankSize_ = comm->rankSize_;
    } else {
        MKI_LOG(ERROR) << "comm is nullptr.";
        comm_ = new (std::nothrow) LcalComm(0, 0);
        if (comm_ == nullptr) {
            MKI_LOG(ERROR) << "LcalComm create failed " << __LINE__;
        }
        rankSize_ = -1;
    }
}

Lccl::Lccl(LcalComm &comm) : comm_(&comm)
{
    rank_ = comm.rank_;
    rankSize_ = comm.rankSize_;
}
}