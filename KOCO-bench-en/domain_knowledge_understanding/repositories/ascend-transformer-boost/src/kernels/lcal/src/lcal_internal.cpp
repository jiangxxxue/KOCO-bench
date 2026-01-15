/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "lcal_internal.h"
#include <map>
#include <mutex>
#include <vector>
#include <mki/utils/log/log.h>
#include <mki/utils/env/env.h>
#include <runtime/kernel.h>
#include "ccl_kernel_args.h"
#include "coc_kernel_args.h"
#include "lcoc.h"

using namespace std;
using namespace Mki;

extern const int LCAL_CCE_BIN_STR[];
asm(R"(.section .rodata, "a", @progbits
LCAL_CCE_BIN_STR:.incbin "lcal_cce.o"
.byte 0
.previous)");

constexpr int LCCL_RT_DEV_BINARY_MAGIC_ELF_AIVEC = 0x41415246;
constexpr int COC_RT_DEV_BINARY_MAGIC_ELF = 0x43554245;

namespace Lcal {
const std::map<HcclDataType, std::string> DATATYPE2NAME = {
    { HCCL_DATA_TYPE_INT32, "int" },
    { HCCL_DATA_TYPE_INT16, "int16_t" },
    { HCCL_DATA_TYPE_INT8, "int8_t" },
    { HCCL_DATA_TYPE_INT64, "int64_t" },
    { HCCL_DATA_TYPE_FP32, "float" },
    { HCCL_DATA_TYPE_FP16, "float16_t" },
    { HCCL_DATA_TYPE_BFP16, "bfloat16_t" }
};

template<class T>
int RegisterBinaryKernel(const string &funcName, int8_t *funSig, const T *binStrPtr, int magic, int len = 0)
{
    rtDevBinary_t binary;
    void *binHandle = nullptr;
    binary.data = binStrPtr;
    binary.length = (len == 0 ? LCAL_1OP_BIN_SIZE : len);

    binary.magic = magic;
    binary.version = 0;
    rtError_t rtRet = rtDevBinaryRegister(&binary, &binHandle);
    if (rtRet != RT_ERROR_NONE) {
        MKI_LOG(WARN) << "rtDevBinaryRegister failed! " << to_string(rtRet) << ", funcName = " << funcName;
        return LCAL_ERROR_INTERNAL;
    }
    rtRet = rtFunctionRegister(binHandle, funSig, funcName.c_str(), funcName.c_str(), 0);
    if (rtRet != RT_ERROR_NONE) {
        MKI_LOG(WARN) << "rtFunctionRegister failed! " << to_string(rtRet) << ", funcName = " << funcName;
        return LCAL_ERROR_INTERNAL;
    }
    return LCAL_SUCCESS;
}

int8_t *GetFunSig(LcalType type, HcclDataType dataType, uint64_t devType = 0)
{
    constexpr int sigOffset = 16;
    constexpr int sigSkew = 0x1000;
    return reinterpret_cast<int8_t *>((static_cast<uint64_t>(type) << sigOffset << sigOffset) +
        (static_cast<uint64_t>(dataType)<< sigOffset) + devType + sigSkew);
}

const int* FindNextOpStart(const int opStartMaigc, const int* cclBinEndPtr, const int* cclBinPtr)
{
    if (cclBinPtr == nullptr) {
        MKI_LOG(ERROR) << "FindNextOpStart failed! cclBinPtr is nullptr";
        return nullptr;
    }
    while (cclBinPtr < cclBinEndPtr && *cclBinPtr != opStartMaigc) {
        cclBinPtr++;
    }
    if (*cclBinPtr == opStartMaigc) {
        cclBinPtr++;
    }
    return cclBinPtr;
}

int RegistCCLOp2Kernel(const int* cclBinPtr, const int* nextPtr)
{
    vector<HcclDataType> registerTypes = { HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT8,
                                           HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_BFP16,
                                           HCCL_DATA_TYPE_INT64 };
    std::vector<LcalType> registerCCLTypesOp2 = { // 完成算子实现后在这里添加算子注册
        LcalType::ALL_GATHER, LcalType::REDUCE_SCATTER, LcalType::ALL2ALL,
    };
    int res = LCAL_SUCCESS;
    for (auto ccl : registerCCLTypesOp2) {
        for (auto t : registerTypes) {
            res = RegisterBinaryKernel(LCAL_TYPE2NAME.at(ccl) + "_" + DATATYPE2NAME.at(t), GetFunSig(ccl, t),
                cclBinPtr, LCCL_RT_DEV_BINARY_MAGIC_ELF_AIVEC, (nextPtr - cclBinPtr) * sizeof(int));
        }
    }
    if (res != LCAL_SUCCESS) {
        return res;
    }
    res = RegisterBinaryKernel(LCAL_TYPE2NAME.at(LcalType::BROADCAST),
        GetFunSig(LcalType::BROADCAST, HCCL_DATA_TYPE_RESERVED), cclBinPtr, LCCL_RT_DEV_BINARY_MAGIC_ELF_AIVEC);
    return res;
}

int RegistCCLOp1Kernel(const int* cclBinPtr, const int* nextPtr)
{
    vector<HcclDataType> registerTypes = { HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT8,
                                           HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_BFP16,
                                           HCCL_DATA_TYPE_INT64 };
    std::vector<LcalType> registerCCLTypesOp1 = { // 完成算子实现后在这里添加算子注册
        LcalType::ALL_REDUCE,
    };
    int res = LCAL_SUCCESS;
    for (auto ccl : registerCCLTypesOp1) {
        for (auto t : registerTypes) {
            res = RegisterBinaryKernel(LCAL_TYPE2NAME.at(ccl) + "_" + DATATYPE2NAME.at(t), GetFunSig(ccl, t),
                cclBinPtr, LCCL_RT_DEV_BINARY_MAGIC_ELF_AIVEC, (nextPtr - cclBinPtr) * sizeof(int));
        }
    }
    return res;
}

int RegistCCLKernel(const int32_t opGroup)
{
    const int* cclBinStr = LCAL_CCE_BIN_STR;
    auto cclBinEndPtr = cclBinStr + LCAL_1OP_BIN_SIZE / sizeof(int);
    const int* cclBinPtr = cclBinStr + 1;
    constexpr int opStartMaigc = 0x44444444;
    const int* nextPtr = FindNextOpStart(opStartMaigc, cclBinEndPtr, cclBinPtr);
    if (nextPtr == nullptr) {
        return LCAL_ERROR_INTERNAL;
    }

    constexpr int32_t smallGroupNum = 2;
    for (int32_t opGroupIdx = 0; opGroupIdx < opGroup; ++opGroupIdx) {
        for (int32_t opIdx = 0; opIdx < smallGroupNum; ++opIdx) {
            cclBinPtr = nextPtr;
            nextPtr = FindNextOpStart(opStartMaigc, cclBinEndPtr, nextPtr);
            if (cclBinPtr == nullptr || cclBinPtr == cclBinEndPtr || nextPtr == nullptr) {
                return LCAL_ERROR_INTERNAL;
            }
        }
    }

    int ret = 0;
    ret = RegistCCLOp1Kernel(cclBinPtr, nextPtr);
    if (ret != LCAL_SUCCESS) {
        return LCAL_ERROR_INTERNAL;
    }

    // 切换到大组内第二个小组是
    cclBinPtr = nextPtr;
    nextPtr = FindNextOpStart(opStartMaigc, cclBinEndPtr, nextPtr);
    if (cclBinPtr == nullptr || cclBinPtr == cclBinEndPtr || nextPtr == nullptr) {
        return LCAL_ERROR_INTERNAL;
    }

    // 大组内第二个小组是 reducescatter, allgather 等
    ret = RegistCCLOp2Kernel(cclBinPtr, nextPtr);
    if (ret != LCAL_SUCCESS) {
        return LCAL_ERROR_INTERNAL;
    }
    return LCAL_SUCCESS;
}

void RegistCoCKernel()
{
    vector<HcclDataType> registerTypes = { HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_BFP16 };
    vector<vector<LcalType>> registerCOCTypes = {
        { LcalType::PURE_MATMUL},
        { LcalType::MATMUL_ALL_REDUCE },
        { LcalType::MATMUL_REDUCE_SCATTER },
        { LcalType::ALL_GATHER_MATMUL, LcalType::ALL_GATHER_MATMUL_V2 },
        { LcalType::ALL_GATHER_MATMUL_REDUCE_SCATTER},
        { LcalType::ALLTOALLV_ALLGATHER_MATMUL, LcalType::ALLTOALLVC_ALLGATHER_MATMUL_HIDDEN},
        { LcalType::MATMUL_REDUCESCATTER_ALLTOALLVC_HIDDEN},
    };

    auto cocCceBinStr = LCAL_CCE_BIN_STR + LCAL_1OP_BIN_SIZE / sizeof(int);
    for (auto lcalTypeGroup : registerCOCTypes) {
        for (auto lcalType : lcalTypeGroup) {
            for (auto t : registerTypes) {
                RegisterBinaryKernel(LCAL_TYPE2NAME.at(lcalType) + "_" + DATATYPE2NAME.at(t), GetFunSig(lcalType, t),
                    cocCceBinStr, COC_RT_DEV_BINARY_MAGIC_ELF);
            }
        }
        cocCceBinStr += LCAL_1OP_BIN_SIZE / sizeof(int);
    }
}

int RegistKernel(const int32_t opGroup)
{
    static bool init = false;
    static mutex mut;
    lock_guard<mutex> guard(mut);
    if (init) {
        return 0;
    }
    RegistCoCKernel();
    RegistCCLKernel(opGroup);
    init = true;
    return LCAL_SUCCESS;
}

int64_t Count2Size(int64_t count, const HcclDataType &dataType)
{
    int64_t dataSize = LCAL_INVALID_VALUE;
    if (dataType == HCCL_DATA_TYPE_INT8 || dataType == HCCL_DATA_TYPE_UINT8) {
        dataSize = count;
    } else if (dataType == HCCL_DATA_TYPE_INT16 || dataType == HCCL_DATA_TYPE_FP16 ||
               dataType == HCCL_DATA_TYPE_BFP16 || dataType == HCCL_DATA_TYPE_UINT16) {
        dataSize = count * sizeof(int16_t);
    } else if (dataType == HCCL_DATA_TYPE_FP32 || dataType == HCCL_DATA_TYPE_INT32 ||
               dataType == HCCL_DATA_TYPE_UINT32) {
        dataSize = count * sizeof(int32_t);
    } else if (dataType == HCCL_DATA_TYPE_INT64 || dataType == HCCL_DATA_TYPE_UINT64) {
        dataSize = count * sizeof(int64_t);
    } else {
        MKI_LOG(ERROR) << "unknown datatype";
    }
    return dataSize;
}

int LoadMTE(LcalType cclType, AscendCCLKernelArgs &args, uint32_t blockDim, HcclDataType dataType, aclrtStream stream)
{
    int error = 0;
    MKI_LOG(DEBUG) << "LoadMTE " << LCAL_TYPE2NAME.at(cclType) << " count:" << args.count << " dataType:" << dataType
                   << " op:" << args.op << " blockDim:" << blockDim << " rootRank:" << args.root
                   << ", magic: " << args.magic;
    int64_t dataSize = Count2Size(args.count, dataType);
    if (dataSize == LCAL_INVALID_VALUE || blockDim == 0) {
        MKI_LOG(ERROR) << ("LoadMTE args are invalid");
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }

    static const char *ENV = Mki::GetEnv("LCCL_PARALLEL");
    if (ENV && (string(ENV) == "1" || string(ENV) == "true") && dataSize >= IPC_BUFF_MAX_SIZE) {
        MKI_LOG(ERROR) << ("LoadMTE args are invalid, because LCCL_PARALLEL is open, and dataSize is too big.");
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }

    rtTaskCfgInfo_t cfgInfo{};
    cfgInfo.schemMode = 1;

    rtArgsEx_t argsInfo{};
    argsInfo.args = &args;
    argsInfo.argsSize = sizeof(args);

    if (cclType == LcalType::BROADCAST || cclType == LcalType::BANDWIDTH) {
        args.count = dataSize;
        error = rtKernelLaunchWithFlagV2(GetFunSig(cclType, HCCL_DATA_TYPE_RESERVED),
                                         blockDim, &argsInfo, nullptr, stream, 0, &cfgInfo);
    } else {
        error = rtKernelLaunchWithFlagV2(GetFunSig(cclType, dataType),
                                         blockDim, &argsInfo, nullptr, stream, 0, &cfgInfo);
    }
    if (error != RT_ERROR_NONE) {
        MKI_LOG(ERROR) << "AsdRtFunctionLaunch -:" << LCAL_TYPE2NAME.at(cclType) << to_string(error);
        return LCAL_ERROR_MKIRT;
    }
    return error;
}

int LoadMTE(LcalType cclType, CCLGatherArgs &args, uint32_t blockDim, HcclDataType dataType, aclrtStream stream)
{
    int error = 0;
    MKI_LOG(DEBUG) << "LoadMTE " << LCAL_TYPE2NAME.at(cclType) << " embTableLen:" << args.embTableLen
                   << " embTableDim:" << args.embTableDim
                   << " lookupLen:" << args.lookupLen;

    rtTaskCfgInfo_t cfgInfo{};
    cfgInfo.schemMode = 1;

    rtArgsEx_t argsInfo{};
    argsInfo.args = &args;
    argsInfo.argsSize = sizeof(args);

    if (cclType == LcalType::GATHER) {
        error = rtKernelLaunchWithFlagV2(GetFunSig(cclType, dataType),
                                         blockDim, &argsInfo, nullptr, stream, 0, &cfgInfo);
    }
    if (error != RT_ERROR_NONE) {
        MKI_LOG(ERROR) << "AsdRtFunctionLaunch -:" << to_string(error);
        return LCAL_ERROR_MKIRT;
    }
    return error;
}

template<typename T, typename M>
size_t OffsetOf(M T::*member, T obj)
{
    return reinterpret_cast<size_t>(&(obj.*member)) - reinterpret_cast<size_t>(&obj);
}

int ComputeOverComm(LcalType cocType, CoCKernelArgs kernelArgs, HcclDataType dataType, aclrtStream stream)
{
    int error = LCAL_SUCCESS;

    size_t tilingAddrOffset = OffsetOf(&CoCKernelArgs::pCocTiling, kernelArgs);
    size_t tilingDataOffset = OffsetOf(&CoCKernelArgs::cocKernelParam, kernelArgs) +
            OffsetOf(&CoCKernelParam::cocTilingData, kernelArgs.cocKernelParam);

    auto &cocTilingData = kernelArgs.cocKernelParam.cocTilingData;
    if (cocTilingData.withSerialMode != 0) {
        static std::vector<int64_t> serialTags(LCAL_MAX_RANK_SIZE, 1);
        cocTilingData.tag = serialTags[cocTilingData.rank];
        serialTags[cocTilingData.rank] = serialTags[cocTilingData.rank] % TAG_MOD + 1;
    }

    rtTaskCfgInfo_t cfgInfo{};
    cfgInfo.schemMode = 1;

    rtArgsEx_t argsInfo{};
    argsInfo.args = static_cast<void *>(&kernelArgs);
    argsInfo.hostInputInfoPtr = nullptr;
    argsInfo.argsSize = sizeof(kernelArgs);
    argsInfo.tilingAddrOffset = tilingAddrOffset;
    argsInfo.tilingDataOffset = tilingDataOffset;
    argsInfo.hostInputInfoNum = 0;
    argsInfo.hasTiling = 1;
    argsInfo.isNoNeedH2DCopy = 0;

    error = rtKernelLaunchWithFlagV2(GetFunSig(cocType, dataType),
                                     kernelArgs.cocKernelParam.cocTilingData.blockDim,
                                     &argsInfo, nullptr, stream, 0, &cfgInfo);
    if (error != RT_ERROR_NONE) {
        MKI_LOG(ERROR) << "AsdRtFunctionLaunch -:" << to_string(error);
        return LCAL_ERROR_MKIRT;
    }
    return error;
}
}