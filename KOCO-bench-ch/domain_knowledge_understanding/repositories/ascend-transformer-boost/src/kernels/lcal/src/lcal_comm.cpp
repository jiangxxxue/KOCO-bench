/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <unordered_map>
#include <lcal_comm.h>
#include "lcal_internal.h"

#include <chrono>
#include <vector>
#include <mutex>
#include <map>
#include <set>
#include <thread>
#include <sstream>
#include <iomanip>

#include <hccl/hccl.h>
#include "mki/utils/log/log.h"
#include "mki/utils/env/env.h"
#include "tools/socket/lcal_sock_exchange.h"

#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "runtime/dev.h"
#include "runtime/rt_ffts.h"
#include "profiling/report_timing.h"

constexpr int AI_CORE_NUM_24 = 24;
constexpr int AI_CORE_NUM_20 = 20;
constexpr int AI_CORE_NUM_2 = 2;

enum TopologyType : int {
    TOPOLOGY_HCCS = 0,
    TOPOLOGY_PIX,
    TOPOLOGY_PIB,
    TOPOLOGY_PHB,
    TOPOLOGY_SYS,
    TOPOLOGY_SIO,
    TOPOLOGY_HCCS_SW
};

using namespace std;
using namespace chrono;
using namespace Mki;

namespace Lcal {
constexpr int HCCL_IPC_PID_ARRAY_SIZE = 1;
constexpr int LCAL_INIT_TIMEOUT = 600;

static map<string, GM_ADDR [LCAL_MAX_RANK_SIZE]> g_localPeerMemMap;
static map<string, int[LCAL_MAX_RANK_SIZE]> g_devList;
static std::mutex g_mtx;

static const std::unordered_map<std::string, ChipName> CHIP_MAP = {
    {"Ascend310P", ChipName::CHIP_310P3},
    {"Ascend910B1", ChipName::CHIP_910B1},
    {"Ascend910B2", ChipName::CHIP_910B2},
    {"Ascend910B2C", ChipName::CHIP_910B2C},
    {"Ascend910B3", ChipName::CHIP_910B3},
    {"Ascend910B4", ChipName::CHIP_910B4},
    {"Ascend910B4-1", ChipName::CHIP_910B41},
    {"Ascend910_9391", ChipName::CHIP_910_9391},
    {"Ascend910_9381", ChipName::CHIP_910_9381},
    {"Ascend910_9392", ChipName::CHIP_910_9392},
    {"Ascend910_9382", ChipName::CHIP_910_9382},
    {"Ascend910_9372", ChipName::CHIP_910_9372},
    {"Ascend910_9361", ChipName::CHIP_910_9361},
    {"Ascend910_9362", ChipName::CHIP_910_9362}
};

ChipName GetChipName()
{
    static ChipName curChipName = ChipName::RESERVED;
    if (curChipName != ChipName::RESERVED) {
        return curChipName;
    }
    constexpr int socVerLength = 100;
    char ver[socVerLength];
    auto ret = rtGetSocVersion(ver, socVerLength);
    if (ret != RT_ERROR_NONE) {
        MKI_LOG(ERROR) << "rtGetSocVersion failed, not sure whether the function is normal, please use it with caution";
        return ChipName::RESERVED;
    }
    string chipName(ver);
    MKI_LOG(DEBUG) << "rtGetSocVersion -- The result after converting ver to string is:" << chipName;

    auto it = CHIP_MAP.find(chipName);
    if (it != CHIP_MAP.end()) {
        curChipName = it->second;
    } else {
        MKI_LOG(WARN) << "There is no commitment to the supported chip types yet," <<
            " and it is not certain whether the functions will work properly.";
    }
    return curChipName;
}

uint32_t GetCoreNum(ChipName chipName)
{
    switch (chipName) {
        case ChipName::CHIP_910B1:
        case ChipName::CHIP_910B2:
        case ChipName::CHIP_910_9391:
        case ChipName::CHIP_910_9381:
        case ChipName::CHIP_910_9392:
        case ChipName::CHIP_910_9382:
        case ChipName::CHIP_910B2C:
            return AI_CORE_NUM_24;
        case ChipName::CHIP_910B3:
        case ChipName::CHIP_910B4:
        case ChipName::CHIP_910B41:
        case ChipName::CHIP_910_9372:
        case ChipName::CHIP_910_9361:
        case ChipName::CHIP_910_9362:
        case ChipName::CHIP_910A5:
            return AI_CORE_NUM_20;
        case ChipName::CHIP_310P3:
            return AI_CORE_NUM_2;
        default:
            MKI_LOG(ERROR) << "Unknown chip name";
            return 0;
    }
}

bool SkipUnusedChannel910B2C(int curRank, int peerRank, ChipName chipName)
{
    if (chipName == ChipName::CHIP_910B2C) {
        constexpr int rankSizePerNode = 8;
        if ((curRank / rankSizePerNode != peerRank / rankSizePerNode) &&
            (std::abs(curRank - peerRank) != rankSizePerNode)) {
            return true;
        }
    }
    return false;
}

int LcalComm::InitDumpAddr()
{
    constexpr uint32_t dumpCoreCnt = 75;
    constexpr uint32_t dumpSizePerCore = 1 * 1024 * 1024;
    constexpr uint32_t dumpWorkspaceSize = dumpCoreCnt * dumpSizePerCore;
    GM_ADDR dumpAddr = nullptr;
    int ret = 0;
    ret = aclrtMalloc(reinterpret_cast<void **>(&dumpAddr), dumpWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        MKI_LOG(ERROR) << "aclrtMalloc err " << __LINE__ << " " << ret;
        return LCAL_ERROR_INTERNAL;
    }
    aclrtMemset(dumpAddr, dumpWorkspaceSize, 0, dumpWorkspaceSize);

    GM_ADDR memory = static_cast<GM_ADDR>(std::malloc(dumpWorkspaceSize));
    if (!memory) {
        MKI_LOG(ERROR) << "std::malloc err " << __LINE__;
        return LCAL_ERROR_INTERNAL;
    }
    errno_t result = memset_s(memory, dumpWorkspaceSize, 0, dumpWorkspaceSize);
    if (result != 0) {
        MKI_LOG(ERROR) << "memset_s err " << result;
    }
    for (uint32_t i = 0; i < dumpCoreCnt; ++i) {
        GM_ADDR block_start = memory + i * dumpSizePerCore;
        GM_ADDR deviceBlockStart = dumpAddr + i * dumpSizePerCore;

        LcclDumpBlockInfo* block_info = reinterpret_cast<LcclDumpBlockInfo*>(block_start);
        block_info->len = dumpSizePerCore;
        block_info->core = i;
        block_info->blockNum = 0;
        block_info->dumpOffset = dumpSizePerCore - sizeof(LcclDumpBlockInfo);
        block_info->magic = 0;
        block_info->dumpAddr = reinterpret_cast<uint64_t>(deviceBlockStart + sizeof(LcclDumpBlockInfo));
    }

    ret = aclrtMemcpy(dumpAddr, dumpWorkspaceSize, memory, dumpWorkspaceSize, ACL_MEMCPY_HOST_TO_DEVICE);
    std::free(memory);
    if (ret != ACL_SUCCESS) {
        MKI_LOG(ERROR) << "aclrtMemcpy err " << __LINE__ << " " << ret;
        return LCAL_ERROR_INTERNAL;
    }

    commArgs_.dumpAddr = dumpAddr;
    return LCAL_SUCCESS;
}

int LcalComm::SyncCommArgs()
{
    commArgs_.rank = rank_;
    commArgs_.localRank = localRank_;
    commArgs_.rankSize = rankSize_;
    commArgs_.localRankSize = localRankSize_;
    for (int i = 0; i < rankSize_; ++i) {
        commArgs_.peerMems[i] = peerMem_[i];
    }

    if (isEnableMsprofOp_ && InitDumpAddr() != LCAL_SUCCESS) {
        return LCAL_ERROR_INTERNAL;
    }

    if (isEnableMix_) {
        uint64_t fftsVal = 0;
        uint32_t fftsLen = 0;
        int error = rtGetC2cCtrlAddr(&fftsVal, &fftsLen);
        if (error != RT_ERROR_NONE) {
            MKI_LOG(ERROR) << "rtGetC2cCtrlAddr err:" << error;
            return LCAL_ERROR_MKIRT;
        }
        commArgs_.fftsVal = fftsVal;
    }

    int ret = 0;
    ret = aclrtMalloc(reinterpret_cast<void **>(&commArgsPtr_), sizeof(commArgs_), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        MKI_LOG(ERROR) << "aclrtMalloc err " << __LINE__ << " " << ret;
        return LCAL_ERROR_INTERNAL;
    }
    ret = aclrtMemcpy(commArgsPtr_, sizeof(commArgs_), &commArgs_, sizeof(commArgs_), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        MKI_LOG(ERROR) << "aclrtMemcpy err " << __LINE__ << " " << ret;
        return LCAL_ERROR_INTERNAL;
    }
    return LCAL_SUCCESS;
}

int LcalComm::InitCommon()
{
    if (EnablePeerAccess() != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "EnablePeerAccess failed!";
        return LCAL_ERROR_INTERNAL;
    }
    const char *lcclDeterministic = Mki::GetEnv("LCCL_DETERMINISTIC");
    if (lcclDeterministic && (string(lcclDeterministic) == "1" || string(lcclDeterministic) == "true")) {
        deterministic_ = true;
        commArgs_.extraFlag |= ExtraFlag::DETERMINISTIC;
    }
    if (GetChipName() == ChipName::CHIP_910B2C) {
        commArgs_.extraFlag |= ExtraFlag::TOPO_910B2C;
    }
    if (GetChipName() >= ChipName::CHIP_910_9391) {
        commArgs_.extraFlag |= ExtraFlag::TOPO_910_93;
    }
    if (GetChipName() > ChipName::CHIP_910_9362) {
        commArgs_.extraFlag |= ExtraFlag::TOPO_910A5;
    }
    if (GetCoreNum(GetChipName()) > AI_CORE_NUM_20) {
        commArgs_.extraFlag |= ExtraFlag::IS_GREATER_THAN_40_AIV;
    }

    ReportTiming report("LcclReporting", rank_, false, nullptr, nullptr);
    MKI_LOG(INFO) << "LcalComm::InitCommon ReportTiming " << std::hex << ReportTiming::ProfilingStatus() << std::dec;
    if (ReportTiming::ProfilingStatus() == ReportTiming::PROF_TASK_TIME_DUMP) {
        isEnableMsprofOp_ = true;
        isEnableMix_ = true;
    }

    int32_t opGroup = 0;
    if (isEnableMsprofOp_) {
        opGroup = 0;
    } else if (isEnableMix_) {
        opGroup = 1;
    } else {
        constexpr int32_t normalOpGroup = 2;
        opGroup = normalOpGroup;
    }
    MKI_LOG(INFO) << "LcalComm::InitCommon RegistKernel opGroup " << opGroup;
    RegistKernel(opGroup);

    localRank_ = rank_ % localRankSize_;
    return LCAL_SUCCESS;
}

void LcalComm::CloseIpcMem()
{
    for (int i = 0; i < rankSize_; ++i) {
        if (i == rank_ || peerMem_[i] == nullptr) {
            continue;
        }

        int ret = rtIpcCloseMemory(static_cast<void *>(peerMem_[i]));
        if (ret != RT_ERROR_NONE) {
            MKI_LOG(WARN) << "Close ipc[" << i << "] memory failed! ret: " << ret;
        }
        peerMem_[i] = nullptr;
    }
}

void LcalComm::FreePeerMem(GM_ADDR &mem) const
{
    if (mem != nullptr) {
        aclError aclRet = aclrtFree(mem);
        if (aclRet != ACL_SUCCESS) {
            MKI_LOG(ERROR) << "Free share memory failed! ret: " << aclRet;
        }
    }
    mem = nullptr;
}

int LcalComm::Init()
{
    if (inited_) {
        return LCAL_SUCCESS;
    }
    if (rank_ < 0 || rank_ >= rankSize_ || rankSize_ <= 0 || rankSize_ > LCAL_MAX_RANK_SIZE) {
        MKI_LOG(ERROR) << "The rank is invalid! rank:" << rank_ << " rankSize:" << rankSize_;
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    if (LcalSockExchange::CheckValid(commId_)) {
        socketExchange_ = new (nothrow) LcalSockExchange(rank_, rankSize_, commId_);
    } else {
        socketExchange_ = new (nothrow) LcalSockExchange(rank_, rankSize_, rankList_, commDomain_);
    }
    if (socketExchange_ == nullptr) {
        MKI_LOG(ERROR) << "LcalSockExchange create failed. rank : " << rank_ << " rankSize:" << rankSize_;
        return LCAL_ERROR_INTERNAL;
    }
    int ret = GetDev();
    if (ret != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "init context failed! ret: " << ret;
        return ret;
    }

    MKI_LOG(INFO) << "rank " << rank_ << "/" << rankSize_ << " running devId:" << devId_;

    if (InitCommon() != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "init common failed!";
        return LCAL_ERROR_INTERNAL;
    }

    MKI_LOG(DEBUG) << "Prepare to InitCommMem localRankSize_ -> " << localRankSize_ << ", localRank_ -> " << localRank_;
    if (InitCommMem() != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "InitCommMem failed!";
        return LCAL_ERROR_INTERNAL;
    }
    MKI_LOG(DEBUG) << "InitCommMem " << rank_ << "/" << rankSize_ << ", localRank_ : " << localRank_ <<
            ", localRankSize_ : " << localRankSize_ << " success";

    SyncCommArgs();
    MKI_LOG(INFO) << "LcalCommInit " << rank_ << "/" << rankSize_ << " success and extraFlag:" << commArgs_.extraFlag <<
        " commArgs_.localRank : " << commArgs_.localRank << " commArgs_.localRankSize : " << commArgs_.localRankSize;
    inited_ = true;
    delete socketExchange_;
    socketExchange_ = nullptr;
    return LCAL_SUCCESS;
}

int LcalComm::InitThread(const std::string &uid)
{
    if (inited_) {
        return LCAL_SUCCESS;
    }
    if (rank_ < 0 || rank_ >= rankSize_ || rankSize_ <= 0 || rankSize_ > LCAL_MAX_RANK_SIZE) {
        MKI_LOG(ERROR) << "The rank is invalid! rank:" << rank_ << "rankSize:" << rankSize_;
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    if (GetDevThread(uid) != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "get devs failed.";
        return LCAL_ERROR_INTERNAL;
    }
    MKI_LOG(INFO) << "rank " << rank_ << "/" << rankSize_ << " running devId:" << devId_ << "uid: " << uid;

    if (InitCommon() != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "init common failed!";
        return LCAL_ERROR_INTERNAL;
    }
    {
        lock_guard<mutex> lock(g_mtx);
        if (g_localPeerMemMap.find(uid) == g_localPeerMemMap.end()) {
            for (int i = 0; i < rankSize_; ++i) {
                g_localPeerMemMap[uid][i] = nullptr;
            }
        }
        uid_ = uid;
    }
    InitMem();
    g_localPeerMemMap[uid][rank_] = peerMem_[rank_];

    auto start = high_resolution_clock::now();
    for (int i = 0; i < rankSize_; ++i) {
        while (g_localPeerMemMap[uid][i] == nullptr) {
            this_thread::sleep_for(1ms);
            auto elapsed = duration_cast<seconds>(high_resolution_clock::now() - start);
            if (elapsed.count() > LCAL_INIT_TIMEOUT) {
                MKI_LOG(ERROR) << "Lccl Init timeout!";
                FreePeerMem(g_localPeerMemMap[uid][rank_]);
                return LCAL_ERROR_TIMEOUT;
            }
        }
        peerMem_[i] = g_localPeerMemMap[uid][i];
    }
    localRank_ = rank_;
    localRankSize_ = rankSize_;
    SyncCommArgs();
    MKI_LOG(INFO) << "Lccl init multi thread " << rank_ << "/" << rankSize_ << " success, uid:" << uid;
    inited_ = true;
    return LCAL_SUCCESS;
}

int LcalComm::EnablePeerAccess()
{
    physicalInfo_.chipName = GetChipName();
    for (auto &dev : devList_) {
        if (devId_ == dev) {
            continue;
        }
        if (SkipUnusedChannel910B2C(dev, devId_, GetChipName())) {
            continue;
        }

        int64_t value = 0;
        if (rtGetPairDevicesInfo(devId_, dev, 0, &value) != RT_ERROR_NONE) {
            MKI_LOG(WARN) << devId_ << " & " << dev << " pair devices info failed to get";
        } else {
            MKI_LOG(DEBUG) << devId_ << " <-----> " << dev << ", halGetPairDevicesInfo: *value = " << value;
        }

        if (value == TOPOLOGY_HCCS || value == TOPOLOGY_SIO || value == TOPOLOGY_HCCS_SW ||
            GetChipName() == ChipName::CHIP_910B2C) {
            physicalInfo_.physicalLink = PhysicalLink::HCCS;
            commArgs_.extraFlag &= ~(ExtraFlag::TOPO_PCIE);
        } else if (physicalInfo_.physicalLink == PhysicalLink::RESERVED) {
            physicalInfo_.physicalLink = PhysicalLink::PCIE;
            commArgs_.extraFlag |= ExtraFlag::TOPO_PCIE;
            if (rankSize_ > PING_PONG_SIZE) {
                MKI_LOG(ERROR) << "do not support pcie > 2 rank! rankSize_ = " << rankSize_;
                return LCAL_ERROR_INTERNAL;
            }
        }

        physicalInfo_.coreNum = GetCoreNum(physicalInfo_.chipName);

        if (physicalInfo_.chipName == ChipName::CHIP_310P3 && value == 0) {
            MKI_LOG(WARN) << "warn aclrtDeviceEnablePeerAccess is skipped! peerDeviceId = " << dev;
            continue;
        }

        aclError ret = aclrtDeviceEnablePeerAccess(dev, 0);
        if (ret != ACL_SUCCESS) {
            MKI_LOG(ERROR) << "err aclrtDeviceEnablePeerAccess failed peerDeviceId = " << dev << " ,rank = " << rank_
                           << ", value = " << value << ", flags = " << 0 << "," << __LINE__ << ": " << ret;
            return LCAL_ERROR_INTERNAL;
        }
    }
    MKI_LOG(DEBUG) << "EnablePeerAccess succeed" << rank_;
    return LCAL_SUCCESS;
}

int LcalComm::GetDev()
{
    int nodeNum = socketExchange_->GetNodeNum();
    if (nodeNum <= 0 || nodeNum > rankSize_) {
        MKI_LOG(ERROR) << "error! node num : " << nodeNum << " rank size: " << rankSize_;
        return LCAL_ERROR_INTERNAL;
    }
    localRankSize_ = rankSize_ / nodeNum;
    localRank_ = rank_ % localRankSize_;
    MKI_LOG(DEBUG) << "GetDev : localRankSize_ : " << localRankSize_ << " localRank_: " << localRank_
                    << "  rank :" << rank_ << "   rankSize :" << rankSize_;
    devList_.resize(rankSize_);
    aclError aclRet = aclrtGetDevice(&devId_);
    if (aclRet != ACL_SUCCESS) {
        MKI_LOG(ERROR) << "aclrtGetDevice error! ret: " << aclRet;
        return LCAL_ERROR_INTERNAL;
    }
    int ret = socketExchange_->AllGather(&devId_, 1, devList_.data());
    if (ret != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "LcalSockExchange AllGather error! ret: " << ret;
        return LCAL_ERROR_INTERNAL;
    }
    std::string devIdStr = "";
    for (int i = 0; i < rankSize_; ++i) {
        devIdStr += (i == 0 ? "" : ", ");
        devIdStr += to_string(devList_[i]);
    }
    MKI_LOG(DEBUG) << "rank " << rank_ << " devId: " << devId_ << ", otherDevList : " << devIdStr;
    MKI_LOG(INFO) << "AllGather: Get other rank dev id success";
    return LCAL_SUCCESS;
}

int LcalComm::GetDevThread(const std::string &uid)
{
    devList_.resize(rankSize_);
    aclError aclRet = aclrtGetDevice(&devId_);
    if (aclRet != ACL_SUCCESS) {
        MKI_LOG(ERROR) << "aclrtGetDevice error! ret: " << aclRet;
        return LCAL_ERROR_INTERNAL;
    }
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        if (g_devList.find(uid) == g_devList.end()) {
            for (int i = 0; i < rankSize_; ++i) {
                g_devList[uid][i] = 0;
            }
        }
    }
    g_devList[uid][rank_] = devId_ + 1;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < rankSize_; ++i) {
        while (g_devList[uid][i] == 0) {
            this_thread::sleep_for(1ms);
            auto elapsed = duration_cast<seconds>(high_resolution_clock::now() - start);
            if (elapsed.count() > LCAL_INIT_TIMEOUT) {
                MKI_LOG(ERROR) << "Lccl Init timeout!";
                return LCAL_ERROR_TIMEOUT;
            }
        }
        devList_.at(i) = g_devList[uid][i] - 1;
    }
    return LCAL_SUCCESS;
}

int LcalComm::InitMem()
{
    constexpr int32_t bufferSizeUint = 1024 * 1024;
    int lcalBuffSize = bufferSize_ * bufferSizeUint + LCAL_FLAG_BUFF_BYTES;

    MKI_LOG(DEBUG) << "lcal buffer size " << lcalBuffSize;
    aclError ret = aclrtMalloc(
        reinterpret_cast<void **>(&peerMem_[rank_]), lcalBuffSize,
        (GetChipName() == ChipName::CHIP_310P3) ? ACL_MEM_MALLOC_HUGE_FIRST_P2P : ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        MKI_LOG(ERROR) << "allocate device mem error " << __FILE__ << ":" << __LINE__ << " " << ret;
        return LCAL_ERROR_INTERNAL;
    }
    MKI_LOG(DEBUG) << "peerMem[rank" << rank_ << "], allocate finished.";
    aclrtMemset(peerMem_[rank_], lcalBuffSize, 0, lcalBuffSize);
    return LCAL_SUCCESS;
}

int LcalComm::GetPid(uint32_t *pids)
{
    if (rtDeviceGetBareTgid(&pids[rank_]) != RT_ERROR_NONE) {
        MKI_LOG(ERROR) << "DeviceGetBareTgid err " << __LINE__;
        return LCAL_ERROR_INTERNAL;
    }
    int ret = socketExchange_->AllGather(&pids[rank_], 1, pids);
    if (ret != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "LcalSockExchange AllGather error! ret: " << ret;
        return ret;
    }
    for (int i = 0; i < rankSize_; ++i) {
        MKI_LOG(DEBUG) << "rank : " << rank_ << ", otherRank : " << i << " pid[" << i << "]: " << pids[i];
    }
    MKI_LOG(DEBUG) << "AllGather: Get other rank pid";
    return LCAL_SUCCESS;
}

int LcalComm::GetSidId(int64_t sdids[LCAL_MAX_RANK_SIZE], int rankSize)
{
    if (rank_ >= rankSize) {
        MKI_LOG(ERROR) << "LcalComm::GetSidId err rank_ >= rankSize " << rank_ << ">=" << rankSize;
        return LCAL_ERROR_INTERNAL;
    }
    if ((physicalInfo_.chipName >= ChipName::CHIP_910_9391) && (physicalInfo_.chipName < ChipName::RESERVED)) {
        const int rtModuleTypeSystem = 0;
        const int infoTypeSdid = 26;
        if (rtGetDeviceInfo(devList_[rank_], rtModuleTypeSystem, infoTypeSdid, &sdids[rank_]) != RT_ERROR_NONE) {
            MKI_LOG(ERROR) << "DeviceGetDeviceInfo err " << __LINE__;
            return LCAL_ERROR_INTERNAL;
        }
        MKI_LOG(DEBUG) << "rank " << rank_ << " dev id: " << devList_[rank_]
                       << " rtGetDeviceInfo sdid: " << sdids[rank_];

        int ret = socketExchange_->AllGather(&sdids[rank_], 1, sdids);
        if (ret != LCAL_SUCCESS) {
            MKI_LOG(ERROR) << "LcalSockExchange AllGather error! ret: " << ret;
            return ret;
        }
        for (int i = 0; i < rankSize_; ++i) {
            MKI_LOG(DEBUG) << "rank " << i << " sdid: " << sdids[i];
        }
        MKI_LOG(DEBUG) << "AllGather: Get other rank sdid";
    }
    return LCAL_SUCCESS;
}

int LcalComm::GetName(string &name, char names[LCAL_MAX_RANK_SIZE][IPC_NAME_SIZE]) const
{
    int ret = socketExchange_->AllGather<char>(name.c_str(), IPC_NAME_SIZE, names[0]);
    if (ret != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "LcalSockExchange AllGather error! ret: " << ret;
        return LCAL_ERROR_INTERNAL;
    }
    for (int i = 0; i < rankSize_; ++i) {
        names[i][IPC_NAME_SIZE - 1] = '\0';
        MKI_LOG(DEBUG) << "rank " << i << " mem name: " << names[i];
    }
    MKI_LOG(DEBUG) << "AllGather: Get other rank mem name";
    return LCAL_SUCCESS;
}

int LcalComm::InitCommMem()
{
    int ret = InitMem();
    if (ret != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "InitMem error! ret: " << ret;
        return ret;
    }

    uint32_t pids[LCAL_MAX_RANK_SIZE] = {0};
    ret = GetPid(pids);
    if (ret != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "GetPid error! ret: " << ret;
        return ret;
    }

    int64_t sdids[LCAL_MAX_RANK_SIZE] = {0};
    ret = GetSidId(sdids, rankSize_);
    if (ret != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "GetSidId error! ret: " << ret;
        return ret;
    }

    string name;
    if (SetMemoryName(name) != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "SetMemoryName err ";
        return LCAL_ERROR_INTERNAL;
    }

    if (SetIpcPidSdid(name, pids, sdids) != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "SetIpcPidSdid failed!";
        return LCAL_ERROR_INTERNAL;
    }

    MKI_LOG(DEBUG) << "rank " << rank_ << " mem name: " << name;
    char names[LCAL_MAX_RANK_SIZE][IPC_NAME_SIZE];
    ret = GetName(name, names);
    if (ret != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "GetName error! ret: " << ret;
        return ret;
    }

    if (OpenIpcMem(names) != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "rank: " << rank_ << " OpenIpcMem failed!";
        return LCAL_ERROR_INTERNAL;
    }
    return LCAL_SUCCESS;
}

int LcalComm::OpenIpcMem(const char names[LCAL_MAX_RANK_SIZE][IPC_NAME_SIZE])
{
    static mutex mut;
    lock_guard<mutex> lock(mut);
    for (int i = 0; i < rankSize_; ++i) {
        if (i == rank_) {
            continue;
        }
        if (SkipUnusedChannel910B2C(rank_, i, GetChipName())) {
            continue;
        }
        int ret = rtIpcOpenMemory(reinterpret_cast<void **>(&peerMem_[i]), names[i]);
        if (ret != RT_ERROR_NONE) {
            CloseIpcMem();
            MKI_LOG(ERROR) << "rank : " << rank_ << " localRank : " << localRank_ << " peerMem: " << i <<
                " IpcOpenMemory err " << ret;
            return LCAL_ERROR_INTERNAL;
        }
    }
    ipcMemInited_ = true;
    return LCAL_SUCCESS;
}

int LcalComm::SetMemoryName(string &name)
{
    char nameModified[IPC_NAME_SIZE] = {};
    int memRank = rank_;
    constexpr int32_t bufferSizeUint = 1024 * 1024;
    int lcalBuffSize = bufferSize_ * bufferSizeUint + LCAL_FLAG_BUFF_BYTES;
    if (rtIpcSetMemoryName(peerMem_[memRank], lcalBuffSize, nameModified, IPC_NAME_SIZE) != RT_ERROR_NONE) {
        return LCAL_ERROR_INTERNAL;
    }
    name = nameModified;
    return LCAL_SUCCESS;
}

int LcalComm::SetIpcPidSdid(string &name, const uint32_t *pids, const int64_t *sdids) const
{
    for (int i = 0; i < rankSize_; ++i) {
        if (i == rank_) {
            continue;
        }

        if (physicalInfo_.chipName < ChipName::CHIP_910_9391) {
            int32_t pidInt32 = pids[i];
            int rtRet = rtSetIpcMemPid(name.c_str(), &pidInt32, HCCL_IPC_PID_ARRAY_SIZE);
            if (rtRet != RT_ERROR_NONE) {
                MKI_LOG(ERROR) << "err " << rtRet;
                return LCAL_ERROR_INTERNAL;
            }
        } else {
            int32_t pidInt32 = pids[i];
            int rtRet = rtSetIpcMemorySuperPodPid(name.c_str(), sdids[i], &pidInt32, HCCL_IPC_PID_ARRAY_SIZE);
            if (rtRet != RT_ERROR_NONE) {
                MKI_LOG(ERROR) << "err " << rtRet;
                return LCAL_ERROR_INTERNAL;
            }
        }
    }
    return LCAL_SUCCESS;
}

LcalComm::~LcalComm()
{
    {
        lock_guard<mutex> lock(g_mtx);
        if (g_localPeerMemMap.find(uid_) != g_localPeerMemMap.end()) {
            g_localPeerMemMap.erase(uid_);
        }
    }

    if (ipcMemInited_) {
#ifndef USE_MSSANITIZER
        CloseIpcMem();
#endif
        ipcMemInited_ = false;
    }
    if (socketExchange_) {
        delete socketExchange_;
        socketExchange_ = nullptr;
    }
    FreePeerMem(commArgs_.dumpAddr);
    FreePeerMem(peerMem_[rank_]);
    FreePeerMem(commArgsPtr_);
}

LcalComm::LcalComm(int rank, int rankSize) : rank_(rank), rankSize_(rankSize)
{
}

LcalComm::LcalComm(int rank, int rankSize, int bufferSize) : rank_(rank), rankSize_(rankSize), bufferSize_(bufferSize)
{
}

LcalComm::LcalComm(int rank, int rankSize, int commDomain, int bufferSize, int isEnableMagic)
    : rank_(rank), rankSize_(rankSize), commDomain_(commDomain), bufferSize_(bufferSize), isEnableMix_(isEnableMagic)
{
}

LcalComm::LcalComm(int rank, int rankSize, LcalUniqueId commId)
    : rank_(rank), rankSize_(rankSize), commId_(commId)
{
}

int LcalComm::GetRank() const
{
    return rank_;
}

int LcalComm::GetRankSize() const
{
    return rankSize_;
}

int LcalComm::GetCommSize() const
{
    return commSize_;
}

int LcalComm::GetBufferSize() const
{
    return bufferSize_;
}

const PhysicalInfo &LcalComm::GetPhysicalInfo() const
{
    return physicalInfo_;
}

GM_ADDR LcalComm::GetCommArgsPtr() const
{
    return commArgsPtr_;
}

CommArgs* LcalComm::GetCommArgs()
{
    return &commArgs_;
}


std::string LcalComm::PrintDFX()
{
    if (commArgsPtr_ == nullptr) {
        return "no comm args";
    }
    int ret = aclrtMemcpy(&commArgs_, sizeof(commArgs_), commArgsPtr_, sizeof(commArgs_),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        MKI_LOG(ERROR) << "aclrtMemcpy err " << __LINE__ << " " << ret;
        return "acl mem copy error";
    }
    stringstream ss;
    ss << "CommArgs {"
       << "\n  rank: " << commArgs_.rank
       << "\n  localRank: " << commArgs_.localRank
       << "\n  rankSize: " << commArgs_.rankSize
       << "\n  localRankSize: " << commArgs_.localRankSize
       << "\n  extraFlag:  0x" << std::hex << std::setfill('0') << commArgs_.extraFlag << std::dec;

    ss << "\n  peerMems: [";
    for (int i = 0; i < LCAL_MAX_RANK_SIZE; ++i) {
        if (commArgs_.peerMems[i] == nullptr) {
            continue;
        }
        if (i > 0) {
            ss << ", ";
        }
        ss << "{id: " << static_cast<void *>(commArgs_.peerMems[i]) << "}";
    }
    ss << "]";

    ss << "\n  magics: [";
    for (int i = 0; i < rankSize_; ++i) {
        ss << std::dec << commArgs_.magics[i] << ",";
    }
    ss << "] \n";

    ss << "\n  dfx: [";
    const int dfxGroupCount = 5;
    for (int i = 0; i < DFX_COUNT; ++i) {
        if (i % dfxGroupCount == 0) {
            ss << "\n    " << std::dec << setw(dfxGroupCount) << i << ": ";
        }
        ss << "0x"<< std::hex << commArgs_.dfx[i] << std::dec << ", ";
    }
    ss << "\n    ]";

    ss << "\n}";
    return ss.str();
}

} // Lcal