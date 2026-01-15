/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <functional>
#include <string>
#include <sstream>
#include <atb/comm.h>
#include <hccl/hccl.h>
#include <mki/utils/file_system/file_system.h>
#include <atb/utils/log.h>
#include <mki/utils/share_memory/share_memory.h>
#include "atb/utils/singleton.h"
#include "atb/utils/comm_pool.h"

namespace atb {
struct CommInitInfo {
    int signal = 0;
    HcclRootInfo hcclRootInfo = {};
    bool barrier[1]; // Flexible array member
};
void ShmGetHcclRootInfo(Mki::ShareMemory &shm, const CommInitInfo &shmInfo, HcclRootInfo &hcclRootInfo)
{
    bool commIdReady = false;
    while (true) {
        shm.SemLock();
        if (shmInfo.signal != 0) {
            hcclRootInfo = shmInfo.hcclRootInfo;
            commIdReady = true;
        }
        shm.SemUnLock();
        if (commIdReady) {
            break;
        }
    }
}

void ShmSetHcclRootInfo(Mki::ShareMemory &shm, CommInitInfo &shmInfo, HcclRootInfo hcclRootInfo)
{
    shm.SemLock();
    shmInfo.hcclRootInfo = hcclRootInfo;
    shmInfo.signal = 1;
    shm.SemUnLock();
}

void ShmSetReady(Mki::ShareMemory &shm, CommInitInfo &shmInfo, int32_t rank)
{
    shm.SemLock();
    shmInfo.barrier[rank] = true;
    shm.SemUnLock();
}

bool ShmBarrier(Mki::ShareMemory &shm, CommInitInfo &shmInfo, int32_t rank, int32_t rankSize)
{
    ATB_LOG(INFO) << "barrier start, rank:" << rank;
    ShmSetReady(shm, shmInfo, rank);

    ATB_LOG(INFO) << "check all ready start";
    const double timeout = 600; // 600: 10 minutes timeout
    time_t startTime = time(nullptr);
    while (true) {
        time_t currentTime = time(nullptr);
        if (difftime(currentTime, startTime) > timeout) {
            ATB_LOG(ERROR) << "barrier fail, check all ready timeout";
            return false;
        }

        bool allReady = true;
        shm.SemLock();
        for (int i = 0; i < rankSize; i++) {
            if (!shmInfo.barrier[i]) {
                allReady = false;
                break;
            }
        }
        shm.SemUnLock();
        if (allReady) {
            ATB_LOG(INFO) << "check all ready success";
            break;
        }
    }

    ATB_LOG(INFO) << "barrier success, rank:" << rank;
    return true;
}

bool CreateHcclRootInfo(HcclRootInfo &hcclRootInfo, int32_t rank, int32_t rankRoot, int32_t rankSize)
{
    uint32_t deviceCount = 0;
    aclError ret = aclrtGetDeviceCount(&deviceCount);
    if (ret != ACL_ERROR_NONE) {
        ATB_LOG(ERROR) << "Failed to get device count, error code: " << ret;
        return false;
    }
    if (static_cast<uint32_t>(rankSize) > deviceCount || rankSize <= 0) {
        ATB_LOG(ERROR) << "rankSize must be greater than zero and smaller than or equal to deviceCount. "
                       << "Device count: " << deviceCount << ", rankSize: " << rankSize;
        return false;
    }
    if (rank < 0 || rank >= rankSize) {
        ATB_LOG(ERROR) << "rank should greater than zero and smaller than ranksize, rank:" << rank
                       << ", rankSize:" << rankSize;
        return false;
    }
    if (rankRoot < 0 || rankRoot >= rankSize) {
        ATB_LOG(ERROR) << "rankRoot should greater than zero and smaller than ranksize, rankRoot:" << rankRoot
                       << ", rankSize:" << rankSize;
        return false;
    }
    std::string shmName = "hcclShareMem";
    Mki::ShareMemory shm(shmName, sizeof(atb::CommInitInfo) + rankSize * sizeof(bool));
    auto *shmInfo = static_cast<atb::CommInitInfo *>(shm.GetShm());
    if (!shmInfo) {
        ATB_LOG(ERROR) << "create share memory fail, rank:" << rank;
        return false;
    }

    // 主进程通过HcclGetRootInfo获取到hcclRootInfo_(包含HostIP信息), 写到共享内存，其他进程读取RoortInfo
    // 等所有的进程都准备好时，再一起往下执行CreateHcclComm
    ATB_LOG(INFO) << "create share memory success, rank:" << rank;
    if (rank == rankRoot) {
        ATB_LOG(INFO) << "rankRoot:" << rankRoot;
        auto hcclResult = HcclGetRootInfo(&hcclRootInfo);
        if (hcclResult != HCCL_SUCCESS) {
            ATB_LOG(ERROR) << "HcclGetRootInfo fail, error:" << hcclResult << ", rank:" << rank;
            return false;
        }
        ATB_LOG(INFO) << "HcclGetRootInfo success, write to share memory";
        ShmSetHcclRootInfo(shm, *shmInfo, hcclRootInfo);
    } else {
        ATB_LOG(INFO) << "get root info from share memory";
        ShmGetHcclRootInfo(shm, *shmInfo, hcclRootInfo);
    }

    return ShmBarrier(shm, *shmInfo, rank, rankSize);
}

HcclComm Comm::CreateHcclComm(int32_t rank, int32_t rankRoot, int32_t rankSize, char *commName)
{
    HcclComm newHcclComm = nullptr;
    HcclRootInfo hcclRootInfo = {};
    if (!CreateHcclRootInfo(hcclRootInfo, rank, rankRoot, rankSize)) {
        return newHcclComm;
    }
    auto ret = HcclCommInitRootInfo(rankSize, &hcclRootInfo, rank, &newHcclComm);
    if (ret != HCCL_SUCCESS || newHcclComm == nullptr) {
        ATB_LOG(ERROR) << "HcclCommInitRootInfo fail, error:" << ret << ", rank:" << rank << ", rankSize:" << rankSize;
        return nullptr;
    }
    ATB_LOG(INFO) << "HcclCommInitRootInfo success, rank:" << rank << ", rankSize:" << rankSize
                  << ", newHcclComm:" << newHcclComm;
    HcclGetCommName(newHcclComm, commName);
    return newHcclComm;
}

HcclComm Comm::CreateHcclCommByRankTableFile(int32_t rank, int32_t rankSize, const char *rankTableFile,
                                             char *commName)
{
    if (rankTableFile == nullptr) {
        ATB_LOG(ERROR) << "rankTableFile is nullptr, please check the file path";
        return nullptr;
    }
    ATB_LOG(INFO) << "HCCL Runner multi server init ";
    HcclComm newHcclComm = nullptr;
    std::string rankTableFileStr(rankTableFile);
    std::string resolvePath = Mki::FileSystem::PathCheckAndRegular(rankTableFileStr);
    if (resolvePath == "") {
        ATB_LOG(ERROR) << "realpath fail, filePath:" << rankTableFileStr;
        return newHcclComm;
    }
    ATB_LOG(INFO) << "rankTableFilePath is :" << resolvePath;
    auto ret = HcclCommInitClusterInfo(rankTableFile, rank, &newHcclComm);
    if (ret != HCCL_SUCCESS || newHcclComm == nullptr) {
        ATB_LOG(ERROR) << "HCCL CommInitClusterInfo ERROR" << ret << " should check rankTableFile config";
        return nullptr;
    }
    ATB_LOG(INFO) << "HcclCommInitClusterInfo success, rank:" << rank << ", rankSize:" << rankSize
                  << ", newHcclComm:" << newHcclComm;
    HcclGetCommName(newHcclComm, commName);
    return newHcclComm;
}

static HcclCommConfig *GetHcclCommConfig(const uint32_t hcclBufferSize)
{
    static HcclCommConfig commConfig;
    static bool inited = false;
    if (!inited) {
        HcclCommConfigInit(&commConfig);
        inited = true;
    }
    commConfig.hcclBufferSize = hcclBufferSize;
    return &commConfig;
}

std::shared_ptr<void> CreateHcclCommByClusterInfo(uint32_t subCommRankId, const char *rankTableFile,
                                                  std::vector<uint32_t> &rankIds)
{
    if (rankTableFile == nullptr) {
        ATB_LOG(ERROR) << "rankTableFile is NULL";
        return std::shared_ptr<HcclComm>();
    }
    std::string rankTableFileStr(rankTableFile);
    ATB_LOG(INFO) << "create hccl comm by clusterInfo, rankTableFile:" << rankTableFileStr;
    std::string resolvePath = Mki::FileSystem::PathCheckAndRegular(rankTableFileStr);
    if (resolvePath.empty()) {
        ATB_LOG(ERROR) << "rank table file is inValid, file path:" << rankTableFileStr;
        return std::shared_ptr<HcclComm>();
    }
    HcclComm newHcclComm = nullptr;
    auto ret = HcclCommInitClusterInfo(rankTableFile, rankIds[subCommRankId], &newHcclComm);
    if (ret != HCCL_SUCCESS || newHcclComm == nullptr) {
        ATB_LOG(ERROR) << "HcclCommInitClusterInfoConfig failed, should check rankTableFile config, error code:"
                       << ret;
        return std::shared_ptr<HcclComm>();
    }
    return std::shared_ptr<HcclComm>(static_cast<HcclComm *>(newHcclComm), [=](void *hcclComm) {
        (void)hcclComm;
        ATB_LOG(INFO) << "destroy hcclComm, but not call HcclCommDestroy hcclComm:" << hcclComm;
    });
}

static void PrintRankIds(const std::vector<uint32_t> &rankIds)
{
    std::ostringstream oss;
    oss << "Create HcclSubComm with rankIds: ";
    for (const auto &id : rankIds) {
        oss << id << " ";
    }
    ATB_LOG(INFO) << oss.str();
}

HcclComm Comm::CreateHcclCrossMulitComm(const char *rankTableFile, uint32_t subCommRankId,
                                        std::vector<uint32_t> &rankIds, uint64_t subCommId, uint32_t hcclBufferSize,
                                        char *commName)
{
    if (rankTableFile == nullptr) {
        ATB_LOG(ERROR) << "rankTableFile is NULL";
        return nullptr;
    }
    std::string rankTableFileStr(rankTableFile);
    std::string resolvePath = Mki::FileSystem::PathCheckAndRegular(rankTableFileStr);
    if (resolvePath.empty()) {
        ATB_LOG(ERROR) << "rank table file is inValid, file path:" << rankTableFileStr;
        return nullptr;
    }
    if (subCommRankId >= rankIds.size()) {
        ATB_LOG(ERROR) << "The subCommRankId must be greater than or equal 0, "
                       << "and cannot exceed the rankIds max index.";
        return nullptr;
    }
    HcclComm globalHcclComm = GetSingleton<atb::CommPool<void>>()
                                  .GetComm(rankTableFileStr, std::bind(CreateHcclCommByClusterInfo, subCommRankId,
                                                                       rankTableFile, rankIds))
                                  .get();
    if (globalHcclComm) {
        ATB_LOG(INFO) << "get hccl comm success by subCommRankId:" << subCommRankId;
    } else {
        ATB_LOG(ERROR) << "get hccl comm fail by subCommRankId:" << subCommRankId;
        return nullptr;
    }
    HcclComm subHcclComm = nullptr;
    PrintRankIds(rankIds);
    auto ret = HcclCreateSubCommConfig(&globalHcclComm, rankIds.size(), rankIds.data(), subCommId, subCommRankId,
                                       GetHcclCommConfig(hcclBufferSize), &subHcclComm);
    if (ret != HCCL_SUCCESS || subHcclComm == nullptr) {
        ATB_LOG(ERROR) << "HcclCreateSubCommConfig failed, should check rank and rankIds, error code:" << ret;
        return nullptr;
    }
    HcclGetCommName(subHcclComm, commName);
    return subHcclComm;
}

Status Comm::DestoryHcclComm(HcclComm comm)
{
    if (comm == nullptr) {
        ATB_LOG(ERROR) << " invalid param, HcclComm is nullptr";
        return ERROR_INVALID_PARAM;
    }
    auto ret = HcclCommDestroy(comm);
    if (ret != HCCL_SUCCESS) {
        ATB_LOG(ERROR) << "DestoryHcclComm fail, error:" << ret;
        return ERROR_HCCL_FAIL;
    }
    return NO_ERROR;
}

} // namespace atb