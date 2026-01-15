/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/runner/hccl_runner.h"
#include <hccl/hccl.h>
#include <mki/utils/file_system/file_system.h>
#include <atb/utils/log.h>
#include <mki/utils/share_memory/share_memory.h>
#include "atb/utils/comm_pool.h"
#include "atb/utils.h"
#include "atb/utils/log.h"
#include "atb/utils/config.h"
#include "atb/utils/common_utils.h"
#include "atb/utils/singleton.h"

namespace atb {
HcclRunner::HcclRunner(const std::string &name, RunnerType runnerType, int rank, int rankSize, int rankRoot,
                       const std::string &commDomain)
    : Runner(name), runnerType_(runnerType), rank_(rank), rankSize_(rankSize), rankRoot_(rankRoot),
      commDomain_(commDomain)
{
    ATB_LOG(INFO) << GetLogPrefix() << "construct, use rank:" << rank << ", rankSize:" << rankSize
                  << ", rankRoot:" << rankRoot << ", commDomain_:" << commDomain_;
    Init();
}

HcclRunner::HcclRunner(const std::string &name, RunnerType runnerType, int rank, const std::string &rankTableFile,
                       const std::string &commDomain)
    : Runner(name), runnerType_(runnerType), rank_(rank), rankTableFile_(rankTableFile), commDomain_(commDomain)
{
    useRankTableFile_ = true;
    ATB_LOG(INFO) << GetLogPrefix() << "construct by rankTableFile, use rank:" << rank
                  << ", rankTableFile_:" << rankTableFile << ", commDomain_:" << commDomain_;
    Init();
}

HcclRunner::HcclRunner(const std::string &name, HcclComm hcclComm, RunnerType runnerType)
    : Runner(name), runnerType_(runnerType)
{
    if (!hcclComm) {
        ATB_LOG(ERROR) << GetLogPrefix() << "construct fail, hcclComm is null";
        return;
    }

#ifdef _DEBUG
    ATB_LOG(INFO) << GetLogPrefix() << "construct, use hcclComm:" << hcclComm;
#else
    ATB_LOG(INFO) << GetLogPrefix() << "construct, use hcclComm";
#endif
    hcclComm_ = HcclCommSharedPtr(
        hcclComm, [](const void *hcclComm) { (void)hcclComm; }); // hcclComm由外部传入时，Runner不负责释放
}

HcclRunner::~HcclRunner()
{
    ATB_LOG(INFO) << "HcclRunner deconstruct";
}

HcclCommSharedPtr HcclRunner::GetHcclCommSharedPtr() const
{
    return hcclComm_;
}

void HcclRunner::Init()
{
    hcclComm_ = GetSingleton<CommPool<void>>().GetComm(std::to_string(rank_) + "_" + commDomain_,
                                                       std::bind(&HcclRunner::CreateHcclComm, this));
    if (hcclComm_) {
        ATB_LOG(INFO) << GetLogPrefix() << "get hccl comm success by rank:" << rank_;
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << "get hccl comm fail by rank:" << rank_;
    }
}

HcclCommSharedPtr HcclRunner::CreateHcclComm()
{
    ATB_LOG(INFO) << GetLogPrefix() << "create hccl comm start, rank:" << rank_ << ", rankSize:" << rankSize_;
    return CreateHcclCommInMulitProcess();
}

HcclCommSharedPtr HcclRunner::CreateHcclCommInMulitProcess()
{
    if (!useRankTableFile_) {
        return CreateHcclCommInMulitProcessByRootInfo();
    } else {
        return CreateHcclCommInMulitProcessByRankFile();
    }
}

HcclCommSharedPtr HcclRunner::CreateHcclCommInMulitProcessByRankFile() const
{
    ATB_LOG(INFO) << "HCCL Runner multi server init ";
    HcclComm newHcclComm = nullptr;
    std::string resolvePath = Mki::FileSystem::PathCheckAndRegular(rankTableFile_);
    if (resolvePath == "") {
        ATB_LOG(ERROR) << "realpath fail, filePath:" << rankTableFile_;
        return HcclCommSharedPtr();
    }
    ATB_LOG(INFO) << GetLogPrefix() << "rankTableFilePath is :" << resolvePath;
    auto ret = HcclCommInitClusterInfo(resolvePath.c_str(), rank_, &newHcclComm);
    if (ret != HCCL_SUCCESS || newHcclComm == nullptr) {
        ATB_LOG(ERROR) << "HCCL CommInitClusterInfo ERROR" << ret << " should check rankTableFile config";
        return HcclCommSharedPtr();
    }
#ifdef _DEBUG
    ATB_LOG(INFO) << GetLogPrefix() << "HcclCommInitClusterInfo success, rank:" << rank_ << ", rankSize:" << rankSize_
                  << ", newHcclComm:" << newHcclComm;
#else
    ATB_LOG(INFO) << GetLogPrefix() << "HcclCommInitClusterInfo success, rank:" << rank_ << ", rankSize:" << rankSize_;
#endif
    return HcclCommSharedPtr(newHcclComm, [=](void *hcclComm) {
        (void)hcclComm;
#ifdef _DEBUG
        ATB_LOG(INFO) << "destroy hcclComm, but not call HcclCommDestroy hcclComm:" << hcclComm;
#else
        ATB_LOG(INFO) << "destroy hcclComm, but not call HcclCommDestroy";
#endif
    });
}

HcclCommSharedPtr HcclRunner::CreateHcclCommInMulitProcessByRootInfo()
{
    ATB_LOG(INFO) << "HCCL Runner single server init ";
    if (!CreateHcclRootInfo()) {
        return HcclCommSharedPtr();
    }

    HcclComm newHcclComm = nullptr;
    auto ret = HcclCommInitRootInfo(rankSize_, &hcclRootInfo_, rank_, &newHcclComm);
    if (ret != HCCL_SUCCESS || newHcclComm == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "HcclCommInitRootInfo fail, error:" << ret << ", rank:" << rank_
                       << ", rankSize:" << rankSize_;
        return HcclCommSharedPtr();
    }
#ifdef _DEBUG
    ATB_LOG(INFO) << GetLogPrefix() << "HcclCommInitRootInfo success, rank:" << rank_ << ", rankSize:" << rankSize_
                  << ", newHcclComm:" << newHcclComm;
#else
    ATB_LOG(INFO) << GetLogPrefix() << "HcclCommInitRootInfo success, rank:" << rank_ << ", rankSize:" << rankSize_;
#endif

    return HcclCommSharedPtr(newHcclComm, [=](void *hcclComm) {
        (void)hcclComm;
#ifdef _DEBUG
        ATB_LOG(INFO) << "destroy hcclComm, but not call HcclCommDestroy hcclComm:" << hcclComm;
#else
        ATB_LOG(INFO) << "destroy hcclComm, but not call HcclCommDestroy";
#endif
    });
}

bool HcclRunner::CreateHcclRootInfo()
{
    std::string shmName = "hcclShareMem" + commDomain_;
    Mki::ShareMemory shm(shmName, sizeof(atb::CommInitInfo) + rankSize_ * sizeof(bool));
    auto *shmInfo = static_cast<atb::CommInitInfo *>(shm.GetShm());
    if (!shmInfo) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create share memory fail, rank:" << rank_;
        return false;
    }

    // 主进程通过HcclGetRootInfo获取到hcclRootInfo_(包含HostIP信息), 写到共享内存，其他进程读取RoortInfo
    // 等所有的进程都准备好时，再一起往下执行CreateHcclComm
    ATB_LOG(INFO) << GetLogPrefix() << "create share memory success, rank:" << rank_;
    if (rank_ == rankRoot_) {
        auto ret = HcclGetRootInfo(&hcclRootInfo_);
        if (ret != HCCL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "HcclGetRootInfo fail, error:" << ret << ", rank:" << rank_;
            return false;
        }
        ATB_LOG(INFO) << GetLogPrefix() << "HcclGetRootInfo success, write to share memory";
        ShmSetHcclRootInfo(shm, *shmInfo);
    } else {
        ATB_LOG(INFO) << GetLogPrefix() << "get root info from share memory";
        ShmGetHcclRootInfo(shm, *shmInfo);
    }

    return ShmBarrier(shm, *shmInfo);
}

void HcclRunner::ShmGetHcclRootInfo(Mki::ShareMemory &shm, const CommInitInfo &shmInfo)
{
    bool commIdReady = false;
    while (true) {
        shm.SemLock();
        if (shmInfo.signal != 0) {
            hcclRootInfo_ = shmInfo.hcclRootInfo;
            commIdReady = true;
        }
        shm.SemUnLock();
        if (commIdReady) {
            break;
        }
    }
}

void HcclRunner::ShmSetHcclRootInfo(Mki::ShareMemory &shm, CommInitInfo &shmInfo)
{
    shm.SemLock();
    shmInfo.hcclRootInfo = hcclRootInfo_;
    shmInfo.signal = 1;
    shm.SemUnLock();
}

void HcclRunner::ShmSetReady(Mki::ShareMemory &shm, CommInitInfo &shmInfo) const
{
    shm.SemLock();
    shmInfo.barrier[rank_] = true;
    shm.SemUnLock();
}

bool HcclRunner::ShmBarrier(Mki::ShareMemory &shm, CommInitInfo &shmInfo)
{
    ATB_LOG(INFO) << GetLogPrefix() << "barrier start, rank:" << rank_;
    ShmSetReady(shm, shmInfo);

    ATB_LOG(INFO) << GetLogPrefix() << "check all ready start";
    const double timeout = 600; // 600: 10 minutes timeout
    time_t startTime = time(nullptr);
    while (true) {
        time_t currentTime = time(nullptr);
        if (difftime(currentTime, startTime) > timeout) {
            ATB_LOG(ERROR) << GetLogPrefix() << "barrier fail, check all ready timeout";
            return false;
        }

        bool allReady = true;
        shm.SemLock();
        for (int i = 0; i < rankSize_; i++) {
            if (!shmInfo.barrier[i]) {
                allReady = false;
                break;
            }
        }
        shm.SemUnLock();
        if (allReady) {
            ATB_LOG(INFO) << GetLogPrefix() << "check all ready success";
            break;
        }
    }

    ATB_LOG(INFO) << GetLogPrefix() << "barrier success, rank:" << rank_;
    return true;
}

static bool IsHcclRunnerTensorValid(const SVector<Tensor> &tensors)
{
    for (const auto &tensor : tensors) {
        if (!tensor.deviceData) {
            ATB_LOG(ERROR) << "tensor devoce is null";
            return false;
        }
    }
    return true;
}

Status HcclRunner::ExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    if (!hcclComm_) {
        return ERROR_COMM_EMPTY;
    }

    if (!IsHcclRunnerTensorValid(runnerVariantPack.inTensors) ||
        !IsHcclRunnerTensorValid(runnerVariantPack.outTensors)) {
        return ERROR_INVALID_PARAM;
    }
    return ErrorType::NO_ERROR;
}

} // namespace atb
