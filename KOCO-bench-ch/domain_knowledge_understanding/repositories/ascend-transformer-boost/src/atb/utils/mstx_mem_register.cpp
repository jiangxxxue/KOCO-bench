/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/mstx_mem_register.h"
#include <unistd.h>
#include <syscall.h>
#include <mstx/ms_tools_ext.h>
#include <mstx/ms_tools_ext_mem.h>
#include "atb/utils/log.h"

static constexpr int32_t DEVICE_UNDEFINED_STATUS = -1;

namespace atb {
thread_local bool MstxMemRegister::isMstxEnable_ = true;

MstxMemRegister::MstxMemRegister() {}

MstxMemRegister::~MstxMemRegister()
{
    if (memPool_) {
        mstxMemHeapUnregister(GetRegisterDomain(), memPool_);
    }
}

mstxDomainHandle_t &MstxMemRegister::GetRegisterDomain()
{
    static mstxDomainHandle_t domain = mstxDomainCreateA("atb");
    return domain;
}

Status MstxMemRegister::MstxHeapRegister(void *workspace, uint64_t workspaceSize)
{
    mstxMemVirtualRangeDesc_t rangeDesc = {};
    rangeDesc.deviceId = GetMstxDevice();
    rangeDesc.ptr = workspace;
    rangeDesc.size = workspaceSize;

    mstxMemHeapDesc_t heapDesc = {};
    heapDesc.usage = MSTX_MEM_HEAP_USAGE_TYPE_SUB_ALLOCATOR;
    heapDesc.type = MSTX_MEM_TYPE_VIRTUAL_ADDRESS;
    heapDesc.typeSpecificDesc = &rangeDesc;
    
    memPool_ = mstxMemHeapRegister(GetRegisterDomain(), &heapDesc);
    if (memPool_ == nullptr) {
        isMstxEnable_ = false;
        return ERROR_INTERNAL_ERROR;
    }
    return NO_ERROR;
}

bool MstxMemRegister::IsMstxEnable()
{
    return isMstxEnable_;
}

void MstxMemRegister::MstxMemRegionsRegister()
{
    regionHandles_.resize(rangesDesc_.size());

    mstxMemRegionsRegisterBatch_t regionsDesc{};
    regionsDesc.heap = memPool_;
    regionsDesc.regionType = MSTX_MEM_TYPE_VIRTUAL_ADDRESS;
    regionsDesc.regionCount = rangesDesc_.size();
    regionsDesc.regionDescArray = rangesDesc_.data();
    regionsDesc.regionHandleArrayOut = regionHandles_.data();

    mstxMemRegionsRegister(GetRegisterDomain(), &regionsDesc);
}

void MstxMemRegister::MstxMemRegionsUnregister()
{
    mstxMemRegionRef_t regionRef[rangesDesc_.size()] = {};
    for (size_t i = 0; i < regionHandles_.size(); ++i) {
        regionRef[i].refType = MSTX_MEM_REGION_REF_TYPE_HANDLE;
        regionRef[i].handle = regionHandles_[i];
    }
    mstxMemRegionsUnregisterBatch_t refsDesc = {};
    refsDesc.refCount = rangesDesc_.size();
    refsDesc.refArray = regionRef;
    mstxMemRegionsUnregister(GetRegisterDomain(), &refsDesc);
}

void MstxMemRegister::ClearMstxMemRegions() noexcept
{
    rangesDesc_.clear();
    regionHandles_.clear();
}

int32_t MstxMemRegister::GetMstxDevice()
{
    int32_t deviceId = DEVICE_UNDEFINED_STATUS;
    aclError aclRet = aclrtGetDevice(&deviceId);
    if (aclRet != ACL_SUCCESS) {
        ATB_LOG(ERROR) << "get device id error!";
        deviceId = DEVICE_UNDEFINED_STATUS;
    }
    return deviceId;
}

void MstxMemRegister::AddTensorMemRegions(void *ptr, uint64_t size)
{
    if (GetMstxDevice() >= 0) {
        mstxMemVirtualRangeDesc_t tensorInfo;
        tensorInfo.deviceId = static_cast<uint32_t>(GetMstxDevice());
        tensorInfo.ptr = ptr;
        tensorInfo.size = size;
        rangesDesc_.push_back(tensorInfo);
    }
}

Status MstxMemRegister::CheckTensorRange()
{
    if (rangesDesc_.empty()) {
        return false;
    } else {
        return true;
    }
}

}  // namespace atb
