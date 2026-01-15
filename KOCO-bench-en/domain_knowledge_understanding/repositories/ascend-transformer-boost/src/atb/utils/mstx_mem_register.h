/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_UTILS_MSTX_REGISTER_H
#define ATB_UTILS_MSTX_REGISTER_H
#include <unistd.h>
#include <syscall.h>
#include <vector>
#include <mstx/ms_tools_ext.h>
#include <mstx/ms_tools_ext_mem.h>
#include "atb/utils/log.h"

namespace atb {
class MstxMemRegister {
public:
    MstxMemRegister();
    ~MstxMemRegister();
    static mstxDomainHandle_t &GetRegisterDomain();
    static bool IsMstxEnable();
    Status CheckTensorRange();
    Status MstxHeapRegister(void *workspace, uint64_t workspaceSize);
    void MstxMemRegionsRegister();
    void MstxMemRegionsUnregister();
    void ClearMstxMemRegions() noexcept;
    void AddTensorMemRegions(void *ptr, uint64_t size);
    int32_t GetMstxDevice();

private:
    thread_local static bool isMstxEnable_;
    static mstxDomainHandle_t domain_;
    mstxMemHeapHandle_t memPool_;
    std::vector<mstxMemVirtualRangeDesc_t> rangesDesc_ = {};
    std::vector<mstxMemRegionHandle_t> regionHandles_ = {};
};

} // namespace atb
#endif // ATB_UTILS_MSTX_REGISTER_H