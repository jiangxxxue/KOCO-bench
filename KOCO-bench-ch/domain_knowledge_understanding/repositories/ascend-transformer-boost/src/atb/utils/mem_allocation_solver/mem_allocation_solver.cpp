/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/mem_allocation_solver/mem_allocation_solver.h"
#include "atb/utils/log.h"

namespace atb {
MemAllocationSolver::MemAllocationSolver() {}

MemAllocationSolver::~MemAllocationSolver() {}

uint64_t MemAllocationSolver::GetSize() const
{
    ATB_LOG(DEBUG) << "MemAllocationSolver::GetSize " << totalSize_;
    return totalSize_;
}

int64_t MemAllocationSolver::GetMallocSize() const
{
    ATB_LOG(DEBUG) << "MemAllocationSolver::GetMallocSize " << mallocTotalSize_;
    return mallocTotalSize_;
}

void MemAllocationSolver::Reset()
{
    totalSize_ = 0;
    mallocTotalSize_ = 0;
}
} // namespace atb