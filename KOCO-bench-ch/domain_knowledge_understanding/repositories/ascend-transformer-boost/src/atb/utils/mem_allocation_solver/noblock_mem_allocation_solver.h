/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDTRANSFORM_NOBLOCK_MEM_ALLOCATION_SOLVER_H
#define ASDTRANSFORM_NOBLOCK_MEM_ALLOCATION_SOLVER_H
#include <map>
#include "atb/utils/mem_allocation_solver/mem_allocation_solver.h"

namespace atb {
class NoblockMemAllocationSolver : public MemAllocationSolver {
public:
    NoblockMemAllocationSolver();
    ~NoblockMemAllocationSolver() override;
    void *GetOffset(int64_t blockSize) override;
    void Free(void *blockAddress) override;
    void Reset() override;

private:
    void *TailExpand(int64_t blockSize, int64_t newOccupiedIntervalStart, int64_t newOccupiedIntervalEnd);

private:
    std::map<int64_t, int64_t> availableIntervals_; // 空闲的区间，左闭右开
    std::map<int64_t, int64_t> occupiedIntervals_;  // 已被占用的区间
};
} // namespace atb
#endif