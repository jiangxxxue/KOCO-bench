/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDTRANSFORM_HEAP_MEM_ALLOCATION_SOLVER_H
#define ASDTRANSFORM_HEAP_MEM_ALLOCATION_SOLVER_H
#include <vector>
#include <memory>
#include "atb/utils/mem_allocation_solver/mem_allocation_solver.h"
#include "block_mem_allocation_solver.h"

namespace atb {
class HeapMemAllocationSolver : public MemAllocationSolver {
public:
    HeapMemAllocationSolver();
    ~HeapMemAllocationSolver() override;
    void *GetOffset(int64_t blockSize) override;
    void Free(void *blockAddress) override;
    void Reset() override;

private:
    void RemoveUselessBlock();

private:
    std::vector<Block> blocks_;
    int64_t curSize_ = 0;
};
} // namespace atb
#endif