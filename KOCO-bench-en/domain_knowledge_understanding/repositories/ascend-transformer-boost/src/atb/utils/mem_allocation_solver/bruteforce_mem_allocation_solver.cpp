/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "bruteforce_mem_allocation_solver.h"
#include "atb/utils/log.h"
#include "atb/utils/config.h"

namespace atb {
BruteforceMemAllocationSolver::BruteforceMemAllocationSolver() {}

BruteforceMemAllocationSolver::~BruteforceMemAllocationSolver() {}

void *BruteforceMemAllocationSolver::GetOffset(int64_t blockSize)
{
    if (blockSize <= 0) {
        ATB_LOG(ERROR) << "BlockMemAllocationSolver::GetOffset, blockSize invalid";
        blockSize = 0;
    }
    ATB_LOG(INFO) << "BruteforceMemAllocationSolver::GetOffset, blockSize:" << blockSize;
    mallocTotalSize_ += blockSize;

    BruteBlock newBlock;
    newBlock.offset = totalSize_;
    newBlock.size = blockSize;
    newBlock.used = true;
    totalSize_ += blockSize;
    ATB_LOG(INFO) << "BruteforceMemAllocationSolver::GetOffset, newTotalSize:" << totalSize_;
    blocks_.push_back(newBlock);

    return reinterpret_cast<void *>(newBlock.offset);
}

void BruteforceMemAllocationSolver::Free(void *blockAddress)
{
    ATB_LOG(INFO) << "BruteforceMemAllocationSolver::Free, blockAddress:" << reinterpret_cast<uintptr_t>(blockAddress);
    for (auto &block : blocks_) {
        intptr_t offset = reinterpret_cast<intptr_t>(blockAddress);
        if (offset == block.offset) {
            block.used = false;
        }
    }
}

void BruteforceMemAllocationSolver::Reset()
{
    blocks_.clear();
    MemAllocationSolver::Reset();
}

} // namespace atb