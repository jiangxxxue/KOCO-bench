/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "heap_mem_allocation_solver.h"
#include <limits>
#include "atb/utils/log.h"
#include "atb/utils/config.h"

namespace atb {
HeapMemAllocationSolver::HeapMemAllocationSolver() {}

HeapMemAllocationSolver::~HeapMemAllocationSolver() {}

void *HeapMemAllocationSolver::GetOffset(int64_t blockSize)
{
    if (blockSize <= 0) {
        ATB_LOG(ERROR) << "BlockMemAllocationSolver::GetOffset, blockSize invalid";
        blockSize = 0;
    }
    ATB_LOG(INFO) << "HeapMemAllocationSolver::GetOffset, blockSize:" << blockSize;
    mallocTotalSize_ += blockSize;
    int validBlockIdx = -1;
    int blockIdx = 0;
    int64_t validBlockSize = 0;
    for (auto &block : blocks_) {
        if (!block.used && block.size >= blockSize) {
            if (validBlockSize == 0) {
                validBlockSize = block.size;
                validBlockIdx = blockIdx;
            } else if (block.size < validBlockSize) {
                validBlockIdx = blockIdx;
                validBlockSize = block.size;
            }
        }
        blockIdx++;
    }
    if (validBlockIdx != -1) {
        blocks_.at(validBlockIdx).used = true;
        return reinterpret_cast<void *>(blocks_.at(validBlockIdx).offset);
    }
    Block newBlock;
    newBlock.offset = curSize_;
    newBlock.size = blockSize;
    newBlock.used = true;
    curSize_ += blockSize;
    totalSize_ = std::max(totalSize_, curSize_);
    blocks_.push_back(newBlock);

    ATB_LOG(INFO) << "HeapMemAllocationSolver::Malloc, newTotalSize:" << totalSize_;
    return reinterpret_cast<void *>(newBlock.offset);
}

void HeapMemAllocationSolver::Free(void *blockAddress)
{
    intptr_t offset = reinterpret_cast<intptr_t>(blockAddress);
    ATB_LOG(INFO) << "HeapMemAllocationSolver::Free, blockAddress:" << offset;
    for (auto &block : blocks_) {
        if (offset == block.offset) {
            if (!block.used) {
                ATB_LOG(ERROR) << "block: " << blockAddress << " is not used, free fail";
                return;
            }
            block.used = false;
            RemoveUselessBlock();
            return;
        }
    }
    ATB_LOG(WARN) << "can't find block: " << blockAddress << ", free fail";
}

void HeapMemAllocationSolver::RemoveUselessBlock()
{
    while (!blocks_.empty() && !blocks_.back().used) {
        curSize_ -= blocks_.back().size;
        blocks_.pop_back();
    }
}

void HeapMemAllocationSolver::Reset()
{
    blocks_.clear();
    curSize_ = 0;
    MemAllocationSolver::Reset();
}
} // namespace atb