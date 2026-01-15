/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "noblock_mem_allocation_solver.h"
#include "atb/utils/log.h"
#include "atb/utils/config.h"

namespace atb {
NoblockMemAllocationSolver::NoblockMemAllocationSolver() {}

NoblockMemAllocationSolver::~NoblockMemAllocationSolver() {}

void *NoblockMemAllocationSolver::GetOffset(int64_t blockSize)
{
    if (blockSize <= 0) {
        ATB_LOG(ERROR) << "BlockMemAllocationSolver::GetOffset, blockSize invalid";
        blockSize = 0;
    }
    ATB_LOG(INFO) << "NoblockMemAllocationSolver::GetOffset, blockSize:" << blockSize;
    mallocTotalSize_ += blockSize;
    bool findInterval = false;
    int64_t resultOffset = 0;
    int64_t newOccupiedIntervalStart = 0;
    int64_t newOccupiedIntervalEnd = 0;
    int64_t newAvailableIntervalStart = 0;
    int64_t newAvailableIntervalEnd = 0;
    std::map<int64_t, int64_t>::iterator iter = availableIntervals_.begin();
    while (iter != availableIntervals_.end()) {
        // 找满足要求的第一个区间
        if (iter->second - iter->first >= blockSize) {
            findInterval = true;
            resultOffset = iter->first;
            newOccupiedIntervalStart = resultOffset;
            newOccupiedIntervalEnd = resultOffset + blockSize;
            newAvailableIntervalStart = resultOffset + blockSize;
            newAvailableIntervalEnd = iter->second;
            availableIntervals_.erase(iter++);
            break;
        }
        iter++;
    }
    if (findInterval) {
        // 更新两个map
        occupiedIntervals_.insert(std::make_pair(newOccupiedIntervalStart, newOccupiedIntervalEnd));
        if (newAvailableIntervalStart != newAvailableIntervalEnd) {
            // 如果新的空闲区间空间为0，则不更新
            // 合并空闲区间
            iter = availableIntervals_.find(newAvailableIntervalEnd);
            if (iter != availableIntervals_.end()) {
                newAvailableIntervalEnd = iter->second;
                availableIntervals_.erase(iter);
            }
            availableIntervals_.insert(std::make_pair(newAvailableIntervalStart, newAvailableIntervalEnd));
        }
        ATB_LOG(INFO) << "NoblockMemAllocationSolver::GetOffset, newTotalSize:" << totalSize_;
        return reinterpret_cast<void *>(resultOffset);
    } else {
        // 尾区间
        return TailExpand(blockSize, newOccupiedIntervalStart, newOccupiedIntervalEnd);
    }
    return nullptr;
}

void NoblockMemAllocationSolver::Free(void *blockAddress)
{
    ATB_LOG(INFO) << "NoblockMemAllocationSolver::Free, blockAddress:" << reinterpret_cast<uintptr_t>(blockAddress);
    intptr_t freeOffset = reinterpret_cast<intptr_t>(blockAddress);
    // 更新占用map
    int64_t newAvailableIntervalStart = freeOffset;
    int64_t newAvailableIntervalEnd = 0;
    std::map<int64_t, int64_t>::iterator iter = occupiedIntervals_.find(freeOffset);
    if (iter == occupiedIntervals_.end()) {
        return;
    }
    newAvailableIntervalEnd = iter->second;
    occupiedIntervals_.erase(iter);
    // 合并空闲区间
    iter = availableIntervals_.begin();
    while (iter != availableIntervals_.end()) {
        if (iter->second == newAvailableIntervalStart) {
            newAvailableIntervalStart = iter->first;
            availableIntervals_.erase(iter++);
            break;
        }
        iter++;
    }
    iter = availableIntervals_.find(newAvailableIntervalEnd);
    if (iter != availableIntervals_.end()) {
        newAvailableIntervalEnd = iter->second;
        availableIntervals_.erase(iter);
    }
    // 更新空闲map
    availableIntervals_.insert(std::make_pair(newAvailableIntervalStart, newAvailableIntervalEnd));
}

void *NoblockMemAllocationSolver::TailExpand(int64_t blockSize, int64_t newOccupiedIntervalStart,
                                             int64_t newOccupiedIntervalEnd)
{
    bool tailAvailable = !availableIntervals_.empty() && availableIntervals_.rbegin()->second == totalSize_;
    if (tailAvailable) {
        newOccupiedIntervalStart = availableIntervals_.rbegin()->first;
        newOccupiedIntervalEnd = newOccupiedIntervalStart + blockSize;
        totalSize_ = newOccupiedIntervalEnd;
        availableIntervals_.erase(availableIntervals_.rbegin()->first);
    } else {
        newOccupiedIntervalStart = totalSize_;
        newOccupiedIntervalEnd = totalSize_ + blockSize;
        totalSize_ += blockSize;
    }
    ATB_LOG(INFO) << "NoblockMemAllocationSolver::Malloc, newTotalSize:" << totalSize_;
    // 更新map
    occupiedIntervals_.insert(std::make_pair(newOccupiedIntervalStart, newOccupiedIntervalEnd));
    return reinterpret_cast<void *>(newOccupiedIntervalStart);
}

void NoblockMemAllocationSolver::Reset()
{
    availableIntervals_.clear();
    occupiedIntervals_.clear();
    MemAllocationSolver::Reset();
}
} // namespace atb
