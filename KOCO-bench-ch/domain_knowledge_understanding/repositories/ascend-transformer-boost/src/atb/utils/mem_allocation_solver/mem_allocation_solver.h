/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDTRANSFORM_MEM_ALLOCATION_SOLVER_H
#define ASDTRANSFORM_MEM_ALLOCATION_SOLVER_H
#include <cstdint>

namespace atb {
class MemAllocationSolver {
public:
    MemAllocationSolver();
    virtual ~MemAllocationSolver();
    virtual void *GetOffset(int64_t blockSize) = 0;
    virtual void Free(void *blockAddress) = 0;
    virtual uint64_t GetSize() const;
    virtual int64_t GetMallocSize() const;
    virtual void Reset();

protected:
    int64_t totalSize_ = 0;
    int64_t mallocTotalSize_ = 0;
};
} // namespace atb
#endif