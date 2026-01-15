/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DYNAMIC_QUANT_UNALIGN_310P_H
#define DYNAMIC_QUANT_UNALIGN_310P_H

#include "kernel_operator.h"
#include "dynamic_quant_align.h"
#include "kernels/elewise/dynamic_quant/dynamic_quant_tiling/dynamic_quant_util.h"


class DynamicQuantUnalign310P : public DynamicQuantAlign<half> {
public:
    __aicore__ inline DynamicQuantUnalign310P() {};

    __aicore__ inline void InitBuffer(__gm__ uint8_t *x, __gm__ uint8_t *z, __gm__ uint8_t *scale,
        __gm__ uint8_t *offset)
    {
        if (isTailLoop_) {
            inGm_.SetGlobalBuffer((__gm__ half*)x + numHeadCore_ * lenHead_ + \
                (blockIdx_ - numHeadCore_) * lenTail_, lenTail_ + lenLastTail_);
            outGm_.SetGlobalBuffer((__gm__ int8_t*)z + numHeadCore_ * lenHead_ + \
                (blockIdx_ - numHeadCore_) * lenTail_, lenTail_ + lenLastTail_);
            scaleGm_.SetGlobalBuffer((__gm__ float*)scale + numHeadCore_ * numHeadRow_ + \
                (blockIdx_ - numHeadCore_) * numTailRow_, numTailRow_ + numCopyRow_);
            if (asymmetric_) {
                offsetGm_.SetGlobalBuffer((__gm__ float*)offset + numHeadCore_ * numHeadRow_ + \
                    (blockIdx_ - numHeadCore_) * numTailRow_, numTailRow_ + numCopyRow_);
            }
        } else {
            DynamicQuantAlign::InitBuffer(x, z, scale, offset);
        }
    }

    __aicore__ inline void Process()
    {
        DynamicQuantAlign::ProcessWithHeadDoublePipLineForAlign();
        ProcessWithTailCopyInForUnalign();
        DynamicQuantAlign::ProcessWithHeadLastComputeAndCopyOutForAlign();
        ProcessWithTailComputeAndCopyOutForUnalign();
    }

private:
    __aicore__ inline void ProcessWithTailCopyInForUnalign()
    {
        if (isTailLoop_) {
            DynamicQuantAlign::CopyIn(loopCount_, lenLastTail_);
        }
    }

    __aicore__ inline void ProcessWithTailComputeAndCopyOutForUnalign()
    {
        if (isTailLoop_) {
            if (asymmetric_) {
                DynamicQuantAlign::ComputeAsymmetric(numLastTailRow_, sizeH_);
            } else {
                DynamicQuantAlign::Compute(numLastTailRow_, sizeH_);
            }
            DynamicQuantAlign::CopyOut(loopCount_, lenLastTail_);
        }
    }
};

#endif // DYNAMIC_QUANT_UNALIGN_310P_H
