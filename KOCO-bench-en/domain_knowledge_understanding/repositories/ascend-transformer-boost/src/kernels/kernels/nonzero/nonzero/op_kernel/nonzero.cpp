
/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "kernels/nonzero/nonzero/tiling/tiling_data.h"
#include "kernels/utils/kernel/kernel_utils.h"

namespace {
static constexpr int32_t BUFFER_NUM = 1;

class Nonzero {
public:
    __aicore__ inline Nonzero() {}
    __aicore__ inline void Init(__gm__ uint8_t *x,
                                __gm__ uint8_t *y,
                                __gm__ uint8_t *numTrues,
                                uint32_t *xDims,
                                uint32_t xdimLength,
                                uint32_t xNumel) {
        xDims_ = xDims;
        xdimLength_ = xdimLength;
        xNumel_ = xNumel;

        xGm_ = (__gm__ int64_t *)x;
        yGm_ = (__gm__ int64_t *)y;
        numTruesGm_ = (__gm__ int64_t *)numTrues;
    }
    __aicore__ inline void Process()
    {
        int64_t numTrues = 0;

        for (uint32_t i = 0; i < xNumel_; i++) {
            if (xGm_[i]) {
                uint32_t numelLeft = xNumel_;
                uint32_t tmp = i;
                for (uint32_t j = 0; j < xdimLength_; j++) {
                    numelLeft /=  xDims_[j];
                    uint32_t idxThis = tmp / numelLeft;
                    __gm__ int64_t *y_now = yGm_ + j * xNumel_;
                    *(y_now + numTrues) = idxThis;
                    tmp %= numelLeft;
                }

                numTrues++;
            }
        }

        numTruesGm_[0] = numTrues;
    }

private:
    __gm__ int64_t *xGm_;
    __gm__ int64_t *yGm_;
    __gm__ int64_t *numTruesGm_;
    uint32_t xdimLength_{1};
    uint32_t xNumel_{1};
    uint32_t *xDims_{nullptr};
};
}

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::NonzeroTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->xdimLength = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->xNumel = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    for (uint32_t i = 0; i < tilingdata->xdimLength; i++) {
        tilingdata->xDims[i] = (*(const __gm__ uint32_t *)(p_tilingdata + 8 + i * sizeof(uint32_t)));
    }
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::NonzeroTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->xdimLength = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->xNumel = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    for (uint32_t i = 0; i < tilingdata->xdimLength; i++) {
        tilingdata->xDims[i] = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8 + i * sizeof(uint32_t)));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}


#define GET_TILING_DATA(tiling_data, tiling_arg)    \
    AsdOps::NonzeroTilingData tiling_data;      \
    InitTilingData((tiling_arg), &(tiling_data))

extern "C" __global__ __aicore__ void nonzero(GM_ADDR x, GM_ADDR y, GM_ADDR numTrues,
                                                    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    Nonzero op;
    op.Init(x, y, numTrues, tiling_data.xDims, tiling_data.xdimLength, tiling_data.xNumel);
    op.Process();
}