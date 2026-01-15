/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_RMS_NORM_FORWARD_RMS_NORM_H_
#define ASCEND_OPS_RMS_NORM_FORWARD_RMS_NORM_H_
#include "rms_norm_base.h"

using namespace AscendC;

template <typename T>
class KernelRmsNorm {
public:
    __aicore__ inline KernelRmsNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma,
                                GM_ADDR y, GM_ADDR rstd,
                                uint32_t numRow_,
                                uint32_t numCol_,
                                uint32_t blockFactor_,
                                uint32_t rowFactor_,
                                uint32_t ubFactor_,
                                float epsilon_,
                                float avgFactor_)
    {
        this->numRow = numRow_;
        this->numCol = numCol_;
        this->blockFactor = blockFactor_;
        this->rowFactor = rowFactor_;
        this->ubFactor = ubFactor_;
        this->epsilon = epsilon_;
        this->avgFactor = (numCol == 0) ? 0.0f : 1.0f / numCol;

        if (GetBlockIdx() < GetBlockNum() - 1) {
            this->row_work = blockFactor;
        } else if (GetBlockIdx() == GetBlockNum() - 1) {
            this->row_work = numRow - (GetBlockNum() - 1) * blockFactor;
        } else {}
        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x + GetBlockIdx() * blockFactor * numCol, row_work * numCol);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);
        yGm.SetGlobalBuffer((__gm__ T*)y + GetBlockIdx() * blockFactor * numCol, row_work * numCol);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor, blockFactor);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, ubFactor * sizeof(T));
        pipe.InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, ubFactor * sizeof(T));
        pipe.InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));

        if constexpr (RmsNormTypeCheck<T, half>::value || RmsNormTypeCheck<T, bfloat16_t>::value) {
            pipe.InitBuffer(x_fp32_buf, ubFactor * sizeof(float));
        }
        pipe.InitBuffer(sqx_buf, ubFactor * sizeof(float));
        pipe.InitBuffer(reduce_fp32_buf, NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();

        uint32_t i_o_max = CeilDiv(row_work, rowFactor);
        uint32_t row_tail = row_work - (i_o_max - 1) * rowFactor;

        for (uint64_t i_o = 0; i_o < i_o_max - 1; i_o++) {
            SubProcess(i_o, rowFactor, gammaLocal);
        }
        SubProcess(i_o_max - 1, row_tail, gammaLocal);
        inQueueGamma.FreeTensor(gammaLocal);
    }

    __aicore__ inline void SubProcess(uint64_t i_o, uint32_t calc_row_num, LocalTensor<T>& gammaLocal)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        for (uint64_t i_i = 0; i_i < calc_row_num; i_i++) {
            uint64_t gm_bias = (i_o * rowFactor + i_i) * numCol;
            CopyIn(gm_bias);
            Compute(i_i, gammaLocal, rstdLocal);
            CopyOutY(gm_bias);
        }
        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(i_o, calc_row_num);
    }
private:
    __aicore__ inline void CopyIn(uint64_t gm_bias)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyCustom<T>(xLocal, xGm[gm_bias], numCol);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void Compute(uint32_t inner_progress, LocalTensor<half> gammaLocal, LocalTensor<float> rstdLocal)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<float> x_fp32 = x_fp32_buf.Get<float>();
        LocalTensor<float> sqx = sqx_buf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduce_fp32_buf.Get<float>();

        Cast(x_fp32, xLocal, RoundMode::CAST_NONE, numCol);
        inQueueX.FreeTensor(xLocal);
        pipe_barrier(PIPE_V);

        Mul(sqx, x_fp32, x_fp32, numCol);
        pipe_barrier(PIPE_V);

        Muls(sqx, sqx, avgFactor, numCol);
        pipe_barrier(PIPE_V);

        ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
        pipe_barrier(PIPE_V);

        Adds(sqx, sqx, epsilon, 1);
        pipe_barrier(PIPE_V);

        Sqrt(sqx, sqx, 1);
        Duplicate(reduce_buf_local, ONE, 1);
        pipe_barrier(PIPE_V);
        Div(sqx, reduce_buf_local, sqx, 1);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float rstdValue = sqx.GetValue(0);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        rstdLocal.SetValue(inner_progress, rstdValue);
        pipe_barrier(PIPE_V);
        Muls(x_fp32, x_fp32, rstdValue, numCol);
        pipe_barrier(PIPE_V);
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        Cast(yLocal, x_fp32, RoundMode::CAST_NONE, numCol);
        pipe_barrier(PIPE_V);
        Mul(yLocal, gammaLocal, yLocal, numCol);
        pipe_barrier(PIPE_V);
        outQueueY.EnQue<half>(yLocal);
    }

    __aicore__ inline void Compute(uint32_t inner_progress, LocalTensor<float> gammaLocal, LocalTensor<float> rstdLocal)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> sqx = sqx_buf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduce_fp32_buf.Get<float>();
        Mul(sqx, xLocal, xLocal, numCol);
        pipe_barrier(PIPE_V);

        Muls(sqx, sqx, avgFactor, numCol);
        pipe_barrier(PIPE_V);

        ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
        pipe_barrier(PIPE_V);
        Adds(sqx, sqx, epsilon, 1);
        pipe_barrier(PIPE_V);

        Sqrt(sqx, sqx, 1);
        Duplicate(reduce_buf_local, ONE, 1);
        pipe_barrier(PIPE_V);
        Div(sqx, reduce_buf_local, sqx, 1);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float rstdValue = sqx.GetValue(0);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        rstdLocal.SetValue(inner_progress, rstdValue);
        pipe_barrier(PIPE_V);
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        Muls(yLocal, xLocal, rstdValue, numCol);
        inQueueX.FreeTensor(xLocal);
        pipe_barrier(PIPE_V);
        Mul(yLocal, gammaLocal, yLocal, numCol);
        pipe_barrier(PIPE_V);
        outQueueY.EnQue<float>(yLocal);
    }

    __aicore__ inline void Compute(uint32_t inner_progress,
                                   LocalTensor<bfloat16_t> gammaLocal,
                                   LocalTensor<float> rstdLocal)
    {
        LocalTensor<bfloat16_t> xLocal = inQueueX.DeQue<bfloat16_t>();
        LocalTensor<float> x_fp32 = x_fp32_buf.Get<float>();
        LocalTensor<float> sqx = sqx_buf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduce_fp32_buf.Get<float>();

        Cast(x_fp32, xLocal, RoundMode::CAST_NONE, numCol);
        inQueueX.FreeTensor(xLocal);
        pipe_barrier(PIPE_V);

        Mul(sqx, x_fp32, x_fp32, numCol);
        pipe_barrier(PIPE_V);

        Muls(sqx, sqx, avgFactor, numCol);
        pipe_barrier(PIPE_V);
        ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
        pipe_barrier(PIPE_V);

        Adds(sqx, sqx, epsilon, 1);
        pipe_barrier(PIPE_V);

        Sqrt(sqx, sqx, 1);
        Duplicate(reduce_buf_local, ONE, 1);
        pipe_barrier(PIPE_V);
        Div(sqx, reduce_buf_local, sqx, 1);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float rstdValue = sqx.GetValue(0);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        rstdLocal.SetValue(inner_progress, rstdValue);
        pipe_barrier(PIPE_V);
        Muls(x_fp32, x_fp32, rstdValue, numCol);
        pipe_barrier(PIPE_V);
        LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
        Cast(yLocal, x_fp32, RoundMode::CAST_RINT, numCol);
        pipe_barrier(PIPE_V);
        Cast(x_fp32, yLocal, RoundMode::CAST_NONE, numCol);
        pipe_barrier(PIPE_V);
        Cast(sqx, gammaLocal, RoundMode::CAST_NONE, numCol);    // gamma_fp32 reuse sqx
        pipe_barrier(PIPE_V);
        Mul(x_fp32, x_fp32, sqx, numCol);
        pipe_barrier(PIPE_V);
        Cast(yLocal, x_fp32, RoundMode::CAST_RINT, numCol);
        pipe_barrier(PIPE_V);
        outQueueY.EnQue<bfloat16_t>(yLocal);
    }

    __aicore__ inline void CopyOutY(uint64_t progress)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyCustom<T>(yGm[progress], yLocal, numCol);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint32_t outer_progress, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
#if __CCE_AICORE__ == 220
            DataCopyCustom<float>(rstdGm[outer_progress * rowFactor], rstdLocal, num);
#endif
        outQueueRstd.FreeTensor(rstdLocal);
    }

private:
    TPipe pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueGamma;
    // create queues for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY, outQueueRstd;

    TBuf<TPosition::VECCALC> x_fp32_buf;
    TBuf<TPosition::VECCALC> sqx_buf;
    TBuf<TPosition::VECCALC> reduce_fp32_buf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> rstdGm;

    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor; // number of calculations rows on each core
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    uint32_t row_work = 1;
};
#endif // ASCEND_OPS_RMS_NORM_FORWARD_RMS_NORM_H_