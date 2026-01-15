/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_RMS_NORM_GRAD_SPLIT_D_H_
#define ASCEND_OPS_RMS_NORM_GRAD_SPLIT_D_H_
#include "rms_norm_grad_common.h"
template <typename T1>
class RmsNormGradSplitD {
public:
    __aicore__ inline RmsNormGradSplitD() {
    }

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd,
                                GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma,
                                GM_ADDR sync, const RmsNormGradTilingData* tiling)
    {
        InitVar(tiling);
        InitInputGmBuffer(dy, x, rstd, gamma, blockDim, coreCalcNum, coreCalcTail);
        InitOutputGmBuffer(dx, dgamma, blockDim, coreCalcNum, coreCalcTail);
        InitInputQue();
        InitOutputQue();
        InitTmpBuffer();
        WaitInitOutput(sync, syncGm_, inQueGamma, blockDim);
    }

    __aicore__ inline void InitVar(const RmsNormGradTilingData* tiling)
    {
        blockDim = tiling->blockDim;
        rowVal = tiling->row;
        colVal = tiling->col;
        avg = tiling->avg;
        dataType = tiling->dataType;
        coreCalcNum = tiling->coreCalcNum;
        coreCalcTail = tiling->coreCalcTail;
        blockFactor = tiling->blockFactor;
        ubFactor = tiling->ubFactor;
        ubCalcNum = tiling->ubCalcNum;
        ubCalcTail = tiling->ubCalcTail;
        ubCalcLoop = tiling->ubCalcLoop;
        ubCalcTailNum = tiling->ubCalcTailNum;
        ubCalcTailTail = tiling->ubCalcTailTail;
        ubCalcTailLoop = tiling->ubCalcTailLoop;
        alignLen = dataType == FLOAT_DTYPE ? ALIGN_32 : ALIGN_16;
        colValAlign = (colVal + alignLen - 1) / alignLen * alignLen;
    }

    __aicore__ inline void InitInputGmBuffer(GM_ADDR dy, GM_ADDR x,
                                             GM_ADDR rstd, GM_ADDR gamma,
                                             uint32_t blockDim,
                                             uint32_t coreCalcNum,
                                             uint32_t coreCalcTail)
    {
        if (GetBlockIdx() < blockDim - 1) {
            coreOffset = blockFactor;
        } else {
            coreOffset = coreCalcTail > 0 ? coreCalcTail : blockFactor;
        }
        coreOffsetStart = blockFactor * colVal;
        coreOffsetLen = coreOffset * colVal;
        dyGm.SetGlobalBuffer((__gm__ T1*)dy + GetBlockIdx() * coreOffsetStart, coreOffsetLen);
        xGm.SetGlobalBuffer((__gm__ T1*)x + GetBlockIdx() * coreOffsetStart, coreOffsetLen);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor, coreOffset);
        gammaGm.SetGlobalBuffer((__gm__ T1*)gamma, colVal);
    }

    __aicore__ inline void InitOutputGmBuffer(GM_ADDR dx, GM_ADDR dgamma,
                                              uint32_t blockDim,
                                              uint32_t coreCalcNum,
                                              uint32_t coreCalcTail)
    {
        dxGm.SetGlobalBuffer((__gm__ T1*)dx + GetBlockIdx() * coreOffsetStart, coreOffsetLen);
        dgammaGm.SetGlobalBuffer((__gm__ float*)dgamma, colVal);
        if (GetBlockIdx() == 0) {
            InitOutput<float>(dgammaGm, colVal, 0);
        }
    }

    __aicore__ inline void InitInputQue()
    {
        ubFactorAlign = (ubFactor + alignLen - 1) / alignLen * alignLen;
        rstdLen = alignLen;
        bufferLenSize = ubFactorAlign * sizeof(T1);
        bufferNum = dataType != FLOAT_DTYPE ? BUFFER_NUM_DB : BUFFER_NUM;
        pipe.InitBuffer(inQueDY, bufferNum, bufferLenSize);
        pipe.InitBuffer(inQueX, bufferNum, bufferLenSize);
        pipe.InitBuffer(inQueRstd, bufferNum, rstdLen * sizeof(float));
        pipe.InitBuffer(inQueGamma, bufferNum, bufferLenSize);
    }

    __aicore__ inline void InitOutputQue()
    {
        pipe.InitBuffer(outQueDX, bufferNum, bufferLenSize);
        pipe.InitBuffer(outQueDgamma, bufferNum, ubFactorAlign * sizeof(float));
    }

    __aicore__ inline void InitTmpBuffer()
    {
        uint32_t tmp_buffer_size = ubFactorAlign * sizeof(float);
        pipe.InitBuffer(ndBufFp32Buf1, tmp_buffer_size);
        pipe.InitBuffer(nFp32Buf, alignLen * sizeof(float));
        if (dataType != FLOAT_DTYPE) {
            pipe.InitBuffer(ndBufFp32Buf2, tmp_buffer_size);
            pipe.InitBuffer(ndBufFp32Buf3, tmp_buffer_size);
            pipe.InitBuffer(dFp32Buf, tmp_buffer_size);
        }
    }

    __aicore__ inline void Process()
    {
        if (coreCalcTail == 0) {
            ProcessMain(blockFactor);
        } else {
            if (GetBlockIdx() < blockDim - 1) {
                ProcessMain(blockFactor);
            } else {
                ProcessMain(coreCalcTail);
            }
        }
    }

    __aicore__ inline void ProcessMain(uint32_t loop_len)
    {
        for (uint64_t i = 0; i < loop_len; i++) {
            // Calc mean firstly
            LocalTensor<float> dySum = nFp32Buf.Get<float>();
            Duplicate(dySum, 0.0f, alignLen);
            pipe_barrier(PIPE_V);
            for (uint64_t j = 0; j < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); j++) {
                ComputeDySum(i, j, ubFactor, ubFactor, dySum);
            }
            if (ubCalcTail != 0) {
                ubTailAlign = (ubCalcTail + alignLen - 1) / alignLen * alignLen;
                ComputeDySum(i, ubCalcLoop - 1, ubCalcTail, ubTailAlign, dySum);
            }
            Muls(dySum, dySum, avg, alignLen);
            pipe_barrier(PIPE_V);
            for (uint64_t j = 0; j < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); j++) {
                CopyGammaIn(j, ubFactor);
                LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
                LocalTensor<float> dgamma;
                CopyIn(i, j, ubFactor, ubFactor);
                ComputeMain(i, j, ubFactor, gamma_ub, dgamma, dySum);
                CopyOut(i, j, ubFactor, ubFactor);
            }
            if (ubCalcTail != 0) {
                CopyGammaIn(ubCalcLoop - 1, ubCalcTail);
                LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
                LocalTensor<float> dgamma;
                CopyIn(i, ubCalcLoop - 1, ubCalcTail, ubTailAlign);
                ComputeMain(i, ubCalcLoop - 1, ubCalcTail, gamma_ub, dgamma, dySum);
                CopyOut(i, ubCalcLoop - 1, ubCalcTail, ubTailAlign);
            }
        }
        for (uint32_t j = 0; j < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); j++) {
            CopyGammaIn(j, ubFactor);
            LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
            LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
            Duplicate(dgamma, 0.0f, ubFactor);
            pipe_barrier(PIPE_V);
            for (uint32_t i = 0; i < loop_len; i++) {
                CopyIn(i, j, ubFactor, ubFactor);
                ComputeDgammaFp32(i, j, ubFactor, gamma_ub, dgamma);
            }
            inQueGamma.FreeTensor(gamma_ub);
            outQueDgamma.EnQue(dgamma);
            CopyDgammaOut(j, ubFactor);
        }
        if (ubCalcTail != 0) {
            CopyGammaIn(ubCalcLoop - 1, ubCalcTail);
            LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
            LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
            Duplicate(dgamma, 0.0f, ubCalcTail);
            pipe_barrier(PIPE_V);
            for (uint32_t i = 0; i < loop_len; i++) {
                CopyIn(i, ubCalcLoop - 1, ubCalcTail, ubTailAlign);
                ComputeDgammaFp32(i, ubCalcLoop - 1, ubCalcTail, gamma_ub, dgamma);
            }
            inQueGamma.FreeTensor(gamma_ub);
            outQueDgamma.EnQue(dgamma);
            CopyDgammaOut(ubCalcLoop - 1, ubCalcTail);
        }
    }

    __aicore__ inline void ComputeDgammaFp32(uint64_t i, uint64_t j,
                                             uint32_t calcLen,
                                             LocalTensor<T1>& gamma_ub,
                                             LocalTensor<float>& dgamma)
    {
        LocalTensor<T1> xUb = inQueX.DeQue<T1>();
        LocalTensor<T1> dyUb = inQueDY.DeQue<T1>();
        LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();
        // dy * (x * rstd) -> sum
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        float rstdValue = rstdUb.GetValue(0);
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
        Muls(xUb, xUb, rstdValue, calcLen);
        pipe_barrier(PIPE_V);
        inQueRstd.FreeTensor(rstdUb);
        Mul(dyUb, dyUb, xUb, calcLen);
        pipe_barrier(PIPE_V);
        inQueX.FreeTensor(xUb);
        Add(dgamma, dgamma, dyUb, calcLen);
        pipe_barrier(PIPE_V);
        inQueDY.FreeTensor(dyUb);
    }

    __aicore__ inline void ComputeDySum(uint64_t i, uint64_t j,
                                        uint32_t calcLen,
                                        uint32_t calc_len_align,
                                        LocalTensor<float>& dySum)
    {
        CopyGammaIn(j, calcLen);
        CopyIn(i, j, calcLen, calc_len_align);
        LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
        LocalTensor<T1> xUb = inQueX.DeQue<T1>();
        LocalTensor<T1> dyUb = inQueDY.DeQue<T1>();
        LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        float rstdValue = rstdUb.GetValue(0);
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
        LocalTensor<float> dySumPart = ndBufFp32Buf1.Get<float>();
        Duplicate(dySumPart, 0.0f, ubTailAlign);
        pipe_barrier(PIPE_V);
        // grad_y = dy * gamma
        Mul(dyUb, dyUb, gamma_ub, calcLen);
        Muls(xUb, xUb, rstdValue, calcLen);
        pipe_barrier(PIPE_V);
        inQueGamma.FreeTensor(gamma_ub);
        inQueRstd.FreeTensor(rstdUb);
        // sum(x * rstd * grad_y)
        Mul(dySumPart, dyUb, xUb, calcLen);
        pipe_barrier(PIPE_V);
        inQueX.FreeTensor(xUb);
        ReduceSumFP32(0, dySumPart, dySumPart, dyUb, calcLen, colValAlign);
        inQueDY.FreeTensor(dyUb);
        Add(dySum, dySum, dySumPart, alignLen);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void CopyGammaIn(uint64_t dIdx, uint32_t calcLen)
    {
        LocalTensor<T1> gamma_ub = inQueGamma.AllocTensor<T1>();
        DataCopyParams data_copy_params{(uint16_t)1, (uint16_t)(calcLen * sizeof(T1)), 0, 0};
        DataCopyPadParams pad_params{false, 0, 0, 0};
        DataCopyPad(gamma_ub, gammaGm[dIdx * ubFactor], data_copy_params, pad_params);
        inQueGamma.EnQue(gamma_ub);
    }

    __aicore__ inline void CopyDgammaOut(uint64_t dIdx, uint32_t calcLen)
    {
        LocalTensor<float> dgamma_out = outQueDgamma.DeQue<float>();
        SetAtomicAdd<float>();
        DataCopyParams data_copy_params{(uint16_t)1, (uint16_t)(calcLen * sizeof(float)), 0, 0};
        DataCopyPad(dgammaGm[dIdx * ubFactor], dgamma_out, data_copy_params);
        SetAtomicNone();
        outQueDgamma.FreeTensor(dgamma_out);
    }

    __aicore__ inline void CopyIn(uint64_t nIdx, uint64_t dIdx, uint32_t calc_unit, uint32_t calc_unit_align)
    {
        // x dy, rstd
        DataCopyParams data_copy_params{(uint16_t)1, (uint16_t)(calc_unit * sizeof(T1)), 0, 0};
        DataCopyPadParams pad_params{true, 0, (uint8_t)(calc_unit_align - calc_unit), 0};
        uint64_t gmOffset = nIdx * colVal + dIdx * ubFactor;
        LocalTensor<float> rstdUb = inQueRstd.AllocTensor<float>();
        DataCopy(rstdUb, rstdGm[nIdx], alignLen);
        inQueRstd.EnQue(rstdUb);
        LocalTensor<T1> xUb = inQueX.AllocTensor<T1>();
        DataCopyPad(xUb, xGm[gmOffset], data_copy_params, pad_params);
        inQueX.EnQue(xUb);
        LocalTensor<T1> dyUb = inQueDY.AllocTensor<T1>();
        DataCopyPad(dyUb, dyGm[gmOffset], data_copy_params, pad_params);
        inQueDY.EnQue(dyUb);
    }

    __aicore__ inline void CopyOut(uint64_t nIdx, uint64_t dIdx, uint32_t calc_unit, uint32_t calc_unit_align)
    {
        LocalTensor<T1> dxUb = outQueDX.DeQue<T1>();
        DataCopyParams data_copy_params{(uint16_t)1, (uint16_t)(calc_unit * sizeof(T1)), 0, 0};
        DataCopyPad(dxGm[nIdx * colVal + dIdx * ubFactor], dxUb, data_copy_params);
        outQueDX.FreeTensor(dxUb);
    }

    __aicore__ inline void ComputeMain(uint64_t nIdx, uint64_t dIdx,
                                       uint32_t calcLen, LocalTensor<T1>& gamma_ub,
                                       LocalTensor<float>& dgamma,
                                       LocalTensor<float>& dySum)
    {
        LocalTensor<T1> xUb = inQueX.DeQue<T1>();
        LocalTensor<T1> dyUb = inQueDY.DeQue<T1>();
        LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        float rstdValue = rstdUb.GetValue(0);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float dySumVal = dySum.GetValue(0);
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);

        Muls(xUb, xUb, rstdValue, calcLen);
        pipe_barrier(PIPE_V);
        inQueRstd.FreeTensor(rstdUb);
        // grad_y = grad* gamma
        Mul(dyUb, dyUb, gamma_ub, calcLen);
        Muls(xUb, xUb, dySumVal, calcLen);
        pipe_barrier(PIPE_V);
        inQueGamma.FreeTensor(gamma_ub);
        Sub(dyUb, dyUb, xUb, calcLen);
        pipe_barrier(PIPE_V);
        inQueX.FreeTensor(xUb);
        LocalTensor<T1> dxUb = outQueDX.AllocTensor<T1>();
        Muls(dxUb, dyUb, rstdValue, calcLen);
        pipe_barrier(PIPE_V);
        inQueDY.FreeTensor(dyUb);
        outQueDX.EnQue(dxUb);
    }

    __aicore__ inline void ProcessFp16()
    {
        if (coreCalcTail == 0) {
            ProcessFp16Main(blockFactor);
        } else {
            if (GetBlockIdx() < blockDim - 1) {
                ProcessFp16Main(blockFactor);
            } else {
                ProcessFp16Main(coreCalcTail);
            }
        }
    }

    __aicore__ inline void ProcessFp16Main(uint32_t loop_len)
    {
        for (uint64_t i = 0; i < loop_len; i++) {
            // Calc mean firstly
            LocalTensor<float> dySum = nFp32Buf.Get<float>();
            Duplicate(dySum, 0.0f, alignLen);
            pipe_barrier(PIPE_V);
            for (uint64_t j = 0; j < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); j++) {
                ComputeDySumFp16(i, j, ubFactor, ubFactor, dySum);
            }
            if (ubCalcTail != 0) {
                ubTailAlign = (ubCalcTail + alignLen - 1) / alignLen * alignLen;
                ComputeDySumFp16(i, ubCalcLoop - 1, ubCalcTail, ubTailAlign, dySum);
            }
            Muls(dySum, dySum, avg, alignLen);
            pipe_barrier(PIPE_V);
            for (uint64_t j = 0; j < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); j++) {
                CopyIn(i, j, ubFactor, ubFactor);
                CopyGammaIn(j, ubFactor);
                LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
                LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
                ComputeMainFp16(i, j, ubFactor, gamma_ub, dgamma, dySum);
                CopyDgammaOut(j, ubFactor);
                CopyOut(i, j, ubFactor, ubFactor);
            }
            if (ubCalcTail != 0) {
                CopyIn(i, ubCalcLoop - 1, ubCalcTail, ubTailAlign);
                CopyGammaIn(ubCalcLoop - 1, ubCalcTail);
                LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
                LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
                ComputeMainFp16(i, ubCalcLoop - 1, ubCalcTail, gamma_ub, dgamma, dySum);
                CopyDgammaOut(ubCalcLoop - 1, ubCalcTail);
                CopyOut(i, ubCalcLoop - 1, ubCalcTail, ubTailAlign);
            }
        }
    }

    __aicore__ inline void ComputeDySumFp16(uint64_t i, uint64_t j,
                                            uint32_t calcLen,
                                            uint32_t calc_len_align,
                                            LocalTensor<float>& dySum)
    {
        CopyGammaIn(j, calcLen);
        CopyIn(i, j, calcLen, calc_len_align);
        LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
        LocalTensor<T1> xUb = inQueX.DeQue<T1>();
        LocalTensor<T1> dyUb = inQueDY.DeQue<T1>();
        LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        float rstdValue = rstdUb.GetValue(0);
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);

        LocalTensor<float> dySumPart = ndBufFp32Buf1.Get<float>();
        LocalTensor<float> tmp32Buf = ndBufFp32Buf2.Get<float>();
        LocalTensor<float> tmp32Buf2 = ndBufFp32Buf3.Get<float>();

        Duplicate(dySumPart, 0.0f, ubTailAlign);
        // grad_y = dy * gamma
        Mul(dyUb, dyUb, gamma_ub, calcLen);
        Cast(tmp32Buf2, xUb, RoundMode::CAST_NONE, calcLen);
        pipe_barrier(PIPE_V);
        inQueGamma.FreeTensor(gamma_ub);
        inQueRstd.FreeTensor(rstdUb);
        inQueX.FreeTensor(xUb);
        Cast(tmp32Buf, dyUb, RoundMode::CAST_NONE, calcLen);
        pipe_barrier(PIPE_V);
        inQueDY.FreeTensor(dyUb);
        // sum(x * rstd * grad_y)
        Muls(tmp32Buf2, tmp32Buf2, rstdValue, calcLen);
        pipe_barrier(PIPE_V);
        Mul(dySumPart, tmp32Buf, tmp32Buf2, calcLen);
        pipe_barrier(PIPE_V);
        ReduceSumFP32(0, dySumPart, dySumPart, tmp32Buf2, calcLen, colValAlign);
        Add(dySum, dySum, dySumPart, alignLen);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void ComputeMainFp16(uint64_t nIdx, uint64_t dIdx,
                                           uint32_t calcLen,
                                           LocalTensor<T1>& gamma_ub,
                                           LocalTensor<float>& dgamma,
                                           LocalTensor<float>& dySum)
    {
        LocalTensor<float> tmp32Buf3 = ndBufFp32Buf3.Get<float>();
        LocalTensor<float> tmp32Buf2 = ndBufFp32Buf2.Get<float>();
        LocalTensor<float> tmp32Buf1 = ndBufFp32Buf1.Get<float>();
        LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();
        LocalTensor<T1> dyUb = inQueDY.DeQue<T1>();
        LocalTensor<T1> xUb = inQueX.DeQue<T1>();
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        float rstdValue = rstdUb.GetValue(0);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float dySumVal = dySum.GetValue(0);
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
        // dg = sum((dy * (x * rstd)), dim=0)
        Cast(tmp32Buf2, xUb, RoundMode::CAST_NONE, calcLen);
        pipe_barrier(PIPE_V);
        inQueRstd.FreeTensor(rstdUb);
        Muls(tmp32Buf2, tmp32Buf2, rstdValue, calcLen);
        pipe_barrier(PIPE_V);
        Cast(xUb, tmp32Buf2, RoundMode::CAST_NONE, calcLen);
        pipe_barrier(PIPE_V);
        Mul(xUb, dyUb, xUb, calcLen);
        pipe_barrier(PIPE_V);
        Mul(dyUb, dyUb, gamma_ub, calcLen);
        Cast(dgamma, xUb, RoundMode::CAST_NONE, calcLen);
        pipe_barrier(PIPE_V);
        inQueGamma.FreeTensor(gamma_ub);
        inQueX.FreeTensor(xUb);
        outQueDgamma.EnQue(dgamma);
        // grad_y = grad* gamma
        Cast(tmp32Buf1, dyUb, RoundMode::CAST_NONE, calcLen);
        // dx = (grad_y - y * mean) * rstd
        Muls(tmp32Buf2, tmp32Buf2, dySumVal, calcLen);
        pipe_barrier(PIPE_V);
        inQueDY.FreeTensor(dyUb);
        Sub(tmp32Buf1, tmp32Buf1, tmp32Buf2, calcLen);
        pipe_barrier(PIPE_V);
        Muls(tmp32Buf1, tmp32Buf1, rstdValue, calcLen);
        pipe_barrier(PIPE_V);
        LocalTensor<T1> dxUb = outQueDX.AllocTensor<T1>();
        Cast(dxUb, tmp32Buf1, RoundMode::CAST_NONE, calcLen);
        pipe_barrier(PIPE_V);
        outQueDX.EnQue(dxUb);
    }

    __aicore__ inline void ProcessBf16()
    {
        if (coreCalcTail == 0) {
            ProcessBf16Main(blockFactor);
        } else {
            if (GetBlockIdx() < blockDim - 1) {
                ProcessBf16Main(blockFactor);
            } else {
                ProcessBf16Main(coreCalcTail);
            }
        }
    }

    __aicore__ inline void ProcessBf16Main(uint32_t loop_len)
    {
        for (uint64_t i = 0; i < loop_len; i++) {
            // Calc mean firstly
            LocalTensor<float> dySum = nFp32Buf.Get<float>();
            Duplicate(dySum, 0.0f, alignLen);
            pipe_barrier(PIPE_V);
            for (uint64_t j = 0; j < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); j++) {
                ComputeDySumBf16(i, j, ubFactor, ubFactor, dySum);
            }
            if (ubCalcTail != 0) {
                ubTailAlign = (ubCalcTail + alignLen - 1) / alignLen * alignLen;
                ComputeDySumBf16(i, ubCalcLoop - 1, ubCalcTail, ubTailAlign, dySum);
            }
            Muls(dySum, dySum, avg, alignLen);
            pipe_barrier(PIPE_V);
            for (uint64_t j = 0; j < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); j++) {
                CopyGammaIn(j, ubFactor);
                LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
                LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
                CopyIn(i, j, ubFactor, ubFactor);
                ComputeMainBf16(i, j, ubFactor, gamma_ub, dgamma, dySum);
                CopyDgammaOut(j, ubFactor);
                CopyOut(i, j, ubFactor, ubFactor);
            }
            if (ubCalcTail != 0) {
                CopyGammaIn(ubCalcLoop - 1, ubCalcTail);
                LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
                LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
                CopyIn(i, ubCalcLoop - 1, ubCalcTail, ubTailAlign);
                ComputeMainBf16(i, ubCalcLoop - 1, ubCalcTail, gamma_ub, dgamma, dySum);
                CopyDgammaOut(ubCalcLoop - 1, ubCalcTail);
                CopyOut(i, ubCalcLoop - 1, ubCalcTail, ubTailAlign);
            }
        }
    }

    __aicore__ inline void ComputeDySumBf16(uint64_t i, uint64_t j,
                                            uint32_t calcLen,
                                            uint32_t calc_len_align,
                                            LocalTensor<float>& dySum)
    {
        CopyGammaIn(j, calcLen);
        CopyIn(i, j, calcLen, calc_len_align);
        LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
        LocalTensor<T1> xUb = inQueX.DeQue<T1>();
        LocalTensor<T1> dyUb = inQueDY.DeQue<T1>();
        LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        float rstdValue = rstdUb.GetValue(0);
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);

        LocalTensor<float> dySumPart = ndBufFp32Buf1.Get<float>();
        LocalTensor<float> tmp32Buf = ndBufFp32Buf2.Get<float>();
        LocalTensor<float> tmp32Buf2 = ndBufFp32Buf3.Get<float>();

        // grad_y = dy * gamma
        Cast(tmp32Buf, dyUb, RoundMode::CAST_NONE, calcLen);
        Cast(tmp32Buf2, gamma_ub, RoundMode::CAST_NONE, calcLen);
        pipe_barrier(PIPE_V);
        inQueGamma.FreeTensor(gamma_ub);
        Mul(tmp32Buf, tmp32Buf, tmp32Buf2, calcLen);
        pipe_barrier(PIPE_V);
        Cast(dyUb, tmp32Buf, RoundMode::CAST_RINT, calcLen);
        pipe_barrier(PIPE_V);
        Cast(tmp32Buf, dyUb, RoundMode::CAST_NONE, calcLen);
        // sum(x * rstd * grad_y)
        Cast(tmp32Buf2, xUb, RoundMode::CAST_NONE, calcLen);
        pipe_barrier(PIPE_V);
        inQueX.FreeTensor(xUb);
        inQueDY.FreeTensor(dyUb);
        inQueRstd.FreeTensor(rstdUb);
        Muls(tmp32Buf2, tmp32Buf2, rstdValue, calcLen);
        pipe_barrier(PIPE_V);
        Mul(dySumPart, tmp32Buf, tmp32Buf2, calcLen);
        pipe_barrier(PIPE_V);
        ReduceSumFP32(0, dySumPart, dySumPart, tmp32Buf2, calcLen, colValAlign);
        Add(dySum, dySum, dySumPart, alignLen);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void ComputeMainBf16(uint64_t nIdx, uint64_t dIdx,
                                           uint32_t calcLen,
                                           LocalTensor<T1>& gamma_ub,
                                           LocalTensor<float>& dgamma,
                                           LocalTensor<float>& dySum)
    {
        LocalTensor<T1> xUb = inQueX.DeQue<T1>();
        LocalTensor<T1> dyUb = inQueDY.DeQue<T1>();
        LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();
        LocalTensor<float> tmp32Buf1 = ndBufFp32Buf1.Get<float>();
        LocalTensor<float> tmp32Buf2 = ndBufFp32Buf2.Get<float>();
        LocalTensor<float> tmp32Buf3 = ndBufFp32Buf3.Get<float>();
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        float rstdValue = rstdUb.GetValue(0);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float dySumVal = dySum.GetValue(0);
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
        // dg = sum((dy * (x * rstd)), dim=0)
        Cast(tmp32Buf2, xUb, RoundMode::CAST_NONE, calcLen);
        Cast(tmp32Buf1, dyUb, RoundMode::CAST_NONE, calcLen);
        pipe_barrier(PIPE_V);
        inQueRstd.FreeTensor(rstdUb);
        Muls(tmp32Buf2, tmp32Buf2, rstdValue, calcLen);
        pipe_barrier(PIPE_V);
        Cast(xUb, tmp32Buf2, RoundMode::CAST_RINT, calcLen);
        pipe_barrier(PIPE_V);
        Cast(tmp32Buf3, xUb, RoundMode::CAST_NONE, calcLen);
        pipe_barrier(PIPE_V);
        Mul(tmp32Buf3, tmp32Buf1, tmp32Buf3, calcLen);
        pipe_barrier(PIPE_V);
        Cast(dyUb, tmp32Buf3, RoundMode::CAST_RINT, calcLen);
        pipe_barrier(PIPE_V);
        Cast(dgamma, dyUb, RoundMode::CAST_NONE, calcLen);
        // grad_y = grad* gamma
        Cast(tmp32Buf3, gamma_ub, RoundMode::CAST_NONE, calcLen);
        Muls(tmp32Buf2, tmp32Buf2, dySumVal, calcLen);
        pipe_barrier(PIPE_V);
        outQueDgamma.EnQue(dgamma);
        inQueGamma.FreeTensor(gamma_ub);
        Mul(tmp32Buf1, tmp32Buf1, tmp32Buf3, calcLen);
        pipe_barrier(PIPE_V);
        Cast(dyUb, tmp32Buf1, RoundMode::CAST_RINT, calcLen);
        pipe_barrier(PIPE_V);
        Cast(tmp32Buf1, dyUb, RoundMode::CAST_NONE, calcLen);
        pipe_barrier(PIPE_V);
        inQueX.FreeTensor(xUb);
        inQueDY.FreeTensor(dyUb);
        // dx = (grad_y - y * mean) * rstd
        Sub(tmp32Buf1, tmp32Buf1, tmp32Buf2, calcLen);
        pipe_barrier(PIPE_V);
        Muls(tmp32Buf1, tmp32Buf1, rstdValue, calcLen);
        pipe_barrier(PIPE_V);
        LocalTensor<T1> dxUb = outQueDX.AllocTensor<T1>();
        Cast(dxUb, tmp32Buf1, RoundMode::CAST_RINT, calcLen);
        pipe_barrier(PIPE_V);
        outQueDX.EnQue(dxUb);
    }

public:
    uint32_t ubTailAlign{0};
    uint32_t rowVal{0};
    uint32_t colVal{0};
    uint32_t colValAlign{0};
    float avg{1.0f};
    uint32_t coreCalcNum{0};
    uint32_t coreCalcTail{0};
    uint32_t blockFactor{0};
    uint32_t blockDim{0};
    uint32_t ubFactor{0};
    uint32_t ubCalcNum{0};
    uint32_t ubCalcTail{0};
    uint64_t ubCalcLoop{0};
    uint32_t ubCalcTailNum{0};
    uint32_t ubCalcTailTail{0};
    uint32_t ubCalcTailLoop{0};
    uint32_t dataType{0};
    uint32_t alignLen{0};
    uint32_t coreOffset{0};
    uint32_t ubFactorAlign{0};
    uint32_t rstdLen{0};
    uint32_t bufferLenSize{0};
    uint32_t coreOffsetStart{0};
    uint32_t coreOffsetLen{0};
    int32_t bufferNum{1};

    TPipe pipe;
    GlobalTensor<int32_t> syncGm_;
    GlobalTensor<T1> dyGm, gammaGm, dxGm, xGm;
    GlobalTensor<float> dgammaGm, rstdGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueDY, inQueX, inQueRstd, inQueGamma;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueDX, outQueDgamma;
    TBuf<TPosition::VECCALC> ndBufFp32Buf1, ndBufFp32Buf2, ndBufFp32Buf3;
    TBuf<TPosition::VECCALC> dFp32Buf;
    TBuf<TPosition::VECCALC> nFp32Buf;
};
#endif // ASCEND_OPS_RMS_NORM_GRAD_SPLIT_D_H_