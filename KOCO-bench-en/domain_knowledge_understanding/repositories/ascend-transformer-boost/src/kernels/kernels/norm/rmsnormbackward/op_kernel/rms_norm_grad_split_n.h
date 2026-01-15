/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_RMS_NORM_GRAD_SPLIT_N_H_
#define ASCEND_OPS_RMS_NORM_GRAD_SPLIT_N_H_
#include "rms_norm_grad_common.h"
template <typename T>
class RmsNormGradSplitN {
public:
    __aicore__ inline RmsNormGradSplitN() {
    }

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd,
                                GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma,
                                GM_ADDR sync,
                                const RmsNormGradTilingData* tiling)
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
        ubCalcTailLoop = tiling->ubCalcTailLoop;
        ubCalcTailTail = tiling->ubCalcTailTail;
        ubCalcTailNum = tiling->ubCalcTailNum;
        ubCalcLoop = tiling->ubCalcLoop;
        ubCalcTail = tiling->ubCalcTail;
        ubCalcNum = tiling->ubCalcNum;
        ubFactor = tiling->ubFactor;
        blockFactor = tiling->blockFactor;
        coreCalcTail = tiling->coreCalcTail;
        coreCalcNum = tiling->coreCalcNum;
        dataType = tiling->dataType;
        avg = tiling->avg;
        colVal = tiling->col;
        rowVal = tiling->row;
        blockDim = tiling->blockDim;
        alignLen = dataType == FLOAT_DTYPE ? ALIGN_32 : ALIGN_16;
        colValAlign = (colVal + alignLen - 1) / alignLen * alignLen;
    }

    __aicore__ inline void InitInputGmBuffer(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd,
                                             GM_ADDR gamma, uint32_t blockDim,
                                             uint32_t coreCalcNum,
                                             uint32_t coreCalcTail)
    {
        if (GetBlockIdx() < blockDim - 1) {
            coreOffset = blockFactor;
        } else {
            coreOffset = coreCalcTail > 0 ? coreCalcTail : blockFactor;
        }
        dyGm.SetGlobalBuffer((__gm__ T*)dy + GetBlockIdx() * blockFactor * colVal, coreOffset * colVal);
        xGm.SetGlobalBuffer((__gm__ T*)x + GetBlockIdx() * blockFactor * colVal, coreOffset * colVal);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor, coreOffset);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, colVal);
    }

    __aicore__ inline void InitOutputGmBuffer(GM_ADDR dx, GM_ADDR dgamma, uint32_t blockDim, uint32_t coreCalcNum,
                                                                                        uint32_t coreCalcTail)
    {
        dxGm.SetGlobalBuffer((__gm__ T*)dx + GetBlockIdx() * blockFactor * colVal, coreOffset * colVal);
        dgammaGm.SetGlobalBuffer((__gm__ float*)dgamma, colVal);
        if (GetBlockIdx() == 0) {
            InitOutput<float>(dgammaGm, colVal, 0);
        }
    }

    __aicore__ inline void InitInputQue()
    {
        ubFactorAlign = ubFactor * colValAlign;
        rstdLen = (ubFactor + alignLen - 1) / alignLen * alignLen;
        bufferLenSize = ubFactorAlign * sizeof(T);
        bufferNum = dataType == BFLOAT16_DTYPE ? BUFFER_NUM_DB : BUFFER_NUM;
        pipe.InitBuffer(inQueDY, bufferNum, bufferLenSize);
        pipe.InitBuffer(inQueX, bufferNum, bufferLenSize);
        pipe.InitBuffer(inQueRstd, bufferNum, rstdLen * sizeof(float));
        pipe.InitBuffer(inQueGamma, BUFFER_NUM, colValAlign * sizeof(T));
    }

    __aicore__ inline void InitOutputQue()
    {
        pipe.InitBuffer(outQueDX, bufferNum, bufferLenSize);
        pipe.InitBuffer(outQueDgamma, BUFFER_NUM, colValAlign * sizeof(float));
    }

    __aicore__ inline void InitTmpBuffer()
    {
        uint32_t ub_factor_len = (ubFactor + alignLen - 1) / alignLen * alignLen;
        uint32_t ubFactorAlignLen = ubFactorAlign * sizeof(float);
        pipe.InitBuffer(ndBufFp32Buf1, ubFactorAlignLen);
        pipe.InitBuffer(nFp32Buf, ub_factor_len * sizeof(float));
        pipe.InitBuffer(dFp32Buf, colValAlign * sizeof(float));
        if (dataType != FLOAT_DTYPE) {
            pipe.InitBuffer(ndBufFp32Buf2, ubFactorAlignLen);
            pipe.InitBuffer(ndBufFp32Buf3, ubFactorAlignLen);
        }
    }

    __aicore__ inline void Process()
    {
        CopyGammaIn();
        LocalTensor<T> gammaUb = inQueGamma.DeQue<T>();
        LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
        Duplicate(dgamma, 0.0f, colValAlign);
        pipe_barrier(PIPE_V);
        if (coreCalcTail == 0) {
            for (uint64_t i = 0; i < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); i++) {
                CopyIn(i, ubCalcNum);
                Compute(i, ubCalcNum, gammaUb, dgamma);
                CopyOut(i, ubCalcNum);
            }
            if (ubCalcTail != 0) {
                CopyIn(ubCalcLoop - 1, ubCalcTail);
                Compute(ubCalcLoop - 1, ubCalcTail, gammaUb, dgamma);
                CopyOut(ubCalcLoop - 1, ubCalcTail);
            }
        } else {
            if (GetBlockIdx() < blockDim - 1) {
                for (uint64_t i = 0; i < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); i++) {
                    CopyIn(i, ubCalcNum);
                    Compute(i, ubCalcNum, gammaUb, dgamma);
                    CopyOut(i, ubCalcNum);
                }
                if (ubCalcTail != 0) {
                    CopyIn(ubCalcLoop - 1, ubCalcTail);
                    Compute(ubCalcLoop - 1, ubCalcTail, gammaUb, dgamma);
                    CopyOut(ubCalcLoop - 1, ubCalcTail);
                }
            } else {
                for (uint64_t i = 0; i < (ubCalcTailTail == 0 ? ubCalcTailLoop : ubCalcTailLoop - 1); i++) {
                    CopyIn(i, ubCalcTailNum);
                    Compute(i, ubCalcTailNum, gammaUb, dgamma);
                    CopyOut(i, ubCalcTailNum);
                }
                if (ubCalcTailTail != 0) {
                    CopyIn(ubCalcTailLoop - 1, ubCalcTailTail);
                    Compute(ubCalcTailLoop - 1, ubCalcTailTail, gammaUb, dgamma);
                    CopyOut(ubCalcTailLoop - 1, ubCalcTailTail);
                }
            }
        }
        inQueGamma.FreeTensor(gammaUb);
        outQueDgamma.EnQue(dgamma);
        CopyDgammaOut();
    }

    __aicore__ inline void CopyGammaIn()
    {
        LocalTensor<T> gammaUb = inQueGamma.AllocTensor<T>();
        DataCopyParams dataCopyParams{(uint16_t)1, (uint16_t)(colVal * sizeof(T)), 0, 0};
        DataCopyPadParams padParams{true, 0, (uint8_t)(colValAlign - colVal), 0};
        DataCopyPad(gammaUb, gammaGm, dataCopyParams, padParams);
        inQueGamma.EnQue(gammaUb);
    }

    __aicore__ inline void CopyIn(uint64_t loopIdx, uint32_t calcLen)
    {
        if (coreCalcTail == 0) {
            DataCopyParams dataCopyParams{(uint16_t)calcLen, (uint16_t)(colVal * sizeof(T)), 0, 0};
            DataCopyPadParams padParams{true, 0, (uint8_t)(colValAlign - colVal), 0};
            LocalTensor<float> rstd = inQueRstd.AllocTensor<float>();
            DataCopy(rstd, rstdGm[loopIdx * ubFactor], (calcLen + alignLen - 1) / alignLen * alignLen);
            inQueRstd.EnQue(rstd);
            LocalTensor<T> x = inQueX.AllocTensor<T>();
            DataCopyPad(x, xGm[loopIdx * ubFactor * colVal], dataCopyParams, padParams);
            inQueX.EnQue(x);
            LocalTensor<T> dy = inQueDY.AllocTensor<T>();
            DataCopyPad(dy, dyGm[loopIdx * ubFactor * colVal], dataCopyParams, padParams);
            inQueDY.EnQue(dy);
        } else {
            if (GetBlockIdx() < blockDim - 1) {
                DataCopyParams dataCopyParams{(uint16_t)calcLen, (uint16_t)(colVal * sizeof(T)), 0, 0};
                DataCopyPadParams padParams{true, 0, (uint8_t)(colValAlign - colVal), 0};
                LocalTensor<float> rstd = inQueRstd.AllocTensor<float>();
                DataCopy(rstd, rstdGm[loopIdx * ubFactor], (calcLen + alignLen - 1) / alignLen * alignLen);
                inQueRstd.EnQue(rstd);
                LocalTensor<T> x = inQueX.AllocTensor<T>();
                DataCopyPad(x, xGm[loopIdx * ubFactor * colVal], dataCopyParams, padParams);
                inQueX.EnQue(x);
                LocalTensor<T> dy = inQueDY.AllocTensor<T>();
                DataCopyPad(dy, dyGm[loopIdx * ubFactor * colVal], dataCopyParams, padParams);
                inQueDY.EnQue(dy);
            } else {
                DataCopyParams dataCopyParams{(uint16_t)calcLen, (uint16_t)(colVal * sizeof(T)), 0, 0};
                DataCopyPadParams padParams{true, 0, (uint8_t)(colValAlign - colVal), 0};
                LocalTensor<float> rstd = inQueRstd.AllocTensor<float>();
                DataCopy(rstd, rstdGm[loopIdx * ubCalcTailNum], (calcLen + alignLen - 1) / alignLen * alignLen);
                inQueRstd.EnQue(rstd);
                LocalTensor<T> x = inQueX.AllocTensor<T>();
                DataCopyPad(x, xGm[loopIdx * ubCalcTailNum * colVal], dataCopyParams, padParams);
                inQueX.EnQue(x);
                LocalTensor<T> dy = inQueDY.AllocTensor<T>();
                DataCopyPad(dy, dyGm[loopIdx * ubCalcTailNum * colVal], dataCopyParams, padParams);
                inQueDY.EnQue(dy);
            }
        }
    }

    __aicore__ inline void CopyOut(uint64_t loopIdx, uint32_t calcLen)
    {
        if (coreCalcTail == 0) {
            DataCopyParams dataCopyParams{(uint16_t)calcLen, (uint16_t)(colVal * sizeof(T)), 0, 0};
            LocalTensor<T> dx = outQueDX.DeQue<T>();
            DataCopyPad(dxGm[loopIdx * ubFactor * colVal], dx, dataCopyParams);
            outQueDX.FreeTensor(dx);
        } else {
            if (GetBlockIdx() < blockDim - 1) {
                DataCopyParams dataCopyParams{(uint16_t)calcLen, (uint16_t)(colVal * sizeof(T)), 0, 0};
                LocalTensor<T> dx = outQueDX.DeQue<T>();
                DataCopyPad(dxGm[loopIdx * ubFactor * colVal], dx, dataCopyParams);
                outQueDX.FreeTensor(dx);
            } else {
                DataCopyParams dataCopyParams{(uint16_t)calcLen, (uint16_t)(colVal * sizeof(T)), 0, 0};
                LocalTensor<T> dx = outQueDX.DeQue<T>();
                DataCopyPad(dxGm[loopIdx * ubCalcTailNum * colVal], dx, dataCopyParams);
                outQueDX.FreeTensor(dx);
            }
        }
    }

    __aicore__ inline void CopyDgammaOut()
    {
        LocalTensor<float> dgamma_out = outQueDgamma.DeQue<float>();
        SetAtomicAdd<float>();
        DataCopyParams dataCopyParams{(uint16_t)1, (uint16_t)(colVal * sizeof(float)), 0, 0};
        DataCopyPad(dgammaGm, dgamma_out, dataCopyParams);
        SetAtomicNone();
        outQueDgamma.FreeTensor(dgamma_out);
    }

    __aicore__ inline void Compute(uint64_t loopIdx, uint32_t calcLen,
                                   LocalTensor<float>& gammaUb,
                                   LocalTensor<float>& dgamma)
    {
        LocalTensor<T> xUb = inQueX.DeQue<T>();
        LocalTensor<T> dyUb = inQueDY.DeQue<T>();
        LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();

        LocalTensor<T> dxUb = outQueDX.AllocTensor<T>();
        LocalTensor<float> tmp_buf = ndBufFp32Buf1.Get<float>(calcLen * colValAlign);
        ComputeMain(xUb, tmp_buf, dxUb, dyUb, rstdUb, gammaUb, dgamma, calcLen);
        outQueDX.EnQue(dxUb);
    }

    __aicore__ inline void ComputeMain(LocalTensor<float>& x1,
                                       LocalTensor<float>& tmp_buf,
                                       LocalTensor<float>& dx,
                                       LocalTensor<float>& dy,
                                       LocalTensor<float>& rstd,
                                       LocalTensor<float>& gamma,
                                       LocalTensor<float>& dgamma,
                                       uint32_t calcLen)
    {
        pipe_barrier(PIPE_ALL);
        for (uint32_t i = 0; i < calcLen; i++) {
            float rstdValue = rstd.GetValue(i);
            set_flag(PIPE_S, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
            // grad_y = dy * g
            Mul(tmp_buf[i * colValAlign], dy[i * colValAlign], gamma, colValAlign);
            // y = x * rstd
            Muls(x1[i * colValAlign], x1[i * colValAlign], rstdValue, colValAlign);
            pipe_barrier(PIPE_V);
            if (i == calcLen - 1) {
                inQueRstd.FreeTensor(rstd);
            }
            // dg = sum(dy * (x * rstd), dim=0)
            Mul(dy[i * colValAlign], dy[i * colValAlign], x1[i * colValAlign], colValAlign);
            pipe_barrier(PIPE_V);
            Add(dgamma, dgamma, dy[i * colValAlign], colValAlign);
            pipe_barrier(PIPE_V);
            // mean = sum(grad_y * y) * avg
            Duplicate(dy[i * colValAlign], 0.0f, colValAlign);
            pipe_barrier(PIPE_V);
            Mul(dy[i * colValAlign], tmp_buf[i * colValAlign], x1[i * colValAlign], colVal);
            pipe_barrier(PIPE_V);
            float reduce_val = ReduceSumFP32_V2(dy[i * colValAlign], colValAlign);
            float dySumVal = reduce_val * avg;
            Muls(x1[i * colValAlign], x1[i * colValAlign], dySumVal, colValAlign);
            pipe_barrier(PIPE_V);
            Sub(tmp_buf[i * colValAlign], tmp_buf[i * colValAlign], x1[i * colValAlign], colValAlign);
            pipe_barrier(PIPE_V);
            if (i == calcLen - 1) {
                inQueX.FreeTensor(x1);
                inQueDY.FreeTensor(dy);
            }
            Muls(dx[i * colValAlign], tmp_buf[i * colValAlign], rstdValue, colValAlign);
            pipe_barrier(PIPE_V);
        }
    }

    __aicore__ inline void ProcessFp16()
    {
        CopyGammaIn();
        LocalTensor<T> gammaUb = inQueGamma.DeQue<T>();
        LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
        Duplicate(dgamma, 0.0f, colValAlign);
        pipe_barrier(PIPE_V);
        if (coreCalcTail == 0) {
            for (uint32_t i = 0; i < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); i++) {
                CopyIn(i, ubCalcNum);
                ComputeFp16(i, ubCalcNum, gammaUb, dgamma);
                CopyOut(i, ubCalcNum);
            }
            if (ubCalcTail != 0) {
                CopyIn(ubCalcLoop - 1, ubCalcTail);
                ComputeFp16(ubCalcLoop - 1, ubCalcTail, gammaUb, dgamma);
                CopyOut(ubCalcLoop - 1, ubCalcTail);
            }
        } else {
            if (GetBlockIdx() < blockDim - 1) {
                for (uint32_t i = 0; i < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); i++) {
                    CopyIn(i, ubCalcNum);
                    ComputeFp16(i, ubCalcNum, gammaUb, dgamma);
                    CopyOut(i, ubCalcNum);
                }
                if (ubCalcTail != 0) {
                    CopyIn(ubCalcLoop - 1, ubCalcTail);
                    ComputeFp16(ubCalcLoop - 1, ubCalcTail, gammaUb, dgamma);
                    CopyOut(ubCalcLoop - 1, ubCalcTail);
                }
            } else {
                for (uint32_t i = 0; i < (ubCalcTailTail == 0 ? ubCalcTailLoop : ubCalcTailLoop - 1); i++) {
                    CopyIn(i, ubCalcTailNum);
                    ComputeFp16(i, ubCalcTailNum, gammaUb, dgamma);
                    CopyOut(i, ubCalcTailNum);
                }
                if (ubCalcTailTail != 0) {
                    CopyIn(ubCalcTailLoop - 1, ubCalcTailTail);
                    ComputeFp16(ubCalcTailLoop - 1, ubCalcTailTail, gammaUb, dgamma);
                    CopyOut(ubCalcTailLoop - 1, ubCalcTailTail);
                }
            }
        }
        inQueGamma.FreeTensor(gammaUb);
        outQueDgamma.EnQue(dgamma);
        CopyDgammaOut();
    }

    __aicore__ inline void ComputeFp16(uint64_t loopIdx,
                                       uint32_t calcLen,
                                       LocalTensor<T>& gammaUb,
                                       LocalTensor<float>& dgamma)
    {
        LocalTensor<T> xUb = inQueX.DeQue<T>();
        LocalTensor<T> dyUb = inQueDY.DeQue<T>();
        LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();
        LocalTensor<T> dxUb = outQueDX.AllocTensor<T>();
        if (colValAlign < SMALLD_THRESHOLD) {
            ComputeMainFp16SmallD(xUb, dxUb, dyUb, rstdUb, gammaUb, dgamma, calcLen);
        } else {
            ComputeMainFp16(xUb, dxUb, dyUb, rstdUb, gammaUb, dgamma, calcLen);
        }
        outQueDX.EnQue(dxUb);
    }

    __aicore__ inline void ComputeMainFp16SmallD(LocalTensor<T>& x,
                                                 LocalTensor<T>& dx,
                                                 LocalTensor<T>& dy,
                                                 LocalTensor<float>& rstd,
                                                 LocalTensor<T>& gamma,
                                                 LocalTensor<float>& dgamma,
                                                 uint32_t calcLen)
    {
        pipe_barrier(PIPE_ALL);
        LocalTensor<float> dySum = ndBufFp32Buf1.Get<float>();
        LocalTensor<float> tmp32buf = ndBufFp32Buf3.Get<float>();
        LocalTensor<float> tmp32Buf2 = ndBufFp32Buf2.Get<float>();
        LocalTensor<T> dTmpBufFp16 = dFp32Buf.Get<T>();
        Cast(tmp32Buf2, x, RoundMode::CAST_NONE, colValAlign * calcLen);
        Duplicate(dySum, 0.0f, colValAlign * calcLen);
        pipe_barrier(PIPE_V);
        for (uint32_t i = 0; i < calcLen; i++) {
            float rstdValue = rstd.GetValue(i);
            set_flag(PIPE_S, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
            // y = x * rstd
            Muls(tmp32Buf2[i * colValAlign], tmp32Buf2[i * colValAlign], rstdValue, colValAlign);
            Mul(dTmpBufFp16, dy[i * colValAlign], gamma, colValAlign);
            pipe_barrier(PIPE_V);
            if (i == calcLen - 1) {
                inQueRstd.FreeTensor(rstd);
            }
            Cast(tmp32buf[i * colValAlign], dTmpBufFp16, RoundMode::CAST_NONE, colValAlign);
            pipe_barrier(PIPE_V);
            // mean = sum (grad_y * y) * avg
            Mul(dySum[i * colValAlign], tmp32buf[i * colValAlign], tmp32Buf2[i * colValAlign], colVal);
            pipe_barrier(PIPE_V);
            float reduce_val = ReduceSumFP32_V2(dySum[i * colValAlign], colValAlign);
            float dySumVal = reduce_val * avg;
            // dx = (grad_y - y * mean) * rstd
            Muls(dySum[i * colValAlign], tmp32Buf2[i * colValAlign], dySumVal, colValAlign);
            pipe_barrier(PIPE_V);
            Sub(dySum[i * colValAlign], tmp32buf[i * colValAlign], dySum[i * colValAlign], colValAlign);
            pipe_barrier(PIPE_V);
            Muls(dySum[i * colValAlign], dySum[i * colValAlign], rstdValue, colValAlign);
            pipe_barrier(PIPE_V);
        }
        Cast(dx, dySum, RoundMode::CAST_NONE, colValAlign * calcLen);
        Cast(x, tmp32Buf2, RoundMode::CAST_NONE, colValAlign * calcLen);
        pipe_barrier(PIPE_V);
        Mul(dy, x, dy, colValAlign * calcLen);
        pipe_barrier(PIPE_V);
        inQueX.FreeTensor(x);
        Cast(tmp32buf, dy, RoundMode::CAST_NONE, colValAlign * calcLen);
        pipe_barrier(PIPE_V);
        inQueDY.FreeTensor(dy);
        for (uint32_t i = 0; i < calcLen; i++) {
            Add(dgamma, tmp32buf[i * colValAlign], dgamma, colValAlign);
            pipe_barrier(PIPE_V);
        }
    }

    __aicore__ inline void ComputeMainFp16(LocalTensor<T>& x, LocalTensor<T>& dx, LocalTensor<T>& dy,
                                           LocalTensor<float>& rstd, LocalTensor<T>& gamma, LocalTensor<float>& dgamma,
                                           uint32_t calcLen)
    {
        pipe_barrier(PIPE_ALL);
        LocalTensor<float> dySum = ndBufFp32Buf1.Get<float>();
        LocalTensor<float> tmp32buf = ndBufFp32Buf3.Get<float>();
        LocalTensor<float> tmp32Buf2 = ndBufFp32Buf2.Get<float>();
        LocalTensor<T> dTmpBufFp16 = dFp32Buf.Get<T>();
        for (uint32_t i = 0; i < calcLen; i++) {
            float rstdValue = rstd.GetValue(i);
            set_flag(PIPE_S, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
            // y = x * rstd
            Cast(tmp32Buf2[i * colValAlign], x[i * colValAlign], RoundMode::CAST_NONE, colValAlign);
            pipe_barrier(PIPE_V);
            if (i == calcLen - 1) {
                inQueRstd.FreeTensor(rstd);
            }
            Muls(tmp32Buf2[i * colValAlign], tmp32Buf2[i * colValAlign], rstdValue, colValAlign);
            Mul(dTmpBufFp16, dy[i * colValAlign], gamma, colValAlign);
            pipe_barrier(PIPE_V);
            Cast(tmp32buf[i * colValAlign], dTmpBufFp16, RoundMode::CAST_NONE, colValAlign);
            pipe_barrier(PIPE_V);
            // mean = sum (grad_y * y) * avg
            Duplicate(dySum[i * colValAlign], 0.0f, colValAlign);
            pipe_barrier(PIPE_V);
            Mul(dySum[i * colValAlign], tmp32buf[i * colValAlign], tmp32Buf2[i * colValAlign], colVal);
            pipe_barrier(PIPE_V);
            float reduce_val = ReduceSumFP32_V2(dySum[i * colValAlign], colValAlign);
            float dySumVal = reduce_val * avg;
            // dx = (grad_y - y * mean) * rstd
            Muls(dySum[i * colValAlign], tmp32Buf2[i * colValAlign], dySumVal, colValAlign);
            pipe_barrier(PIPE_V);
            Sub(dySum[i * colValAlign], tmp32buf[i * colValAlign], dySum[i * colValAlign], colValAlign);
            Cast(x[i * colValAlign], tmp32Buf2[i * colValAlign], RoundMode::CAST_NONE, colValAlign);
            pipe_barrier(PIPE_V);
            Muls(dySum[i * colValAlign], dySum[i * colValAlign], rstdValue, colValAlign);
            Mul(dy[i * colValAlign], x[i * colValAlign], dy[i * colValAlign], colValAlign);
            pipe_barrier(PIPE_V);
            if (i == calcLen - 1) {
                inQueX.FreeTensor(x);
            }
            Cast(dx[i * colValAlign], dySum[i * colValAlign], RoundMode::CAST_NONE, colValAlign);
            Cast(tmp32buf[i * colValAlign], dy[i * colValAlign], RoundMode::CAST_NONE, colValAlign);
            pipe_barrier(PIPE_V);
            if (i == calcLen - 1) {
                inQueDY.FreeTensor(dy);
            }
            Add(dgamma, tmp32buf[i * colValAlign], dgamma, colValAlign);
            pipe_barrier(PIPE_V);
        }
    }

    __aicore__ inline void ProcessBf16()
    {
        CopyGammaIn();
        LocalTensor<T> gammaUb = inQueGamma.DeQue<T>();
        LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
        Duplicate(dgamma, 0.0f, colValAlign);
        pipe_barrier(PIPE_V);
        if (coreCalcTail == 0) {
            for (uint64_t i = 0; i < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); i++) {
                CopyIn(i, ubCalcNum);
                ComputeBf16(i, ubCalcNum, gammaUb, dgamma);
                CopyOut(i, ubCalcNum);
            }
            if (ubCalcTail != 0) {
                CopyIn(ubCalcLoop - 1, ubCalcTail);
                ComputeBf16(ubCalcLoop - 1, ubCalcTail, gammaUb, dgamma);
                CopyOut(ubCalcLoop - 1, ubCalcTail);
            }
        } else {
            if (GetBlockIdx() < blockDim - 1) {
                for (uint64_t i = 0; i < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); i++) {
                    CopyIn(i, ubCalcNum);
                    ComputeBf16(i, ubCalcNum, gammaUb, dgamma);
                    CopyOut(i, ubCalcNum);
                }
                if (ubCalcTail != 0) {
                    CopyIn(ubCalcLoop - 1, ubCalcTail);
                    ComputeBf16(ubCalcLoop - 1, ubCalcTail, gammaUb, dgamma);
                    CopyOut(ubCalcLoop - 1, ubCalcTail);
                }
            } else {
                for (uint32_t i = 0; i < (ubCalcTailTail == 0 ? ubCalcTailLoop : ubCalcTailLoop - 1); i++) {
                    CopyIn(i, ubCalcTailNum);
                    ComputeBf16(i, ubCalcTailNum, gammaUb, dgamma);
                    CopyOut(i, ubCalcTailNum);
                }
                if (ubCalcTailTail != 0) {
                    CopyIn(ubCalcTailLoop - 1, ubCalcTailTail);
                    ComputeBf16(ubCalcTailLoop - 1, ubCalcTailTail, gammaUb, dgamma);
                    CopyOut(ubCalcTailLoop - 1, ubCalcTailTail);
                }
            }
        }
        inQueGamma.FreeTensor(gammaUb);
        outQueDgamma.EnQue(dgamma);
        CopyDgammaOut();
    }

    __aicore__ inline void ComputeBf16(uint64_t loopIdx,
                                       uint32_t calcLen,
                                       LocalTensor<T>& gammaUb,
                                       LocalTensor<float>& dgamma)
    {
        LocalTensor<T> xUb = inQueX.DeQue<T>();
        LocalTensor<T> dyUb = inQueDY.DeQue<T>();
        LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();
        LocalTensor<T> dxUb = outQueDX.AllocTensor<T>();
        if (colValAlign < SMALLD_THRESHOLD) {
            ComputeMainBf16SmallD(xUb, dxUb, dyUb, rstdUb, gammaUb, dgamma, calcLen);
        } else {
            ComputeMainBf16(xUb, dxUb, dyUb, rstdUb, gammaUb, dgamma, calcLen);
        }
        outQueDX.EnQue(dxUb);
    }

    __aicore__ inline void ComputeMainBf16SmallD(LocalTensor<T>& x,
                                                 LocalTensor<T>& dx,
                                                 LocalTensor<T>& dy,
                                                 LocalTensor<float>& rstd,
                                                 LocalTensor<T>& gamma,
                                                 LocalTensor<float>& dgamma,
                                                 uint32_t calcLen)
    {
        LocalTensor<float> dySum = ndBufFp32Buf1.Get<float>();
        LocalTensor<float> tmp32buf = ndBufFp32Buf3.Get<float>();
        LocalTensor<float> tmp32Buf2 = ndBufFp32Buf2.Get<float>();
        LocalTensor<float> gammaFp32 = dFp32Buf.Get<float>();
        Cast(tmp32Buf2, x, RoundMode::CAST_NONE, colValAlign * calcLen);
        Cast(tmp32buf, dy, RoundMode::CAST_NONE, colValAlign * calcLen);
        Duplicate(dySum, 0.0f, colValAlign * calcLen);
        pipe_barrier(PIPE_V);
        for (uint32_t i = 0; i < calcLen; i++) {
            Cast(gammaFp32, gamma, RoundMode::CAST_NONE, colValAlign);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            float rstdValue = rstd.GetValue(i);
            set_flag(PIPE_S, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
            // y = x * rstd
            Muls(tmp32Buf2[i * colValAlign], tmp32Buf2[i * colValAlign], rstdValue, colValAlign);
            pipe_barrier(PIPE_V);
            Mul(tmp32buf[i * colValAlign], tmp32buf[i * colValAlign], gammaFp32, colValAlign);
            pipe_barrier(PIPE_V);
            // mean = sum (grad_y * y) * avg
            Cast(x[i * colValAlign], tmp32buf[i * colValAlign], RoundMode::CAST_RINT, colValAlign);
            pipe_barrier(PIPE_V);
            Cast(tmp32buf[i * colValAlign], x[i * colValAlign], RoundMode::CAST_NONE, colValAlign);
            pipe_barrier(PIPE_V);
            if (i == calcLen - 1) {
                inQueRstd.FreeTensor(rstd);
                inQueX.FreeTensor(x);
            }
            Mul(dySum[i * colValAlign], tmp32buf[i * colValAlign], tmp32Buf2[i * colValAlign], colVal);
            pipe_barrier(PIPE_V);
            float reduce_val = ReduceSumFP32_V2(dySum[i * colValAlign], colValAlign);
            float dySumVal = reduce_val * avg;
            // dx = (grad_y - y * mean) * rstd
            Muls(dySum[i * colValAlign], tmp32Buf2[i * colValAlign], dySumVal, colValAlign);
            pipe_barrier(PIPE_V);
            Sub(dySum[i * colValAlign], tmp32buf[i * colValAlign], dySum[i * colValAlign], colValAlign);
            pipe_barrier(PIPE_V);
            Muls(dySum[i * colValAlign], dySum[i * colValAlign], rstdValue, colValAlign);
            pipe_barrier(PIPE_V);
        }
        Cast(dx, dySum, RoundMode::CAST_RINT, colValAlign * calcLen);
        Cast(tmp32buf, dy, RoundMode::CAST_NONE, colValAlign * calcLen);
        pipe_barrier(PIPE_V);
        // dg=sum(dy * (x * rstd), dim=0)
        Cast(dy, tmp32Buf2, RoundMode::CAST_RINT, colValAlign * calcLen);
        pipe_barrier(PIPE_V);
        Cast(dySum, dy, RoundMode::CAST_NONE, colValAlign * calcLen);
        pipe_barrier(PIPE_V);
        Mul(dySum, tmp32buf, dySum, colValAlign * calcLen);
        pipe_barrier(PIPE_V);
        Cast(dy, dySum, RoundMode::CAST_RINT, colValAlign * calcLen);
        pipe_barrier(PIPE_V);
        Cast(dySum, dy, RoundMode::CAST_NONE, colValAlign * calcLen);
        pipe_barrier(PIPE_V);
        inQueDY.FreeTensor(dy);
        for (uint32_t i = 0; i < calcLen; i++) {
            Add(dgamma, dySum[i * colValAlign], dgamma, colValAlign);
            pipe_barrier(PIPE_V);
        }
    }

    __aicore__ inline void ComputeMainBf16(LocalTensor<T>& x, LocalTensor<T>& dx,
                                           LocalTensor<T>& dy, LocalTensor<float>& rstd,
                                           LocalTensor<T>& gamma, LocalTensor<float>& dgamma, uint32_t calcLen)
    {
        LocalTensor<float> gammaFp32 = dFp32Buf.Get<float>();
        LocalTensor<float> tmp32Buf2 = ndBufFp32Buf2.Get<float>();
        LocalTensor<float> tmp32buf = ndBufFp32Buf3.Get<float>();
        LocalTensor<float> dySum = ndBufFp32Buf1.Get<float>();
        for (uint32_t i = 0; i < calcLen; i++) {
            Cast(gammaFp32, gamma, RoundMode::CAST_NONE, colValAlign);
            Cast(tmp32Buf2[i * colValAlign], x[i * colValAlign], RoundMode::CAST_NONE, colValAlign);
            Cast(tmp32buf[i * colValAlign], dy[i * colValAlign], RoundMode::CAST_NONE, colValAlign);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            float rstdValue = rstd.GetValue(i);
            set_flag(PIPE_S, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
            // y = x * rstd
            Muls(tmp32Buf2[i * colValAlign], tmp32Buf2[i * colValAlign], rstdValue, colValAlign);
            pipe_barrier(PIPE_V);
            Mul(tmp32buf[i * colValAlign], tmp32buf[i * colValAlign], gammaFp32, colValAlign);
            pipe_barrier(PIPE_V);
            // mean = sum (grad_y * y) * avg
            Duplicate(dySum[i * colValAlign], 0.0f, colValAlign);
            Cast(x[i * colValAlign], tmp32buf[i * colValAlign], RoundMode::CAST_RINT, colValAlign);
            pipe_barrier(PIPE_V);
            Cast(tmp32buf[i * colValAlign], x[i * colValAlign], RoundMode::CAST_NONE, colValAlign);
            pipe_barrier(PIPE_V);
            if (i == calcLen - 1) {
                inQueRstd.FreeTensor(rstd);
                inQueX.FreeTensor(x);
            }
            Mul(dySum[i * colValAlign], tmp32buf[i * colValAlign], tmp32Buf2[i * colValAlign], colVal);
            pipe_barrier(PIPE_V);
            float reduce_val = ReduceSumFP32_V2(dySum[i * colValAlign], colValAlign);
            float dySumVal = reduce_val * avg;
            // dx = (grad_y - y * mean) * rstd
            Muls(dySum[i * colValAlign], tmp32Buf2[i * colValAlign], dySumVal, colValAlign);
            pipe_barrier(PIPE_V);
            Sub(dySum[i * colValAlign], tmp32buf[i * colValAlign], dySum[i * colValAlign], colValAlign);
            pipe_barrier(PIPE_V);
            Muls(dySum[i * colValAlign], dySum[i * colValAlign], rstdValue, colValAlign);
            pipe_barrier(PIPE_V);
            Cast(dx[i * colValAlign], dySum[i * colValAlign], RoundMode::CAST_RINT, colValAlign);
            Cast(tmp32buf[i * colValAlign], dy[i * colValAlign], RoundMode::CAST_NONE, colValAlign);
            pipe_barrier(PIPE_V);
            // dg=sum(dy * (x * rstd), dim=0)
            Cast(dy[i * colValAlign], tmp32Buf2[i * colValAlign], RoundMode::CAST_RINT, colValAlign);
            pipe_barrier(PIPE_V);
            Cast(dySum[i * colValAlign], dy[i * colValAlign], RoundMode::CAST_NONE, colValAlign);
            pipe_barrier(PIPE_V);
            Mul(dySum[i * colValAlign], tmp32buf[i * colValAlign], dySum[i * colValAlign], colValAlign);
            pipe_barrier(PIPE_V);
            Cast(dy[i * colValAlign], dySum[i * colValAlign], RoundMode::CAST_RINT, colValAlign);
            pipe_barrier(PIPE_V);
            Cast(dySum[i * colValAlign], dy[i * colValAlign], RoundMode::CAST_NONE, colValAlign);
            pipe_barrier(PIPE_V);
            if (i == calcLen - 1) {
                inQueDY.FreeTensor(dy);
            }
            Add(dgamma, dySum[i * colValAlign], dgamma, colValAlign);
            pipe_barrier(PIPE_V);
        }
    }

public:
    uint32_t ubTailAlign;
    uint32_t rowVal;
    uint32_t colVal;
    uint32_t colValAlign;
    float avg{1.0f};
    uint32_t coreCalcNum;
    uint32_t coreCalcTail;
    uint32_t blockFactor;
    uint32_t blockDim;
    uint32_t ubFactor;
    uint32_t ubCalcNum;
    uint32_t ubCalcTail;
    uint32_t ubCalcLoop;
    uint32_t ubCalcTailNum;
    uint32_t ubCalcTailTail;
    uint32_t ubCalcTailLoop;
    uint32_t dataType;
    uint32_t alignLen;
    uint32_t coreOffset;
    uint32_t ubFactorAlign;
    uint32_t rstdLen;
    uint32_t bufferLenSize;
    int32_t bufferNum;

    TPipe pipe;
    GlobalTensor<int32_t> syncGm_;
    GlobalTensor<T> dyGm, gammaGm, dxGm, xGm;
    GlobalTensor<float> dgammaGm, rstdGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueDY, inQueX, inQueRstd, inQueGamma;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueDX, outQueDgamma;
    TBuf<TPosition::VECCALC> ndBufFp32Buf1, ndBufFp32Buf2, ndBufFp32Buf3;
    TBuf<TPosition::VECCALC> dFp32Buf;
    TBuf<TPosition::VECCALC> nFp32Buf;
};
#endif // ASCEND_OPS_RMS_NORM_GRAD_SPLIT_N_H_