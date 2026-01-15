/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPP_SWI_GLU_BACKWARD_H
#define OPP_SWI_GLU_BACKWARD_H
#include "kernel_operator.h"
#include "kernels/activation/swiglu_backward/tiling/tiling_data.h"

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t SPLIT_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 32;

template <typename T>
__aicore__ inline T CeilDiv(T x, T y) {
    return y == 0 ? 0 : (x + y - 1) / y;
}

template<typename inType, typename outType>
class SwigluBackward {
public:
    __aicore__ inline SwigluBackward() {}
    __aicore__ inline ~SwigluBackward() {}

    __aicore__ inline void Init(GM_ADDR y_grad, GM_ADDR x, GM_ADDR x_grad,
        AsdOps::SwiGluBackwardTilingData& tiling_data)
    {
        this->rowLen = tiling_data.rowLen;
        this->colLen = tiling_data.colLen;
        this->rowLenPerCore = tiling_data.rowLenPerCore;
        this->colLenPerCore = tiling_data.colLenPerCore;
        this->basicRowLen = tiling_data.basicRowLen;
        this->basicColLen = tiling_data.basicColLen;
        this->coreNumPerLine = tiling_data.coreNumPerLine;
        xGm.SetGlobalBuffer((__gm__ inType*)x, SPLIT_NUM * rowLen * colLen);
        yGradGm.SetGlobalBuffer((__gm__ inType*)y_grad, rowLen * colLen);
        xGradGm.SetGlobalBuffer((__gm__ outType*)x_grad, SPLIT_NUM * rowLen * colLen);
        uint64_t tileLength = basicRowLen * (basicColLen == 0 ? (BLOCK_SIZE / sizeof(inType)) : basicColLen);
        pipe.InitBuffer(inQueueA, BUFFER_NUM, tileLength * sizeof(inType));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, tileLength * sizeof(inType));
        pipe.InitBuffer(inQueueYGrad, BUFFER_NUM, tileLength * sizeof(inType));
        pipe.InitBuffer(outQueueAGrad, BUFFER_NUM, tileLength * sizeof(outType));
        pipe.InitBuffer(outQueueBGrad, BUFFER_NUM, tileLength * sizeof(outType));
        pipe.InitBuffer(tmpA, tileLength * sizeof(float));
        pipe.InitBuffer(tmpB, tileLength * sizeof(float));
        pipe.InitBuffer(tmpYGrad, tileLength * sizeof(float));
        pipe.InitBuffer(tmpAGrad, tileLength * sizeof(float));
        pipe.InitBuffer(tmpBGrad, tileLength * sizeof(float));
        pipe.InitBuffer(tmpSigmoid, tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (basicColLen == 0) { // 边界情况，n在32B以内
            ProcessGradSmallN();
        } else if (colLenPerCore == colLen) { // 一核处理多行的情况
            if (basicRowLen == 1) { // ub空间可容纳1行内数据
                ProcessGradCoreMultiUbSingle();
            } else { // ub空间可容纳多行
                ProcessGradCoreMultiUbMulti();
            }
        } else { // 一行多核处理的情况
            ProcessGradCoreSingle();
        }
    }

    __aicore__ inline void ProcessGradSmallN() // 边界情况，n在32B以内
    {
        uint64_t coreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint32_t rowLenPerCoreCal = coreIdx == AscendC::GetBlockNum() - 1 ?
            rowLen - (AscendC::GetBlockNum() - 1) * rowLenPerCore : rowLenPerCore; // 最后一核单独处理

        uint32_t rowLoop = CeilDiv(rowLenPerCoreCal, static_cast<uint32_t>(basicRowLen));
        uint16_t basicRowLenCal = basicRowLen;
        for (uint32_t rIdx = 0; rIdx < rowLoop; rIdx++) {
            basicRowLenCal = static_cast<uint16_t>((rIdx == rowLoop - 1) ?
                (rowLenPerCoreCal - (rowLoop - 1) * basicRowLen) : basicRowLen); // 每核处理的最后一个行循环单独处理

            AscendC::DataCopyExtParams splitCopyinParams;
            AscendC::DataCopyExtParams yGradCopyinParams;
            AscendC::DataCopyExtParams splitCopyoutParams;
            uint64_t splitCopyinOffset;
            uint64_t yGradCopyinOffset;
            uint64_t splitCopyoutOffset;
            AscendC::DataCopyPadExtParams<inType> padParams;
            splitCopyinParams = {basicRowLenCal, colLen * sizeof(inType), colLen * sizeof(inType), 0, 0};
            splitCopyinOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen + rIdx * basicRowLen * SPLIT_NUM * colLen;
            yGradCopyinParams = {basicRowLenCal, colLen * sizeof(inType), 0, 0, 0};
            yGradCopyinOffset = coreIdx * rowLenPerCore * colLen + rIdx * basicRowLen * colLen;
            padParams = {false, 0, 0, 0};
            splitCopyoutParams = {basicRowLenCal, colLen * sizeof(outType), 0, colLen * sizeof(outType), 0};
            splitCopyoutOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen + rIdx * basicRowLen * SPLIT_NUM * colLen;
            CopyInPad(splitCopyinOffset, splitCopyinParams, yGradCopyinOffset, yGradCopyinParams, padParams);
            Compute(basicRowLen * BLOCK_SIZE / sizeof(inType));
            CopyOutPad(splitCopyoutOffset, splitCopyoutParams);
        }
    }

    __aicore__ inline void ProcessGradCoreMultiUbSingle() // 一核处理多行, ub空间可容纳1行内数据
    {
        uint64_t coreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint32_t rowLenPerCoreCal = coreIdx == AscendC::GetBlockNum() - 1 ?
            rowLen - (AscendC::GetBlockNum() - 1) * rowLenPerCore : rowLenPerCore; // 最后一核单独处理

        uint32_t colLoop = CeilDiv(colLenPerCore, static_cast<uint32_t>(basicColLen));
        uint16_t basicColLenCal = basicColLen;
        for (uint32_t rIdx = 0; rIdx < rowLenPerCoreCal; rIdx++) {
            for (uint32_t cidx = 0; cidx < colLoop; cidx++) {
                basicColLenCal = static_cast<uint16_t>(cidx == colLoop - 1 ?
                    colLenPerCore - (colLoop - 1) * basicColLen : basicColLen);

                AscendC::DataCopyParams splitCopyinParams;
                AscendC::DataCopyParams yGradCopyinParams;
                AscendC::DataCopyParams splitCopyoutParams;
                uint64_t splitCopyinOffset;
                uint64_t yGradCopyinOffset;
                uint64_t splitCopyoutOffset;
                if (cidx == colLoop - 1) { // 无论colLen是否32B对齐，每行最后一块都回退到(列basicColLen)再处理
                    splitCopyinParams = {basicRowLen, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
                    splitCopyinOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen +
                        rIdx * basicRowLen * SPLIT_NUM * colLen + colLenPerCore - basicColLen;
                    yGradCopyinParams = {basicRowLen, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
                    yGradCopyinOffset = coreIdx * rowLenPerCore * colLen + rIdx * basicRowLen * colLen +
                        colLenPerCore - basicColLen;
                    splitCopyoutParams = {basicRowLen, basicColLen * sizeof(outType) / BLOCK_SIZE, 0, 0};
                    splitCopyoutOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen + rIdx * basicRowLen * SPLIT_NUM *
                        colLen + colLenPerCore - basicColLen;
                } else {
                    splitCopyinParams = {basicRowLen, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
                    splitCopyinOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen +
                        rIdx * basicRowLen * SPLIT_NUM * colLen + cidx * basicColLen;
                    yGradCopyinParams = {basicRowLen, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
                    yGradCopyinOffset = coreIdx * rowLenPerCore * colLen + rIdx * basicRowLen * colLen +
                        cidx * basicColLen;
                    splitCopyoutParams = {basicRowLen, basicColLen * sizeof(outType) / BLOCK_SIZE, 0, 0};
                    splitCopyoutOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen + rIdx * basicRowLen * SPLIT_NUM *
                        colLen + cidx * basicColLen;
                }
                CopyIn(splitCopyinOffset, splitCopyinParams, yGradCopyinOffset, yGradCopyinParams);
                Compute(basicRowLen * basicColLen);
                CopyOut(splitCopyoutOffset, splitCopyoutParams);
            }
        }
    }

    __aicore__ inline void ProcessGradCoreSingle() // 一行多核处理的情况
    {
        uint32_t core_num_per_line = coreNumPerLine;
        uint64_t coreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint64_t core_ridx = coreIdx / core_num_per_line;
        uint64_t core_cidx = coreIdx % core_num_per_line;
        uint32_t colLenPerCoreCal = core_cidx == core_num_per_line - 1 ?
            colLen - (core_num_per_line - 1) * colLenPerCore: colLenPerCore;
        uint32_t colLoop = CeilDiv(colLenPerCoreCal, static_cast<uint32_t>(basicColLen));
        AscendC::DataCopyParams splitCopyinParams;
        AscendC::DataCopyParams yGradCopyinParams;
        AscendC::DataCopyParams splitCopyoutParams;
        uint64_t splitCopyinOffset;
        uint64_t yGradCopyinOffset;
        uint64_t splitCopyoutOffset;
        for (uint32_t cidx = 0; cidx < colLoop; cidx++) {
            if (cidx == colLoop - 1) { // 无论colLen是否32B对齐，每行最后一块都回退到(列basicColLen)再处理
                splitCopyinParams = {basicRowLen, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
                splitCopyinOffset = core_ridx * basicRowLen * SPLIT_NUM * colLen + core_cidx * colLenPerCore +
                    colLenPerCoreCal - basicColLen;
                yGradCopyinParams = {basicRowLen, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
                yGradCopyinOffset = core_ridx * basicRowLen * colLen + core_cidx * colLenPerCore +
                    colLenPerCoreCal - basicColLen;
                splitCopyoutParams = {basicRowLen, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
                splitCopyoutOffset = core_ridx * basicRowLen * SPLIT_NUM * colLen + core_cidx * colLenPerCore +
                    colLenPerCoreCal - basicColLen;
            } else {
                splitCopyinParams = {basicRowLen, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
                splitCopyinOffset = core_ridx * basicRowLen * SPLIT_NUM * colLen + core_cidx * colLenPerCore +
                    cidx * basicColLen;
                yGradCopyinParams = {basicRowLen, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
                yGradCopyinOffset = core_ridx * basicRowLen * colLen + core_cidx * colLenPerCore +
                    cidx * basicColLen;
                splitCopyoutParams = {basicRowLen, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
                splitCopyoutOffset = core_ridx * basicRowLen * SPLIT_NUM * colLen + core_cidx * colLenPerCore +
                    cidx * basicColLen;
            }
            CopyIn(splitCopyinOffset, splitCopyinParams, yGradCopyinOffset, yGradCopyinParams);
            Compute(basicRowLen * basicColLen);
            CopyOut(splitCopyoutOffset, splitCopyoutParams);
        }
    }

    __aicore__ inline void ProcessGradCoreMultiUbMultiAlign(uint32_t ridx, uint64_t coreIdx, uint16_t basicRowLenCal)
    {
        AscendC::DataCopyParams splitCopyinParams;
        AscendC::DataCopyParams yGradCopyinParams;
        AscendC::DataCopyParams splitCopyoutParams;
        uint64_t splitCopyinOffset;
        uint64_t yGradCopyinOffset;
        uint64_t splitCopyoutOffset;
        splitCopyinParams = {basicRowLenCal, basicColLen * sizeof(inType) / BLOCK_SIZE,
            (SPLIT_NUM * colLen - basicColLen) * sizeof(inType) / BLOCK_SIZE, 0};
        splitCopyinOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen +
            ridx * basicRowLen * SPLIT_NUM * colLen;
        yGradCopyinParams = {basicRowLenCal, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
        yGradCopyinOffset = coreIdx * rowLenPerCore * colLen + ridx * basicRowLen * colLen;
        splitCopyoutParams = {basicRowLenCal, basicColLen * sizeof(outType) / BLOCK_SIZE, 0,
            (SPLIT_NUM * colLen - basicColLen) * sizeof(outType) / BLOCK_SIZE};
        splitCopyoutOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen + ridx * basicRowLen * SPLIT_NUM * colLen;
        CopyIn(splitCopyinOffset, splitCopyinParams, yGradCopyinOffset, yGradCopyinParams);
        Compute(basicRowLen * basicColLen);
        CopyOut(splitCopyoutOffset, splitCopyoutParams);
    }

    __aicore__ inline void ProcessGradCoreMultiUbMultiNotAlign(uint32_t ridx, uint64_t coreIdx, uint16_t basicRowLenCal)
    {
        AscendC::DataCopyExtParams splitCopyinParams;
        AscendC::DataCopyExtParams yGradCopyinParams;
        AscendC::DataCopyExtParams splitCopyoutParams;
        uint64_t splitCopyinOffset;
        uint64_t yGradCopyinOffset;
        uint64_t splitCopyoutOffset;
        AscendC::DataCopyPadExtParams<inType> padParams;
        // 先处理一个basicblock
        splitCopyinParams = {basicRowLenCal, basicColLen * sizeof(inType),
                             (SPLIT_NUM * colLen - basicColLen) * sizeof(inType), 0, 0};
        splitCopyinOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen +
            ridx * basicRowLen * SPLIT_NUM * colLen;
        yGradCopyinParams = {basicRowLenCal, basicColLen * sizeof(inType),
                             (colLen - basicColLen) * sizeof(inType), 0, 0};
        yGradCopyinOffset = coreIdx * rowLenPerCore * colLen + ridx * basicRowLen * colLen;
        padParams = {false, 0, 0, 0};
        splitCopyoutParams = {basicRowLenCal, basicColLen * sizeof(outType), 0,
                              (SPLIT_NUM * colLen - basicColLen) * sizeof(outType), 0};
        splitCopyoutOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen + ridx * basicRowLen * SPLIT_NUM * colLen;
        CopyInPad(splitCopyinOffset, splitCopyinParams, yGradCopyinOffset, yGradCopyinParams, padParams);
        Compute(basicRowLen * basicColLen);
        CopyOutPad(splitCopyoutOffset, splitCopyoutParams);
        // 再处理最后一个(列32B)的block
        splitCopyinParams = {basicRowLenCal, (colLenPerCore - basicColLen) * sizeof(inType),
            (SPLIT_NUM * colLen - (colLenPerCore - basicColLen)) * sizeof(inType), 0, 0};
        splitCopyinOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen +
            ridx * basicRowLen * SPLIT_NUM * colLen + basicColLen;
        yGradCopyinParams = {basicRowLenCal, (colLenPerCore - basicColLen) * sizeof(inType),
            (colLen - (colLenPerCore - basicColLen)) * sizeof(inType), 0, 0};
        yGradCopyinOffset = coreIdx * rowLenPerCore * colLen + ridx * basicRowLen * colLen + basicColLen;
        padParams = {false, 0, 0, 0};
        splitCopyoutParams = {basicRowLenCal, (colLenPerCore - basicColLen) * sizeof(outType), 0,
            (SPLIT_NUM * colLen - (colLenPerCore - basicColLen)) * sizeof(outType), 0};
        splitCopyoutOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen + ridx * basicRowLen * SPLIT_NUM * colLen +
            basicColLen;
        CopyInPad(splitCopyinOffset, splitCopyinParams, yGradCopyinOffset, yGradCopyinParams, padParams);
        Compute(basicRowLen * basicColLen);
        CopyOutPad(splitCopyoutOffset, splitCopyoutParams);
    }

    __aicore__ inline void ProcessGradCoreMultiUbMulti() // 一核处理多行, ub空间可容纳多行
    {
        uint64_t coreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint32_t rowLenPerCoreCal = coreIdx == AscendC::GetBlockNum() - 1 ?
            rowLen - (AscendC::GetBlockNum() - 1) * rowLenPerCore : rowLenPerCore; // 最后一核单独处理

        uint32_t rowLoop = CeilDiv(rowLenPerCoreCal, static_cast<uint32_t>(basicRowLen));
        uint16_t basicRowLenCal = basicRowLen;
        for (uint32_t ridx = 0; ridx < rowLoop; ridx++) {
            basicRowLenCal = static_cast<uint16_t>((ridx == rowLoop - 1) ?
                (rowLenPerCoreCal - (rowLoop - 1) * basicRowLen) : basicRowLen); // 每核处理的最后一个行循环单独处理

            if (colLen % (BLOCK_SIZE / sizeof(inType)) == 0) { // colLen 32B对齐, 此时basicColLen = colLenPerCore
                ProcessGradCoreMultiUbMultiAlign(ridx, coreIdx, basicRowLenCal);
            } else { // colLen 非32B对齐
                ProcessGradCoreMultiUbMultiNotAlign(ridx, coreIdx, basicRowLenCal);
            }
        }
    }

    __aicore__ inline void CopyIn(uint64_t splitCopyinOffset, AscendC::DataCopyParams& splitCopyinParams,
                                  uint64_t yGradCopyinOffset, AscendC::DataCopyParams& yGradCopyinParams)
    {
        // Copy A
        AscendC::LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        DataCopy(aLocal, this->xGm[splitCopyinOffset], splitCopyinParams);
        this->inQueueA.template EnQue(aLocal);
        // Copy B
        AscendC::LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();
        DataCopy(bLocal, this->xGm[splitCopyinOffset + colLen], splitCopyinParams);
        this->inQueueB.template EnQue(bLocal);
        // Copy y_grad
        AscendC::LocalTensor<inType> yGradLocal = this->inQueueYGrad.template AllocTensor<inType>();
        DataCopy(yGradLocal, this->yGradGm[yGradCopyinOffset], yGradCopyinParams);
        this->inQueueYGrad.template EnQue(yGradLocal);
    }

    __aicore__ inline void CopyOut(uint64_t splitCopyoutOffset, AscendC::DataCopyParams& splitCopyoutParams)
    {
        // Copy AGrad
        AscendC::LocalTensor<outType> aGradLocal = this->outQueueAGrad.template DeQue<outType>();
        DataCopy(this->xGradGm[splitCopyoutOffset], aGradLocal, splitCopyoutParams);
        this->outQueueAGrad.template FreeTensor(aGradLocal);
        // Copy BGrad
        AscendC::LocalTensor<outType> bGradLocal = this->outQueueBGrad.template DeQue<outType>();
        DataCopy(this->xGradGm[splitCopyoutOffset + colLen], bGradLocal, splitCopyoutParams);
        this->outQueueBGrad.template FreeTensor(bGradLocal);
    }

    __aicore__ inline void CopyInPad(uint64_t splitCopyinOffset, AscendC::DataCopyExtParams& splitCopyinParams,
        uint64_t yGradCopyinOffset, AscendC::DataCopyExtParams& yGradCopyinParams,
        AscendC::DataCopyPadExtParams<inType>& padParams)
    {
        // Copy A
        AscendC::LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        DataCopyPad(aLocal, this->xGm[splitCopyinOffset], splitCopyinParams, padParams);
        this->inQueueA.template EnQue(aLocal);
        // Copy B
        AscendC::LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();
        DataCopyPad(bLocal, this->xGm[splitCopyinOffset + colLen], splitCopyinParams, padParams);
        this->inQueueB.template EnQue(bLocal);
        // Copy yGrad
        AscendC::LocalTensor<inType> yGradLocal = this->inQueueYGrad.template AllocTensor<inType>();
        DataCopyPad(yGradLocal, this->yGradGm[yGradCopyinOffset], yGradCopyinParams, padParams);
        this->inQueueYGrad.template EnQue(yGradLocal);
    }

    __aicore__ inline void CopyOutPad(uint64_t splitCopyoutOffset, AscendC::DataCopyExtParams& splitCopyoutParams)
    {
        // Copy AGrad
        AscendC::LocalTensor<outType> aGradLocal = this->outQueueAGrad.template DeQue<outType>();
        DataCopyPad(this->xGradGm[splitCopyoutOffset], aGradLocal, splitCopyoutParams);
        this->outQueueAGrad.template FreeTensor(aGradLocal);
        // Copy BGrad
        AscendC::LocalTensor<outType> bGradLocal = this->outQueueBGrad.template DeQue<outType>();
        DataCopyPad(this->xGradGm[splitCopyoutOffset + colLen], bGradLocal, splitCopyoutParams);
        this->outQueueBGrad.template FreeTensor(bGradLocal);
    }

    __aicore__ inline void Compute(uint64_t tileLength)
    {
        AscendC::LocalTensor<inType> aLocal = inQueueA.template DeQue<inType>();
        AscendC::LocalTensor<inType> bLocal = inQueueB.template DeQue<inType>();
        AscendC::LocalTensor<inType> yGradLocal = inQueueYGrad.template DeQue<inType>();
        AscendC::LocalTensor<float> tmpALocal = tmpA.Get<float>();
        AscendC::LocalTensor<float> tmpBLocal = tmpB.Get<float>();
        AscendC::LocalTensor<float> tmpYGradLocal = tmpYGrad.Get<float>();
        AscendC::LocalTensor<float> tmpSigmoidLocal = tmpSigmoid.Get<float>();
        AscendC::LocalTensor<float> tmpAGradLocal = tmpAGrad.Get<float>();
        AscendC::LocalTensor<float> tmpBGradLocal = tmpBGrad.Get<float>();

        if constexpr (sizeof(inType) == sizeof(float)) {
            DataCopy(tmpALocal, aLocal, tileLength);
            DataCopy(tmpBLocal, bLocal, tileLength);
            DataCopy(tmpYGradLocal, yGradLocal, tileLength);
        } else {
            Cast(tmpALocal, aLocal, AscendC::RoundMode::CAST_NONE, tileLength);
            Cast(tmpBLocal, bLocal, AscendC::RoundMode::CAST_NONE, tileLength);
            Cast(tmpYGradLocal, yGradLocal, AscendC::RoundMode::CAST_NONE, tileLength);
        }
        inQueueA.FreeTensor(aLocal);
        inQueueB.FreeTensor(bLocal);
        inQueueYGrad.FreeTensor(yGradLocal);

        // compute sigmoid
        AscendC::PipeBarrier<PIPE_V>();
        Muls(tmpSigmoidLocal, tmpALocal, static_cast<float>(-1.0), tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        Exp(tmpSigmoidLocal, tmpSigmoidLocal, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(tmpSigmoidLocal, tmpSigmoidLocal, static_cast<float>(1.0), tileLength);
        Duplicate(tmpAGradLocal, static_cast<float>(1.0), tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        Div(tmpSigmoidLocal, tmpAGradLocal, tmpSigmoidLocal, tileLength);

        // compute b_grad
        AscendC::PipeBarrier<PIPE_V>();
        Mul(tmpBGradLocal, tmpSigmoidLocal, tmpALocal, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(tmpBGradLocal, tmpBGradLocal, tmpYGradLocal, tileLength);

        // compute a_grad
        Sub(tmpAGradLocal, tmpAGradLocal, tmpSigmoidLocal, tileLength); // 1-sigmoid
        AscendC::PipeBarrier<PIPE_V>();
        Mul(tmpAGradLocal, tmpAGradLocal, tmpALocal, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(tmpAGradLocal, tmpAGradLocal, tmpSigmoidLocal, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        Add(tmpAGradLocal, tmpAGradLocal, tmpSigmoidLocal, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(tmpYGradLocal, tmpYGradLocal, tmpBLocal, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(tmpAGradLocal, tmpAGradLocal, tmpYGradLocal, tileLength);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<outType> AGradLocal = outQueueAGrad.template AllocTensor<outType>();
        AscendC::LocalTensor<outType> BGradLocal = outQueueBGrad.template AllocTensor<outType>();
        if constexpr (sizeof(inType) == sizeof(float)) {
            DataCopy(AGradLocal, tmpAGradLocal, tileLength);
            DataCopy(BGradLocal, tmpBGradLocal, tileLength);
        } else {
            Cast(AGradLocal, tmpAGradLocal, AscendC::RoundMode::CAST_RINT, tileLength);
            Cast(BGradLocal, tmpBGradLocal, AscendC::RoundMode::CAST_RINT, tileLength);
        }
        AscendC::PipeBarrier<PIPE_V>();
        outQueueAGrad.template EnQue<outType>(AGradLocal);
        outQueueBGrad.template EnQue<outType>(BGradLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueA;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueB, inQueueYGrad;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueAGrad, outQueueBGrad;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpA, tmpB, tmpYGrad, tmpAGrad, tmpBGrad, tmpSigmoid;

    AscendC::GlobalTensor<inType> xGm;
    AscendC::GlobalTensor<inType> yGradGm;
    AscendC::GlobalTensor<outType> xGradGm;

    uint32_t rowLen;
    uint32_t colLen;
    uint32_t rowLenPerCore;
    uint32_t colLenPerCore;
    uint16_t basicRowLen;
    uint16_t basicColLen;
    uint32_t coreNumPerLine;
};
#endif // OPP_SWI_GLU_BACKWARD_H