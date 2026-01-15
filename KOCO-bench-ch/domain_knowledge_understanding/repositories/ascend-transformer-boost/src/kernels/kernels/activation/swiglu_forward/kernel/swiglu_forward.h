/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPP_SWI_GLU_FORWARD_H
#define OPP_SWI_GLU_FORWARD_H
#include "kernel_operator.h"
#include "kernels/activation/swiglu_forward/tiling/tiling_data.h"

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t SPLIT_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 32;

template <typename T>
__aicore__ inline T CeilDiv(T x, T y) {
    return y == 0 ? 0 : (x + y - 1) / y;
}

template<typename inType, typename outType>
class SwigluForward {
public:
    __aicore__ inline SwigluForward() {}
    __aicore__ inline ~SwigluForward() {}

    __aicore__ inline void Init(GM_ADDR input_gm, GM_ADDR output_gm, AsdOps::SwiGluForwardTilingData &tiling_data)
    {
        this->rowLen = tiling_data.rowLen;
        this->colLen = tiling_data.colLen;
        this->rowLenPerCore = tiling_data.rowLenPerCore;
        this->colLenPerCore = tiling_data.colLenPerCore;
        this->basicRowLen = tiling_data.basicRowLen;
        this->basicColLen = tiling_data.basicColLen;
        this->coreNumPerLine = tiling_data.coreNumPerLine;
        xGm.SetGlobalBuffer((__gm__ inType*)input_gm, SPLIT_NUM * rowLen * colLen);
        yGm.SetGlobalBuffer((__gm__ outType*)output_gm, rowLen * colLen);
        uint64_t tileLength = basicRowLen * (basicColLen == 0 ? (BLOCK_SIZE / sizeof(inType)) : basicColLen);
        pipe.InitBuffer(inQueueA, BUFFER_NUM, tileLength * sizeof(inType));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, tileLength * sizeof(inType));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(outType));
        pipe.InitBuffer(tempBufferA, tileLength * sizeof(float));
        pipe.InitBuffer(tempBufferB, tileLength * sizeof(float));
        pipe.InitBuffer(tempBufferY, tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (basicColLen == 0) { // 边界情况，n在32B以内
            ProcessSmallN();
        } else if (colLenPerCore == colLen) { // 一核处理多行的情况
            if (basicRowLen == 1) { // ub空间可容纳1行内数据
                ProcessCoreMultiUbSingle();
            } else { // ub空间可容纳多行
                ProcessCoreMultiUbMulti();
            }
        } else { // 一行多核处理的情况
            ProcessCoreSingle();
        }
    }

    __aicore__ inline void ProcessSmallN() // // 边界情况，n在32B以内
    {
        uint64_t coreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint32_t rowLenPerCoreCal = coreIdx == AscendC::GetBlockNum() - 1 ?
            rowLen - (AscendC::GetBlockNum() - 1) * rowLenPerCore : rowLenPerCore; // 最后一核单独处理

        uint32_t rowLoop = CeilDiv(rowLenPerCoreCal, (uint32_t)basicRowLen);
        uint16_t basicRowLenCal = basicRowLen;
        for (uint32_t ridx = 0; ridx < rowLoop; ridx++) {
            // 每核处理的最后一个行循环单独处理
            basicRowLenCal = ((ridx == rowLoop - 1) ?
                static_cast<uint16_t>(rowLenPerCoreCal - (rowLoop - 1) * basicRowLen) : basicRowLen);

            AscendC::DataCopyExtParams splitCopyinParams;
            AscendC::DataCopyExtParams splitCopyoutParams;
            uint64_t splitCopyinOffset;
            uint64_t splitCopyoutOffset;
            AscendC::DataCopyPadExtParams<inType> padParams;
            splitCopyinParams = {basicRowLenCal, colLen * sizeof(inType), colLen * sizeof(inType), 0, 0};
            splitCopyinOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen + ridx * basicRowLen * SPLIT_NUM * colLen;
            padParams = {false, 0, 0, 0};
            splitCopyoutParams = {basicRowLenCal, colLen * sizeof(outType), 0, 0, 0};
            splitCopyoutOffset = coreIdx * rowLenPerCore * colLen + ridx * basicRowLen * colLen;
            CopyInPad(splitCopyinOffset, splitCopyinParams, padParams);
            Compute(basicRowLen * BLOCK_SIZE / sizeof(inType));
            CopyOutPad(splitCopyoutOffset, splitCopyoutParams);
        }
    }

    __aicore__ inline void ProcessCoreSingle() // 一行多核处理的情况
    {
        uint32_t core_num_per_line = coreNumPerLine;
        uint64_t coreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint64_t core_ridx = coreIdx / core_num_per_line;
        uint64_t core_cidx = coreIdx % core_num_per_line;
        uint32_t colLenPerCoreCal = core_cidx == core_num_per_line - 1 ?
            colLen - (core_num_per_line - 1) * colLenPerCore: colLenPerCore;
        uint32_t colLoop = CeilDiv(colLenPerCoreCal, static_cast<uint32_t>(basicColLen));
        AscendC::DataCopyParams splitCopyinParams;
        AscendC::DataCopyParams splitCopyoutParams;
        uint64_t splitCopyinOffset;
        uint64_t splitCopyoutOffset;
        for (uint32_t cidx = 0; cidx < colLoop; cidx++) {
            if (cidx == colLoop - 1) { // 无论colLen是否32B对齐，每行最后一块都回退到(列basicColLen)再处理
                splitCopyinParams = {basicRowLen, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
                splitCopyinOffset = core_ridx * basicRowLen * SPLIT_NUM * colLen + core_cidx * colLenPerCore +
                    colLenPerCoreCal - basicColLen;
                splitCopyoutParams = {basicRowLen, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
                splitCopyoutOffset = core_ridx * basicRowLen * colLen + core_cidx * colLenPerCore +
                    colLenPerCoreCal - basicColLen;
            } else {
                splitCopyinParams = {basicRowLen, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
                splitCopyinOffset = core_ridx * basicRowLen * SPLIT_NUM * colLen + core_cidx * colLenPerCore +
                    cidx * basicColLen;
                splitCopyoutParams = {basicRowLen, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
                splitCopyoutOffset = core_ridx * basicRowLen * colLen + core_cidx * colLenPerCore +
                    cidx * basicColLen;
            }
            CopyIn(splitCopyinOffset, splitCopyinParams);
            Compute(basicRowLen * basicColLen);
            CopyOut(splitCopyoutOffset, splitCopyoutParams);
        }
    }

    __aicore__ inline void ProcessCoreMultiUbSingle() // 一核处理多行, ub空间可容纳1行内数据
    {
        uint64_t coreIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint32_t rowLenPerCoreCal = coreIdx == AscendC::GetBlockNum() - 1 ?
            rowLen - (AscendC::GetBlockNum() - 1) * rowLenPerCore : rowLenPerCore; // 最后一核单独处理

        uint32_t colLoop = CeilDiv(colLenPerCore, static_cast<uint32_t>(basicColLen));
        uint16_t basicColLenCal = basicColLen;
        for (uint32_t ridx = 0; ridx < rowLenPerCoreCal; ridx++) {
            for (uint32_t cidx = 0; cidx < colLoop; cidx++) {
                basicColLenCal = cidx == colLoop - 1 ?
                    static_cast<uint16_t>(colLenPerCore - (colLoop - 1) * basicColLen) : basicColLen;

                AscendC::DataCopyParams splitCopyinParams;
                AscendC::DataCopyParams splitCopyoutParams;
                uint64_t splitCopyinOffset;
                uint64_t splitCopyoutOffset;
                if (cidx == colLoop - 1) { // 无论colLen是否32B对齐，每行最后一块都回退到(列basicColLen)再处理
                    splitCopyinParams = {basicRowLen, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
                    splitCopyinOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen +
                        ridx * basicRowLen * SPLIT_NUM * colLen + colLenPerCore - basicColLen;
                    splitCopyoutParams = {basicRowLen, basicColLen * sizeof(outType) / BLOCK_SIZE, 0, 0};
                    splitCopyoutOffset = coreIdx * rowLenPerCore * colLen + ridx * basicRowLen * colLen +
                        colLenPerCore - basicColLen;
                } else {
                    splitCopyinParams = {basicRowLen, basicColLen * sizeof(inType) / BLOCK_SIZE, 0, 0};
                    splitCopyinOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen +
                        ridx * basicRowLen * SPLIT_NUM * colLen + cidx * basicColLen;
                    splitCopyoutParams = {basicRowLen, basicColLen * sizeof(outType) / BLOCK_SIZE, 0, 0};
                    splitCopyoutOffset = coreIdx * rowLenPerCore * colLen + ridx * basicRowLen * colLen +
                        cidx * basicColLen;
                }
                CopyIn(splitCopyinOffset, splitCopyinParams);
                Compute(basicRowLen * basicColLen);
                CopyOut(splitCopyoutOffset, splitCopyoutParams);
            }
        }
    }

    __aicore__ inline void ProcessCoreMultiUbMulti() // 一核处理多行, ub空间可容纳多行
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
                ProcessCoreMultiUbMultiAlign(ridx, coreIdx, basicRowLenCal);
            } else { // colLen 非32B对齐
                ProcessCoreMultiUbMultiNotAlign(ridx, coreIdx, basicRowLenCal);
            }
        }
    }

    __aicore__ inline void ProcessCoreMultiUbMultiAlign(uint32_t ridx, uint64_t coreIdx, uint16_t basicRowLenCal)
    {
        AscendC::DataCopyParams splitCopyinParams;
        AscendC::DataCopyParams splitCopyoutParams;
        uint64_t splitCopyinOffset;
        uint64_t splitCopyoutOffset;
        splitCopyinParams = {basicRowLenCal, basicColLen * sizeof(inType) / BLOCK_SIZE,
            (SPLIT_NUM * colLen - basicColLen) * sizeof(inType) / BLOCK_SIZE, 0};
        splitCopyinOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen +
            ridx * basicRowLen * SPLIT_NUM * colLen;
        splitCopyoutParams = {basicRowLenCal, basicColLen * sizeof(outType) / BLOCK_SIZE, 0,
            (colLenPerCore - basicColLen) * sizeof(outType) / BLOCK_SIZE};
        splitCopyoutOffset = coreIdx * rowLenPerCore * colLen + ridx * basicRowLen * colLen;
        CopyIn(splitCopyinOffset, splitCopyinParams);
        Compute(basicRowLen * basicColLen);
        CopyOut(splitCopyoutOffset, splitCopyoutParams);
    }

    __aicore__ inline void ProcessCoreMultiUbMultiNotAlign(uint32_t ridx, uint64_t coreIdx, uint16_t basicRowLenCal)
    {
        AscendC::DataCopyExtParams splitCopyinParams;
        AscendC::DataCopyExtParams splitCopyoutParams;
        uint64_t splitCopyinOffset;
        uint64_t splitCopyoutOffset;
        AscendC::DataCopyPadExtParams<inType> padParams;
        // 先处理一个basicblock
        splitCopyinParams = {basicRowLenCal, basicColLen * sizeof(inType),
                             (SPLIT_NUM * colLen - basicColLen) * sizeof(inType), 0, 0};
        splitCopyinOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen +
            ridx * basicRowLen * SPLIT_NUM * colLen;
        padParams = {false, 0, 0, 0};
        splitCopyoutParams = {basicRowLenCal, (basicColLen * sizeof(outType)), 0,
                              (colLenPerCore - basicColLen) * sizeof(outType), 0};
        splitCopyoutOffset = coreIdx * rowLenPerCore * colLen + ridx * basicRowLen * colLen;
        CopyInPad(splitCopyinOffset, splitCopyinParams, padParams);
        Compute(basicRowLen * basicColLen);
        CopyOutPad(splitCopyoutOffset, splitCopyoutParams);
        // 再处理最后一个(列32B)的block
        splitCopyinParams = {basicRowLenCal, (colLenPerCore - basicColLen) * sizeof(inType),
            (SPLIT_NUM * colLen - (colLenPerCore - basicColLen)) * sizeof(inType), 0, 0};
        splitCopyinOffset = coreIdx * rowLenPerCore * SPLIT_NUM * colLen +
            ridx * basicRowLen * SPLIT_NUM * colLen + basicColLen;
        padParams = {false, 0, 0, 0};
        splitCopyoutParams = {basicRowLenCal, (colLenPerCore - basicColLen) * sizeof(outType), 0,
            (colLen - (colLenPerCore - basicColLen)) * sizeof(outType), 0};
        splitCopyoutOffset = coreIdx * rowLenPerCore * colLen + ridx * basicRowLen * colLen + basicColLen;
        CopyInPad(splitCopyinOffset, splitCopyinParams, padParams);
        Compute(basicRowLen * basicColLen);
        CopyOutPad(splitCopyoutOffset, splitCopyoutParams);
    }

    __aicore__ inline void CopyIn(uint64_t splitCopyinOffset, AscendC::DataCopyParams& splitCopyinParams)
    {
        // Copy A
        AscendC::LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        DataCopy(aLocal, this->xGm[splitCopyinOffset], splitCopyinParams);
        this->inQueueA.template EnQue(aLocal);
        // Copy B
        AscendC::LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();
        DataCopy(bLocal, this->xGm[splitCopyinOffset + colLen], splitCopyinParams);
        this->inQueueB.template EnQue(bLocal);
    }

    __aicore__ inline void CopyOut(uint64_t splitCopyoutOffset, AscendC::DataCopyParams& splitCopyoutParams)
    {
        AscendC::LocalTensor<outType> cLocal = this->outQueueY.template DeQue<outType>();
        DataCopy(this->yGm[splitCopyoutOffset], cLocal, splitCopyoutParams);
        this->outQueueY.template FreeTensor(cLocal);
    }

    __aicore__ inline void CopyInPad(uint64_t splitCopyinOffset, AscendC::DataCopyExtParams& splitCopyinParams,
        AscendC::DataCopyPadExtParams<inType>& padParams)
    {
        // Copy A
        AscendC::LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        DataCopyPad(aLocal, this->xGm[splitCopyinOffset], splitCopyinParams, padParams);
#endif
        this->inQueueA.template EnQue(aLocal);
        // Copy B
        AscendC::LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        DataCopyPad(bLocal, this->xGm[splitCopyinOffset + colLen], splitCopyinParams, padParams);
#endif
        this->inQueueB.template EnQue(bLocal);
    }

    __aicore__ inline void CopyOutPad(uint64_t splitCopyoutOffset, AscendC::DataCopyExtParams& splitCopyoutParams)
    {
        AscendC::LocalTensor<outType> cLocal = this->outQueueY.template DeQue<outType>();
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        DataCopyPad(this->yGm[splitCopyoutOffset], cLocal, splitCopyoutParams);
#endif
        this->outQueueY.template FreeTensor(cLocal);
    }

    __aicore__ inline void Compute(uint64_t tileLength)
    {
        AscendC::LocalTensor<float> tmpALocal = tempBufferA.Get<float>();
        AscendC::LocalTensor<float> tmpBLocal = tempBufferB.Get<float>();
        AscendC::LocalTensor<float> tmpYLocal = tempBufferY.Get<float>();

        AscendC::LocalTensor<inType> aLocal = inQueueA.template DeQue<inType>();
        if constexpr (sizeof(inType) == sizeof(float)) {
            DataCopy(tmpALocal, aLocal, tileLength);
        } else {
            Cast(tmpALocal, aLocal, AscendC::RoundMode::CAST_NONE, tileLength);
        }
        inQueueA.template FreeTensor(aLocal);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(tmpYLocal, tmpALocal, static_cast<float>(-1.0), tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        Exp(tmpYLocal, tmpYLocal, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(tmpYLocal, tmpYLocal, static_cast<float>(1.0), tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        Div(tmpYLocal, tmpALocal, tmpYLocal, tileLength);

        AscendC::LocalTensor<inType> bLocal = inQueueB.template DeQue<inType>();
        if constexpr (sizeof(inType) == sizeof(float)) {
            DataCopy(tmpBLocal, bLocal, tileLength);
        } else {
            Cast(tmpBLocal, bLocal, AscendC::RoundMode::CAST_NONE, tileLength);
        }
        inQueueB.template FreeTensor(bLocal);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(tmpYLocal, tmpYLocal, tmpBLocal, tileLength);

        AscendC::LocalTensor<outType> cLocal = outQueueY.template AllocTensor<outType>();
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (sizeof(inType) == sizeof(float)) {
            DataCopy(cLocal, tmpYLocal, tileLength);
        } else {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
            Cast(cLocal, tmpYLocal, AscendC::RoundMode::CAST_RINT, tileLength);
#else
            Cast(cLocal, tmpYLocal, AscendC::RoundMode::CAST_NONE, tileLength);
#endif
        }
        AscendC::PipeBarrier<PIPE_V>();
        outQueueY.template EnQue<outType>(cLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueA;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueB;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBufferA;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBufferB;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBufferY;

    AscendC::GlobalTensor<inType> xGm;
    AscendC::GlobalTensor<outType> yGm;

    uint32_t rowLen;
    uint32_t colLen;
    uint32_t rowLenPerCore;
    uint32_t colLenPerCore;
    uint16_t basicRowLen;
    uint16_t basicColLen;
    uint32_t coreNumPerLine;
};
#endif // OPP_SWI_GLU_FORWARD_H