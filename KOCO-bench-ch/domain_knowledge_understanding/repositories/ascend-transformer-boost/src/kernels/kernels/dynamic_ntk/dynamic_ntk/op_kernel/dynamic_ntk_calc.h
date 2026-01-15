/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GTSOPS_DYNAMIC_NTK_CALC_H
#define GTSOPS_DYNAMIC_NTK_CALC_H

#include "kernel_operator.h"
#include "kernels/utils/kernel/kernel_utils.h"
#include "dynamic_ntk_tiling_kernel.h"

using namespace AscendC;


template <typename OutType>
class DynamicNTKCalc {
public:
    // todo 测试非DB场景的性能
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t POS_DSIZE = sizeof(int32_t);
    static constexpr uint32_t CALC_DSIZE = sizeof(float);
    static constexpr uint32_t OUT_DSIZE = sizeof(OutType);
    static constexpr uint32_t SEQLEN_DSIZE = sizeof(int32_t);
    static constexpr uint32_t CALC_MASK = 256 / CALC_DSIZE;
    static constexpr uint32_t ONE_BLK_CAL_ELEMENTS = ONE_BLK_SIZE / CALC_DSIZE;
    static constexpr uint32_t ONE_BLK_POS_ELEMENTS = ONE_BLK_SIZE / POS_DSIZE;
    static constexpr uint32_t ONE_BLK_OUT_ELEMENTS = ONE_BLK_SIZE / OUT_DSIZE;
    static constexpr uint32_t ONE_BLK_SEQLEN_ELEMENTS = ONE_BLK_SIZE / SEQLEN_DSIZE;
#if __CCE_AICORE__ == 200
    static constexpr RoundMode OUT_CAST_MODE = RoundMode::CAST_NONE;
#else
    static constexpr RoundMode OUT_CAST_MODE = RoundMode::CAST_ROUND;
#endif

    __aicore__ inline DynamicNTKCalc() {}

    __aicore__ inline void Init(TPipe& tPipe, GM_ADDR positionIds, GM_ADDR invFreqs, GM_ADDR seqLens, GM_ADDR sin,
        GM_ADDR cos, GM_ADDR workspace, const AsdOps::DynamicNTKTilingData* tilingData)
    {
        tiling.copyTiling(tilingData);
        posGm.SetGlobalBuffer((__gm__ int32_t *)positionIds, tiling.numTokens);
        invFreqGm.SetGlobalBuffer((__gm__ float *)invFreqs, tiling.batchNum * tiling.halfHeadDim);
        seqLenGm.SetGlobalBuffer((__gm__ int32_t *)seqLens, tiling.batchNum);
        uint64_t sinSize = static_cast<uint64_t>(tiling.numTokens) * tiling.headDim;
        sinGm.SetGlobalBuffer((__gm__ OutType *)sin, sinSize);
        cosGm.SetGlobalBuffer((__gm__ OutType *)cos, sinSize);

#if __CCE_AICORE__ == 200
        TBuf<> tbuf;
        tPipe.InitBuffer(tbuf, ONE_BLK_SIZE);
#endif
        uint64_t sinBufSize = tiling.posTileLen * tiling.freqTileLen * CALC_DSIZE;
        tPipe.InitBuffer(posQue, BUFFER_NUM, RoundUp(tiling.posTileLen * POS_DSIZE, ONE_BLK_SIZE));
        tPipe.InitBuffer(invFreqQue, BUFFER_NUM, RoundUp(tiling.freqTileLen * CALC_DSIZE, ONE_BLK_SIZE));
        tPipe.InitBuffer(sinQue, BUFFER_NUM, RoundUp(sinBufSize, ONE_BLK_SIZE));
        tPipe.InitBuffer(cosQue, BUFFER_NUM, RoundUp(sinBufSize, ONE_BLK_SIZE));
        tPipe.InitBuffer(tBuf, RoundUp(tiling.batchNum * SEQLEN_DSIZE, ONE_BLK_SIZE));
        mte2ScalarEvents[0] = EVENT_ID3;
        if constexpr (BUFFER_NUM > 1) {
            mte2ScalarEvents[1] = EVENT_ID2;
        }
        isPingPong = 0;
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> seqLenBuf = tBuf.Get<int32_t>();
        uint32_t alignUpLen = (tiling.batchNum + ONE_BLK_SEQLEN_ELEMENTS - 1) / ONE_BLK_SEQLEN_ELEMENTS
            * ONE_BLK_SEQLEN_ELEMENTS;
        AscendC::DataCopy(seqLenBuf, seqLenGm, alignUpLen);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);

        for (uint32_t fId = 0; fId < tiling.freqTileNum; fId++) {
            ProcessInner(seqLenBuf, fId, tiling.freqTileLen);
        }
        if (tiling.freqTailLen > 0) {
            ProcessInner(seqLenBuf, tiling.freqTileNum, tiling.freqTailLen);
        }
    }

    __aicore__ inline void ProcessInner(LocalTensor<int32_t> &seqLenBuf, uint32_t fId, uint32_t fLen)
    {
        // 找到当前核需要处理的起始batch
        uint32_t batchS = 0;
        uint64_t batchPosOff = 0;
        uint64_t curPosOff = tiling.posGmOff;
        uint64_t endPosOff = tiling.posGmOff + tiling.posTotalLen;
        for (uint32_t i = 0; i < tiling.batchNum; i++) {
            uint32_t seqLen = seqLenBuf.GetValue(i);
            if (batchPosOff + seqLen > curPosOff) {
                batchS = i;
                break;
            }
            batchPosOff += seqLen;
        }
/*
              batchPosOff
    |----seq1----|--------------------seq2-------------------|-------------------seq3-------------------|
    |--------------core0--------------|--------------core1--------------|--------------core2------------|
    |---tile0---|---tile1---|---tile2-|---tile0---|---tile1---|---tile2-|---tile0---|---tile1---|--tile2|
                                   posGmOff
                                   curPosOff
    以core1为例:
    batchS = 1
 */
        // 按PositionIds在行方向上，InvFreq在列方向上同时做切分，每次计算一个小矩阵
        uint64_t firstSeqLen = batchPosOff + seqLenBuf.GetValue(batchS) - curPosOff;
        for (uint32_t batchId = batchS; batchId < tiling.batchNum; batchId++) {
            if (curPosOff >= endPosOff) {
                break;
            }
            uint64_t leftLen = (batchId == batchS) ? firstSeqLen : seqLenBuf.GetValue(batchId);
            if (leftLen + curPosOff > endPosOff) {
                leftLen = endPosOff - curPosOff;
            }
            uint64_t curOff = 0;
            for (; (curOff + tiling.posTileLen) <= leftLen; curOff += tiling.posTileLen) {
                uint64_t gmOff = curPosOff + curOff;
                CopyIn<false>(fId, gmOff, fLen, tiling.posTileLen, batchId);
                Compute(fId, fLen, tiling.posTileLen);
                CopyOut(fId, gmOff, fLen, tiling.posTileLen);
                isPingPong = !isPingPong;
            }
            if (leftLen > curOff) {
                uint32_t pLen = leftLen - curOff;
                uint64_t gmOff = curPosOff + curOff;
                CopyIn<true>(fId, gmOff, fLen, pLen, batchId);
                Compute(fId, fLen, pLen);
                CopyOut(fId, gmOff, fLen, pLen);
                isPingPong = !isPingPong;
            }
            curPosOff += leftLen;
        }
    }

    template<bool IsTail>
    __aicore__ inline void CopyIn(uint32_t fId, uint64_t posOff, uint32_t fLen, uint32_t pLen, uint32_t batchId)
    {
        // 拷贝PositionIds的数据
        LocalTensor<int32_t> posBuf = posQue.AllocTensor<int32_t>();
        if constexpr (IsTail) {
            uint32_t alignUpLen = (pLen + ONE_BLK_POS_ELEMENTS - 1) / ONE_BLK_POS_ELEMENTS * ONE_BLK_POS_ELEMENTS;
            // 向后方多拷贝几个数据
            AscendC::DataCopy(posBuf, posGm[posOff], alignUpLen);
        } else {
            AscendC::DataCopy(posBuf, posGm[posOff], pLen);
        }
        posQue.EnQue(posBuf);
        set_flag(PIPE_MTE2, PIPE_S, mte2ScalarEvents[isPingPong]);

        // 拷贝InvFreqIn或者auxArray的数据
        // invFreq默认是32对齐的
        LocalTensor<float> freqBuf = invFreqQue.AllocTensor<float>();
        uint64_t gmOff = static_cast<uint64_t>(batchId) * tiling.halfHeadDim + fId * tiling.freqTileLen;
        DataCopy(freqBuf, invFreqGm[gmOff], fLen);
        invFreqQue.EnQue(freqBuf);
    }

    __aicore__ inline void Compute(uint32_t fId, uint32_t fLen, uint32_t pLen)
    {
        // 先将PositionIds转为float类型
        LocalTensor<int32_t> posIntBuf = posQue.DeQue<int32_t>();
        LocalTensor<float> posBuf = posIntBuf.template ReinterpretCast<float>();
        Cast(posBuf, posIntBuf, RoundMode::CAST_NONE, pLen);

        LocalTensor<float> freqBuf = invFreqQue.DeQue<float>();
        // emb = outer(positionIds, invFreq)
        LocalTensor<float> embBuf = cosQue.AllocTensor<float>();
        wait_flag(PIPE_MTE2, PIPE_S, mte2ScalarEvents[isPingPong]);
        for (uint32_t i = 0; i < pLen; i++) {
            set_flag(PIPE_V, PIPE_S, mte2ScalarEvents[isPingPong]);
            wait_flag(PIPE_V, PIPE_S, mte2ScalarEvents[isPingPong]);
            auto posId = posBuf.GetValue(i);
            ::Muls(embBuf[i * fLen], freqBuf, posId, fLen);
        }
        pipe_barrier(PIPE_V);
        invFreqQue.FreeTensor(freqBuf);
        posQue.FreeTensor(posBuf);

        LocalTensor<float> sinBuf = sinQue.AllocTensor<float>();
        Sin<float, false>(sinBuf, embBuf);
        if constexpr (std::is_same_v<OutType, float>) {
            sinQue.EnQue(sinBuf);
        } else {
            LocalTensor<OutType> sinOutBuf = sinBuf.template ReinterpretCast<OutType>();
            pipe_barrier(PIPE_V);
            Cast(sinOutBuf, sinBuf, OUT_CAST_MODE, pLen * fLen);
            sinQue.EnQue(sinOutBuf);
        }

        Cos<float, false>(embBuf, embBuf);
        if constexpr (std::is_same_v<OutType, float>) {
            cosQue.EnQue(embBuf);
        } else {
            pipe_barrier(PIPE_V);
            LocalTensor<OutType> cosOutBuf = embBuf.template ReinterpretCast<OutType>();
            Cast(cosOutBuf, embBuf, OUT_CAST_MODE, pLen * fLen);
            cosQue.EnQue(cosOutBuf);
        }
    }

    __aicore__ inline void CopyOut(uint32_t fId, uint64_t posOff, uint32_t fLen, uint32_t pLen)
    {
        LocalTensor<OutType> sinBuf = sinQue.DeQue<OutType>();
        DataCopyParams copyParam = {(uint16_t)pLen, (uint16_t)(fLen / ONE_BLK_OUT_ELEMENTS), 0,
            (uint16_t)((tiling.headDim - fLen) / ONE_BLK_OUT_ELEMENTS)};
        uint64_t gmOff1 = posOff * static_cast<uint64_t>(tiling.headDim)
            + static_cast<uint64_t>(fId) * tiling.freqTileLen;
        uint64_t gmOff2 = gmOff1 + static_cast<uint64_t>(tiling.halfHeadDim);
        DataCopy(sinGm[gmOff1], sinBuf, copyParam);
        DataCopy(sinGm[gmOff2], sinBuf, copyParam);
        sinQue.FreeTensor(sinBuf);

        LocalTensor<OutType> cosBuf = cosQue.DeQue<OutType>();
        DataCopy(cosGm[gmOff1], cosBuf, copyParam);
        DataCopy(cosGm[gmOff2], cosBuf, copyParam);
        cosQue.FreeTensor(cosBuf);
    }

private:
    GlobalTensor<int32_t> posGm;
    GlobalTensor<float> invFreqGm;
    GlobalTensor<int32_t> seqLenGm;
    GlobalTensor<OutType> sinGm;
    GlobalTensor<OutType> cosGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> posQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> invFreqQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> sinQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> cosQue;
    TBuf<> tBuf;
    event_t mte2ScalarEvents[BUFFER_NUM];
    uint32_t isPingPong;
    DynamicNTKTilingKernel tiling;
};


#endif // GTSOPS_DYNAMIC_NTK_CALC_H
