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
#include "mixkernels/unpad/tiling/tiling_data.h"
#include "mixkernels/utils/common/kernel/kernel_utils.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t ELE_PER_BLK = 8;
constexpr int32_t MAX_BATCH_NUM = 64;

class KernelUnpad {
public:
    __aicore__ inline KernelUnpad() {}
    __aicore__ inline void Init(GM_ADDR input_ids,
                                GM_ADDR cum_offsets_now, GM_ADDR token_num,
                                GM_ADDR seq_len, GM_ADDR x_remove_padding,
                                GM_ADDR cum_offsets_out, GM_ADDR padding_offset,
                                uint32_t padLength_, uint32_t batch)
    {
        this->padLength_ = padLength_;
        this->batch_ = batch;
        inputIdsGm.SetGlobalBuffer((__gm__ int64_t *)input_ids, padLength_ * batch_);
        cumOffsetsNowGm.SetGlobalBuffer((__gm__ int32_t *)cum_offsets_now, batch_);
        tokenNumGm.SetGlobalBuffer((__gm__ int64_t *)token_num, 1);
        seqLenGm.SetGlobalBuffer((__gm__ int32_t *)seq_len, batch_);
        xRemovePaddingGm.SetGlobalBuffer((__gm__ int64_t *)x_remove_padding, padLength_ * batch_);
        cumOffsetOutGm.SetGlobalBuffer((__gm__ int32_t *)cum_offsets_out, batch_);
        paddingOffsetGm.SetGlobalBuffer((__gm__ int32_t *)padding_offset, padLength_ * batch_);
        padLengthAlign_ = ((padLength_ + ELE_PER_BLK - 1) / ELE_PER_BLK) * ELE_PER_BLK;
        batchAlign_ = ((batch_ + ELE_PER_BLK - 1) / ELE_PER_BLK) * ELE_PER_BLK;
        pipe_.InitBuffer(inputIdsQueue_, BUFFER_NUM, padLengthAlign_ * sizeof(int64_t));
        pipe_.InitBuffer(cumOffsetsQueue_, BUFFER_NUM, MAX_BATCH_NUM * sizeof(int32_t));
        pipe_.InitBuffer(seqLenQueue_, BUFFER_NUM, MAX_BATCH_NUM * sizeof(int32_t));
        pipe_.InitBuffer(xRemovePaddingQueue_, padLengthAlign_ * sizeof(int64_t));
        pipe_.InitBuffer(cumOffsetOutQueue_, BUFFER_NUM, MAX_BATCH_NUM * sizeof(int32_t));
        pipe_.InitBuffer(broadCastBuf_, padLengthAlign_ *  sizeof(int32_t));
        pipe_.InitBuffer(cumOffsetsBuf_, MAX_BATCH_NUM *  sizeof(int32_t));
        pipe_.InitBuffer(seqLenBuf_, MAX_BATCH_NUM *  sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint64_t i = 0; i < batch_; i++) {
            CopyIn(i);
            CopyOnce();
            AscendC::PipeBarrier<PIPE_MTE2>();
            ComputeOnce(i);
            ComputeRemovePadding();
            AscendC::PipeBarrier<PIPE_V>();
            CopyOut(i);
            CopyOutOnce(i);
        }
        ClearStep();
    }

private:
    __aicore__ inline void CopyOnce()
    {
        LocalTensor<int32_t> seqLenLocal = seqLenQueue_.AllocTensor<int32_t>();
        AscendC::LocalTensor<int32_t> cumOffsetsBuffer = cumOffsetsBuf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> seqLenBufBuffer = seqLenBuf_.Get<int32_t>();
        DataCopy(seqLenLocal, seqLenGm, batchAlign_);
        DataCopy(cumOffsetsBuffer, cumOffsetsNowGm, batchAlign_);
        DataCopy(seqLenBufBuffer, seqLenGm, batchAlign_);
        seqLenQueue_.EnQue(seqLenLocal);
    }

    __aicore__ inline void ComputeOnce(uint32_t progress)
    {
        AscendC::LocalTensor<int32_t> cumOffsetsBuffer = cumOffsetsBuf_.Get<int32_t>();
        LocalTensor<int32_t> seqLenLocal = seqLenQueue_.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> broadCast = broadCastBuf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> cumOffsetOutLocal = cumOffsetOutQueue_.AllocTensor<int32_t>();
        seqLenZero_ = seqLenLocal.GetValue(0);
        if (progress == 0) {
            Duplicate(broadCast, (int32_t)0, padLengthAlign_);
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            Duplicate(broadCast, (int32_t)cumOffsetsBuffer.GetValue(progress - 1), padLengthAlign_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        cumOffsetOutLocal.SetValue(0, (int32_t)0);
        for (uint32_t x = 1; x < batch_; x++) {
                cumOffsetOutLocal.SetValue(x, cumOffsetsBuffer.GetValue(x - 1));
        }
        cumOffsetOutQueue_.EnQue<int32_t>(cumOffsetOutLocal);
        seqLenQueue_.FreeTensor(seqLenLocal);
    }

    __aicore__ inline void CopyOutOnce(uint64_t progress)
    {
        AscendC::LocalTensor<int32_t> cumOffsetsBuffer = cumOffsetsBuf_.Get<int32_t>();
        LocalTensor<int32_t> cumOffsetOutLocal = cumOffsetOutQueue_.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> broadCast = broadCastBuf_.Get<int32_t>();
        if (progress == 0) {
            DataCopy(paddingOffsetGm, broadCast, padLengthAlign_);
        } else {
            DataCopy(paddingOffsetGm[seqLenZero_ + (progress - 1) * padLength_ -
                     cumOffsetsBuffer.GetValue(progress - 1) +
                     cumOffsetsBuffer.GetValue(0)], broadCast, padLengthAlign_);
        }
        DataCopy(cumOffsetOutGm, cumOffsetOutLocal, batchAlign_);
        cumOffsetOutQueue_.FreeTensor(cumOffsetOutLocal);
    }

    __aicore__ inline void CopyIn(uint64_t progress)
    {
        LocalTensor<int64_t> inputIdsLocal = inputIdsQueue_.AllocTensor<int64_t>();
        DataCopy(inputIdsLocal, inputIdsGm[progress * padLength_], padLengthAlign_);
        inputIdsQueue_.EnQue(inputIdsLocal);
    }

    __aicore__ inline void ComputeRemovePadding()
    {
        LocalTensor<int64_t> inputIdsLocal = inputIdsQueue_.DeQue<int64_t>();
        AscendC::LocalTensor<int32_t> broadCast = broadCastBuf_.Get<int32_t>();
        AscendC::LocalTensor<int64_t> xRemovePaddingLocal = xRemovePaddingQueue_.Get<int64_t>();
        DataCopy(xRemovePaddingLocal, inputIdsLocal, padLengthAlign_);
        AscendC::PipeBarrier<PIPE_ALL>();
        inputIdsQueue_.FreeTensor(inputIdsLocal);
    }
    __aicore__ inline void CopyOut(uint64_t progress)
    {
        AscendC::LocalTensor<int32_t> cumOffsetsBuffer = cumOffsetsBuf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> seqLenBufBuffer = seqLenBuf_.Get<int32_t>();
        LocalTensor<int64_t> xRemovePaddingLocal = xRemovePaddingQueue_.Get<int64_t>();
        AscendC::LocalTensor<int32_t> broadCast = broadCastBuf_.Get<int32_t>();

        if (progress == 0) {
            DataCopy(xRemovePaddingGm, xRemovePaddingLocal, padLengthAlign_);
        } else {
            DataCopy(xRemovePaddingGm[progress * padLength_ - cumOffsetsBuffer.GetValue(progress - 1)],
                     xRemovePaddingLocal, padLengthAlign_);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        for (int32_t i = 0; i < padLengthAlign_; i++) {
            xRemovePaddingLocal.SetValue(i, (int64_t)0);
        }
        DataCopy(xRemovePaddingGm[batch_ * padLength_ - cumOffsetsBuffer.GetValue(batch_ - 1)],
            xRemovePaddingLocal, padLengthAlign_);
    }

    __aicore__ inline void ClearStep()
    {
        AscendC::LocalTensor<int32_t> cumOffsetsBuffer = cumOffsetsBuf_.Get<int32_t>();
        LocalTensor<int64_t> xRemovePaddingLocal = xRemovePaddingQueue_.Get<int64_t>();
        AscendC::LocalTensor<int32_t> broadCast = broadCastBuf_.Get<int32_t>();
        AscendC::PipeBarrier<PIPE_ALL>();

        for (int32_t i = 0; i < padLengthAlign_; i++) {
            xRemovePaddingLocal.SetValue(i, (int64_t)0);
        }
        Duplicate(broadCast, (int32_t)0, padLengthAlign_);
        AscendC::PipeBarrier<PIPE_ALL>();

        DataCopy(paddingOffsetGm[batch_ * padLength_ - cumOffsetsBuffer.GetValue(batch_ - 1)],
            broadCast, padLengthAlign_);

        DataCopy(xRemovePaddingGm[batch_ * padLength_ - cumOffsetsBuffer.GetValue(batch_ - 1)],
            xRemovePaddingLocal, padLengthAlign_);
    }
private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM>  inputIdsQueue_, cumOffsetsQueue_, tokenNumQueue_, seqLenQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> cumOffsetOutQueue_, paddingOffsetQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> broadCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cumOffsetsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> seqLenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xRemovePaddingQueue_;

    GlobalTensor<int32_t> cumOffsetsNowGm, seqLenGm, cumOffsetOutGm, paddingOffsetGm;
    GlobalTensor<int64_t> inputIdsGm, tokenNumGm, xRemovePaddingGm;
    uint32_t padLength_{1};
    uint64_t batch_{1};
    uint32_t padLengthAlign_{16};
    uint32_t batchAlign_{8};
    int64_t seqLenZero_{0};
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AtbOps::UnpadTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->padLength = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->batch = (*(const __gm__ uint32_t *)(p_tilingdata + 4));

#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AtbOps::UnpadTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->padLength = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->batch = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    AtbOps::UnpadTilingData tiling_data;         \
    InitTilingData(tiling_arg, &(tiling_data))

extern "C" __global__ __aicore__ void unpad(GM_ADDR input_ids,
    GM_ADDR cum_offsets_now, GM_ADDR token_num, GM_ADDR seq_len, GM_ADDR x_remove_padding,
    GM_ADDR cum_offsets_out, GM_ADDR padding_offset, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelUnpad op;
    op.Init(input_ids, cum_offsets_now, token_num, seq_len,
            x_remove_padding, cum_offsets_out, padding_offset, tilingData.padLength, tilingData.batch);
    op.Process();
}