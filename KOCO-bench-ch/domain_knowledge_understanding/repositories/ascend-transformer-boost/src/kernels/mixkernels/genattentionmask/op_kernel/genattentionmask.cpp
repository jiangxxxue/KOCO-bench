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
#include "mixkernels/genattentionmask/tiling/tiling_data.h"

using namespace AscendC;
namespace {
    constexpr uint32_t MAX_PROCESS_NUM = 192 * 1024 / sizeof(half) / 4;
    constexpr int32_t BUFFER_NUM = 1;                                     // tensor num for each queue
} // namespace

class GenAttentionMask {
public:
    __aicore__ inline GenAttentionMask(uint32_t batch, uint32_t maxSeqLen,
    uint32_t blockNum, uint32_t headNum, int32_t *qSeqLen)
    {
        this->batch = batch;
        this->maxSeqLen = maxSeqLen;
        this->blockNum = blockNum;
        this->headNum = headNum;

        for (int i = 0; i < batch; i++) {
            this->qSeqLen[i] = *(qSeqLen + i);
        }
    }

    __aicore__ inline void Init(GM_ADDR attentionMask, GM_ADDR attentionMask2)
    {
        pipe.InitBuffer(attentionMaskUb, MAX_PROCESS_NUM * sizeof(half));

        this->currentSeqLen = this->qSeqLen[GetBlockIdx()];
        int64_t offset = 0;
        for (int i = 0; i < GetBlockIdx(); i++) {
            offset += static_cast<int64_t>(this->qSeqLen[i]) * this->qSeqLen[i] * this->headNum;
        }

        // get start index for current core, core parallel
        attentionMaskGm.SetGlobalBuffer((__gm__ half*)attentionMask +
        this->maxSeqLen * this->maxSeqLen * GetBlockIdx(), this->currentSeqLen * this->maxSeqLen);
        attentionMask2Gm.SetGlobalBuffer((__gm__ half*)attentionMask2 + offset,
        this->currentSeqLen * this->currentSeqLen * this->headNum);

        // pipe alloc memory to queue, the unit is Bytes
        int currentSeqLenRound = (this->currentSeqLen * sizeof(half) + 31) / 32 * 32 / sizeof(half);

        this->rowsPerLoop = MAX_PROCESS_NUM / currentSeqLenRound;
        int rowsRepeat = this->currentSeqLen / rowsPerLoop;
        int rowsRemain = this->currentSeqLen % rowsPerLoop;

        for (int i = 0; i < rowsRepeat; i++) {
            Process(i, this->rowsPerLoop);
        }
        if (rowsRemain > 0) {
            Process(rowsRepeat, rowsRemain);
        }
    }

private:
    __aicore__ inline void Process(int32_t progress, int rows)
    {
        int currentSeqLenRound = (this->currentSeqLen * sizeof(half) + 31) / 32 * 32 / sizeof(half);
        // alloc tensor from queue memory
        LocalTensor<half> attentionMaskUbPreUb = attentionMaskUb.Get<half>(MAX_PROCESS_NUM * sizeof(half));

        // copy progress_th tile from global tensor to local tensor
        pipe_barrier(PIPE_ALL);
        copy_gm_to_ubuf_align_b16(
            (__ubuf__ half *)attentionMaskUbPreUb.GetPhyAddr(),
            (__gm__ half *)attentionMaskGm.GetPhyAddr() + progress * this->rowsPerLoop * this->maxSeqLen,
            0,
            rows,
            this->currentSeqLen * sizeof(half),
            0,
            0,
            (this->maxSeqLen - this->currentSeqLen) * sizeof(half),
            0);
        pipe_barrier(PIPE_ALL);

        for (int i = 0; i < this->headNum; i++) {
            copy_ubuf_to_gm_align_b16(
                (__gm__ half *)attentionMask2Gm.GetPhyAddr() + i * this->currentSeqLen * this->currentSeqLen +
                progress * this->rowsPerLoop * this->currentSeqLen,
                (__ubuf__ half *)attentionMaskUbPreUb.GetPhyAddr(),
                0,
                rows,
                this->currentSeqLen * sizeof(half),
                0,
                0,
                0,
                0);
        }
        pipe_barrier(PIPE_ALL);
    }
private:
    int32_t batch = 0;
    int32_t maxSeqLen = 0;
    int32_t blockNum = 0;
    int32_t headNum = 0;
    int32_t qSeqLen[AtbOps::GEN_ATTENTION_MASK_MAX_BATCH];
    int32_t currentSeqLen = 0;
    int32_t rowsPerLoop = 0;
    TPipe pipe;
    TBuf<> attentionMaskUb;
    GlobalTensor<half> attentionMaskGm, attentionMask2Gm;
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata,
AtbOps::GenAttentionMaskTilingData *tilingData)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingData->batch = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingData->maxSeqLen = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingData->blockNum = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingData->headNum = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    for (int i = 0; i < tilingData->batch; i++) {
        tilingData->qSeqLen[i] = (*(const __gm__ uint32_t *)(p_tilingdata + 16 + 4 * i));
    }
#else
    __ubuf__ uint8_t *tilingdata_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    copy_gm_to_ubuf(((__ubuf__ uint8_t *)tilingdata_in_ub), p_tilingdata, 0, 1, 2, 0, 0);
    pipe_barrier(PIPE_ALL);
    tilingData->batch = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 0));
    tilingData->maxSeqLen = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 4));
    tilingData->blockNum = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 8));
    tilingData->headNum = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 12));
    for (int i = 0; i < tilingData->batch; i++) {
        tilingData->qSeqLen[i] = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 16 + 4 * i);
    }
    pipe_barrier(PIPE_ALL);
#endif
}

// implementation of kernel function
extern "C" __global__ __aicore__ void genattentionmask(GM_ADDR attentionMask, GM_ADDR attentionMask2, GM_ADDR tiling)
{
    AtbOps::GenAttentionMaskTilingData tiling_data;
    InitTilingData(tiling, &tiling_data);
    GenAttentionMask op(tiling_data.batch, tiling_data.maxSeqLen,
        tiling_data.blockNum, tiling_data.headNum, tiling_data.qSeqLen);
    op.Init(attentionMask, attentionMask2);
}