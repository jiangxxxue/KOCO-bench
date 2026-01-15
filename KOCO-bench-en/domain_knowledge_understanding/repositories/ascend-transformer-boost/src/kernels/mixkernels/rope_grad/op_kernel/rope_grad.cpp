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
#include "mixkernels/rope_grad/tiling/tiling_data.h"
using namespace AscendC;

namespace {
    constexpr uint32_t MAX_PROCESS_NUM = 192 * 1024 / sizeof(half) / 8; // max process num of one loop
    constexpr int32_t BUFFER_NUM = 1;                                   // tensor num for each queue
} // namespace

class RopeGrad {
public:
    __aicore__ inline RopeGrad(int64_t maxSeqLen, int64_t hiddenSize, int64_t headSize, int64_t sumSeqLen,
                                    int64_t batch)
    {
        this->maxSeqLen = maxSeqLen;
        this->hiddenSize = hiddenSize;
        this->headSize = headSize;
        this->sumSeqLen = sumSeqLen;
        this->batch = batch;
    }

    __aicore__ inline void Init(GM_ADDR qembedGrad, GM_ADDR kembedGrad, GM_ADDR cos, GM_ADDR sin,
                                GM_ADDR qGrad, GM_ADDR kGrad)
    {
        // get start index for current core, core parallel
        qembedgradGm.SetGlobalBuffer((__gm__ half*)qembedGrad + GetBlockIdx() * this->headSize,
            this->sumSeqLen * this->headSize);
        kembedgradGm.SetGlobalBuffer((__gm__ half*)kembedGrad + GetBlockIdx() * this->headSize,
            this->sumSeqLen * this->headSize);
        cosGm.SetGlobalBuffer((__gm__ half*)cos, this->maxSeqLen * this->headSize);
        sinGm.SetGlobalBuffer((__gm__ half*)sin, this->maxSeqLen * this->headSize);
        qgradGm.SetGlobalBuffer((__gm__ half*)qGrad + GetBlockIdx() * this->headSize,
            this->sumSeqLen * this->headSize);
        kgradGm.SetGlobalBuffer((__gm__ half*)kGrad + GetBlockIdx() * this->headSize,
            this->sumSeqLen * this->headSize);
        this->rowsPerLoop = MAX_PROCESS_NUM / this->headSize;
        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(qembedgradQueue, BUFFER_NUM, MAX_PROCESS_NUM * sizeof(half));
        pipe.InitBuffer(kembedgradQueue, BUFFER_NUM, MAX_PROCESS_NUM * sizeof(half));
        pipe.InitBuffer(cosQueue, BUFFER_NUM, MAX_PROCESS_NUM * sizeof(half));
        pipe.InitBuffer(sinQueue, BUFFER_NUM, MAX_PROCESS_NUM * sizeof(half));
        pipe.InitBuffer(qgradQueue, BUFFER_NUM, MAX_PROCESS_NUM * sizeof(half));
        pipe.InitBuffer(kgradQueue, BUFFER_NUM, MAX_PROCESS_NUM * sizeof(half));
        pipe.InitBuffer(workbuffer, MAX_PROCESS_NUM * sizeof(half));
    } // 初始化函数，完成内存初始化相关操作

    __aicore__ inline void Process(int32_t curnum, int32_t seqleni)
    {
        uint64_t rowsRepeat = seqleni / this->rowsPerLoop;
        uint64_t rowsRemain = seqleni % this->rowsPerLoop;
        for (uint64_t j = 0; j < rowsRepeat; j++) {
            CopyIn(curnum, j, this->rowsPerLoop);
            Compute(this->rowsPerLoop);
            CopyOut(curnum, j, this->rowsPerLoop);
        }
        if (rowsRemain > 0) {
            CopyIn(curnum, rowsRepeat, rowsRemain);
            Compute(rowsRemain);
            CopyOut(curnum, rowsRepeat, rowsRemain);
        }
        this->cursumseqlen += seqleni;
        pipe_barrier(PIPE_ALL);
    } // 核心处理函数，实现算子逻辑，调用私有成员函数CopyIn、Compute、CopyOut完成矢量算子的三级流水操作
private:
    __aicore__ inline void CopyIn(int32_t b, uint64_t progress, int64_t currentloopRows)
    {
        // alloc tensor from queue memory
        LocalTensor<half> qembedgradLocal = qembedgradQueue.AllocTensor<half>();
        LocalTensor<half> kembedgradLocal = kembedgradQueue.AllocTensor<half>();
        LocalTensor<half> cosLocal = cosQueue.AllocTensor<half>();
        LocalTensor<half> sinLocal = sinQueue.AllocTensor<half>();
        // copy progress_th tile from global tensor to local tensor
        DataCopy(qembedgradLocal, qembedgradGm[(cursumseqlen + progress * this->rowsPerLoop) * this->hiddenSize],
            {static_cast<uint16_t>(currentloopRows), static_cast<uint16_t>(this->headSize / 16),
             static_cast<uint16_t>((this->hiddenSize - this->headSize) / 16), 0});
        DataCopy(kembedgradLocal, kembedgradGm[(cursumseqlen + progress * this->rowsPerLoop) * this->hiddenSize],
            {static_cast<uint16_t>(currentloopRows), static_cast<uint16_t>(this->headSize / 16),
             static_cast<uint16_t>((this->hiddenSize - this->headSize) / 16), 0});
        DataCopy(cosLocal, cosGm[progress * this->rowsPerLoop * this->headSize], currentloopRows * this->headSize);
        DataCopy(sinLocal, sinGm[progress * this->rowsPerLoop * this->headSize], currentloopRows * this->headSize);
        // enque input tensors to VECIN queue
        qembedgradQueue.EnQue(qembedgradLocal);
        kembedgradQueue.EnQue(kembedgradLocal);
        cosQueue.EnQue(cosLocal);
        sinQueue.EnQue(sinLocal);
    }
    __aicore__ inline void Compute(uint64_t currentloopRows)
    {
        // deque input tensors from VECIN queue
        LocalTensor<half> qembedgradLocal = qembedgradQueue.DeQue<half>();
        LocalTensor<half> kembedgradLocal = kembedgradQueue.DeQue<half>();
        LocalTensor<half> cosLocal = cosQueue.DeQue<half>();
        LocalTensor<half> sinLocal = sinQueue.DeQue<half>();
        LocalTensor<half> qgradLocal = qgradQueue.AllocTensor<half>();
        LocalTensor<half> kgradLocal = kgradQueue.AllocTensor<half>();
        LocalTensor<half> workLocal = workbuffer.Get<half>();
        half scalars = -1;
        uint64_t mask = this->headSize / 2;
        DataCopy(workLocal, sinLocal[mask],
            {static_cast<uint16_t>(currentloopRows), static_cast<uint16_t>(this->headSize / 32),
             static_cast<uint16_t>(this->headSize / 32), static_cast<uint16_t>(this->headSize / 32)});
        pipe_barrier(PIPE_V);
        Muls(workLocal[mask], sinLocal, scalars, mask, currentloopRows, {1, 1, 8, 8});
        pipe_barrier(PIPE_V);
        Add(workLocal, cosLocal, workLocal, currentloopRows * this->headSize);
        pipe_barrier(PIPE_V);
        Mul(qgradLocal, qembedgradLocal, workLocal, currentloopRows * this->headSize);
        pipe_barrier(PIPE_V);
        Mul(kgradLocal, kembedgradLocal, workLocal, currentloopRows * this->headSize);
        pipe_barrier(PIPE_V);
        // enque the output tensor to VECOUT queue
        qgradQueue.EnQue<half>(qgradLocal);
        kgradQueue.EnQue<half>(kgradLocal);
        // free input tensors for reuse
        qembedgradQueue.FreeTensor(qembedgradLocal);
        kembedgradQueue.FreeTensor(kembedgradLocal);
        cosQueue.FreeTensor(cosLocal);
        sinQueue.FreeTensor(sinLocal);
    }
    __aicore__ inline void CopyOut(int32_t b, int64_t progress, int64_t currentloopRows)
    {
        // deque output tensor from VECOUT queue
        LocalTensor<half> qgradLocal = qgradQueue.DeQue<half>();
        LocalTensor<half> kgradLocal = kgradQueue.DeQue<half>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(qgradGm[(cursumseqlen + progress * this->rowsPerLoop) * this->hiddenSize], qgradLocal,
            {static_cast<uint16_t>(currentloopRows), static_cast<uint16_t>(this->headSize / 16), 0,
             static_cast<uint16_t>((this->hiddenSize - this->headSize) / 16)});
        DataCopy(kgradGm[(cursumseqlen + progress * this->rowsPerLoop) * this->hiddenSize], kgradLocal,
            {static_cast<uint16_t>(currentloopRows), static_cast<uint16_t>(this->headSize / 16), 0,
             static_cast<uint16_t>((this->hiddenSize - this->headSize) / 16)});
        // free output tensor for reuse
        qgradQueue.FreeTensor(qgradLocal);
        kgradQueue.FreeTensor(kgradLocal);
    }
private:
    int64_t maxSeqLen = 0;
    int64_t hiddenSize = 0;
    int64_t headSize = 0;
    int64_t sumSeqLen = 0;
    int64_t batch = 0;
    int64_t rowsPerLoop = 0;
    uint64_t cursumseqlen = 0;

    TPipe pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, 1> qembedgradQueue, kembedgradQueue, cosQueue, sinQueue;
    // create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, 1> qgradQueue, kgradQueue;
    TBuf<> workbuffer;
    GlobalTensor<half> qembedgradGm, kembedgradGm, cosGm, sinGm, qgradGm, kgradGm;
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AtbOps::RopeGradTilingData *tilingData)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingData->maxSeqLen = (*(const __gm__ int64_t *)(p_tilingdata + 0));
    tilingData->hiddenSize = (*(const __gm__ int64_t *)(p_tilingdata + 8));
    tilingData->headSize = (*(const __gm__ int64_t *)(p_tilingdata + 16));
    tilingData->sumSeqLen = (*(const __gm__ int64_t *)(p_tilingdata + 24));
    tilingData->batch = (*(const __gm__ int64_t *)(p_tilingdata + 32));
    pipe_barrier(PIPE_ALL);
#else
    __ubuf__ uint8_t *tilingdata_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    copy_gm_to_ubuf(((__ubuf__ uint8_t *)tilingdata_in_ub), p_tilingdata, 0, 1, 42, 0, 0);
    pipe_barrier(PIPE_ALL);
    tilingData->maxSeqLen = (*(__ubuf__ int64_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 0));
    tilingData->hiddenSize = (*(__ubuf__ int64_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 8));
    tilingData->headSize = (*(__ubuf__ int64_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 16));
    tilingData->sumSeqLen = (*(__ubuf__ int64_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 24));
    tilingData->batch = (*(__ubuf__ int64_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 32));
    pipe_barrier(PIPE_ALL);
#endif
}

inline __aicore__ AtbOps::RopeGradSampleTilingData GetSampleTilingData(const __gm__ uint8_t *p_tilingdata)
{
    AtbOps::RopeGradSampleTilingData sampleTilingData;
    sampleTilingData.qSeqLen = (*(const __gm__ int32_t *)p_tilingdata);
    return sampleTilingData;
}

// implementation of kernel function
extern "C" __global__ __aicore__ void rope_grad(GM_ADDR qembedGrad, GM_ADDR kembedGrad,
                                                    GM_ADDR cos, GM_ADDR sin, GM_ADDR qGrad,
                                                    GM_ADDR kGrad, GM_ADDR tiling)
{
    AtbOps::RopeGradTilingData tilingData;
    InitTilingData(tiling, &tilingData);
    RopeGrad op(tilingData.maxSeqLen, tilingData.hiddenSize, tilingData.headSize,
                        tilingData.sumSeqLen, tilingData.batch);
    op.Init(qembedGrad, kembedGrad, cos, sin, qGrad, kGrad);
    tiling += sizeof(AtbOps::RopeGradTilingData);
    for (uint32_t i = 0; i < tilingData.batch; i++) {
        AtbOps::RopeGradSampleTilingData sampleTilingData = GetSampleTilingData(tiling);
        op.Process(i, sampleTilingData.qSeqLen);
        tiling += sizeof(AtbOps::RopeGradSampleTilingData);
    }
}
