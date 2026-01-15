/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "mixkernels/utils/common/kernel/kernel_utils.h"

#include "kernels/logprobs_sample/tiling/tiling_data.h"

static constexpr int32_t BUFFER_NUM = 1;               // 不使用double buffer，仅使用queue接口来简化同步逻辑
static constexpr float DEFAULT_LOG_VALUE = -9999.0f;
static constexpr int32_t INPUT_PAD_SIZE = 16;          // 输入sortedProbs为half或者bf16
static constexpr int32_t OUTPUT_PAD_SIZE = 8;          // 输出logprobs为float32
static constexpr int32_t SYNC_OUT_MIN_SIZE = 32;
static constexpr int32_t SYNC_OUT_EVENT_ID = 0;

template<typename T>
class LogprobsSample {
public:
    __aicore__ inline LogprobsSample() {}
    __aicore__ inline void Init(GM_ADDR sortedProbs, GM_ADDR cumsumedProbs, GM_ADDR selectRange, GM_ADDR logprobsSize,
                                GM_ADDR syncWorkspace, GM_ADDR outputLogprobs,
                                const AsdOps::LogprobsSampleTilingData &tilingData)
    {
        coreNums_ = AscendC::GetBlockNum();
        isMultiCore_ = coreNums_ > 1;
        currentCoreId_ = AscendC::GetBlockIdx();
        currentSelectRange_ = 0;
        currentSelectProb_ = 0.0f;
        batchSize_ = tilingData.batchSize;
        logprobsSize_ = *((__gm__ int32_t*)logprobsSize);
        calculateSizePad_ = (logprobsSize_ + INPUT_PAD_SIZE - 1) / INPUT_PAD_SIZE * INPUT_PAD_SIZE;
        outputSizePad_ = (logprobsSize_ + OUTPUT_PAD_SIZE - 1) / OUTPUT_PAD_SIZE * OUTPUT_PAD_SIZE;
        isOutputAligned_ = outputSizePad_ == logprobsSize_;
        collectOutputLocally_ = logprobsSize_ < OUTPUT_PAD_SIZE;
        probsSize_ = tilingData.probsSize;
        notLastCoreRun_ = (batchSize_ + coreNums_ - 1) / coreNums_;
        lastCoreRun_ = batchSize_ - (coreNums_ - 1) * notLastCoreRun_;
        coreRun_ = (currentCoreId_ == coreNums_ - 1) ? lastCoreRun_ : notLastCoreRun_;

        cumsumedProbsGm_.SetGlobalBuffer((__gm__ T *)cumsumedProbs);
        selectRangeGm_.SetGlobalBuffer((__gm__ int32_t *)selectRange);

        pipe_.InitBuffer(calcTmpBuf_, calculateSizePad_ * sizeof(float));
        sortedProbsGm_.SetGlobalBuffer((__gm__ T *)sortedProbs);
        logprobsOutputGm_.SetGlobalBuffer((__gm__ float *)outputLogprobs);

        pipe_.InitBuffer(inQueueSortedProbs_, BUFFER_NUM,  calculateSizePad_ * sizeof(T));
        pipe_.InitBuffer(outQueueLogprobs_, BUFFER_NUM, outputSizePad_ * sizeof(float));

#if (__CCE_AICORE__ != 220)
        if (isMultiCore_) {
            if (collectOutputLocally_) {
                pipe_.InitBuffer(outputLocalBuf_, coreRun_ * OUTPUT_PAD_SIZE * sizeof(float));
                syncGm_.SetGlobalBuffer((__gm__ int32_t *)syncWorkspace, coreNums_ * SYNC_OUT_MIN_SIZE);
                pipe_.InitBuffer(syncOutQueue_, BUFFER_NUM, SYNC_OUT_MIN_SIZE);
            } else {
                pipe_.InitBuffer(gatherMaskPattenBuf_, sizeof(uint32_t));
                pipe_.InitBuffer(outQueueLogprobsTail_, BUFFER_NUM, OUTPUT_PAD_SIZE * sizeof(float));
            }
        }
#endif
    }
    __aicore__ inline void Process()
    {
        if (logprobsSize_ == 0) {
            return;
        }
        for (int32_t relativeBatchId = 0; relativeBatchId < coreRun_; ++relativeBatchId) {
            int32_t absoluteBatchId = currentCoreId_ * notLastCoreRun_ + relativeBatchId;
            InitSelectRangeAndProb(absoluteBatchId);
            CopyIn(absoluteBatchId);
            Compute();
            CopyOut(relativeBatchId, absoluteBatchId);
        }
#if (__CCE_AICORE__ != 220)
        if (isMultiCore_ && collectOutputLocally_) {
            SyncOut();
        }
#endif
    }

private:
    __aicore__ inline void InitSelectRangeAndProb(int32_t absoluteBatchId)
    {
        currentSelectRange_ = selectRangeGm_.GetValue(absoluteBatchId);
        // 当选择范围为0，为异常情况。根本原因是ToppSample应当选择the minimum that >= topp，而不是the maximum that <= topp
        // 这里强制设置为1，意为选择范围只有第一个值
        if (currentSelectRange_ == 0)
            currentSelectRange_ = 1;
        int64_t selectProbOffset = currentSelectRange_ - 1;
        selectProbOffset += probsSize_ * absoluteBatchId;
#if (__CCE_AICORE__ == 220)
        if constexpr(AscendC::IsSameType<T, bfloat16_t>::value) {
            currentSelectProb_ = AscendC::ToFloat(cumsumedProbsGm_.GetValue(selectProbOffset));
        } else {
            currentSelectProb_ = (float)cumsumedProbsGm_.GetValue(selectProbOffset);
        }
#else
        currentSelectProb_ = (float)cumsumedProbsGm_.GetValue(selectProbOffset);
#endif
    }

    __aicore__ inline void CopyIn(int32_t absoluteBatchId)
    {
        AscendC::LocalTensor<T> sortedProbsLocal = inQueueSortedProbs_.AllocTensor<T>();
        int32_t blockOffset = absoluteBatchId * probsSize_;
        AscendC::DataCopy(sortedProbsLocal, sortedProbsGm_[blockOffset], calculateSizePad_);
        inQueueSortedProbs_.EnQue(sortedProbsLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> sortedProbsLocal = inQueueSortedProbs_.DeQue<T>();
        AscendC::LocalTensor<float> logprobsLocal = outQueueLogprobs_.AllocTensor<float>();
        if (logprobsSize_ <= currentSelectRange_) {
            CalculateLogOutput(sortedProbsLocal, logprobsLocal, outputSizePad_);
        } else {
            uint32_t fullBlockSize = (currentSelectRange_ / INPUT_PAD_SIZE) * INPUT_PAD_SIZE;
            uint32_t tailSize = currentSelectRange_ - fullBlockSize;
            if (fullBlockSize > 0) {
                CalculateLogOutput(sortedProbsLocal, logprobsLocal, fullBlockSize);
            }
            if (fullBlockSize < logprobsSize_) {
                AscendC::Duplicate(logprobsLocal[fullBlockSize], DEFAULT_LOG_VALUE, logprobsSize_ - fullBlockSize);
                AscendC::PipeBarrier<PIPE_V>();
            }
            if (tailSize > 0) {
                CalculateLogOutput(sortedProbsLocal[fullBlockSize], logprobsLocal[fullBlockSize], tailSize);
            }
        }

#if (__CCE_AICORE__ != 220)
        if (!isOutputAligned_ && isMultiCore_ && !collectOutputLocally_) {
            AscendC::LocalTensor<uint32_t> gatherMaskPattern = gatherMaskPattenBuf_.Get<uint32_t>();
            /* 比如logprobsSize为20，对应的索引为[0, 19]
             * 则在拷出时，需要先把[0, 15]拷出，再把[16, 19]拷出
             * 这里通过在[8, 19]的输入里，GatherMask [12, 19]的位置拷贝到tailLocal，对应的pattern为
             * 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23
             * 0  0  0  0  1  1  1  1  1  1  1  1  0  0  0  0
             * 注意8为低地址，[12, 15]拷贝了两次
             * */
            int32_t remainCount = logprobsSize_ + OUTPUT_PAD_SIZE - outputSizePad_;
            uint32_t pattern = ((1u << OUTPUT_PAD_SIZE) - 1) << remainCount;
            gatherMaskPattern.SetValue(0, pattern);

            AscendC::LocalTensor<float> tailLocal = outQueueLogprobsTail_.AllocTensor<float>();
            uint32_t mask = OUTPUT_PAD_SIZE * 2;
            uint32_t mainOffset = outputSizePad_ - mask;
            uint64_t rsvdCnt = 0;
            AscendC::GatherMask(tailLocal, logprobsLocal[mainOffset], gatherMaskPattern, true, mask, {1, 1, 8, 8},
                                rsvdCnt);
            outQueueLogprobsTail_.EnQue(tailLocal);
        }
#endif
        outQueueLogprobs_.EnQue(logprobsLocal);
        inQueueSortedProbs_.FreeTensor(sortedProbsLocal);
    }

    __aicore__ inline void CalculateLogOutput(const AscendC::LocalTensor<T> &srcTensor,
                                              const AscendC::LocalTensor<float> &dstTensor,
                                              int calcCount)
    {
        AscendC::LocalTensor<float> dividendBuf = calcTmpBuf_.Get<float>();
        // topp select prob不可能为0，因为cumsumed prob是逆序概率求部分和算出来的
        AscendC::Duplicate(dividendBuf, currentSelectProb_, calcCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(dstTensor, srcTensor, AscendC::RoundMode::CAST_NONE, calcCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(dstTensor, dstTensor, dividendBuf, calcCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Log(dstTensor, dstTensor, calcCount);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOut(int32_t relativeBatchId, int32_t absoluteBatchId)
    {
        AscendC::LocalTensor<float> logprobsLocal = outQueueLogprobs_.DeQue<float>();
        uint32_t offsetMain = absoluteBatchId * logprobsSize_;
        if (isOutputAligned_ || !isMultiCore_) {
            // 对齐场景，全量拷贝；单核场景，正常覆盖
            AscendC::DataCopy(logprobsOutputGm_[offsetMain], logprobsLocal, outputSizePad_);
        } else {
#if (__CCE_AICORE__ == 220)
            AscendC::DataCopyExtParams copyParams{1, logprobsSize_ * (uint32_t)sizeof(float), 0, 0, 0};
            AscendC::DataCopyPad(logprobsOutputGm_[offsetMain], logprobsLocal, copyParams);
#else
            // 非对齐场景
            if (collectOutputLocally_) {
                AscendC::LocalTensor<float> outputLocal = outputLocalBuf_.Get<float>();
                AscendC::DataCopy(outputLocal[relativeBatchId * OUTPUT_PAD_SIZE], logprobsLocal, OUTPUT_PAD_SIZE);
            } else {
                uint32_t mainBlockSize = logprobsSize_ / OUTPUT_PAD_SIZE * OUTPUT_PAD_SIZE;
                AscendC::DataCopy(logprobsOutputGm_[offsetMain], logprobsLocal, mainBlockSize);
                AscendC::PipeBarrier<PIPE_MTE3>();
                uint32_t offsetTail = offsetMain + logprobsSize_ - OUTPUT_PAD_SIZE;
                AscendC::LocalTensor<float> tailLocal = outQueueLogprobsTail_.DeQue<float>();
                AscendC::DataCopy(logprobsOutputGm_[offsetTail], tailLocal, OUTPUT_PAD_SIZE);
                outQueueLogprobsTail_.FreeTensor(tailLocal);
            }
#endif
        }
        outQueueLogprobs_.FreeTensor(logprobsLocal);
    }

    __aicore__ inline void SyncOut()
    {
        if (currentCoreId_ == 0) {
            auto syncBuf = syncOutQueue_.AllocTensor<int32_t>();
            CopyLocalBufOut();
            AscendC::IBSet(syncGm_, syncBuf, 0, SYNC_OUT_EVENT_ID);
            syncOutQueue_.FreeTensor(syncBuf);
        } else if (currentCoreId_ == coreNums_ - 1) {
            auto syncBuf = syncOutQueue_.AllocTensor<int32_t>();
            AscendC::IBWait(syncGm_, syncBuf, currentCoreId_ - 1, SYNC_OUT_EVENT_ID);
            CopyLocalBufOut();
            syncOutQueue_.FreeTensor(syncBuf);
        } else {
            auto syncBuf = syncOutQueue_.AllocTensor<int32_t>();
            AscendC::IBWait(syncGm_, syncBuf, currentCoreId_ - 1, SYNC_OUT_EVENT_ID);
            CopyLocalBufOut();
            AscendC::IBSet(syncGm_, syncBuf, currentCoreId_, SYNC_OUT_EVENT_ID);
            syncOutQueue_.FreeTensor(syncBuf);
        }
    }

    __aicore__ inline void CopyLocalBufOut()
    {
        AscendC::LocalTensor<float> localBuf = outputLocalBuf_.Get<float>();
        for (int32_t relativeBatchId = 0; relativeBatchId < coreRun_; ++relativeBatchId) {
            int32_t absoluteBatchId = currentCoreId_ * notLastCoreRun_ + relativeBatchId;
            uint32_t offsetMain = absoluteBatchId * logprobsSize_;
            // GM里的数据位置有重叠，需要同步
            AscendC::PipeBarrier<PIPE_MTE3>();
            AscendC::DataCopy(logprobsOutputGm_[offsetMain], localBuf[relativeBatchId * OUTPUT_PAD_SIZE],
                              OUTPUT_PAD_SIZE);
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueSortedProbs_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueLogprobs_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueLogprobsTail_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> syncOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherMaskPattenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputLocalBuf_;
    AscendC::GlobalTensor<T> sortedProbsGm_;
    AscendC::GlobalTensor<T> cumsumedProbsGm_;
    AscendC::GlobalTensor<int32_t> selectRangeGm_;
    AscendC::GlobalTensor<float> logprobsOutputGm_;
    AscendC::GlobalTensor<int32_t> syncGm_;
    uint32_t coreNums_;
    uint32_t currentCoreId_;
    int32_t currentSelectRange_;
    float currentSelectProb_;
    uint32_t batchSize_;
    uint32_t logprobsSize_;
    uint32_t calculateSizePad_;
    uint32_t outputSizePad_;
    bool isOutputAligned_;
    uint32_t probsSize_;
    uint32_t notLastCoreRun_;
    uint32_t lastCoreRun_;
    uint32_t coreRun_;
    bool collectOutputLocally_ = false;
    bool isMultiCore_;
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::LogprobsSampleTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->batchSize = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->probsSize = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
#else
    AscendC::TPipe pipe_;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::LogprobsSampleTilingData), &pipe_);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->batchSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->probsSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}


/**
  * @brief  计算logprobs的kernel入口函数
  * @param  sortedProbs:    逆序排序概率的，shape为(batch, voc_size)
  * @param  cumsumedProbs:  逆序排序概率l累积和，shape为(batch, vec_size)
  * @param  selectRange:    topp筛选出的范围，shape为(batch, 1)，每个值的值域为[0, voc_size]
  * @param  logprobsSize:   logprobs输出第二维的大小，shape为(1)
  * @param  outputLogprobs: logprobs的计算结果，shape为(batch, logprobsSize)
  * @param  syncWorkspace:  用于拷出时的核间同步
  * @param  tiling:         logprobs tiling数据
  * @retval None
 */
extern "C" __global__ __aicore__ void logprobs_sample(GM_ADDR sortedProbs, GM_ADDR cumsumedProbs, GM_ADDR selectRange,
                                                      GM_ADDR logprobsSize, GM_ADDR outputLogprobs,
                                                      GM_ADDR syncWorkspace, GM_ADDR tiling)
{
    AsdOps::LogprobsSampleTilingData tiling_data;
    InitTilingData(tiling, &tiling_data);
    if (TILING_KEY_IS(0)) {
        LogprobsSample<half> op;
        op.Init(sortedProbs, cumsumedProbs, selectRange, logprobsSize, syncWorkspace, outputLogprobs, tiling_data);
        op.Process();
    }
#if (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(1)) {
        LogprobsSample<bfloat16_t> op;
        op.Init(sortedProbs, cumsumedProbs, selectRange, logprobsSize, syncWorkspace, outputLogprobs, tiling_data);
        op.Process();
    }
#endif
}
