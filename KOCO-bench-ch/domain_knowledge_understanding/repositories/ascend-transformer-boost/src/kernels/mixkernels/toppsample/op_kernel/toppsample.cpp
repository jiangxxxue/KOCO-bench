/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "mixkernels/toppsample/tiling/tiling_data.h"
#include "mixkernels/utils/common/kernel/kernel_utils.h"
namespace {
static constexpr int32_t BUFFER_NUM = 1;
static constexpr uint32_t DATA_BYTE = 2;
static constexpr uint32_t MAX_CORE_NUM = 512;
static constexpr uint32_t BLK_SIZE = 32;
static constexpr uint32_t DEFAULT_STRIDE = 8;
static constexpr uint32_t FP32_PER_REPEAT = 64;
static constexpr uint32_t FP16_PER_BLOCK = 16;
static constexpr uint32_t NUM_4 = 4;

using AscendC::HardEvent;

template <bool MULTI, typename T> class KernelToppsample {
public:
    __aicore__ inline KernelToppsample() {}
    __aicore__ inline void Init(GM_ADDR cumsumed_probs, GM_ADDR topp, GM_ADDR select_index, GM_ADDR select_range,
                                AtbOps::ToppsampleTilingData &tiling_data)
    {
        realLastDim_ = tiling_data.realLastDim;
        firstDim_ = tiling_data.firstDim;
        expandLastDim_ = tiling_data.expandLastDim;
        tilingUb_ = tiling_data.randValList;
        numSamplesMax_ = tiling_data.numSamplesMax;
        perCoreRunNum_ = tiling_data.perCoreRunNum;
        nlElePerCorePerRun_ = tiling_data.nlElePerCorePerRun;
        lElePerCoreLastRun_ = tiling_data.lElePerCoreLastRun;
        tempUbEleAligened_ = tiling_data.tempUbEleAligened;
        realCore_ = AscendC::GetBlockNum();
        blockIdx_ = AscendC::GetBlockIdx();
        nlCoreRun_ = (firstDim_ + realCore_ - 1) / realCore_;
        lCoreRun_ = firstDim_ - (realCore_ - 1) * nlCoreRun_;
        dynamicRound_ = (blockIdx_ == realCore_ - 1) ? lCoreRun_ : nlCoreRun_;
        maxBatch_ = (firstDim_ + FP16_PER_BLOCK - 1) / FP16_PER_BLOCK * FP16_PER_BLOCK;

        xGm_.SetGlobalBuffer((__gm__ T *)cumsumed_probs);
        yGm_.SetGlobalBuffer((__gm__ T *)topp); // batch,num_samples
        zGm_.SetGlobalBuffer((__gm__ int32_t *)select_index);
        selectRangeGm_.SetGlobalBuffer((__gm__ int32_t *)select_range);

        pipe_.InitBuffer(inputBuf_, tempUbEleAligened_ * DATA_BYTE);
        pipe_.InitBuffer(tempBuf_, tempUbEleAligened_ * DATA_BYTE * DATA_BYTE);
        pipe_.InitBuffer(fp32Buf_, tempUbEleAligened_ * DATA_BYTE * DATA_BYTE);
        pipe_.InitBuffer(yBuf_, maxBatch_ * DATA_BYTE);                    // topp
        pipe_.InitBuffer(yF32Buf_, maxBatch_ * DATA_BYTE * DATA_BYTE);     // toppfp32
        pipe_.InitBuffer(int8Buf_, tempUbEleAligened_ / DEFAULT_STRIDE);   // compare
        pipe_.InitBuffer(blockBuf_, BLK_SIZE);                             // 存下标
        pipe_.InitBuffer(int32Buf_, MAX_CORE_NUM * DATA_BYTE * DATA_BYTE); // 每个核做几个batch
        pipe_.InitBuffer(vecIn_, 1, NUM_4 * MAX_CORE_NUM * sizeof(int32_t));   // 核间同步专享
        pipe_.InitBuffer(selectRangeBlkBuf_, MAX_CORE_NUM * DATA_BYTE * DATA_BYTE);
    }
    __aicore__ inline void PickUpRand()
    {
        AscendC::LocalTensor<T> buf = yBuf_.Get<T>();
        DataCopy(buf, yGm_, maxBatch_);
    }

    __aicore__ inline void FirstPick(uint32_t cid, uint32_t offset)
    {
        // 寻找第一个比截断数大的数字。removeVal选择的值一定比removeVal小， 因此找到第一个比截断大的值存下标，
        // 从此记录，等待后续往前找。
        AscendC::LocalTensor<T> buf = inputBuf_.Get<T>();
        AscendC::LocalTensor<float> fp32Buf = fp32Buf_.Get<float>();
        flag_ = 0;
        tempRandVal_ = removeVal_ * (*(tilingUb_ + offset)); // 得到截断数

        for (int j = 0; j < perCoreRunNum_; j++) {
            uint32_t RunNum = (j == perCoreRunNum_ - 1) ? lElePerCoreLastRun_ : nlElePerCorePerRun_;
            uint64_t offsetXgm = (uint64_t)blockIdx_ * nlCoreRun_ * realLastDim_ + (uint64_t)cid * realLastDim_ +
                                 (uint64_t)j * tempUbEleAligened_;
            uint32_t RunNumF16Align = (RunNum + FP16_PER_BLOCK - 1) / FP16_PER_BLOCK * FP16_PER_BLOCK;
            AscendC::PipeBarrier<PIPE_MTE2>();
            DataCopy(buf, xGm_[offsetXgm], RunNumF16Align); // 拷贝单核每次的一块数。
            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(fp32Buf, buf, AscendC::RoundMode::CAST_NONE, RunNumF16Align);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);

            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
            // 此处多拷贝，底下仍然根据原长度做判断。
            if (fp32Buf.GetValue(RunNum - 1) < tempRandVal_) {
                continue;
            } else if (flag_ == 0) {
                flag_ = 1;      // 只需选择一次，增加标志位
                idxReturn_ = j; // 保留这一块的idx
            }
            if (fp32Buf.GetValue(RunNum - 1) < removeVal_) {
                continue;
            } else {
                idxForward_ = j; // 保留这一块的最后一个值的绝对idx
                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                break;
            }
        }
    }

    __aicore__ inline void Process(__gm__ uint8_t *sync)
    {
        AscendC::LocalTensor<T> buf = inputBuf_.Get<T>();
        AscendC::LocalTensor<float> fp32Buf = fp32Buf_.Get<float>();
        AscendC::LocalTensor<T> toppBuf_ = yBuf_.Get<T>();
        AscendC::LocalTensor<float> toppBufF32_ = yF32Buf_.Get<float>();
        AscendC::LocalTensor<int32_t> int32BlkBuf = int32Buf_.Get<int32_t>();

        PickUpRand();
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        // 最开始清空数据
        Duplicate(int32BlkBuf, (int32_t)0, MAX_CORE_NUM);
        AscendC::LocalTensor<uint32_t> uint32Buf_ = int8Buf_.Get<uint32_t>();
        Duplicate(uint32Buf_, uint32_t(0), tempUbEleAligened_ / BLK_SIZE);
        // 截断数可能是batch个，也可能是1个
        // 每个batch往后取一个随机数。(*(tilingUb_ + batchOffset))
        Cast(toppBufF32_, toppBuf_, AscendC::RoundMode::CAST_NONE, maxBatch_);
        for (int cid = 0; cid < dynamicRound_; cid++) { // 每个核做多少次
            absIdx_ = 0;
            uint32_t batchOffset = (blockIdx_ * nlCoreRun_ + cid) % MAX_CORE_NUM;
            if constexpr (MULTI) {
                AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);

                AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
                removeVal_ = toppBufF32_.GetValue((blockIdx_ * nlCoreRun_ + cid));
            } else {
                AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);

                AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
                removeVal_ = toppBufF32_.GetValue(0);
            }
            tempRandVal_ = removeVal_ * (*(tilingUb_ + batchOffset)); // 得到截断数
            FirstPick(cid, batchOffset);
            uint32_t RunNum = (idxForward_ == perCoreRunNum_ - 1) ? lElePerCoreLastRun_ : nlElePerCorePerRun_;
            Compute(RunNum, removeVal_, cid, 0);
            Cast(fp32Buf, buf, AscendC::RoundMode::CAST_NONE, RunNum);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);

            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
            float finalVal = FigureOutValue(cid); // 寻找到最后一个比截断值肖的值;
            float cutOff = finalVal * (*(tilingUb_ + batchOffset));
            // 拿到比removeVal 小的第一个数，开始往前做，取下标。
            finalPick(cutOff, cid);
        }
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        CoreSyncOut(sync);
    }

private:
    __aicore__ inline void CopyIn(uint32_t coreIdx, uint32_t loopIdx, uint32_t copyEleNum)
    {
        AscendC::LocalTensor<T> buf = inputBuf_.Get<T>();

        uint32_t copyEleNumAligned_ = (copyEleNum + FP16_PER_BLOCK - 1) / FP16_PER_BLOCK * FP16_PER_BLOCK;
        uint64_t xGmOffsetSec = (uint64_t)blockIdx_ * nlCoreRun_ * realLastDim_ + (uint64_t)coreIdx * realLastDim_ +
                                (uint64_t)loopIdx * tempUbEleAligened_;
        DataCopy(buf, xGm_[xGmOffsetSec], copyEleNumAligned_); // 拷贝单核每次的一块数。
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    // 首个找到的一块的最后一个数一定是> removeVal的,并且前一块一定是小于的
    __aicore__ inline void Compute(uint32_t copyEleNum, float compareVal, uint32_t cid, uint32_t flag)
    {
        AscendC::LocalTensor<T> buf = inputBuf_.Get<T>();
        AscendC::LocalTensor<half> tempBuf = tempBuf_.Get<half>();
        AscendC::LocalTensor<uint8_t> uint8Buf = int8Buf_.Get<uint8_t>();
        AscendC::LocalTensor<float> blkBuf = blockBuf_.Get<float>();
        AscendC::LocalTensor<float> fp32Buf = fp32Buf_.Get<float>();
        AscendC::LocalTensor<half> fp16Buf = fp32Buf_.Get<half>();
        AscendC::LocalTensor<float> fp32TempBuf = tempBuf_.Get<float>();
        uint32_t copyEleNumAlignF16_ = (copyEleNum + FP16_PER_BLOCK - 1) / FP16_PER_BLOCK * FP16_PER_BLOCK;
        uint32_t copyEleNumAlignF32_ = (copyEleNum + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
        for (uint32_t dupVal = copyEleNum; dupVal < copyEleNumAlignF16_; dupVal++) {
            buf.SetValue(dupVal, T(1));
        }
        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Cast(fp32TempBuf, buf, AscendC::RoundMode::CAST_NONE, copyEleNumAlignF16_);
        AscendC::PipeBarrier<PIPE_V>();
        Duplicate(fp32Buf, compareVal, tempUbEleAligened_); // duplicate一个removeVal的tensor去比较 250
        AscendC::PipeBarrier<PIPE_V>();
        if (flag == 0) {
            Compare(uint8Buf, fp32TempBuf, fp32Buf, AscendC::CMPMODE::LT, copyEleNumAlignF32_); // 找到小于removeVal的；
        } else {
            Compare(uint8Buf, fp32TempBuf, fp32Buf, AscendC::CMPMODE::LE,
                    copyEleNumAlignF32_); // 找到小于等于removeVal的；
        }
        AscendC::PipeBarrier<PIPE_V>();

        Duplicate(fp16Buf, (half)1, tempUbEleAligened_); // duplicate一个removeVal的tensor去比较
        AscendC::PipeBarrier<PIPE_V>();

        Select(tempBuf, uint8Buf, fp16Buf, (half)0, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
               copyEleNumAlignF16_); // 根据compare去选择1 还是 0；
        AscendC::PipeBarrier<PIPE_V>();

        Cast(fp32Buf, tempBuf, AscendC::RoundMode::CAST_NONE, copyEleNum);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);

        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t dupVal = copyEleNum; dupVal < copyEleNumAlignF32_; dupVal++) {
            fp32Buf.SetValue(dupVal, (float)0);
        }
        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);

        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        ReduceSum(blkBuf, fp32Buf, fp32TempBuf, copyEleNumAlignF32_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float FigureOutValue(uint32_t cid)
    {
        AscendC::LocalTensor<float> blkBuf = blockBuf_.Get<float>();
        AscendC::LocalTensor<float> fp32Buf = fp32Buf_.Get<float>();
        AscendC::LocalTensor<int32_t> selectRangeBlkBuf = selectRangeBlkBuf_.Get<int32_t>();
        float sec2Last = blkBuf.GetValue(0);
        auto relativeSelectRange = static_cast<int32_t>(sec2Last);
        lastVal_ = relativeSelectRange - 1;
        relativeSelectRange += idxForward_ * tempUbEleAligened_;
        selectRangeBlkBuf.SetValue(cid, relativeSelectRange);
        if (lastVal_ < 0) {
            return fp32Buf.GetValue(0);
        }
        return fp32Buf.GetValue((uint32_t)lastVal_); // buf里面本就是相对idx 因此不需要修改。
    }

    __aicore__ inline void finalPick(float cutOff, uint32_t cid)
    {
        AscendC::LocalTensor<float> blkBuf = blockBuf_.Get<float>();
        AscendC::LocalTensor<int32_t> int32BlkBuf = int32Buf_.Get<int32_t>();
        for (int j = idxReturn_; j >= 0; j--) {
            uint32_t RunNum = (j == perCoreRunNum_ - 1) ? lElePerCoreLastRun_ : nlElePerCorePerRun_;
            CopyIn(cid, j, RunNum);
            AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Compute(RunNum, cutOff, cid, 1);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);

            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
            lastValTemp = blkBuf.GetValue(0);
            if (lastValTemp > (float)0.0) {
                absIdx_ = j;
                break;
            }
        }
        lastVal_ = (int32_t)lastValTemp + (int32_t)absIdx_ * (int32_t)tempUbEleAligened_; // 绝对位置
        if (lastVal_ > 0) {
            int32BlkBuf.SetValue(cid, lastVal_);
        }
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<int32_t> int32BlkBuf = int32Buf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> selectRangeBlkBuf = selectRangeBlkBuf_.Get<int32_t>();
        if (blockIdx_ != realCore_ - 1) {
            uint32_t dynamicRoundAlign_ = DEFAULT_STRIDE * ((dynamicRound_ + DEFAULT_STRIDE - 1) / DEFAULT_STRIDE);
            DataCopy(zGm_[static_cast<uint64_t>(blockIdx_) * nlCoreRun_], int32BlkBuf, dynamicRoundAlign_);
            DataCopy(selectRangeGm_[static_cast<uint64_t>(blockIdx_) * nlCoreRun_], selectRangeBlkBuf,
                     dynamicRoundAlign_);
        }
        else {
            uint32_t dynamicRoundAlign_ = dynamicRound_ / DEFAULT_STRIDE * DEFAULT_STRIDE;
            if (dynamicRoundAlign_ > 0) {
                DataCopy(zGm_[static_cast<uint64_t>(blockIdx_) * nlCoreRun_], int32BlkBuf, dynamicRound_);
                DataCopy(selectRangeGm_[static_cast<uint64_t>(blockIdx_) * nlCoreRun_], selectRangeBlkBuf,
                         dynamicRound_);
            }
            AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            for (int i = dynamicRoundAlign_; i < dynamicRound_; i++) {
                zGm_(static_cast<uint64_t>(blockIdx_) * nlCoreRun_ + i) = int32BlkBuf.GetValue(i);
                selectRangeGm_(static_cast<uint64_t>(blockIdx_) * nlCoreRun_ + i) = selectRangeBlkBuf.GetValue(i);
            }
        }
    }

    __aicore__ inline void CoreSyncOut(__gm__ uint8_t *sync)
    {
        syncGm_.SetGlobalBuffer((__gm__ int32_t *)(sync), BLK_SIZE * MAX_CORE_NUM * firstDim_);
        if (realCore_ != 1) {
            if (blockIdx_ == realCore_ - 1) {
                auto syncBuf = vecIn_.AllocTensor<int32_t>();
                AscendC::IBWait(syncGm_, syncBuf, realCore_ - 2, 0);
                CopyOut();
                vecIn_.FreeTensor(syncBuf);
            } else if (blockIdx_ == 0) {
                auto syncBuf = vecIn_.AllocTensor<int32_t>();
                CopyOut();
                AscendC::IBSet(syncGm_, syncBuf, 0, 0);
                vecIn_.FreeTensor(syncBuf);
            } else {
                auto syncBuf = vecIn_.AllocTensor<int32_t>();
                AscendC::IBWait(syncGm_, syncBuf, blockIdx_ - 1, 0);
                CopyOut();
                AscendC::IBSet(syncGm_, syncBuf, blockIdx_, 0);
                vecIn_.FreeTensor(syncBuf);
            }
        } else {
            CopyOut();
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> vecIn_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_, yF32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> int8Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> blockBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> int32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selectRangeBlkBuf_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<int32_t> syncGm_;
    AscendC::GlobalTensor<int32_t> zGm_;
    AscendC::GlobalTensor<int32_t> selectRangeGm_;
    uint32_t realLastDim_{0};
    uint32_t expandLastDim_{0};
    uint32_t numSamplesMax_{0};
    uint32_t firstDim_{0};
    uint32_t maxBatch_{0};
    float maxNum_{0};
    float tempValue_{0};
    uint32_t perCoreRunNum_{0};
    uint32_t nlElePerCorePerRun_{0};
    uint32_t lElePerCoreLastRun_{0};
    uint32_t tempUbEleAligened_{19456};
    uint32_t realLastDimTemp_{0};
    float *tilingUb_{nullptr};
    int32_t *resOut_{nullptr};
    float removeVal_{0};
    float tempRandVal_{0};
    uint32_t idxReturn_{0};
    uint32_t idxForward_{0};
    uint32_t loopReturn_{0};
    uint32_t flag_{0};
    int32_t lastVal_{0};
    uint32_t absIdx_{0};
    uint32_t nlCoreRun_{1};
    uint32_t lCoreRun_{1};
    uint32_t dynamicRound_{1};
    uint32_t realCore_{1};
    uint32_t blockIdx_{0};
    uint32_t realLastDimAlignF32_{64};
    uint32_t realLastDimAlignF16_{128};
    float lastValTemp;
};
} // namespace

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AtbOps::ToppsampleTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->realLastDim = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->expandLastDim = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->firstDim = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->numSamplesMax = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->perCoreRunNum = (*(const __gm__ int32_t *)(p_tilingdata + 16));
    tilingdata->nlElePerCorePerRun = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->lElePerCoreLastRun = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
    tilingdata->tempUbEleAligened = (*(const __gm__ uint32_t *)(p_tilingdata + 28));
    for (uint32_t i = 0; i < tilingdata->numSamplesMax; i++) {
        tilingdata->randValList[i] = (*(const __gm__ float *)(p_tilingdata + 32 + i * sizeof(float)));
    }
#else
    AscendC::TPipe pipe_;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AtbOps::ToppsampleTilingData), &pipe_);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->realLastDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->expandLastDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->firstDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->numSamplesMax = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingdata->perCoreRunNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingdata->nlElePerCorePerRun = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 20));
    tilingdata->lElePerCoreLastRun = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    tilingdata->tempUbEleAligened = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 28));
    for (uint32_t i = 0; i < tilingdata->numSamplesMax; i++) {
        tilingdata->randValList[i] = (*(__ubuf__ float *)(tilingdata_in_ub + 32 + i * sizeof(float)));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                                       \
    AtbOps::ToppsampleTilingData tiling_data;                                                                          \
    InitTilingData((tiling_arg), &(tiling_data))

extern "C" __global__ __aicore__ void toppsample(GM_ADDR cumsumed_probs, GM_ADDR topp, GM_ADDR select_index,
                                                 GM_ADDR select_range, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        KernelToppsample<false, half> op;
        op.Init(cumsumed_probs, topp, select_index, select_range, tiling_data);
        op.Process(workspace);
    } else if (TILING_KEY_IS(0)) {
        KernelToppsample<true, half> op;
        op.Init(cumsumed_probs, topp, select_index, select_range, tiling_data);
        op.Process(workspace);
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(3)) {
        KernelToppsample<false, bfloat16_t> op;
        op.Init(cumsumed_probs, topp, select_index, select_range, tiling_data);
        op.Process(workspace);
    } else if (TILING_KEY_IS(2)) {
        KernelToppsample<true, bfloat16_t> op;
        op.Init(cumsumed_probs, topp, select_index, select_range, tiling_data);
        op.Process(workspace);
    }
#endif
}