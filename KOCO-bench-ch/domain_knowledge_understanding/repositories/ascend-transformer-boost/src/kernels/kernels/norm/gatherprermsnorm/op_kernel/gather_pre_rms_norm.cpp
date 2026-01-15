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
#include "kernels/utils/kernel/kernel_utils.h"
#include "kernels/norm/gatherprermsnorm/tiling/gather_pre_rms_norm_tiling_data.h"
#include "kernels/norm/rmsnormforward/op_kernel/rms_norm_base.h"
#include "kernels/norm/common/common_pre_post/comm_pre_post.h"

using AscendC::HardEvent;

static constexpr uint32_t DOUBLE_BUFFER_NUM = 2;  // split the UB to 2 equal part to enable ping-pong techniques.

template <typename T>
class GatherPreRmsnormSmallIndices {
public:
    __aicore__ inline GatherPreRmsnormSmallIndices(AscendC::TPipe *pipe, __gm__ uint8_t *x, __gm__ uint8_t *rIn,
        __gm__ uint8_t *indices, __gm__ uint8_t *g, __gm__ uint8_t *y, __gm__ uint8_t *rOut,
        AsdOps::GatherPreRmsNormTilingData &tilingData)
    {
        numCore_ = tilingData.numCore;
        numCol_ = tilingData.numCol;
        numRow_ = tilingData.numRow;
        avgFactor_ = tilingData.avgFactor;
        epsilon_ = tilingData.epsilon;
        numRowPerCore_ = tilingData.numRowPerCore;
        numRowPerCoreAlign_ = tilingData.numRowPerCoreAlign;
        numRowTailCore_ = tilingData.numRowTailCore;

        if (AscendC::GetBlockIdx() != numCore_ - 1) {
            rowWork_ = numRowPerCore_;
        } else {
            rowWork_ = numRowTailCore_;
        }
        gmIndicesOffset_ = static_cast<uint64_t>(numRowPerCore_);
        gmOffset_ = static_cast<uint64_t>(numRowPerCore_ * numCol_);

        repeatTimes_ = numCol_ / FP32_PER_REPEAT;
        alignRepeatOffset_ = repeatTimes_ * FP32_PER_REPEAT;
        tailRepeatNum_ = numCol_ - alignRepeatOffset_;

        gmX_.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gmOffset_);
        gmResIn_.SetGlobalBuffer((__gm__ T *)rIn);
        gmIndices_.SetGlobalBuffer((__gm__ int32_t *)indices + AscendC::GetBlockIdx() * gmIndicesOffset_);
        gmGamma_.SetGlobalBuffer((__gm__ T *)g);
        gmY_.SetGlobalBuffer((__gm__ float *)y + AscendC::GetBlockIdx() * gmOffset_);
        gmResOut_.SetGlobalBuffer((__gm__ T *)rOut + AscendC::GetBlockIdx() * gmOffset_);

        pipe->InitBuffer(fp16BufX_, DOUBLE_BUFFER_NUM * numCol_ * sizeof(T));  // double buffer
        pipe->InitBuffer(fp16BufResIn_, DOUBLE_BUFFER_NUM * numCol_ * sizeof(T));  // double buffer
        pipe->InitBuffer(fp16BufIndices_, numRowPerCoreAlign_ * sizeof(int32_t));
        pipe->InitBuffer(fp16BufGamma_, numCol_ * sizeof(T));
        pipe->InitBuffer(fp16BufResOut_, numCol_ * sizeof(T));
        pipe->InitBuffer(fp32BufY_, numCol_ * sizeof(float));
        pipe->InitBuffer(fp32BufXy_, numCol_ * sizeof(float));
        pipe->InitBuffer(calcBuf_, numCol_ * sizeof(float));
        pipe->InitBuffer(brcbBuf_, AsdOps::BRCB_BYTE);
    }

    __aicore__ inline void Launch()
    {
        event_t eventCur;
        event_t eventCur1;
        AscendC::LocalTensor<T> g = fp16BufGamma_.Get<T>();
        AscendC::LocalTensor<int32_t> indices = fp16BufIndices_.Get<int32_t>();
        DataCopy(g, gmGamma_[0], numCol_);
        DataCopy(indices, gmIndices_[0], numRowPerCoreAlign_);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);  // Synchronised EventId between ping and pong
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);  // Synchronised EventId between ping and pong

        for (uint32_t pid = 0; pid < rowWork_; pid++) {
            eventCur = pid % DOUBLE_BUFFER_NUM == 0 ? EVENT_ID0: EVENT_ID1;
            eventCur1 = pid % DOUBLE_BUFFER_NUM == 0 ? EVENT_ID4: EVENT_ID5;
            auto ind = indices.GetValue(pid);
            uint64_t offsetX = static_cast<uint64_t>(pid * numCol_);
            uint64_t offsetResIn = static_cast<uint64_t>(ind * numCol_);
            uint64_t bufOffset = static_cast<uint64_t>((pid % DOUBLE_BUFFER_NUM) * numCol_);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventCur);

            CopyInAll(offsetX, offsetResIn, numCol_, bufOffset);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventCur);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventCur);

            Compute(offsetX, numCol_, bufOffset, eventCur1);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventCur);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventCur);

            CopyOutY(offsetX, numCol_);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventCur);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);  // Synchronised EventId between ping and pong
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);  // Synchronised EventId between ping and pong
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);  // Synchronised EventId between ping and pong
    }

private:
    __aicore__ inline void CopyInAll(uint64_t offsetX, uint64_t offsetResIn, uint32_t numel, uint64_t bufOffset)
    {
        AscendC::LocalTensor<T> xIn = fp16BufX_.Get<T>();
        AscendC::LocalTensor<T> resIn = fp16BufResIn_.Get<T>();
        DataCopy(xIn[bufOffset], gmX_[offsetX], numel);
        DataCopy(resIn[bufOffset], gmResIn_[offsetResIn], numel);
    }

    __aicore__ inline void Compute(uint64_t offsetX, uint32_t numel, uint64_t bufOffset, event_t eventId)
    {
        AscendC::LocalTensor<T> x = fp16BufX_.Get<T>();
        AscendC::LocalTensor<T> fp16X = x[bufOffset];
        AscendC::LocalTensor<T> resIn = fp16BufResIn_.Get<T>();
        AscendC::LocalTensor<T> fp16ResIn = resIn[bufOffset];
        AscendC::LocalTensor<T> g = fp16BufGamma_.Get<T>();

        AscendC::LocalTensor<float> fp32Y = fp32BufY_.Get<float>();
        AscendC::LocalTensor<T> resOut = fp16BufResOut_.Get<T>();

        AscendC::LocalTensor<float> fp32Xy = fp32BufXy_.Get<float>();
        AscendC::LocalTensor<float> fp32Tmp = calcBuf_.Get<float>();
        AscendC::LocalTensor<float> brcbWorkspace = brcbBuf_.Get<float>();

        AddResAndCast<T>(fp16X, fp16ResIn, fp32Xy, fp32Tmp, numel);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);  // Synchronised EventId between ping and pong
        CastFrom32To16(resOut, fp32Xy, numel);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);

        CopyOutRes(offsetX, numel);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);  // Synchronised EventId between ping and pong

        FigureOutNorm(fp32Tmp, fp32Xy, brcbWorkspace, avgFactor_, epsilon_, numel, repeatTimes_,
                      alignRepeatOffset_, tailRepeatNum_);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);  // Synchronised EventId between ping and pong

        MultiplyGamma(g, fp32Tmp, fp32Xy, fp32Y, numel);
    }

    __aicore__ inline void CopyOutY(uint64_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<float> fp32Y = fp32BufY_.Get<float>();
        DataCopy(gmY_[offset], fp32Y, numel);
    }

    __aicore__ inline void CopyOutRes(uint64_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<T> resOut = fp16BufResOut_.Get<T>();
        DataCopy(gmResOut_[offset], resOut, numel);
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufX_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufResIn_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufIndices_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufGamma_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufResOut_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32BufY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32BufXy_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> brcbBuf_;
    AscendC::GlobalTensor<T> gmX_;
    AscendC::GlobalTensor<T> gmResIn_;
    AscendC::GlobalTensor<int32_t> gmIndices_;
    AscendC::GlobalTensor<T> gmGamma_;
    AscendC::GlobalTensor<float> gmY_;
    AscendC::GlobalTensor<T> gmResOut_;
    uint32_t numCore_{0};  // 一共激活多少AICORE
    uint32_t numCol_{0};  // 输入的列数
    uint32_t numRow_{0};  // 索引后的输入行数
    float avgFactor_{1.0};     // numCol_的倒数
    float epsilon_{1e-12f};    // norm平滑参数
    uint32_t numRowPerCore_{0};  // 非尾核处理的行数
    uint32_t numRowPerCoreAlign_{0};  // 每个核处理的行数 32字节对齐
    uint32_t numRowTailCore_{0};  // 尾核处理的行数
    uint32_t rowWork_{0};  // 每个核处理的行数
    uint64_t gmOffset_{0};  // GM数据起始位置偏移量
    uint64_t gmIndicesOffset_{0};  // GM indices数据起始位置偏移量
    uint32_t repeatTimes_{1};  // 用于Brcb计算
    uint32_t alignRepeatOffset_{0}; // 用于Brcb计算
    uint32_t tailRepeatNum_{0};  // 用于Brcb计算
};

template <typename T>
class GatherPreRmsnormLargeIndices {
public:
    __aicore__ inline GatherPreRmsnormLargeIndices(AscendC::TPipe *pipe, __gm__ uint8_t *x, __gm__ uint8_t *rIn,
        __gm__ uint8_t *indices, __gm__ uint8_t *g, __gm__ uint8_t *y, __gm__ uint8_t *rOut,
        AsdOps::GatherPreRmsNormTilingData &tilingData)
    {
        numCore_ = tilingData.numCore;
        numCol_ = tilingData.numCol;
        numRow_ = tilingData.numRow;
        avgFactor_ = tilingData.avgFactor;
        epsilon_ = tilingData.epsilon;
        ubResByte_ = tilingData.ubResByte;
        numRowPerCore_ = tilingData.numRowPerCore;
        numRowTailCore_ = tilingData.numRowTailCore;

        if (AscendC::GetBlockIdx() != numCore_ - 1) {
            rowWork_ = numRowPerCore_;
        } else {
            rowWork_ = numRowTailCore_;
        }
        gmIndicesOffset_ = static_cast<uint64_t>(numRowPerCore_);
        gmOffset_ = static_cast<uint64_t>(numRowPerCore_ * numCol_);

        repeatTimes_ = numCol_ / FP32_PER_REPEAT;
        alignRepeatOffset_ = repeatTimes_ * FP32_PER_REPEAT;
        tailRepeatNum_ = numCol_ - alignRepeatOffset_;

        gmX_.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gmOffset_);
        gmResIn_.SetGlobalBuffer((__gm__ T *)rIn);
        gmIndices_.SetGlobalBuffer((__gm__ int32_t *)indices + AscendC::GetBlockIdx() * gmIndicesOffset_);
        gmGamma_.SetGlobalBuffer((__gm__ T *)g);
        gmY_.SetGlobalBuffer((__gm__ float *)y + AscendC::GetBlockIdx() * gmOffset_);
        gmResOut_.SetGlobalBuffer((__gm__ T *)rOut + AscendC::GetBlockIdx() * gmOffset_);

        pipe->InitBuffer(fp16BufX_, DOUBLE_BUFFER_NUM * numCol_ * sizeof(T));  // double buffer
        pipe->InitBuffer(fp16BufResIn_, DOUBLE_BUFFER_NUM * numCol_ * sizeof(T));  // double buffer
        pipe->InitBuffer(fp16BufIndices_, ubResByte_);
        pipe->InitBuffer(fp16BufGamma_, numCol_ * sizeof(T));
        pipe->InitBuffer(fp16BufResOut_, numCol_ * sizeof(T));
        pipe->InitBuffer(fp32BufY_, numCol_ * sizeof(float));
        pipe->InitBuffer(fp32BufXy_, numCol_ * sizeof(float));
        pipe->InitBuffer(calcBuf_, numCol_ * sizeof(float));
        pipe->InitBuffer(brcbBuf_, AsdOps::BRCB_BYTE);
    }

    __aicore__ inline void Launch()
    {
        uint32_t indicesNumInOneLoop = ubResByte_ / sizeof(int32_t);
        uint32_t loopNum = (rowWork_ + indicesNumInOneLoop - 1) / indicesNumInOneLoop;
        uint32_t tailIndicesNum = rowWork_ - (loopNum - 1) * indicesNumInOneLoop;
        uint32_t tailIndicesNumAlign = (tailIndicesNum + AsdOps::INT32_ALIGN_NUM - 1) /
                                        AsdOps::INT32_ALIGN_NUM * AsdOps::INT32_ALIGN_NUM;
        for (uint32_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
            uint64_t indiceLoopOffset = static_cast<uint64_t>(loopIdx * indicesNumInOneLoop);
            uint32_t inputXLoopOffset = static_cast<uint32_t>(loopIdx * indicesNumInOneLoop * numCol_);
            if (loopIdx != loopNum - 1) {
                ComputeOneLoop(indiceLoopOffset, indicesNumInOneLoop, indicesNumInOneLoop, inputXLoopOffset);
            } else {
                ComputeOneLoop(indiceLoopOffset, tailIndicesNumAlign, tailIndicesNum, inputXLoopOffset);
            }
        }
    }

private:
    __aicore__ inline void ComputeOneLoop(uint64_t indiceLoopOffset, uint32_t indicesCopyInNum,
        uint32_t indicesComputeNum, uint32_t inputXLoopOffset)
    {
        event_t eventCur;
        event_t eventCur1;
        AscendC::LocalTensor<T> g = fp16BufGamma_.Get<T>();
        AscendC::LocalTensor<int32_t> indices = fp16BufIndices_.Get<int32_t>();
        DataCopy(g, gmGamma_[0], numCol_);
        DataCopy(indices, gmIndices_[indiceLoopOffset], indicesCopyInNum);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);  // Synchronised EventId between ping and pong
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);  // Synchronised EventId between ping and pong

        for (uint32_t pid = 0; pid < indicesComputeNum; pid++) {
            eventCur = pid % DOUBLE_BUFFER_NUM == 0 ? EVENT_ID0: EVENT_ID1;
            eventCur1 = pid % DOUBLE_BUFFER_NUM == 0 ? EVENT_ID4: EVENT_ID5;
            auto ind = indices.GetValue(pid);
            uint64_t offsetX = static_cast<uint64_t>(inputXLoopOffset + pid * numCol_);
            uint64_t offsetResIn = static_cast<uint64_t>(ind * numCol_);
            uint64_t bufOffset = static_cast<uint64_t>((pid % DOUBLE_BUFFER_NUM) * numCol_);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventCur);

            CopyInAll(offsetX, offsetResIn, numCol_, bufOffset);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventCur);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventCur);

            Compute(offsetX, numCol_, bufOffset, eventCur1);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventCur);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventCur);

            CopyOutY(offsetX, numCol_);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventCur);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);  // Synchronised EventId between ping and pong
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);  // Synchronised EventId between ping and pong
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);  // Synchronised EventId between ping and pong
    }

private:
    __aicore__ inline void CopyInAll(uint64_t offsetX, uint64_t offsetResIn, uint32_t numel, uint64_t bufOffset)
    {
        AscendC::LocalTensor<T> xIn = fp16BufX_.Get<T>();
        AscendC::LocalTensor<T> resIn = fp16BufResIn_.Get<T>();
        DataCopy(xIn[bufOffset], gmX_[offsetX], numel);
        DataCopy(resIn[bufOffset], gmResIn_[offsetResIn], numel);
    }

    __aicore__ inline void Compute(uint64_t offsetX, uint32_t numel, uint64_t bufOffset, event_t eventId)
    {
        AscendC::LocalTensor<T> x = fp16BufX_.Get<T>();
        AscendC::LocalTensor<T> fp16X = x[bufOffset];
        AscendC::LocalTensor<T> resIn = fp16BufResIn_.Get<T>();
        AscendC::LocalTensor<T> fp16ResIn = resIn[bufOffset];
        AscendC::LocalTensor<T> g = fp16BufGamma_.Get<T>();

        AscendC::LocalTensor<float> fp32Y = fp32BufY_.Get<float>();
        AscendC::LocalTensor<T> resOut = fp16BufResOut_.Get<T>();

        AscendC::LocalTensor<float> fp32Xy = fp32BufXy_.Get<float>();
        AscendC::LocalTensor<float> fp32Tmp = calcBuf_.Get<float>();
        AscendC::LocalTensor<float> brcbWorkspace = brcbBuf_.Get<float>();

        AddResAndCast<T>(fp16X, fp16ResIn, fp32Xy, fp32Tmp, numel);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);  // Synchronised EventId between ping and pong
        CastFrom32To16(resOut, fp32Xy, numel);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);

        CopyOutRes(offsetX, numel);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);  // Synchronised EventId between ping and pong

        FigureOutNorm(fp32Tmp, fp32Xy, brcbWorkspace, avgFactor_, epsilon_, numel, repeatTimes_,
                      alignRepeatOffset_, tailRepeatNum_);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);  // Synchronised EventId between ping and pong

        MultiplyGamma(g, fp32Tmp, fp32Xy, fp32Y, numel);
    }

    __aicore__ inline void CopyOutY(uint64_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<float> fp32Y = fp32BufY_.Get<float>();
        DataCopy(gmY_[offset], fp32Y, numel);
    }

    __aicore__ inline void CopyOutRes(uint64_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<T> resOut = fp16BufResOut_.Get<T>();
        DataCopy(gmResOut_[offset], resOut, numel);
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufX_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufResIn_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufIndices_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufGamma_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufResOut_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32BufY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32BufXy_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> brcbBuf_;
    AscendC::GlobalTensor<T> gmX_;
    AscendC::GlobalTensor<T> gmResIn_;
    AscendC::GlobalTensor<int32_t> gmIndices_;
    AscendC::GlobalTensor<T> gmGamma_;
    AscendC::GlobalTensor<float> gmY_;
    AscendC::GlobalTensor<T> gmResOut_;
    uint32_t numCore_{0};  // 一共激活多少AICORE
    uint32_t numCol_{0};  // 输入的列数
    uint32_t numRow_{0};  // 索引后的输入行数
    float avgFactor_{1.0};     // numCol_的倒数
    float epsilon_{1e-12f};    // norm平滑参数
    uint32_t ubResByte_{0};  // 能够用于存放indices的ub大小 单位字节
    uint32_t numRowPerCore_{0};  // 非尾核处理的行数
    uint32_t numRowTailCore_{0};  // 尾核处理的行数
    uint32_t rowWork_{0};  // 每个核处理的行数
    uint64_t gmOffset_{0};  // GM数据起始位置偏移量
    uint64_t gmIndicesOffset_{0};  // GM indices数据起始位置偏移量
    uint32_t repeatTimes_{1};  // 用于Brcb计算
    uint32_t alignRepeatOffset_{0}; // 用于Brcb计算
    uint32_t tailRepeatNum_{0};  // 用于Brcb计算
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *pTilingdata, AsdOps::GatherPreRmsNormTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numCore = (*(const __gm__ uint32_t *)(pTilingdata + 0));
    tilingdata->numCol = (*(const __gm__ uint32_t *)(pTilingdata + 4));
    tilingdata->numRow = (*(const __gm__ uint32_t *)(pTilingdata + 8));
    tilingdata->avgFactor = (*(const __gm__ float *)(pTilingdata + 12));
    tilingdata->epsilon = (*(const __gm__ float *)(pTilingdata + 16));
    tilingdata->ubResByte = (*(const __gm__ uint32_t *)(pTilingdata + 20));
    tilingdata->numRowPerCore = (*(const __gm__ uint32_t *)(pTilingdata + 24));
    tilingdata->numRowPerCoreAlign = (*(const __gm__ uint32_t *)(pTilingdata + 28));
    tilingdata->numRowTailCore = (*(const __gm__ uint32_t *)(pTilingdata + 32));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdataInUb = nullptr;
    CopyGmTilingToUb(tilingdataInUb, pTilingdata, sizeof(AsdOps::GatherPreRmsNormTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numCore = (*(__ubuf__ uint32_t *)(tilingdataInUb + 0));
    tilingdata->numCol = (*(__ubuf__ uint32_t *)(tilingdataInUb + 4));
    tilingdata->numRow = (*(__ubuf__ uint32_t *)(tilingdataInUb + 8));
    tilingdata->avgFactor = (*(__ubuf__ float *)(tilingdataInUb + 12));
    tilingdata->epsilon = (*(__ubuf__ float *)(tilingdataInUb + 16));
    tilingdata->ubResByte = (*(__ubuf__ uint32_t *)(tilingdataInUb + 20));
    tilingdata->numRowPerCore = (*(__ubuf__ uint32_t *)(tilingdataInUb + 24));
    tilingdata->numRowPerCoreAlign = (*(__ubuf__ uint32_t *)(tilingdataInUb + 28));
    tilingdata->numRowTailCore = (*(__ubuf__ uint32_t *)(tilingdataInUb + 32));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tilingData, tilingArg)                                                                       \
    AsdOps::GatherPreRmsNormTilingData tilingData;                                                                  \
    InitTilingData((tilingArg), &(tilingData))

extern "C" __global__ __aicore__ void gather_pre_rms_norm(GM_ADDR x, GM_ADDR rIn, GM_ADDR indices, GM_ADDR g,
    GM_ADDR y, GM_ADDR rOut, GM_ADDR tiling)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    PRELOAD(3);  // icache预取, 减少icache miss
#endif
    AscendC::TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(100)) { // fp16, no slice, small indices
        GatherPreRmsnormSmallIndices<half> kernel(&pipe, x, rIn, indices, g, y, rOut, tilingData);
        kernel.Launch();
    }
    if (TILING_KEY_IS(101)) { // fp16, no slice, large indices
        GatherPreRmsnormLargeIndices<half> kernel(&pipe, x, rIn, indices, g, y, rOut, tilingData);
        kernel.Launch();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(110)) { // bf16, no slice, small indices
        GatherPreRmsnormSmallIndices<bfloat16_t> kernel(&pipe, x, rIn, indices, g, y, rOut, tilingData);
        kernel.Launch();
    }
    if (TILING_KEY_IS(111)) { // bf16, no slice, large indices
        GatherPreRmsnormLargeIndices<bfloat16_t> kernel(&pipe, x, rIn, indices, g, y, rOut, tilingData);
        kernel.Launch();
    }
#endif
}
