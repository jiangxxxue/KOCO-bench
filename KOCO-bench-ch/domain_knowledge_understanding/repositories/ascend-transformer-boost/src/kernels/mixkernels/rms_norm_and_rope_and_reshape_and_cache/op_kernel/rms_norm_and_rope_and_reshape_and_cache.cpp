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
#include "kernels/utils/kernel/simd.h"
#include "../tiling/rms_norm_and_rope_and_reshape_and_cache_tiling_data.h"

namespace RmsNormAndRopeAndReshapeAndCacheFusion {
using namespace AscendC;

constexpr uint32_t BUF_FACTOR = 2; // 1(g) + 1(workspace) = 2
constexpr uint32_t OFFSET_GAMMA = 0;
constexpr uint32_t OFFSET_CALC = 1;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr int64_t SLICE_SIZE_FP16 = 12288;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t BLK_SIZE = 32; // 一个block字节数
const constexpr uint8_t REPEAT_STRIDE = 8; // 默认stride, 8 * 32 = 256
constexpr uint32_t ELE_NUM_FP16 = 16; // 一个block fp16元素个数
// constexpr uint32_t MAX_ROW = 30; // 一个循环处理的最大行数

template <typename K_DTYPE>
class RmsNormAndRopeAndReshapeAndCache {
public:
    __aicore__ inline RmsNormAndRopeAndReshapeAndCache() {};
    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR x, GM_ADDR gamma, GM_ADDR keyRope, GM_ADDR cos, GM_ADDR sin,
        GM_ADDR slotMapping, GM_ADDR keycacheIn, GM_ADDR keycacheOut,
        const AtbOps::RmsNormAndRopeAndReshapeAndCacheTilingData& tilingData);
    __aicore__ inline void Process();

protected:
    __aicore__ inline float ComputeSliceSquareSum(LocalTensor<float>& in, LocalTensor<float>& tmp,
        LocalTensor<float>& workLocal, uint32_t count);
    __aicore__ inline void ComputeRmsNormFast(LocalTensor<K_DTYPE>& out, LocalTensor<float> in, float rms,
        LocalTensor<K_DTYPE>& gamma, uint32_t count, LocalTensor<float>& tmp, LocalTensor<float>& gammaFp32);
    __aicore__ inline void CopykVCache(LocalTensor<K_DTYPE>& ubAddr, GlobalTensor<K_DTYPE>& dst, uint64_t cacheStart,
        DataCopyParams& copyParamsOut);
    __aicore__ inline void ComputeRmsNorm(LocalTensor<K_DTYPE>& RACTokenLocal, LocalTensor<K_DTYPE>& gamma,
        LocalTensor<float>& gammaFp32);
    __aicore__ inline void ExpandNeg(LocalTensor<K_DTYPE>& tempBuf, uint32_t headNumTemp, uint32_t rowWorkNum);
    __aicore__ inline void CosSinBroadcast(LocalTensor<K_DTYPE> &tempBuf, uint32_t Calclen, uint32_t rowWorkNum,
        uint32_t cosSinOffset);
    __aicore__ inline void QkComm(const GlobalTensor<K_DTYPE> &src, uint32_t hiddenSizeTmp,
        LocalTensor<K_DTYPE> &tempBuf, uint32_t headNumTemp, uint32_t ropeOffset);
    __aicore__ inline void CalcRopeAlign(LocalTensor<K_DTYPE> &tempBuf, uint32_t repeatTimes1, uint32_t oriPosTemp,
        uint32_t removeTemp, uint32_t padTemp);

    GlobalTensor<K_DTYPE> xGm, gammaGm, ropeGm, cosGm, sinGm, keyCacheOutGm;
    GlobalTensor<int32_t> slotMappingGm;
    TBuf<TPosition::VECCALC> xFp16TmpBuffer, xFp32TmpBuffer, gammaFp16TmpBuffer, gammaFp32TmpBuffer, calcBuffer,
        keyRopeFp16TmpBuffer, keyRopeFp32TmpBuffer, keyRACFp16TmpBuffer, keyCacheInTmpBuffer;

    LocalTensor<K_DTYPE> xFp16, gamma, keyRopeFp16, RACTokenLocal;
    LocalTensor<float> xFp32, gammaFp32, keyRopeFp32;

    uint32_t numRow_;
    uint32_t rowWork;
    uint32_t numCol_;
    uint32_t numCore_;
    uint32_t rowWork_;
    float avgFactor_;
    uint32_t precisionMode_;
    float epsilon_;
    uint32_t ropeHiddenSizeK_;
    uint32_t ropeHeadDimK_;
    uint32_t ropeHeadNumK_;
    uint32_t rACNumTokens_;
    uint32_t rACHeadSizeK_;
    uint32_t repeatSize{128};
    uint32_t rotaryCoeff_;
    uint32_t currentCoreIdx;
    uint32_t xGmOffset;
    uint32_t ropeGmOffset;
    uint32_t rACTokenSizeK_;
    uint32_t rACNumBlocks_;
    uint32_t ropeRotateStride;
    uint32_t negOne_{0}; // -1 -1 -1 0 0 0在uB中的位置
    uint32_t cosPad_{0}; // broadcast的cos在uB中的位置
    uint32_t sinPad_{0}; // broadcast的sin在uB中的位置
    uint32_t oriPos_{0}; // q,k在uB中的位置
    uint32_t rowWorkNum{0};
    uint32_t cosPadTmp{0};
    uint32_t sinPadTmp{0};
    uint32_t oriPosTmp{0};
    uint32_t removeBeforeTmp{0};
    uint32_t padBefore_{0}; // 保存qk[-x : hiddensize - x]
    uint32_t removeBefore_{0}; // 保存qk[x : hiddensize + x]
    uint32_t sliceTimeK_{1};
    uint32_t rotaryStrideOffset{0}; // 每次旋转长度
    uint32_t maxRow_; // 一个循环处理的最大行数
    
private:
    TPipe* pipe;
    int32_t eventIdMTE22VA1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
};

template <typename K_DTYPE>
__aicore__ inline void RmsNormAndRopeAndReshapeAndCache<K_DTYPE>::Init(TPipe* pipeIn, GM_ADDR x, GM_ADDR gamma,
    GM_ADDR keyRope, GM_ADDR cos, GM_ADDR sin, GM_ADDR slotMapping, GM_ADDR keycacheIn, GM_ADDR keycacheOut,
    const AtbOps::RmsNormAndRopeAndReshapeAndCacheTilingData& tiling)
{
    numRow_ = tiling.numRow;
    numCol_ = tiling.numCol;
    numCore_ = tiling.numCore;
    avgFactor_ = tiling.avgFactor;
    precisionMode_ = tiling.precisionMode;
    epsilon_ = tiling.epsilon;
    ropeHiddenSizeK_ = tiling.ropeHiddenSizeK;
    ropeHeadDimK_ = tiling.ropeHeadDimK;
    ropeHeadNumK_ = tiling.ropeHeadNumK;
    rACNumTokens_ = tiling.racNumTokens;
    rACHeadSizeK_ = tiling.racHeadSizeK;
    rotaryCoeff_ = tiling.rotaryCoeff;
    maxRow_ = tiling.maxRow;
    currentCoreIdx = GetBlockIdx();
    if (currentCoreIdx >= numCore_) {
        return;
    }
    rowWork = (numRow_ + numCore_ - 1) / numCore_;
    rowWork_ = (currentCoreIdx < numCore_ - 1) ? rowWork : (numRow_ - rowWork * (numCore_ - 1));
    xGmOffset = static_cast<uint32_t>(rowWork) * numCol_;
    ropeGmOffset = static_cast<uint32_t>(rowWork) * ropeHiddenSizeK_;
    ropeRotateStride = ropeHeadDimK_ / rotaryCoeff_;
    cosPad_ = 0;
    sinPad_ = cosPad_ + ropeHeadDimK_ * maxRow_;
    negOne_ = sinPad_ + ropeHeadDimK_ * maxRow_;
    oriPos_ = negOne_ + ropeHeadDimK_ * maxRow_;
    padBefore_ = oriPos_ + ropeHeadDimK_;
    removeBefore_ = padBefore_ + ropeHeadDimK_ * maxRow_;
    rotaryStrideOffset = (ropeHeadDimK_ == rotaryCoeff_) ? 1 : ropeRotateStride; // 32
    xGm.SetGlobalBuffer((__gm__ K_DTYPE*)x + currentCoreIdx * xGmOffset);
    gammaGm.SetGlobalBuffer((__gm__ K_DTYPE*)gamma);
    ropeGm.SetGlobalBuffer((__gm__ K_DTYPE*)keyRope + currentCoreIdx * ropeGmOffset);
    cosGm.SetGlobalBuffer((__gm__ K_DTYPE*)cos + currentCoreIdx * ropeGmOffset);
    sinGm.SetGlobalBuffer((__gm__ K_DTYPE*)sin + currentCoreIdx * ropeGmOffset);
    slotMappingGm.SetGlobalBuffer((__gm__ int32_t*)slotMapping);
    keyCacheOutGm.SetGlobalBuffer((__gm__ K_DTYPE*)keycacheOut);

    pipe = pipeIn;
    pipe->InitBuffer(xFp16TmpBuffer, numCol_ * maxRow_ * sizeof(K_DTYPE));
    pipe->InitBuffer(gammaFp16TmpBuffer, numCol_ * sizeof(K_DTYPE));
    pipe->InitBuffer(xFp32TmpBuffer, numCol_ * sizeof(float));
    pipe->InitBuffer(gammaFp32TmpBuffer, numCol_ * sizeof(float));
    pipe->InitBuffer(calcBuffer, BUF_FACTOR * numCol_ * sizeof(float));
    pipe->InitBuffer(keyRopeFp16TmpBuffer, numCol_ * maxRow_ * sizeof(K_DTYPE));
    pipe->InitBuffer(keyRopeFp32TmpBuffer, numCol_ * maxRow_ * sizeof(float));
    pipe->InitBuffer(keyRACFp16TmpBuffer, rACHeadSizeK_ * sizeof(K_DTYPE));
}

template <typename K_DTYPE>
__aicore__ inline float RmsNormAndRopeAndReshapeAndCache<K_DTYPE>::ComputeSliceSquareSum(LocalTensor<float>& in,
    LocalTensor<float>& tmp, LocalTensor<float>& workLocal, uint32_t count)
{
    Mul(tmp, in, in, count);
    PipeBarrier<PIPE_V>();
    ReduceSum(tmp, tmp, workLocal, count);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    return tmp.GetValue(0);
}

template <typename K_DTYPE>
__aicore__ inline void RmsNormAndRopeAndReshapeAndCache<K_DTYPE>::ComputeRmsNormFast(LocalTensor<K_DTYPE>& out,
    LocalTensor<float> in, float rms, LocalTensor<K_DTYPE>& gamma, uint32_t count, LocalTensor<float>& tmp,
    LocalTensor<float>& gammaFp32)
{
    Duplicate(tmp, rms, count);
    PipeBarrier<PIPE_V>();
    Div(tmp, in, tmp, count);
    PipeBarrier<PIPE_V>();
    Mul(in, gammaFp32, tmp, count);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    Cast(out, in, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
}

template <typename K_DTYPE>
__aicore__ inline void RmsNormAndRopeAndReshapeAndCache<K_DTYPE>::ComputeRmsNorm(LocalTensor<K_DTYPE>& RACTokenLocal,
    LocalTensor<K_DTYPE>& gamma, LocalTensor<float>& gammaFp32)
{
    LocalTensor<float> calTask = calcBuffer.Get<float>(OFFSET_CALC * numCol_);
    float rms = sqrt(ComputeSliceSquareSum(xFp32, calTask, calTask, numCol_) * avgFactor_ + epsilon_);
    PipeBarrier<PIPE_V>();
    ComputeRmsNormFast(RACTokenLocal, xFp32, rms, gamma, numCol_, calTask, gammaFp32);
}

template <typename K_DTYPE>
__aicore__ inline void RmsNormAndRopeAndReshapeAndCache<K_DTYPE>::ExpandNeg(LocalTensor<K_DTYPE>& tempBuf,
    uint32_t headNumTemp, uint32_t rowWorkNum)
{
    Duplicate(tempBuf[negOne_], (K_DTYPE)-1, ropeRotateStride);
    Duplicate(tempBuf[ropeRotateStride + negOne_], (K_DTYPE)1, ropeRotateStride);
    PipeBarrier<PIPE_V>();
}

template <typename K_DTYPE>
__aicore__ inline void RmsNormAndRopeAndReshapeAndCache<K_DTYPE>::CosSinBroadcast(LocalTensor<K_DTYPE> &tempBuf,
    uint32_t Calclen, uint32_t rowWorkNum, uint32_t cosSinOffset)
{
    DataCopyParams copyParams = {
        1, static_cast<uint16_t>((ropeHeadDimK_ * rowWorkNum * sizeof(K_DTYPE) + BLK_SIZE - 1) / BLK_SIZE), 0, 0};
    DataCopy(tempBuf[cosPad_], cosGm[cosSinOffset], copyParams);
    DataCopy(tempBuf[sinPad_], sinGm[cosSinOffset], copyParams);
    PipeBarrier<PIPE_MTE2>();
}

template <typename K_DTYPE>
__aicore__ inline void RmsNormAndRopeAndReshapeAndCache<K_DTYPE>::QkComm(const GlobalTensor<K_DTYPE> &src,
    uint32_t hiddenSizeTmp, LocalTensor<K_DTYPE> &tempBuf, uint32_t headNumTemp, uint32_t ropeOffset)
{
    uint16_t hiddenSizeBlk = static_cast<uint16_t>(hiddenSizeTmp * rowWorkNum / ELE_NUM_FP16); // 4
    DataCopy(tempBuf[oriPos_], src[ropeOffset], {1, hiddenSizeBlk, 0, 0}); // 搬运keyRope
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    // {2, 32, 32, 32}  rotaryStrideOffset:32
    DataCopyParams copyParams = {static_cast<uint16_t>(headNumTemp * rowWorkNum),
        static_cast<uint16_t>(rotaryStrideOffset / ELE_NUM_FP16),
        static_cast<uint16_t>(rotaryStrideOffset / ELE_NUM_FP16),
        static_cast<uint16_t>(rotaryStrideOffset / ELE_NUM_FP16)};
    // 旋转keyRope
    DataCopy(tempBuf[removeBefore_ + rotaryStrideOffset], tempBuf[oriPos_], copyParams); // 搬后半段
    DataCopy(tempBuf[removeBefore_], tempBuf[oriPos_ + rotaryStrideOffset], copyParams); // 搬前半段
}

template <typename K_DTYPE>
__aicore__ inline void RmsNormAndRopeAndReshapeAndCache<K_DTYPE>::CalcRopeAlign(LocalTensor<K_DTYPE> &tempBuf,
    uint32_t repeatTimes1, uint32_t oriPosTemp, uint32_t removeTemp, uint32_t padTemp)
{
    Cast(keyRopeFp32, tempBuf, RoundMode::CAST_NONE, numCol_ * maxRow_);
    PipeBarrier<PIPE_V>();
    mul_v<ArchType::ASCEND_V220, float>(keyRopeFp32[oriPosTemp], keyRopeFp32[cosPadTmp], keyRopeFp32[oriPosTemp],
        repeatTimes1, 1, 1, 1, REPEAT_STRIDE, REPEAT_STRIDE, REPEAT_STRIDE); // keyRope*cos
    PipeBarrier<PIPE_V>();
    mul_v<ArchType::ASCEND_V220, float>(keyRopeFp32[removeTemp], keyRopeFp32[negOne_], keyRopeFp32[removeTemp],
        repeatTimes1, 1, 1, 1, REPEAT_STRIDE, REPEAT_STRIDE, REPEAT_STRIDE); // rotateHalf(keyRope)
    PipeBarrier<PIPE_V>();
    mul_v<ArchType::ASCEND_V220, float>(keyRopeFp32[removeTemp], keyRopeFp32[sinPadTmp], keyRopeFp32[removeTemp],
        repeatTimes1, 1, 1, 1, REPEAT_STRIDE, REPEAT_STRIDE, REPEAT_STRIDE); // rotateHalf(keyRope)*sin
    PipeBarrier<PIPE_V>();
    add_v<ArchType::ASCEND_V220, float>(keyRopeFp32[padTemp], keyRopeFp32[removeTemp], keyRopeFp32[oriPosTemp],
        repeatTimes1, 1, 1, 1, REPEAT_STRIDE, REPEAT_STRIDE, REPEAT_STRIDE); // keyRope*cos + rotateHalf(keyRope)*sin
    PipeBarrier<PIPE_V>();
    Cast(RACTokenLocal[numCol_], keyRopeFp32[padTemp], RoundMode::CAST_RINT, ropeHiddenSizeK_);
    PipeBarrier<PIPE_V>();
}
 
 template <typename K_DTYPE>
__aicore__ inline void RmsNormAndRopeAndReshapeAndCache<K_DTYPE>::CopykVCache(LocalTensor<K_DTYPE>& ubAddr,
    GlobalTensor<K_DTYPE>& dst, uint64_t cacheStart, DataCopyParams& copyParamsOut)
{
    DataCopy(dst[cacheStart], ubAddr, copyParamsOut);
    PipeBarrier<PIPE_MTE3>();
}

template <typename K_DTYPE>
__aicore__ inline void RmsNormAndRopeAndReshapeAndCache<K_DTYPE>::Process()
{
    uint32_t loopNums = (rowWork_ + maxRow_ - 1) / maxRow_;
    uint32_t tailLoopRowNums = rowWork_ % maxRow_ == 0 ? maxRow_ : rowWork_ % maxRow_;
    for (uint32_t loopNum = 0; loopNum < loopNums; ++loopNum) {
        rowWorkNum = loopNum == loopNums - 1 ? tailLoopRowNums : maxRow_;
        uint32_t headNumTempK = ropeHiddenSizeK_ / ropeHeadDimK_; // 1
        xFp16 = xFp16TmpBuffer.Get<K_DTYPE>();
        xFp32 = xFp32TmpBuffer.Get<float>();
        gamma = gammaFp16TmpBuffer.Get<K_DTYPE>();
        gammaFp32 = gammaFp32TmpBuffer.Get<float>();
        keyRopeFp16 = keyRopeFp16TmpBuffer.Get<K_DTYPE>();
        keyRopeFp32 = keyRopeFp32TmpBuffer.Get<float>();
        uint32_t repeatTimeOnce = (ropeHiddenSizeK_ + repeatSize - 1) / repeatSize; // 1
        RACTokenLocal = keyRACFp16TmpBuffer.Get<K_DTYPE>();
        rACTokenSizeK_ = rACHeadSizeK_; // 576
        rACNumBlocks_ = rACTokenSizeK_ * sizeof(K_DTYPE) / BLOCK_SIZE; // 2304
        DataCopyParams copyParams = {1, static_cast<uint16_t>(rACNumBlocks_), 0, 0};

        DataCopy(xFp16, xGm[loopNum * maxRow_ * numCol_], numCol_ * rowWorkNum);
        PipeBarrier<PIPE_MTE2>();
        DataCopy(gamma, gammaGm[0], numCol_);
        SetFlag<HardEvent::MTE2_V>(eventIdMTE22VA1);
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE22VA1);
        Cast(gammaFp32, gamma, RoundMode::CAST_NONE, numCol_);

        ExpandNeg(keyRopeFp16, ropeHeadNumK_, rowWorkNum); // 填充-1,1
        uint32_t ropeOffset = loopNum * maxRow_ * ropeHeadDimK_;
        CosSinBroadcast(keyRopeFp16, ropeHiddenSizeK_, rowWorkNum, ropeOffset); // 搬运cos sin
        PipeBarrier<PIPE_MTE2>();
        QkComm(ropeGm, ropeHiddenSizeK_, keyRopeFp16, headNumTempK, ropeOffset);

        for (uint32_t loop = 0; loop < rowWorkNum; ++loop) {
            uint32_t offset = loop * numCol_;
            cosPadTmp = cosPad_ + loop * ropeHeadDimK_;
            sinPadTmp = sinPad_ + loop * ropeHeadDimK_;
            oriPosTmp = oriPos_ + loop * ropeHeadDimK_;
            removeBeforeTmp = removeBefore_ + loop * ropeHeadDimK_;

            uint32_t slotValueOffset = currentCoreIdx * rowWork + loopNum * maxRow_;
            int64_t slotValue = static_cast<int64_t>(slotMappingGm.GetValue(loop + slotValueOffset));
            uint64_t cacheStart = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(rACTokenSizeK_);
            Cast(xFp32, xFp16[offset], RoundMode::CAST_NONE, numCol_);
            PipeBarrier<PIPE_V>();

            ComputeRmsNorm(RACTokenLocal, gamma, gammaFp32); // rmsNorm

            CalcRopeAlign(keyRopeFp16, repeatTimeOnce, oriPosTmp, removeBeforeTmp, padBefore_);

            if (slotValue < 0) {
                continue;
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CopykVCache(RACTokenLocal, keyCacheOutGm, cacheStart, copyParams);
            PipeBarrier<PIPE_ALL>();
        }
    }
}
}

inline __aicore__ void InitTilingData(const __gm__ uint8_t *pTilingdata,
    AtbOps::RmsNormAndRopeAndReshapeAndCacheTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numRow = (*(const __gm__ uint32_t *)(pTilingdata + 0));
    tilingdata->numCol = (*(const __gm__ uint32_t *)(pTilingdata + 4));
    tilingdata->numCore = (*(const __gm__ uint32_t *)(pTilingdata + 8));
    tilingdata->avgFactor = (*(const __gm__ float *)(pTilingdata + 12));
    tilingdata->precisionMode = (*(const __gm__ uint32_t *)(pTilingdata + 16));
    tilingdata->epsilon = (*(const __gm__ float *)(pTilingdata + 20));
    tilingdata->ropeNumTokens = (*(const __gm__ uint32_t *)(pTilingdata + 24));
    tilingdata->ropeHiddenSizeK = (*(const __gm__ uint32_t *)(pTilingdata + 28));
    tilingdata->ropeHeadDimK = (*(const __gm__ uint32_t *)(pTilingdata + 32));
    tilingdata->ropeHeadNumK = (*(const __gm__ uint32_t *)(pTilingdata + 36));
    tilingdata->racNumTokens = (*(const __gm__ uint32_t *)(pTilingdata + 40));
    tilingdata->slotMappingLen = (*(const __gm__ uint32_t *)(pTilingdata + 44));
    tilingdata->racNumHeads = (*(const __gm__ uint32_t *)(pTilingdata + 48));
    tilingdata->racHeadSizeK = (*(const __gm__ uint32_t *)(pTilingdata + 52));
    tilingdata->rotaryCoeff = (*(const __gm__ uint32_t *)(pTilingdata + 56));
    tilingdata->maxRow = (*(const __gm__ uint32_t *)(pTilingdata + 60));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdataInUb = nullptr;
    CopyGmTilingToUb(tilingdataInUb, pTilingdata, sizeof(AsdOps::RmsNormAndRopeAndReshapeAndCacheTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numRow = (*(__ubuf__ uint32_t *)(tilingdataInUb + 0));
    tilingdata->numCol = (*(__ubuf__ uint32_t *)(tilingdataInUb + 4));
    tilingdata->numCore = (*(__ubuf__ uint32_t *)(tilingdataInUb + 8));
    tilingdata->avgFactor = (*(__ubuf__ float *)(tilingdataInUb + 12));
    tilingdata->precisionMode = (*(__ubuf__ uint32_t *)(tilingdataInUb + 16));
    tilingdata->epsilon = (*(__ubuf__ float *)(tilingdataInUb + 20));
    tilingdata->ropeNumTokens = (*(__ubuf__ uint32_t *)(tilingdataInUb + 24));
    tilingdata->ropeHiddenSizeK = (*(__ubuf__ uint32_t *)(tilingdataInUb + 28));
    tilingdata->ropeHeadDimK = (*(__ubuf__ uint32_t *)(tilingdataInUb + 32));
    tilingdata->ropeHeadNumK = (*(__ubuf__ uint32_t *)(tilingdataInUb + 36));
    tilingdata->racNumTokens = (*(__ubuf__ uint32_t *)(tilingdataInUb + 40));
    tilingdata->slotMappingLen = (*(__ubuf__ uint32_t *)(tilingdataInUb + 44));
    tilingdata->racNumHeads = (*(__ubuf__ uint32_t *)(tilingdataInUb + 48));
    tilingdata->racHeadSizeK = (*(__ubuf__ uint32_t *)(tilingdataInUb + 52));
    tilingdata->rotaryCoeff = (*(__ubuf__ uint32_t *)(tilingdataInUb + 56));
    tilingdata->maxRow = (*(__ubuf__ uint32_t *)(tilingdataInUb + 60));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}
 
#define GET_TILING_DATA(tilingData, tilingArg)                                                                       \
    AtbOps::RmsNormAndRopeAndReshapeAndCacheTilingData tilingData;                                                   \
    InitTilingData(tilingArg, &(tilingData))
 
extern "C" __global__ __aicore__ void rms_norm_and_rope_and_reshape_and_cache(GM_ADDR x, GM_ADDR gamma,
    GM_ADDR keyRope, GM_ADDR cos, GM_ADDR sin, GM_ADDR slotMapping, GM_ADDR keycachein, GM_ADDR keycacheout,
    GM_ADDR tiling)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    PRELOAD(2);
#endif
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(101)) {
        RmsNormAndRopeAndReshapeAndCacheFusion::RmsNormAndRopeAndReshapeAndCache<half> op;
        op.Init(&pipe, x, gamma, keyRope, cos, sin, slotMapping, keycachein, keycacheout, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(102)) {
        RmsNormAndRopeAndReshapeAndCacheFusion::RmsNormAndRopeAndReshapeAndCache<bfloat16_t> op;
        op.Init(&pipe, x, gamma, keyRope, cos, sin, slotMapping, keycachein, keycacheout, tilingData);
        op.Process();
    }
}