/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mla_preprocess_tiling.h"
#include "atbops/params/params.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "mki/utils/math/math.h"
#include "mki/utils/platform/platform_info.h"
#include "mla_preprocess_tilingdata.h"

#include <cmath>
#include <string>

constexpr uint32_t AXES_ALIGN_SIZE = 512;
constexpr uint32_t BASE_BLOCK_STEP = 2;
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t CONST_32 = 32;
constexpr uint32_t CONST_128 = 128;
constexpr uint32_t CONST_256 = 256;
constexpr uint32_t CONST_512 = 512;
constexpr uint32_t L1_BUFFER_SIZE = 524288;
constexpr uint32_t L1_PINGPONG_BUFFER_LEN = 262144;
constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN = 131072;
constexpr uint32_t L1_SCALE_SIZE = 4096;
constexpr uint32_t L1_BIAS_SIZE = 2048;
constexpr uint32_t L0C_SIZE = 128 * 1024;
constexpr uint32_t CONCAT_SIZE = 512;
constexpr uint32_t HIDDEN_STRATE = 7168;
constexpr uint32_t HIDDEN_STRATE_ROPE = 192;
constexpr uint32_t HIDDEN_STRATE_MM = 2112;
constexpr uint32_t HIDDEN_STRATE_RMS = 1536;
constexpr uint32_t UB_SIZE = 196352;
constexpr uint32_t HEADDIM = 64;
constexpr uint32_t FP32_REPEAT_MASK = 64;
constexpr uint32_t FP16_REPEAT_MASK = 128;

const int32_t NUM1 = 1;
const int32_t NUM2 = 2;
const int32_t NUM3 = 3;
const int32_t NUM4 = 4;
const int32_t NUM8 = 8;
constexpr uint32_t INDEX_WDQKV = 5;
constexpr uint32_t INDEX_WUQ = 18;
constexpr uint32_t INDEX_WUK = 20;

inline uint32_t CeilDiv(const uint32_t dividend, const uint32_t divisor)
{
    if (divisor == 0) {
        return UINT32_MAX;
    }
    return (dividend + divisor - 1) / divisor;
}

inline uint32_t RoundUp(const uint32_t val, const uint32_t align = 16)
{
    if (align == 0) {
        return 0;
    }
    return (val + align - 1) / align * align;
}

inline uint32_t RoundDown(const uint32_t val, const uint32_t align = 16)
{
    if (align == 0) {
        return 0;
    }
    return val / align * align;
}

template <typename T = uint32_t> inline T Max(const T a, const T b) { return a > b ? a : b; }

template <typename T = uint32_t> inline T Min(const T a, const T b) { return a < b ? a : b; }

namespace AtbOps {

using namespace Mki;
using QuantMode = OpParam::MlaPreprocess::QuantMode;
class PpMatmulTilingApi {
public:
    PpMatmulTilingApi(uint32_t numBatch, uint32_t m, uint32_t k, uint32_t n, bool transA, bool transB, bool enDequant,
        bool deqOnTheFly)
        : numBatch_(numBatch), m_(m), k_(k), n_(n), transA_(transA), transB_(transB), enDequant_(enDequant),
          deqOnTheFly_(deqOnTheFly)
    {
        inDataSize_ = enDequant ? sizeof(uint8_t) : sizeof(uint16_t);
    }
    void GetTilingData(AtbOps::PpMatmulTilingData &tiling);

private:
    void GetTileSize();
    float GetCost(const uint32_t m0, const uint32_t n0) const;
    void UpdateTileSize(const uint32_t m0, const uint32_t n0);
    void Swizzle();
    uint32_t ComputeL1AbSize() const;
    uint32_t ComputeK0ForABpingpong(uint32_t l1AbSize) const;
    bool IsLoadAllAmat(uint32_t l1AbSize) const;
    uint32_t ComputeK0ForOnlyBpingpong(uint32_t l1AbSize);

private:
    uint32_t numBatch_{0};
    uint32_t m_{0};
    uint32_t k_{0};
    uint32_t n_{0};
    uint32_t m0_{0};
    uint32_t k0_{0};
    uint32_t n0_{0};
    uint32_t mLoop_{0};
    uint32_t kLoop_{0};
    uint32_t nLoop_{0};
    uint32_t coreLoop_{0};
    uint32_t swizzleCount_{0};
    uint32_t blockDim_{0};
    uint32_t swizzleDirect_{0};
    uint32_t inDataSize_{0};
    uint32_t b0matPingPongBufferLen_{L1_PINGPONG_BUFFER_LEN};
    bool transA_{false};
    bool transB_{false};
    bool enDequant_{false};
    bool enShuffleK_{false};
    bool enLoadAllAmat_{false};
    bool deqOnTheFly_{false};
};

void PpMatmulTilingApi::GetTilingData(AtbOps::PpMatmulTilingData &tiling)
{
    GetTileSize();
    tiling.numBatch = numBatch_;
    tiling.m = m_;
    tiling.k = k_;
    tiling.n = n_;
    tiling.m0 = m0_;
    tiling.k0 = k0_;
    tiling.n0 = n0_;
    tiling.mLoop = mLoop_;
    tiling.kLoop = kLoop_;
    tiling.nLoop = nLoop_;
    tiling.coreLoop = coreLoop_;
    tiling.swizzleCount = swizzleCount_;
    tiling.swizzleDirect = swizzleDirect_;
    tiling.enShuffleK = static_cast<uint32_t>(enShuffleK_);
    tiling.blockDim = blockDim_;
    tiling.enLoadAllAmat = static_cast<uint32_t>(enLoadAllAmat_);
    tiling.b0matPingPongBufferLen = b0matPingPongBufferLen_;
}

void PpMatmulTilingApi::GetTileSize()
{
    bool priFlag = !(m_ < n_);
    uint32_t roundBase = pow(2, ceil(log(CeilDiv(priFlag ? n_ : m_, CONST_16)))) * CONST_16;
    uint32_t priAxes = RoundUp(priFlag ? m_ : n_, CONST_16);
    uint32_t subAxes = RoundUp(priFlag ? n_ : m_, roundBase);
    float minCost = __FLT_MAX__;
    uint32_t maxAxes0 = AXES_ALIGN_SIZE;
    uint32_t maxPriAxes0 = Min(maxAxes0, priAxes);
    uint32_t maxSubAxes0 = Min(maxAxes0, subAxes);
    for (uint32_t priAxes0 = CONST_16; priAxes0 <= maxPriAxes0; priAxes0 *= BASE_BLOCK_STEP) {
        for (uint32_t subAxes0 = CONST_16; subAxes0 <= maxSubAxes0; subAxes0 *= BASE_BLOCK_STEP) {
            if (priAxes0 * subAxes0 * sizeof(float) > PlatformInfo::Instance().GetL0CSize()) {
                continue;
            }
            uint32_t newM0 = priFlag ? priAxes0 : subAxes0;
            uint32_t newN0 = priFlag ? subAxes0 : priAxes0;
            if (newN0 > CONST_256 && enDequant_) {
                continue;
            }
            float cost = GetCost(newM0, newN0);
            if (cost < minCost) {
                minCost = cost;
                UpdateTileSize(newM0, newN0);
            }
        }
    }

    Swizzle();

    uint32_t l1AbSize = ComputeL1AbSize();
    k0_ = ComputeK0ForABpingpong(l1AbSize);
    kLoop_ = CeilDiv(k_, k0_);
    // 对于MM1和MM2, 如果一个核一轮跑不完, 选择全载A, 并更新k0
    if (0) { // IsLoadAllAmat(l1AbSize)
        k0_ = ComputeK0ForOnlyBpingpong(l1AbSize);
        kLoop_ = CeilDiv(k_, k0_);
    }
}

uint32_t PpMatmulTilingApi::ComputeK0ForOnlyBpingpong(uint32_t l1AbSize)
{
    enLoadAllAmat_ = true;
    b0matPingPongBufferLen_ = static_cast<uint32_t>(
        static_cast<float>((l1AbSize - RoundUp(m_, CONST_16) * RoundUp(k_, CONST_32) * inDataSize_) / DIM_2));
    uint32_t k0MaxB0 =
        static_cast<uint32_t>(static_cast<float>(b0matPingPongBufferLen_ / (RoundUp(n0_, CONST_16) * inDataSize_)));
    uint32_t k0B0 = k0MaxB0 < CONST_512 ? RoundDown(k0MaxB0, CONST_32) : RoundDown(k0MaxB0, CONST_512);
    return k0B0 > CONST_512 ? RoundDown(k0B0, CONST_512) : k0B0;
}

bool PpMatmulTilingApi::IsLoadAllAmat(uint32_t l1AbSize) const
{
    return (coreLoop_ > blockDim_) && enDequant_ && (kLoop_ > 1) &&
           (l1AbSize > RoundUp(m_, CONST_16) * RoundUp(k_, CONST_32) * inDataSize_) && (mLoop_ == 1);
}

uint32_t PpMatmulTilingApi::ComputeK0ForABpingpong(uint32_t l1AbSize) const
{
    uint32_t k0Max = static_cast<uint32_t>(static_cast<float>(l1AbSize / DIM_2) / ((m0_ + n0_) * inDataSize_));
    uint32_t tmpK0;
    if (enDequant_) {
        tmpK0 = k0Max < CONST_512 ? RoundDown(k0Max, CONST_32) : RoundDown(k0Max, CONST_512);
    } else {
        tmpK0 = k0Max < CONST_256 ? RoundDown(k0Max, CONST_16) : RoundDown(k0Max, CONST_256);
    }
    if (tmpK0 > CONST_512) {
        tmpK0 = RoundDown(tmpK0, CONST_512);
    }
    return tmpK0;
}

uint32_t PpMatmulTilingApi::ComputeL1AbSize() const
{
    if (enDequant_ && deqOnTheFly_) {
        return L1_BUFFER_SIZE;
    }
    return enDequant_ ? (L1_BUFFER_SIZE - L1_BIAS_SIZE - L1_SCALE_SIZE) : L1_BUFFER_SIZE;
}

float PpMatmulTilingApi::GetCost(const uint32_t m0, const uint32_t n0) const
{
    float aCoef = 1.0;
    float bCoef = 1.0;
    float bwCoef = 5.0;
    uint32_t mLoop = CeilDiv(m_, m0);
    uint32_t nLoop = CeilDiv(n_, n0);
    if (mLoop == 0 || nLoop == 0) {
        return __FLT_MAX__;
    }
    uint32_t rqdNumCore = numBatch_ * mLoop * nLoop;
    uint32_t blockDim = Min(rqdNumCore, PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));
    uint32_t mOnce = blockDim < nLoop ? m0 : blockDim / nLoop * m0;
    uint32_t nOnce = blockDim < nLoop ? PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE) * n0 : n_;
    if (mOnce * k_ * sizeof(uint16_t) > PlatformInfo::Instance().GetL2Size()) {
        aCoef = bwCoef;
    }
    if (nOnce * k_ * sizeof(uint16_t) > PlatformInfo::Instance().GetL2Size()) {
        bCoef = bwCoef;
    }
    if (transA_ && m0 % CONST_256 == 0) {
        aCoef *= NUM2;
    }
    if (!transB_ && n0 % CONST_256 == 0) {
        bCoef *= NUM2;
    }
    return 1 / (aCoef * static_cast<float>(n0)) + 1 / (bCoef * static_cast<float>(m0));
}

void PpMatmulTilingApi::UpdateTileSize(const uint32_t m0, const uint32_t n0)
{
    m0_ = m0;
    n0_ = n0;
    mLoop_ = CeilDiv(m_, m0_);
    nLoop_ = CeilDiv(n_, n0_);
    coreLoop_ = numBatch_ * mLoop_ * nLoop_;
    const uint32_t maxNumCubeCore = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (mLoop_ == 1 && transB_ && coreLoop_ % maxNumCubeCore < maxNumCubeCore / NUM4 * NUM3) {
        uint32_t tmpM0 = RoundUp(m_, CONST_16);
        uint32_t maxN0 = L0C_SIZE / (tmpM0 * sizeof(float));
        if (enDequant_) {
            maxN0 = maxN0 < CONST_256 ? maxN0 : CONST_256;
        }
        uint32_t x = CeilDiv(n_, maxNumCubeCore);
        uint32_t y = CeilDiv(x, maxN0);
        uint32_t tmpN0 = RoundUp(CeilDiv(x, y), CONST_16);
        uint32_t rqdL0cSize = tmpM0 * tmpN0 * sizeof(float);
        if (rqdL0cSize < L0C_SIZE && (tmpM0 + tmpN0) * CONST_256 * inDataSize_ < L1_BUFFER_SIZE) {
            m0_ = tmpM0;
            n0_ = tmpN0;
            nLoop_ = CeilDiv(n_, n0_);
            coreLoop_ = numBatch_ * nLoop_;
        }
    }
    blockDim_ = Min(coreLoop_, maxNumCubeCore);
}

void PpMatmulTilingApi::Swizzle()
{
    float minCost = m_ * k_ + k_ * n_;
    for (uint32_t i = 1; i <= blockDim_; ++i) {
        int c = static_cast<int32_t>((blockDim_ + i - 1) / i);
        float cost;
        // B0 + A < A0 + B
        if (i * n0_ + m_ < m0_ * c + n_) {
            swizzleDirect_ = 1; // Nz
            cost = n0_ * i + m0_ * c;
            if (cost <= minCost) {
                minCost = cost;
                swizzleCount_ = i;
            }
        } else {
            swizzleDirect_ = 0; // Zn
            cost = m0_ * i + n0_ * c;
            if (cost < minCost) {
                minCost = cost;
                swizzleCount_ = i;
            }
        }
    }
}

class MlaPreprocessTiling {
public:
    AtbOps::MlaTilingData tilingData;
    Mki::Status Init(const Mki::LaunchParam &launchParam, Mki::KernelInfo &kernelInfo);
    void RmsNormQuantTiling(const uint32_t numTokens);
    void RopeConcatTiling(const OpParam::MlaPreprocess &param, const uint32_t &aicNum);
    void EinSumQuantTiling(const OpParam::MlaPreprocess &param, const uint32_t &aicNum, const TensorDType inDtype);
    void SetTiling(AtbOps::MlaTilingData &tilingParam) const;
    void SetTilingKey(const Mki::LaunchParam &launchParam, Mki::KernelInfo &kernelInfo) const;
    void SetMlapoWorkSpace(const TensorDType inDtype, const OpParam::MlaPreprocess &param, Mki::KernelInfo &kernelInfo);
};

void MlaPreprocessTiling::RmsNormQuantTiling(const uint32_t numTokens)
{
    const uint32_t numVectorCore = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    tilingData.rmsNumCore1 = numVectorCore;
    tilingData.rmsNumCol1 = HIDDEN_STRATE;
    tilingData.rmsNumRow1 = numTokens;
    tilingData.rmsQuantMin1 = -CONST_128;
    tilingData.rmsNumCore2 = numVectorCore;
    tilingData.rmsNumCol2 = HIDDEN_STRATE_MM;
    tilingData.rmsNumRow2 = numTokens;
    tilingData.rmsQuantMin2 = -CONST_128;
}

void MlaPreprocessTiling::RopeConcatTiling(const OpParam::MlaPreprocess &param, const uint32_t &aicNum)
{
    uint32_t ntokens = param.N;
    uint32_t hiddenSizeQ = HEADDIM * param.headNum;
    uint32_t headDim = HEADDIM;
    uint32_t headNumQ = hiddenSizeQ / headDim;
    uint32_t concatSize = CONCAT_SIZE;
    uint32_t maxCore = aicNum * 2;
    uint32_t maxUbSize = UB_SIZE;
    uint32_t allHeadNum = ntokens * headNumQ;

    uint32_t tempCore = (allHeadNum + maxCore - 1) / maxCore;
    uint32_t realCore = (allHeadNum + tempCore - 1) / tempCore;  // 实际运算核数
    uint32_t nlCoreRun = (allHeadNum + realCore - 1) / realCore; // 前核运算head数
    uint32_t lCoreRun = allHeadNum - (realCore - 1) * nlCoreRun; // 尾核运算head数

    uint32_t dataTypeSize = 2;

    // 计算一次能搬几行 q 4+2、reverseq 4、neg 4、sin 4+2、cos 4+2  + concat 2
    uint32_t allSize = headDim * (3 * (4 + dataTypeSize) + 2 * 4) + concatSize * dataTypeSize; // rope内部升精度计算
    uint32_t maxNPerLoopForUb = maxUbSize / allSize; // ub每次能载入最大行数（包括所有计算数据）
    uint32_t preCoreLoopTime = (nlCoreRun + maxNPerLoopForUb - 1) / maxNPerLoopForUb;  // 前核循环次数
    uint32_t preCoreLoopNLast = nlCoreRun - (preCoreLoopTime - 1) * maxNPerLoopForUb;  // 前核最后一批处理数据行数
    uint32_t lastCoreLoopTime = (lCoreRun + maxNPerLoopForUb - 1) / maxNPerLoopForUb;  // 尾核循环次数
    uint32_t lastCoreLoopNLast = lCoreRun - (lastCoreLoopTime - 1) * maxNPerLoopForUb; // 尾核最后一批处理数据行数

    tilingData.hiddenSizeQ = hiddenSizeQ;
    tilingData.headNumQ = headNumQ;
    tilingData.headDim = headDim;
    tilingData.concatSize = concatSize;
    tilingData.rotaryCoeff = NUM2;
    tilingData.ntokens = ntokens;
    tilingData.realCore = realCore;
    tilingData.nlCoreRun = nlCoreRun;
    tilingData.lCoreRun = nlCoreRun;
    tilingData.maxNPerLoopForUb = maxNPerLoopForUb;
    tilingData.preCoreLoopTime = preCoreLoopTime;
    tilingData.preCoreLoopNLast = preCoreLoopNLast;
    tilingData.lastCoreLoopTime = lastCoreLoopTime;
    tilingData.lastCoreLoopNLast = lastCoreLoopNLast;
}

void MlaPreprocessTiling::EinSumQuantTiling(const OpParam::MlaPreprocess &param, const uint32_t &aicNum,
                                            const TensorDType inDtype)
{
    uint32_t aivCore = aicNum * 2;
    uint32_t ubSize = UB_SIZE - 1024;

    // input shape
    uint32_t esqBatch = param.N;          // tokenNum
    uint32_t esqHeadNum = param.headNum;  // headNum
    uint32_t esqColNum = AXES_ALIGN_SIZE; // 512

    // split core
    uint32_t esqFrontCore = esqBatch % aivCore;
    uint32_t esqTailCore = aivCore - esqFrontCore;
    uint32_t esqFrontCoreBatch = CeilDiv(esqBatch, aivCore);
    uint32_t esqTailCoreBatch = esqBatch / aivCore;

    // split ub --> calc H' <-- 一次ub循环中搬运处理的行数
    uint32_t splitFactor = 0;
    uint32_t esqHeadPerLoop = 0; // ub每次计算的head行数
    uint32_t repeatMask = 0;

    if (inDtype == TENSOR_DTYPE_BF16 || param.quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
        // 将scale一次性搬入、广播、缓存 H * 32bytes
        uint32_t scaleUb = RoundUp(esqHeadNum) * CONST_32;
        // bf16 input [H', colNum](f16 + fp32 + int8), ub reuse
        splitFactor = esqColNum * (sizeof(uint16_t) + sizeof(float) + sizeof(uint8_t));
        splitFactor *= NUM2;
        esqHeadPerLoop = (ubSize - scaleUb) / splitFactor;  // 26
        repeatMask = FP32_REPEAT_MASK;
    } else {
        // fp16 input [H', cloNum](fp16*2 + int8) + [H', 1](fp16) + [H', 16](fp16)
        splitFactor =
            esqColNum * (NUM2 * sizeof(uint16_t) + sizeof(uint8_t)) + sizeof(uint16_t) + (CONST_16 * sizeof(uint16_t));
        esqHeadPerLoop = ubSize / splitFactor;
        repeatMask = FP16_REPEAT_MASK;
        esqHeadPerLoop = RoundDown(esqHeadPerLoop);           // 向下16对齐
    }
    uint32_t esqUbHeadLoop = esqHeadNum / esqHeadPerLoop; // ub完整循环次数
    uint32_t esqHeadTail = esqHeadNum % esqHeadPerLoop;   // ub最后一次处理head的行数
    uint32_t esqColLoop = esqColNum / repeatMask;         // 每行按列计算要循环处理的次数
    uint32_t esqColTail = esqColNum % repeatMask;         // colNum非64/128对齐时，最后一次计算列数

    tilingData.esqFrontCore = esqFrontCore;
    tilingData.esqTailCore = esqTailCore;
    tilingData.esqFrontCoreBatch = esqFrontCoreBatch;
    tilingData.esqTailCoreBatch = esqTailCoreBatch;
    tilingData.esqHeadNum = esqHeadNum;
    tilingData.esqColNum = esqColNum;
    tilingData.esqUbHeadLoop = esqUbHeadLoop;
    tilingData.esqHeadPerLoop = esqHeadPerLoop;
    tilingData.esqHeadTail = esqHeadTail;
    tilingData.esqColLoop = esqColLoop;
    tilingData.esqColTail = esqColTail;
}

void MlaPreprocessTiling::SetTiling(AtbOps::MlaTilingData &tilingParam) const
{
    tilingParam.n = tilingData.n;
    tilingParam.perTaskNum = tilingData.perTaskNum;
    tilingParam.resTaskNum = tilingData.resTaskNum;
    tilingParam.numCore = tilingData.numCore;

    tilingParam.rmsNumCore1 = tilingData.rmsNumCore1;
    tilingParam.rmsNumCol1 = tilingData.rmsNumCol1;

    tilingParam.rmsNumCore2 = tilingData.rmsNumCore2;
    tilingParam.rmsNumCol2 = tilingData.rmsNumCol2;

    tilingParam.hiddenSizeQ = tilingData.hiddenSizeQ;
    tilingParam.headNumQ = tilingData.headNumQ;
    tilingParam.headDim = tilingData.headDim;
    tilingParam.concatSize = tilingData.concatSize;
    tilingParam.rotaryCoeff = tilingData.rotaryCoeff;
    tilingParam.ntokens = tilingData.ntokens;
    tilingParam.realCore = tilingData.realCore;
    tilingParam.nlCoreRun = tilingData.nlCoreRun;
    tilingParam.lCoreRun = tilingData.lCoreRun;
    tilingParam.maxNPerLoopForUb = tilingData.maxNPerLoopForUb;
    tilingParam.preCoreLoopTime = tilingData.preCoreLoopTime;
    tilingParam.preCoreLoopNLast = tilingData.preCoreLoopNLast;
    tilingParam.lastCoreLoopTime = tilingData.lastCoreLoopTime;
    tilingParam.lastCoreLoopNLast = tilingData.lastCoreLoopNLast;

    tilingParam.esqFrontCore = tilingData.esqFrontCore;
    tilingParam.esqTailCore = tilingData.esqTailCore;
    tilingParam.esqFrontCoreBatch = tilingData.esqFrontCoreBatch;
    tilingParam.esqTailCoreBatch = tilingData.esqTailCoreBatch;
    tilingParam.esqHeadNum = tilingData.esqHeadNum;
    tilingParam.esqColNum = tilingData.esqColNum;
    tilingParam.esqUbHeadLoop = tilingData.esqUbHeadLoop;
    tilingParam.esqHeadPerLoop = tilingData.esqHeadPerLoop;
    tilingParam.esqHeadTail = tilingData.esqHeadTail;
    tilingParam.esqColLoop = tilingData.esqColLoop;
    tilingParam.esqColTail = tilingData.esqColTail;
}

void MlaPreprocessTiling::SetTilingKey(const Mki::LaunchParam &launchParam, Mki::KernelInfo &kernelInfo) const
{
    OpParam::MlaPreprocess param = AnyCast<OpParam::MlaPreprocess>(launchParam.GetParam());
    TensorFormat formatWeight1 = launchParam.GetInTensor(INDEX_WDQKV).desc.format;
    TensorFormat formatWeight2 = launchParam.GetInTensor(INDEX_WUQ).desc.format;
    TensorFormat formatWeight3 = launchParam.GetInTensor(INDEX_WUK).desc.format;
    uint64_t tilingKey = static_cast<uint64_t>(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16);
    tilingKey = (tilingKey << 2) + static_cast<uint64_t>(param.cacheMode); // 2bit for cacheMode.
    tilingKey = (tilingKey << 1) + static_cast<uint64_t>(formatWeight1 == TENSOR_FORMAT_FRACTAL_NZ);
    tilingKey = (tilingKey << 1) + static_cast<uint64_t>(formatWeight2 == TENSOR_FORMAT_FRACTAL_NZ);
    tilingKey = (tilingKey << 1) + static_cast<uint64_t>(formatWeight3 == TENSOR_FORMAT_FRACTAL_NZ);
    tilingKey = (tilingKey << 2) + static_cast<uint64_t>(param.quantMode); // 2bit for quantMode.
    kernelInfo.SetTilingId(tilingKey);
    MKI_LOG_INFO << "tilingKey = " << tilingKey;
}

std::ostream &operator<<(std::ostream &os, const PpMatmulTilingData &tilingData)
{
    os << "(bSize, mSize, kSize, nSize) = (" << tilingData.numBatch << ", " << tilingData.m << ", " << tilingData.k
       << ", " << tilingData.n << ")" << std::endl;
    os << "(m0, k0, n0) = (" << tilingData.m0 << ", " << tilingData.k0 << ", " << tilingData.n0 << ")" << std::endl;
    os << "(mLoop, kLoop, nLoop) = (" << tilingData.mLoop << ", " << tilingData.kLoop << ", " << tilingData.nLoop << ")"
       << std::endl;
    os << "coreLoop = " << tilingData.coreLoop << std::endl;
    os << "(SwizzleDirect, SwizzleCount) = (" << tilingData.swizzleDirect << ", " << tilingData.swizzleCount << ")"
       << std::endl;
    os << "blockDim = " << tilingData.blockDim << std::endl;
    return os;
}

std::ostream &operator<<(std::ostream &os, const MlaTilingData &tilingData)
{
    os << "tiling data:" << std::endl;
    os << "numCore = " << tilingData.numCore << std::endl;
    os << "n = " << tilingData.n << std::endl;
    os << "perTaskNum = " << tilingData.perTaskNum << std::endl;
    os << "resTaskNum = " << tilingData.resTaskNum << std::endl;
    os << tilingData.mm1 << std::endl;
    os << tilingData.mm2 << std::endl;
    os << tilingData.mm3 << std::endl;
    os << "rmsNumCore1 = " << tilingData.rmsNumCore1 << std::endl;
    os << "rmsNumCol1 = " << tilingData.rmsNumCol1 << std::endl;
    os << "rmsNumRow1 = " << tilingData.rmsNumRow1 << std::endl;
    os << "rmsQuantMin1 = " << tilingData.rmsQuantMin1 << std::endl;
    os << "rmsNumCore2 = " << tilingData.rmsNumCore2 << std::endl;
    os << "rmsNumCol2 = " << tilingData.rmsNumCol2 << std::endl;
    os << "rmsNumRow2 = " << tilingData.rmsNumRow2 << std::endl;
    os << "rmsQuantMin2 = " << tilingData.rmsQuantMin2 << std::endl;
    os << "hiddenSizeQ = " << tilingData.hiddenSizeQ << std::endl;
    os << "headNumQ = " << tilingData.headNumQ << std::endl;
    os << "headDim = " << tilingData.headDim << std::endl;
    os << "concatSize = " << tilingData.concatSize << std::endl;
    os << "rotaryCoeff = " << tilingData.rotaryCoeff << std::endl;
    os << "ntokens = " << tilingData.ntokens << std::endl;
    os << "realCore = " << tilingData.realCore << std::endl;
    os << "nlCoreRun = " << tilingData.nlCoreRun << std::endl;
    os << "lCoreRun = " << tilingData.lCoreRun << std::endl;
    os << "maxNPerLoopForUb = " << tilingData.maxNPerLoopForUb << std::endl;
    os << "preCoreLoopTime = " << tilingData.preCoreLoopTime << std::endl;
    os << "preCoreLoopNLast = " << tilingData.preCoreLoopNLast << std::endl;
    os << "lastCoreLoopTime = " << tilingData.lastCoreLoopTime << std::endl;
    os << "lastCoreLoopNLast = " << tilingData.lastCoreLoopNLast << std::endl;
    os << "esqFrontCore      = " << tilingData.esqFrontCore << std::endl;
    os << "esqTailCore       = " << tilingData.esqTailCore << std::endl;
    os << "esqFrontCoreBatch = " << tilingData.esqFrontCoreBatch << std::endl;
    os << "esqTailCoreBatch  = " << tilingData.esqTailCoreBatch << std::endl;
    os << "esqHeadNum        = " << tilingData.esqHeadNum << std::endl;
    os << "esqColNum         = " << tilingData.esqColNum << std::endl;
    os << "esqUbHeadLoop     = " << tilingData.esqUbHeadLoop << std::endl;
    os << "esqHeadPerLoop    = " << tilingData.esqHeadPerLoop << std::endl;
    os << "esqHeadTail       = " << tilingData.esqHeadTail << std::endl;
    os << "esqColLoop        = " << tilingData.esqColLoop << std::endl;
    os << "esqColTail        = " << tilingData.esqColTail << std::endl;
    return os;
}

void MlaPreprocessTiling::SetMlapoWorkSpace(const TensorDType inDtype, const OpParam::MlaPreprocess &param,
                                            Mki::KernelInfo &kernelInfo)
{
    uint64_t s1wsFactor = static_cast<uint64_t>(
        param.cacheMode == 2 ?
        std::max(HIDDEN_STRATE * sizeof(int8_t), param.headNum * AXES_ALIGN_SIZE * sizeof(uint16_t)) :
        HIDDEN_STRATE * sizeof(int8_t)
    );
    uint64_t workSizeS1 = static_cast<uint64_t>(tilingData.n) * s1wsFactor;
    uint64_t workSizeS2 = static_cast<uint64_t>(tilingData.n) * param.headNum * HIDDEN_STRATE_ROPE * sizeof(uint16_t);
    uint64_t workSizeS3 = static_cast<uint64_t>(tilingData.n) * HIDDEN_STRATE_MM * sizeof(uint16_t);
    uint64_t workSizeS4 = static_cast<uint64_t>(tilingData.n) *
                          std::max(param.headNum * HIDDEN_STRATE_ROPE, HIDDEN_STRATE_MM) * sizeof(uint32_t);
    uint64_t pertokenWorkspace = static_cast<uint64_t>(tilingData.n) * sizeof(float) * 2;
    uint64_t maxWorkspaceSize = 0;
    maxWorkspaceSize = std::max(maxWorkspaceSize, workSizeS1);
    maxWorkspaceSize = std::max(maxWorkspaceSize, workSizeS2);
    maxWorkspaceSize = std::max(maxWorkspaceSize, workSizeS3);
    maxWorkspaceSize = std::max(maxWorkspaceSize, workSizeS4);
    if (inDtype == TENSOR_DTYPE_BF16 || param.quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
        kernelInfo.GetScratchSizes() = {
            maxWorkspaceSize, maxWorkspaceSize, maxWorkspaceSize, maxWorkspaceSize, pertokenWorkspace};
    } else {
        kernelInfo.GetScratchSizes() = {
            maxWorkspaceSize, maxWorkspaceSize, maxWorkspaceSize};
    }
}

Mki::Status MlaPreprocessTiling::Init(const Mki::LaunchParam &launchParam, Mki::KernelInfo &kernelInfo)
{
    OpParam::MlaPreprocess param = AnyCast<OpParam::MlaPreprocess>(launchParam.GetParam());
    auto inDtype = launchParam.GetInTensor(0).desc.dtype;
    const uint32_t &aicNum = Mki::PlatformInfo::Instance().GetCoreNum(Mki::CoreType::CORE_TYPE_CUBE);
    tilingData.n = param.N;
    tilingData.numCore = aicNum;
    bool deqOnTheFly = false;
    if (inDtype == TENSOR_DTYPE_BF16 || param.quantMode == QuantMode::PER_TOKEN_SYMM_QUANT) {
        deqOnTheFly = true;
    }
    MKI_LOG(INFO) << "tilingSize: " << kernelInfo.GetTilingSize();
    auto tilingParam = reinterpret_cast<AtbOps::MlaTilingData *>(kernelInfo.GetTilingHostAddr());
    RmsNormQuantTiling(param.N);
    RopeConcatTiling(param, aicNum);
    EinSumQuantTiling(param, aicNum, inDtype);
    PpMatmulTilingApi mm1TilingApi(1,  // numBatch
        param.N,                       // m
        HIDDEN_STRATE,                 // k
        HIDDEN_STRATE_MM,              // n
        false,                         // transA
        true,                          // transB
        true,                          // enDequant
        deqOnTheFly);                     // in bf16.cce?
    mm1TilingApi.GetTilingData(tilingParam->mm1);
    PpMatmulTilingApi mm2TilingApi(1,        // numBatch
        param.N,                             // m
        HIDDEN_STRATE_RMS,                   // k
        param.headNum * HIDDEN_STRATE_ROPE,  // n
        false,                               // transA
        true,                                // transB
        true,                                // enDequant
        deqOnTheFly);                           // in bf16.cce?
    mm2TilingApi.GetTilingData(tilingParam->mm2);
    PpMatmulTilingApi mm3TilingApi(param.headNum,  // numBatch
        param.N,                                   // m
        CONST_128,                                 // k
        CONCAT_SIZE,                               // n
        false,                                     // transA
        false,                                     // transB
        false,                                     // enDequant
        deqOnTheFly);                                 // in bf16.cce?
    mm3TilingApi.GetTilingData(tilingParam->mm3);
    SetTiling(*tilingParam);
    MKI_LOG(INFO) << *tilingParam;
    SetMlapoWorkSpace(inDtype, param, kernelInfo);
    kernelInfo.SetBlockDim(aicNum);
    SetTilingKey(launchParam, kernelInfo);
    return Mki::Status::OkStatus();
}

Mki::Status GetMLAProprecessTiling(const Mki::LaunchParam &launchParam, Mki::KernelInfo &kernelInfo)
{
    MlaPreprocessTiling mlaTiling;
    MKI_CHECK(mlaTiling.Init(launchParam, kernelInfo).Ok(), "Init MlaPreprocessTiling failed",
              return Mki::Status::FailStatus(Mki::ERROR_ATTR_INVALID_TYPE));
    return Status::OkStatus();
}
} // namespace AtbOps
