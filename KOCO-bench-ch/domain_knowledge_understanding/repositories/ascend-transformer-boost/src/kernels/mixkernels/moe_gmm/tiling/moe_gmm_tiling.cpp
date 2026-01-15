/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "moe_gmm_tiling.h"
#include "tiling_data.h"
#include "atbops/params/params.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "mki/utils/math/math.h"
#include "mki/utils/platform/platform_info.h"

namespace AtbOps {
constexpr uint32_t L1_DESCALE_BUFFER_LEN_MAX = 6144;
constexpr uint32_t CONST_2 = 2;
constexpr uint32_t CONST_3 = 3;
constexpr uint32_t CONST_4 = 4;
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t CONST_32 = 32;
constexpr uint32_t CONST_128 = 128;
constexpr uint32_t CONST_256 = 256;
constexpr uint32_t CONST_512 = 512;
constexpr uint32_t TRANS_B_MASK = 0b1000;
constexpr uint32_t IN_TENSOR_COUNT = 4;
constexpr uint32_t IN_TENSOR_W8A8_COUNT = 6;
constexpr uint32_t BASE_BLOCK_STEP = 2;
constexpr uint32_t L1AB_PINGPONG_BUFFER_LEN = 262144;

using namespace Mki;

class MoeGmmTiling {
public:
    Status Init(const LaunchParam &launchParam);
    Status CalcTiling(KernelInfo &kernelInfo);

private:
    void GetBestTileSize();
    float CostFunc(const uint32_t tileM, const uint32_t tileN) const;
    void UpdateTileSize(const uint32_t tileM, const uint32_t tileN);
    void Swizzle();
    void SetTilingData(KernelInfo &kernelInfo) const;
    void End();
    void GetMoEGmmOPTiling();

private:
    uint32_t bSize_{0};
    uint32_t mSize_{0};
    uint32_t kSize_{0};
    uint32_t nSize_{0};
    uint32_t tileM_{0};
    uint32_t tileK_{0};
    uint32_t tileN_{0};
    uint32_t mLoop_{0};
    uint32_t kLoop_{0};
    uint32_t nLoop_{0};
    uint32_t coreLoop_{0};
    uint32_t blockDim_{0};
    uint32_t swizzleDir_{0};
    uint32_t swizzleCnt_{0};
    uint32_t allM_{0};
    int32_t dequantType_{0};
    bool transposeB_{false};
    bool isMoeModeUp_{false};
    bool enDequant_{false};
    TensorDType dtypeA_{TENSOR_DTYPE_UNDEFINED};
    TensorFormat formatB_{TENSOR_FORMAT_UNDEFINED};
};

Status MoeGmmTiling::Init(const LaunchParam &launchParam)
{
    MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MoeGmm), "Invalid op param.",
              return Status::FailStatus(ERROR_ATTR_INVALID_TYPE));
    const auto &attrs = AnyCast<OpParam::MoeGmm>(launchParam.GetParam());
    MKI_CHECK(attrs.moeGmmMode != OpParam::MoeGmm::MOE_GMM_UNDEFINED, "Invalid MoeGmmMode.",
              return Status::FailStatus(ERROR_NOT_CONSISTANT));
    transposeB_ = attrs.transposeB;
    isMoeModeUp_ = attrs.moeGmmMode == OpParam::MoeGmm::MOE_GMM_UP;
    enDequant_ = attrs.moeGmmDequantType != OpParam::MoeGmm::NO_DEQUANT;
    dequantType_ = static_cast<int32_t>(attrs.moeGmmDequantType);
    MKI_CHECK(enDequant_ || launchParam.GetInTensorCount() == IN_TENSOR_COUNT, "Invalid number of input tensor.",
              return Status::FailStatus(ERROR_NOT_CONSISTANT));
    MKI_CHECK(!enDequant_ || launchParam.GetInTensorCount() == IN_TENSOR_W8A8_COUNT, "Invalid number of input tensor.",
              return Status::FailStatus(ERROR_NOT_CONSISTANT));

    const TensorDesc &inTensorDesc0 = launchParam.GetInTensor(0).desc;
    const TensorDesc &inTensorDesc1 = launchParam.GetInTensor(1).desc;
    const TensorDesc &inTensorDesc3 = launchParam.GetInTensor(3).desc;
    dtypeA_ = inTensorDesc0.dtype;
    formatB_ = inTensorDesc1.format;
    MKI_CHECK(formatB_ == TENSOR_FORMAT_ND || formatB_ == TENSOR_FORMAT_FRACTAL_NZ, "Invalid format of input[1].",
              return Status::FailStatus(ERROR_NOT_CONSISTANT));
    bSize_ = inTensorDesc1.dims.at(0);
    allM_ = isMoeModeUp_ ? inTensorDesc3.dims.at(0) : inTensorDesc0.dims.at(0);
    mSize_ = static_cast<uint32_t>(Utils::CeilDiv<int32_t>(allM_, bSize_));
    if (!isMoeModeUp_) {
        allM_ = static_cast<uint32_t>(inTensorDesc0.dims.at(0)) / attrs.topK;
    }
    kSize_ = attrs.hiddenSize.at(0);
    nSize_ = attrs.hiddenSize.at(1);
    return Status::OkStatus();
}

void MoeGmmTiling::GetMoEGmmOPTiling()
{
    if (isMoeModeUp_) {
        if (mLoop_ > 1) {
            // pass
        } else {
            tileM_ = CONST_32;
            mLoop_ = Utils::CeilDiv(mSize_, tileM_);
            tileN_ = enDequant_ ? CONST_512 : CONST_256;
            nLoop_ = Utils::CeilDiv(nSize_, tileN_);
            tileK_ = CONST_256;
            kLoop_ = Utils::CeilDiv(kSize_, tileK_);
            swizzleCnt_ = 1;
        }
    } else {
        if (mLoop_ > 1) {
            tileM_ = CONST_128;
            mLoop_ = Utils::CeilDiv(mSize_, tileM_);
            tileN_ = CONST_256;
            nLoop_ = Utils::CeilDiv(nSize_, tileN_);
            tileK_ = enDequant_ ? CONST_256 : CONST_128;
            kLoop_ = Utils::CeilDiv(kSize_, tileK_);
            swizzleCnt_ = 1;
        } else {
            tileM_ = CONST_32;
            mLoop_ = Utils::CeilDiv(mSize_, tileM_);
            tileN_ = CONST_512;
            nLoop_ = Utils::CeilDiv(nSize_, tileN_);
            tileK_ = enDequant_ ? CONST_256 : CONST_128;
            kLoop_ = Utils::CeilDiv(kSize_, tileK_);
            swizzleCnt_ = 1;
        }
    }
}

Status MoeGmmTiling::CalcTiling(KernelInfo &kernelInfo)
{
    GetBestTileSize();
    GetMoEGmmOPTiling();
    SetTilingData(kernelInfo);
    uint64_t workspaceSize;
    uint64_t workspaceHpSize = 16;
    if (isMoeModeUp_) {
        workspaceSize = static_cast<uint64_t>(blockDim_ * tileM_ * kSize_ * CONST_2 * sizeof(uint16_t));
        if (enDequant_) {
            workspaceHpSize = static_cast<uint64_t>(allM_ * nSize_ * sizeof(uint32_t));
        }
    } else {
        workspaceSize = static_cast<uint64_t>(blockDim_ * tileM_ * nSize_ * CONST_2 * sizeof(uint32_t));
        workspaceHpSize = static_cast<uint64_t>(allM_ * nSize_ * sizeof(uint32_t));
    }
    kernelInfo.GetScratchSizes() = {workspaceSize, workspaceHpSize};
    kernelInfo.SetBlockDim(blockDim_);
    return Status::OkStatus();
}

void MoeGmmTiling::End()
{
    using namespace Mki::Utils;
    uint32_t cubeBlockSize = enDequant_ ? CONST_512 : CONST_256;
    uint32_t kBlockSize = enDequant_ ? CONST_32 : CONST_16;
    uint32_t scaleBlockSize = enDequant_ ? L1_DESCALE_BUFFER_LEN_MAX : 0;
    uint32_t shapeSum = tileM_ + (enDequant_ ? RoundUp(tileN_, CONST_32) : tileN_);
    uint32_t k0Max = shapeSum == 0
                         ? L1AB_PINGPONG_BUFFER_LEN
                         : static_cast<uint32_t>(static_cast<float>(L1AB_PINGPONG_BUFFER_LEN - scaleBlockSize) /
                                                 (shapeSum * CONST_2));
    tileK_ = k0Max < cubeBlockSize ? RoundDown(k0Max, kBlockSize) : RoundDown(k0Max, cubeBlockSize);
    if (tileK_ > CONST_512) {
        tileK_ = Utils::RoundDown(tileK_, CONST_512);
    }
    kLoop_ = Utils::CeilDiv(kSize_, tileK_);
}

void MoeGmmTiling::GetBestTileSize()
{
    using namespace Utils;
    bool priFlag = !(mSize_ < nSize_);
    uint32_t roundBase =
        static_cast<uint32_t>(pow(2, ceil(log(CeilDiv(priFlag ? nSize_ : mSize_, CONST_16)))) * CONST_16);
    uint32_t priAxes = RoundUp(priFlag ? mSize_ : nSize_, CONST_16);
    uint32_t subAxes = RoundUp(priFlag ? nSize_ : mSize_, roundBase);
    float minCost = __FLT_MAX__;
    uint32_t maxAxes0 = enDequant_ ? CONST_512 : CONST_256;
    uint32_t maxPriAxes0 = std::min(maxAxes0, priAxes);
    uint32_t maxSubAxes0 = std::min(maxAxes0, subAxes);
    uint32_t maxTileN = CONST_256;
    for (uint32_t priAxes0 = CONST_16; priAxes0 <= maxPriAxes0; priAxes0 *= BASE_BLOCK_STEP) {
        for (uint32_t subAxes0 = CONST_16; subAxes0 <= maxSubAxes0; subAxes0 *= BASE_BLOCK_STEP) {
            if (priAxes0 * subAxes0 * sizeof(float) > PlatformInfo::Instance().GetL0CSize()) {
                continue;
            }
            uint32_t newM0 = priFlag ? priAxes0 : subAxes0;
            uint32_t newN0 = priFlag ? subAxes0 : priAxes0;
            if (enDequant_ && newN0 > maxTileN) {
                continue;
            }
            float cost = CostFunc(newM0, newN0);
            if (cost < minCost) {
                UpdateTileSize(newM0, newN0);
                minCost = cost;
            }
        }
    }
    Swizzle();
    End();
    return;
}

float MoeGmmTiling::CostFunc(const uint32_t tileM, const uint32_t tileN) const
{
    float aCoef = 1.0f;
    float bCoef = 1.0f;
    float bwCoef = 5.0f;  // base on experiments.
    uint32_t mLoop = Utils::CeilDiv(mSize_, tileM);
    uint32_t nLoop = Utils::CeilDiv(nSize_, tileN);
    if (mLoop == 0 || nLoop == 0) {
        return 1;
    }
    uint32_t coreNeed = bSize_ * mLoop * nLoop;
    const uint32_t numCubeCore = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    uint32_t blockDim = std::min(coreNeed, numCubeCore);
    uint32_t mOnce = blockDim < nLoop ? tileM : blockDim / nLoop * tileM;
    uint32_t nOnce = blockDim < nLoop ? numCubeCore * tileN : nSize_;
    if (mOnce * kSize_ * sizeof(uint16_t) > PlatformInfo::Instance().GetL2Size()) {
        aCoef = bwCoef;
    }
    if (nOnce * kSize_ * sizeof(uint16_t) > PlatformInfo::Instance().GetL2Size()) {
        bCoef = bwCoef;
    }
    return 1 / (aCoef * static_cast<float>(tileN)) + 1 / (bCoef * static_cast<float>(tileM));
}

void MoeGmmTiling::UpdateTileSize(const uint32_t tileM, const uint32_t tileN)
{
    tileM_ = tileM;
    tileN_ = tileN;
    mLoop_ = Utils::CeilDiv(mSize_, tileM_);
    nLoop_ = Utils::CeilDiv(nSize_, tileN_);
    coreLoop_ = bSize_ * mLoop_ * nLoop_;
    const uint32_t numCore = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (mLoop_ == 1 && transposeB_ && coreLoop_ % numCore < numCore / CONST_4 * CONST_3) {
        uint32_t x = Utils::CeilDiv(nSize_, numCore);
        uint32_t y = Utils::CeilDiv(x, CONST_256);
        uint32_t nBase = Utils::RoundUp(Utils::CeilDiv(x, y), CONST_16);
        if (tileM * nBase * sizeof(float) < PlatformInfo::Instance().GetL0CSize()) {
            tileN_ = nBase;
            nLoop_ = Utils::CeilDiv(nSize_, tileN_);
            coreLoop_ = bSize_ * nLoop_;
        }
    }
    blockDim_ = std::min(coreLoop_, numCore);
}

void MoeGmmTiling::Swizzle()
{
    float minCost = mSize_ * kSize_ + kSize_ * nSize_;
    for (uint32_t i = 1; i <= blockDim_; ++i) {
        int c = static_cast<int32_t>((blockDim_ + i - 1) / i);
        float cost = 0.0f;
        if (i * tileN_ + mSize_ < tileM_ * static_cast<uint32_t>(c) + nSize_) {
            swizzleDir_ = 1; // Nz
            cost = tileN_ * i + tileM_ * static_cast<uint32_t>(c);
            if (cost <= minCost) {
                minCost = cost;
                swizzleCnt_ = i;
            }
        } else {
            swizzleDir_ = 0; // Zn
            cost = tileM_ * i + tileN_ * static_cast<uint32_t>(c);
            if (cost < minCost) {
                minCost = cost;
                swizzleCnt_ = i;
            }
        }
    }
}

void PrintMoeGmmTilingData(KernelInfo &kernelInfo)
{
    auto tilingDataPtr = reinterpret_cast<MoeGmmTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_LOG_INFO << ">>> batch: " << tilingDataPtr->batch;
    MKI_LOG_INFO << ">>> m: " << tilingDataPtr->m;
    MKI_LOG_INFO << ">>> k: " << tilingDataPtr->k;
    MKI_LOG_INFO << ">>> n: " << tilingDataPtr->n;
    MKI_LOG_INFO << ">>> m0: " << tilingDataPtr->m0;
    MKI_LOG_INFO << ">>> k0: " << tilingDataPtr->k0;
    MKI_LOG_INFO << ">>> n0: " << tilingDataPtr->n0;
    MKI_LOG_INFO << ">>> mLoop: " << tilingDataPtr->mLoop;
    MKI_LOG_INFO << ">>> kLoop: " << tilingDataPtr->kLoop;
    MKI_LOG_INFO << ">>> nLoop: " << tilingDataPtr->nLoop;
    MKI_LOG_INFO << ">>> coreLoop: " << tilingDataPtr->coreLoop;
    MKI_LOG_INFO << ">>> swizzlCount: " << tilingDataPtr->swizzlCount;
    MKI_LOG_INFO << ">>> blockDim: " << tilingDataPtr->blockDim;
    MKI_LOG_INFO << ">>> swizzlDirect: " << tilingDataPtr->swizzlDirect;
    MKI_LOG_INFO << ">>> splitk: " << tilingDataPtr->splitk;
    MKI_LOG_INFO << ">>> enShuffleK: " << tilingDataPtr->enShuffleK;
    MKI_LOG_INFO << ">>> allM: " << tilingDataPtr->allM;
    MKI_LOG_INFO << ">>> moeUp: " << tilingDataPtr->moeUp;
    MKI_LOG_INFO << ">>> tilingKey: " << tilingDataPtr->tilingKey;
    MKI_LOG_INFO << ">>> workspace: " << kernelInfo.GetScratchSizes()[0];
    MKI_LOG_INFO << ">>> workspaceHp: " << kernelInfo.GetScratchSizes()[1];
}

void MoeGmmTiling::SetTilingData(KernelInfo &kernelInfo) const
{
    auto tilingDataPtr = reinterpret_cast<MoeGmmTilingData *>(kernelInfo.GetTilingHostAddr());
    tilingDataPtr->batch = bSize_;
    tilingDataPtr->m = mSize_;
    tilingDataPtr->k = kSize_;
    tilingDataPtr->n = nSize_;
    tilingDataPtr->m0 = tileM_;
    tilingDataPtr->k0 = tileK_;
    tilingDataPtr->n0 = tileN_;
    tilingDataPtr->mLoop = mLoop_;
    tilingDataPtr->kLoop = kLoop_;
    tilingDataPtr->nLoop = nLoop_;
    tilingDataPtr->coreLoop = coreLoop_;
    tilingDataPtr->swizzlCount = swizzleCnt_;
    tilingDataPtr->blockDim = blockDim_;
    tilingDataPtr->swizzlDirect = swizzleDir_;
    tilingDataPtr->splitk = 0;
    tilingDataPtr->enShuffleK = 0;
    tilingDataPtr->allM = allM_;
    tilingDataPtr->moeUp = isMoeModeUp_;

    uint32_t outDtype = static_cast<uint32_t>(dtypeA_ == TENSOR_DTYPE_BF16);
    if (enDequant_) {
        outDtype = static_cast<uint32_t>(dequantType_ == OpParam::MoeGmm::DEQ_BF16);
    }
    uint32_t tilingKey = swizzleDir_;
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(0);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(transposeB_);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(outDtype);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(0);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(formatB_ == TENSOR_FORMAT_FRACTAL_NZ);
    tilingDataPtr->tilingKey = tilingKey;
}

Mki::Status GetMoeGmmTiling(const Mki::LaunchParam &launchParam, Mki::KernelInfo &kernelInfo)
{
    MoeGmmTiling moeGmmTiling;
    MKI_CHECK(moeGmmTiling.Init(launchParam).Ok(), "Init tiling function failed.",
              return Status::FailStatus(ERROR_NOT_CONSISTANT));
    MKI_CHECK(moeGmmTiling.CalcTiling(kernelInfo).Ok(), "Calc tiling failed.",
              return Status::FailStatus(ERROR_NOT_CONSISTANT));
    PrintMoeGmmTilingData(kernelInfo);
    return Status::OkStatus();
}
} // namespace AtbOps
