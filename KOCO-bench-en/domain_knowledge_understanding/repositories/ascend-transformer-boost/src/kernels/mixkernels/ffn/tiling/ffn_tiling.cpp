/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/types.h>
#include "atbops/params/params.h"
#include "tiling_data.h"
#include "ffn_tiling.h"

namespace AtbOps {
constexpr uint32_t BLOCK_NUM_PER_FRACTAL = 16;
constexpr uint32_t NUM_PER_FRACTAL = 512;
constexpr uint32_t BLOCK_SIZE_32 = 32;
constexpr uint32_t BLOCK_SIZE_16 = 16;
constexpr uint32_t BLOCK_NUM_PER_VEC = 8;
constexpr uint32_t AXES_M_SIZE = 256;
constexpr uint32_t AXES_N_SIZE = 128;
constexpr uint32_t CONST_2 = 2;
constexpr uint32_t CONST_3 = 3;
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t CONST_32 = 32;
constexpr uint32_t CONST_256 = 256;
constexpr uint32_t UB_HALF_BUFFER_LEN = 131072;
constexpr uint32_t HBM_BW = 17;
constexpr uint32_t L2_BW = 114;
constexpr uint32_t FFN_WORKSPACE_TENSOR_IDX = 10;
constexpr uint32_t DTYPE_BIT_COUNT = 2;
constexpr uint32_t FORMAT_BIT_COUNT = 1;
constexpr uint32_t MODE_BIT_COUNT = 1;

const std::unordered_map<TensorDType, uint32_t> TENSOR_DTYPE_MAP = {
    {TENSOR_DTYPE_INT8, 0},
    {TENSOR_DTYPE_FLOAT16, 1},
    {TENSOR_DTYPE_BF16, 2},
    {TENSOR_DTYPE_FLOAT, 3}};

const std::unordered_map<TensorFormat, uint32_t> TENSOR_FORMAT_MAP = {
    {TENSOR_FORMAT_ND, 0},
    {TENSOR_FORMAT_FRACTAL_NZ, 1}};

struct FFNHardwareInfo {
    uint32_t coreNum{0};
    uint64_t l1Size{0};
    uint64_t l2Size{0};
    uint64_t l0ASize{0};
    uint64_t l0CSize{0};
    uint64_t ubSize{0};
    uint64_t hbmBandWidth{0};
    uint64_t l2BandWidth{0};
    explicit FFNHardwareInfo(PlatformInfo &platInfo)
        : coreNum(platInfo.GetCoreNum(CoreType::CORE_TYPE_CUBE)),
          l1Size(platInfo.GetL1Size()),
          l2Size(platInfo.GetL2Size()),
          l0ASize(platInfo.GetL0ASize()),
          l0CSize(platInfo.GetL0CSize()),
          ubSize(platInfo.GetUbSize()),
          hbmBandWidth(HBM_BW),
          l2BandWidth(L2_BW) {}
};

inline Status FFNCheck(const LaunchParam &launchParam, const OpParam::FFN &param)
{
    MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FFN), "OpParam is not FFN",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(param.activationType >= OpParam::FFN::ActivationType::GELU &&
        param.activationType < OpParam::FFN::ActivationType::INVALID_ACTIVATION_TYPE,
        "activationType only support 0,1,2", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(param.implMode >= OpParam::FFN::ImplMode::HIGH_PRECISION &&
        param.implMode < OpParam::FFN::ImplMode::INVALID_IMPL_MODE,
        "implMode only support 0,1", return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}

inline float CostFunc(const FFNHardwareInfo &hwInfo, const FFNMatmulTilingData &tilingTmp)
{
    float aCoef = 1;
    float bCoef = 1;
    float bwCoef = static_cast<float>(hwInfo.l2BandWidth) / static_cast<float>(hwInfo.hbmBandWidth);
    uint32_t mLoops = Utils::CeilDiv(tilingTmp.m, tilingTmp.baseM);
    uint32_t nLoops = Utils::CeilDiv(tilingTmp.n, tilingTmp.baseN);
    if (mLoops == 0 || nLoops == 0) {
        return 1;
    }
    uint32_t coreNeed = mLoops * nLoops;
    uint32_t blockDim = std::min(coreNeed, hwInfo.coreNum);
    uint32_t mOnce = blockDim < nLoops ? tilingTmp.baseM : blockDim / nLoops * tilingTmp.baseM;
    uint32_t nOnce = blockDim < nLoops ? hwInfo.coreNum * tilingTmp.baseN : tilingTmp.n;
    if (mOnce * tilingTmp.k * sizeof(uint16_t) > hwInfo.l2Size) {
        aCoef = bwCoef;
    }
    if (nOnce * tilingTmp.k * sizeof(uint16_t) > hwInfo.l2Size) {
        bCoef = bwCoef;
    }
    return 1 / (aCoef * static_cast<float>(tilingTmp.baseN)) + 1 / (bCoef * static_cast<float>(tilingTmp.baseM));
}

inline void SetBaseOp(FFNMatmulTilingData &tilingTmp)
{
    tilingTmp.mLoops = Utils::CeilDiv(tilingTmp.m, tilingTmp.baseM);
    tilingTmp.nLoops = Utils::CeilDiv(tilingTmp.n, tilingTmp.baseN);
    tilingTmp.coreLoops = tilingTmp.mLoops * tilingTmp.nLoops;

    uint32_t l1AbPpBuffLen = UB_HALF_BUFFER_LEN;
    uint32_t shapeCount = tilingTmp.baseM + tilingTmp.baseN;
    uint32_t k0Max = (shapeCount == 0) ? l1AbPpBuffLen : (l1AbPpBuffLen / shapeCount);
    uint32_t k0Init = CONST_256;
    tilingTmp.baseK = k0Max < k0Init ? k0Max / CONST_32 * CONST_32 : k0Max / k0Init * k0Init;
    tilingTmp.kLoops = Utils::CeilDiv(tilingTmp.k, tilingTmp.baseK);
}

inline void GetSwizzlParam(const FFNHardwareInfo &hwInfo, FFNMatmulTilingData &tilingTmp)
{
    tilingTmp.swizzlDirect = 0;
    tilingTmp.swizzlCount = 1;
    float baseM = tilingTmp.baseM;
    float baseN = tilingTmp.baseN;
    float m = tilingTmp.m;
    float k = tilingTmp.k;
    float n = tilingTmp.n;
    float mincost = m * k + k * n;
    uint32_t blockDim = std::min(tilingTmp.coreLoops, hwInfo.coreNum);
    for (uint32_t i = 1; i <= blockDim; ++i) {
        uint32_t c = (blockDim + i - 1) / i;
        float cost;
        // B0 + A < A0 + B
        if (i * baseN + m < baseM * c + n) {
            tilingTmp.swizzlDirect = 1; // Nz
            cost = baseN * i + baseM * c;
            if (cost <= mincost) {
                mincost = cost;
                tilingTmp.swizzlCount = i;
            }
        } else {
            tilingTmp.swizzlDirect = 0; // Zn
            cost = baseM * i + baseN * c;
            if (cost < mincost) {
                mincost = cost;
                tilingTmp.swizzlCount = i;
            }
        }
    }
    return;
}

inline void GetMatmulTiling(const FFNHardwareInfo &hwInfo, FFNMatmulTilingData &tilingTmp)
{
    float costMin = 1;
    uint32_t mUpLimit = Utils::RoundUp(tilingTmp.m, CONST_16);
    uint32_t nUpLimit = Utils::RoundUp(tilingTmp.n, CONST_32);

    for (uint32_t mAxes = BLOCK_SIZE_16; mAxes <= mUpLimit && mAxes <= AXES_M_SIZE; mAxes += BLOCK_SIZE_16) {
        for (uint32_t nAxes = BLOCK_SIZE_32; nAxes <= nUpLimit && nAxes <= AXES_N_SIZE; nAxes += BLOCK_SIZE_32) {
            if (mAxes * nAxes * sizeof(float) > hwInfo.l0CSize) {
                continue;
            }
            tilingTmp.baseM = mAxes;
            tilingTmp.baseN = nAxes;
            float cost = CostFunc(hwInfo, tilingTmp);
            if (cost < costMin) {
                costMin = cost;
                SetBaseOp(tilingTmp);
            }
        }
    }
    GetSwizzlParam(hwInfo, tilingTmp);
}

inline void GetOffsetTiling(FFNTilingData &tiling)
{
    uint64_t scaleOffset = tiling.mm1.baseN * CONST_2 * (sizeof(uint64_t) + sizeof(int32_t));
    uint64_t biasOffset = scaleOffset + tiling.mm1.baseM / BLOCK_NUM_PER_VEC * tiling.mm1.baseN * sizeof(uint16_t);
    uint64_t syncOffset = biasOffset + tiling.mm1.baseN * sizeof(uint16_t);
    uint64_t cOffset = syncOffset + BLOCK_SIZE_32;
    uint64_t geluOffset = cOffset + UB_HALF_BUFFER_LEN;
    tiling.scaleOffset = scaleOffset;
    tiling.biasOffset = biasOffset;
    tiling.syncOffset = syncOffset;
    tiling.cOffset = cOffset;
    tiling.geluOffset = geluOffset;
}

inline void SetTilingKey(const OpParam::FFN &param, KernelInfo &kernelInfo)
{
    if (param.implMode == OpParam::FFN::ImplMode::HIGH_PRECISION) {
        kernelInfo.SetTilingId(0);
    } else  if (param.implMode == OpParam::FFN::ImplMode::HIGH_PERFORMANCE) {
        kernelInfo.SetTilingId(1);
    }
}

inline void FFNTilingParam(const LaunchParam &launchParam, KernelInfo &kernelInfo,
    OpParam::FFN &param, FFNTilingData &tiling)
{
    uint32_t blockDim;
    uint32_t tmpSize;
    FFNHardwareInfo hwInfo(PlatformInfo::Instance());
    
    auto xTensor = launchParam.GetInTensor(DIM_0);
    auto w1Tensor = launchParam.GetInTensor(DIM_1);
    auto w2Tensor = launchParam.GetInTensor(DIM_2);
    
    tiling.mm1.m = static_cast<uint32_t>(xTensor.desc.dims[DIM_0]);
    tiling.mm1.k = static_cast<uint32_t>(xTensor.desc.dims[DIM_1]);
    tiling.mm1.n = static_cast<uint32_t>(w1Tensor.desc.dims[DIM_0]);
    tiling.mm2.m = static_cast<uint32_t>(xTensor.desc.dims[DIM_0]);
    tiling.mm2.k = static_cast<uint32_t>(w1Tensor.desc.dims[DIM_0]);
    tiling.mm2.n = static_cast<uint32_t>(w2Tensor.desc.dims[DIM_0]);
    uint32_t roundM = Utils::RoundUp(tiling.mm1.m, BLOCK_SIZE_16);
    uint32_t roundN = Utils::RoundUp(tiling.mm1.n, BLOCK_SIZE_32);
    GetMatmulTiling(hwInfo, tiling.mm1);
    GetMatmulTiling(hwInfo, tiling.mm2);
    GetOffsetTiling(tiling);
    SetTilingKey(param, kernelInfo);

    tiling.activationType = static_cast<uint32_t>(param.activationType);

    uint32_t coreLoops = std::max(tiling.mm1.coreLoops, tiling.mm2.coreLoops);
    blockDim = (hwInfo.coreNum < coreLoops) ? hwInfo.coreNum : coreLoops;
    tmpSize = NUM_PER_FRACTAL + roundM * roundN * sizeof(int8_t);
    kernelInfo.SetBlockDim(blockDim);
    kernelInfo.GetScratchSizes() = {tmpSize};
    kernelInfo.SetMemsetInfo(FFN_WORKSPACE_TENSOR_IDX, tmpSize);
}

Status FFNTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto param = AnyCast<OpParam::FFN>(launchParam.GetParam());
    OP_TILING_CHECK_STATUS_RETURN(FFNCheck(launchParam, param));
    
    FFNTilingData *tiling = reinterpret_cast<FFNTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tiling != nullptr, "tiling should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    FFNTilingParam(launchParam, kernelInfo, param, *tiling);

    MKI_LOG(INFO) << "activationType = " << tiling->activationType;
    MKI_LOG(INFO) << "scaleOffset = " << tiling->scaleOffset;
    MKI_LOG(INFO) << "biasOffset = " << tiling->biasOffset;
    MKI_LOG(INFO) << "syncOffset = " << tiling->syncOffset;
    MKI_LOG(INFO) << "cOffset = " << tiling->cOffset;
    MKI_LOG(INFO) << "geluOffset = " << tiling->geluOffset;
    MKI_LOG(INFO) << "mm1.baseM = " << tiling->mm1.baseM;
    MKI_LOG(INFO) << "mm1.baseK = " << tiling->mm1.baseK;
    MKI_LOG(INFO) << "mm1.baseN = " << tiling->mm1.baseN;
    MKI_LOG(INFO) << "mm2.baseM = " << tiling->mm2.baseM;
    MKI_LOG(INFO) << "mm2.baseK = " << tiling->mm2.baseK;
    MKI_LOG(INFO) << "mm2.baseN = " << tiling->mm2.baseN;

    return Status::OkStatus();
}

}  // namespace AtbOps