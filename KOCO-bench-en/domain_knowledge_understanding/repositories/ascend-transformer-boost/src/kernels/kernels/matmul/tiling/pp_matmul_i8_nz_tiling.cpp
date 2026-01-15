/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pp_matmul_i8_nz_tiling.h"
#include <map>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/matmul.h"
#include "kernels/matmul/common/common_tiling.h"

namespace AsdOps {
constexpr uint32_t CONST_2 = 2;
constexpr uint32_t L2_BW = 5;
constexpr uint32_t ALIGNMENT_16 = 16;
constexpr uint32_t ALIGNMENT_256 = 256;
constexpr uint32_t CONST_3 = 3;
constexpr uint32_t CONST_4 = 4;
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t CONST_256 = 256;
constexpr uint32_t CONST_512 = 512;
constexpr uint32_t TRANS_B_MASK = 0b001000;

HardwareInfo310P::HardwareInfo310P(PlatformInfo &platInfo)
    : coreNum(platInfo.GetCoreNum(CoreType::CORE_TYPE_CUBE)), l2Size(platInfo.GetL2Size()),
      l1Size(platInfo.GetL1Size()), l0aSize(platInfo.GetL0ASize()), l0bSize(platInfo.GetL0BSize()),
      l0cSize(platInfo.GetL0CSize()), hbmBandWidth(1), l2BandWidth(L2_BW)
{
}

void PpTilingData310P::SetBaseShape(uint32_t batchSize, uint32_t m, uint32_t k, uint32_t n)
{
    opShape.batchSize = batchSize;
    opShape.m = m;
    opShape.k = k;
    opShape.n = n;
}

void PpTilingData310P::SetBaseOp(uint32_t coreNum, uint32_t mBase, uint32_t nBase)
{
    opShape.m0 = mBase;
    opShape.n0 = nBase;
    mLoop = CeilDiv(opShape.m, opShape.m0);
    nLoop = CeilDiv(opShape.n, opShape.n0);
    coreLoop = opShape.batchSize * mLoop * nLoop;

    bool transB = tilingKey & TRANS_B_MASK;
    if (mLoop == 1 && (tilingK == 0) && transB && coreLoop % coreNum < coreNum / CONST_4 * CONST_3) {
        uint32_t x = CeilDiv(opShape.n, coreNum);
        uint32_t y = CeilDiv(x, CONST_256);
        nBase = RoundUp(CeilDiv(x, y), CONST_16);
        uint32_t baseLimitSize = (PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910A)
                                   ? UB_LIMIT_SIZE_910A
                                   : PlatformInfo::Instance().GetL0CSize();
        if (mBase * nBase * sizeof(float) < baseLimitSize) {
            opShape.n0 = nBase;
            nLoop = CeilDiv(opShape.n, opShape.n0);
            coreLoop = opShape.batchSize * nLoop;
        }
    }
    blockDim = std::min(coreLoop, coreNum);
}

void PpTilingData310P::SetTilingKey(const MatMulInfo310P &mmInfo)
{
    tilingKey = swizzlDirect;
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.transA);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.transB);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.isInt8);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.biasFlag);
    tilingKey = (tilingKey << 1) + splitK;
}

uint32_t PpTilingData310P::End(const MatMulInfo310P &mmInfo)
{
    uint32_t l1AbPpBuffLen = L0AB_PINGPONG_BUFFER_LEN_FP16;
    if (mmInfo.isCompress) {
        l1AbPpBuffLen = L1AB_PINGPONG_BUFFER_LEN_INT8_SPARSE;
        uint32_t compressNTile = CeilDiv((opShape.n % opShape.n0), ALIGNMENT_16) % mmInfo.tilingN;
        compressOverlapN = compressNTile == 0 ? 0 : mmInfo.tilingN - compressNTile;
    }
    uint32_t shapeCount = opShape.m0 + opShape.n0;
    uint32_t k0Max = (shapeCount == 0) ? l1AbPpBuffLen : (l1AbPpBuffLen / shapeCount);
    uint32_t k0Init = mmInfo.isCompress ? mmInfo.tilingK * BLOCK_SIZE_INT8_K : ALIGNMENT_256;
    opShape.k0 = k0Max < k0Init ? k0Max / ALIGNMENT_16 * ALIGNMENT_16 : k0Max / k0Init * k0Init;
    kLoop = CeilDiv(opShape.k, opShape.k0);
    return blockDim;
}

void GetSwizzlParam(PpTilingData310P &tilingData)
{
    tilingData.swizzlDirect = 0;
    tilingData.swizzlCount = 1;
    float m0 = tilingData.opShape.m0;
    float n0 = tilingData.opShape.n0;
    float m = tilingData.opShape.m;
    float k = tilingData.opShape.k;
    float n = tilingData.opShape.n;
    float mincost = m * k + k * n;

    for (uint32_t i = 1; i <= tilingData.blockDim; ++i) {
        uint32_t c = (tilingData.blockDim + i - 1) / i;
        float cost;
        // B0 + A < A0 + B
        if (i * n0 + m < m0 * c + n) {
            tilingData.swizzlDirect = 1; // Nz
            cost = n0 * i + m0 * c;
            if (cost <= mincost) {
                mincost = cost;
                tilingData.swizzlCount = i;
            }
        } else {
            tilingData.swizzlDirect = 0; // Zn
            cost = m0 * i + n0 * c;
            if (cost < mincost) {
                mincost = cost;
                tilingData.swizzlCount = i;
            }
        }
    }
    return;
}

void GetSplitKParam(PpTilingData310P &tilingData)
{
    tilingData.splitK = 0;
    return;
}

void GetPpTiling(const MatMulInfo310P &mmInfo, const HardwareInfo310P &hwInfo, uint32_t &blockDim,
                 PpTilingData310P &tilingData)
{
    OpShape310P opShape;
    opShape.batchSize = mmInfo.batchSize;
    opShape.m = mmInfo.m;
    opShape.n = mmInfo.n;
    opShape.k = mmInfo.k;
    tilingData.opShape = opShape;
    tilingData.tilingK = mmInfo.tilingK;
    tilingData.tilingN = mmInfo.tilingN;
    tilingData.SetTilingKey(mmInfo);
    if (opShape.m < opShape.n) {
        TilingFunc<false, OpShape310P, PpTilingData310P, HardwareInfo310P, MatMulInfo310P>(
            opShape, tilingData, hwInfo, mmInfo, mmInfo.isCompress, mmInfo.tilingN);
    } else {
        TilingFunc<true, OpShape310P, PpTilingData310P, HardwareInfo310P, MatMulInfo310P>(
            opShape, tilingData, hwInfo, mmInfo, mmInfo.isCompress, mmInfo.tilingN);
    }
    GetSwizzlParam(tilingData);
    GetSplitKParam(tilingData);
    blockDim = tilingData.End(mmInfo);
    tilingData.SetTilingKey(mmInfo);
}

Status PpTiling310P(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    std::map<TensorDType, float> dTypeMap = {
        {TENSOR_DTYPE_INT8, 1.0}, {TENSOR_DTYPE_FLOAT16, 2.0}, {TENSOR_DTYPE_FLOAT, 4.0}};
    MatMulInfo310P mmInfo;
    HardwareInfo310P hwInfo(PlatformInfo::Instance());
    uint32_t blockDim = 0;
    const auto &attrs = AnyCast<OpParam::MatMul>(launchParam.GetParam());
    const auto &inputDType = launchParam.GetInTensor(0).desc.dtype;
    const auto &inputADim = launchParam.GetInTensor(0).desc.dims;
    mmInfo.transA = attrs.transposeA;
    mmInfo.transB = attrs.transposeB;
    mmInfo.tilingK = attrs.tilingK;
    mmInfo.tilingN = attrs.tilingN;
    mmInfo.isInt8 = (inputDType == TENSOR_DTYPE_INT8);
    mmInfo.isCompress = (attrs.tilingK > 0 && attrs.tilingN > 0);
    mmInfo.biasFlag = (launchParam.GetInTensorCount() > CONST_2);
    mmInfo.batchSize = inputADim[0];
    Status ret = PpMatMulOriShapeCheck(attrs);
    MKI_CHECK_NO_LOG(ret.Ok(), return ret);
    mmInfo.m = attrs.oriShape[0]; // oriShape[0]: m
    mmInfo.k = attrs.oriShape[1]; // oriShape[1]: k
    mmInfo.n = attrs.oriShape[2]; // oriShape[2]: n
    mmInfo.inDtype = dTypeMap[inputDType];
    PpTilingData310P *tilingdata = reinterpret_cast<AsdOps::PpTilingData310P *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingdata != nullptr, "tilingDataPtr should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    GetPpTiling(mmInfo, hwInfo, blockDim, *tilingdata);
    kernelInfo.SetBlockDim(blockDim);

    ret = PpMatmulTilingCheck<PpTilingData310P>(*tilingdata);
    MKI_CHECK_NO_LOG(ret.Ok(), return ret);

    MKI_LOG(INFO) << "blockDim = " << blockDim;
    MKI_LOG(INFO) << "(b, m, k, n) = (" << tilingdata->opShape.batchSize << ", " << tilingdata->opShape.m << ", "
                  << tilingdata->opShape.k << ", " << tilingdata->opShape.n << ")";
    MKI_LOG(INFO) << "(m0, k0, n0) = (" << tilingdata->opShape.m0 << ", " << tilingdata->opShape.k0 << ", "
                  << tilingdata->opShape.n0 << ")";
    MKI_LOG(INFO) << "(mLoop, kLoop, nLoop) = (" << tilingdata->mLoop << ", " << tilingdata->kLoop << ", "
                  << tilingdata->nLoop << ")";
    MKI_LOG(INFO) << "coreloop = " << tilingdata->coreLoop;
    MKI_LOG(INFO) << "swizzlCount = " << tilingdata->swizzlCount;
    MKI_LOG(INFO) << "isCompress " << mmInfo.isCompress;
    MKI_LOG(INFO) << "tilingK = " << tilingdata->tilingK;
    MKI_LOG(INFO) << "tilingN = " << tilingdata->tilingN;
    MKI_LOG(INFO) << "compressOverlapN = " << tilingdata->compressOverlapN;
    MKI_LOG(INFO) << "swizzlDirect = " << tilingdata->swizzlDirect;
    MKI_LOG(INFO) << "splitK = " << tilingdata->splitK;
    return Status::OkStatus();
}
} // namespace AsdOps