/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pp_matmul_nz_tiling.h"
#include <map>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/matmul.h"
#include "kernels/matmul/common/common_tiling.h"

namespace AsdOps {
constexpr uint32_t BLOCK_SIZE_K = 32;

HardwareInfoNz::HardwareInfoNz(PlatformInfo &platInfo)
    : coreNum(platInfo.GetCoreNum(CoreType::CORE_TYPE_CUBE)),
      l2Size(platInfo.GetL2Size()),
      l1Size(platInfo.GetL1Size()),
      l0aSize(platInfo.GetL0ASize()),
      l0bSize(platInfo.GetL0BSize()),
      l0cSize(platInfo.GetL0CSize()),
      hbmBandWidth(1),
      l2BandWidth(5) {} // 5 L2 bandwidth

void PpTilingDataNz::SetBaseShape(uint32_t batchSize, uint32_t m, uint32_t k, uint32_t n)
{
    opShape.batchSize = batchSize;
    opShape.m = m;
    opShape.k = k;
    opShape.n = n;
}

void PpTilingDataNz::SetBaseOp(uint32_t coreNum, uint32_t mBase, uint32_t nBase)
{
    opShape.m0 = mBase;
    opShape.n0 = nBase;
    mLoop = CeilDiv(opShape.m, opShape.m0);
    nLoop = CeilDiv(opShape.n, opShape.n0);
    coreLoop = opShape.batchSize * mLoop * nLoop;
    blockDim = std::min(coreLoop, coreNum);
}

void PpTilingDataNz::SetTilingKey(const MatMulInfoNz &mmInfo, uint32_t isSwizlDirect, uint32_t isSplitk)
{
    bool enPreload = CeilDiv(coreLoop, blockDim) > 1u;
    tilingKey = isSwizlDirect;
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.transA);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.transB);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.isInt8);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.biasFlag);
    tilingKey = (tilingKey << 1) + isSplitk;
    tilingKey = (tilingKey << 1) + enPreload;
}

uint32_t PpTilingDataNz::End()
{
    uint32_t shapeSum = opShape.m0 + opShape.n0;
    uint32_t k0Max = (shapeSum == 0) ? L0AB_PINGPONG_BUFFER_LEN_FP16 : (L0AB_PINGPONG_BUFFER_LEN_FP16 / shapeSum);
    opShape.k0 = k0Max < CUBE_BLOCK_SIZE ? k0Max / BLOCK_SIZE * BLOCK_SIZE : \
        k0Max / CUBE_BLOCK_SIZE * CUBE_BLOCK_SIZE; // k0Max less than 256, matrix 16
    kLoop = CeilDiv(opShape.k, opShape.k0);
    return blockDim;
}

uint32_t Splitk(const PpTilingDataNz &tilingData)
{
    (void)tilingData;
    return 0;
}

void GetTiling(const MatMulInfoNz &mmInfo, const HardwareInfoNz &hwInfo, uint32_t &blockDim, PpTilingDataNz &tilingData)
{
    OpShapeNz opShape;
    opShape.batchSize = mmInfo.batchSize;
    opShape.m = mmInfo.m;
    opShape.n = mmInfo.n;
    opShape.k = mmInfo.k;
    tilingData.opShape = opShape;

    if (opShape.m < opShape.n) {
        TilingFunc<false, OpShapeNz, PpTilingDataNz, HardwareInfoNz, MatMulInfoNz>(opShape, tilingData, hwInfo, mmInfo);
    } else {
        TilingFunc<true, OpShapeNz, PpTilingDataNz, HardwareInfoNz, MatMulInfoNz>(opShape, tilingData, hwInfo, mmInfo);
    }
    blockDim = tilingData.End();
    uint32_t direct = Swizzl<PpTilingDataNz>(tilingData);
    uint32_t splitk = Splitk(tilingData);
    tilingData.SetTilingKey(mmInfo, direct, splitk);
    return;
}

Status PpTilingNz(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    std::map<TensorDType, float> dTypeMap = {
        {TENSOR_DTYPE_INT8, 1.0}, {TENSOR_DTYPE_FLOAT16, 2.0}, {TENSOR_DTYPE_FLOAT, 4.0}
    };
    MatMulInfoNz mmInfo;
    HardwareInfoNz hwInfo(PlatformInfo::Instance());
    uint32_t blockDim = 0;
    const auto &attrs = AnyCast<OpParam::MatMul>(launchParam.GetParam());
    const auto &inputDType = launchParam.GetInTensor(0).desc.dtype;
    const auto &inputADim = launchParam.GetInTensor(0).desc.dims;
    mmInfo.transA = attrs.transposeA;
    mmInfo.transB = attrs.transposeB;
    mmInfo.isInt8 = (inputDType == TENSOR_DTYPE_INT8);
    mmInfo.biasFlag = (launchParam.GetInTensorCount() > 2); // InTensor count greater than 2
    Status ret = Status::OkStatus();
    if ((PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_310P ||
         PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910A) &&
        launchParam.GetInTensor(1).desc.format == TENSOR_FORMAT_FRACTAL_NZ) {
        mmInfo.batchSize = static_cast<uint32_t>(inputADim.at(0));
        ret = PpMatMulOriShapeCheck(attrs);
        MKI_CHECK_NO_LOG(ret.Ok(), return ret);
        mmInfo.m = attrs.oriShape[0];
        mmInfo.k = attrs.oriShape[1];
        mmInfo.n = attrs.oriShape[2]; // orishape 2
    }
    mmInfo.inDtype = dTypeMap[inputDType];
    PpTilingDataNz *tilingData =
        reinterpret_cast<AsdOps::PpTilingDataNz *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingData != nullptr, "tilingData should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    if ((PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_310P ||
         PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910A) &&
        launchParam.GetInTensor(1).desc.format == TENSOR_FORMAT_FRACTAL_NZ) {
        GetTiling(mmInfo, hwInfo, blockDim, *tilingData);
    }
    kernelInfo.SetBlockDim(blockDim);
    ret = PpMatmulTilingCheck<PpTilingDataNz>(*tilingData);
    MKI_CHECK_NO_LOG(ret.Ok(), return ret);
    MKI_LOG(INFO) << "block dim = " << blockDim;
    MKI_LOG(INFO) << "(b, m, k, n) = (" << tilingData->opShape.batchSize << ", " << tilingData->opShape.m << ", "
                  << tilingData->opShape.k << ", " << tilingData->opShape.n << ")";
    MKI_LOG(INFO) << "(m0, k0, n0) = (" << tilingData->opShape.m0 << ", " << tilingData->opShape.k0 << ", "
                  << tilingData->opShape.n0 << ")";
    MKI_LOG(INFO) << "(mLoop, kLoop, nLoop) = (" << tilingData->mLoop << ", " << tilingData->kLoop << ", "
                  << tilingData->nLoop << ")";
    MKI_LOG(INFO) << "coreloop = " << tilingData->coreLoop;
    MKI_LOG(INFO) << "swizzlCount = " << tilingData->swizzlCount;
    MKI_LOG(INFO) << "swizzlDirect = " << tilingData->swizzlDirect;
    MKI_LOG(INFO) << "split k = " << tilingData->splitk;
    return Status::OkStatus();
}
} // namespace AsdOps