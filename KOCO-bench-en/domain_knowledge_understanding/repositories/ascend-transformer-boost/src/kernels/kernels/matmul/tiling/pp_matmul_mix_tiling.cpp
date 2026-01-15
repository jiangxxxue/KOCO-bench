/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pp_matmul_mix_tiling.h"
#include <map>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/matmul.h"
#include "pp_matmul_info.h"
#include "kernels/matmul/common/common_tiling.h"

namespace AsdOps {
constexpr uint32_t CONST_2 = 2;
constexpr uint32_t L2_BW = 5;

void GetHardwareInfoPPMatmulMix(HardwareInfo& hwInfor, PlatformInfo &platInfo)
{
    hwInfor.coreNum = platInfo.GetCoreNum(CoreType::CORE_TYPE_CUBE);
    hwInfor.l2Size = platInfo.GetL2Size();
    hwInfor.l1Size = platInfo.GetL1Size();
    hwInfor.l0aSize = platInfo.GetL0ASize();
    hwInfor.l0bSize = platInfo.GetL0BSize();
    hwInfor.l0cSize = platInfo.GetL0CSize();
    hwInfor.hbmBandWidth = 1;
    hwInfor.l2BandWidth = L2_BW;
}

void PpMixTilingData::SetBaseShape(uint32_t batchSize, uint32_t m, uint32_t k, uint32_t n)
{
    opShape.batchSize = batchSize;
    opShape.m = m;
    opShape.k = k;
    opShape.n = n;
}

void PpMixTilingData::SetBaseOp(uint32_t coreNum, uint32_t mBase, uint32_t nBase)
{
    opShape.m0 = mBase;
    opShape.n0 = nBase;
    mLoop = CeilDiv(opShape.m, opShape.m0);
    nLoop = CeilDiv(opShape.n, opShape.n0);
    coreLoop = opShape.batchSize * mLoop * nLoop;
    blockDim = std::min(coreLoop, coreNum);
}

void PpMixTilingData::SetTilingKey(const MatMulInfo &mmInfo, uint32_t isSwizlDirect, uint32_t isSplitk)
{
    tilingKey = isSwizlDirect;
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.transA);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.transB);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.isInt8);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(mmInfo.biasFlag);
    tilingKey = (tilingKey << 1) + isSplitk;
}

uint32_t PpMixTilingData::End(const MatMulInfo &mmInfo)
{
    uint32_t cubeBlockSize = mmInfo.isInt8 ? CUBE_BLOCK_SIZE_INT8 : CUBE_BLOCK_SIZE;
    uint32_t kBlockSize = mmInfo.isInt8 ? BLOCK_SIZE_INT8_K : BLOCK_SIZE;
    uint32_t shapeCount = opShape.m0 + opShape.n0;
    uint32_t k0Max = shapeCount == 0 ? L1AB_PINGPONG_BUFFER_LEN :
        static_cast<uint32_t>(static_cast<float>(L1AB_PINGPONG_BUFFER_LEN) / (shapeCount * mmInfo.inDtype));
    MKI_LOG(INFO) << "k0Max = " << k0Max;
    opShape.k0 =
        k0Max < cubeBlockSize ? k0Max / kBlockSize * kBlockSize : k0Max / cubeBlockSize * cubeBlockSize;
    kLoop = CeilDiv(opShape.k, opShape.k0);
    return blockDim;
}

uint32_t Splitk(const PpMixTilingData &tilingData)
{
    (void)tilingData;
    return 0;
}

void GetPpMatmulMixTiling(const MatMulInfo &mmInfo, const HardwareInfo &hwInfor,
                          uint32_t &blockDim, PpMixTilingData &tilingData)
{
    OpShape opShape;
    opShape.batchSize = mmInfo.batchSize;
    opShape.m = mmInfo.m;
    opShape.n = mmInfo.n;
    opShape.k = mmInfo.k;
    tilingData.opShape = opShape;

    if (opShape.m < opShape.n) {
        TilingFunc<false, OpShape, PpMixTilingData, HardwareInfo, MatMulInfo>(opShape, tilingData, hwInfor, mmInfo);
    } else {
        TilingFunc<true, OpShape, PpMixTilingData, HardwareInfo, MatMulInfo>(opShape, tilingData, hwInfor, mmInfo);
    }

    uint32_t direct = Swizzl<PpMixTilingData>(tilingData);
    uint32_t splitk = Splitk(tilingData);
    tilingData.splitk = splitk;
    blockDim = tilingData.End(mmInfo);
    tilingData.SetTilingKey(mmInfo, direct, splitk);
    return;
}

Status PpMatmulMixTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    std::map<TensorDType, float> dTypeMap = {
        {TENSOR_DTYPE_INT8, 1.0}, {TENSOR_DTYPE_FLOAT16, 2.0}, {TENSOR_DTYPE_FLOAT, 4.0}};
    MatMulInfo mmInfo;
    HardwareInfo hwInfo;
    GetHardwareInfoPPMatmulMix(hwInfo, PlatformInfo::Instance());
    uint32_t blockDim = 0;
    const auto &attrs = AnyCast<OpParam::MatMul>(launchParam.GetParam());
    const auto &inputDType = launchParam.GetInTensor(0).desc.dtype;
    const auto &inputADim = launchParam.GetInTensor(0).desc.dims;
    const auto &inputBDim = launchParam.GetInTensor(1).desc.dims;
    mmInfo.transA = attrs.transposeA;
    mmInfo.transB = attrs.transposeB;
    mmInfo.isInt8 = (inputDType == TENSOR_DTYPE_INT8);
    mmInfo.biasFlag = (launchParam.GetInTensorCount() > CONST_2);
    if (inputADim.size() > CONST_2 && inputBDim.size() > CONST_2) {
        mmInfo.batchSize = static_cast<uint32_t>(inputBDim.at(0));
        mmInfo.m = static_cast<uint32_t>(inputADim.at(1));
        mmInfo.k = static_cast<uint32_t>(inputBDim.at(1));
        mmInfo.n = static_cast<uint32_t>(inputBDim.at(CONST_2));
        if (mmInfo.transA) {
            mmInfo.m = static_cast<uint32_t>(inputADim.at(CONST_2));
        }
        if (mmInfo.transB) {
            mmInfo.n = static_cast<uint32_t>(inputBDim.at(1));
            mmInfo.k = static_cast<uint32_t>(inputBDim.at(CONST_2));
        }
    } else {
        PpMatmulTilingBDimTwo(mmInfo, inputADim, inputBDim);
    }
    mmInfo.inDtype = dTypeMap[inputDType];
    Status ret = MatmulOriginShapeCheck(mmInfo);
    MKI_CHECK_NO_LOG(ret.Ok(), return ret);

    PpMixTilingData *tilingData =
        reinterpret_cast<AsdOps::PpMixTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingData != nullptr, "tilingData should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE));

    GetPpMatmulMixTiling(mmInfo, hwInfo, blockDim, *tilingData);
    kernelInfo.SetBlockDim(blockDim);

    ret = PpMatmulTilingCheck<PpMixTilingData>(*tilingData);
    MKI_CHECK_NO_LOG(ret.Ok(), return ret);

    MKI_LOG(INFO) << "block dim = " << blockDim;
    MKI_LOG(INFO) << "batchsize, m, k, n = " << tilingData->opShape.batchSize << " " << tilingData->opShape.m << " "
                  << tilingData->opShape.k << " " << tilingData->opShape.n;
    MKI_LOG(INFO) << "m0, k0, n0 = " << tilingData->opShape.m0 << " " << tilingData->opShape.k0 << " "
                  << tilingData->opShape.n0 << " ";
    MKI_LOG(INFO) << "mLoop, kLoop, nLoop = " << tilingData->mLoop << " " << tilingData->kLoop << " "
                  << tilingData->nLoop << "coreloop = " << tilingData->coreLoop << " "
                  << "swizzlCount = " << tilingData->swizzlCount << " ";
    MKI_LOG(INFO) << "swizzlDirect = " << tilingData->swizzlDirect << "split k  = " << tilingData->splitk;
    return Status::OkStatus();
}
} // namespace AsdOps