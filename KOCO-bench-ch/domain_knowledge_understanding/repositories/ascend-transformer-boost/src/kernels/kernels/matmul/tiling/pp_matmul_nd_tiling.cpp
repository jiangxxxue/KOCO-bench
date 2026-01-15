/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pp_matmul_nd_tiling.h"
#include <map>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/matmul.h"
#include "pp_matmul_info.h"
#include "kernels/matmul/common/common_tiling.h"

namespace AsdOps {
constexpr uint32_t BLOCK_SIZE_K = 32;
static constexpr uint32_t SHAPE_INDEX_2 = 2;

void PpTilingDataNd::SetTilingKey(bool transB)
{
    tilingKey = 0;
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(transB);
}

bool ValidNzFormat(const AsdOps::TensorDesc &tensorDesc)
{
    MKI_CHECK_NO_LOG(tensorDesc.format == TENSOR_FORMAT_FRACTAL_NZ, return false);
    MKI_CHECK_NO_LOG(tensorDesc.dims.size() == 4, return false);
    return true;
}

static SVector<SVector<int64_t>> GetOriShapeTable(const LaunchParam &launchParam)
{
    const auto &inputADim = launchParam.GetInTensor(0).desc.dims;
    const auto &inputBDim = launchParam.GetInTensor(1).desc.dims;

    SVector<SVector<int64_t>> oriShapeTable;
    if (inputBDim.size() > 2 && inputADim.size() > 2) { // Input dimsize greater than 2
        oriShapeTable = {
            {inputADim.at(0), inputADim.at(1), inputADim.at(2), inputBDim.at(2)}, // NN
            {inputADim.at(0), inputADim.at(1), inputADim.at(2), inputBDim.at(1)}, // NT
            {inputADim.at(0), inputADim.at(2), inputADim.at(1), inputBDim.at(2)}, // TN
            {inputADim.at(0), inputADim.at(2), inputADim.at(1), inputBDim.at(1)}  // TT
        };
    } else {
        oriShapeTable = {
            {1, inputADim.at(0), inputADim.at(1), inputBDim.at(1)}, // NN
            {1, inputADim.at(0), inputADim.at(1), inputBDim.at(0)}, // NT
            {1, inputADim.at(1), inputADim.at(0), inputBDim.at(1)}, // TN
            {1, inputADim.at(1), inputADim.at(0), inputBDim.at(0)}  // TT
        };
    }
    return oriShapeTable;
}

Status PpTilingNd(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    const auto &attrs = AnyCast<OpParam::MatMul>(launchParam.GetParam());
    auto inTensorDesc0 = launchParam.GetInTensor(0).desc;
    auto inTensorDesc1 = launchParam.GetInTensor(1).desc;

    MKI_CHECK(inTensorDesc0.format == TENSOR_FORMAT_ND, "Invalid tensor format.",
                 return Status::FailStatus(ERROR_NOT_CONSISTANT));
    MKI_CHECK(inTensorDesc1.format == TENSOR_FORMAT_ND || ValidNzFormat(inTensorDesc1), "Invalid tensor format.",
                 return Status::FailStatus(ERROR_NOT_CONSISTANT));

    kernelInfo.SetBlockDim(1);
    PpTilingDataNd *tilingData = reinterpret_cast<AsdOps::PpTilingDataNd *>(kernelInfo.GetTilingHostAddr());

    auto oriShapeTable = GetOriShapeTable(launchParam);

    bool transA = attrs.transposeA;
    bool transB = attrs.transposeB;
    uint32_t oriShapeIdx = (static_cast<uint32_t>(transA) << 1) + static_cast<uint32_t>(transB);
    int64_t batchSize = oriShapeTable.at(oriShapeIdx).at(0);
    int64_t mValue = oriShapeTable.at(oriShapeIdx).at(1);
    int64_t kValue = oriShapeTable.at(oriShapeIdx).at(2);
    int64_t nValue = oriShapeTable.at(oriShapeIdx).at(3);
    MKI_CHECK(batchSize >= 0 && batchSize <= UINT32_MAX, "Invalid batchSize value.",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingData->batchSize = static_cast<uint32_t>(batchSize);
    MKI_CHECK(mValue >= 0 && mValue <= UINT32_MAX, "Invalid m value.",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingData->m = static_cast<uint32_t>(mValue);
    MKI_CHECK(kValue >= 0 && kValue <= UINT32_MAX, "Invalid k value.",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingData->k = static_cast<uint32_t>(kValue); // dim 2
    MKI_CHECK(nValue >= 0 && nValue <= UINT32_MAX, "Invalid n value.",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingData->n = static_cast<uint32_t>(nValue); // dim 3
    if (inTensorDesc1.format == TENSOR_FORMAT_FRACTAL_NZ) {
        tilingData->n = attrs.oriShape[SHAPE_INDEX_2];
    }

    MKI_LOG(INFO) << "b = " << tilingData->batchSize;
    MKI_LOG(INFO) << "m = " << tilingData->m;
    MKI_LOG(INFO) << "k = " << tilingData->k;
    MKI_LOG(INFO) << "n = " << tilingData->n;

    tilingData->m0 = CONST_128;
    tilingData->k0 = CONST_256;
    tilingData->n0 = CONST_256;
    tilingData->mLoop = CeilDiv(tilingData->m, tilingData->m0);
    tilingData->nLoop = CeilDiv(tilingData->n, tilingData->n0);
    tilingData->kLoop = CeilDiv(tilingData->k, tilingData->k0);
    tilingData->coreLoop = tilingData->batchSize * tilingData->mLoop * tilingData->nLoop;
    tilingData->SetTilingKey(attrs.transposeB);
    return Status::OkStatus();
}
} // namespace AsdOps