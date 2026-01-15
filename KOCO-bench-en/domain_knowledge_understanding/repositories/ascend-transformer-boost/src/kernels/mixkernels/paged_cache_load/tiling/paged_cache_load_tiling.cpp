/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <climits>
#include <mki/launch_param.h>
#include <mki/kernel_info.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/const/op_const.h>
#include "atbops/params/params.h"
#include "paged_cache_load_tiling_dependency.h"

namespace AtbOps {
using namespace Mki;

bool CommonPagedCacheLoadTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto &kCacheShape = launchParam.GetInTensor(DIM_0).desc.dims;
    auto &blockTablesShape = launchParam.GetInTensor(DIM_2).desc.dims;

    auto &kShape = launchParam.GetOutTensor(DIM_0).desc.dims;
    auto &vShape = launchParam.GetOutTensor(DIM_1).desc.dims;

    auto param = AnyCast<OpParam::PagedCacheLoad>(launchParam.GetParam());
    int32_t blockSize;
    int32_t tokenSizeK;
    int32_t tokenSizeV;
    switch (param.type) {
        case OpParam::PagedCacheLoad::PAGED_CACHE_LOAD_NZ:
            blockSize = static_cast<int32_t>(kCacheShape.at(DIM_2));
            tokenSizeK = static_cast<int32_t>(kShape.at(DIM_1));
            tokenSizeV = static_cast<int32_t>(vShape.at(DIM_1));
            break;
        case OpParam::PagedCacheLoad::PAGED_CACHE_LOAD_ND:
            blockSize = static_cast<int32_t>(kCacheShape.at(DIM_1));
            tokenSizeK = static_cast<int32_t>(kShape.at(DIM_1)*kShape.at(DIM_2));
            tokenSizeV = static_cast<int32_t>(vShape.at(DIM_1)*vShape.at(DIM_2));
            break;
        default:
            return false;
    }

    int32_t numTokens = static_cast<int32_t>(blockTablesShape.at(DIM_0)); // block tables row
    int32_t numblkTabCol = static_cast<int32_t>(blockTablesShape.at(DIM_1)); // block tables column

    TensorDType inDtype = launchParam.GetInTensor(DIM_0).desc.dtype;
    uint32_t typeByte = static_cast<uint32_t>(GetTensorElementSize(inDtype));

    MKI_CHECK(blockSize > 0 && blockSize <= INT_MAX, "blockSize is invalid", return false);
    MKI_CHECK(numTokens > 0 && numTokens <= INT_MAX, "numTokens is invalid", return false);
    MKI_CHECK(numblkTabCol > 0 && numblkTabCol <= INT_MAX, "numblkTabCol is invalid", return false);
    MKI_CHECK(tokenSizeK > 0 && tokenSizeK <= INT_MAX, "tokenSizeK is invalid", return false);
    MKI_CHECK(tokenSizeV > 0 && tokenSizeV <= INT_MAX, "tokenSizeV is invalid", return false);
    MKI_CHECK(typeByte > 0 && typeByte <= INT_MAX, "typeByte is invalid", return false);
    MKI_CHECK(tokenSizeK <= INT_MAX / blockSize / static_cast<int32_t>(typeByte),
        "tokenSizeK * blockSize is too large", return false);
    MKI_CHECK(tokenSizeV <= INT_MAX / blockSize / static_cast<int32_t>(typeByte),
        "tokenSizeV * blockSize is too large", return false);

    PagedCacheLoadTilingData *tilingDataPtr =
            reinterpret_cast<PagedCacheLoadTilingData *>(kernelInfo.GetTilingHostAddr());

    tilingDataPtr->blockSize = blockSize;
    tilingDataPtr->numTokens = numTokens;
    tilingDataPtr->numblkTabCol = numblkTabCol;
    tilingDataPtr->tokenSizeK = tokenSizeK;
    tilingDataPtr->tokenSizeV = tokenSizeV;
    tilingDataPtr->typeByte = typeByte;
    tilingDataPtr->hasSeqStarts = param.hasSeqStarts;
    tilingDataPtr->cuSeqLens = param.cuSeqLens;

    MKI_LOG(INFO) << "blockSize: " << tilingDataPtr->blockSize << ", numTokens: " <<
        tilingDataPtr->numTokens << ", numblkTabCol: " << tilingDataPtr->numblkTabCol <<
        ", tokenSizeK: " << tilingDataPtr->tokenSizeK << ", tokenSizeV: " <<
        tilingDataPtr->tokenSizeV << ", typeByte: " << tilingDataPtr->typeByte;

    return true;
}


uint32_t PagedCacheLoadGetBlockDim(const LaunchParam &launchParam)
{
    auto &kShape = launchParam.GetOutTensor(DIM_0).desc.dims;
    uint32_t sumContextLens = static_cast<uint32_t>(kShape.at(DIM_0));

    uint32_t blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    auto param = AnyCast<OpParam::PagedCacheLoad>(launchParam.GetParam());
    switch (param.type) {
        case OpParam::PagedCacheLoad::PAGED_CACHE_LOAD_NZ:
            return sumContextLens < blockDim ? sumContextLens : blockDim;
        case OpParam::PagedCacheLoad::PAGED_CACHE_LOAD_ND:
            return blockDim;
        default:
            return blockDim;
    }
    return blockDim;
}


Status PagedCacheLoadTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    PagedCacheLoadTilingData *tilingDataPtr =
            reinterpret_cast<PagedCacheLoadTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingHost should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingHost should not be empty"));
    MKI_CHECK(CommonPagedCacheLoadTiling(launchParam, kernelInfo), "value is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    kernelInfo.SetBlockDim(PagedCacheLoadGetBlockDim(launchParam));
    return Status::OkStatus();
}
} // namespace AtbOps
