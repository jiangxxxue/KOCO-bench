/*
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "blockcopy_tiling.h"
#include <limits>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "tiling_data.h"
static constexpr int32_t NZ_SIZE = 16;
static constexpr uint64_t BLOCKCOPY_WKSP_TENSOR_IDX = 7;
static constexpr uint64_t BLOCKCOPY_SYNC_WORKSPACE_TENSOR_IDX = 8;
static constexpr int32_t BYTES_32 = 32;
namespace AtbOps {
using namespace Mki;

bool BlockCopyTilingNd(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    BlockCopyTilingData *tilingDataPtr = reinterpret_cast<AtbOps::BlockCopyTilingData*>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingHost should not be empty",
              return false);
    auto &kShape = launchParam.GetInTensor(DIM_0).desc.dims;
    auto &vShape = launchParam.GetInTensor(DIM_1).desc.dims;
    // 910B kv shape [#B, B, head, DIM]
    uint32_t blockSize = static_cast<uint32_t>(kShape.at(DIM_1));
    uint32_t numHead = static_cast<uint32_t>(kShape.at(DIM_2));
    uint32_t headSizeK = static_cast<uint32_t>(kShape.at(DIM_3));
    uint32_t headSizeV = static_cast<uint32_t>(vShape.at(DIM_3));
    MKI_CHECK(blockSize > 0, "blockSize is invalid", return false);
    MKI_CHECK(numHead > 0, "numHead is invalid", return false);
    tilingDataPtr->blockSize = blockSize;
    tilingDataPtr->numHead = numHead;
    MKI_CHECK(headSizeK > 0, "headSizeK is invalid", return false);
    MKI_CHECK(headSizeV > 0, "headSizeV is invalid", return false);
    tilingDataPtr->headSizeK = headSizeK;
    tilingDataPtr->headSizeV = headSizeV;
    uint64_t maxVal = std::numeric_limits<int64_t>::max();
    MKI_CHECK(numHead <= maxVal / blockSize,
          "blockSize * numHead exceeds uint64_t limit", return false);
    uint64_t tmp = static_cast<uint64_t>(blockSize) * static_cast<uint64_t>(numHead);
    MKI_CHECK(headSizeK <= maxVal / tmp,
          "blockSize * numHead * headSizeK exceeds uint64_t limit", return false);
    return true;
}
bool BlockCopyTilingNz(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    BlockCopyTilingData *tilingDataPtr = reinterpret_cast<AtbOps::BlockCopyTilingData*>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingHost should not be empty",
              return false);
    auto &kShape = launchParam.GetInTensor(DIM_0).desc.dims;
    auto &vShape = launchParam.GetInTensor(DIM_1).desc.dims;
    // nz：无法获取头和头大小；numhead弃置，改用headSizeV来保存大小
    // 310p的kv：[#B, head*head_size // 16, B, 16]
    uint32_t blockSize = static_cast<uint32_t>(kShape.at(DIM_2));
    // 头参数固定设置为1
    uint32_t numHead = static_cast<uint32_t>(1);
    uint32_t headSizeK = static_cast<uint32_t>(kShape.at(DIM_1) * NZ_SIZE);
    uint32_t headSizeV = static_cast<uint32_t>(vShape.at(DIM_1) * NZ_SIZE);
    MKI_CHECK(blockSize > 0, "blockSize is invalid", return false);
    MKI_CHECK(numHead > 0, "numHead is invalid", return false);
    MKI_CHECK(headSizeK > 0, "headSizeK is invalid", return false);
    MKI_CHECK(headSizeV > 0, "headSizeV is invalid", return false);
    tilingDataPtr->blockSize = blockSize;
    tilingDataPtr->numHead = numHead;
    tilingDataPtr->headSizeK = headSizeK;
    tilingDataPtr->headSizeV = headSizeV;
    uint64_t maxVal = std::numeric_limits<int64_t>::max();
    MKI_CHECK(numHead <= maxVal / blockSize,
          "blockSize * numHead exceeds uint64_t limit", return false);
    uint64_t tmp = static_cast<uint64_t>(blockSize) * static_cast<uint64_t>(numHead);
    MKI_CHECK(headSizeK <= maxVal / tmp,
          "blockSize * numHead * headSizeK exceeds uint64_t limit", return false);
    return true;
}

bool BlockCopyTilingCheck310P(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    TensorDType inDtype = launchParam.GetInTensor(0).desc.dtype;
    uint32_t typeByte = static_cast<uint32_t>(GetTensorElementSize(inDtype));
    MKI_CHECK(typeByte > 0, "typeByte is invalid", return false);
    BlockCopyTilingData *tilingDataPtr = reinterpret_cast<AtbOps::BlockCopyTilingData*>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingHost should not be empty",
              return false);
    auto headDim = tilingDataPtr->blockSize * tilingDataPtr->numHead;
    auto totalKCacheItems = tilingDataPtr->headSizeK * headDim;
    auto totalVCacheItems = tilingDataPtr->headSizeV * headDim;
    MKI_CHECK(totalKCacheItems % (BYTES_32 / typeByte) == 0, "310P Platform KCache should be aligned with 32 bytes!",
              return false);
    MKI_CHECK(totalVCacheItems % (BYTES_32 / typeByte) == 0, "310P Platform VCache should be aligned with 32 bytes!",
              return false);
    return true;
}

Status BlockCopyTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto param = AnyCast<OpParam::BlockCopy>(launchParam.GetParam());
    auto &kShape = launchParam.GetInTensor(DIM_0).desc.dims;
    auto &srcShape = launchParam.GetInTensor(DIM_2).desc.dims;
    auto &dstShape = launchParam.GetInTensor(DIM_3).desc.dims;
    auto &cumShape = launchParam.GetInTensor(DIM_4).desc.dims;
    uint32_t blockCount = static_cast<uint32_t>(kShape.at(DIM_0));
    uint32_t sourceCount = static_cast<uint32_t>(srcShape.at(DIM_0));
    uint32_t destinationCount = static_cast<uint32_t>(dstShape.at(DIM_0));
    uint32_t cumSumCount = static_cast<uint32_t>(cumShape.at(DIM_0));

    if (param.type == OpParam::BlockCopy::BLOCK_COPY_CACHE_ND) {
        MKI_CHECK(BlockCopyTilingNd(launchParam, kernelInfo), "K/V Cache size is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
    } else {
        MKI_CHECK(BlockCopyTilingNz(launchParam, kernelInfo), "K/V Cache size is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
    }
    bool is310P = PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_310P ? true : false;
    if (is310P) {
        MKI_CHECK(BlockCopyTilingCheck310P(launchParam, kernelInfo), "K/V Cache size aligned error in 310P",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
    }

    MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::BlockCopy), "OpParam is invalid",
        return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
    MKI_CHECK(sourceCount > 0, "sourceCount is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(destinationCount > 0, "destinationCount is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(blockCount > 0, "blockCount is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(sourceCount == cumSumCount, "src&cum's shape must be same ",
        return Status::FailStatus(ERROR_INVALID_VALUE));

    BlockCopyTilingData *tilingDataPtr = reinterpret_cast<AtbOps::BlockCopyTilingData*>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingHost should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingHost should not be empty"));

    tilingDataPtr->blockCount = blockCount;
    tilingDataPtr->sourceCount = sourceCount;
    tilingDataPtr->destinationCount = destinationCount;

    TensorDType inDtype = launchParam.GetInTensor(0).desc.dtype;
    uint32_t typeByte = static_cast<uint32_t>(GetTensorElementSize(inDtype));
    tilingDataPtr->typeByte = typeByte;

    uint32_t tilingKey = TILING_DTYPE_IDX * typeByte;

    uint64_t maxCore = static_cast<uint64_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    uint32_t actualCore = destinationCount < maxCore ? destinationCount : maxCore;

    tilingDataPtr->blockDim = actualCore;

    tilingDataPtr->perCoreCopyCount = destinationCount / actualCore;
    tilingDataPtr->tailCoreCopyCount = destinationCount % actualCore;

    MKI_LOG(INFO) << "blockCount: " << blockCount << ", sourceCount: " << sourceCount
                  << ", destinationCount: " << destinationCount;
    kernelInfo.SetBlockDim(actualCore);
    kernelInfo.SetTilingId(tilingKey);
    return Status::OkStatus();
}
} // namespace AtbOps
