/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "customize_blockcopy_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "customize_blockcopy/kernel_implement/include/customizeblockcopy.h"
#include "tiling_data.h"
static constexpr int32_t NZ_SIZE = 16;
static constexpr uint64_t BLOCKCOPY_WKSP_TENSOR_IDX = 7;
static constexpr uint64_t BLOCKCOPY_SYNC_WORKSPACE_TENSOR_IDX = 8;
static constexpr int32_t BYTES_32 = 32;
namespace AtbOps {
using namespace Mki;
/**
 * @brief 针对“ND”格式的 K/V Cache 计算 tilingData。
 *
 * “ND”格式：Tensor 形状为 [B, blockSize, numHead, headSize]。
 * 本函数读取输入维度并校验合法性，然后填充 blockSize、numHead、
 * headSizeK、headSizeV 等字段到 tilingData 中。
 *
 * @param[in]  launchParam   运行时输入信息
 * @param[out] kernelInfo    填充后的 kernelInfo，其中包含 tilingData
 * @return bool              成功返回 true，失败返回 false
 */
bool BlockCopyTilingNd(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    CustomizeBlockCopyTilingData *tilingDataPtr =
        reinterpret_cast<AtbOps::CustomizeBlockCopyTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingHost should not be empty", return false);
    auto &kShape = launchParam.GetInTensor(0).desc.dims;
    auto &vShape = launchParam.GetInTensor(1).desc.dims;
    // 910B kv shape [#B, B, head, DIM]
    uint32_t blockSize = static_cast<uint32_t>(kShape.at(1));
    uint32_t numHead = static_cast<uint32_t>(kShape.at(2));
    uint32_t headSizeK = static_cast<uint32_t>(kShape.at(3));
    uint32_t headSizeV = static_cast<uint32_t>(vShape.at(3));
    MKI_CHECK(blockSize > 0, "blockSize is invalid", return false);
    MKI_CHECK(numHead > 0, "numHead is invalid", return false);
    tilingDataPtr->blockSize = blockSize;
    tilingDataPtr->numHead = numHead;
    MKI_CHECK(headSizeK > 0, "headSizeK is invalid", return false);
    MKI_CHECK(headSizeV > 0, "headSizeV is invalid", return false);
    tilingDataPtr->headSizeK = headSizeK;
    tilingDataPtr->headSizeV = headSizeV;
    return true;
}
/**
 * @brief 针对“NZ”格式的 K/V Cache 计算 tilingData。
 *
 * “NZ”格式：310P 特殊布局，Tensor 形状为
 * [B, head*headSize/16, blockSize, 16]，
 * 并强制 numHead=1。根据 dim1*16 计算 headSizeK/V。
 *
 * @param[in]  launchParam   运行时输入信息
 * @param[out] kernelInfo    填充后的 kernelInfo，其中包含 tilingData
 * @return bool              成功返回 true，失败返回 false
 */
bool BlockCopyTilingNz(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    CustomizeBlockCopyTilingData *tilingDataPtr =
        reinterpret_cast<AtbOps::CustomizeBlockCopyTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingHost should not be empty", return false);
    auto &kShape = launchParam.GetInTensor(0).desc.dims;
    auto &vShape = launchParam.GetInTensor(1).desc.dims;
    // nz：无法获取头和头大小；numhead弃置，改用headSizeV来保存大小
    // 310p的kv：[#B, head*head_size // 16, B, 16]
    uint32_t blockSize = static_cast<uint32_t>(kShape.at(2));
    // 头参数固定设置为1
    uint32_t numHead = static_cast<uint32_t>(1);
    uint32_t headSizeK = static_cast<uint32_t>(kShape.at(1) * NZ_SIZE);
    uint32_t headSizeV = static_cast<uint32_t>(vShape.at(1) * NZ_SIZE);
    MKI_CHECK(blockSize > 0, "blockSize is invalid", return false);
    MKI_CHECK(numHead > 0, "numHead is invalid", return false);
    MKI_CHECK(headSizeK > 0, "headSizeK is invalid", return false);
    MKI_CHECK(headSizeV > 0, "headSizeV is invalid", return false);
    tilingDataPtr->blockSize = blockSize;
    tilingDataPtr->numHead = numHead;
    tilingDataPtr->headSizeK = headSizeK;
    tilingDataPtr->headSizeV = headSizeV;
    return true;
}
/**
 * @brief 仅在 310P 平台上检查 K/V Cache 总元素数是否与 typeByte 对齐。
 *
 * 防止后续的 vector 访问越界，要求按 32 字节对齐。
 *
 * @param[in]  launchParam   运行时输入信息
 * @param[out] kernelInfo    kernel 调度信息（仅读取 tilingData）
 * @return bool              对齐检查通过返回 true，否则返回 false
 */
bool BlockCopyTilingCheck310P(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    TensorDType inDtype = launchParam.GetInTensor(0).desc.dtype;
    uint32_t typeByte = static_cast<uint32_t>(GetTensorElementSize(inDtype));
    CustomizeBlockCopyTilingData *tilingDataPtr =
        reinterpret_cast<AtbOps::CustomizeBlockCopyTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingHost should not be empty", return false);
    auto headDim = tilingDataPtr->blockSize * tilingDataPtr->numHead;
    auto totalKCacheItems = tilingDataPtr->headSizeK * headDim;
    auto totalVCacheItems = tilingDataPtr->headSizeV * headDim;
    MKI_CHECK(totalKCacheItems % typeByte == 0, "310P Platform KCache should be aligned with 32 bytes!", return false);
    MKI_CHECK(totalVCacheItems % typeByte == 0, "310P Platform VCache should be aligned with 32 bytes!", return false);
    return true;
}

Status CustomizeBlockCopyTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto param = AnyCast<OpParam::CustomizeBlockCopy>(launchParam.GetParam());
    auto &kShape = launchParam.GetInTensor(0).desc.dims;
    auto &srcShape = launchParam.GetInTensor(2).desc.dims;
    auto &dstShape = launchParam.GetInTensor(3).desc.dims;
    auto &cumShape = launchParam.GetInTensor(4).desc.dims;
    uint32_t blockCount = static_cast<uint32_t>(kShape.at(0));
    uint32_t sourceCount = static_cast<uint32_t>(srcShape.at(0));
    uint32_t destinationCount = static_cast<uint32_t>(dstShape.at(0));
    uint32_t cumSumCount = static_cast<uint32_t>(cumShape.at(0));

    if (param.type == OpParam::CustomizeBlockCopy::BLOCK_COPY_CACHE_ND) {
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

    MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::CustomizeBlockCopy), "OpParam is invalid",
              return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
    MKI_CHECK(sourceCount > 0, "sourceCount is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(destinationCount > 0, "destinationCount is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(blockCount > 0, "blockCount is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(sourceCount == cumSumCount, "src&cum's shape must be same ",
              return Status::FailStatus(ERROR_INVALID_VALUE));

    CustomizeBlockCopyTilingData *tilingDataPtr =
        reinterpret_cast<AtbOps::CustomizeBlockCopyTilingData *>(kernelInfo.GetTilingHostAddr());
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
