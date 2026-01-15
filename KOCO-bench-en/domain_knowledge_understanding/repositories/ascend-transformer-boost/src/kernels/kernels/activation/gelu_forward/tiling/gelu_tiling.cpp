/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gelu_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "tiling_data.h"

namespace {
    constexpr uint32_t NUM_32 = 32;
    constexpr uint32_t NUM_2 = 2;
    constexpr uint32_t NUM_4 = 4;
}
namespace AsdOps {
using namespace Mki;
/**
核外
1. 数据能被core整除
2. 数据不能被core整除，留一个core处理余数

核内
3. 数据在core上能放的下
4. 数据在core上放不下，多次循环，不同的length
*/
bool FillTilingParam(const LaunchParam &launchParam, GeluForwardTilingData &tilingDataPtr, uint32_t &coreNum)
{
    tilingDataPtr.bufferNum = AsdOps::GELU_FORWARD_BUFF_NUM;
    // 获取可用核数
    coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    // UB空间大小,输入数据信息
    uint32_t maxUbSize = static_cast<uint32_t>(PlatformInfo::Instance().GetUbSize());
    auto type = launchParam.GetInTensor(0).desc.dtype;
    uint32_t inputSize = static_cast<uint32_t>(GetTensorElementSize(launchParam.GetInTensor(0).desc.dtype));
    uint32_t totalBufferNum = 1;
    // 每一个输入数据需要多少字节的空间
    if (type == Mki::TENSOR_DTYPE_FLOAT) {
        totalBufferNum = NUM_2 * AsdOps::GELU_FORWARD_BUFF_NUM * NUM_4;
    } else if (type == Mki::TENSOR_DTYPE_FLOAT16 || type == Mki::TENSOR_DTYPE_BF16) {
        totalBufferNum = NUM_2 * AsdOps::GELU_FORWARD_BUFF_NUM * (inputSize + NUM_4);
    } else {
        MKI_LOG(ERROR) << "type is not support";
        return false;
    }
    auto alignSize = NUM_32 / inputSize;
    // 单次搬运的最大数据量，32B向下对齐
    uint32_t maxPerElemBytes = (maxUbSize / totalBufferNum) / alignSize * alignSize;
    if (launchParam.GetInTensor(0).Numel() < (std::numeric_limits<uint32_t>::max() - alignSize + 1)) {
        MKI_LOG(ERROR) << "Numel is not invalid";
        return false;
    }
    auto alignDataLen = (static_cast<uint32_t>(launchParam.GetInTensor(0).Numel()) + alignSize  - 1) /
         alignSize * alignSize;
    // 判断是否为小shape来决定是否重置单次搬运数据
    if (alignDataLen <= maxPerElemBytes * GELU_FORWARD_BUFF_NUM * coreNum) {
        tilingDataPtr.bufferNum = 1;
        maxPerElemBytes = maxPerElemBytes * NUM_2;
    }

    tilingDataPtr.blockLength = (static_cast<uint32_t>(launchParam.GetInTensor(0).Numel()) + coreNum - 1) / coreNum;
    tilingDataPtr.blockLength = (tilingDataPtr.blockLength + alignSize - 1) / alignSize * alignSize;
    // 每个核要算的数据能否在UB上放的下
    tilingDataPtr.tileNum = tilingDataPtr.blockLength / maxPerElemBytes;
    tilingDataPtr.tailLength = tilingDataPtr.blockLength % maxPerElemBytes;
    if (tilingDataPtr.tileNum == 0) {
        tilingDataPtr.tileLength = tilingDataPtr.tailLength;
    } else {
        tilingDataPtr.tileLength = maxPerElemBytes;
    }
    return true;
}
Status GeluForwardTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    uint32_t blockDim = 0;
    int32_t dataType = launchParam.GetInTensor(0).desc.dtype;
    GeluForwardTilingData *tilingDataPtr =
        reinterpret_cast<GeluForwardTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr should not be empty"));
    MKI_CHECK(FillTilingParam(launchParam, *tilingDataPtr, blockDim), "FillTilingParam Failed.",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "FillTilingParam Failed."));
    kernelInfo.SetBlockDim(blockDim);
    kernelInfo.SetTilingId(dataType); // 不同的数据类型用不同的分核策略，所以暂时用数据类型的枚举来表示分核ID
    MKI_LOG(INFO) << "Tiling Data: blockLength " << tilingDataPtr->blockLength << ", tileLength "
                  << tilingDataPtr->tileLength << ", tileNum: " << tilingDataPtr->tileNum
                  << ", tailLength " << tilingDataPtr->tailLength;

    return Status::OkStatus();
}
} // namespace AsdOps
