/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faster_gelu_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "tiling_data.h"

namespace AsdOps {
using namespace Mki;
constexpr uint32_t PACK_SIZE = 512; // L2 Cache 搬运对齐约束
constexpr uint32_t TBUF_NUM = 2;
constexpr uint32_t TQUE_NUM = 2;
constexpr uint32_t UB_RESERVED_BUFF = 0; // reserve ubSize
constexpr uint32_t ALIGN_SIZE = 32;

void CalcVectorTiling512Align(const LaunchParam &launchParam, FasterGeluForwardTilingData &tilingDataPtr,
                              uint32_t &usedCoreNum)
{
    uint64_t dataLen = static_cast<uint64_t>(launchParam.GetInTensor(0).Numel());
    uint32_t totalCore = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    if (totalCore > MAX_CORE_SIZE) {
        MKI_LOG(WARN) << "faster_gelu_forward can not support soc that VEC CoreNum exceed 128. The Max Core Num will "
                         "be 128 instead";
        totalCore = MAX_CORE_SIZE;
    }
    uint64_t ubSize = PlatformInfo::Instance().GetUbSize();
    uint32_t inputDTypeSize = static_cast<uint32_t>(GetTensorElementSize(launchParam.GetInTensor(0).desc.dtype));
    uint32_t alignNum = ALIGN_SIZE / inputDTypeSize;
    uint32_t packLen = PACK_SIZE / inputDTypeSize;

    // 向上对齐 ALIGN_SIZE  这里直接把数据长度向上32B对齐
    uint64_t dataLenAlign = (dataLen + alignNum - 1) / alignNum * alignNum;
    uint32_t alignDataNum = dataLenAlign - dataLen;

    // 开启DB
    uint32_t bufferNum = AsdOps::FASTER_GELU_FORWARD_BUFF_NUM;
    uint64_t dataLenPerUB = 1;

    // 计算UB单元素计算使用的字节数 和TBuf和TQue有关
    uint32_t singleDataSize = 0;
    // singleDataSize = bufferNum * TQUE_NUM * inputDTypeSize + TBuf_NUM * 实际的DtypeLen
    if (inputDTypeSize == sizeof(float)) {
        singleDataSize = bufferNum * TQUE_NUM * inputDTypeSize;
    } else {
        singleDataSize = bufferNum * TQUE_NUM * inputDTypeSize + TBUF_NUM * sizeof(float);
    }

    ubSize -= UB_RESERVED_BUFF;
    // 计算UB单次迭代最大可计算的32字节对齐的元素个数
    dataLenPerUB = ubSize / singleDataSize;
    uint64_t availableUB = dataLenPerUB / alignNum * alignNum;

    // 评估需要的核数
    if (dataLenAlign <= availableUB) {
        usedCoreNum = 1;
    } else if (dataLenAlign > availableUB && dataLenAlign <= availableUB * totalCore) {
        usedCoreNum = (dataLenAlign + availableUB - 1) / availableUB;
    } else {
        usedCoreNum = totalCore;
    }
    // 每个核计算的block_length 最均匀的分法
    uint64_t baseBlockLength = dataLenAlign / (usedCoreNum * packLen) * packLen; // 搬运向下512B对齐
    uint64_t resDataLenAlign = dataLenAlign - usedCoreNum * baseBlockLength;
    tilingDataPtr.usedCoreNum = usedCoreNum;
    std::fill(tilingDataPtr.singleCoreDataLen, tilingDataPtr.singleCoreDataLen + usedCoreNum, baseBlockLength);
    uint32_t index = 0;
    for (uint32_t i = packLen; i <= resDataLenAlign; i += packLen) {
        tilingDataPtr.singleCoreDataLen[index % usedCoreNum] += packLen;
        index++;
    }
    tilingDataPtr.singleCoreDataLen[usedCoreNum - 1] += resDataLenAlign % packLen;

    tilingDataPtr.maxTileLen =
        availableUB > tilingDataPtr.singleCoreDataLen[0] ? tilingDataPtr.singleCoreDataLen[0] : availableUB;

    // 如果只用一个核心，对齐数量置为0，防止核内计算偏移时访问非法内存
    tilingDataPtr.alignDataNum = usedCoreNum > 1 ? alignDataNum : 0;
}

Status FasterGeluForwardTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    uint32_t blockDim = 0;
    uint64_t dataType = launchParam.GetInTensor(0).desc.dtype;
    FasterGeluForwardTilingData *tilingDataPtr =
        reinterpret_cast<FasterGeluForwardTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr should not be empty"));
    CalcVectorTiling512Align(launchParam, *tilingDataPtr, blockDim);

    for (uint32_t i = 0; i < tilingDataPtr->usedCoreNum; i++) {
        MKI_LOG(INFO) << "Core-" << i << "  singleCoreDataLen num is " << tilingDataPtr->singleCoreDataLen[i];
    }
    kernelInfo.SetBlockDim(blockDim);
    kernelInfo.SetTilingId(dataType);
    return Status::OkStatus();
}
} // namespace AsdOps
