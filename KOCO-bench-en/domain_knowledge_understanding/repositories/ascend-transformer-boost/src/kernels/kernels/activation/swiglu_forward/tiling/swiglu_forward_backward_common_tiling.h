/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_SWIGLU_FORWARD_BACKWARD_COMMOND_TILING_H
#define OPS_SWIGLU_FORWARD_BACKWARD_COMMOND_TILING_H

#include <limits>
#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include "kernels/activation/swiglu_backward/tiling/tiling_data.h"
#include "kernels/activation/swiglu_forward/tiling/tiling_data.h"


namespace AsdOps {
using namespace Mki;
constexpr uint32_t SINGLE_UB_SIZE_FP32 = 24;
constexpr uint32_t SINGLE_UB_SIZE_NFP32 = 18;
constexpr uint32_t SINGLE_UB_SIZE_FP32_GRAD = 44;
constexpr uint32_t SINGLE_UB_SIZE_NFP32_GRAD = 34;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_CORE_NUMBER = 64;
constexpr uint32_t UB_RESERVED_BUFF = 0;
constexpr uint32_t INSTRUCTION_LIMIT = 4095;

template<typename T>
T AlignUp(T num, T div)
{
    return (div == 0) ? 0 : (num + div - 1) / div * div;
}
template<typename T>
inline T AlignDown(T num, T div)
{
    return (div == 0) ? 0 : num / div * div;
}

template<typename T>
inline T CeilDiv(T num, T div)
{
    return div == 0 ? 0 : (num + div - 1) / div;
}

template<typename T>
inline T Min(T num, T div)
{
    return num < div ? num : div;
}

inline uint32_t GetTypeLength(int32_t dtype)
{
    switch (dtype) {
        case TENSOR_DTYPE_FLOAT16:
        case TENSOR_DTYPE_BF16:
            return sizeof(int16_t);
        case TENSOR_DTYPE_FLOAT:
            return sizeof(int32_t);
        default:
            return 0;
    }
}

template<typename T>
inline bool SetTotalShape(const SVector<int64_t>& inShape, const int32_t inDim, T tilingData)
{
    int64_t shapeBefore = 1;
    int64_t shapeAfter = 1;
    int32_t dimNum = static_cast<int32_t>(inShape.size());
    for (int32_t i = 0; i < inDim; i++) { // inDim >= 0 && inDim < xShape.size()
        MKI_CHECK(shapeBefore < std::numeric_limits<int64_t>::max() / inShape[i] && inShape[i] > 0,
                "inShape[i] is not invalid", return false);
        shapeBefore *= inShape[i];
    }
    for (int32_t j = inDim; j < dimNum; j++) {
        MKI_CHECK(shapeAfter < std::numeric_limits<int64_t>::max() / inShape[j] && inShape[j] > 0,
                "inShape[j] is not invalid", return false);
        shapeAfter *= inShape[j];
    }
    // 如果shape不是2的倍数,返回
    if (shapeBefore <= 0 || shapeAfter <= 0 || shapeAfter % 2 != 0) {
        MKI_LOG(ERROR) << "SwiGlu SetTotalShape Unsupported inDim " + std::to_string(inDim) +
                          ", shapeBefore " + std::to_string(shapeBefore) + ", shapeAfter " +
                          std::to_string(shapeAfter) + " %% 2 != 0";
        return false;
    }
    MKI_CHECK(shapeBefore <= std::numeric_limits<uint32_t>::max(), "rowLen exceeds the upper limit",
        return false);
    MKI_CHECK(shapeAfter <= std::numeric_limits<uint32_t>::max(), "colLen exceeds the upper limit",
        return false);
    tilingData->rowLen = static_cast<uint32_t>(shapeBefore);
    // colLen为原shape除以2
    tilingData->colLen = static_cast<uint32_t>(shapeAfter) / 2;
    return true;
}

// 计算每核处理的数据情况
template<typename T>
inline uint64_t CalDataPerCore(uint32_t totalCore, int32_t dataType, T tilingData, KernelInfo &kernelInfo)
{
    uint32_t coreNumUsed = 0; // 实际使用的核数
    // 每核处理多少数据
    uint64_t dataNumPerCore = CeilDiv(static_cast<uint64_t>(tilingData->rowLen) * tilingData->colLen,
        static_cast<uint64_t>(totalCore));
    if (dataNumPerCore == 0) { // 处理边界情况
        dataNumPerCore = tilingData->colLen;
    }
    if (dataNumPerCore >= tilingData->colLen) { // 一核处理多行
        tilingData->rowLenPerCore = CeilDiv(dataNumPerCore, static_cast<uint64_t>(tilingData->colLen)); // 每核计算多少行
        tilingData->colLenPerCore = tilingData->colLen; // 每核计算多少列
        tilingData->coreNumPerLine = 1;
        dataNumPerCore = tilingData->rowLenPerCore * tilingData->colLenPerCore;
        coreNumUsed = CeilDiv(tilingData->rowLen, tilingData->rowLenPerCore);
    } else { // 一行被多核处理
        uint64_t coreNumPerLine = totalCore / tilingData->rowLen; // 一行被多少核处理
        dataNumPerCore = CeilDiv(static_cast<uint64_t>(tilingData->colLen), coreNumPerLine);
        uint32_t typeLength = GetTypeLength(dataType);
        dataNumPerCore = AlignUp(dataNumPerCore, static_cast<uint64_t>((typeLength == 0) ? 0 :
            (BLOCK_SIZE / typeLength)));
        tilingData->rowLenPerCore = 1; // 每核计算多少行
        tilingData->colLenPerCore = dataNumPerCore; // 每核计算多少列
        coreNumPerLine = CeilDiv(tilingData->colLen, tilingData->colLenPerCore); // 更新值：一行被多少核处理
        coreNumUsed = coreNumPerLine * tilingData->rowLen;
        tilingData->coreNumPerLine = coreNumPerLine;
    }
    kernelInfo.SetBlockDim(coreNumUsed);
    return dataNumPerCore;
}

// 处理每核内的循环情况
template<typename T>
inline bool CalDataPerLoop(uint32_t ubSize, uint64_t dataNumPerCore, int32_t dataType, T tilingData, bool isGrad)
{
    // ub空间情况
    uint32_t dataNumSingleUb = 0;
    if (isGrad) {
        if (dataType == TENSOR_DTYPE_FLOAT) { // 反向fp32
            dataNumSingleUb = ubSize / SINGLE_UB_SIZE_FP32_GRAD;
        } else { // 反向bf16 fp16
            dataNumSingleUb = ubSize / SINGLE_UB_SIZE_NFP32_GRAD;
        }
    } else {
        if (dataType == TENSOR_DTYPE_FLOAT) { // 反向fp32
            dataNumSingleUb = ubSize / SINGLE_UB_SIZE_FP32;
        } else { // 反向bf16 fp16
            dataNumSingleUb = ubSize / SINGLE_UB_SIZE_NFP32;
        }
    }
    uint32_t typeLength = GetTypeLength(dataType);
    dataNumSingleUb = AlignDown(dataNumSingleUb, (typeLength == 0) ? 0 : (BLOCK_SIZE / typeLength));
    // 算出的ub空间数据量可能大于1行，可能小于1行
    uint32_t dataLineSingleUb = dataNumSingleUb / AlignUp(tilingData->colLen, (typeLength == 0) ? 0 : (BLOCK_SIZE / typeLength));
    if (dataLineSingleUb == 0) { // ub空间容纳数据量少于1行
        dataLineSingleUb = 1;
        tilingData->basicRowLen = 1;
        uint32_t basicColLenTmp = AlignDown(Min(tilingData->colLenPerCore, dataNumSingleUb),
            (typeLength == 0) ? 0 : (BLOCK_SIZE / typeLength));
        if (basicColLenTmp > std::numeric_limits<uint16_t>::max()) {
            MKI_LOG(ERROR) << "SwiGlu basicColLen cannot cast to uint16";
            return false;
        }
        tilingData->basicColLen = static_cast<uint16_t>(basicColLenTmp);
    } else { // ub空间容纳数据量大于等于1行
        uint32_t basicRowLenTmp = Min(tilingData->rowLenPerCore, dataLineSingleUb);
        uint32_t basicColLenTmp = AlignDown(Min(static_cast<uint64_t>(tilingData->colLen), dataNumPerCore),
            static_cast<uint64_t>(((typeLength == 0) ? 0 : (BLOCK_SIZE / typeLength))));
        if (basicColLenTmp > std::numeric_limits<uint16_t>::max() || basicRowLenTmp > INSTRUCTION_LIMIT) {
            MKI_LOG(ERROR) << "SwiGlu basicColLen cannot cast to uint16 or basicRowLen > 4095";
            return false;
        }
        tilingData->basicRowLen = static_cast<uint16_t>(basicRowLenTmp);
        tilingData->basicColLen = static_cast<uint16_t>(basicColLenTmp);
    }
    if (tilingData->coreNumPerLine <= 0) {
        MKI_LOG(ERROR) << "SwiGlu coreNumPerLine <= 0";
        return false;
    }
    return true;
}
} // namespace AsdOps
#endif // OPS_SWIGLU_FORWARD_BACKWARD_COMMOND_TILING_H