/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "laser_attention_tiling.h"
#include <iostream>
#include "atbops/params/params.h"
#include "mki/utils/const/op_const.h"
#include "mki/utils/log/log.h"
#include "mki/utils/math/math.h"
#include "mki/utils/checktensor/check_tensor.h"
#include "mki/utils/platform/platform_info.h"
#include "tiling_data.h"

namespace AtbOps {
using namespace Mki;
constexpr size_t RESERVE_SIZE = 100 * 1024 * 1024;

constexpr int32_t FLOAT_SIZE = sizeof(float);
constexpr int32_t DOUBLE_BUFFER = 2;
constexpr int32_t BASE_BLOCK_SIDE_LEN = 128;

constexpr int32_t ROW_SIZE_8K = 1024 * 8;
constexpr int32_t ROW_SIZE_16K = 1024 * 16;
constexpr int32_t ROW_SIZE_32K = 1024 * 32;

constexpr int32_t AI_CORE_CNT_910B = 24;
constexpr int32_t AI_CORE_CNT_910B1 = 25;
constexpr int32_t AI_CORE_CNT_910B3 = 20;

constexpr int32_t ONE_CORE_PER_GROUP = 1;
constexpr int32_t TWO_CORE_PER_GROUP = 2;
constexpr int32_t FOUR_CORE_PER_GROUP = 4;
constexpr int32_t EIGHT_CORE_PER_GROUP = 8;

constexpr size_t CONST_ONES_SIZE = 16384 * 2;

constexpr int32_t SINGLE_GROUP = 1;

constexpr int32_t LOW_PRECISION_FACTOR = 1;
constexpr int32_t HIGH_PRECISION_FACTOR = 2;

constexpr int32_t ATTEN_TYPE_MHA = 0;
constexpr int32_t ATTEN_TYPE_GQA = 1;

constexpr int32_t SPARSE_MODE_DENSE = 0;
constexpr int32_t SPARSE_MODE_SPARSE = 1;

constexpr int32_t EMPTY_WIN_LEN = 0;

constexpr uint64_t TILING_ID_BF16 = 1;
constexpr uint64_t TILING_ID_FLOAT16 = 0;

constexpr size_t QUERY_IN_TENSOR_INDEX = 0;
constexpr size_t KEY_IN_TENSOR_INDEX = 1;
constexpr size_t VALUE_IN_TENSOR_INDEX = 2;
constexpr size_t ATTEN_MASK_IN_TENSOR_INDEX = 5;

constexpr int32_t BLANK_SHAPE = 0;

constexpr int ERROR_NULL_PTR = 1;

constexpr int32_t ATTEN_OUT_INDEX = 9;
constexpr int32_t GM_BYTE_SIZE = 4;

constexpr int32_t DIM_192 = 192;
constexpr int32_t DIM_256 = 256;

enum class InputLayout {
    SBH = 0,
    BNSD = 3
};

void SetBlockDim(const uint32_t coreNum, KernelInfo &kernelInfo)
{
    if (coreNum == AI_CORE_CNT_910B1) {
        kernelInfo.SetBlockDim(AI_CORE_CNT_910B);
    } else {
        kernelInfo.SetBlockDim(coreNum);
    }
}

int32_t GetCoreNumPerGroup(const uint32_t coreNum, const int32_t kSeqLength, const int32_t isHighPrecision)
{
    int32_t coreNumPerGroup = SINGLE_GROUP;
    int32_t factor = isHighPrecision == 1 ? HIGH_PRECISION_FACTOR : LOW_PRECISION_FACTOR;
    if (kSeqLength <= ROW_SIZE_8K / factor) {
        coreNumPerGroup = ONE_CORE_PER_GROUP;
    } else if (kSeqLength > ROW_SIZE_8K / factor && kSeqLength <= ROW_SIZE_16K / factor) {
        coreNumPerGroup = TWO_CORE_PER_GROUP;
    } else if (kSeqLength > ROW_SIZE_16K / factor && kSeqLength < ROW_SIZE_32K / factor) {
        coreNumPerGroup = FOUR_CORE_PER_GROUP;
    } else {
        if (coreNum == AI_CORE_CNT_910B3) {
            coreNumPerGroup = FOUR_CORE_PER_GROUP;
        } else {
            coreNumPerGroup = EIGHT_CORE_PER_GROUP;
        }
    }
    return coreNumPerGroup;
}

int32_t GetAttenType(const LaunchParam &launchParam, const TensorDesc queryShape, const TensorDesc keyShape)
{
    auto param = AnyCast<OpParam::LaserAttention>(launchParam.GetParam());
    if (param.inputLayout == "BNSD") {
        return queryShape.dims[DIM_1] != keyShape.dims[DIM_1] ? ATTEN_TYPE_GQA : ATTEN_TYPE_MHA;
    } else if (param.inputLayout == "SBH") {
        return (queryShape.dims[DIM_2] / DIM_192) != (keyShape.dims[DIM_2] / DIM_256) ? ATTEN_TYPE_GQA : ATTEN_TYPE_MHA;
    }
    return 0;
}

int32_t GetHeadGroupSize(const LaunchParam &launchParam, const TensorDesc queryShape, const TensorDesc keyShape)
{
    auto param = AnyCast<OpParam::LaserAttention>(launchParam.GetParam());
    if (param.inputLayout == "BNSD") {
        return keyShape.dims[DIM_1] == BLANK_SHAPE ? BLANK_SHAPE : queryShape.dims[DIM_1] / keyShape.dims[DIM_1];
    } else if (param.inputLayout == "SBH") {
        return (keyShape.dims[DIM_2] / DIM_256) == BLANK_SHAPE ?
                   BLANK_SHAPE :
                   (queryShape.dims[DIM_2] / DIM_192) / (keyShape.dims[DIM_2] / DIM_256);
    }
    return 1;
}

int32_t GetSparseMode(const bool isTriangle, const int32_t preTokens, const int32_t seqSize)
{
    return isTriangle && preTokens < seqSize ? SPARSE_MODE_SPARSE : SPARSE_MODE_DENSE;
}

int32_t GetWindowLen(const bool isTriangle, const int32_t preTokens, const int32_t seqSize)
{
    int32_t windowLen = EMPTY_WIN_LEN;
    if (isTriangle) {
        if (preTokens < seqSize) {
            windowLen = preTokens;
        } else {
            windowLen = seqSize;
        }
    }
    return windowLen;
}

void SetWorkspace(KernelInfo &kernelInfo, const int32_t colSize, const int32_t coreGroupNum,
                  const LaserAttentionTilingData &tilingData)
{
    int32_t rowSumSize = tilingData.batchSize * tilingData.headNum * tilingData.qSeqLength * FLOAT_SIZE;

    uint64_t qSize = static_cast<uint64_t>(static_cast<int64_t>(tilingData.batchSize) * static_cast<int64_t>(tilingData.headNum) *
                                           static_cast<int64_t>(tilingData.qSeqLength) * DIM_192 * 2);
    uint64_t kSize = static_cast<uint64_t>(static_cast<int64_t>(tilingData.batchSize) * static_cast<int64_t>(tilingData.headNum) *
                                           static_cast<int64_t>(tilingData.kSeqLength) * DIM_256 * 2);
    uint64_t vSize = static_cast<uint64_t>(static_cast<int64_t>(tilingData.batchSize) * static_cast<int64_t>(tilingData.headNum) *
                                           static_cast<int64_t>(tilingData.kSeqLength) * 128 * 2);

    auto cube2WorkspaceSize = tilingData.batchSize * tilingData.headNum * tilingData.qSeqLength * 128 * FLOAT_SIZE;

    uint64_t sysWorkspaceSize = static_cast<uint64_t>(coreGroupNum) *
                                    static_cast<uint64_t>(colSize + BASE_BLOCK_SIDE_LEN) *
                                    static_cast<uint64_t>(BASE_BLOCK_SIDE_LEN * DOUBLE_BUFFER * FLOAT_SIZE) +
                                static_cast<uint64_t>(CONST_ONES_SIZE + rowSumSize + RESERVE_SIZE) +
                                static_cast<uint64_t>(qSize + kSize + vSize + cube2WorkspaceSize);
    kernelInfo.GetScratchSizes().push_back(sysWorkspaceSize);
}

void PrintTiling(const LaserAttentionTilingData &tilingData)
{
    MKI_LOG(INFO) << "tilingData: "
                  << "\ntilingData->batchSize:" << tilingData.batchSize
                  << "\ntilingData->headNum:" << tilingData.headNum << "\ntilingData->seqSize:" << tilingData.seqSize
                  << "\ntilingData->headDim:" << tilingData.headDim
                  << "\ntilingData->coreNumPerGroup:" << tilingData.coreNumPerGroup
                  << "\ntilingData->coreGroupNum:" << tilingData.coreGroupNum
                  << "\ntilingData->qSeqLength:" << tilingData.qSeqLength
                  << "\ntilingData->kSeqLength:" << tilingData.kSeqLength
                  << "\ntilingData->vSeqLength:" << tilingData.vSeqLength
                  << "\ntilingData->maskSeqLength:" << tilingData.maskSeqLength
                  << "\ntilingData->scale:" << tilingData.scale << "\ntilingData->keepProb:" << tilingData.keepProb
                  << "\ntilingData->preTokens:" << tilingData.preTokens
                  << "\ntilingData->nextTokens:" << tilingData.nextTokens
                  << "\ntilingData->attenType:" << tilingData.attenType
                  << "\ntilingData->sparseMode:" << tilingData.sparseMode
                  << "\ntilingData->headGroupSize:" << tilingData.headGroupSize
                  << "\ntilingData->windowLen:" << tilingData.windowLen
                  << "\ntilingData->isTriangle:" << tilingData.isTriangle
                  << "\ntilingData->inputLayout:" << tilingData.inputLayout
                  << "\ntilingData->isHighPrecision:" << tilingData.isHighPrecision;
}

Status LaserAttentionTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    uint32_t coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    SetBlockDim(coreNum, kernelInfo);

    auto tilingDataPtr = reinterpret_cast<LaserAttentionTilingData *>(kernelInfo.GetTilingHostAddr());
    if (tilingDataPtr == nullptr) {
        return Status::FailStatus(ERROR_NULL_PTR, "tiling data addr is nullptr");
    }

    LaserAttentionTilingData &tilingData = *tilingDataPtr;
    auto param = AnyCast<OpParam::LaserAttention>(launchParam.GetParam());

    const auto &queryShape = launchParam.GetInTensor(QUERY_IN_TENSOR_INDEX).desc;
    const auto &keyShape = launchParam.GetInTensor(KEY_IN_TENSOR_INDEX).desc;
    const auto &valueShape = launchParam.GetInTensor(VALUE_IN_TENSOR_INDEX).desc;

    Tensor maskTensor = launchParam.GetInTensor(ATTEN_MASK_IN_TENSOR_INDEX);
    if (!CheckEmptyTensor(maskTensor)) {
        const auto &maskShape = launchParam.GetInTensor(ATTEN_MASK_IN_TENSOR_INDEX).desc;
        tilingData.maskSeqLength = maskShape.dims[DIM_0];
    }

    int32_t seqSize = 0;
    int32_t kSeqLength = 0;

    if (param.inputLayout == "BNSD") {
        tilingData.batchSize = queryShape.dims[DIM_0];
        tilingData.headNum = param.headNum;
        seqSize = queryShape.dims[DIM_2];
        tilingData.seqSize = seqSize;
        tilingData.headDim = queryShape.dims[DIM_3];
        kSeqLength = keyShape.dims[DIM_2];
        tilingData.vSeqLength = valueShape.dims[DIM_2];
    } else if (param.inputLayout == "SBH") {
        tilingData.batchSize = queryShape.dims[1];
        tilingData.headNum = param.headNum;
        seqSize = queryShape.dims[0];
        tilingData.seqSize = seqSize;
        tilingData.headDim = DIM_192;
        kSeqLength = keyShape.dims[0];
        tilingData.vSeqLength = valueShape.dims[0];
    }

    int32_t isHighPrecision = param.innerPrecise;

    int32_t coreNumPerGroup = GetCoreNumPerGroup(coreNum, kSeqLength, isHighPrecision);
    tilingData.coreNumPerGroup = coreNumPerGroup;

    int32_t coreGroupNum = static_cast<int32_t>(coreNum) / coreNumPerGroup;
    tilingData.coreGroupNum = coreGroupNum;

    tilingData.qSeqLength = seqSize;
    tilingData.kSeqLength = kSeqLength;

    bool isTriangle = (launchParam.GetInTensor(ATTEN_MASK_IN_TENSOR_INDEX).data != nullptr);
    tilingData.isTriangle = isTriangle ? 1 : 0;

    tilingData.attenType = GetAttenType(launchParam, queryShape, keyShape);
    tilingData.headGroupSize = GetHeadGroupSize(launchParam, queryShape, keyShape);

    int32_t preTokens = param.preTokens;

    tilingData.sparseMode = GetSparseMode(isTriangle, preTokens, seqSize);

    tilingData.windowLen = GetWindowLen(isTriangle, preTokens, seqSize);

    tilingData.isHighPrecision = isHighPrecision;

    tilingData.scale = param.scaleValue;

    if (param.inputLayout == "BNSD") {
        tilingData.inputLayout = static_cast<int>(InputLayout::BNSD);
    } else if (param.inputLayout == "SBH") {
        tilingData.inputLayout = static_cast<int>(InputLayout::SBH);
    }

    int32_t colSize = tilingData.sparseMode == SPARSE_MODE_SPARSE ? tilingData.windowLen : kSeqLength;

    PrintTiling(tilingData);
    SetWorkspace(kernelInfo, colSize, coreGroupNum, tilingData);

    return Status::OkStatus();
}
} // namespace AtbOps
