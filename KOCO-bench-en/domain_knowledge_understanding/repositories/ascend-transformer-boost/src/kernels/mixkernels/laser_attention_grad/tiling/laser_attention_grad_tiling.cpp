/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "laser_attention_grad_tiling.h"
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
constexpr size_t RESERVE_SIZE = 16 * 1024 * 1024;
constexpr size_t RESERVE_SIZE_END = 10 * 1024 * 1024;

constexpr int32_t ROW_SIZE_8K = 8192;
constexpr int32_t AI_CORE_CNT_910B = 24;
constexpr int32_t AI_CORE_CNT_910B1 = 25;
constexpr int32_t AI_CORE_CNT_910B3 = 20;

constexpr int32_t FLOAT_SIZE = sizeof(float);
constexpr int32_t DOUBLE_BUFFER = 2;
constexpr int32_t BASE_BLOCK_SIDE_LEN = 128;

constexpr int32_t BLOCK_BUM_PER_CORE_16 = 16;
constexpr int32_t BLOCK_BUM_PER_CORE_32 = 16;

constexpr int32_t SPARSE_MODE_DENSE = 0;
constexpr int32_t SPARSE_MODE_SPARSE = 1;

constexpr int32_t ATTEN_TYPE_MHA = 0;
constexpr int32_t ATTEN_TYPE_GQA = 1;

constexpr int32_t EMPTY_WIN_LEN = 0;

constexpr uint64_t TEMP_SPACE_FACTOR = 2;

constexpr uint64_t TILING_ID_FLOAT16 = 0;
constexpr uint64_t TILING_ID_FLOAT16_BANDOP = 1;
constexpr uint64_t TILING_ID_BF16 = 2;
constexpr uint64_t TILING_ID_BF16_BANDOP = 3;

constexpr size_t QUERY_IN_TENSOR_INDEX = 0;
constexpr size_t KEY_IN_TENSOR_INDEX = 1;
constexpr size_t VALUE_IN_TENSOR_INDEX = 2;
constexpr size_t ATTEN_MASK_IN_TENSOR_INDEX = 6;

constexpr int32_t BLANK_SHAPE = 0;

constexpr int ERROR_NULL_PTR = 1;

constexpr int32_t QUERY_GRAD_INDEX = 11;
constexpr int32_t KEY_GRAD_INDEX = 12;
constexpr int32_t VALUE_GRAD_INDEX = 13;
constexpr int32_t WORKSPACE_INDEX = 14;
constexpr int32_t GM_BYTE_SIZE = 4;

constexpr int32_t ALIGN_BYTES = 32;

constexpr uint32_t UB_RESERVE_SIZE = 1024;
constexpr uint32_t UB_DIVIDE_FACTOR = 6;
constexpr uint32_t BF16_DTYPE_SIZE = 2;

constexpr int32_t HEAD_DIM_128 = 128;
constexpr int32_t HEAD_DIM_192 = 192;
constexpr int32_t HEAD_DIM_256 = 256;

enum class InputLayout {
    SBH = 0,
    BNSD = 3
};

uint32_t GetAicNum(const uint32_t coreNum)
{
    return coreNum == AI_CORE_CNT_910B1 ? AI_CORE_CNT_910B : coreNum;
}

int32_t GetBlockNumPerCore(const int32_t seqSize)
{
    return BLOCK_BUM_PER_CORE_16;
}

int32_t GetAttenTypeGrad(const LaunchParam &launchParam, const TensorDesc queryShape, const TensorDesc keyShape)
{
    auto param = AnyCast<OpParam::LaserAttentionGrad>(launchParam.GetParam());
    if (param.inputLayout == "BNSD") {
        return queryShape.dims.at(DIM_1) != keyShape.dims.at(DIM_1) ? ATTEN_TYPE_GQA : ATTEN_TYPE_MHA;
    } else if (param.inputLayout == "SBH") {
        return (queryShape.dims[DIM_2] / HEAD_DIM_192) != (keyShape.dims[DIM_2] / HEAD_DIM_256) ? ATTEN_TYPE_GQA :
                                                                                                  ATTEN_TYPE_MHA;
    }
    return static_cast<int>(InputLayout::BNSD);
}

int32_t GetSparseModeGrad(const bool isTriangle, const int32_t preTokens, const int32_t seqSize)
{
    return isTriangle && preTokens < seqSize ? SPARSE_MODE_SPARSE : SPARSE_MODE_DENSE;
}

int32_t GetHeadGroupSizeGrad(const LaunchParam &launchParam, const TensorDesc queryShape, const TensorDesc keyShape)
{
    auto param = AnyCast<OpParam::LaserAttentionGrad>(launchParam.GetParam());
    if (param.inputLayout == "BNSD") {
        return keyShape.dims[DIM_1] == BLANK_SHAPE ? BLANK_SHAPE : queryShape.dims[DIM_1] / keyShape.dims[DIM_1];
    } else if (param.inputLayout == "SBH") {
        return (keyShape.dims[DIM_2] / HEAD_DIM_256) == BLANK_SHAPE ?
                   BLANK_SHAPE :
                   (queryShape.dims[DIM_2] / HEAD_DIM_192) / (keyShape.dims[DIM_2] / HEAD_DIM_256);
    }
    return 1;
}

int32_t GetWindowLenGrad(const bool isTriangle, const int32_t preTokens, const int32_t seqSize)
{
    int32_t windowLength = EMPTY_WIN_LEN;
    if (isTriangle) {
        if (preTokens < seqSize) {
            windowLength = preTokens;
        } else {
            windowLength = seqSize;
        }
    }
    return windowLength;
}

void SetWorkspace(KernelInfo &kernelInfo, const uint32_t aicNum, const int32_t blockNumPerCore,
                  LaserAttentionGradTilingData &tilingData, uint32_t keyHeadNum, int32_t keyHeadDim,
                  int32_t valueHeadDim)
{
    uint32_t batchSize = static_cast<uint32_t>(tilingData.batchSize);
    uint32_t headNum = static_cast<uint32_t>(tilingData.headNum);
    uint32_t qSeqLength = static_cast<uint32_t>(tilingData.qSeqLength);
    uint32_t kSeqLength = static_cast<uint32_t>(tilingData.kSeqLength);
    uint32_t headDim = static_cast<uint32_t>(tilingData.headDim);

    uint64_t querySize = 0;
    uint64_t keySize = 0;
    uint64_t valueSize = 0;

    auto outputHead = (tilingData.headNum + tilingData.headGroupSize - 1) /
                      tilingData.headGroupSize; // gqa时output_head=head/group_size,其他情景下output_head=head
    uint64_t qSize = static_cast<uint64_t>(static_cast<int64_t>(tilingData.batchSize) * static_cast<int64_t>(tilingData.headNum) *
                                           static_cast<int64_t>(tilingData.qSeqLength) * HEAD_DIM_192 * 2);
    uint64_t kSize = static_cast<uint64_t>(static_cast<int64_t>(tilingData.batchSize) * static_cast<int64_t>(outputHead) *
                                           static_cast<int64_t>(tilingData.kSeqLength) * HEAD_DIM_256 * 2);
    uint64_t vSize = static_cast<uint64_t>(static_cast<int64_t>(tilingData.batchSize) * static_cast<int64_t>(outputHead) *
                                           static_cast<int64_t>(tilingData.kSeqLength) * 128 * 2);
    uint64_t atentionScoreSize = static_cast<uint64_t>(static_cast<int64_t>(tilingData.batchSize) *
                                                       static_cast<int64_t>(tilingData.headNum) *
                                                       static_cast<int64_t>(tilingData.qSeqLength) * 128 * 2);

    if (headDim == HEAD_DIM_128) {
        querySize = GM_BYTE_SIZE * batchSize * headNum * qSeqLength * headDim;
        keySize = GM_BYTE_SIZE * batchSize * keyHeadNum * kSeqLength * headDim;
        valueSize = GM_BYTE_SIZE * batchSize * keyHeadNum * kSeqLength * headDim;
    } else if (headDim == HEAD_DIM_192) {
        uint32_t kHeadDim = static_cast<uint32_t>(keyHeadDim);
        uint32_t vHeadDim = static_cast<uint32_t>(valueHeadDim);
        querySize = GM_BYTE_SIZE * batchSize * headNum * qSeqLength * headDim;
        keySize = GM_BYTE_SIZE * batchSize * keyHeadNum * kSeqLength * kHeadDim;
        valueSize = GM_BYTE_SIZE * batchSize * keyHeadNum * kSeqLength * vHeadDim;
    }

    uint64_t orgWorkspaceSize = aicNum * blockNumPerCore * BASE_BLOCK_SIDE_LEN * BASE_BLOCK_SIDE_LEN * DOUBLE_BUFFER *
                                    FLOAT_SIZE * TEMP_SPACE_FACTOR +
                                RESERVE_SIZE + RESERVE_SIZE_END;
    uint64_t sysWorkspaceSize =
        querySize + keySize + valueSize + orgWorkspaceSize + qSize + kSize + vSize + atentionScoreSize * 2;

    tilingData.outputWorkspaceOffset = static_cast<int64_t>(orgWorkspaceSize) / FLOAT_SIZE;
    tilingData.queryShape = static_cast<int64_t>(querySize) / GM_BYTE_SIZE;
    tilingData.keyShape = static_cast<int64_t>(keySize) / GM_BYTE_SIZE;
    tilingData.valueShape = static_cast<int64_t>(valueSize) / GM_BYTE_SIZE;

    MKI_LOG(INFO) << "aicNum " << aicNum << "blockNumPerCore " << blockNumPerCore << "outputWorkspaceOffset "
                  << tilingData.outputWorkspaceOffset << "queryShape " << tilingData.queryShape << "keyShape "
                  << tilingData.keyShape << "valueShape " << tilingData.valueShape;
    kernelInfo.GetScratchSizes().push_back(sysWorkspaceSize);

    // 清空dq, dk, dv对应的workspace
    kernelInfo.SetMemsetInfo(WORKSPACE_INDEX, RESERVE_SIZE + querySize + keySize + valueSize);
}

int64_t AlignDown(int64_t num, int64_t rnd)
{
    return ((rnd) == 0 ? 0 : ((num / rnd) * rnd));
}

void SetPostTiling(LaserAttentionGradTilingData &tilingData)
{
    uint32_t ubSize = PlatformInfo::Instance().GetUbSize();
    uint32_t ubSizeLeft = ubSize - UB_RESERVE_SIZE;
    auto baseNum = ubSizeLeft / UB_DIVIDE_FACTOR / DOUBLE_BUFFER;
    auto baseNumAlign = AlignDown(baseNum, ALIGN_BYTES / BF16_DTYPE_SIZE);
    tilingData.postBaseNum = baseNumAlign;

    int64_t coreNum = static_cast<int64_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));

    // tiling for dq
    tilingData.postDqFrontCoreNum = tilingData.queryShape % coreNum;
    tilingData.postDqTailCoreNum = coreNum - tilingData.postDqFrontCoreNum;
    tilingData.postDqFrontDataNum = (tilingData.queryShape + coreNum - 1) / coreNum;
    tilingData.postDqTailDataNum = tilingData.queryShape / coreNum;

    // tiling for dk
    tilingData.postDkFrontCoreNum = tilingData.keyShape % coreNum;
    tilingData.postDkTailCoreNum = coreNum - tilingData.postDkFrontCoreNum;
    tilingData.postDkFrontDataNum = (tilingData.keyShape + coreNum - 1) / coreNum;
    tilingData.postDkTailDataNum = tilingData.keyShape / coreNum;

    // tiling for dv
    tilingData.postDvFrontCoreNum = tilingData.valueShape % coreNum;
    tilingData.postDvTailCoreNum = coreNum - tilingData.postDvFrontCoreNum;
    tilingData.postDvFrontDataNum = (tilingData.valueShape + coreNum - 1) / coreNum;
    tilingData.postDvTailDataNum = tilingData.valueShape / coreNum;
}

void PrintTiling(const LaserAttentionGradTilingData &tilingData)
{
    MKI_LOG(INFO) << "tilingData: "
                  << "\ntilingData->batchSize:" << tilingData.batchSize
                  << "\ntilingData->headNum:" << tilingData.headNum << "\ntilingData->seqSize:" << tilingData.seqSize
                  << "\ntilingData->headDim:" << tilingData.headDim << "\ntilingData->blockDim:" << tilingData.blockDim
                  << "\ntilingData->blockNumPerCore:" << tilingData.blockNumPerCore
                  << "\ntilingData->attenType:" << tilingData.attenType
                  << "\ntilingData->sparseMode:" << tilingData.sparseMode
                  << "\ntilingData->headGroupSize:" << tilingData.headGroupSize
                  << "\ntilingData->windowLen:" << tilingData.windowLen
                  << "\ntilingData->qSeqLength:" << tilingData.qSeqLength
                  << "\ntilingData->kSeqLength:" << tilingData.kSeqLength
                  << "\ntilingData->vSeqLength:" << tilingData.vSeqLength << "\ntilingData->scale:" << tilingData.scale
                  << "\ntilingData->keepProb:" << tilingData.keepProb
                  << "\ntilingData->preTokens:" << tilingData.preTokens
                  << "\ntilingData->nextTokens:" << tilingData.nextTokens
                  << "\ntilingData->maskSeqLength:" << tilingData.maskSeqLength
                  << "\ntilingData->isTriangle:" << tilingData.isTriangle
                  << "\ntilingData->isHighPrecision:" << tilingData.isHighPrecision
                  << "\ntilingData->postBaseNum:" << tilingData.postBaseNum
                  << "\ntilingData->queryShape:" << tilingData.queryShape
                  << "\ntilingData->keyShape:" << tilingData.keyShape
                  << "\ntilingData->valueShape:" << tilingData.valueShape
                  << "\ntilingData->inputLayout:" << tilingData.inputLayout
                  << "\ntilingData->postDqFrontCoreNum:" << tilingData.postDqFrontCoreNum
                  << "\ntilingData->postDqTailCoreNum:" << tilingData.postDqTailCoreNum
                  << "\ntilingData->postDqFrontDataNum:" << tilingData.postDqFrontDataNum
                  << "\ntilingData->postDqTailDataNum:" << tilingData.postDqTailDataNum
                  << "\ntilingData->postDkFrontCoreNum:" << tilingData.postDkFrontCoreNum
                  << "\ntilingData->postDkTailCoreNum:" << tilingData.postDkTailCoreNum
                  << "\ntilingData->postDkFrontDataNum:" << tilingData.postDkFrontDataNum
                  << "\ntilingData->postDkTailDataNum:" << tilingData.postDkTailDataNum
                  << "\ntilingData->postDvFrontCoreNum:" << tilingData.postDvFrontCoreNum
                  << "\ntilingData->postDvTailCoreNum:" << tilingData.postDvTailCoreNum
                  << "\ntilingData->postDvFrontDataNum:" << tilingData.postDvFrontDataNum
                  << "\ntilingData->postDvTailDataNum:" << tilingData.postDvTailDataNum;
}

Status LaserAttentionGradTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    uint32_t coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    uint32_t aicNum = GetAicNum(coreNum);
    kernelInfo.SetBlockDim(aicNum);

    auto tilingDataPtr = reinterpret_cast<LaserAttentionGradTilingData *>(kernelInfo.GetTilingHostAddr());
    if (tilingDataPtr == nullptr) {
        return Status::FailStatus(ERROR_NULL_PTR, "tiling data is null");
    }

    LaserAttentionGradTilingData &tilingData = *tilingDataPtr;
    auto param = AnyCast<OpParam::LaserAttentionGrad>(launchParam.GetParam());
    const auto &queryShape = launchParam.GetInTensor(QUERY_IN_TENSOR_INDEX).desc;
    const auto &keyShape = launchParam.GetInTensor(KEY_IN_TENSOR_INDEX).desc;
    const auto &valueShape = launchParam.GetInTensor(VALUE_IN_TENSOR_INDEX).desc;

    Tensor maskTensor = launchParam.GetInTensor(ATTEN_MASK_IN_TENSOR_INDEX);
    if (!CheckEmptyTensor(maskTensor)) {
        const auto &maskShape = launchParam.GetInTensor(ATTEN_MASK_IN_TENSOR_INDEX).desc;
        tilingData.maskSeqLength = maskShape.dims[DIM_0];
    }

    // set scale value
    tilingData.scale = param.scaleValue;

    tilingData.headNum = param.headNum;

    int32_t seqSize = 0;

    if (param.inputLayout == "BNSD") {
        tilingData.inputLayout = static_cast<int>(InputLayout::BNSD);
        tilingData.batchSize = queryShape.dims[DIM_0];
        seqSize = queryShape.dims[DIM_2];
        tilingData.seqSize = seqSize;
    } else if (param.inputLayout == "SBH") {
        tilingData.inputLayout = static_cast<int>(InputLayout::SBH);
        tilingData.batchSize = queryShape.dims[DIM_1];
        seqSize = queryShape.dims[DIM_0];
        tilingData.seqSize = seqSize;
    }
    if (param.inputLayout == "BNSD") {
        tilingData.headDim = queryShape.dims[DIM_3];
    } else if (param.inputLayout == "SBH") {
        tilingData.headDim = HEAD_DIM_192;
    }

    tilingData.blockDim = static_cast<int32_t>(coreNum);

    int32_t blockNumPerCore = GetBlockNumPerCore(seqSize);
    tilingData.blockNumPerCore = blockNumPerCore;

    bool isTriangle = launchParam.GetInTensor(ATTEN_MASK_IN_TENSOR_INDEX).data != nullptr;
    tilingData.isTriangle = isTriangle;
    tilingData.attenType = GetAttenTypeGrad(launchParam, queryShape, keyShape);

    int32_t preTokens = param.preTokens;
    int32_t sparseMode = GetSparseModeGrad(isTriangle, preTokens, seqSize);
    tilingData.sparseMode = sparseMode;

    if (sparseMode == 1) {
        kernelInfo.SetTilingId(TILING_ID_BF16);
    } else {
        kernelInfo.SetTilingId(TILING_ID_BF16_BANDOP);
    }
    MKI_LOG(INFO) << "tilingKey is : " << kernelInfo.GetTilingId();

    tilingData.headGroupSize = GetHeadGroupSizeGrad(launchParam, queryShape, keyShape);
    int32_t windowLen = GetWindowLenGrad(isTriangle, preTokens, seqSize);
    tilingData.windowLen = windowLen;

    uint32_t keyHeadNum = 0;

    if (param.inputLayout == "BNSD") {
        tilingData.qSeqLength = seqSize;
        tilingData.kSeqLength = keyShape.dims[DIM_2];
        tilingData.vSeqLength = valueShape.dims[DIM_2];
        tilingData.isHighPrecision = param.innerPrecise;
        keyHeadNum = keyShape.dims[DIM_1];
    } else if (param.inputLayout == "SBH") {
        tilingData.qSeqLength = seqSize;
        tilingData.kSeqLength = keyShape.dims[DIM_0];
        tilingData.vSeqLength = valueShape.dims[DIM_0];
        tilingData.isHighPrecision = param.innerPrecise;
        keyHeadNum = keyShape.dims[DIM_2] / HEAD_DIM_256;
    }

    SetWorkspace(kernelInfo, aicNum, blockNumPerCore, tilingData, keyHeadNum, HEAD_DIM_256, HEAD_DIM_128);

    // tiling for post kernel
    SetPostTiling(tilingData);

    PrintTiling(tilingData);

    return Status::OkStatus();
}
} // namespace AtbOps
