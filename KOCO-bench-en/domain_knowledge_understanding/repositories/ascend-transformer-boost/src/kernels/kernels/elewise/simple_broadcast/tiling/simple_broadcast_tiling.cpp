/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "simple_broadcast_tiling.h"
#include <cstring>
#include <mki/utils/env/env.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>

namespace AsdOps {
Status SimplyBroadcastTiling(BroadcastInfo &broadcastInfo, const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto tilingData = reinterpret_cast<AsdOps::BroadcastTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingData != nullptr, "tilingData should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    // 计算dimN、dimB元素的个数
    // scalar
    bool isScalar = false;
    if (launchParam.GetInTensor(TENSOR_SRC_0_IDX).Numel() == 1) {
        tilingData->dimN = launchParam.GetInTensor(TENSOR_DST_IDX).Numel();
        tilingData->dimB = 1;
        isScalar = true;
    } else {
        tilingData->dimN = launchParam.GetInTensor(TENSOR_SRC_0_IDX).Numel();
        MKI_CHECK(tilingData->dimN > 0, "dimN is 0 or negative", return Status::FailStatus(ERROR_INVALID_VALUE));
        tilingData->dimB = launchParam.GetInTensor(TENSOR_DST_IDX).Numel() / tilingData->dimN;
    }
    MKI_CHECK((tilingData->dimN > 0 && tilingData->dimB > 0),
        "dimN or dimB is 0 or negative", return Status::FailStatus(ERROR_INVALID_VALUE));
    // 仅910B支持非32字节对齐
    if (AsdOps::PlatformInfo::Instance().GetPlatformType() != AsdOps::PlatformType::ASCEND_910B) {
        MKI_CHECK(tilingData->dimN % (32 / sizeof(int8_t)) == 0,
            "dimN is not 32 bytes align", return Status::FailStatus(ERROR_INVALID_VALUE)); // offset是int8_t类型
    }
    // 计算dimN、dimB分核个数，每个核至少处理40960字节数据
    TensorDType inTensorDtype = launchParam.GetInTensor(TENSOR_DST_IDX).desc.dtype;
    int64_t inTensorBytes =  static_cast<int64_t>(GetTensorElementSize(inTensorDtype));
    MKI_CHECK((inTensorBytes != 0),
        "inTensorDtype is invalid!", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK((tilingData->dimN < INT64_MAX / inTensorBytes),
        "dimN is invalid!", return Status::FailStatus(ERROR_INVALID_VALUE));
    int64_t dimNBytes = tilingData->dimN * inTensorBytes;
    int64_t coreNum = static_cast<int64_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    int64_t dimNBlockNum = std::min(Utils::CeilDiv(dimNBytes, 40960l), coreNum);

    // 计算dimN、dimB每个核元素个数，256字节对齐
    int64_t oneRepeatElemCount = 256 / inTensorBytes;
    int64_t dimNBlockSize = std::min(static_cast<int64_t>(Utils::RoundUp(
        Utils::CeilDiv(tilingData->dimN, dimNBlockNum), oneRepeatElemCount)), tilingData->dimN);
    tilingData->dimNBlockSize = dimNBlockSize;
    tilingData->dimNBlockNum = Utils::CeilDiv(tilingData->dimN, tilingData->dimNBlockSize);
    MKI_CHECK((tilingData->dimNBlockNum > 0),
        "dimNBlockNum is 0 or negative", return Status::FailStatus(ERROR_INVALID_VALUE));

    int64_t dimBBlockNum = std::min(coreNum / tilingData->dimNBlockNum, tilingData->dimB);
    tilingData->dimBBlockSize = Utils::CeilDiv(tilingData->dimB, dimBBlockNum);
    tilingData->dimBBlockNum = Utils::CeilDiv(tilingData->dimB, tilingData->dimBBlockSize);

    int64_t oneElemNeedUbSize = static_cast<int64_t>(broadcastInfo.broadcastBytesPerElem) +
        static_cast<int64_t>(broadcastInfo.normalBytesPerElem);
    int64_t maxUbBytes = static_cast<int64_t>(PlatformInfo::Instance().GetUbSize() * 0.8); // 预留20%
    tilingData->dimNLoop = std::min(static_cast<int64_t>(Utils::RoundDown(
        maxUbBytes / oneElemNeedUbSize, oneRepeatElemCount)), dimNBlockSize);
    bool cutN = false;
    if (tilingData->dimNLoop == tilingData->dimN) {
        // 如果能搬一整行，那么进一步计算可以进一步搬多少行
        tilingData->dimBLoop = std::min((maxUbBytes - broadcastInfo.normalBytesPerElem * tilingData->dimNLoop) /
                                        (tilingData->dimNLoop * broadcastInfo.broadcastBytesPerElem),
                                        static_cast<size_t>(tilingData->dimBBlockSize));
    } else {
        tilingData->dimBLoop = 1;
        cutN = true;
    }

    uint64_t tilingKey = TILING_ID_BASE;
    if (!isScalar) {
        if (cutN) {
            tilingKey += TILING_ID_CUT * 1; // 1: loop cut N
        } else {
            tilingKey += TILING_ID_CUT * 2; // 2: loop cut B
        }
    }
    kernelInfo.SetTilingId(tilingKey);
    kernelInfo.SetBlockDim(tilingData->dimNBlockNum * tilingData->dimBBlockNum);
    MKI_LOG(INFO) << "Broadcast Tiling Data: dimN: " << tilingData->dimN
                  << ", dimB: " << tilingData->dimB
                  << ", dimNBlockNum: " << tilingData->dimNBlockNum
                  << ", dimBBlockNum: " << tilingData->dimBBlockNum
                  << ", dimNBlockSize: " << tilingData->dimNBlockSize
                  << ", dimBBlockSize: " << tilingData->dimBBlockSize
                  << ", dimNLoop: " << tilingData->dimNLoop
                  << ", dimBLoop: " << tilingData->dimBLoop;

    return Status::OkStatus();
}

Status QuantPerChannelTiling(BroadcastInfo &broadcastInfo, const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    Status status = SimplyBroadcastTiling(broadcastInfo, launchParam, kernelInfo);
    if (!status.Ok()) {
        return status;
    }

    bool isQuantMinSetEnable = false;
    const char *quantMinSetPtr = Mki::GetEnv("ASDOPS_QUANT_MIN_NEG_127");
    if (quantMinSetPtr != nullptr && strcmp(quantMinSetPtr, "1") == 0) {
        isQuantMinSetEnable = true;
    }
    bool hasOffset = !CheckEmptyTensor(launchParam.GetInTensor(TENSOR_OFFSET_IDX));
    bool isBf16 = launchParam.GetInTensor(TENSOR_DST_IDX).desc.dtype == TENSOR_DTYPE_BF16;

    uint64_t tilingKey = kernelInfo.GetTilingId();
    tilingKey += isQuantMinSetEnable ? TILING_ID_MIN_NEG_127 : 0;
    tilingKey += hasOffset ? TILING_ID_HAS_OFFSET : 0;
    tilingKey += isBf16 ? TILING_ID_DTYPE : 0;
    kernelInfo.SetTilingId(tilingKey);
    MKI_LOG(INFO) << "Quant Per Channel tilingKey is : " << tilingKey;

    return Status::OkStatus();
}

Status DequantPerChannelTiling(BroadcastInfo &broadcastInfo, const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    Status status = SimplyBroadcastTiling(broadcastInfo, launchParam, kernelInfo);
    if (!status.Ok()) {
        return status;
    }
    bool hasOffset = !CheckEmptyTensor(launchParam.GetInTensor(TENSOR_OFFSET_IDX));

    uint64_t tilingKey = kernelInfo.GetTilingId();
    tilingKey = hasOffset ? tilingKey + TILING_ID_HAS_OFFSET : tilingKey;
    kernelInfo.SetTilingId(tilingKey);
    MKI_LOG(INFO) << "Dequant Per Channel tilingKey is : " << tilingKey;

    return Status::OkStatus();
}
} // namespace AsdOps