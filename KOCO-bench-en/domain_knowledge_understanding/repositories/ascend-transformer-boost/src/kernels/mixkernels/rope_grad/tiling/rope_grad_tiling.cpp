/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/assert/assert.h>
#include "atbops/params/params.h"
#include "tiling_data.h"
#include "rope_grad_tiling.h"

static constexpr uint32_t MAX_PROCESS_NUM = 192 * 1024 / 16;
namespace AtbOps {
Status RopeGradTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    MKI_LOG(INFO) << " RopeGradTiling START ";
    uint8_t *tilingHost = kernelInfo.GetTilingHostAddr();
    RopeGradTilingData *tilingDataPtr = reinterpret_cast<RopeGradTilingData *>(tilingHost);
    if (tilingDataPtr == nullptr) {
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr should not be empty");
    }
    int32_t coreNum = static_cast<int32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    const auto &inTensor0Dims = launchParam.GetInTensor(0).desc.dims;
    const auto &inTensor2Dims = launchParam.GetInTensor(2).desc.dims;
    auto sumSeqLen = inTensor0Dims[0];
    auto hiddenSize = inTensor0Dims[1];
    auto maxSeqLen = inTensor2Dims[0];
    auto headSize = inTensor2Dims[1];
    tilingDataPtr->maxSeqLen = maxSeqLen;
    tilingDataPtr->hiddenSize = hiddenSize;
    tilingDataPtr->headSize = headSize;
    MKI_CHECK(tilingDataPtr->headSize > 0 && tilingDataPtr->headSize < MAX_PROCESS_NUM, "headSize invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(hiddenSize % headSize == 0, "hiddenSize should be an integer multiple of headSize",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->sumSeqLen = sumSeqLen;
    auto param = AnyCast<OpParam::RopeGrad>(launchParam.GetParam());
    tilingDataPtr->batch = static_cast<int64_t>(param.qSeqLen.size());
    MKI_CHECK(tilingDataPtr->batch > 0, "batch is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    for (int i = 0; i < tilingDataPtr->batch; i++) {
        auto sampleTilingDataPtr = reinterpret_cast<RopeGradSampleTilingData *>(
            tilingHost + sizeof(RopeGradTilingData) + i * sizeof(RopeGradSampleTilingData));
        MKI_CHECK(sampleTilingDataPtr != nullptr, "sampleTilingDataPtr is nullptr",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        sampleTilingDataPtr->qSeqLen = param.qSeqLen[i];
        MKI_CHECK(sampleTilingDataPtr->qSeqLen > 0 && sampleTilingDataPtr->qSeqLen <= maxSeqLen, "qSeqLen is invalid",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_LOG(INFO) << ", qSeqLen" << i << ": " << sampleTilingDataPtr->qSeqLen;
    }
    MKI_LOG(INFO) << "RopeGrad tilingHost Data: maxSeqLen " << tilingDataPtr->maxSeqLen << ", hiddenSize "
                  << tilingDataPtr->hiddenSize << ", headSize " << tilingDataPtr->headSize << ", sumSeqLen "
                  << tilingDataPtr->sumSeqLen << ", batch " << tilingDataPtr->batch;
    int32_t blockNum = hiddenSize / headSize;
    blockNum = blockNum < coreNum ? blockNum : coreNum;
    kernelInfo.SetBlockDim(blockNum);
    return Status::OkStatus();
}
} // namespace AtbOps
