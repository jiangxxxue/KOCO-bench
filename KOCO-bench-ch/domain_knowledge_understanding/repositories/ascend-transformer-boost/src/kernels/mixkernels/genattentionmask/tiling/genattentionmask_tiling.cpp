/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "genattentionmask_tiling.h"

#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "tiling_data.h"

namespace AtbOps {
Status GenAttentionMaskTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto tilingDataPtr = reinterpret_cast<GenAttentionMaskTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    const auto &inTensor0Dims = launchParam.GetInTensor(0).desc.dims;
    auto batch = inTensor0Dims[0];
    auto maxSeqLen = inTensor0Dims[2];
    MKI_CHECK(batch <= GEN_ATTENTION_MASK_MAX_BATCH, "batch exceeds the upper limit.",
        return Status::FailStatus(ERROR_INVALID_VALUE));

    tilingDataPtr->batch = batch;
    tilingDataPtr->maxSeqLen = maxSeqLen;
    tilingDataPtr->blockNum = batch;

    auto param = AnyCast<OpParam::GenAttentionMask>(launchParam.GetParam());
    tilingDataPtr->headNum = param.headNum;
    MKI_CHECK(param.qSeqLen.size() >= static_cast<size_t>(batch), "size of param.qSeqLen must be larger than batch.",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    for (int i = 0; i < batch; i++) {
        tilingDataPtr->qSeqLen[i] = param.qSeqLen.data()[i];
        MKI_LOG(INFO) << "qSeqLen " << i << ": " << tilingDataPtr->qSeqLen[i];
    }

    MKI_LOG(INFO) << "GenAttentionMask Tiling Data: batch " << tilingDataPtr->batch
                  << ", maxSeqLen " << tilingDataPtr->maxSeqLen
                  << ", blockNum " << tilingDataPtr->blockNum
                  << ", headNum " << tilingDataPtr->headNum
                  << ", qSeqLen " << tilingDataPtr->qSeqLen;

    kernelInfo.SetBlockDim(batch);
    return Status::OkStatus();
}
} // namespace AtbOps
