/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "unpad_with_hidden_state_tiling.h"

#include <securec.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "tiling_data.h"

namespace AtbOps {

inline uint32_t AlignDownToBasicBlock(uint32_t size)
{
    return size & ~(BASIC_BLOCK_SIZE - 1);  // align to power of 2
}

inline ErrorType GetTileInfo(TileInfo &tileInfo, uint32_t dataLength, uint32_t tileLength, uint32_t coreNum)
{
    MKI_CHECK(tileLength > 0 && coreNum > 0,
        "create tile info failed", return ERROR_INVALID_VALUE);
    uint32_t tileNum = dataLength / tileLength;
    tileInfo.formerCoreNum = tileNum % coreNum;
    tileInfo.formerCoreTileNum = tileNum / coreNum + 1;
    tileInfo.lastTileLength = dataLength % tileLength;
    return NO_ERROR;
}

Status UnpadWithHiddenStateTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    uint32_t coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::UnpadWithHiddenState), "param is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    auto param = AnyCast<OpParam::UnpadWithHiddenState>(launchParam.GetParam());
    uint32_t batchSize = param.qSeqLen.size();
    uint32_t maxSeqLen = param.maxSeqLen;
    uint32_t hiddenSize = launchParam.GetInTensor(DIM_0).desc.dims[DIM_2];
    MKI_CHECK(batchSize > 0 && batchSize <= MAX_BATCH_SIZE, "batch size is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(maxSeqLen > 0 && maxSeqLen <= SEQLEN_LIMIT, "max sequence length is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    uint32_t bufferSize = AlignDownToBasicBlock(UB_SIZE / BUFFER_NUM / UNPAD_WITH_HIDDEN_STATE_COMPUTE_DIM);
    uint32_t tileLength = bufferSize / ELEMENT_SIZE;

    void *tilingdata = kernelInfo.GetTilingHostAddr();
    auto ret = memset_s(tilingdata, sizeof(UnpadWithHiddenStateTilingData), 0,
        sizeof(UnpadWithHiddenStateTilingData));
    MKI_CHECK(ret == EOK, "memory set failed", return Status::FailStatus(ERROR_INVALID_VALUE));
    auto tilingDataPointer = reinterpret_cast<UnpadWithHiddenStateTilingData *>(tilingdata);
    tilingDataPointer->coreNum = coreNum;
    tilingDataPointer->batchSize = batchSize;
    tilingDataPointer->maxSeqLen = maxSeqLen;
    tilingDataPointer->bufferSize = bufferSize;
    tilingDataPointer->tileLength = tileLength;

    uint32_t inputOffset = 0;
    uint32_t outputOffset = 0;
    for (uint32_t sampleIndex = 0; sampleIndex < batchSize; ++sampleIndex) {
        uint32_t sampleSeqLen = static_cast<uint32_t>(param.qSeqLen[sampleIndex]);
        MKI_CHECK(sampleSeqLen > 0 && sampleSeqLen <= maxSeqLen, "sequence length is invalid",
            return Status::FailStatus(ERROR_INVALID_VALUE));
        uint32_t sampleDataLength = sampleSeqLen * hiddenSize;

        tilingDataPointer->inputOffset[sampleIndex] = inputOffset;
        tilingDataPointer->outputOffset[sampleIndex] = outputOffset;
        auto infoRet = GetTileInfo(tilingDataPointer->tileInfo[sampleIndex], sampleDataLength, tileLength, coreNum);
        MKI_CHECK(infoRet == NO_ERROR, "get tile info failed", return Status::FailStatus(ERROR_INVALID_VALUE));

        inputOffset += maxSeqLen * hiddenSize;
        outputOffset += sampleDataLength;
    }

    tilingDataPointer->totalDataLength = inputOffset;
    tilingDataPointer->unpadDataLength = outputOffset;

    kernelInfo.SetBlockDim(coreNum);
    return Status::OkStatus();
}

}  // namespace AtbOps
