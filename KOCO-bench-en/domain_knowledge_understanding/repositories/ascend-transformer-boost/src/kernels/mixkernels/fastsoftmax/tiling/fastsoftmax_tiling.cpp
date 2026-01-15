/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "fastsoftmax_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "tiling/softmax/softmax_tiling.h"
#include "tiling_data.h"

namespace AtbOps {

inline uint32_t AlignUpToBasicBlock(uint32_t size)
{
    return (size + 0xF) & ~0xF;
}

inline void SaveSoftMaxTilingToBuffer(uint32_t outer, uint32_t inner, uint8_t buffer[])
{
    if (outer > 0) {
        ge::Shape shape({ outer, inner });
        uint32_t workLocalSize = AscendC::GetSoftMaxMaxTmpSize(shape, DATA_BYTESIZE, false);
        if (workLocalSize > SHARED_TMP_SIZE) {
            workLocalSize = SHARED_TMP_SIZE;
        }
        optiling::SoftMaxTiling tiling;
        AscendC::SoftMaxTilingFunc(shape, DATA_BYTESIZE, workLocalSize, tiling);
        tiling.SaveToBuffer(buffer, tiling.GetDataSize());
    }
}

inline void SetSampleTilingData(FastSoftMaxSampleTilingData *sampleTilingDataPointer, const uint32_t ubSize,
    const uint32_t coreNum, uint32_t headNum, uint32_t sampleSeqLenOrigin)
{
    uint32_t sampleSeqLen = AlignUpToBasicBlock(sampleSeqLenOrigin);
    uint32_t outerSize = headNum * sampleSeqLenOrigin;
    uint32_t innerSize = sampleSeqLen * DATA_BYTESIZE;
    uint32_t maxTileRowNum = (ubSize - SHARED_TMP_SIZE) /
        (BUFFER_NUM * SOFTMAX_COMPUTE_DIM * innerSize + 2 * BASICBLOCK_SIZE);
    uint32_t maxCoreTileNum = Utils::CeilDiv(outerSize, maxTileRowNum * coreNum);
    uint32_t tileRowNum = Utils::CeilDiv(outerSize, maxCoreTileNum * coreNum);
    uint32_t tailTileRowNum = outerSize % tileRowNum;
    uint32_t formerCoreTileNum = maxCoreTileNum;
    uint32_t latterCoreTileNum = maxCoreTileNum - 1;
    uint32_t formerCoreNum = outerSize / tileRowNum - coreNum * latterCoreTileNum;
    uint32_t latterCoreNum = coreNum - formerCoreNum;

    sampleTilingDataPointer->sampleSeqLenOrigin = sampleSeqLenOrigin;
    sampleTilingDataPointer->sampleSeqLen = sampleSeqLen;
    sampleTilingDataPointer->outerSize = outerSize;
    sampleTilingDataPointer->innerSize = innerSize;
    sampleTilingDataPointer->tileRowNum = tileRowNum;
    sampleTilingDataPointer->tailTileRowNum = tailTileRowNum;
    sampleTilingDataPointer->formerCoreNum = formerCoreNum;
    sampleTilingDataPointer->latterCoreNum = latterCoreNum;
    sampleTilingDataPointer->formerCoreTileNum = formerCoreTileNum;
    sampleTilingDataPointer->latterCoreTileNum = latterCoreTileNum;
    SaveSoftMaxTilingToBuffer(tileRowNum, sampleSeqLen, sampleTilingDataPointer->softMaxTilingBuffer);
    SaveSoftMaxTilingToBuffer(tailTileRowNum, sampleSeqLen, sampleTilingDataPointer->tailSoftMaxTilingBuffer);
}

Status FastSoftMaxTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    uint32_t coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    uint32_t ubSize = PlatformInfo::Instance().GetUbSize();
    auto param = AnyCast<OpParam::FastSoftMax>(launchParam.GetParam());
    uint32_t batchSize = param.qSeqLen.size();
    uint32_t headNum = static_cast<uint32_t>(param.headNum);
    MKI_CHECK(headNum > 0, "head Num is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));

    uint8_t *tiling = kernelInfo.GetTilingHostAddr();
    auto tilingDataPointer = reinterpret_cast<FastSoftMaxTilingData *>(tiling);
    tiling += sizeof(FastSoftMaxTilingData);
    MKI_CHECK(tilingDataPointer != nullptr, "tilingData should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPointer->batchSize = batchSize;
    tilingDataPointer->headNum = headNum;

    uint32_t dataOffset = 0;
    for (uint32_t sampleIndex = 0; sampleIndex < batchSize; ++sampleIndex) {
        uint32_t sampleSeqLenOrigin = static_cast<uint32_t>(param.qSeqLen[sampleIndex]);
        MKI_CHECK(sampleSeqLenOrigin > 0 && sampleSeqLenOrigin <= MAX_SEQ_LEN, "seqlen is invalid",
            return Status::FailStatus(ERROR_INVALID_VALUE));
        auto sampleTilingDataPointer = reinterpret_cast<FastSoftMaxSampleTilingData *>(tiling);
        tiling += sizeof(FastSoftMaxSampleTilingData);
        SetSampleTilingData(sampleTilingDataPointer, ubSize, coreNum, headNum, sampleSeqLenOrigin);
        uint32_t dataLength = headNum * sampleSeqLenOrigin * sampleSeqLenOrigin;
        sampleTilingDataPointer->dataLength = dataLength;
        sampleTilingDataPointer->dataOffset = dataOffset;
        dataOffset += dataLength;
    }

    kernelInfo.SetBlockDim(coreNum);
    return Status::OkStatus();
}

}  // namespace AtbOps