/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rope_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "tiling_data.h"

namespace AtbOps {
static constexpr uint32_t BLOCK_SIZE = 16;
static constexpr uint32_t BLOCK_BYTE = 32;
static constexpr uint32_t REMAIN_SPACE = 16 * 1024 * 1024;
static constexpr uint32_t NUM_COSIN = 2;
static constexpr uint32_t NUM_EIGHT = 8;
static constexpr uint32_t TOTAL_OFFSET = 6;
static constexpr uint32_t ELE_NUM_FP16 = 16; // 一个block fp16元素个数
static constexpr uint32_t LARGE_NTOKENS_THRESHOLD = 64;
static constexpr uint32_t SLICE_SIZE_FP16_LARGE_NTOKENS = 4096;
static constexpr uint32_t TILING_BF16_ALIGN_BROARD = 33;
static constexpr uint32_t TILING_BF16_BROARD = 32;
static constexpr uint32_t TILING_HIGH_PREC_BOARD = 31;
static constexpr uint32_t TILING_HIGH_PERF_BROARD = 30;
static constexpr uint32_t TILING_BF16_ALIGN = 24;
static constexpr uint32_t TILING_BF16 = 22;
static constexpr uint32_t TILING_HIGH_PREC = 21;
static constexpr uint32_t TILING_HIGH_PERF = 20;
static constexpr uint32_t TILING_HIGH_PERF_LARGE_NTOKENS = 23;

void RopeNdProcess(const LaunchParam &launchParam, KernelInfo &kernelInfo, RopeTilingData &tilingDataPtr)
{
    uint32_t hiddenSizeQ = static_cast<uint32_t>(launchParam.GetInTensor(0).desc.dims[1]);
    uint32_t hiddenSizeK = static_cast<uint32_t>(launchParam.GetInTensor(1).desc.dims[1]);
    auto cosSize = launchParam.GetInTensor(2).desc.dims.size();
    uint32_t headDim = static_cast<uint32_t>(launchParam.GetInTensor(2).desc.dims[cosSize - 1]);
    uint32_t ntokens = static_cast<uint32_t>(launchParam.GetInTensor(0).desc.dims[0]);
    uint32_t batch = static_cast<uint32_t>(launchParam.GetInTensor(4).desc.dims[0]);
    uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    auto maxUbSize = static_cast<uint32_t>(PlatformInfo::Instance().GetUbSize());
    tilingDataPtr.maxUbSize = maxUbSize;

    uint32_t multiple = 1;
    bool condition = tilingDataPtr.cosFormat == 0 && cosSize == NUM_COSIN &&
                     launchParam.GetInTensor(NUM_COSIN).desc.dtype == TENSOR_DTYPE_FLOAT16 &&
                     ntokens >= LARGE_NTOKENS_THRESHOLD && headDim / tilingDataPtr.rotaryCoeff % ELE_NUM_FP16 == 0;
    if (condition) { // 不对齐场景, multiple为1
        uint32_t hiddenSize = hiddenSizeK > hiddenSizeQ ? hiddenSizeK : hiddenSizeQ;
        multiple = SLICE_SIZE_FP16_LARGE_NTOKENS / hiddenSize;
        multiple = multiple > 0 ? multiple : 1;
        while (ntokens % multiple != 0 && multiple > 1) {
            --multiple;
        }
        if (ntokens / multiple < maxCore || UINT32_MAX / multiple < hiddenSize) {
            multiple = 1;
        } else {
            ntokens /= multiple;
            batch /= multiple;
            hiddenSizeQ *= multiple;
            hiddenSizeK *= multiple;
            headDim *= multiple;
        }
    }
    uint32_t tempCore = (ntokens + maxCore - 1) / maxCore;
    uint32_t realCore = (ntokens + tempCore - 1) / tempCore;
    tilingDataPtr.realCore = realCore;
    tilingDataPtr.hiddenSizeQ = hiddenSizeQ;
    tilingDataPtr.hiddenSizeK = hiddenSizeK;
    tilingDataPtr.headDim = headDim;
    tilingDataPtr.ntokens = ntokens;
    tilingDataPtr.batch = batch;
    tilingDataPtr.multiple = multiple;
    kernelInfo.SetBlockDim(realCore);
    MKI_LOG(DEBUG) << "Ntokens is " << ntokens;
    MKI_LOG(DEBUG) << "Batch is " << batch;
    MKI_LOG(DEBUG) << "HiddenSizeQ is " << hiddenSizeQ;
    MKI_LOG(DEBUG) << "HiddenSizeK is " << hiddenSizeK;
    MKI_LOG(DEBUG) << "HeadDim is " << headDim;
    MKI_LOG(DEBUG) << "Multiple is " << multiple;
    MKI_LOG(DEBUG) << "RealCore is " << realCore;
}
Status TilingKeyChose(const LaunchParam &launchParam, KernelInfo &kernelInfo, const RopeTilingData &tilingDataPtr)
{
    auto platformType = PlatformInfo::Instance().GetPlatformType();
    auto cosSize = launchParam.GetInTensor(NUM_COSIN).desc.dims.size();
    if (cosSize == NUM_COSIN) {
        if (launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16) {
            if (platformType != PlatformType::ASCEND_910B) {
                MKI_LOG(ERROR) << "BF16 only supports 800I A2";
                return Status::FailStatus(ERROR_INVALID_VALUE);
            }
            uint32_t alignRotary = (tilingDataPtr.headDim / tilingDataPtr.rotaryCoeff) % ELE_NUM_FP16;
            bool condition = (alignRotary == 0) && (tilingDataPtr.ntokens >= LARGE_NTOKENS_THRESHOLD);
            if (condition) { // ntokens >= 64时，走TILING_BF16_ALIGN
                kernelInfo.SetTilingId(TILING_BF16_ALIGN); // first 2 for shape dims of cos
            } else {
                kernelInfo.SetTilingId(TILING_BF16);
            }
        } else if (launchParam.GetInTensor(NUM_COSIN).desc.dtype == TENSOR_DTYPE_FLOAT) {
            kernelInfo.SetTilingId(TILING_HIGH_PREC); // second 1 for FP32
        } else {
            bool condition = tilingDataPtr.ntokens * tilingDataPtr.multiple >= LARGE_NTOKENS_THRESHOLD &&
                             tilingDataPtr.cosFormat == 0;
            if (condition) { // ntokens >= 64时，走TILING_HIGH_PERF_LARGE_NTOKENS
                kernelInfo.SetTilingId(TILING_HIGH_PERF_LARGE_NTOKENS);
            } else {
                kernelInfo.SetTilingId(TILING_HIGH_PERF); // second 0 for FP16
            }
        }
    } else {
        if (launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16) {
            if (platformType != PlatformType::ASCEND_910B) {
                MKI_LOG(ERROR) << "BF16 only supports 800I A2";
                return Status::FailStatus(ERROR_INVALID_VALUE);
            }
            uint32_t alignRotary = (tilingDataPtr.headDim / tilingDataPtr.rotaryCoeff) % ELE_NUM_FP16;
            bool condition = (alignRotary == 0) && (tilingDataPtr.ntokens >= LARGE_NTOKENS_THRESHOLD);
            if (condition) { // ntokens >= 64时，走TILING_BF16_ALIGN_BROARD
                kernelInfo.SetTilingId(TILING_BF16_ALIGN_BROARD); // first 2 for shape dims of cos
            } else {
                kernelInfo.SetTilingId(TILING_BF16_BROARD); // first 3 for shape dims of cos
            }
        } else if (launchParam.GetInTensor(NUM_COSIN).desc.dtype == TENSOR_DTYPE_FLOAT) {
            kernelInfo.SetTilingId(TILING_HIGH_PREC_BOARD); // second 1 for FP32
        } else {
            kernelInfo.SetTilingId(TILING_HIGH_PERF_BROARD); // second 0 for fp16
        }
    }
    return Status::OkStatus();
}
Status RopeTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    RopeTilingData *tilingDataPtr = reinterpret_cast<RopeTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    auto attrs = AnyCast<OpParam::Rope>(launchParam.GetParam());
    MKI_CHECK(attrs.rotaryCoeff > 0, "attrs.rotaryCoeff is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->rotaryCoeff = static_cast<uint32_t>(attrs.rotaryCoeff);
    MKI_CHECK(attrs.cosFormat == 0 || attrs.cosFormat == 1, "wrong cosFormat, cosFormat should be 0 or 1",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->cosFormat = static_cast<uint32_t>(attrs.cosFormat);
    MKI_LOG(DEBUG) << "cosFormat is " << tilingDataPtr->cosFormat;
    uint32_t headNumQ = 1;
    uint32_t headNumK = 1;
    RopeNdProcess(launchParam, kernelInfo, *tilingDataPtr);
    if (tilingDataPtr->headDim != 0) {
        headNumQ = tilingDataPtr->hiddenSizeQ / tilingDataPtr->headDim;
        headNumK = tilingDataPtr->hiddenSizeK / tilingDataPtr->headDim;
    } else {
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr->headDim is wrong");
    }
    tilingDataPtr->headNumQ = headNumQ;
    tilingDataPtr->headNumK = headNumK;
    auto ret = TilingKeyChose(launchParam, kernelInfo, *tilingDataPtr);
    if (!ret.Ok()) {
        return Status::FailStatus(ERROR_INVALID_VALUE);
    }
    uint64_t sysWorkspaceSize =
        static_cast<uint64_t>(REMAIN_SPACE) +
        static_cast<uint64_t>(TOTAL_OFFSET) * tilingDataPtr->realCore * tilingDataPtr->hiddenSizeQ * sizeof(uint16_t) +
        static_cast<uint64_t>(tilingDataPtr->ntokens) * tilingDataPtr->headDim * NUM_COSIN * sizeof(uint16_t);
    uint64_t syncWorkspaceSize = tilingDataPtr->realCore * BLOCK_BYTE;
    kernelInfo.GetScratchSizes() = {sysWorkspaceSize, syncWorkspaceSize};
    if ((PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_310P ||
         PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B) &&
         attrs.cosFormat == 1) {
        kernelInfo.SetMemsetInfo(NUM_EIGHT, syncWorkspaceSize);
    }
    MKI_LOG(INFO) << "workspace: " << sysWorkspaceSize;
    return Status::OkStatus();
}
} // namespace AtbOps