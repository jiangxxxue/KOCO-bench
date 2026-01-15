/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "matmul_nd_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/params.h"
#include "tbe_tiling_runner.h"
#include "kernels/matmul/tiling/tiling_data.h"

namespace AsdOps {
constexpr int32_t MAT_N_STEP = 256;
static inline uint32_t Min(const uint32_t x, const uint32_t y)
{
    return x < y ? x : y;
}

Status MatMulNdTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                      const BinHandle &binHandle)
{
    auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
    const TensorDesc &tensorDescA = launchParam.GetInTensor(0).desc;
    const TensorDesc &tensorDescB = launchParam.GetInTensor(1).desc;
    const TensorDesc &tensorDescOut = launchParam.GetOutTensor(0).desc;

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("MatMulV2")
        .SetKernelName(kernelName)
        .AddInput(tensorDescA.dtype, tensorDescA.format, tensorDescA.dims)
        .AddInput(tensorDescB.dtype, tensorDescB.format, tensorDescB.dims)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims)
        .AddAttrBool(opParam.transposeA)
        .AddAttrBool(opParam.transposeB)
        .AddAttrInt64(0); // 0x40 is high precision
    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status BatchMatMulNdTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                           const BinHandle &binHandle)
{
    auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
    const TensorDesc &tensorDescA = launchParam.GetInTensor(0).desc;
    const TensorDesc &tensorDescB = launchParam.GetInTensor(1).desc;
    const TensorDesc &tensorDescOut = launchParam.GetOutTensor(0).desc;

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("BatchMatMulV2")
        .SetKernelName(kernelName)
        .AddInput(tensorDescA.dtype, tensorDescA.format, tensorDescA.dims)
        .AddInput(tensorDescB.dtype, tensorDescB.format, tensorDescB.dims)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims)
        .AddAttrBool(opParam.transposeA)
        .AddAttrBool(opParam.transposeB)
        .AddAttrInt64(0); // 0x40 is high precision
    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status MatMulNdGemvTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto inTensorsBDim = launchParam.GetInTensor(1).desc.dims;
    auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
    GemvMatmulTilingData *tilingDataPtr =
        reinterpret_cast<AsdOps::GemvMatmulTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE));

    uint32_t cubeCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    tilingDataPtr->n = static_cast<uint32_t>(inTensorsBDim.at(0));
    tilingDataPtr->k = static_cast<uint32_t>(inTensorsBDim.at(1));
    tilingDataPtr->transposeB = static_cast<uint32_t>(opParam.transposeB);
    uint32_t needCoreNum = Min(Utils::CeilDiv(static_cast<int32_t>(tilingDataPtr->n), MAT_N_STEP), cubeCoreNum);
    MKI_LOG(INFO) << "MatMulGemv tiling k, n transposeB = "
                  << " " << tilingDataPtr->k << " " << tilingDataPtr->n << " " << tilingDataPtr->transposeB << " "
                  << opParam.transposeB;
    MKI_LOG(INFO) << "MatMulGemv num_core: " << needCoreNum;
    kernelInfo.SetBlockDim(needCoreNum);
    return Status::OkStatus();
}
} // namespace AsdOps