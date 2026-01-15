/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "matmul_nz_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
Status MatMulNzTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                      const BinHandle &binHandle)
{
    auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
    const TensorDesc &tensorDescA = launchParam.GetInTensor(0).desc;
    const TensorDesc &tensorDescB = launchParam.GetInTensor(1).desc;
    const TensorDesc &tensorDescOut = launchParam.GetOutTensor(0).desc;
    MKI_CHECK(opParam.oriShape.size() == 3, "size of oriShape is invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE)); // oriShape包含m,k,n这3个参数
    int64_t m = opParam.oriShape[0];
    int64_t k = opParam.oriShape[1];
    int64_t n = opParam.oriShape[2];
    SVector<int64_t> oriShapeA = {m, k};
    if (opParam.transposeA) {
        oriShapeA = {k, m};
    }
    SVector<int64_t> oriShapeB = {k, n};
    if (opParam.transposeB) {
        oriShapeB = {n, k};
    }
    SVector<int64_t> oriShapeOut = {m, n};

    auto runner = AsdOpsGeRt::TbeTilingRunner()
                      .SetName("MatMulV2")
                      .SetKernelName(kernelName)
                      .AddInput(tensorDescA.dtype, tensorDescA.format, oriShapeA)
                      .AddInput(tensorDescB.dtype, tensorDescB.format, oriShapeB)
                      .AddOutput(tensorDescOut.dtype, tensorDescOut.format, oriShapeOut)
                      .AddAttrBool(opParam.transposeA)
                      .AddAttrBool(opParam.transposeB)
                      .AddAttrInt64(0); // 0x40 is high precision
    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status BatchMatMulNzTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                           const BinHandle &binHandle)
{
    auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
    const TensorDesc &tensorDescA = launchParam.GetInTensor(0).desc;
    const TensorDesc &tensorDescB = launchParam.GetInTensor(1).desc;
    const TensorDesc &tensorDescOut = launchParam.GetOutTensor(0).desc;
    MKI_CHECK(opParam.oriShape.size() == 3, "size of oriShape is invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE)); // oriShape包含m,k,n这3个参数
    int64_t m = opParam.oriShape[0];
    int64_t k = opParam.oriShape[1];
    int64_t n = opParam.oriShape[2];
    SVector<int64_t> oriShapeA = {tensorDescA.dims[0], m, k};
    if (opParam.transposeA) {
        oriShapeA = {tensorDescA.dims[0], k, m};
    }
    SVector<int64_t> oriShapeB = {tensorDescB.dims[0], k, n};
    if (opParam.transposeB) {
        oriShapeB = {tensorDescB.dims[0], n, k};
    }
    SVector<int64_t> oriShapeOut = {m, n};

    auto runner = AsdOpsGeRt::TbeTilingRunner()
                      .SetName("BatchMatMulV2")
                      .SetKernelName(kernelName)
                      .AddInput(tensorDescA.dtype, tensorDescA.format, oriShapeA)
                      .AddInput(tensorDescB.dtype, tensorDescB.format, oriShapeB)
                      .AddOutput(tensorDescOut.dtype, tensorDescOut.format, oriShapeOut)
                      .AddAttrBool(opParam.transposeA)
                      .AddAttrBool(opParam.transposeB)
                      .AddAttrInt64(0); // 0x40 is high precision
    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
} // namespace AsdOps