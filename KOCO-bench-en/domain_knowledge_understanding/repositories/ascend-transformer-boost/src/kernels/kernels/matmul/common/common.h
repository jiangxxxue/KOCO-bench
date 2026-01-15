/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMMON_H
#define COMMMON_H

#include <mki/launch_param.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "common_tiling.h"

namespace AsdOps {
// size: 输入参数数量
inline bool CheckAsdOpsND(const LaunchParam &launchParam, size_t size)
{
    MKI_CHECK(launchParam.GetInTensorCount() == size, "inTensor count invalid", return false);
    MKI_CHECK(launchParam.GetOutTensorCount() == 1, "outTensor count invalid", return false);
    MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MatMul), "check param type failed!", return false);

    const auto &inTensor0 = launchParam.GetInTensor(0);
    const auto &inTensor1 = launchParam.GetInTensor(1);
    const auto &outTensor = launchParam.GetOutTensor(0);
    MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
    MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16 || inTensor0.desc.dtype == TENSOR_DTYPE_BF16 ||
                  inTensor0.desc.dtype == TENSOR_DTYPE_INT8,
              "tensor dtype invalid", return false);
    MKI_CHECK(inTensor0.desc.dims.size() == 2 || inTensor0.desc.dims.size() == 3, "tensor dims invalid", return false);
    MKI_CHECK(inTensor1.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
    MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16 || inTensor1.desc.dtype == TENSOR_DTYPE_BF16 ||
                  inTensor1.desc.dtype == TENSOR_DTYPE_INT8,
              "tensor dtype invalid", return false);
    MKI_CHECK(inTensor1.desc.dims.size() == 2 || inTensor1.desc.dims.size() == 3, "tensor dims invalid", return false);
    MKI_CHECK(outTensor.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
    MKI_CHECK(outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16 || outTensor.desc.dtype == TENSOR_DTYPE_BF16 ||
                  outTensor.desc.dtype == TENSOR_DTYPE_INT8,
              "tensor dtype invalid", return false);
    MKI_CHECK(outTensor.desc.dims.size() == 2 || outTensor.desc.dims.size() == 3, "tensor dims invalid", return false);
    return true;
}

inline bool CheckAsdOpsWeightNZ(const LaunchParam &launchParam, size_t size)
{
    MKI_CHECK(launchParam.GetInTensorCount() == size, "inTensor count invalid", return false);
    MKI_CHECK(launchParam.GetOutTensorCount() == 1, "outTensor count invalid", return false);
    MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MatMul), "check param type failed!", return false);

    const auto &inTensor0 = launchParam.GetInTensor(0);
    const auto &inTensor1 = launchParam.GetInTensor(1);
    const auto &outTensor = launchParam.GetOutTensor(0);
    MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
    MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16 || inTensor0.desc.dtype == TENSOR_DTYPE_BF16 ||
                  inTensor0.desc.dtype == TENSOR_DTYPE_INT8,
              "tensor dtype invalid", return false);
    MKI_CHECK(inTensor0.desc.dims.size() == 2 || inTensor0.desc.dims.size() == 3, "tensor dims invalid", return false);
    MKI_CHECK(inTensor1.desc.format == TENSOR_FORMAT_FRACTAL_NZ, "tensor format invalid", return false);
    MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16 || inTensor1.desc.dtype == TENSOR_DTYPE_BF16 ||
                  inTensor1.desc.dtype == TENSOR_DTYPE_INT8,
              "tensor dtype invalid", return false);
    MKI_CHECK(inTensor1.desc.dims.size() == 4, "tensor dims invalid", return false);
    MKI_CHECK(outTensor.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
    MKI_CHECK(outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16 || outTensor.desc.dtype == TENSOR_DTYPE_BF16 ||
                  outTensor.desc.dtype == TENSOR_DTYPE_INT8,
              "tensor dtype invalid", return false);
    MKI_CHECK(outTensor.desc.dims.size() == 2 || outTensor.desc.dims.size() == 3, "tensor dims invalid", return false);
    return true;
}

template <typename T>
inline uint64_t GetMatmulTilingSizeCommon(const LaunchParam &launchParam)
{
    (void)launchParam;
    constexpr uint32_t CONST_256 = 256;
    return Round<CONST_256>(sizeof(T));
}
} // namespace AsdOps

#endif
