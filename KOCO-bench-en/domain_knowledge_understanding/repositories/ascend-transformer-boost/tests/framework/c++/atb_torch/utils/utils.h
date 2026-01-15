/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef EXAMPLE_UTIL_H
#define EXAMPLE_UTIL_H
#include <vector>
#include <string>
#include <atb/types.h>
#include <torch/torch.h>
#include "atb/runner/runner.h"
#include "atb/operation.h"

class Utils {
public:
    static void *GetCurrentStream();
    static int64_t GetTensorNpuFormat(const at::Tensor &tensor);
    static at::Tensor NpuFormatCast(const at::Tensor &tensor);
    static void BuildVariantPack(const std::vector<torch::Tensor> &inTensors,
                                 const std::vector<torch::Tensor> &outTensors, atb::VariantPack &variantPack);
    static atb::Tensor AtTensor2Tensor(const at::Tensor &atTensor);
    static at::Tensor CreateAtTensorFromTensorDesc(const atb::TensorDesc &tensorDesc);
    static void ContiguousAtTensor(std::vector<torch::Tensor> &atTensors);
    static void ContiguousAtTensor(torch::Tensor &atTensor);
};

#endif