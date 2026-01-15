/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TORCH_ATB_UTILS_H
#define TORCH_ATB_UTILS_H
#include "atb/atb_infer.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#include <torch/torch.h>
#pragma GCC diagnostic pop

namespace TorchAtb {
namespace Utils {
atb::Context *GetAtbContext();
atb::Tensor ConvertToAtbTensor(torch::Tensor &torchTensor);
torch::Tensor CreateTorchTensorFromTensorDesc(const atb::TensorDesc &tensorDesc);
bool IsTaskQueueEnable();
aclrtStream GetCurrentStream();
} // namespace Utils

} // namespace TorchAtb
#endif // TORCH_ATB_UTILS_H