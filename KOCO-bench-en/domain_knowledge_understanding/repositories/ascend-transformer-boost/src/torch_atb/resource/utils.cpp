/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils.h"
#include <thread>
#include <cstdlib>
#include <stdexcept>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#pragma GCC diagnostic pop
#include "atb/utils/tensor_util.h"

namespace TorchAtb {
constexpr int32_t DECIMAL = 10;
namespace Utils {
aclrtStream GetCurrentStream()
{
    int32_t devId = 0;
    aclrtGetDevice(&devId);
    aclrtStream stream = c10_npu::getCurrentNPUStream(devId).stream();
    if (stream == nullptr) {
        throw std::runtime_error("get current npu stream fail");
    }
    return stream;
}

atb::Context *GetAtbContext()
{
    static thread_local std::shared_ptr<atb::Context> atbContext;
    if (atbContext) {
        return atbContext.get();
    }
    atb::Context *context = nullptr;
    atb::Status st = atb::CreateContext(&context);
    if (st != atb::NO_ERROR || !context) {
        throw std::runtime_error("create ATB context fail");
    }
    context->SetExecuteStream(GetCurrentStream());
    std::shared_ptr<atb::Context> shardContext(context, [](atb::Context *context) { atb::DestroyContext(context); });
    atbContext = shardContext;
    return atbContext.get();
}

atb::Tensor ConvertToAtbTensor(torch::Tensor &torchTensor)
{
    static const std::map<at::ScalarType, aclDataType> TORCH_TO_ACL_DTYPE_MAP = {
        {at::ScalarType::Bool, ACL_BOOL},    {at::ScalarType::Byte, ACL_UINT8},    {at::ScalarType::Char, ACL_INT8},
        {at::ScalarType::Half, ACL_FLOAT16}, {at::ScalarType::Float, ACL_FLOAT},   {at::ScalarType::Int, ACL_INT32},
        {at::ScalarType::Long, ACL_INT64},   {at::ScalarType::BFloat16, ACL_BF16}, {at::ScalarType::Short, ACL_INT16},
    };

    atb::Tensor atbTensor;
    if (!torchTensor.is_contiguous()) {
        torchTensor = torchTensor.contiguous();
    }
    if (!torchTensor.is_cpu()) {
        atbTensor.desc.format = static_cast<aclFormat>(at_npu::native::get_npu_format(torchTensor));
        atbTensor.deviceData = torchTensor.data_ptr();
    } else {
        atbTensor.hostData = torchTensor.data_ptr();
        atbTensor.desc.format = ACL_FORMAT_ND;
    }
    if (atbTensor.desc.format == ACL_FORMAT_NCHW) {
        atbTensor.desc.format = ACL_FORMAT_ND;
    }
    atbTensor.desc.shape.dimNum = torchTensor.sizes().size();
    for (uint64_t i = 0; i < torchTensor.sizes().size(); i++) {
        atbTensor.desc.shape.dims[i] = torchTensor.sizes()[i];
    }

    auto it = TORCH_TO_ACL_DTYPE_MAP.find(torchTensor.scalar_type());
    if (it != TORCH_TO_ACL_DTYPE_MAP.end()) {
        atbTensor.desc.dtype = it->second;
    }
    atbTensor.dataSize = atb::TensorUtil::CalcTensorDataSize(atbTensor);
    return atbTensor;
}

torch::Tensor CreateTorchTensorFromTensorDesc(const atb::TensorDesc &tensorDesc)
{
    static const std::map<aclDataType, at::ScalarType> ACL_TO_TORCH_DTYPE_MAP = {
        {ACL_BOOL, at::ScalarType::Bool},    {ACL_UINT8, at::ScalarType::Byte},    {ACL_INT8, at::ScalarType::Char},
        {ACL_FLOAT16, at::ScalarType::Half}, {ACL_FLOAT, at::ScalarType::Float},   {ACL_INT32, at::ScalarType::Int},
        {ACL_INT64, at::ScalarType::Long},   {ACL_BF16, at::ScalarType::BFloat16}, {ACL_INT16, at::ScalarType::Short},
    };
    at::TensorOptions options = at::TensorOptions();
    auto it = ACL_TO_TORCH_DTYPE_MAP.find(tensorDesc.dtype);
    if (it != ACL_TO_TORCH_DTYPE_MAP.end()) {
        options = options.dtype(it->second);
    }
    options = options.layout(torch::kStrided).requires_grad(false).device(torch_npu::utils::get_npu_device_type());
    torch::Tensor newTensor = at_npu::native::empty_with_format(
        at::IntArrayRef(tensorDesc.shape.dims, tensorDesc.shape.dimNum), options, tensorDesc.format);
    if (!newTensor.is_contiguous()) {
        newTensor = newTensor.contiguous();
    }
    return newTensor;
}
static bool IsEnvEnable(const char *envStr, bool defaultVal)
{
    const char *env = std::getenv(envStr);
    if (env == nullptr) {
        return defaultVal;
    }
    return strtol(env, nullptr, DECIMAL) != 0;
}

bool IsTaskQueueEnable()
{
    static bool isTaskQueueEnable =
        (!IsEnvEnable("ASCEND_LAUNCH_BLOCKING", false) && IsEnvEnable("TASK_QUEUE_ENABLE", true));
    return isTaskQueueEnable;
}
} // namespace Utils
} // namespace TorchAtb
