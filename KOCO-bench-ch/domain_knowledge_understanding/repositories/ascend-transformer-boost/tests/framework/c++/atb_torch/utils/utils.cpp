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
#include <iostream>
#include <sys/stat.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#pragma GCC diagnostic pop
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <mki/utils/file_system/file_system.h>
#include <acl/acl_rt.h>
#include <atb/utils/log.h>
#include "atb/utils/config.h"
#include "atb/utils/tensor_util.h"

aclrtStream Utils::GetCurrentStream()
{
    int32_t devId = 0;
    aclrtGetDevice(&devId);
    aclrtStream stream = c10_npu::getCurrentNPUStream(devId).stream();
    ATB_LOG_IF(stream == nullptr, ERROR) << "get current stream fail";
    return stream;
}

int64_t Utils::GetTensorNpuFormat(const at::Tensor &tensor)
{
    return at_npu::native::get_npu_format(tensor);
}

at::Tensor Utils::NpuFormatCast(const at::Tensor &tensor)
{
    return at_npu::native::npu_format_cast(tensor, GetTensorNpuFormat(tensor));
}

void Utils::BuildVariantPack(const std::vector<torch::Tensor> &inTensors, const std::vector<torch::Tensor> &outTensors,
                             atb::VariantPack &variantPack)
{
    for (size_t i = 0; i < inTensors.size(); ++i) {
        variantPack.inTensors.push_back(AtTensor2Tensor(inTensors.at(i)));
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        variantPack.outTensors.push_back(AtTensor2Tensor(outTensors.at(i)));
    }
}

atb::Tensor Utils::AtTensor2Tensor(const at::Tensor &atTensor)
{
    static std::map<at::ScalarType, aclDataType> dtypeMap = {
        {at::ScalarType::Bool, ACL_BOOL},   {at::ScalarType::Byte, ACL_UINT8},
        {at::ScalarType::Char, ACL_INT8},   {at::ScalarType::Half, ACL_FLOAT16},
        {at::ScalarType::Float, ACL_FLOAT}, {at::ScalarType::Int, ACL_INT32},
        {at::ScalarType::Long, ACL_INT64}, {at::ScalarType::BFloat16, ACL_BF16},
        {at::ScalarType::Short, ACL_INT16},
    };

    ATB_LOG_IF(!atTensor.is_contiguous(), FATAL) << "atTensor is not contiguous";
    atb::Tensor tensor;
    tensor.desc.format = static_cast<aclFormat>(GetTensorNpuFormat(atTensor));
    tensor.deviceData = atTensor.data_ptr();
    if (tensor.deviceData != nullptr) {
        tensor.desc.shape.dimNum = atTensor.sizes().size();
        for (uint64_t i = 0; i < atTensor.sizes().size(); i++) {
            tensor.desc.shape.dims[i] = atTensor.sizes()[i];
        }
    }

    auto it = dtypeMap.find(atTensor.scalar_type());
    if (it != dtypeMap.end()) {
        tensor.desc.dtype = it->second;
    } else {
        ATB_LOG(ERROR) << "not support dtype:" << atTensor.scalar_type();
    }

    tensor.dataSize = atb::TensorUtil::CalcTensorDataSize(tensor);

    return tensor;
}

at::Tensor Utils::CreateAtTensorFromTensorDesc(const atb::TensorDesc &tensorDesc)
{
    static std::map<aclDataType, at::ScalarType> dtypeMap = {
        {ACL_BOOL, at::ScalarType::Bool},   {ACL_UINT8, at::ScalarType::Byte},
        {ACL_INT8, at::ScalarType::Char},   {ACL_FLOAT16, at::ScalarType::Half},
        {ACL_FLOAT, at::ScalarType::Float}, {ACL_INT32, at::ScalarType::Int},
        {ACL_INT64, at::ScalarType::Long}, {ACL_BF16, at::ScalarType::BFloat16},
        {ACL_INT16, at::ScalarType::Short},
    };
    at::TensorOptions options = at::TensorOptions();
    auto it = dtypeMap.find(tensorDesc.dtype);
    if (it != dtypeMap.end()) {
        options = options.dtype(it->second);
    } else {
        ATB_LOG(ERROR) << "not support dtype:" << tensorDesc.dtype;
    }

    options = options.layout(torch::kStrided).requires_grad(false).device(torch_npu::utils::get_npu_device_type());

    ATB_LOG(INFO) << "empty_with_format stat, " << atb::TensorUtil::TensorDescToString(tensorDesc);
    at::Tensor newTensor = at_npu::native::empty_with_format(
        at::IntArrayRef(tensorDesc.shape.dims, tensorDesc.shape.dimNum), options, tensorDesc.format);
    newTensor.zero_();
    ATB_LOG(INFO) << "empty_with_format end, newTensor.format:" << GetTensorNpuFormat(newTensor)
                  << ", is_contiguous:" << newTensor.is_contiguous();
    if (GetTensorNpuFormat(newTensor) != tensorDesc.format) {
        ATB_LOG(WARN) << "empty_with_format newTensor.format:" << GetTensorNpuFormat(newTensor)
                      << " != " << tensorDesc.format;
    }
    if (!newTensor.is_contiguous()) {
        newTensor = newTensor.contiguous();
    }

    ATB_LOG(INFO) << "empty_with_format success, newTensor.options:" << newTensor.options()
                  << ", format:" << GetTensorNpuFormat(newTensor) << ", is_contiguous:" << newTensor.is_contiguous();

    return newTensor;
}

void Utils::ContiguousAtTensor(std::vector<torch::Tensor> &atTensors)
{
    for (size_t i = 0; i < atTensors.size(); ++i) {
        if (!atTensors.at(i).is_contiguous()) {
            atTensors.at(i) = atTensors.at(i).contiguous();
        }
    }
}

void Utils::ContiguousAtTensor(torch::Tensor &atTensor)
{
    if (!atTensor.is_contiguous()) {
        atTensor = atTensor.contiguous();
    }
}
