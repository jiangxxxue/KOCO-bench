/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/base/operation_base.h>
#include <mki/utils/log/log.h>
#include <mki_loader/op_register.h>

#include "asdops/params/scatter_elements_v2.h"

namespace AsdOps {
using namespace Mki;
constexpr int64_t INPUT_NUM = 3;
constexpr int64_t OUTPUT_NUM = 1;
class ScatterElementsV2Operation : public OperationBase {
public:
    explicit ScatterElementsV2Operation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);

        // 检查输入张量数量是否为 3（input, indices, updates）
        MKI_CHECK(launchParam.GetInTensorCount() == 3, "Parameters are abnormal,input num invalid,", return nullptr);

        const auto &tensorDesc0 = launchParam.GetInTensor(0).desc; // 输入张量 0
        const auto &tensorDesc1 = launchParam.GetInTensor(1).desc; // 输入张量 1（例如索引张量）
        auto tensor0dtype = tensorDesc0.dtype;
        auto tensor1dtype = tensorDesc1.dtype;

        const auto &param = AnyCast<OpParam::ScatterElementsV2>(launchParam.GetParam());
        std::string reductionStr = "";

        if (param.reduction == OpParam::ScatterElementsV2::ReductionType::NONE) {
            reductionStr = "none";
        } else if (param.reduction == OpParam::ScatterElementsV2::ReductionType::ADD) {
            reductionStr = "add";
        } else {
            MKI_LOG(ERROR) << "reduction only support none or add";
        }

        switch (tensor0dtype) {
            case TENSOR_DTYPE_UINT8: return GetUint8BestKernel(tensor1dtype, reductionStr);
            case TENSOR_DTYPE_INT8: return GetInt8BestKernel(tensor1dtype, reductionStr);
            case TENSOR_DTYPE_INT32: return GetInt32BestKernel(tensor1dtype, reductionStr);
            case TENSOR_DTYPE_FLOAT16: return GetFloat16BestKernel(tensor1dtype, reductionStr);
            case TENSOR_DTYPE_FLOAT: return GetFloat32BestKernel(tensor1dtype, reductionStr);
            case TENSOR_DTYPE_BF16: return GetBfloat16BestKernel(tensor1dtype, reductionStr);
            default: return nullptr;
        }
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return INPUT_NUM; // ScatterElementsV2Operation has 3 inputs
    }
    int64_t GetOutputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return OUTPUT_NUM; // ScatterElementsV2Operation has 1 output
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        // 检查参数类型是否为 Scatter
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::ScatterElementsV2),
                  "ScatterElementsV2: no match param type",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "no match specificParam type."));

        // 获取 Scatter 参数
        OpParam::ScatterElementsV2 param = AnyCast<OpParam::ScatterElementsV2>(launchParam.GetParam());
        int64_t axis = param.axis;

        // 检查输入张量的形状是否合法
        const auto &inputTensorDesc = launchParam.GetInTensor(0).desc;  // input 张量
        const auto &indiceTensorDesc = launchParam.GetInTensor(1).desc; // indices 张量
        const auto &updateTensorDesc = launchParam.GetInTensor(2).desc; // updates 张量

        // axis 只能等于 -1 （与torch存在差异）
        MKI_CHECK(axis == -1, "ScatterElementsV2Operation: axis only support -1 ",
                  return Status::FailStatus(ERROR_INVALID_VALUE, "wrong input axis."));

        MKI_CHECK(inputTensorDesc.dtype == updateTensorDesc.dtype,
                  "ScatterElementsV2Operation: The dtype values of input_tensor and update must be the same.",
                  return Status::FailStatus(ERROR_INVALID_VALUE,
                                            "The dtype values of input_tensor and update must be the same."));

        Status dimsStatus = CheckDims(inputTensorDesc, indiceTensorDesc, updateTensorDesc);
        if (dimsStatus.Code() != 0)
            return dimsStatus;

        Status shapeStatus = CheckShape(inputTensorDesc, indiceTensorDesc, updateTensorDesc);
        if (shapeStatus.Code() != 0)
            return shapeStatus;

        // 设置输出张量的形状
        outTensors[0].desc = inputTensorDesc;
        return Status::OkStatus();
    }

private:
    Kernel *GetInt8BestKernel(TensorDType indiceDtype, std::string reduction) const
    {
        if (indiceDtype == TENSOR_DTYPE_INT32) {
            if (reduction == "add") {
                return GetKernelByName("ScatterElementsV2Int8Int32AddKernel");
            } else {
                return GetKernelByName("ScatterElementsV2Int8Int32NoneKernel");
            }
        } else if (indiceDtype == TENSOR_DTYPE_INT64) {
            if (reduction == "add") {
                return GetKernelByName("ScatterElementsV2Int8Int64AddKernel");
            } else {
                return GetKernelByName("ScatterElementsV2Int8Int64NoneKernel");
            }
        } else {
            MKI_LOG(ERROR) << "indiceDtype only support int32 or int64";
            return nullptr;
        }
    }

    Kernel *GetUint8BestKernel(TensorDType indiceDtype, std::string reduction) const
    {
        if (indiceDtype == TENSOR_DTYPE_INT32) {
            if (reduction == "add") {
                return GetKernelByName("ScatterElementsV2Uint8Int32AddKernel");
            } else {
                return GetKernelByName("ScatterElementsV2Uint8Int32NoneKernel");
            }
        } else if (indiceDtype == TENSOR_DTYPE_INT64) {
            if (reduction == "add") {
                return GetKernelByName("ScatterElementsV2Uint8Int64AddKernel");
            } else {
                return GetKernelByName("ScatterElementsV2Uint8Int64NoneKernel");
            }
        } else {
            MKI_LOG(ERROR) << "indiceDtype only support int32 or int64";
            return nullptr;
        }
    }

    Kernel *GetInt32BestKernel(TensorDType indiceDtype, std::string reduction) const
    {
        if (indiceDtype == TENSOR_DTYPE_INT32) {
            if (reduction == "add") {
                return GetKernelByName("ScatterElementsV2Int32Int32AddKernel");
            } else {
                return GetKernelByName("ScatterElementsV2Int32Int32NoneKernel");
            }
        } else if (indiceDtype == TENSOR_DTYPE_INT64) {
            if (reduction == "add") {
                return GetKernelByName("ScatterElementsV2Int32Int64AddKernel");
            } else {
                return GetKernelByName("ScatterElementsV2Int32Int64NoneKernel");
            }
        } else {
            MKI_LOG(ERROR) << "indiceDtype only support int32 or int64";
            return nullptr;
        }
    }

    Kernel *GetFloat16BestKernel(TensorDType indiceDtype, std::string reduction) const
    {
        if (indiceDtype == TENSOR_DTYPE_INT32) {
            if (reduction == "add") {
                return GetKernelByName("ScatterElementsV2Float16Int32AddKernel");
            } else {
                return GetKernelByName("ScatterElementsV2Float16Int32NoneKernel");
            }
        } else if (indiceDtype == TENSOR_DTYPE_INT64) {
            if (reduction == "add") {
                return GetKernelByName("ScatterElementsV2Float16Int64AddKernel");
            } else {
                return GetKernelByName("ScatterElementsV2Float16Int64NoneKernel");
            }
        } else {
            MKI_LOG(ERROR) << "indiceDtype only support int32 or int64";
            return nullptr;
        }
    }

    Kernel *GetFloat32BestKernel(TensorDType indiceDtype, std::string reduction) const
    {
        if (indiceDtype == TENSOR_DTYPE_INT32) {
            if (reduction == "add") {
                return GetKernelByName("ScatterElementsV2Float32Int32AddKernel");
            } else {
                return GetKernelByName("ScatterElementsV2Float32Int32NoneKernel");
            }
        } else if (indiceDtype == TENSOR_DTYPE_INT64) {
            if (reduction == "add") {
                return GetKernelByName("ScatterElementsV2Float32Int64AddKernel");
            } else {
                return GetKernelByName("ScatterElementsV2Float32Int64NoneKernel");
            }
        } else {
            MKI_LOG(ERROR) << "indiceDtype only support int32 or int64";
            return nullptr;
        }
    }

    Kernel *GetBfloat16BestKernel(TensorDType indiceDtype, std::string reduction) const
    {
        if (indiceDtype == TENSOR_DTYPE_INT32) {
            if (reduction == "add") {
                return GetKernelByName("ScatterElementsV2Bfloat16Int32AddKernel");
            } else {
                return GetKernelByName("ScatterElementsV2Bfloat16Int32NoneKernel");
            }
        } else if (indiceDtype == TENSOR_DTYPE_INT64) {
            if (reduction == "add") {
                return GetKernelByName("ScatterElementsV2Bfloat16Int64AddKernel");
            } else {
                return GetKernelByName("ScatterElementsV2Bfloat16Int64NoneKernel");
            }
        } else {
            MKI_LOG(ERROR) << "indiceDtype only support int32 or int64";
            return nullptr;
        }
    }

    Status CheckDims(TensorDesc inputTensorDesc, TensorDesc indiceTensorDesc, TensorDesc updateTensorDesc) const
    {
        // dims size 是tensor的维度
        // indice和update的dim需要完全相同
        MKI_CHECK(indiceTensorDesc.dims.size() == updateTensorDesc.dims.size(),
                  "ScatterElementsV2Operation: The dimension of indice must be the same as that of update.",
                  return Status::FailStatus(ERROR_INVALID_VALUE,
                                            "The dimension of indice must be the same as that of update."));

        // indice的维度需要和input_tensor的维度相同
        MKI_CHECK(inputTensorDesc.dims.size() == indiceTensorDesc.dims.size(),
                  "ScatterElementsV2Operation: The dimension of indice must be the same as that of input_tensor.",
                  return Status::FailStatus(ERROR_INVALID_VALUE,
                                            "The dimension of indice must be the same as that of input_tensor."));
        return Status::OkStatus();
    }

    Status CheckShape(TensorDesc inputTensorDesc, TensorDesc indiceTensorDesc, TensorDesc updateTensorDesc) const
    {
        // dims 数组中存储的是tensor的shape
        // indice和update的shape需要完全相同
        for (unsigned int i = 0; i <= indiceTensorDesc.dims.size() - 1; i++) {
            MKI_CHECK(indiceTensorDesc.dims[i] == updateTensorDesc.dims[i],
                      "ScatterElementsV2Operation: The shapes of indice must be the same as that of update.",
                      return Status::FailStatus(ERROR_INVALID_VALUE,
                                                "The shapes of indice must be the same as that of update."));
            MKI_CHECK(indiceTensorDesc.dims[i] != 0,
                      "ScatterElementsV2Operation: The value of indice_tensor cannot be 0 for any dimension.",
                      return Status::FailStatus(ERROR_INVALID_VALUE,
                                                "The value of indice_tensor cannot be 0 for any dimension."));
            MKI_CHECK(inputTensorDesc.dims[i] != 0,
                      "ScatterElementsV2Operation: The value of input_tensor cannot be 0 for any dimension.",
                      return Status::FailStatus(ERROR_INVALID_VALUE,
                                                "The value of input_tensor cannot be 0 for any dimension."));
        }

        // input_tensor 和 indice 在非尾轴和非0轴上的shape必须一致
        for (unsigned int i = 1; i < inputTensorDesc.dims.size() - 1; i++) {
            MKI_CHECK(
                inputTensorDesc.dims[i] == indiceTensorDesc.dims[i],
                "ScatterElementsV2Operation: The shapes of input_tensor and indices_tensor on the non-tail axis and "
                "non-zero axis must be the same.",
                return Status::FailStatus(ERROR_INVALID_VALUE, "The shapes of input_tensor and indices_tensor s on the "
                                                               "non-tail axis and non-zero axis must be the same."));
        }

        // indices_tensor 0轴和尾轴不大于inpute_tensor的0轴和尾轴
        MKI_CHECK(inputTensorDesc.dims[0] >= indiceTensorDesc.dims[0],
                  "ScatterElementsV2Operation: The shapes of input_tensor and indices_tensor on the non-tail axis and "
                  "non-zero axis must be the same.",
                  return Status::FailStatus(ERROR_INVALID_VALUE,
                                            "The shapes of input_tensor and indices_tensor s on the "
                                            "non-tail axis and non-zero axis must be the same."));

        // indices_tensor 0轴和尾轴不大于inpute_tensor的0轴和尾轴
        MKI_CHECK(inputTensorDesc.dims[inputTensorDesc.dims.size() - 1] >=
                      indiceTensorDesc.dims[indiceTensorDesc.dims.size() - 1],
                  "ScatterElementsV2Operation: The 0 axis and tail axis of inpute_tensor are not greater than the 0 "
                  "axis and tail axis of inpute_tensor.",
                  return Status::FailStatus(ERROR_INVALID_VALUE,
                                            "The 0 axis and tail axis of inpute_tensor are not greater "
                                            "than the 0 axis and tail axis of inpute_tensor."));

        return Status::OkStatus();
    }
};

REG_OPERATION(ScatterElementsV2Operation);

} // namespace AsdOps