/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/log/log.h>
#include "asdops/params/activation.h"

namespace {
constexpr int64_t SWIGLU_SPLIT_NUM = 2;
} // namespace

namespace AsdOps {
using namespace Mki;
class ActivationOperation : public OperationBase {
public:
    explicit ActivationOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        auto dtype = launchParam.GetOutTensor(0).desc.dtype;
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Activation), "OpParam is invalid", return nullptr);
        OpParam::Activation param = launchParam.GetParam<OpParam::Activation>();
        switch (param.activationType) {
            case OpParam::Activation::ACTIVATION_RELU:
                if (dtype == TENSOR_DTYPE_FLOAT) {
                    return GetKernelByName("ReluF32Kernel");
                } else if (dtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("ReluBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ACTIVATION_RELU dtype " << GetStrWithDType(dtype);
                return nullptr;
            case OpParam::Activation::ACTIVATION_GELU:
                if (dtype == TENSOR_DTYPE_FLOAT) {
                    return (param.approx != 0) ? GetKernelByName("GeluF32Kernel")
                                               : GetKernelByName("GeluApproxF32Kernel");
                }
                if (dtype == TENSOR_DTYPE_FLOAT16) {
                    return (param.approx != 0) ? GetKernelByName("GeluF16Kernel")
                                               : GetKernelByName("GeluApproxF16Kernel");
                }
                if (dtype == TENSOR_DTYPE_BF16) {
                    return (param.approx != 0) ? GetKernelByName("GeluBF16Kernel")
                                               : GetKernelByName("GeluApproxBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ACTIVATION_GELU dtype " << GetStrWithDType(dtype);
                return nullptr;
            case OpParam::Activation::ACTIVATION_FAST_GELU:
                if (dtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("FastGeluF16Kernel");
                }
                if (dtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("FastGeluBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ACTIVATION_FAST_GELU dtype " << GetStrWithDType(dtype);
                return nullptr;
            case OpParam::Activation::ACTIVATION_SWISH:
                if (dtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("SwishF16Kernel");
                }
                if (dtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("SwishBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ACTIVATION_SWISH dtype " << GetStrWithDType(dtype);
                return nullptr;
            case OpParam::Activation::ACTIVATION_LOG:
                if (dtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("LogF16Kernel");
                }
                if (dtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("LogBF16Kernel");
                }
                if (dtype == TENSOR_DTYPE_FLOAT) {
                    return GetKernelByName("LogF32Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ACTIVATION_LOG inDtype " << GetStrWithDType(dtype);
                return nullptr;
            case OpParam::Activation::ACTIVATION_SWIGLU_FORWARD:
                return GetKernelByName("SwiGluForwardKernel");
            case OpParam::Activation::ACTIVATION_SWIGLU_BACKWARD:
                return GetKernelByName("SwiGluBackwardKernel");
            case OpParam::Activation::ACTIVATION_SIGMOID:
                if (dtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("SigmoidF16Kernel");
                }
                if (dtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("SigmoidBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ACTIVATION_SIGMOID inDtype " << GetStrWithDType(dtype);
                return nullptr;
            case OpParam::Activation::ACTIVATION_FASTER_GELU_FORWARD:
                return GetKernelByName("FasterGeluForwardKernel");
            default:
                return nullptr;
        }
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::Activation),
                     "OpParam is invalid", return false);
        auto param = AnyCast<OpParam::Activation>(specificParam);
        switch (param.activationType) {
            case OpParam::Activation::ACTIVATION_SWIGLU_FORWARD:
                return 1; // swiglu_forward has 1 input
            case OpParam::Activation::ACTIVATION_SWIGLU_BACKWARD:
                return 2; // swiglu_backward has 2 input
            default:
                return 1;
        }
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Activation), "OpParam is invalid",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        auto param = AnyCast<OpParam::Activation>(launchParam.GetParam());
        switch (param.activationType) {
            case OpParam::Activation::ACTIVATION_SWIGLU_FORWARD:
                return InferShapeSwiGluForward(launchParam, outTensors);
            case OpParam::Activation::ACTIVATION_SWIGLU_BACKWARD:
                return InferShapeSwiGluBackward(launchParam, outTensors);
            default:
                // 以下代码勿删，gelu、relu等算子运行需要
                outTensors[0].desc = launchParam.GetInTensor(0).desc;
                return Status::OkStatus();
        }
    }

private:
    Status InferShapeSwiGluForward(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        MKI_LOG(INFO) << "InferShapeSwiGluForward Start";
        const SVector<int64_t> &xshape = launchParam.GetInTensor(0).desc.dims;
        MKI_CHECK(!xshape.empty(), "xshape should not be empty",
                     return Status::FailStatus(ERROR_INVALID_VALUE, "xshape should not be empty"));
        auto param = AnyCast<OpParam::Activation>(launchParam.GetParam());
        int32_t splitDim = param.dim;
        if (splitDim < 0) {
            splitDim += static_cast<int32_t>(xshape.size());
        }
        MKI_CHECK(splitDim >= 0 && splitDim < static_cast<int32_t>(xshape.size()),
                     "The value of attr [dim] must be in the range [-" << xshape.size() <<", " <<
                     xshape.size() - 1 << "], but got [" << splitDim << "].",
                     return Status::FailStatus(ERROR_INVALID_VALUE, "input dim error"));
        const uint32_t splitNum = 2;
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        outTensors[0].desc.dims[splitDim] = xshape[splitDim] / splitNum;
        MKI_LOG(INFO) << "InferShapeSwiGluForward Passed";
        return Status::OkStatus();
    }

    Status InferShapeSwiGluBackward(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        MKI_LOG(INFO) << "InferShapeSwiGluBackward Start";
        const SVector<int64_t> &xshape = launchParam.GetInTensor(DIM_1).desc.dims;
        MKI_CHECK(!xshape.empty(), "xshape should not be empty",
                     return Status::FailStatus(ERROR_INVALID_VALUE, "xshape should not be empty"));
        auto param = AnyCast<OpParam::Activation>(launchParam.GetParam());
        int32_t splitDim = param.dim;
        if (splitDim < 0) {
            splitDim += static_cast<int32_t>(xshape.size());
        }
        MKI_CHECK(splitDim >= 0 && splitDim < static_cast<int64_t>(xshape.size()),
                     "The value of attr [dim] must be in the range [-" << xshape.size() <<", " <<
                     xshape.size() - 1 << "], but got [" << splitDim << "].",
                     return Status::FailStatus(ERROR_INVALID_VALUE, "input dim error")); // 校验属性

        auto yGradDim = launchParam.GetInTensor(DIM_0).desc.dims;
        auto xDim = launchParam.GetInTensor(DIM_1).desc.dims;
        int32_t yGradDimNum = static_cast<int32_t>(launchParam.GetInTensor(DIM_0).desc.dims.size());
        int32_t xDimNum = static_cast<int32_t>(launchParam.GetInTensor(DIM_1).desc.dims.size());
        MKI_CHECK(xDimNum == yGradDimNum,
            "inTensor0 & inTensor1 dimNum does not match, should be same",
            return Status::FailStatus(ERROR_INVALID_VALUE, "input tensor dimNum error")); // 校验输入dimNum
        for (int32_t i = 0; i < yGradDimNum; i++) { // 校验输入shape信息
            if (i == splitDim) {
                MKI_CHECK(xDim[i] == SWIGLU_SPLIT_NUM * yGradDim[i],
                    "Dims[" << i << "] of inTensor0 should be half of inTnesor1.",
                    return Status::FailStatus(ERROR_INVALID_VALUE, "input tensor shape error"));
            } else {
                MKI_CHECK(xDim[i] == yGradDim[i],
                    "Dims[" << i << "] of inTensor0 and inTensor1 does not match, should be same.",
                    return Status::FailStatus(ERROR_INVALID_VALUE, "input tensor shape error"));
            }
        }
        outTensors[0].desc = launchParam.GetInTensor(DIM_1).desc;
        MKI_LOG(INFO) << "InferShapeSwiGluBackward Passed";
        return Status::OkStatus();
    }
};
REG_OPERATION(ActivationOperation);
} // namespace AsdOps
