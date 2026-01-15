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
#include <mki/utils/log/log.h>
#include "asdops/params/softmax.h"

namespace AsdOps {
using namespace Mki;
class SoftmaxOperation : public OperationBase {
public:
    explicit SoftmaxOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        auto dtype = launchParam.GetOutTensor(0).desc.dtype;
        if (dtype == TENSOR_DTYPE_FLOAT16) {
            return GetKernelByName("SoftmaxF16Kernel");
        } else if (dtype == TENSOR_DTYPE_FLOAT) {
            return GetKernelByName("SoftmaxF32Kernel");
        } else if (dtype == TENSOR_DTYPE_BF16) {
            return GetKernelByName("SoftmaxBF16Kernel");
        } else {
            return nullptr;
        }
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Softmax), "softmax: no match param type",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "no match specificParam type."));
        OpParam::Softmax param = AnyCast<OpParam::Softmax>(launchParam.GetParam());
        SVector<int64_t> axes = param.axes;
        MKI_CHECK(axes.size() != 0, "softmax: axes should be set.",
            return Status::FailStatus(ERROR_INVALID_VALUE, "no input axes."));
        for (size_t i = 1; i < axes.size(); i++) {
            MKI_CHECK((axes[i] == axes[i - 1] + 1) && axes[i] != 0, "softmax: input axes should be consistent.",
                return Status::FailStatus(ERROR_INVALID_VALUE, "input axes is not consistent."));
        }
        const auto &inTensorDesc0 = launchParam.GetInTensor(0).desc;
        const auto axesBound = static_cast<int64_t>(inTensorDesc0.dims.size());
        MKI_CHECK(axes[axes.size() - 1] < axesBound && axes[0] >= -axesBound,
                     "softmax: axes values should fall in the range [" << -axesBound << ", " << axesBound << ").",
                     return Status::FailStatus(ERROR_INVALID_VALUE, "wrong input axes."));

        outTensors[0].desc = inTensorDesc0;
        return Status::OkStatus();
    }
};
REG_OPERATION(SoftmaxOperation);
} // namespace AsdOps
