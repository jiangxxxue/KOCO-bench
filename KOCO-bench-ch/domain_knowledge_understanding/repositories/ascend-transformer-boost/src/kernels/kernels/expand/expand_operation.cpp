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
#include "asdops/params/expand.h"
#include "kernels/utils/common.h"
#include "asdops/common.h"

namespace AsdOps {
using namespace Mki;
class ExpandOperation : public OperationBase {
public:
    explicit ExpandOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        auto dtype = launchParam.GetOutTensor(0).desc.dtype;
        if (dtype == TENSOR_DTYPE_FLOAT16) {
            return GetKernelByName("ExpandF16Kernel");
        } else if (dtype == TENSOR_DTYPE_BF16) {
            return GetKernelByName("ExpandBF16Kernel");
        }
        MKI_LOG(ERROR) << "No kernel for Expand dtype " << GetStrWithDType(dtype);
        return nullptr;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        TensorDType dtype0 = launchParam.GetInTensor(0).desc.dtype;
        TensorFormat format = launchParam.GetInTensor(0).desc.format;
        SVector<int64_t> dims0 = launchParam.GetInTensor(0).desc.dims;
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Expand), "OpParam is invalid",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        OpParam::Expand param = AnyCast<OpParam::Expand>(launchParam.GetParam());
        SVector<int64_t> dims1 = param.shape;
        if (dims1.size() > DIM_8) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "param_.multiples dimNum should <= MAX_DIM(8)");
        }
        if (dims0.size() > dims1.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "shape dim should larger than A tensor");
        }

        if (dtype0 != TENSOR_DTYPE_FLOAT16 && dtype0 != TENSOR_DTYPE_BF16) {
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Unsupported input descriptor.");
        }

        MKI_CHECK(CheckParamAllPositive(param.shape), "OpParam shape exists negetive number.",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        int64_t largerSize = static_cast<int64_t>(dims1.size());
        int64_t idx = 0;
        while (idx < largerSize) {
            auto dimIdx0 = static_cast<int64_t>(dims0.size()) - idx - 1;
            auto dimIdx1 = static_cast<int64_t>(dims1.size()) - idx - 1;
            auto dim0 = dimIdx0 < 0 ? 1 : dims0[dimIdx0];
            auto dim1 = dims1[dimIdx1];
            if ((dim0 <= 0) || !((dim0 == dim1) || ((dim0 < dim1) && (dim0 == 1)))) {
                MKI_LOG(ERROR) << "cannot support this dim of expand";
                return Status::FailStatus(ERROR_INVALID_VALUE);
            }
            idx++;
        }
        outTensors[0].desc = {dtype0, format, param.shape, {}, 0};
        return Status::OkStatus();
    }
};
REG_OPERATION(ExpandOperation);
} //    namespace AsdOps