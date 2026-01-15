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
#include "asdops/params/params.h"

namespace AsdOps {
using namespace Mki;
class ConcatOperation : public OperationBase {
public:
    explicit ConcatOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Fail to check consistent", return nullptr);
        auto dtype = launchParam.GetOutTensor(0).desc.dtype;
        if (dtype == TENSOR_DTYPE_FLOAT) {
            return GetKernelByName("ConcatF32Input2Kernel");
        } else {
            return GetKernelByName("ConcatF16Input2Kernel");
        }
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return 2; // concat has 2 inputs
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Concat), "concat: no match param type",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        OpParam::Concat param = AnyCast<OpParam::Concat>(launchParam.GetParam());
        TensorDType dtype0 = launchParam.GetInTensor(0).desc.dtype;
        TensorDType dtype1 = launchParam.GetInTensor(1).desc.dtype;
        SVector<int64_t> dims = launchParam.GetInTensor(0).desc.dims;
        SVector<int64_t> dims1 = launchParam.GetInTensor(1).desc.dims;

        TensorFormat format = launchParam.GetInTensor(0).desc.format;
        int64_t dimSize = static_cast<int64_t>(dims.size());
        int64_t dim1Size = static_cast<int64_t>(dims1.size());
        int64_t concatDim = param.concatDim;
        if (concatDim < 0) {
            concatDim += dimSize;
        }
        if (concatDim < 0 || concatDim >= dimSize || concatDim >= dim1Size) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "Incorrect concatDim.");
        }

        if ((dtype0 == TENSOR_DTYPE_FLOAT16 || dtype0 == TENSOR_DTYPE_FLOAT || dtype0 == TENSOR_DTYPE_BF16) &&
            (dtype1 == dtype0)) {
            dims.at(concatDim) = dims.at(concatDim) + dims1.at(concatDim);
            outTensors[0].desc = {dtype0, format, dims, {}, 0};
            return Status::OkStatus();
        } else {
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Unsupported input descriptor.");
        }
    }
};
REG_OPERATION(ConcatOperation);
} // namespace AsdOps