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
#include <mki/utils/log/log.h>
#include <mki_loader/op_register.h>
#include "asdops/params/asstrided.h"
#include "kernels/utils/common.h"

namespace AsdOps {
using namespace Mki;
static const size_t INPUT_TENSOR_INDEX = 0;

class AsStridedOperation : public OperationBase {
public:
    explicit AsStridedOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        auto dtype = launchParam.GetOutTensor(0).desc.dtype;
        if (dtype == TENSOR_DTYPE_FLOAT16) {
            return GetKernelByName("AsStridedF16Int64Kernel");
        } else if (dtype == TENSOR_DTYPE_FLOAT) {
            return GetKernelByName("AsStridedF32Int64Kernel");
        } else if (dtype == TENSOR_DTYPE_INT64) {
            return GetKernelByName("AsStridedInt64Int64Kernel");
        } else {
            return nullptr;
        }
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        TensorDType dtype = launchParam.GetInTensor(INPUT_TENSOR_INDEX).desc.dtype;
        TensorFormat format = launchParam.GetInTensor(INPUT_TENSOR_INDEX).desc.format;
        MKI_CHECK(dtype == TENSOR_DTYPE_FLOAT16 || dtype == TENSOR_DTYPE_FLOAT || dtype == TENSOR_DTYPE_INT64,
                  "input tensor dtype is invalid: " << GetStrWithDType(dtype),
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR,
                                            "input tensor dtype should be float16 or float or int64"));
        MKI_CHECK(format == TENSOR_FORMAT_ND, "input tensor format is invalid: " << GetStrWithFormat(format),
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "input tensor format should be ND"));
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::AsStrided), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        auto param = AnyCast<OpParam::AsStrided>(launchParam.GetParam());
        MKI_CHECK(param.size.size() == param.stride.size(),
                  "OpParam size dim num and stride dim num are different: " << param.size.size() << " and "
                                                                            << param.stride.size(),
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR,
                                            "OpParam size dim num and stride dim num are different"));
        MKI_CHECK(param.offset.size() == 1, "OpParam offset dim is invalid: " << param.offset.size(),
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam offset dim is invalid"));
        MKI_CHECK(CheckParamAllPositive(param.size), "OpParam size exists negetive number",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam size is invalid"));
        outTensors[0].desc = {dtype, format, param.size, {}, 0};
        // check tensor num
        return Status::OkStatus();
    }
};
REG_OPERATION(AsStridedOperation);
} //    namespace AsdOps