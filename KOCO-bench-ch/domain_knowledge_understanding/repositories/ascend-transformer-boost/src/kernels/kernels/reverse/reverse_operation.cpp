/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <mki/utils/log/log.h>
#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include "asdops/params/reverse.h"

namespace AsdOps {
using namespace Mki;
class ReverseOperation : public OperationBase {
public:
    explicit ReverseOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        auto dtype = launchParam.GetOutTensor(0).desc.dtype;
        if (dtype == TENSOR_DTYPE_FLOAT16) {
            return GetKernelByName("ReverseF16Kernel");
        } else if (dtype == TENSOR_DTYPE_BF16) {
            return GetKernelByName("ReverseBF16Kernel");
        } else if (dtype == TENSOR_DTYPE_FLOAT) {
            return GetKernelByName("ReverseF32Kernel");
        }
        MKI_LOG(ERROR) << "No kernel for Reverse dtype " << GetStrWithDType(dtype);
        return nullptr;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        SVector<int64_t> dimsIn = launchParam.GetInTensor(0).desc.dims;
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Reverse), "OpParam is invalid",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        OpParam::Reverse param = AnyCast<OpParam::Reverse>(launchParam.GetParam());
        SVector<int32_t> axis = param.axis;
        if (axis.size() > dimsIn.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "axis is out of range");
        }

        std::sort(axis.begin(), axis.end());
        for (uint64_t i = 0; i < axis.size(); i++) {
            if (axis[i] >= static_cast<int64_t>(dimsIn.size())) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "axis is out of range");
            }
            if ((i < axis.size() - 1) && (axis[i] == axis[i + 1])) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "axis is specified more than once");
            }
        }
        outTensors[0].desc= launchParam.GetInTensor(0).desc;
        return Status::OkStatus();
    }
};
REG_OPERATION(ReverseOperation);
} //    namespace AsdOps