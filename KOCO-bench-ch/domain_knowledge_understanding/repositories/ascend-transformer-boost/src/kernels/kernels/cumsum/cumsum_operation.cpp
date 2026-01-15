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
#include "asdops/params/cumsum.h"

namespace AsdOps {
using namespace Mki;
class CumsumOperation : public OperationBase {
public:
    explicit CumsumOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        auto dtype = launchParam.GetOutTensor(0).desc.dtype;
        auto &param = AnyCast<OpParam::Cumsum>(launchParam.GetParam());
        if (dtype == TENSOR_DTYPE_FLOAT16) {
            return param.deterministic ? GetKernelByName("CumsumF16DtmKernel") : GetKernelByName("CumsumF16Kernel");
        } else if (dtype == TENSOR_DTYPE_BF16) {
            return param.deterministic ? GetKernelByName("CumsumBF16DtmKernel") : GetKernelByName("CumsumBF16Kernel");
        } else {
            MKI_LOG(ERROR) << "Wrong Output dtype:" << GetStrWithDType(dtype);
            return nullptr;
        }
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Cumsum), "OpParam is invalid",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        OpParam::Cumsum param = AnyCast<OpParam::Cumsum>(launchParam.GetParam());
        SVector<int64_t> axis = param.axis;
        if (axis.size() != 1) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "null or multi axis is not support now");
        }

        outTensors[0].desc.dtype = launchParam.GetInTensor(0).desc.dtype;
        outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;
        outTensors[0].desc.dims = launchParam.GetInTensor(0).desc.dims;

        return Status::OkStatus();
    }
};
REG_OPERATION(CumsumOperation);
} // namespace AsdOps