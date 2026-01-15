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
#include "asdops/params/copy.h"

namespace AsdOps {
using namespace Mki;
class CopyOperation : public OperationBase {
public:
    explicit CopyOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        auto dtype = launchParam.GetOutTensor(0).desc.dtype;
        if (dtype == TENSOR_DTYPE_INT64) {
            return GetKernelByName("ViewCopyInt64Kernel");
        } else if (dtype == TENSOR_DTYPE_INT32) {
            return GetKernelByName("ViewCopyInt32Kernel");
        } else if (dtype == TENSOR_DTYPE_FLOAT) {
            return GetKernelByName("ViewCopyF32Kernel");
        } else if (dtype == TENSOR_DTYPE_FLOAT16) {
            return GetKernelByName("ViewCopyF16Kernel");
        } else if (dtype == TENSOR_DTYPE_BF16) {
            return GetKernelByName("ViewCopyBF16Kernel");
        } else {
            return nullptr;
        }
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return 2; // viewcopy 2 input
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        TensorDType dtype = launchParam.GetInTensor(0).desc.dtype;
        TensorFormat format = launchParam.GetInTensor(0).desc.format;
        SVector<int64_t> dims = launchParam.GetInTensor(0).desc.dims;

        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Copy), "OpParam is invalid",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        auto param = AnyCast<OpParam::Copy>(launchParam.GetParam());
        MKI_CHECK(param.dstSize.size() == param.dstStride.size(),
            "OpParam size dim num and stride dim num are different",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        outTensors[0].desc = {dtype, format, dims, {}, 0};
        return Status::OkStatus();
    }
};
REG_OPERATION(CopyOperation);
} //    namespace AsdOps