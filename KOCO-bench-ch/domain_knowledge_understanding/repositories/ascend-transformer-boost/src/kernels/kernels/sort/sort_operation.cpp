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
#include "asdops/params/sort.h"

namespace AsdOps {
using namespace Mki;
class SortOperation : public OperationBase {
public:
    explicit SortOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        auto dtype = launchParam.GetOutTensor(0).desc.dtype;
        if (dtype == TENSOR_DTYPE_FLOAT16) {
            return GetKernelByName("TopKDescF16Kernel");
        } else if (dtype == TENSOR_DTYPE_BF16) {
            return GetKernelByName("TopKDescBF16Kernel");
        } else if (dtype == TENSOR_DTYPE_FLOAT) {
            return GetKernelByName("TopKDescF32Kernel");
        } else {
            return nullptr;
        }
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return 2; // sort has 2 outputs
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        TensorDesc desc0 = launchParam.GetInTensor(0).desc;
        const SVector<int64_t> &dims0 = desc0.dims;
        if (dims0.size() == 0) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "TopK input tensor dim should be >=1.");
        }

        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Sort), "sort: no match param type",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "no match specificParam type."));
        OpParam::Sort param = AnyCast<OpParam::Sort>(launchParam.GetParam());
        const SVector<int32_t> &num = param.num;
        if (num.size() == 0 || num[0] <= 0) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "TopK num invalid.");
        }

        if (dims0[dims0.size() - 1] < num[0]) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "TopK input tensor last dim at least k.");
        }

        SVector<int64_t> outDim = dims0;
        outDim[dims0.size() - 1] = num[0];

        TensorDType dtype0 = desc0.dtype;
        TensorFormat format = desc0.format;
        outTensors[0].desc = {dtype0, format, outDim, {}, 0};
        outTensors[1].desc = {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, outDim, {}, 0};

        return Status::OkStatus();
    }
};
REG_OPERATION(SortOperation);
} // namespace AsdOps
