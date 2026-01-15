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
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/multinomial.h"

namespace {
constexpr size_t MULTINOMIAL_TENSOR_IN_COUNT = 1;
constexpr size_t MULTINOMIAL_TENSOR_OUT_COUNT = 1;
constexpr size_t TENSOR_THE_SECOND = 1;
constexpr size_t TENSOR_THE_THIRD = 2;
constexpr size_t TENSOR_THE_FOURTH = 3;
constexpr size_t TENSOR_THE_FIFTH = 4;
}

namespace AsdOps {
using namespace Mki;
class MultinomialOperation : public OperationBase {
public:
    explicit MultinomialOperation(const std::string &opName) noexcept : OperationBase(opName) {}
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        return GetKernelByName("MultinomialKernel");
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Multinomial), "OpParam is invalid",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() > 1,
                     "input size is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        return MultinomialInferShape(launchParam, outTensors);
    }

private:
    Status MultinomialInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        OpParam::Multinomial param = AnyCast<OpParam::Multinomial>(launchParam.GetParam());
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 2, "dim size of inTensor0 is invalid",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        outTensors[0].desc.dims[launchParam.GetInTensor(0).desc.dims.size() - 1] = param.numSamples;
        outTensors[0].desc.dtype = TENSOR_DTYPE_INT32;
        return Status::OkStatus();
    }
};
REG_OPERATION(MultinomialOperation);
} //    namespace AsdOps