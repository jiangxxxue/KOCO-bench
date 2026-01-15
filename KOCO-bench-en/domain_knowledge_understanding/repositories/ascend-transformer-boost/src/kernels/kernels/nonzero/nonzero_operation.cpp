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

namespace {
constexpr size_t NONZERO_TENSOR_IN_COUNT = 1;
constexpr size_t NONZERO_TENSOR_OUT_COUNT = 2;
}

namespace AsdOps {
using namespace Mki;
class NonzeroOperation : public OperationBase {
public:
    explicit NonzeroOperation(const std::string &opName) noexcept : OperationBase(opName) {}
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        return GetKernelByName("NonzeroKernel");
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return NONZERO_TENSOR_IN_COUNT;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return NONZERO_TENSOR_OUT_COUNT;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        return NonzeroInferShape(launchParam, outTensors);
    }

private:
    Status NonzeroInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        outTensors[1].desc = launchParam.GetInTensor(0).desc;
        outTensors[0].desc.dims.resize(NONZERO_TENSOR_OUT_COUNT);
        outTensors[0].desc.dims[0] = static_cast<int64_t>(launchParam.GetInTensor(0).desc.dims.size());
        outTensors[0].desc.dims[1] = launchParam.GetInTensor(0).Numel();
        outTensors[1].desc.dims.resize(1);
        outTensors[1].desc.dims[0] = 1;
        
        return Status::OkStatus();
    }
};
REG_OPERATION(NonzeroOperation);
} //    namespace AsdOps