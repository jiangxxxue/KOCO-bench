/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <mki/base/operation_base.h>
#include <mki/utils/log/log.h>
#include <mki/utils/const/op_const.h>
#include <mki_loader/op_register.h>
#include "atbops/params/params.h"

namespace AtbOps {
using namespace Mki;
class SwiGluQuantOperation : public OperationBase {
public:
    explicit SwiGluQuantOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        return DIM_1;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        return DIM_2;
    }

    bool CheckSwiGluQuant(const LaunchParam &launchParam) const
    {
        auto &inTensor0 = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(inTensor0.desc.dims.size() == DIM_2, "dim size of inTensor0 is invalid", return false);
        size_t dimNum = inTensor0.desc.dims.size();
        for (size_t dimIndex = DIM_0; dimIndex < dimNum; dimIndex++) {
            MKI_CHECK(inTensor0.desc.dims[dimIndex] != DIM_0,
                      "Any dimension of the input vector cannot be equal to zero.", return false);
        }
        MKI_CHECK(inTensor0.desc.dims[DIM_1] % DIM_2 == 0,
                  "The size of the second dim of the input tensor should be divisible by 2.", return false);
        return true;
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(CheckSwiGluQuant(launchParam), "Failed to check run info",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        auto &inTensor0 = launchParam.GetInTensor(DIM_0);
        outTensors[DIM_0].desc = launchParam.GetInTensor(DIM_0).desc;
        outTensors[DIM_0].desc.dtype = TENSOR_DTYPE_INT8;
        outTensors[DIM_0].desc.dims[DIM_1] = inTensor0.desc.dims[inTensor0.desc.dims.size() - 1] / DIM_2;
        outTensors[DIM_1].desc = launchParam.GetInTensor(0).desc;
        outTensors[DIM_1].desc.dims.resize(DIM_1);
        outTensors[DIM_1].desc.dims[DIM_0] = inTensor0.desc.dims[DIM_0];
        outTensors[DIM_1].desc.dtype = TENSOR_DTYPE_FLOAT;
        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        return GetKernelByName("SwiGluQuantKernel");
    }
};

REG_OPERATION(SwiGluQuantOperation);
} // namespace AtbOps 11