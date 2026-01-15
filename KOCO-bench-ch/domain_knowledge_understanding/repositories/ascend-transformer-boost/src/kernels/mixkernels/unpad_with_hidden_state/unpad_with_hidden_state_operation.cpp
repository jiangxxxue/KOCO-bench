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
#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/checktensor/check_tensor.h>
#include "atbops/params/params.h"

namespace AtbOps {
using namespace Mki;
class UnpadWithHiddenStateOperation : public OperationBase {
public:
    explicit UnpadWithHiddenStateOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        return GetKernelByName("UnpadWithHiddenStateKernel");
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::UnpadWithHiddenState), "OpParam is invalid", return 0);
        return DIM_1; // 1 inputs
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::UnpadWithHiddenState), "OpParam is invalid", return 0);
        return DIM_1; // 1 output
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(CheckUnpadWithHiddenState(launchParam), "Failed to check run info",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        auto param = AnyCast<OpParam::UnpadWithHiddenState>(launchParam.GetParam());
        int64_t outerSize = 0;
        for (auto sampleSeqLen : param.qSeqLen) {
            outerSize += sampleSeqLen;
        }
        int64_t hiddenSize = launchParam.GetInTensor(DIM_0).desc.dims[DIM_2];
        outTensors[DIM_0].desc.dtype = TENSOR_DTYPE_FLOAT16;
        outTensors[DIM_0].desc.format = TENSOR_FORMAT_ND;
        outTensors[DIM_0].desc.dims = { outerSize, hiddenSize };

        return Status::OkStatus();
    }
private:
    bool CheckUnpadWithHiddenState(const LaunchParam &launchParam) const
    {
        auto param = AnyCast<OpParam::UnpadWithHiddenState>(launchParam.GetParam());
        auto &dataInputTensor = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(dataInputTensor.desc.dtype == TENSOR_DTYPE_FLOAT16,
            "Input dim 0 (dataInput) dtype invalid, should be float16", return false);
        MKI_CHECK(dataInputTensor.desc.format == TENSOR_FORMAT_ND,
            "Input dim 0 (dataInput) format invalid, should be ND", return false);
        MKI_CHECK(dataInputTensor.desc.dims.size() == 3,
            "Input dim 0 (dataInput) dims invalid", return false);
        MKI_CHECK(dataInputTensor.desc.dims[DIM_0] == static_cast<int64_t>(param.qSeqLen.size()),
            "Input dim 0 (dataInput) dims invalid", return false);
        MKI_CHECK(dataInputTensor.desc.dims[DIM_1] == param.maxSeqLen,
            "Input dim 0 (dataInput) dims invalid", return false);
        return true;
    }
};

REG_OPERATION(UnpadWithHiddenStateOperation);
} //    namespace AtbOps
