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
#include <mki/utils/const/op_const.h>
#include <mki_loader/op_register.h>
#include "atbops/params/params.h"

namespace AtbOps {
using namespace Mki;
class PadWithHiddenStateOperation : public OperationBase {
public:
    explicit PadWithHiddenStateOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::PadWithHiddenState), "OpParam is invalid", return 0);
        return DIM_1;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::PadWithHiddenState), "OpParam is invalid", return 0);
        return DIM_1;
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(CheckPadWithHiddenState(launchParam), "Failed to check run info",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        auto param = AnyCast<OpParam::PadWithHiddenState>(launchParam.GetParam());
        int64_t batchSize = static_cast<int64_t>(param.qSeqLen.size());
        int64_t hiddenSize = launchParam.GetInTensor(DIM_0).desc.dims[DIM_1];
        outTensors[DIM_0].desc.dtype = TENSOR_DTYPE_FLOAT16;
        outTensors[DIM_0].desc.format = TENSOR_FORMAT_ND;
        outTensors[DIM_0].desc.dims = { batchSize, param.maxSeqLen, hiddenSize };

        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        return GetKernelByName("PadWithHiddenStateKernel");
    }

private:
    bool CheckPadWithHiddenState(const LaunchParam &launchParam) const
    {
        auto param = AnyCast<OpParam::PadWithHiddenState>(launchParam.GetParam());
        int64_t outerSize = 0;
        for (auto sampleSeqLen : param.qSeqLen) {
            outerSize += sampleSeqLen;
        }
        auto &dataInputTensor = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(dataInputTensor.desc.dtype == TENSOR_DTYPE_FLOAT16,
            "Input dim 0 (dataInput) dtype invalid, should be float16", return false);
        MKI_CHECK(dataInputTensor.desc.format == TENSOR_FORMAT_ND,
            "Input dim 0 (dataInput) format invalid, should be ND", return false);
        MKI_CHECK(dataInputTensor.desc.dims.size() == 2,
            "Input dim 0 (dataInput) dims invalid", return false);
        MKI_CHECK(dataInputTensor.desc.dims[DIM_0] == outerSize,
            "Input dim 0 (dataInput) dims invalid", return false);
        return true;
    }
};

REG_OPERATION(PadWithHiddenStateOperation);
} // namespace AtbOps
