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
constexpr int32_t MAX_BATCH_NUM = 64;
using namespace Mki;
class PadOperation : public OperationBase {
public:
    explicit PadOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::Pad), "OpParam is invalid", return 0);
        return DIM_4;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::Pad), "OpParam is invalid", return 0);
        return DIM_1;
    }

    bool CheckPad(const LaunchParam &launchParam) const
    {
        MKI_CHECK(launchParam.GetInTensor(DIM_0).desc.dims.size() == DIM_2,
            "dim size of inTensor0 is invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(DIM_3).desc.dims.size() == DIM_2,
            "dim size of inTensor3 is invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(DIM_0).desc.dims[DIM_0] == launchParam.GetInTensor(DIM_1).desc.dims[DIM_1],
            "input0 dim[0] and input1 dim[1] should be equal", return false);
        MKI_CHECK(launchParam.GetInTensor(DIM_1).desc.dims[DIM_0] == DIM_1
            && launchParam.GetInTensor(DIM_2).desc.dims[DIM_1] == DIM_1,
            "input1 dim[0] and input2 dim[1] should be equal", return false);
        MKI_CHECK(launchParam.GetInTensor(DIM_2).desc.dims[0] == launchParam.GetInTensor(DIM_3).desc.dims[0],
            "input_ids or seqlen is wrong", return false);
        MKI_CHECK(launchParam.GetInTensor(DIM_3).desc.dims[DIM_0] <= MAX_BATCH_NUM,
            "batch should be less than 64", return false);
        return true;
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
       MKI_CHECK(CheckPad(launchParam), "Failed to check launch param",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        outTensors[0].desc = launchParam.GetInTensor(DIM_0).desc; // out
        auto hiddenDim = launchParam.GetInTensor(DIM_0).desc.dims[1];
        outTensors[0].desc.dims[1] = hiddenDim;
        outTensors[0].desc.dims[0] = launchParam.GetInTensor(DIM_3).desc.dims[0];

        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        return GetKernelByName("PadKernel");
    }
};

REG_OPERATION(PadOperation);
} // namespace AtbOps
