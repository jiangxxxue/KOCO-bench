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
class RopeGradOperation : public OperationBase {
public:
    explicit RopeGradOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::RopeGrad), "OpParam is invalid", return 0);
        return 4; // RopeGrad Op has 4 inputs: q, k, cos, sin
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::RopeGrad), "OpParam is invalid", return 0);
        return 2; // RopeGrad Op has 2 outpts: rope_q, rope_k
    }

    bool CheckRopeGrad(const LaunchParam &launchParam) const
    {
        auto &inTensor0 = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(inTensor0.desc.dims.size() == 2, "dim size of inTensor0 is invalid", return false);
        auto &inTensor1 = launchParam.GetInTensor(DIM_1);
        MKI_CHECK(inTensor1.desc.dims.size() == 2, "dim size of inTensor1 is invalid", return false);
        MKI_CHECK(inTensor0.desc.dims[0] == inTensor1.desc.dims[0], "Shape of inTensor0/inTensor1 should be same",
                     return false);
        auto &cos = launchParam.GetInTensor(DIM_2);
        auto &sin = launchParam.GetInTensor(DIM_3);
        MKI_CHECK(cos.desc.dims.size() == sin.desc.dims.size(), "dim size of cos and sin is not equal",
                     return false);
        MKI_CHECK(cos.desc.dims[1] == 128, "headdim shoule be 128", return false);
        MKI_CHECK(cos.desc.dims[0] == sin.desc.dims[0] && cos.desc.dims[1] == sin.desc.dims[1],
                     "Shape of cos/sin should be same", return false);
        return true;
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(CheckRopeGrad(launchParam), "Failed to check run info",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        outTensors[DIM_0].desc = launchParam.GetInTensor(DIM_0).desc;
        outTensors[DIM_1].desc = launchParam.GetInTensor(DIM_1).desc;
        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        return GetKernelByName("RopeGradKernel");
    }
};

REG_OPERATION(RopeGradOperation);
} // namespace AtbOps
