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
class StridedBatchMatmulOperation : public OperationBase {
public:
    explicit StridedBatchMatmulOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        return GetKernelByName("StridedBatchMatmulKernel");
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::StridedBatchMatmul), "OpParam is invalid", return 0);
        return 2; // StridedBatchMatmul Op has 2 inputs: MatA, MatB
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::StridedBatchMatmul), "OpParam is invalid", return 0);
        return 1; // StridedBatchMatmul Op has 1 output: MatC
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        auto opParam = AnyCast<OpParam::StridedBatchMatmul>(launchParam.GetParam());
        int64_t outdims = 0;
        for (int i = 0; i < opParam.batch; i++) {
            outdims += opParam.m[i] * opParam.n[i];
        }
        outdims = opParam.headNum * outdims;
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        outTensors[0].desc.dims = { outdims };
        return Status::OkStatus();
    }
};
REG_OPERATION(StridedBatchMatmulOperation);
} //    namespace AtbOps
