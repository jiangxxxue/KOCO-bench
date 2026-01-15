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

static constexpr uint32_t ELE_NUM_FP16 = 16;

namespace AtbOps {
using namespace Mki;
class RopeOperation : public OperationBase {
public:
    explicit RopeOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::Rope), "OpParam is invalid", return 0);
        return DIM_5; // Rope Op has 5 inputs: q, k, cos, sin, seqlen
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::Rope), "OpParam is invalid", return 0);
        return DIM_2; // Rope Op has 2 outputs: rope_q, rope_k
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_LOG(INFO) << "RopeInferShape enter";
        const SVector<int64_t> &inputQ = launchParam.GetInTensor(DIM_0).desc.dims;
        const SVector<int64_t> &inputK = launchParam.GetInTensor(DIM_1).desc.dims;
        const SVector<int64_t> &inputCos = launchParam.GetInTensor(DIM_2).desc.dims;
        const SVector<int64_t> &inputSin = launchParam.GetInTensor(DIM_3).desc.dims;
        if (inputQ.size() != inputK.size() || inputQ.size() == 0) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "dim size of inputQ is wrong");
        }
        for (size_t i = 0; i < inputQ.size() - 1; i++) {
            if (inputQ[i] != inputK[i]) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "inputQ are not equal to inputK");
            }
        }
        if (inputCos != inputSin) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "inputCos are not equal to inputSin");
        }
        if (inputQ[inputQ.size()-1] % ELE_NUM_FP16 != 0 || inputK[inputK.size()-1] % ELE_NUM_FP16 != 0) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "the shapes of inputQ and inputK must be 32 bytes aligned.");
        }
        outTensors[DIM_0].desc = launchParam.GetInTensor(DIM_0).desc;
        outTensors[DIM_1].desc = launchParam.GetInTensor(DIM_1).desc;
 
        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Rope), "OpParam is invalid", return nullptr);
        return GetKernelByName("AtbRopeKernel");
    }
};

REG_OPERATION(RopeOperation);
} // namespace AtbOps
