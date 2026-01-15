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
class LaserAttentionOperation : public OperationBase {
public:
    explicit LaserAttentionOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::LaserAttention), "OpParam is invalid", return 0);
        return DIM_6; // LaserAttention Op has 6 inputs: q, k, v, pseShift, dropMask, attenMask
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::LaserAttention), "OpParam is invalid", return 0);
        return DIM_3; // LaserAttention Op has 3 outputs: softmaxMax, softmaxSum, attenOut
    }

    bool CheckLaserAttention(const LaunchParam &launchParam) const
    {
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorQ.desc.dtype == TENSOR_DTYPE_BF16,
                  "Input Q dtype " << GetStrWithDType(tensorQ.desc.dtype) << " invalid, should be float16 or bfloat16.",
                  return false);

        auto &tensorK = launchParam.GetInTensor(DIM_1);
        auto &tensorV = launchParam.GetInTensor(DIM_2);
        MKI_CHECK((tensorK.desc.dtype == tensorQ.desc.dtype) && (tensorV.desc.dtype == tensorQ.desc.dtype),
                  "The dtype of Q, K, V should be the same.", return false);

        MKI_CHECK(tensorV.desc.dims.size() == tensorK.desc.dims.size(), "The dim of K, V should be the same.",
                  return false);

        return true;
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(CheckLaserAttention(launchParam), "Failed to check run info",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto &tensorV = launchParam.GetInTensor(DIM_2);
        auto param = AnyCast<OpParam::LaserAttention>(launchParam.GetParam());
        if (param.inputLayout == "BNSD") {
            outTensors[DIM_0].desc.dtype = TENSOR_DTYPE_FLOAT;
            outTensors[DIM_0].desc.format = TENSOR_FORMAT_ND;
            outTensors[DIM_0].desc.dims = {tensorQ.desc.dims[DIM_0], param.headNum, tensorQ.desc.dims[DIM_2]};

            outTensors[DIM_1].desc.dtype = TENSOR_DTYPE_FLOAT;
            outTensors[DIM_1].desc.format = TENSOR_FORMAT_ND;
            outTensors[DIM_1].desc.dims = {tensorQ.desc.dims[DIM_0], param.headNum, tensorQ.desc.dims[DIM_2]};
            outTensors[DIM_2].desc.dtype = TENSOR_DTYPE_BF16;
            outTensors[DIM_2].desc.format = TENSOR_FORMAT_ND;
            outTensors[DIM_2].desc.dims = {tensorQ.desc.dims[DIM_0], param.headNum, tensorQ.desc.dims[DIM_2],
                                           tensorV.desc.dims[DIM_3]};
        } else if (param.inputLayout == "SBH") {
            outTensors[DIM_0].desc.dtype = TENSOR_DTYPE_FLOAT;
            outTensors[DIM_0].desc.format = TENSOR_FORMAT_ND;
            outTensors[DIM_0].desc.dims = {tensorQ.desc.dims[DIM_1], param.headNum, tensorQ.desc.dims[DIM_0]};

            outTensors[DIM_1].desc.dtype = TENSOR_DTYPE_FLOAT;
            outTensors[DIM_1].desc.format = TENSOR_FORMAT_ND;
            outTensors[DIM_1].desc.dims = {tensorQ.desc.dims[DIM_1], param.headNum, tensorQ.desc.dims[DIM_0]};

            outTensors[DIM_2].desc.dtype = TENSOR_DTYPE_BF16;
            outTensors[DIM_2].desc.format = TENSOR_FORMAT_ND;
            outTensors[DIM_2].desc.dims = {tensorQ.desc.dims[DIM_0], tensorQ.desc.dims[DIM_1], 128 * param.headNum};
        }

        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::LaserAttention), "OpParam is invalid",
                  return nullptr);
        auto param = AnyCast<OpParam::LaserAttention>(launchParam.GetParam());

        return GetKernelByName("LaserAttentionKernel");
    }
};

REG_OPERATION(LaserAttentionOperation);
} // namespace AtbOps
