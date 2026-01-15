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
class LaserAttentionGradOperation : public OperationBase {
public:
    explicit LaserAttentionGradOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::LaserAttentionGrad), "OpParam is invalid", return 0);
        return DIM_10; // LaserAttentionGrad Op has 10 inputs: q, k, v, attenOutGrad, pseShift, dropMask, attenMask,
                       // softmaxMax, softmaxSum, attenIn
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::LaserAttentionGrad), "OpParam is invalid", return 0);
        return DIM_3; // LaserAttentionGrad Op has 3 outputs: dQ, dK, dV
    }

    bool CheckLaserAttentionGrad(const LaunchParam &launchParam) const
    {
        // Q.shape = [batch, qNumHead, seqSize, headDim]
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorQ.desc.dtype == TENSOR_DTYPE_BF16,
                  "Input Q dtype " << GetStrWithDType(tensorQ.desc.dtype) << " invalid, should be float16 or bfloat16.",
                  return false);

        auto &tensorGrad = launchParam.GetInTensor(DIM_3);
        auto &tensorO = launchParam.GetInTensor(DIM_9);
        MKI_CHECK((tensorGrad.desc.dtype == tensorQ.desc.dtype) && (tensorO.desc.dtype == tensorQ.desc.dtype),
                  "The dtype of Q, attenOutGrad, attenIn should be the same.", return false);
        MKI_CHECK((tensorGrad.desc.dims.size() == tensorQ.desc.dims.size()) &&
                      (tensorO.desc.dims.size() == tensorQ.desc.dims.size()),
                  "The shape of Q, attenOutGrad, attenIn should be the same.", return false);

        // softmaxLogMaxSum.shape = [batch, qNumHead, seqSize]
        auto &tensorM = launchParam.GetInTensor(DIM_7);
        auto &tensorS = launchParam.GetInTensor(DIM_8);
        MKI_CHECK(tensorM.desc.dtype == TENSOR_DTYPE_FLOAT,
                  "Input SoftmaxMax dtype " << GetStrWithDType(tensorM.desc.dtype) << " invalid, should be float32.",
                  return false);
        MKI_CHECK(tensorS.desc.dtype == TENSOR_DTYPE_FLOAT,
                  "Input SoftmaxSum dtype " << GetStrWithDType(tensorS.desc.dtype) << " invalid, should be float32.",
                  return false);

        MKI_CHECK(tensorS.desc.dims.size() == tensorM.desc.dims.size(),
                  "The shape of SoftmaxMax, SoftmaxSum should be the same.", return false);

        return true;
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(CheckLaserAttentionGrad(launchParam), "Failed to check run info",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        auto &query = launchParam.GetInTensor(DIM_0);
        auto &key = launchParam.GetInTensor(DIM_1);
        auto &value = launchParam.GetInTensor(DIM_2);
        auto param = AnyCast<OpParam::LaserAttentionGrad>(launchParam.GetParam());
        if (param.inputLayout == "BNSD") {
            outTensors[DIM_0].desc.dtype = TENSOR_DTYPE_BF16;
            outTensors[DIM_0].desc.format = TENSOR_FORMAT_ND;
            outTensors[DIM_0].desc.dims = {query.desc.dims[DIM_0], query.desc.dims[DIM_1], query.desc.dims[DIM_2],
                                           query.desc.dims[DIM_3]};

            outTensors[DIM_1].desc.dtype = TENSOR_DTYPE_BF16;
            outTensors[DIM_1].desc.format = TENSOR_FORMAT_ND;
            outTensors[DIM_1].desc.dims = {key.desc.dims[DIM_0], key.desc.dims[DIM_1], key.desc.dims[DIM_2],
                                           key.desc.dims[DIM_3]};

            outTensors[DIM_2].desc.dtype = TENSOR_DTYPE_BF16;
            outTensors[DIM_2].desc.format = TENSOR_FORMAT_ND;
            outTensors[DIM_2].desc.dims = {value.desc.dims[DIM_0], value.desc.dims[DIM_1], value.desc.dims[DIM_2],
                                           value.desc.dims[DIM_3]};
        } else if (param.inputLayout == "SBH") {
            outTensors[DIM_0].desc.dtype = TENSOR_DTYPE_BF16;
            outTensors[DIM_0].desc.format = TENSOR_FORMAT_ND;
            outTensors[DIM_0].desc.dims = {query.desc.dims[DIM_0], query.desc.dims[DIM_1], query.desc.dims[DIM_2]};

            outTensors[DIM_1].desc.dtype = TENSOR_DTYPE_BF16;
            outTensors[DIM_1].desc.format = TENSOR_FORMAT_ND;
            outTensors[DIM_1].desc.dims = {key.desc.dims[DIM_0], key.desc.dims[DIM_1], key.desc.dims[DIM_2]};

            outTensors[DIM_2].desc.dtype = TENSOR_DTYPE_BF16;
            outTensors[DIM_2].desc.format = TENSOR_FORMAT_ND;
            outTensors[DIM_2].desc.dims = {value.desc.dims[DIM_0], value.desc.dims[DIM_1], value.desc.dims[DIM_2]};
        }

        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::LaserAttentionGrad), "OpParam is invalid",
                  return nullptr);
        MKI_LOG(INFO) << "find laser attention grad kernel \n";
        auto param = AnyCast<OpParam::LaserAttentionGrad>(launchParam.GetParam());
        return GetKernelByName("LaserAttentionGradKernel");
    }
};

REG_OPERATION(LaserAttentionGradOperation);
} // namespace AtbOps
