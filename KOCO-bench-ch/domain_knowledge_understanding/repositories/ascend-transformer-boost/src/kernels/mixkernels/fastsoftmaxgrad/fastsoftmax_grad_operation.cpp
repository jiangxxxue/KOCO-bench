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
constexpr int32_t MAX_SEQLEN = 4096;
using namespace Mki;
class FastSoftMaxGradOperation : public OperationBase {
public:
    explicit FastSoftMaxGradOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::FastSoftMaxGrad), "OpParam is invalid", return 0);
        return DIM_2;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::FastSoftMaxGrad), "OpParam is invalid", return 0);
        return DIM_1;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FastSoftMaxGrad),
            "OpParam is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
            
        auto param = AnyCast<OpParam::FastSoftMaxGrad>(launchParam.GetParam());
        MKI_LOG(INFO) << "infer shape param: " << param.headNum;
        MKI_CHECK(CheckUnpadFastSoftMaxGrad(launchParam), "Failed to check run info",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check run info"));
        auto &yInputTensor = launchParam.GetInTensor(DIM_0);
        outTensors[DIM_0].desc = yInputTensor.desc;
        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FastSoftMaxGrad),
            "OpParam is invalid", return nullptr);
        return GetKernelByName("FastSoftMaxGradKernel");
    }

private:
    bool CheckUnpadFastSoftMaxGrad(const LaunchParam &launchParam) const
    {
        auto param = AnyCast<OpParam::FastSoftMaxGrad>(launchParam.GetParam());
        MKI_CHECK(param.headNum > 0,
            "headNum is invalid for FastSoftMaxGradOperation.", return false);
        constexpr size_t maxSeqLenSize = 32;
        MKI_CHECK(param.qSeqLen.size() > 0,
            "qSeqLen list should not be empty!", return false);
        MKI_CHECK(param.qSeqLen.size() <= maxSeqLenSize,
            "qSeqLen list size should be less than 32!", return false);
        for (auto sampleSeqLen : param.qSeqLen) {
            if (sampleSeqLen <= 0 || sampleSeqLen > MAX_SEQLEN) {
                MKI_LOG(ERROR) << "Invalid qSeqLen: " << sampleSeqLen;
                return false;
            }
        }
        int64_t nSquareToken = 0;
        for (int32_t sampleSeqLen : param.qSeqLen) {
            nSquareToken += static_cast<int64_t>(sampleSeqLen) * sampleSeqLen;
        }
        MKI_CHECK(nSquareToken < std::numeric_limits<uint32_t>::max() / static_cast<uint32_t>(param.headNum),
            "nSquareToken * headNum is overflow for FastSoftMaxGradOperation.", return false);
        int64_t inDim = 0;
        inDim = nSquareToken * param.headNum;
        auto &yInputTensor = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(yInputTensor.desc.dtype == TENSOR_DTYPE_FLOAT16,
            "Input dim 0 (yInput) dtype invalid, should be float16", return false);
        MKI_CHECK(yInputTensor.desc.format == TENSOR_FORMAT_ND,
            "Input dim 0 (yInput) format invalid, should be ND", return false);
        MKI_CHECK(yInputTensor.desc.dims.size() == 1,
            "Input dim 0 (yInput) dims invalid", return false);
        MKI_CHECK(yInputTensor.desc.dims[DIM_0] == inDim,
            "Input dim 0 (yInput) dims invalid", return false);
        auto &yGradTensor = launchParam.GetInTensor(DIM_1);
        MKI_CHECK(yGradTensor.desc.dtype == TENSOR_DTYPE_FLOAT16,
            "Input dim 1 (yGrad) dtype invalid, should be float16", return false);
        MKI_CHECK(yGradTensor.desc.format == TENSOR_FORMAT_ND,
            "Input dim 1 (yGrad) format invalid, should be ND", return false);
        MKI_CHECK(yGradTensor.desc.dims.size() == 1,
            "Input dim 1 (yGrad) dims invalid", return false);
        MKI_CHECK(yGradTensor.desc.dims[DIM_0] == inDim,
            "Input dim 1 (yGrad) dims invalid", return false);
        return true;
    }
};

REG_OPERATION(FastSoftMaxGradOperation);
} // namespace AtbOps
