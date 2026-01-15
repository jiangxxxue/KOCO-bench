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
class GenAttentionMaskOperation : public OperationBase {
public:
    explicit GenAttentionMaskOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::GenAttentionMask), "OpParam is invalid", return 0);
        return DIM_1;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::GenAttentionMask), "OpParam is invalid", return 0);
        return DIM_1;
    }

    bool CheckGenAttentionMask(const LaunchParam &launchParam) const
    {
        auto &dataInputTensor = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(dataInputTensor.desc.dtype == TENSOR_DTYPE_FLOAT16,
            "Input dim 0 (dataInput) dtype invalid, should be float16", return false);

        return true;
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::GenAttentionMask),
            "OpParam is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
            
        auto opParam = AnyCast<OpParam::GenAttentionMask>(launchParam.GetParam());
        MKI_LOG(INFO) << "infer shape param: " << opParam.headNum;
        MKI_CHECK(CheckGenAttentionMask(launchParam), "Failed to check launch param",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        int64_t outdims = 0;
        for (int i = 0; i < launchParam.GetInTensor(0).desc.dims[0]; i++) {
            outdims += static_cast<int64_t>(opParam.headNum) * opParam.qSeqLen[i] * opParam.qSeqLen[i];
        }
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        outTensors[0].desc.dims = { outdims };

        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::GenAttentionMask),
            "OpParam is invalid", return nullptr);
        return GetKernelByName("GenAttentionMaskKernel");
    }
};

REG_OPERATION(GenAttentionMaskOperation);
} // namespace AtbOps
