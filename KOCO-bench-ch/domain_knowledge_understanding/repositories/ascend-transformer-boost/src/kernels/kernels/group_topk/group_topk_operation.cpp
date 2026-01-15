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
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"

constexpr int64_t IN_TENSOR_NUM = 2;
constexpr int64_t OUT_TENSOR_NUM = 1;

namespace AsdOps {
using namespace Mki;
class GroupTopkOperation : public OperationBase {
public:
    explicit GroupTopkOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Fail to check consistent", return nullptr);
        auto inDtype0 = launchParam.GetInTensor(0).desc.dtype;
        auto inDtype1 = launchParam.GetInTensor(1).desc.dtype;
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::GroupTopk), "OpParam is invalid", return nullptr);
        if ((inDtype0 == TENSOR_DTYPE_FLOAT16 || inDtype0 == TENSOR_DTYPE_BF16) && inDtype1 == TENSOR_DTYPE_INT32) {
            MKI_LOG(INFO) << "get GroupTopkKernel";
            return GetKernelByName("GroupTopkKernel");
        } else {
            MKI_LOG(ERROR) << "GroupTopk op expected dtype is error";
            return nullptr;
        }
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return IN_TENSOR_NUM;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return OUT_TENSOR_NUM;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::GroupTopk), "no match param type",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        MKI_LOG(INFO) << "GroupTopk InferShapeImpl launchParam.GetInTensor(0).desc="
                      << launchParam.GetInTensor(0).desc.ToString();
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        return Status::OkStatus();
    }
};
REG_OPERATION(GroupTopkOperation);
} // namespace AsdOps