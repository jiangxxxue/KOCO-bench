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
class GatingOperation : public OperationBase {
public:
    explicit GatingOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::Gating), "OpParam is invalid", return 0);
        return DIM_2;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::Gating), "OpParam is invalid", return 0);
        return DIM_4;
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Gating),
            "OpParam is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        auto param = AnyCast<OpParam::Gating>(launchParam.GetParam());
        MKI_LOG(INFO) << "infer shape param: headSize:" << param.headSize << ", headNum:" << param.headNum
                      << ", cumSumInt64:" << param.cumSumInt64;
        if (launchParam.GetParam().Type() == typeid(OpParam::Gating)) {
            outTensors[0].desc = launchParam.GetInTensor(0).desc;
            outTensors[1].desc = launchParam.GetInTensor(0).desc;
            outTensors[1].desc.dtype = param.cumSumInt64 ? TENSOR_DTYPE_INT64 : TENSOR_DTYPE_INT32;
            outTensors[DIM_2].desc = launchParam.GetInTensor(0).desc;
            outTensors[DIM_3].desc = launchParam.GetInTensor(0).desc;
            if (param.deviceExpert.size() > 0) {
                outTensors[DIM_3].desc.dims[0] = 1;
                outTensors[1].desc.dims[0] = static_cast<int64_t>(param.deviceExpert.size());
            } else {
                outTensors[DIM_3].desc.dims[0] = 0;
                outTensors[1].desc.dims[0] = static_cast<int64_t>(param.headSize == 0 ? 1 : param.headSize);
            }
            return Status::OkStatus();
        } else {
            return Status::FailStatus(
                ERROR_ATTR_INVALID_TYPE,
                "Failed to check gating param, type of specificParam is not equals to OpParam::Gating");
        }
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Gating),
            "OpParam is invalid", return nullptr);
        return GetKernelByName("GatingKernel");
    }
};

REG_OPERATION(GatingOperation);
} // namespace AtbOps
