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
class GmmAddOperation : public OperationBase {
public:
    explicit GmmAddOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override { return GetKernelByName("GmmAddKernel"); }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::GmmAdd), "OpParam is invalid.", return 0);
        return 4; // GmmAddOperation has 4 inputs.
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::GmmAdd), "OpParam is invalid.", return 0);
        return 1;
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == DIM_4, "invalid count of inTensor",
                  return Mki::Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        MKI_CHECK(launchParam.GetOutTensorCount() == DIM_1, "invalid count of outTensor",
                  return Mki::Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        uint32_t numGroup = static_cast<uint32_t>(launchParam.GetInTensor(2).desc.Numel());
        MKI_CHECK(numGroup > 0 && numGroup <= 128, "Invalid number of group.",
                  return Mki::Status::FailStatus(ERROR_INVALID_VALUE));
        auto inTensorDescY = launchParam.GetInTensor(DIM_3).desc;
        MKI_CHECK(inTensorDescY.dims.size() == DIM_2, "invalid shape of input y",
                  return Mki::Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        TensorDesc &tensorDescOut = outTensors[DIM_0].desc;
        tensorDescOut.dtype = TENSOR_DTYPE_FLOAT;
        tensorDescOut.format = TENSOR_FORMAT_ND;
        auto &outDims = tensorDescOut.dims;
        outDims.clear();
        outDims.emplace_back(inTensorDescY.dims[DIM_0]);
        outDims.emplace_back(inTensorDescY.dims[DIM_1]);
        return Status::OkStatus();
    }
};
REG_OPERATION(GmmAddOperation);
} //    namespace AtbOps
