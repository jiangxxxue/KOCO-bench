/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki_loader/op_register.h>
#include <mki/base/operation_base.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"

namespace AsdOps {
using namespace Mki;
class LogprobsSampleOperation : public OperationBase {
public:
    explicit LogprobsSampleOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        return GetKernelByName("LogprobsSampleKernel");
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::LogprobsSample), "OpParam is invalid", return 0);
        return DIM_3; // 3 inputs
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::LogprobsSample), "OpParam is invalid", return 0);
        return DIM_1; // 1 output
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetInTensor(DIM_0).desc.dims.size() == DIM_2, "dim size of inTensor0 is invalid",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_0).desc.dtype == TENSOR_DTYPE_FLOAT16
                      || launchParam.GetInTensor(DIM_0).desc.dtype == TENSOR_DTYPE_BF16,
                  "inTensor0 dtype invalid, should be float16 or bf16",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_0).desc.format == TENSOR_FORMAT_ND,
                  "inTensor0 format invalid, should be ND", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        MKI_CHECK(launchParam.GetInTensor(DIM_1).desc.dims.size() == DIM_2, "dim size of inTensor1 is invalid",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_1).desc.dtype == TENSOR_DTYPE_FLOAT16
                      || launchParam.GetInTensor(DIM_1).desc.dtype == TENSOR_DTYPE_BF16,
                  "inTensor1 dtype invalid, should be float16 or bf16",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_0).desc.dtype == launchParam.GetInTensor(DIM_1).desc.dtype,
                  "inTensor1 dtype is inconsistent with inTensor0", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_1).desc.format == TENSOR_FORMAT_ND,
                  "inTensor1 format invalid, should be ND", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        MKI_CHECK(launchParam.GetInTensor(DIM_2).desc.dims.size() == DIM_1, "dim size of inTensor3 is invalid",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_2).desc.dims[DIM_0] == launchParam.GetInTensor(DIM_0).desc.dims[DIM_0],
                  "inTensor3 dim[0] is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_2).desc.dtype == TENSOR_DTYPE_INT32,
                  "inTensor3 dtype invalid, should be int32",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::LogprobsSample), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        auto param = AnyCast<OpParam::LogprobsSample>(launchParam.GetParam());
        outTensors[DIM_0].desc.dims = { launchParam.GetInTensor(DIM_0).desc.dims[0], (int64_t)param.logprobsSize };
        outTensors[DIM_0].desc.dtype = TENSOR_DTYPE_FLOAT;
        return Status::OkStatus();
    }
};

REG_OPERATION(LogprobsSampleOperation);
} //    namespace AsdOps
