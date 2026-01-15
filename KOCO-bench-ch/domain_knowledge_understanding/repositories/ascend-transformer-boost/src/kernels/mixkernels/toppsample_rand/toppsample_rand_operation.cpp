/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <algorithm>
#include <mki_loader/op_register.h>
#include <mki/base/operation_base.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/log/log.h>
#include "atbops/params/params.h"

namespace AtbOps {
using namespace Mki;
class ToppsampleRandOperation : public OperationBase {
public:
    explicit ToppsampleRandOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        return GetKernelByName("ToppsampleRandKernel");
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::ToppsampleRand), "OpParam is invalid", return 0);
        return DIM_3; // 3 inputs
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::ToppsampleRand), "OpParam is invalid", return 0);
        return DIM_2;  // select_index, select_range
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetInTensor(DIM_0).desc.dims.size() == DIM_2, "dim size of inTensor0 is invalid",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_0).desc.dtype == TENSOR_DTYPE_FLOAT16
         || launchParam.GetInTensor(DIM_0).desc.dtype == TENSOR_DTYPE_BF16,
            "inTensor0 dtype invalid, should be float16 or bf16", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_0).desc.format == TENSOR_FORMAT_ND,
            "inTensor0 format invalid, should be ND", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_1).desc.dims.size() == DIM_2, "dim size of inTensor1 is invalid",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_1).desc.dims[DIM_1] == DIM_1, "inTensor1 dim[1] is invalid",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_1).desc.dtype == TENSOR_DTYPE_FLOAT16
         || launchParam.GetInTensor(DIM_1).desc.dtype == TENSOR_DTYPE_BF16,
            "inTensor1 dtype invalid, should be float16 or bf16", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_0).desc.dtype == launchParam.GetInTensor(DIM_1).desc.dtype,
            "inTensor1 dtype is inconsistent with inTensor0", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_1).desc.format == TENSOR_FORMAT_ND,
            "inTensor1 format invalid, should be ND", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_2).desc.dims.size() == DIM_2, "dim size of inTensor2 is invalid",
         return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_2).desc.dtype == TENSOR_DTYPE_FLOAT,
            "inTensor2 dtype invalid, should be float", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_2).desc.format == TENSOR_FORMAT_ND,
            "inTensor2 format invalid, should be ND", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(DIM_2).desc.dims[DIM_1] == DIM_1, "inTensor2 dim[1] is invalid",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        outTensors[DIM_0].desc = launchParam.GetInTensor(DIM_0).desc;
        outTensors[DIM_0].desc.dims[launchParam.GetInTensor(DIM_0).desc.dims.size() - 1] = 1; // num_samples
        outTensors[DIM_0].desc.dtype = TENSOR_DTYPE_INT32;

        outTensors[DIM_1].desc.dtype = TENSOR_DTYPE_INT32;
        outTensors[DIM_1].desc.format = launchParam.GetInTensor(DIM_0).desc.format;
        outTensors[DIM_1].desc.dims = { launchParam.GetInTensor(DIM_0).desc.dims[0] };

        return Status::OkStatus();
    }
};

REG_OPERATION(ToppsampleRandOperation);
} //    namespace AtbOps
