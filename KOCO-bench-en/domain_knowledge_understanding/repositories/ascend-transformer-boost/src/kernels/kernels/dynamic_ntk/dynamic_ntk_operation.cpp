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
#include <mki/utils/assert/assert.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/log/log.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/common.h"
#include "asdops/params/dynamic_ntk.h"

namespace {
constexpr size_t DYNAMIC_NTK_TENSOR_IN_COUNT = 3;
constexpr size_t DYNAMIC_NTK_TENSOR_OUT_COUNT = 2;
constexpr size_t INPUT_SEQ_LEN_IDX = 2;
constexpr uint32_t SHIFT_AMOUNT = 2;
constexpr uint32_t MAX_HEAD_SIZE = 2048;
}

namespace AsdOps {
using namespace Mki;
class DynamicNTKOperation : public OperationBase {
public:
    explicit DynamicNTKOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        return GetKernelByName("DynamicNTKKernel");
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return DYNAMIC_NTK_TENSOR_IN_COUNT;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return DYNAMIC_NTK_TENSOR_OUT_COUNT;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::DynamicNTK), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 1,
                  "input position_ids dim size is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(1).desc.dims.size() == Mki::DIM_2,
                  "input inv_freqs dim size is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensor(INPUT_SEQ_LEN_IDX).desc.dims.size() == 1,
                  "input seq_lens dim size is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        return DynamicNTKInferShape(launchParam, outTensors);
    }

private:
    Status DynamicNTKInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        OpParam::DynamicNTK param = AnyCast<OpParam::DynamicNTK>(launchParam.GetParam());
        auto freqTensor = launchParam.GetInTensor(1);
        auto freqDims = freqTensor.desc.dims;
        MKI_CHECK(freqDims[1] <= MAX_HEAD_SIZE / 2, "input invFreqs dim is invalid",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        int64_t headDim = freqDims[1] * SHIFT_AMOUNT;
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        outTensors[0].desc.dims.emplace_back(headDim);
        outTensors[0].desc.dtype = param.outType == 0 ? TENSOR_DTYPE_FLOAT16
            : (param.outType == 1 ? TENSOR_DTYPE_BF16 : TENSOR_DTYPE_FLOAT);
        outTensors[1].desc = outTensors[0].desc;
        return Status::OkStatus();
    }
};
REG_OPERATION(DynamicNTKOperation);
} //    namespace AsdOps
