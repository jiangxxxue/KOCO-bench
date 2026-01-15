/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

constexpr uint32_t TENSOR_INPUT_NUM = 4;
constexpr uint32_t TENSOR_OUTPUT_NUM = 1;
constexpr uint32_t TENSOR_DIMS_SIZE_2 = 2;
constexpr uint32_t TENSOR_DIMS_SIZE_3 = 3;
constexpr uint32_t CONST_ZERO = 0;
constexpr uint32_t HEAD_DIM_CONVENTION_NUM = 16;
constexpr uint32_t HEAD_DIM_MAX = 64;

namespace AtbOps {
using namespace Mki;
class RopeQConcatOperation : public OperationBase {
public:
    explicit RopeQConcatOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::RopeQConcat), "OpParam is invalid", return 0);
        return TENSOR_INPUT_NUM; // RopeQConcat Op has 4 inputs: q, k, cos, sin
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::RopeQConcat), "OpParam is invalid", return 0);
        return TENSOR_OUTPUT_NUM; // RopeQConcat Op has 1 outpts: ropeQConcat
    }

    bool CheckRopeQConcat(const LaunchParam &launchParam) const
    {
        auto &inTensor0 = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(inTensor0.desc.dims.size() == TENSOR_DIMS_SIZE_2, "dim size of inTensor0 is invalid", return false);
        auto &inTensor1 = launchParam.GetInTensor(DIM_1);
        MKI_CHECK(inTensor1.desc.dims.size() == TENSOR_DIMS_SIZE_2, "dim size of inTensor1 is invalid", return false);
        auto &inTensor2 = launchParam.GetInTensor(DIM_2);
        MKI_CHECK(inTensor2.desc.dims.size() == TENSOR_DIMS_SIZE_2, "dim size of inTensor2 is invalid", return false);
        auto &inTensor3 = launchParam.GetInTensor(DIM_3);
        MKI_CHECK(inTensor3.desc.dims.size() == TENSOR_DIMS_SIZE_3, "dim size of inTensor3 is invalid", return false);
        MKI_CHECK(inTensor1.desc.dims[DIM_0] == inTensor2.desc.dims[DIM_0] &&
                      inTensor1.desc.dims[DIM_1] == inTensor2.desc.dims[DIM_1],
                  "Shape of cos/sin should be same", return false);
        for (size_t i = 0; i < TENSOR_INPUT_NUM; i++) {
            auto inTensor = launchParam.GetInTensor(i);
            size_t dimNum = inTensor.desc.dims.size();
            for (size_t dimIndex = DIM_0; dimIndex < dimNum; dimIndex++) {
                MKI_CHECK(inTensor.desc.dims[dimIndex] != CONST_ZERO,
                          "Any dimension of the input vector cannot be equal to zero.", return false);
            }
        }
        size_t headDim = inTensor1.desc.dims[DIM_1];
        MKI_CHECK(headDim % HEAD_DIM_CONVENTION_NUM == CONST_ZERO && headDim <= HEAD_DIM_MAX,
                  "headDim is invalid. It must be a multiple of 16 and less than or equal to 64.", return false);
        size_t concatSize = inTensor3.desc.dims[DIM_2];
        MKI_CHECK(concatSize % HEAD_DIM_CONVENTION_NUM == CONST_ZERO,
                  "concatSize is invalid. It must be a multiple of 16.", return false);
        return true;
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(CheckRopeQConcat(launchParam), "Failed to check run info",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        auto &inTensor1 = launchParam.GetInTensor(DIM_1);
        auto &inTensor3 = launchParam.GetInTensor(DIM_3);
        outTensors[DIM_0].desc = launchParam.GetInTensor(DIM_3).desc;
        outTensors[DIM_0].desc.dims[DIM_2] = inTensor1.desc.dims[DIM_1] + inTensor3.desc.dims[DIM_2];

        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        return GetKernelByName("RopeQConcatKernel");
    }
};

REG_OPERATION(RopeQConcatOperation);
} // namespace AtbOps
