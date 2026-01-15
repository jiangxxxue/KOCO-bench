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

static constexpr uint32_t TENSOR_INPUT_NUM = 4;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 2;
static constexpr uint32_t EXPECT_DIM_X = 2;
static constexpr uint32_t EXPECT_DIM_ADD_NUM = 1;
static constexpr uint32_t MAX_TABLE_LIMIT = 128;

namespace AtbOps {
using namespace Mki;
class FusedAddTopkDivOperation : public OperationBase {
public:
    explicit FusedAddTopkDivOperation(const std::string &opName) noexcept : OperationBase(opName) {}
 
    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::FusedAddTopkDiv), "OpParam is invalid", return 0);
        return TENSOR_INPUT_NUM; // FusedAddTopkDiv Op has 4 inputs
    }
 
    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::FusedAddTopkDiv), "OpParam is invalid", return 0);
        return TENSOR_OUTPUT_NUM; // FusedAddTopkDiv Op has 2 outpts
    }
 
    bool CheckFusedAddTopkDivShape(const LaunchParam &launchParam) const
    {
        auto &inTensor0 = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(inTensor0.desc.dims.size() == EXPECT_DIM_X, "dim size of x is invalid", return false);
        auto &inTensor1 = launchParam.GetInTensor(DIM_1);
        MKI_CHECK(inTensor1.desc.dims.size() == EXPECT_DIM_ADD_NUM, "dim size of add_num is invalid", return false);
        MKI_CHECK(inTensor0.desc.dims[DIM_1] == inTensor1.desc.dims[DIM_0],
                  "Shape of x[1], add_num[0] should be same", return false);
        auto param = AnyCast<OpParam::FusedAddTopkDiv>(launchParam.GetParam());
        if (param.enableExpertMapping) {
            auto &inTensor2 = launchParam.GetInTensor(DIM_2);
            auto &inTensor3 = launchParam.GetInTensor(DIM_3);
            MKI_CHECK(inTensor2.desc.dims.size() == EXPECT_DIM_ADD_NUM,
                      "dim size of mapping_num is invalid", return false);
            MKI_CHECK(inTensor3.desc.dims.size() == EXPECT_DIM_X, "dim size of mapping_table is invalid", return false);
            MKI_CHECK(inTensor0.desc.dims[DIM_1] == inTensor2.desc.dims[DIM_0],
                    "Shape of x[1], mapping_num[0] should be same", return false);
            MKI_CHECK(inTensor0.desc.dims[DIM_1] == inTensor3.desc.dims[DIM_0],
                    "Shape of x[1], mapping_table[0] should be same", return false);
            MKI_CHECK(inTensor3.desc.dims[DIM_1] <= MAX_TABLE_LIMIT,
                    "mapping_table[0] should less or equal to 128", return false);
        }
        return true;
    }
 
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(CheckFusedAddTopkDivShape(launchParam), "Failed to check run info",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        auto param = AnyCast<OpParam::FusedAddTopkDiv>(launchParam.GetParam());
        MKI_CHECK(param.k <= launchParam.GetInTensor(DIM_0).desc.dims[DIM_1],
                  "k should be less or equal to the shape of x[1]", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        outTensors[DIM_0].desc = launchParam.GetInTensor(DIM_0).desc;
        outTensors[DIM_1].desc = launchParam.GetInTensor(DIM_0).desc;
        outTensors[DIM_0].desc.dims[DIM_1] = param.k;
        outTensors[DIM_1].desc.dims[DIM_1] = param.k;
        outTensors[DIM_0].desc.dtype = TENSOR_DTYPE_FLOAT;
        outTensors[DIM_1].desc.dtype = TENSOR_DTYPE_INT32;
 
        return Status::OkStatus();
    }
 
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        return GetKernelByName("FusedAddTopkDivKernel");
    }
};
 
REG_OPERATION(FusedAddTopkDivOperation);
} // namespace AtbOps