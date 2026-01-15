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
class FFNOperation : public OperationBase {
public:
    explicit FFNOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::FFN), "OpParam is invalid", return 0);
        return DIM_9;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::FFN), "OpParam is invalid", return 0);
        return DIM_1;
    }

    bool CheckTensor(const Tensor tensor, const uint32_t tensorIdx, const std::vector<TensorDType> dtypeList,
        const std::vector<TensorFormat> formatList, const std::vector<size_t> dimSizeList) const
    {
        bool condition = false;
        for (uint32_t i = 0; i < dtypeList.size(); i++) {
            if (tensor.desc.dtype == dtypeList[i]) {
                condition = true;
                break;
            }
        }
        MKI_CHECK(condition, "Input dim " << tensorIdx << "dtype invalid", return false);

        condition = false;
        for (uint32_t i = 0; i < formatList.size(); i++) {
            if (tensor.desc.format == formatList[i]) {
                condition = true;
                break;
            }
        }
        MKI_CHECK(condition, "Input dim " << tensorIdx << "format invalid", return false);

        condition = false;
        for (uint32_t i = 0; i < dimSizeList.size(); i++) {
            if (tensor.desc.dims.size() == dimSizeList[i]) {
                condition = true;
                break;
            }
        }
        MKI_CHECK(condition, "Input dim " << tensorIdx << "dims invalid", return false);
        return true;
    }

    bool CheckFFN(const LaunchParam &launchParam) const
    {
        auto &xTensor = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(CheckTensor(xTensor, DIM_0, {TENSOR_DTYPE_INT8}, {TENSOR_FORMAT_ND},
            {DIM_2}), "Check xTensor failed", return false);
        int64_t h = xTensor.desc.dims[DIM_1];
        MKI_CHECK(h > 0 && h < std::numeric_limits<uint32_t>::max(), "h should > 0 && < UINT32_MAX", return false);
 
        auto &weight1Tensor = launchParam.GetInTensor(DIM_1);
        MKI_CHECK(CheckTensor(weight1Tensor, DIM_1, {TENSOR_DTYPE_INT8}, {TENSOR_FORMAT_ND},
            {DIM_2}), "Check weight1Tensor failed", return false);
        MKI_CHECK(weight1Tensor.desc.dims[DIM_1] == h, "Input dim 1 (weight1) dims invalid", return false);
        int64_t n = weight1Tensor.desc.dims[DIM_0];
        MKI_CHECK(n > 0 && n < std::numeric_limits<uint32_t>::max(), "n should > 0 && < UINT32_MAX", return false);
 
        auto &weight2Tensor = launchParam.GetInTensor(DIM_2);
        MKI_CHECK(CheckTensor(weight2Tensor, DIM_2, {TENSOR_DTYPE_INT8}, {TENSOR_FORMAT_ND},
            {DIM_2}), "Check weight2Tensor failed", return false);
        MKI_CHECK(weight2Tensor.desc.dims[DIM_0] == h, "Input dim 2 (weight2) dims invalid", return false);
        MKI_CHECK(weight2Tensor.desc.dims[DIM_1] == n, "Input dim 2 (weight2) dims invalid", return false);
        
        auto &scaleTensor = launchParam.GetInTensor(DIM_3);
        MKI_CHECK(CheckTensor(scaleTensor, DIM_3, {TENSOR_DTYPE_FLOAT16}, {TENSOR_FORMAT_ND},
            {1}), "Check scaleTensor failed", return false);
        MKI_CHECK(scaleTensor.desc.dims[DIM_0] == n, "Input dim 3 (scale) dims invalid", return false);
 
        auto &deqScale1Tensor = launchParam.GetInTensor(DIM_4);
        MKI_CHECK(CheckTensor(deqScale1Tensor, DIM_4, {TENSOR_DTYPE_UINT64, TENSOR_DTYPE_INT64}, {TENSOR_FORMAT_ND},
            {1}), "Check deqScale1Tensor failed", return false);
        MKI_CHECK(deqScale1Tensor.desc.dims[DIM_0] == n, "Input dim 4 (deq_scale1) dims invalid", return false);
 
        auto &deqScale2Tensor = launchParam.GetInTensor(DIM_5);
        MKI_CHECK(CheckTensor(deqScale2Tensor, DIM_5, {TENSOR_DTYPE_UINT64, TENSOR_DTYPE_INT64}, {TENSOR_FORMAT_ND},
            {1}), "Check deqScale2Tensor failed", return false);
        MKI_CHECK(deqScale2Tensor.desc.dims[DIM_0] == h, "Input dim 5 (deq_scale2) dims invalid", return false);
 
        auto &biasTensor = launchParam.GetInTensor(DIM_6);
        MKI_CHECK(CheckTensor(biasTensor, DIM_6, {TENSOR_DTYPE_FLOAT16}, {TENSOR_FORMAT_ND},
            {1}), "Check biasTensor failed", return false);
        MKI_CHECK(biasTensor.desc.dims[DIM_0] == n, "Input dim 6 (bias) dims invalid", return false);
 
        auto &deqBias1Tensor = launchParam.GetInTensor(DIM_7);
        MKI_CHECK(CheckTensor(deqBias1Tensor, DIM_7, {TENSOR_DTYPE_INT32}, {TENSOR_FORMAT_ND},
            {1}), "Check deqBias1Tensor failed", return false);
        MKI_CHECK(deqBias1Tensor.desc.dims[DIM_0] == n, "Input dim 7 (deq_bias1) dims invalid", return false);
 
        auto &deqBias2Tensor = launchParam.GetInTensor(DIM_8);
        MKI_CHECK(CheckTensor(deqBias2Tensor, DIM_8, {TENSOR_DTYPE_INT32}, {TENSOR_FORMAT_ND},
            {1}), "Check deqBias2Tensor failed", return false);
        MKI_CHECK(deqBias2Tensor.desc.dims[DIM_0] == h, "Input dim 8 (deq_bias2) dims invalid", return false);
        
        return true;
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FFN),
            "OpParam is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
            
        auto param = AnyCast<OpParam::FFN>(launchParam.GetParam());
        MKI_LOG(INFO) << "activationType: " << param.activationType;
        MKI_CHECK(CheckFFN(launchParam), "Failed to check run info",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check run info"));
        auto &xTensor = launchParam.GetInTensor(DIM_0);
        outTensors[DIM_0].desc.dtype = TENSOR_DTYPE_FLOAT16;
        outTensors[DIM_0].desc.format = xTensor.desc.format;
        outTensors[DIM_0].desc.dims = xTensor.desc.dims;
        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FFN),
            "OpParam is invalid", return nullptr);
        return GetKernelByName("FFNKernel");
    }
};

REG_OPERATION(FFNOperation);
} // namespace AtbOps
