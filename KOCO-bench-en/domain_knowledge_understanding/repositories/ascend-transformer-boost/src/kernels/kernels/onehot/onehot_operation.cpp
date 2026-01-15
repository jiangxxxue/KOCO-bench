/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include <mki/utils/checktensor/check_tensor.h>
#include "asdops/params/onehot.h"

namespace AsdOps {
using namespace Mki;
class OnehotOperation : public OperationBase {
public:
    explicit OnehotOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        auto dtypeOut = launchParam.GetOutTensor(0).desc.dtype;
        if (dtypeOut == TENSOR_DTYPE_INT64) {
            return GetKernelByName("OnehotInt64Kernel");
        } else if (dtypeOut == TENSOR_DTYPE_INT32) {
            return GetKernelByName("OnehotInt32Kernel");
        }
        MKI_LOG(ERROR) << "No kernel for Onehot dtype " << dtypeOut;
        return nullptr;
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return 3; // onehot has 3 inputs
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return 1; // onehot has 1 output
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        SVector<int64_t> dimsIn = launchParam.GetInTensor(0).desc.dims;
        const Tensor &zeros = launchParam.GetInTensor(1);
        const Tensor &ones = launchParam.GetInTensor(2);
        if (!CheckEmptyTensor(zeros) && !CheckEmptyTensor(ones)) {
            if (zeros.Numel() != 1) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "Zeros shape is not [1]");
            }
            if (ones.Numel() != 1) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "Ones shape is not [1]");
            }
        } else {
            return Status::FailStatus(ERROR_INVALID_VALUE, "Ones and Zeros are empty tensor");
        }
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Onehot), "OpParam is invalid",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        OpParam::Onehot param = AnyCast<OpParam::Onehot>(launchParam.GetParam());
        int64_t axis = param.axis;
        SVector<int64_t> &depth = param.depth;
        MKI_CHECK(depth.size() >= 1, "OpParam depth is invalid",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam depth is invalid"));
        int64_t maxSize = static_cast<int64_t>(dimsIn.size() + 1);
        if (axis > maxSize || axis < -1 * maxSize) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "axis is out of range");
        }
        int32_t indicesIndex = 0;
        SVector<int64_t> newShape;
        if (axis >= 0) {
            for (int i = 0; i < maxSize; ++i) {
                if (i == axis) {
                    newShape.push_back(depth[0]);
                } else {
                    newShape.push_back(dimsIn[indicesIndex++]);
                }
            }
        } else {
            for (int i = 0; i < maxSize; ++i) {
                if (i == maxSize + axis) {
                    newShape.push_back(depth[0]);
                } else {
                    newShape.push_back(dimsIn[indicesIndex++]);
                }
            }
        }
        outTensors[0].desc= launchParam.GetInTensor(0).desc;
        outTensors[0].desc.dims = newShape;
        return Status::OkStatus();
    }
};
REG_OPERATION(OnehotOperation);
} //    namespace AsdOps