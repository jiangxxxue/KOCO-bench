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
#include "asdops/params/reduce.h"
#include "kernels/utils/common.h"

namespace AsdOps {
using namespace Mki;
class ReduceOperation : public OperationBase {
public:
    explicit ReduceOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        auto inDtype = launchParam.GetInTensor(0).desc.dtype;
        auto outDtype = launchParam.GetOutTensor(0).desc.dtype;
        OpParam::Reduce param = AnyCast<OpParam::Reduce>(launchParam.GetParam());
        switch (param.reduceType) {
            case OpParam::Reduce::REDUCE_MAX:
                if (inDtype == TENSOR_DTYPE_INT32 && outDtype == TENSOR_DTYPE_INT32) {
                    return GetKernelByName("ReduceMaxInt32Kernel");
                } else if (inDtype == TENSOR_DTYPE_BF16 && outDtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("ReduceMaxBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for REDUCE_MAX inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Reduce::REDUCE_MIN:
                if (inDtype == TENSOR_DTYPE_INT32 && outDtype == TENSOR_DTYPE_INT32) {
                    return GetKernelByName("ReduceMinInt32Kernel");
                } else if (inDtype == TENSOR_DTYPE_BF16 && outDtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("ReduceMinBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for REDUCE_MIN inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Reduce::REDUCE_SUM:
                if (inDtype == TENSOR_DTYPE_FLOAT16 && outDtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("ReduceSumF16Kernel");
                } else if (inDtype == TENSOR_DTYPE_BF16 && outDtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("ReduceSumBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for REDUCE_SUM inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            default:
                MKI_LOG(ERROR) << "No kernel for Reduce inDtype " << GetStrWithDType(inDtype);
                return nullptr;
        }
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        SVector<int64_t> dimsIn = launchParam.GetInTensor(0).desc.dims;

        OpParam::Reduce param = AnyCast<OpParam::Reduce>(launchParam.GetParam());
        SVector<int64_t> axis = param.axis;

        if (axis.size() < 1 || axis.size() > dimsIn.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "axis size is not support");
        }

        SVector<int64_t> dimOut = dimsIn;
        std::sort(axis.begin(), axis.end());

        MKI_CHECK(!CheckParamAnyNegative(axis),
            "OpParam axis exists negetive number.",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam axis is invalid"));

        if (axis[axis.size() - 1] >= static_cast<int64_t>(dimsIn.size())) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "value of axis is out of range");
        }

        for (uint64_t i = axis.size() - 1; i > 0; i--) {
            if (axis[i] == axis[i - 1]) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "axis is specified more than once");
            }
            dimOut.erase(dimOut.begin() + axis[i]);
        }
        dimOut.erase(dimOut.begin() + axis[0]);

        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        outTensors[0].desc.dims = dimOut;
        return Status::OkStatus();
    }
};
REG_OPERATION(ReduceOperation);
} //    namespace AsdOps
