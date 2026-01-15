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
#include <mki/utils/log/log.h>
#include "asdops/params/slice.h"

namespace AsdOps {
using namespace Mki;
class SliceOperation : public OperationBase {
public:
    explicit SliceOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        auto dtype = launchParam.GetOutTensor(0).desc.dtype;
        if (dtype == TENSOR_DTYPE_FLOAT16 || dtype == TENSOR_DTYPE_BF16) {
            return GetKernelByName("SliceF16Int64Kernel");
        } else if (dtype == TENSOR_DTYPE_INT8 || dtype == TENSOR_DTYPE_UINT8 || dtype == TENSOR_DTYPE_BOOL) {
            return GetKernelByName("SliceInt8Int64Kernel");
        } else if (dtype == TENSOR_DTYPE_FLOAT || dtype == TENSOR_DTYPE_UINT32 || dtype == TENSOR_DTYPE_INT32) {
            return GetKernelByName("SliceInt32Int64Kernel");
        } else {
            return nullptr;
        }
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Slice), "slice: no match param type",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        OpParam::Slice param = AnyCast<OpParam::Slice>(launchParam.GetParam());
        outTensors[0].desc.dtype = launchParam.GetInTensor(0).desc.dtype;
        outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;

        Mki::SVector<int64_t> xDims = launchParam.GetInTensor(0).desc.dims;

        SVector<int64_t> offsets = param.offsets;
        SVector<int64_t> size = param.size;

        if (offsets.size() != xDims.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "Wrong input offsets");
        }
        if (size.size() != xDims.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "Wrong input size");
        }

        Mki::SVector<int64_t> outputDims;
        for (size_t i = 0; i < xDims.size(); ++i) {
            int64_t offsetValue = offsets[i];
            int64_t sizeValue = size[i];
            if (offsetValue < 0) {
                offsetValue = offsets[i] + xDims[i];
            }
            if (sizeValue == -1) {
                sizeValue = xDims[i] - offsetValue;
            } else if (sizeValue < -1) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "The value of size can not < -1");
            }
            if (offsetValue + sizeValue > xDims[i]) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "Wrong offsets or size");
            }
            outputDims.push_back(sizeValue);
        }
        outTensors[0].desc.dims = outputDims;
        return Status::OkStatus();
    }
};
REG_OPERATION(SliceOperation);
} // namespace AsdOps