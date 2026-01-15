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
#include "asdops/params/params.h"
#include "kernels/utils/common.h"

namespace AsdOps {
using namespace Mki;
class FillOperation : public OperationBase {
public:
    explicit FillOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        auto param = AnyCast<OpParam::Fill>(launchParam.GetParam());
        auto dtype = launchParam.GetOutTensor(0).desc.dtype;
        if (param.withMask) {
            if (dtype == TENSOR_DTYPE_FLOAT16) {
                return GetKernelByName("MaskedFillF16Kernel");
            } else if (dtype == TENSOR_DTYPE_INT32) {
                return GetKernelByName("MaskedFillInt32Kernel");
            } else if (dtype == TENSOR_DTYPE_BF16) {
                return GetKernelByName("MaskedFillBF16Kernel");
            }
            MKI_LOG(ERROR) << "No kernel for MaskedFill dtype " << GetStrWithDType(dtype);
        } else {
            if (dtype == TENSOR_DTYPE_FLOAT16) {
                return GetKernelByName("FillF16Kernel");
            } else if (dtype == TENSOR_DTYPE_BF16) {
                return GetKernelByName("FillBF16Kernel");
            }
            MKI_LOG(ERROR) << "No kernel for Fill dtype " << GetStrWithDType(dtype);
        }
        return nullptr;
    }
    
    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::Fill), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::Fill>(specificParam);
        if (param.withMask) {
            return 2; // MaskedFill has 2 inputs
        }
        return 0; // Fill has 0 inputs
    }
    bool CheckIsTensorBroadcastable(const TensorDesc &input, const TensorDesc &mask) const
    {
        if (input.dims.size() < mask.dims.size()) {
            MKI_LOG(ERROR) << "inTensors should have same dimension";
            return false;
        }
        uint64_t lenDiff = input.dims.size() - mask.dims.size();
        for (uint64_t j = 0; j < mask.dims.size(); j++) {
            uint64_t i = j + lenDiff;
            if (input.dims[i] != mask.dims[j] && mask.dims[j] != static_cast<int64_t>(1)) {
                MKI_LOG(ERROR) << "inTensors' shapes are not broadcastable";
                return false;
            }
        }
        return true;
    }
protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Fill),
                  "OpParam is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        auto param = AnyCast<OpParam::Fill>(launchParam.GetParam());
        if (param.value.size() != 1) {
            MKI_LOG(ERROR) << "fillParam value size should be 1, actually: ", param.value.size();
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR);
        }
        if (param.withMask) {
            outTensors[0].desc = launchParam.GetInTensor(0).desc;
            MKI_CHECK(CheckIsTensorBroadcastable(launchParam.GetInTensor(0).desc, launchParam.GetInTensor(1).desc),
                      "OpParam outDim exists negetive number.", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        } else {
            MKI_CHECK(param.outTensorType == TENSOR_DTYPE_FLOAT16 || param.outTensorType == TENSOR_DTYPE_BF16,
                  "OpParam outTensorType is unsupport", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
            MKI_CHECK(!CheckParamAnyNegative(param.outDim), "OpParam outDim exists negetive number.",
                      return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
            outTensors[0].desc = {param.outTensorType, TENSOR_FORMAT_ND, param.outDim, {}, 0};
        }
        return Status::OkStatus();
    }
};
REG_OPERATION(FillOperation);
} //    namespace AsdOps