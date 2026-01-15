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
#include <mki_loader/op_register.h>

#include "asdops/common.h"
#include "asdops/params/params.h"

namespace AsdOps {
using namespace Mki;
class GatherOperation : public OperationBase {
public:
    explicit GatherOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        auto dtype = launchParam.GetInTensor(0).desc.dtype;
        auto dtype1 = launchParam.GetInTensor(1).desc.dtype;
        if (dtype == TENSOR_DTYPE_FLOAT16 || dtype == TENSOR_DTYPE_BF16) {
            if (dtype1 == TENSOR_DTYPE_UINT32 || dtype1 == TENSOR_DTYPE_INT32) {
                return GetKernelByName("Gather16I32Kernel");
            } else if (dtype1 == TENSOR_DTYPE_INT64) {
                return GetKernelByName("Gather16I64Kernel");
            }
        } else if (dtype == TENSOR_DTYPE_FLOAT || dtype == TENSOR_DTYPE_INT32 || dtype == TENSOR_DTYPE_UINT32) {
            if (dtype1 == TENSOR_DTYPE_UINT32 || dtype1 == TENSOR_DTYPE_INT32) {
                return GetKernelByName("Gather32I32Kernel");
            } else if (dtype1 == TENSOR_DTYPE_INT64) {
                return GetKernelByName("Gather32I64Kernel");
            }
        } else if (dtype == TENSOR_DTYPE_INT64) {
            if (dtype1 == TENSOR_DTYPE_UINT32 || dtype1 == TENSOR_DTYPE_INT32) {
                return GetKernelByName("Gather64I32Kernel");
            } else if (dtype1 == TENSOR_DTYPE_INT64) {
                return GetKernelByName("Gather64I64Kernel");
            }
        }
        MKI_LOG(ERROR) << "No kernel for Gather dtype " << GetStrWithDType(dtype) << ", " << GetStrWithDType(dtype1);
        return nullptr;
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return 2; // gather has 2 inputs
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return 1; // gather has 1 output
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Gather), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        OpParam::Gather param = AnyCast<OpParam::Gather>(launchParam.GetParam());
        const SVector<int64_t> &axisVector = param.axis;
        // only support one axis
        MKI_CHECK(axisVector.size() == 1 && axisVector[0] >= 0, "multi axis is not support now",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(axisVector[0] >= param.batchDims, "batch_dims must be less than or equal to axis",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(param.batchDims >= 0, "batchDims is not support now", return Status::FailStatus(ERROR_INVALID_VALUE));

        size_t axis = static_cast<size_t>(axisVector[0]);
        outTensors[0].desc.dtype = launchParam.GetInTensor(0).desc.dtype;
        outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;

        SVector<int64_t> tensorDims = launchParam.GetInTensor(0).desc.dims;
        SVector<int64_t> indicesDims = launchParam.GetInTensor(1).desc.dims;
        SVector<int64_t> outTensorDims;
        uint64_t dim1 = static_cast<uint64_t>(axisVector[0]);
        uint64_t dim2 = indicesDims.size() - static_cast<uint64_t>(param.batchDims);
        uint64_t dim3 = tensorDims.size() - dim1 - 1;
        uint64_t outTensordimNum = dim1 + dim2 + dim3;
        MKI_CHECK(outTensordimNum <= DIM_8, "outTensorDescs dimNum is out of range",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(static_cast<uint64_t>(axisVector[0]) < tensorDims.size(),
                  "param.axis should  < inTensorDescs(0) dimNum", return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(static_cast<uint64_t>(param.batchDims) <= indicesDims.size(),
                  "param.batchDims should  <= inTensorDescs(1) dimNum", return Status::FailStatus(ERROR_INVALID_VALUE));

        for (size_t i = 0; i < axis && i < tensorDims.size(); i++) {
            outTensorDims.push_back(tensorDims[i]);
        }
        for (size_t i = static_cast<size_t>(param.batchDims); i < indicesDims.size(); ++i) {
            outTensorDims.push_back(indicesDims[i]);
        }
        for (size_t i = axis + 1; i < tensorDims.size(); ++i) {
            outTensorDims.push_back(tensorDims[i]);
        }

        outTensors[0].desc.dims = outTensorDims;
        return Status::OkStatus();
    }
};
REG_OPERATION(GatherOperation);
} //    namespace AsdOps