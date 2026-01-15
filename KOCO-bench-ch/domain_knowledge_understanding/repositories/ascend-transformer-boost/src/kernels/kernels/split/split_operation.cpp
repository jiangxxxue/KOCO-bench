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
#include <mki/types.h>
#include <mki/utils/log/log.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/split.h"

namespace AsdOps {
using namespace Mki;
static const int SPLIT_SUPPORT_SPLIT_NUM2 = 2;
static const int SPLIT_SUPPORT_SPLIT_NUM3 = 3;
static const int MAX_SUPPORT_SPLIT_NUM = 3;
class SplitOperation : public OperationBase {
public:
    explicit SplitOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        auto dtype = launchParam.GetOutTensor(0).desc.dtype;
        auto param = AnyCast<OpParam::Split>(launchParam.GetParam());
        size_t outputNum = launchParam.GetOutTensorCount();
        if (dtype == TENSOR_DTYPE_FLOAT16 || dtype == TENSOR_DTYPE_BF16) {
            if (outputNum == SPLIT_SUPPORT_SPLIT_NUM2) {
                if (param.splitSize.size() > 0) {
                    return GetKernelByName("SplitVF16Output2Kernel");
                } else {
                    return GetKernelByName("SplitF16Output2Kernel");
                }
            } else if (outputNum == SPLIT_SUPPORT_SPLIT_NUM3) {
                if (param.splitSize.size() > 0) {
                    return GetKernelByName("SplitVF16Output3Kernel");
                } else {
                    return GetKernelByName("SplitF16Output3Kernel");
                }
            } else {
                MKI_LOG(ERROR) << "outputNum is wrong:" << outputNum;
                return nullptr;
            }
        } else if (dtype == TENSOR_DTYPE_INT64) {
            if (outputNum == SPLIT_SUPPORT_SPLIT_NUM2) {
                if (param.splitSize.size() > 0) {
                    return GetKernelByName("SplitVInt64Output2Kernel");
                } else {
                    return GetKernelByName("SplitInt64Output2Kernel");
                }
            } else {
                MKI_LOG(ERROR) << "outputNum is wrong:" << outputNum;
                return nullptr;
            }
        } else {
            MKI_LOG(ERROR) << "tensor data type is wrong:" << dtype;
            return nullptr;
        }
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::Split), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::Split>(specificParam);
        if (param.splitNum == SPLIT_SUPPORT_SPLIT_NUM2) {
            return 2; // split has 2 inputs
        } else if (param.splitNum == SPLIT_SUPPORT_SPLIT_NUM3) {
            return 3; // split has 3 inputs
        }
        MKI_LOG(ERROR) << "splitNum is wrong";
        return 0;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Split),
            "split: param type invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        OpParam::Split param = AnyCast<OpParam::Split>(launchParam.GetParam());
        SVector<int64_t> dims = launchParam.GetInTensor(0).desc.dims;
        TensorFormat format = launchParam.GetInTensor(0).desc.format;
        int64_t realSplitDim = param.splitDim;
        if ((param.splitDim < 0)) {
            if ((static_cast<int>(dims.size()) + param.splitDim) >= 0) {
                param.splitDim += static_cast<int>(dims.size());
                realSplitDim = param.splitDim;
            } else {
                return Status::FailStatus(ERROR_INVALID_VALUE, "Incorrect splitDim.");
            }
        }
        if (param.splitNum <= 0 || param.splitNum > MAX_SUPPORT_SPLIT_NUM) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "Incorrect splitNum.");
        }
        if (param.splitDim >= static_cast<int>(dims.size())) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "Incorrect splitDim.");
        }
        if (param.splitVDim[0] >= static_cast<int>(dims.size()) || param.splitVDim[0] < 0) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "Incorrect splitVDim.");
        }
        if (format != TENSOR_FORMAT_ND) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "tensor format should be ND");
        }
        if (dims[param.splitDim] % param.splitNum != 0 && param.splitSize.size() == 0) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "dims[splitDim] mod splitNum is not zero");
        }
        if (param.splitSize.size() > 0) {
            return SplitVInferShape(launchParam, outTensors);
        } else {
            return SplitInferShape(launchParam, outTensors, realSplitDim);
        }
    }

private:
    Status SplitVInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        OpParam::Split param = AnyCast<OpParam::Split>(launchParam.GetParam());
        TensorDType dtype = launchParam.GetInTensor(0).desc.dtype;
        SVector<int64_t> dims = launchParam.GetInTensor(0).desc.dims;
        TensorFormat format = launchParam.GetInTensor(0).desc.format;
        if (dtype == TENSOR_DTYPE_FLOAT16 || dtype == TENSOR_DTYPE_BF16) {
            for (size_t i = 0; i < launchParam.GetOutTensorCount(); i++) {
                dims.at(param.splitVDim[0]) = static_cast<int64_t>(param.splitSize[i]);
                outTensors[i].desc = {dtype, format, dims, {}, 0};
            }
            return Status::OkStatus();
        } else if (dtype == TENSOR_DTYPE_INT64) {
            for (size_t i = 0; i < launchParam.GetOutTensorCount(); i++) {
                dims.at(param.splitVDim[0]) = static_cast<int64_t>(param.splitSize[i]);
                outTensors[i].desc = {TENSOR_DTYPE_INT64, format, dims, {}, 0};
            }
            return Status::OkStatus();
        } else {
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Unsupported input descriptor.");
        }
    }

    Status SplitInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors, int64_t realSplitDim) const
    {
        OpParam::Split param = AnyCast<OpParam::Split>(launchParam.GetParam());
        TensorDType dtype = launchParam.GetInTensor(0).desc.dtype;
        SVector<int64_t> dims = launchParam.GetInTensor(0).desc.dims;
        TensorFormat format = launchParam.GetInTensor(0).desc.format;
        if (dtype == TENSOR_DTYPE_FLOAT16 || dtype == TENSOR_DTYPE_BF16) {
            dims.at(realSplitDim) = dims.at(realSplitDim) / param.splitNum;
            for (size_t i = 0; i < launchParam.GetOutTensorCount(); i++) {
                outTensors[i].desc = {dtype, format, dims, {}, 0};
            }
            return Status::OkStatus();
        } else if (dtype == TENSOR_DTYPE_INT64) {
            dims.at(realSplitDim) = dims.at(realSplitDim) / param.splitNum;
            for (size_t i = 0; i < launchParam.GetOutTensorCount(); i++) {
                outTensors[i].desc = {TENSOR_DTYPE_INT64, format, dims, {}, 0};
            }
            return Status::OkStatus();
        } else {
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Unsupported input descriptor.");
        }
    }
};
REG_OPERATION(SplitOperation);
} // namespace AsdOps
