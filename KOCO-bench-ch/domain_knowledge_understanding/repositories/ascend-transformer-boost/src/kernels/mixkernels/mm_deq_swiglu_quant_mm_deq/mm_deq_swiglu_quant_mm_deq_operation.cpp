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

#include "mm_deq_swiglu_quant_mm_deq_common.h"

namespace AtbOps {

using namespace Mki;
class MmDeqSwigluQuantMmDeqOperation : public OperationBase {
public:
    explicit MmDeqSwigluQuantMmDeqOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::MmDeqSwigluQuantMmDeq), "OpParam is invalid", return 0);
        return INPUT_NUM;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::MmDeqSwigluQuantMmDeq), "OpParam is invalid", return 0);
        return OUTPUT_NUM;
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(CheckMmDeqSwigluQuantMmDeq(launchParam), "Failed to check run info",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check run info"));

        int64_t m = launchParam.GetInTensor(InTensorIndex::X1).desc.dims.at(XDimIndex::M);

        MKI_CHECK(m <= MAX_M, "MmDeqSwigluQuantMmDeq only support m not greater than 131072",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Input shape is invalid"));

        MKI_CHECK(CheckX1(launchParam, m), "Unsupport x1",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check run info"));
        MKI_CHECK(CheckWeight1(launchParam), "Unsupport weight1",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check run info"));
        MKI_CHECK(CheckScale1(launchParam), "Unsupport scale1",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check run info"));
        MKI_CHECK(CheckPerTokenScale1(launchParam, m), "Unsupport perTokenScale1",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check run info"));
        MKI_CHECK(CheckWeight2(launchParam), "Unsupport weight2",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check run info"));
        MKI_CHECK(CheckScale2(launchParam), "Unsupport scale2",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check run info"));

        auto &outDesc = outTensors[0].desc;
        outDesc.dtype = TensorDType::TENSOR_DTYPE_FLOAT16;
        outDesc.format = TensorFormat::TENSOR_FORMAT_ND;
        outDesc.dims = {m, SUPPORTED_N2};

        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MmDeqSwigluQuantMmDeq),
            "OpParam is invalid", return nullptr);
        MKI_CHECK(CheckMmDeqSwigluQuantMmDeq(launchParam), "Failed to check run info", return nullptr);

        auto param = AnyCast<OpParam::MmDeqSwigluQuantMmDeq>(launchParam.GetParam());
        if (param.weightUpPermuteType == OpParam::MmDeqSwigluQuantMmDeq::PERMUTE_N256) {
            return GetKernelByName("MmDeqSwigluQuantMmDeqN256Kernel");
        } else if (param.weightUpPermuteType == OpParam::MmDeqSwigluQuantMmDeq::PERMUTE_N128) {
            return GetKernelByName("MmDeqSwigluQuantMmDeqN128Kernel");
        } else {
            return nullptr;
        }
    }

private:
    bool CheckMmDeqSwigluQuantMmDeq(const LaunchParam &launchParam) const
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MmDeqSwigluQuantMmDeq),
            "OpParam is invalid", return false);
        auto param = AnyCast<OpParam::MmDeqSwigluQuantMmDeq>(launchParam.GetParam());
        MKI_CHECK(param.outputType == OpParam::MmDeqSwigluQuantMmDeq::OUTPUT_FLOAT16,
            "Param outputType only support OUTPUT_FLOAT16 (0).", return false);
        MKI_CHECK(param.weightUpPermuteType != OpParam::MmDeqSwigluQuantMmDeq::PERMUTE_INVALID,
            "Param weightUpPermuteType has invalid value.", return false);
        MKI_CHECK(!param.transposeWeightUp,
            "Param transposeWeightUp only support false.", return false);
        MKI_CHECK(param.transposeWeightDown,
            "Param transposeWeightDown only support true.", return false);
        return true;
    }

    bool CheckX1(const LaunchParam &launchParam, int64_t m) const
    {
        auto &x1Desc = launchParam.GetInTensor(InTensorIndex::X1).desc;

        MKI_CHECK(x1Desc.dims.size() == X_DIMS,
            "Unsupported x1 shape of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(x1Desc.dims.at(XDimIndex::M) == m,
            "Unsupported x1 shape of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(x1Desc.dims.at(XDimIndex::K) == SUPPORTED_K1,
            "Unsupported x1 shape of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(x1Desc.dtype == TENSOR_DTYPE_INT8,
            "Unsupported x1 dtype of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(x1Desc.format == TENSOR_FORMAT_ND,
            "Unsupported x1 format of MmDeqSwigluQuantMmDeq", return false);
        return true;
    }

    bool CheckWeight1(const LaunchParam &launchParam) const
    {
        auto &weight1Desc = launchParam.GetInTensor(InTensorIndex::WEIGHT1).desc;

        MKI_CHECK(weight1Desc.dims.size() == WEIGHT_DIMS,
            "Unsupported weight1 shape of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(weight1Desc.dims.at(WeightDimIndex<false>::K) == SUPPORTED_K1,
            "Unsupported weight1 shape of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(weight1Desc.dims.at(WeightDimIndex<false>::N) == SUPPORTED_N1,
            "Unsupported weight1 shape of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(weight1Desc.dtype == TENSOR_DTYPE_INT8,
            "Unsupported weight1 dtype of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(weight1Desc.format == TENSOR_FORMAT_FRACTAL_NZ,
            "Unsupported weight1 format of MmDeqSwigluQuantMmDeq", return false);
        return true;
    }

    bool CheckScale1(const LaunchParam &launchParam) const
    {
        auto &scale1Desc = launchParam.GetInTensor(InTensorIndex::SCALE1).desc;

        MKI_CHECK(scale1Desc.dims.size() == SCALE_DIMS,
            "Unsupported scale1 shape of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(scale1Desc.dims.at(ScaleDimIndex::N) == SUPPORTED_N1,
            "Unsupported scale1 shape of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(scale1Desc.dtype == TENSOR_DTYPE_FLOAT,
            "Unsupported scale1 dtype of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(scale1Desc.format == TENSOR_FORMAT_ND,
            "Unsupported scale1 format of MmDeqSwigluQuantMmDeq", return false);
        return true;
    }

    bool CheckPerTokenScale1(const LaunchParam &launchParam, int64_t m) const
    {
        auto &perTokenScale1Desc = launchParam.GetInTensor(InTensorIndex::PER_TOKEN_SCALE1).desc;

        MKI_CHECK(perTokenScale1Desc.dims.size() == PER_TOKEN_SCALE_DIMS,
            "Unsupported perTokenScale1 shape of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(perTokenScale1Desc.dims.at(PerTokenScaleDimIndex::M) == m,
            "Unsupported perTokenScale1 shape of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(perTokenScale1Desc.dtype == TENSOR_DTYPE_FLOAT,
            "Unsupported perTokenScale1 dtype of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(perTokenScale1Desc.format == TENSOR_FORMAT_ND,
            "Unsupported perTokenScale1 format of MmDeqSwigluQuantMmDeq", return false);
        return true;
    }

    bool CheckWeight2(const LaunchParam &launchParam) const
    {
        auto &weight2Desc = launchParam.GetInTensor(InTensorIndex::WEIGHT2).desc;

        MKI_CHECK(weight2Desc.dims.size() == WEIGHT_DIMS,
            "Unsupported weight2 shape of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(weight2Desc.dims.at(WeightDimIndex<true>::N) == SUPPORTED_N2,
            "Unsupported weight2 shape of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(weight2Desc.dims.at(WeightDimIndex<true>::K) == SUPPORTED_K2,
            "Unsupported weight2 shape of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(weight2Desc.dtype == TENSOR_DTYPE_INT8,
            "Unsupported weight2 dtype of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(weight2Desc.format == TENSOR_FORMAT_FRACTAL_NZ,
            "Unsupported weight2 format of MmDeqSwigluQuantMmDeq", return false);
        return true;
    }

    bool CheckScale2(const LaunchParam &launchParam) const
    {
        auto &scale2Desc = launchParam.GetInTensor(InTensorIndex::SCALE2).desc;

        MKI_CHECK(scale2Desc.dims.size() == SCALE_DIMS,
            "Unsupported scale2 shape of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(scale2Desc.dims.at(ScaleDimIndex::N) == SUPPORTED_N2,
            "Unsupported scale2 shape of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(scale2Desc.dtype == TENSOR_DTYPE_FLOAT,
            "Unsupported scale2 dtype of MmDeqSwigluQuantMmDeq", return false);
        MKI_CHECK(scale2Desc.format == TENSOR_FORMAT_ND,
            "Unsupported scale2 format of MmDeqSwigluQuantMmDeq", return false);
        return true;
    }
};

REG_OPERATION(MmDeqSwigluQuantMmDeqOperation);
} // namespace AtbOps
