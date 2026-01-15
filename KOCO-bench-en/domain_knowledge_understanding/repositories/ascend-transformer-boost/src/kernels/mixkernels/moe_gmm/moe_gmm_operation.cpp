/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "mki/base/operation_base.h"
#include "mki/utils/log/log.h"
#include "mki/utils/const/op_const.h"
#include "mki_loader/op_register.h"
#include "atbops/params/params.h"

const uint64_t K_NUM_IN_TENSOR = 4;
const uint64_t K_NUM_IN_TENSOR_W8A8 = 6;
const uint64_t K_NUM_OUT_TENSOR = 1;

namespace AtbOps {
using namespace Mki;
class MoeGmmOperation : public OperationBase {
public:
    explicit MoeGmmOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::MoeGmm), "Invalid Op Param.", return 0);
        auto attr = AnyCast<OpParam::MoeGmm>(specificParam);
        switch (attr.moeGmmDequantType) {
            case OpParam::MoeGmm::NO_DEQUANT: return K_NUM_IN_TENSOR;
            case OpParam::MoeGmm::DEQ_FP16:
            case OpParam::MoeGmm::DEQ_BF16: return K_NUM_IN_TENSOR_W8A8;
            default: break;
        }
        return -1;
    }

    int64_t GetOutputNum(const Any &specificParam) const override { return K_NUM_OUT_TENSOR; }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(CheckUpLaunchParam(launchParam), "Infershape failed.",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        auto inTensorDescA = launchParam.GetInTensor(0).desc;
        auto index = launchParam.GetInTensor(3).desc;

        TensorDesc &tensorDescOut = outTensors[0].desc;

        auto attrs = AnyCast<OpParam::MoeGmm>(launchParam.GetParam());
        MKI_CHECK(attrs.hiddenSize.size() == 2, "Infershape failed.",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(attrs.hiddenSize[0] > 0, "Infershape failed.", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(attrs.hiddenSize[1] > 0, "Infershape failed.", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(attrs.topK > 0, "Infershape failed.", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        tensorDescOut.dtype = inTensorDescA.dtype;
        tensorDescOut.format = inTensorDescA.format;
        auto &outDims = tensorDescOut.dims;
        outDims.clear();
        switch (attrs.moeGmmMode) {
            case OpParam::MoeGmm::MOE_GMM_UP:
                outDims.emplace_back(index.dims[0]);
                outDims.emplace_back(attrs.hiddenSize[1]);
                break;
            case OpParam::MoeGmm::MOE_GMM_DOWN:
                tensorDescOut = inTensorDescA;
                tensorDescOut.dims[0] = tensorDescOut.dims[0] / static_cast<int64_t>(attrs.topK);
                tensorDescOut.dims[1] = attrs.hiddenSize[1];
                break;
            default: break;
        }
        switch (attrs.moeGmmDequantType) {
            case OpParam::MoeGmm::DEQ_BF16: tensorDescOut.dtype = TENSOR_DTYPE_BF16; break;
            case OpParam::MoeGmm::DEQ_FP16: tensorDescOut.dtype = TENSOR_DTYPE_FLOAT16; break;
            default: break;
        }
        return Mki::Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MoeGmm), "Invalid Op Param.", return nullptr);
        auto attr = AnyCast<OpParam::MoeGmm>(launchParam.GetParam());
        switch (attr.moeGmmDequantType) {
            case OpParam::MoeGmm::NO_DEQUANT: return GetKernelByName("MoeGmmKernel");
            case OpParam::MoeGmm::DEQ_BF16:
            case OpParam::MoeGmm::DEQ_FP16: return GetKernelByName("MoeGmmW8a8Kernel");
            default: break;
        }
        return nullptr;
    }

private:
    bool CheckUpLaunchParam(const LaunchParam &launchParam) const
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MoeGmm), "Invalid op param.", return false);
        return true;
    }
};

REG_OPERATION(MoeGmmOperation);
} // namespace AtbOps
