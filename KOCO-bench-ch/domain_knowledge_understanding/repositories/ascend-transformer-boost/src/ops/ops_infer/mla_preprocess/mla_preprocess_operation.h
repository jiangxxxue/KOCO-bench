/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_MLAPREPROCESS_OPERATION_H
#define ATB_MLAPREPROCESS_OPERATION_H
#include "atb/operation/operation_base.h"
#include "atb/infer_op_params.h"

namespace atb {
class MlaPreprocessOperation : public OperationBase {
public:
    explicit MlaPreprocessOperation(const infer::MlaPreprocessParam &param);
    ~MlaPreprocessOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;
    Status InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const override;
    Status SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const override;
    nlohmann::json GetParamJson() const override;

private:
    Status DimCheck(const SVector<TensorDesc> &inTensorDesc) const;
    Status TensorShapeCheck(const atb::SVector<atb::TensorDesc> &inTensorDesc,
                            const std::vector<std::vector<int64_t>> &inTensorShapes) const;
    void ModifyInTensorShapeTable(const atb::SVector<atb::TensorDesc> &inTensorDesc,
                                  const std::vector<std::vector<int64_t>> &inTensorShapes, const int64_t headNum) const;
    Status OutTensorCheck(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const;
    Status OutTensorCheckSplit(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const;
    Status BlockSizeCheck(const SVector<TensorDesc> &inTensorDesc) const;
    // check the hiddenSize of input, gamma0 and beta0
    Status HiddenSizeCheck(const SVector<TensorDesc> &inTensorDesc) const;
    Status CheckAclnnKernel(const SVector<TensorDesc> &inTensorDesc) const;
private:
    infer::MlaPreprocessParam param_;
    mutable bool useAclnnKernel_ = false;
    mutable bool doRmsNorm_ = true;
};
} // namespace atb
#endif