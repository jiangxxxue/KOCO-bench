/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TOPP_WHEREOPERATION_H
#define OPS_TOPP_WHEREOPERATION_H
#include "atb/operation/operation_base.h"
#include "atb/infer_op_params.h"

namespace atb {
class TopkToppSamplingOperation : public OperationBase {
public:
    explicit TopkToppSamplingOperation(const infer::TopkToppSamplingParam &param);
    ~TopkToppSamplingOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    infer::TopkToppSamplingParam GetParam() const;
    void SetParam(const infer::TopkToppSamplingParam &param);

protected:
    Status CheckBatchSize(const SVector<TensorDesc> &inTensorDescs) const;
    Status CheckSingleTopk(const SVector<TensorDesc> &inTensorDescs) const;
    Status CheckMutinomial(const SVector<TensorDesc> &inTensorDescs) const;
    Status CheckMutinomialLogProbs(const SVector<TensorDesc> &inTensorDescs) const;
    Status CheckBatchTopkTopp(const SVector<TensorDesc> &inTensorDescs) const;
    Status CheckExponential(const SVector<TensorDesc> &inTensorDescs) const;
    Status CheckExponentialLogProbs(const SVector<TensorDesc> &inTensorDescs) const;
    Status CheckIntensorAndParam(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;
    Status InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const override;
    Status SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const override;
    nlohmann::json GetParamJson() const override;

private:
    infer::TopkToppSamplingParam param_;
    Status CheckLogProbsSize(const SVector<TensorDesc> &inTensorDescs) const;
};
} // namespace atb
#endif
