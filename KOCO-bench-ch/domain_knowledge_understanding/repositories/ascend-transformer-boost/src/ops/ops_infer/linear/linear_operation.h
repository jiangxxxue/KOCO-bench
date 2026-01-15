/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_LINEAR_OPERATION_H
#define ATB_LINEAR_OPERATION_H
#include "atb/infer_op_params.h"
#include "atb/operation/operation_base.h"
#include "atb/utils/operation_util.h"

namespace atb {
class LinearOperation : public OperationBase {
public:
    explicit LinearOperation(const infer::LinearParam &param);
    ~LinearOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    Status InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const override;
    Status SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const override;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;
    nlohmann::json GetParamJson() const override;

private:
    Status InTensorDescsCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status XWeightCheck(const TensorDesc &xTensorDesc, const TensorDesc &weightTensorDesc, int64_t align) const;
    Status BiasCheck(const TensorDesc &biasTensorDesc, const TensorDesc &weightTensorDesc) const;
    Status AccumCheck(const TensorDesc &accumTensorDesc, const TensorDesc &xTensorDesc,
                      const TensorDesc &weightTensorDesc) const;
    Status DeqScaleCheck(const TensorDesc &deqScaleTensorDesc, const TensorDesc &weightTensorDesc) const;
    Status PerTokenDeqScaleCheck(const TensorDesc &deqScaleTensorDesc, const TensorDesc &weightTensorDesc,
                                 const TensorDesc &xTensorDesc, const TensorDesc &perTokendeqScaleTensorDesc) const;
    Status OutTensorCheck(const SVector<TensorDesc> &inTensorDescs, const SVector<Tensor> &outTensors) const;
    bool XWeightDimNumCheck(const TensorDesc &xTensorDesc, const TensorDesc &weightTensorDesc) const;
    bool XWeightBatchCheck(const TensorDesc &xTensorDesc, const TensorDesc &weightTensorDesc) const;
    bool XWeightKEqualCheck(const TensorDesc &xTensorDesc, const TensorDesc &weightTensorDesc) const;
    bool WeightAlignCheck(const TensorDesc &weightTensorDesc, int64_t align) const;
    bool PerTokenXWeightDimNumCheck(const TensorDesc &xTensorDesc, const TensorDesc &weightTensorDesc) const;

private:
    infer::LinearParam param_;
    MatmulCommonCheckParam commonCheckParam_;
};
} // namespace atb
#endif