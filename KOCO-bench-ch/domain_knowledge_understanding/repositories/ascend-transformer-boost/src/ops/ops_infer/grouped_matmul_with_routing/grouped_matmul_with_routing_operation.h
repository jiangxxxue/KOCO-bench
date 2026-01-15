/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_MOE_OPERATION_H
#define ATB_MOE_OPERATION_H
#include "atb/operation/operation_base.h"
#include "atb/infer_op_params.h"

namespace atb {
class GroupedMatmulWithRoutingOperation : public OperationBase {
public:
    explicit GroupedMatmulWithRoutingOperation(const infer::GroupedMatmulWithRoutingParam &param);
    ~GroupedMatmulWithRoutingOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;
    Status InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const override;
    Status SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const override;
    nlohmann::json GetParamJson() const override;

private:
    Status OutTensorDimCheck(const SVector<TensorDesc> &inTensorDescs, const SVector<Tensor> &outTensors) const;
    bool TokenExpertTensorCheck(const TensorDesc &acTensorDesc, const TensorDesc &expertWeightDesc) const;
    bool ExpertCountTensorCheck(const TensorDesc &expertCount, const TensorDesc &expertWeightDesc) const;
    bool ExpertIndexTensorCheck(const TensorDesc &acTensorDesc, const TensorDesc &expertIndex) const;
    bool NScaleTensorCheck(const TensorDesc &nScaleDesc, const TensorDesc &expertWeightDesc) const;
    bool MScaleTensorCheck(const TensorDesc &mScaleDesc, const TensorDesc &acTensorDesc) const;
    infer::GroupedMatmulWithRoutingParam param_;
};
} // namespace atb

#endif // ATB_ALL_REDUCE_OPERATION_H
