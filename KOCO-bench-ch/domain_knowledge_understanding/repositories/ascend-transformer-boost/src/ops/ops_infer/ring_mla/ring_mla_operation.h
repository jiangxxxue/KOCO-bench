/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_RING_MLA_OPERATION_H
#define ATB_RING_MLA_OPERATION_H

#include "atb/operation/operation_base.h"
#include "atb/infer_op_params.h"

namespace atb {


class RingMLAOperation : public OperationBase {
public:
    explicit RingMLAOperation(const infer::RingMLAParam &opParam);
    ~RingMLAOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    bool DimNumCheck(const SVector<TensorDesc> &inTensorDescs, ExternalError &extError) const;
    bool QSplitDimCheck(const SVector<TensorDesc> &inTensorDescs, ExternalError &extError) const;
    bool KSplitDimCheck(const SVector<TensorDesc> &inTensorDescs, ExternalError &extError) const;
    bool InputLseDimCheck(const SVector<TensorDesc> &inTensorDescs, ExternalError &extError) const;
    Status DimCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const override;
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    Status SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const override;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;
    nlohmann::json GetParamJson() const override;

private:
    infer::RingMLAParam param_;
    infer::RingMLAParam::CalcType calcType = infer::RingMLAParam::CalcType::CALC_TYPE_DEFAULT;
    bool isInputSoftmaxLse_ = false;
};
} // namespace atb
#endif // ATB_RING_MLA_OPERATION_H
