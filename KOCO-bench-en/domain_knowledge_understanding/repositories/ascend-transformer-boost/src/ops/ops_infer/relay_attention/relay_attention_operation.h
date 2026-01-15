/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_RELAY_ATTENTION_OPERATION_H
#define ATB_RELAY_ATTENTION_OPERATION_H
#include "atb/operation/operation_base.h"
#include "atb/infer_op_params.h"

namespace atb {
class RelayAttentionOperation : public OperationBase {
public:
    explicit RelayAttentionOperation(const infer::RelayAttentionParam &param);
    ~RelayAttentionOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    void InitOpIni();
    Status InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const override;
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    Status SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const override;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;
    nlohmann::json GetParamJson() const override;

private:
    infer::RelayAttentionParam param_;
    Status InferShapeDimCheck(const SVector<TensorDesc> &inTensorDescs) const;
    bool InferShapeDimCheckKVShare(const SVector<TensorDesc> &inTensorDescs) const;
    bool InferShapeDimCheckKV(const SVector<TensorDesc> &inTensorDescs) const;
};
} // namespace atb
#endif