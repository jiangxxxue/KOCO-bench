/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_GATING_OPERATION_H
#define ATB_GATING_OPERATION_H

#include "atb/infer_op_params.h"
#include "atb/operation/operation_base.h"

namespace atb {
class GatingOperation : public OperationBase {
public:
    explicit GatingOperation(const infer::GatingParam &param);
    ~GatingOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    Status InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const override;
    Status SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const override;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;

private:
    Status InTensorDescsCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status OutTensorsCheck(const SVector<Tensor> &outTensors, int64_t topkDim0) const;

private:
    infer::GatingParam param_;
};
} // namespace atb
#endif // ATB_GATING_OPERATION_H
