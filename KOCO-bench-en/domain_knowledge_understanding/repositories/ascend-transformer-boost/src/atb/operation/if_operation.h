/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_IF_OPERATION_H
#define ATB_IF_OPERATION_H
#include <string>
#include <functional>
#include <memory>
#include "atb/common_op_params.h"
#include "atb/svector.h"
#include "operation_base.h"

namespace atb {
class IfOperation : public OperationBase {
public:
    explicit IfOperation(const common::IfCondParam &param);
    ~IfOperation() override;
    std::string GetName() const override;
    Status Setup(const VariantPack &variantPack, uint64_t &workspaceSize, Context *context) override;
    Status Execute(const VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                   Context *context) override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    void SetExecuteStreamId(uint32_t streamId) override;

protected:
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;

private:
    Status GetOperationFromCondition(Operation *&op) const;

private:
    common::IfCondParam param_;
    mutable Operation *opSelected_ = nullptr;
};
} // namespace atb
#endif // ATB_IF_OPERATION_H