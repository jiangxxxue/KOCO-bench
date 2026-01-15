/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_PLUGINOPERATION_H
#define ATB_PLUGINOPERATION_H
#include <string>
#include <functional>
#include <memory>
#include "atb/svector.h"
#include "operation_base.h"

namespace atb {
class PluginOperation : public OperationBase {
public:
    explicit PluginOperation(Operation *operation);
    ~PluginOperation() override;
    std::string GetName() const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    void SetExecuteStreamId(uint32_t streamId) override;
    uint32_t GetExecuteStreamId() const override;

protected:
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;

private:
    std::unique_ptr<Operation> operation_;
};
} // namespace atb
#endif
