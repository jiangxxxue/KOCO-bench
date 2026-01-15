/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_ELEWISE_ELEWISEOPERATION_H
#define OPS_ELEWISE_ELEWISEOPERATION_H
#include <acl/acl.h>
#include "atb/operation/operation_base.h"
#include "atb/infer_op_params.h"

namespace atb {
class ElewiseOperation : public OperationBase {
public:
    explicit ElewiseOperation(const infer::ElewiseParam &param);
    ~ElewiseOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    Status InferShapeImplCast(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const;
    Status InferShapeImplQuant(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const;
    Status InferShapeImplDynamicQuant(const SVector<TensorDesc> &inTensorDescs,
                                      SVector<TensorDesc> &outTensorDescs) const;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;
    SVector<bool> GetEmptyInTensorPermissions() const override;
    nlohmann::json GetParamJson() const override;

private:
    infer::ElewiseParam param_;
    Status InferShapeCommon(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const;
    Status InferShapeImplQuantChannel(const SVector<TensorDesc> &inTensorDescs,
                                      SVector<TensorDesc> &outTensorDescs) const;
    Status InferShapeImplDequantChannel(const SVector<TensorDesc> &inTensorDescs,
                                        SVector<TensorDesc> &outTensorDescs) const;
    bool InferShapeCheckDynamicQuant(const SVector<TensorDesc> &inTensorDescs) const;
    bool InferShapeCheckDynamicQuant310P(const SVector<TensorDesc> &inTensorDescs) const;
};
} // namespace atb
#endif