/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SCATTER_ELEMENTS_V2_OPS_OPERATION_H
#define SCATTER_ELEMENTS_V2_OPS_OPERATION_H
#include "atb/operation/operation_base.h"
#include "atb/infer_op_params.h"
 
namespace atb {
class ScatterElementsV2Operation : public OperationBase {
public:
    explicit ScatterElementsV2Operation(const infer::ScatterElementsV2Param &param);
    ~ScatterElementsV2Operation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
 
protected:
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;
    Status InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const override;
    Status SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const override;
    nlohmann::json GetParamJson() const override;
 
private:
    Status ParamCheck(const TensorDesc &inputTensorDesc, const TensorDesc &indicesTensorDesc,
                      const TensorDesc &updateTensorDesc) const;
    Status CheckDims(const TensorDesc &inputTensorDesc, const TensorDesc &indicesTensorDesc,
                     const TensorDesc &updateTensorDesc) const;
    Status CheckShape(const TensorDesc &inputTensorDesc, const TensorDesc &indicesTensorDesc,
                     const TensorDesc &updateTensorDesc) const;
    infer::ScatterElementsV2Param param_;
};
} // namespace atb
#endif