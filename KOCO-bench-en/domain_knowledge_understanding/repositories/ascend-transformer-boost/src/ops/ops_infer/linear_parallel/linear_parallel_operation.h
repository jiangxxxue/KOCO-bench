/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_LINEAR_PARALLEL_OPERATION_H
#define ATB_LINEAR_PARALLEL_OPERATION_H
#include "atb/operation/graph_operation.h"
#include "atb/infer_op_params.h"
#include "atb/utils/operation_util.h"

namespace atb {
class LinearParallelOperation : public OperationBase {
public:
    explicit LinearParallelOperation(const infer::LinearParallelParam &param);
    ~LinearParallelOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    Status InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const override;
    Status SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const override;
    nlohmann::json GetParamJson() const override;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;

private:
    Status InferShapeLinearReduceScatter(const SVector<TensorDesc> &inTensorDescs,
                                         SVector<TensorDesc> &outTensorDescs) const;
    Status InferShapeAllGatherLinear(const SVector<TensorDesc> &inTensorDescs,
                                     SVector<TensorDesc> &outTensorDescs) const;
    Status InferShapeAllGatherLinearReduceScatter(const SVector<TensorDesc> &inTensorDescs,
                                                  SVector<TensorDesc> &outTensorDescs) const;
    Status InferShapeAllToAllvcAllGatherGmm(const SVector<TensorDesc> &inTensorDescs,
                                            SVector<TensorDesc> &outTensorDescs) const;
    Status CheckResidual(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapeCheckLinearAllReduce(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapeCheckLinearReduceScatter(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapeCheckAllGatherLinear(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapeCheckAllGatherLinearReduceScatter(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapeCheckAllToAllvcAllGatherGmm(const SVector<TensorDesc> &inTensorDescs) const;
    Status SetupCheckLinearAllReduce(const SVector<TensorDesc> &inTensorDescs, const TensorDesc &outTensorDesc) const;
    Status SetupCheckLinearReduceScatter(const SVector<TensorDesc> &inTensorDescs, TensorDesc &outTensorDesc) const;
    Status SetupCheckAllGatherLinear(SVector<TensorDesc> &inTensorDescs,
                                     const SVector<TensorDesc> &outTensorDescs) const;
    Status SetupCheckAllGatherLinearReduceScatter(SVector<TensorDesc> &inTensorDescs, TensorDesc &outTensorDesc) const;
    Status SetupCheckAllToAllvcAllGatherGmm(const SVector<TensorDesc> &inTensorDescs,
                                            const TensorDesc &outTensorDesc) const;

private:
    infer::LinearParallelParam param_;
    MatmulCommonCheckParam commonCheckParam_;
};
} // namespace atb
#endif