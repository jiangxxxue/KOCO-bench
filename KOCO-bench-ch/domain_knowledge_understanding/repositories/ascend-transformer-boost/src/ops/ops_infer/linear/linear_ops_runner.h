/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_LINEAR_OPS_RUNNER_H
#define ATB_LINEAR_OPS_RUNNER_H
#include <asdops/params/params.h>
#include "atb/infer_op_params.h"
#include "atb/runner/ops_runner.h"

namespace atb {
class LinearOpsRunner : public OpsRunner {
public:
    explicit LinearOpsRunner(const infer::LinearParam &param);
    ~LinearOpsRunner() override;

protected:
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    Status SetupKernelGraphMatmul910B();
    Status SetupKernelGraphMatmulWeightNdNot910B();
    Status SetupKernelGraphMatmulWeightNzNot910B();
    Status SetupKernelGraphMatmulElewiseAdd910B();
    Status SetupKernelGraphMatmulElewiseAddWeightNdNot910B();
    Status SetupKernelGraphMatmulElewiseAddWeightNzNot910B();
    Status SetupKernelGraphMatmulWithBias();
    Status SetupKernelGraphMatmulAccum();
    Status SetupKernelGraphMatmulEin();
    Status SetupKernelGraphMatmulEinElewiseAdd();
    Status SetupKernelGraphMatmulDequant910B();
    Status SetupKernelGraphMatmulDequantPerToken910B();
    Status SetupKernelGraphMatmulDequantWeightNdNot910B();
    Status SetupKernelGraphMatmulDequantWeightNzNot910B();
    Status SetupKernelGraphMoeGateCorr();
    void SetupMatmulOriShape(const Mki::Tensor &xTensor, const Mki::Tensor &weightTensor);
    void SetupMatmulOriShapeEin(const Mki::Tensor &xTensor, const Mki::Tensor &weightTensor);
    void InitKernelGraph(size_t inTensorNum, size_t outTensorNum, size_t internalTensorNum, size_t nodeNum);
    void SetupNeedMergeAxis(const Mki::Tensor &xTensor, const Mki::Tensor &weightTensor);

private:
    infer::LinearParam param_;
    bool isWeightNz_ = false;
    bool xNeedMergeAxis_ = false;
    bool weightNeedMergeAxis_ = false;
    AsdOps::OpParam::MatMul matmulParam_;
    AsdOps::OpParam::MoeGateCorr moeGateCorrParam_;
    AsdOps::OpParam::Elewise elewiseAddParam_;
    AsdOps::OpParam::Transdata transdataNdToNzParam_;
    AsdOps::OpParam::Transdata transdataNzToNdParam_;
    Mki::Tensor nullTensor_ = {};
    ViewFunc matmulMergeAxis_;
    ViewFunc matmulNzReshape_;
    ViewFunc transdataUnsqueeze_;
    ViewFunc elewiseAddUnsqueeze_;
};
} // namespace atb

#endif // ATB_LINEAR_OPS_RUNNER_H
