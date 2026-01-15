/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LAYER_NORM_OPS_RUNNER_H
#define LAYER_NORM_OPS_RUNNER_H

#include <asdops/params/params.h>
#include "atb/infer_op_params.h"
#include "atb/runner/ops_runner.h"

namespace atb {
class LayerNormOpsRunner : public OpsRunner {
public:
    explicit LayerNormOpsRunner(const infer::LayerNormParam &param);
    ~LayerNormOpsRunner() override;

private:
    void SetLayerNormParam(const infer::LayerNormParam &inferParam, AsdOps::OpParam::Norm &asdopsParam) const;
    void SetLayerNormQuantParam(const infer::LayerNormParam &inferParam, AsdOps::OpParam::Norm &asdopsParam) const;
    void SetPreLayerNormParam(const infer::LayerNormParam &inferParam, AsdOps::OpParam::Norm &asdopsParam) const;
    void SetPostLayerNormParam(const infer::LayerNormParam &inferParam, AsdOps::OpParam::Norm &asdopsParam) const;
    void SetPostLayerNormQuantParam(const infer::LayerNormParam &inferParam, AsdOps::OpParam::Norm &asdopsParam) const;
    void BuildLayerNormGraph(const AsdOps::OpParam::Norm &layerNormParam);
    void BuildLayerNormQuantGraph(const AsdOps::OpParam::Norm &layerNormParam);
    void BuildLayerNormDynamicQuantGraph(const AsdOps::OpParam::Norm &layerNormParam);
    void BuildPreLayerNormGraph(const AsdOps::OpParam::Norm &layerNormParam);
    void BuildPostLayerNormGraph(const AsdOps::OpParam::Norm &layerNormParam);
    void BuildPostLayerNormQuantGraph(const AsdOps::OpParam::Norm &layerNormParam);

private:
    infer::LayerNormParam param_;
    Mki::Tensor nullTensor_ = {};
};
} // namespace atb
#endif