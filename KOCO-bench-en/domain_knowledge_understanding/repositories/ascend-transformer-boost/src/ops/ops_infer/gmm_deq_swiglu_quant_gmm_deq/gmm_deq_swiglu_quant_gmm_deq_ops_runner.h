/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_GMM_DEQ_SWIGLU_QUANT_GMM_DEQ_OPS_RUNNER_H
#define ATB_GMM_DEQ_SWIGLU_QUANT_GMM_DEQ_OPS_RUNNER_H

#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class GmmDeqSwigluQuantGmmDeqOpsRunner : public OpsRunner {
public:
    explicit GmmDeqSwigluQuantGmmDeqOpsRunner(const infer::GmmDeqSwigluQuantGmmDeqParam &param);
    ~GmmDeqSwigluQuantGmmDeqOpsRunner() override;
    void SetParam(const Mki::Any &param) override;

protected:
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    infer::GmmDeqSwigluQuantGmmDeqParam param_;
};

namespace infer {
inline bool operator==(const GmmDeqSwigluQuantGmmDeqParam &left, const GmmDeqSwigluQuantGmmDeqParam &right)
{
    return left.outputType == right.outputType && left.groupListType == right.groupListType &&
        left.weightUpPermuteType == right.weightUpPermuteType && left.transposeWeightUp == right.transposeWeightUp &&
        left.transposeWeightDown == right.transposeWeightDown;
}
} // namespace infer
} // namespace atb

#endif