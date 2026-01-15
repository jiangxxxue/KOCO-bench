/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_MLAPREPROCESS_OPS_RUNNER_H
#define ATB_MLAPREPROCESS_OPS_RUNNER_H
#include <cfloat>
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"
#include "atb/utils/utils_internal.h"

namespace atb {
class MlaPreprocessOpsRunner : public OpsRunner {
public:
    explicit MlaPreprocessOpsRunner(const infer::MlaPreprocessParam &param);
    ~MlaPreprocessOpsRunner() override;

protected:
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    Status ModifyKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    infer::MlaPreprocessParam param_;
    Mki::Tensor nullTensor_ = {}; // 空tensor，作为{true, 0}场景占位符
};

namespace infer {
inline bool operator==(const MlaPreprocessParam &left, const MlaPreprocessParam &right)
{
    return left.wdqDim == right.wdqDim && left.qRopeDim == right.qRopeDim && left.kRopeDim == right.kRopeDim &&
           UtilsInternal::IsFloatEqual(left.epsilon, right.epsilon) && left.qRotaryCoeff == right.qRotaryCoeff &&
           left.kRotaryCoeff == right.kRotaryCoeff && left.transposeWdq == right.transposeWdq &&
           left.transposeWuq == right.transposeWuq && left.transposeWuk == right.transposeWuk &&
           left.cacheMode == right.cacheMode;
}
} // namespace infer
} // namespace atb

#endif