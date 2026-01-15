/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_FUSED_ADD_TOPK_DIV_OPS_RUNNER_H
#define ATB_FUSED_ADD_TOPK_DIV_OPS_RUNNER_H
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"
#include "atb/utils/utils_internal.h"

namespace atb {
class FusedAddTopkDivOpsRunner : public OpsRunner {
public:
    explicit FusedAddTopkDivOpsRunner(const infer::FusedAddTopkDivParam &param);
    ~FusedAddTopkDivOpsRunner() override;
    void SetParam(const Mki::Any &param) override;

protected:
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    infer::FusedAddTopkDivParam param_;
    Mki::Tensor nullTensor_ = {}; // 空tensor，作为enableExpertMapping==false场景占位符
};

namespace infer {
inline bool operator==(const FusedAddTopkDivParam &left, const FusedAddTopkDivParam &right)
{
    return left.groupNum == right.groupNum && left.groupTopk == right.groupTopk && left.n == right.n &&
           left.k == right.k && left.activationType == right.activationType && left.isNorm == right.isNorm &&
           UtilsInternal::IsFloatEqual(left.scale, right.scale) &&
           left.enableExpertMapping == right.enableExpertMapping;
}
}  // namespace infer
}  // namespace atb
#endif  // ATB_FUSED_ADD_TOPK_DIV_OPS_RUNNER_H