/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_SELF_ATTENTION_FUSION_BYPASS_OPS_RUNNER_910A_H
#define ATB_SELF_ATTENTION_FUSION_BYPASS_OPS_RUNNER_910A_H

#include <atbops/params/params.h>
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"
#include "param.h"

namespace atb {
class SelfAttentionFusionBypassOpsRunner910A : public OpsRunner {
public:
    explicit SelfAttentionFusionBypassOpsRunner910A(const infer::SelfAttentionParam &param);
    ~SelfAttentionFusionBypassOpsRunner910A() override;
    void SetParam(const Mki::Any &param) override;

protected:
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    Status ModifyKernelGraph(const OpsTensorPack &opsTensorPack) override;
    AtbOps::OpParam::UnpadFlashAttentionNz::Type GetFaType();
    void SetFAParam(AtbOps::OpParam::UnpadFlashAttentionNz &flashAttentionParam);
    Status SetKVCacheTensorList(AtbOps::OpParam::UnpadFlashAttentionNz &param, KernelGraphNode &flashAttentionNode);

private:
    infer::SelfAttentionParam param_;
    SelfAttentionFusionVariantPackParam newParam_;
    int64_t ntokens_ = 0;
    int64_t hiddenSize_ = 0;
    int64_t batch_ = 0;
    Mki::Tensor nullTensor_ = {}; // 空tensor
};
} // namespace atb
#endif // ATB_SELF_ATTENTION_FUSION_OPS_RUNNER_910A_H