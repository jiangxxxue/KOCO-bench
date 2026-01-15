/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_RELAY_ATTENTION_RUNNER_H
#define OPS_RELAY_ATTENTION_RUNNER_H
#include <atbops/params/params.h>
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"
#include "param.h"

namespace atb {
class RelayAttentionOpsRunner : public OpsRunner {
public:
    explicit RelayAttentionOpsRunner(const infer::RelayAttentionParam &param);
    ~RelayAttentionOpsRunner() override;
    void SetRAParam(AtbOps::OpParam::UnpadFlashAttention &flashAttentionParam);

private:
    infer::RelayAttentionParam param_;
    Status ModifyKernelGraph(const OpsTensorPack &opsTensorPack) override;
    Mki::Tensor nullTensor_ = {}; // 空tensor
    RelayAttentionVariantPackParam newParam_;
};
} // namespace atb
#endif