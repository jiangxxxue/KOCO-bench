/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_RING_MLA_OPS_RUNNER_H
#define ATB_RING_MLA_OPS_RUNNER_H


#include <atbops/params/params.h>
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class RingMLAOpsRunner : public OpsRunner {
public:
    explicit RingMLAOpsRunner(const infer::RingMLAParam &param);
    ~RingMLAOpsRunner() override;

private:
    Status ModifyKernelGraph(const OpsTensorPack &opsTensorPack) override;
    void SetRingMLAParam(AtbOps::OpParam::RINGMLA &RingMLAParam);

    infer::RingMLAParam param_;
    bool isInputSoftmaxLse_ = false;
    bool isNoMask_ = false;
    Mki::Tensor nullTensor_ = {}; // 空tensor，作为layerId或mask
};
} // namespace atb
#endif // ATB_RING_MLA_OPS_RUNNER_H
