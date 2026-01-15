/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_LASER_ATTENTION_OPS_RUNNER_H
#define ATB_LASER_ATTENTION_OPS_RUNNER_H
#include "atb/runner/ops_runner.h"
#include "atb/train_op_params.h"
#include "atb/utils/utils_internal.h"

namespace atb {
class LaserAttentionOpsRunner : public OpsRunner {
public:
    explicit LaserAttentionOpsRunner(const train::LaserAttentionParam &param);
    ~LaserAttentionOpsRunner() override;
    void SetParam(const Mki::Any &param) override;

protected:
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    train::LaserAttentionParam param_;
};

namespace train {
inline bool operator==(const LaserAttentionParam &left, const LaserAttentionParam &right)
{
    return left.headNum == right.headNum && left.inputLayout == right.inputLayout &&
           UtilsInternal::IsFloatEqual(left.scaleValue, right.scaleValue) &&
           UtilsInternal::IsFloatEqual(left.keepProb, right.keepProb) && left.preTokens == right.preTokens &&
           left.nextTokens == right.nextTokens && left.sparseMode == right.sparseMode &&
           left.innerPrecise == right.innerPrecise;
}
} // namespace train
} // namespace atb
#endif // ATB_LASER_ATTENTION_OPS_RUNNER_H
