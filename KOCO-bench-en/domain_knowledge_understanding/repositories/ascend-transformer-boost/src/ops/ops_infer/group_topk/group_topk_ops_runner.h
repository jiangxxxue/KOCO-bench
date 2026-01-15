/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_GROUP_TOPK_OPS_RUNNER_H
#define ATB_GROUP_TOPK_OPS_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/ops_runner.h"

namespace atb {
class GroupTopkOpsRunner : public OpsRunner {
public:
    explicit GroupTopkOpsRunner(const infer::GroupTopkParam &param);
    ~GroupTopkOpsRunner() override;
    void SetParam(const Mki::Any &param) override;

protected:
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    infer::GroupTopkParam param_;
};

inline bool operator==(const infer::GroupTopkParam &left, const infer::GroupTopkParam &right)
{
    return left.groupNum == right.groupNum && left.k == right.k;
}
} // namespace atb
#endif // ATB_GROUP_TOPK_OPS_RUNNER_H