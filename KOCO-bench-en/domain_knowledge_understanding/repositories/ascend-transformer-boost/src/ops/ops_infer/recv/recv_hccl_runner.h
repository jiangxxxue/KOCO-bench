/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_RECV_HCCL_RUNNER_H
#define ATB_RECV_HCCL_RUNNER_H
#include "atb/runner/hccl_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class RecvHcclRunner : public HcclRunner {
public:
    explicit RecvHcclRunner(const infer::RecvParam &param);
    RecvHcclRunner(const infer::RecvParam &param, bool useRankTableFile);
    RecvHcclRunner(const infer::RecvParam &param, HcclComm hcclComm);
    ~RecvHcclRunner() override;

protected:
    Status ExecuteImpl(RunnerVariantPack &runnerVariantPack) override;

private:
    infer::RecvParam param_;
};
} // namespace atb
#endif
