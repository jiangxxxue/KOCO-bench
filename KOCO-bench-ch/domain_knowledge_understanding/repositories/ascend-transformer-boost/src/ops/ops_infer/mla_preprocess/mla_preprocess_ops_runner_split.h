/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_MLAPREPROCESS_OPS_RUNNER_SPLIT_H
#define ATB_MLAPREPROCESS_OPS_RUNNER_SPLIT_H
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class MlaPreprocessOpsRunnerSplit : public OpsRunner {
public:
    explicit MlaPreprocessOpsRunnerSplit(const infer::MlaPreprocessParam &param);
    ~MlaPreprocessOpsRunnerSplit() override;

protected:
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    Status ModifyKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    infer::MlaPreprocessParam param_;
};
} // namespace atb

#endif