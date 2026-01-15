/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_INDEX_ADD_OPS_RUNNER_H
#define ATB_INDEX_ADD_OPS_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/ops_runner.h"

namespace atb {
class IndexAddOpsRunner : public OpsRunner {
public:
    explicit IndexAddOpsRunner(const infer::IndexAddParam &param);
    ~IndexAddOpsRunner() override;

protected:
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    Status SetupKernelGraphIndexAdd();
    Status SetupKernelGraphIndexAddValid();

private:
    infer::IndexAddParam param_;
};
} // namespace atb
#endif // ATB_INDEX_ADD_OPS_RUNNER_H