/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_DYNAMICNTK_RUNNER_H
#define OPS_DYNAMICNTK_RUNNER_H
#include <asdops/params/params.h>
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class DynamicNTKOpsRunner : public OpsRunner {
public:
    explicit DynamicNTKOpsRunner(const infer::DynamicNTKParam &param);
    ~DynamicNTKOpsRunner() override;

private:
    infer::DynamicNTKParam param_;
    AsdOps::OpParam::DynamicNTK asdopsParam_;
};
} // namespace atb
#endif