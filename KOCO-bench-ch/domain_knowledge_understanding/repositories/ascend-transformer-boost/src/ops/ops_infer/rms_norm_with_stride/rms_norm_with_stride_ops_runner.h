/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_RMS_NORM_OPS_RUNNER_H
#define ATB_RMS_NORM_OPS_RUNNER_H

#include <asdops/params/params.h>
#include "atb/infer_op_params.h"
#include "atb/runner/ops_runner.h"

namespace atb {
class RmsNormWithStrideOpsRunner : public OpsRunner {
public:
    explicit RmsNormWithStrideOpsRunner(const infer::RmsNormWithStrideParam &param);
    ~RmsNormWithStrideOpsRunner() override;

private:
    void SetRmsNormParam(const infer::RmsNormWithStrideParam &inferParam, AsdOps::OpParam::Norm &asdopsParam) const;
    void BuildRmsNormGraph(const AsdOps::OpParam::Norm &rmsNormParam);
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    infer::RmsNormWithStrideParam param_;
    Mki::Tensor nullTensor_ = {}; // 空tensor，作为quant参数的占位符
};
} // namespace atb
#endif