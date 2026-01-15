/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TOPP_WHEREOPSRUNNER_H
#define OPS_TOPP_WHEREOPSRUNNER_H
#include <mki/utils/SVector/SVector.h>
#include <atbops/params/params.h>
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"


namespace atb {
class TopkToppSamplingOpsRunner : public OpsRunner {
public:
    explicit TopkToppSamplingOpsRunner(const infer::TopkToppSamplingParam &param);
    ~TopkToppSamplingOpsRunner() override;
    void SetParam(const Mki::Any &param) final;

protected:
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    Status SetupBatchTopKExponentialSampling();
    Status SetupBatchTopKMultinomialSampling();
    Status SetupLogProbsBatchTopKExponentialSampling();
    Status SetupLogProbsBatchTopKMultinomialSampling();
    Status SetupSingleTopKSampling();
    infer::TopkToppSamplingParam param_;

    size_t dimNum_{0};
    int64_t lastDim_{0};
};
} // namespace atb
#endif
