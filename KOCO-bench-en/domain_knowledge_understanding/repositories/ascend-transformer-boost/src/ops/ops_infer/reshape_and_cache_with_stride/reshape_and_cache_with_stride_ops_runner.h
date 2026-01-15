/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef OPS_RESHAPE_AND_CACHE_RESHAPEANDCACHEWITHSTRIDEOPSRUNNER_H
#define OPS_RESHAPE_AND_CACHE_RESHAPEANDCACHEWITHSTRIDEOPSRUNNER_H

#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class ReshapeAndCacheWithStrideOpsRunner : public OpsRunner {
public:
    explicit ReshapeAndCacheWithStrideOpsRunner(const infer::ReshapeAndCacheWithStrideParam &param);
    ~ReshapeAndCacheWithStrideOpsRunner() override;

private:
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    infer::ReshapeAndCacheWithStrideParam param_;
    Mki::Tensor nullTensor_ = {}; // 空tensor
};
} // namespace atb
#endif