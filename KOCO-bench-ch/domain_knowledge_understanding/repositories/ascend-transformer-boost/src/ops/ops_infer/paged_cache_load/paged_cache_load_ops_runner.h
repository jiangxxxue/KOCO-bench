/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef OPS_PAGED_CACHE_LOAD_OPS_RUNNER_H
#define OPS_PAGED_CACHE_LOAD_OPS_RUNNER_H

#include <atbops/params/params.h>
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class PagedCacheLoadOpsRunner : public OpsRunner {
public:
    explicit PagedCacheLoadOpsRunner(const infer::PagedCacheLoadParam &param);
    void SetPaParam(AtbOps::OpParam::PagedCacheLoad &pagedCacheLoadParam);
    ~PagedCacheLoadOpsRunner() override;

private:
    infer::PagedCacheLoadParam param_;
    Mki::Tensor nullTensor_ = {}; // 空tensor
};
} // namespace atb
#endif