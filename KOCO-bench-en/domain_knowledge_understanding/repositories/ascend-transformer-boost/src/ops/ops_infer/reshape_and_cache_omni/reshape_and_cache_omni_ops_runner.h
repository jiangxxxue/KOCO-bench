/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_RESHAPE_AND_CACHE_OMNI_RESHAPEANDCACHEOMNIOPSRUNNER_H
#define OPS_RESHAPE_AND_CACHE_OMNI_RESHAPEANDCACHEOMNIOPSRUNNER_H

#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class ReshapeAndCacheOmniOpsRunner : public OpsRunner {
public:
    explicit ReshapeAndCacheOmniOpsRunner(const infer::ReshapeAndCacheOmniParam &param);
    ~ReshapeAndCacheOmniOpsRunner() override;
    void SetParam(const Mki::Any &param) override;

private:
    infer::ReshapeAndCacheOmniParam param_;
};

namespace infer {
inline bool operator==(const ReshapeAndCacheOmniParam &left, const ReshapeAndCacheOmniParam &right)
{
    (void) left;
    (void) right;
    return true;
}
}  // namespace infer
} // namespace atb
#endif