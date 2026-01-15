/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_FILL_FILLOPSRUNNER_H
#define OPS_FILL_FILLOPSRUNNER_H
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"
#include "atb/utils/utils_internal.h"

namespace atb {
class FillOpsRunner : public OpsRunner {
public:
    explicit FillOpsRunner(const infer::FillParam &param);
    ~FillOpsRunner() override;

private:
    void SetFillGraph();
    void SetMaskedFillGraph();

    infer::FillParam param_;
};

namespace infer {
inline bool operator==(const FillParam &left, const FillParam &right)
{
    return left.withMask == right.withMask &&
           [](const SVector<float> &v1, const SVector<float> &v2) {
               if (v1.size() != v2.size()) {
                   return false;
               }
               for (size_t i = 0; i < v1.size(); ++i) {
                   if (!UtilsInternal::IsFloatEqual(v1[i], v2[i])) {
                       return false;
                   }
               }
               return true;
           }(left.value, right.value) &&
           left.outDim == right.outDim;
}
} // namespace infer
} // namespace atb
#endif