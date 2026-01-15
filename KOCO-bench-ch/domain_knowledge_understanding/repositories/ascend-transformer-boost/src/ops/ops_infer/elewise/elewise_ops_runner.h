/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_ELEWISE_ELEWISEOPSRUNNER_H
#define OPS_ELEWISE_ELEWISEOPSRUNNER_H
#include <asdops/params/params.h>
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class ElewiseOpsRunner : public OpsRunner {
public:
    explicit ElewiseOpsRunner(const infer::ElewiseParam &param);
    ~ElewiseOpsRunner() override;

private:
    infer::ElewiseParam param_;
    Mki::Tensor nullTensor_ = {}; // 空tensor占位符
    bool SetIntensor(KernelGraphNode &elewiseNode);
    void SetOuttensor(KernelGraphNode &elewiseNode);
    uint32_t GetIntensorSize() const;
    AsdOps::OpParam::Elewise::ElewiseType GetOpElwiseType() const;
    Mki::TensorDType GetOutTensorType(const aclDataType outType) const;
};
} // namespace atb
#endif