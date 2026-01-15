/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TORCH_ATB_GRAPH_NODE_H
#define TORCH_ATB_GRAPH_NODE_H
#include <acl/acl_rt.h>
#include "atb/context/context_base.h"
#include "atb/operation.h"
#include "atb/operation/operation_base.h"
#include "atb/types.h"
#include "resource/utils.h"

namespace TorchAtb {
class GraphNode {
public:
    void SetOperation(atb::Operation *op);
    std::string GetOutput(size_t index) const;
    bool FindOutput(const std::string &id) const;
    void SetStreamId(uint32_t streamId) const;
    uint32_t GetStreamId() const;

    std::vector<std::string> inTensorIds;
    std::vector<std::string> outTensorIds;
    atb::Operation *operation = nullptr;

private:
    std::string name;
    std::vector<GraphNode> graphNodes_;
};

} // namespace TorchAtb
#endif // TORCH_ATB_GRAPH_OPERATION_WRAPPER_H