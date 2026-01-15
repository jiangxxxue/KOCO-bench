/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "graph_node.h"
#include "atb/utils/log.h"

namespace TorchAtb {
using namespace atb;

void GraphNode::SetOperation(atb::Operation *op)
{
    operation = op;
}

std::string GraphNode::GetOutput(size_t index) const
{
    if (index >= outTensorIds.size()) {
        throw std::out_of_range("Output index is out of range");
    }
    return outTensorIds[index];
}

bool GraphNode::FindOutput(const std::string &id) const
{
    auto it = std::find(outTensorIds.begin(), outTensorIds.end(), id);
    return it != outTensorIds.end();
}

void GraphNode::SetStreamId(uint32_t streamId) const
{
    if (!operation) {
        throw std::runtime_error("Set execute stream id fail, operation is nullptr");
    }
    SetExecuteStreamId(operation, streamId);
}

uint32_t GraphNode::GetStreamId() const
{
    if (!operation) {
        throw std::runtime_error("Get execute stream id fail, operation is nullptr");
    }
    return GetExecuteStreamId(operation);
}
} // namespace TorchAtb