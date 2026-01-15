/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <sstream>
#include <mki/operation.h>
#include <asdops/ops.h>
#include <atbops/ops.h>
#include "atb/utils/tensor_util.h"
#include "atb/runner/mki_node_implement.h"
#include "atb/runner/kernel_graph.h"

namespace atb {
std::string KernelGraph::ToString() const
{
    std::stringstream ss;
    for (size_t i = 0; i < inTensors.size(); ++i) {
        ss << "inTensors[" << i << "]: " << TensorUtil::AsdOpsTensorToString(inTensors[i]) << std::endl;
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        ss << "outTensors[" << i << "]: " << TensorUtil::AsdOpsTensorToString(outTensors[i]) << std::endl;
    }
    for (size_t i = 0; i < internalTensors.size(); ++i) {
        ss << "internalTensors[" << i << "]: " << TensorUtil::AsdOpsTensorToString(internalTensors[i]) << std::endl;
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (size_t j = 0; j < nodes[i].inTensors.size(); ++j) {
            ss << "node[" << i << "] inTensors[" << j
               << "]: " << TensorUtil::AsdOpsTensorToString(*nodes[i].inTensors[j]) << std::endl;
        }
        for (size_t j = 0; j < nodes[i].outTensors.size(); ++j) {
            ss << "node[" << i << "] outTensors[" << j
               << "]: " << TensorUtil::AsdOpsTensorToString(*nodes[i].outTensors[j]) << std::endl;
        }
    }
    return ss.str();
}

void KernelGraph::Init()
{
    for (auto &node : nodes) {
        node.Reset();
        node.inTensorsType.reserve(node.inTensors.size());
        node.inTensorsType.resize(node.inTensors.size());
        node.outTensorsType.reserve(node.outTensors.size());
        node.outTensorsType.resize(node.outTensors.size());

        for (size_t i = 0; i < node.inTensors.size(); i++) {
            auto inTensor = node.inTensors.at(i);
            if (IsInternalTensor(inTensor)) {
                node.inTensorsType.at(i) = TensorType::INTERMEDIATE_TENSOR;
            } else {
                node.inTensorsType.at(i) = TensorType::IN_TENSOR;
            }
        }

        for (size_t i = 0; i < node.outTensors.size(); i++) {
            auto outTensor = node.outTensors.at(i);
            if (IsInternalTensor(outTensor)) {
                node.outTensorsType.at(i) = TensorType::INTERMEDIATE_TENSOR;
            } else {
                node.outTensorsType.at(i) = TensorType::OUT_TENSOR;
            }
        }
    }
}

bool KernelGraph::IsInternalTensor(const Mki::Tensor *tensor) const
{
    for (auto &internalTensor : internalTensors) {
        if (tensor == &internalTensor) {
            return true;
        }
    }

    return false;
}

void KernelGraphNode::Reset()
{
    if (impl) {
        impl->Reset();
    }
}

bool KernelGraphNode::CreateImplement()
{
    Mki::Operation *op = AsdOps::Ops::Instance().GetOperationByName(opDesc.opName);
    if (op) {
        impl = std::make_shared<MkiNodeImplement>(op, inferShapePreFunc);
    } else {
        op = AtbOps::Ops::Instance().GetOperationByName(opDesc.opName);
        if (op) {
            impl = std::make_shared<MkiNodeImplement>(op, mkiInferShapePreFunc);
        } else {
            impl.reset();
        }
    }

    if (!impl) {
        ATB_LOG(ERROR) << "node " << opDesc.opName << " implement is null";
        return false;
    }
    return true;
}

std::string KernelGraphNode::GetName() const
{
    return impl == nullptr ? opDesc.opName : impl->GetName();
}
} // namespace atb