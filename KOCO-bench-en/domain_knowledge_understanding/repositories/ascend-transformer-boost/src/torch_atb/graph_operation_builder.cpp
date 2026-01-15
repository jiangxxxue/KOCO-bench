/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "graph_operation_builder.h"
#include <stdexcept>
#include "atb/utils/log.h"

namespace TorchAtb {

using namespace atb;
using namespace atb::infer;

static const size_t MAX_TENSOR_NUM = 256;
GraphOperationBuilder::GraphOperationBuilder(const std::string &graphName)
{
    graphParam_.name = graphName;
}

GraphOperationBuilder &GraphOperationBuilder::SetInputOutput(const std::vector<std::string> &inTensorNames,
                                                             const std::vector<std::string> &outTensorNames)
{
    uint32_t id = 0;
    size_t inTensorNum = inTensorNames.size();
    if (inTensorNum > MAX_TENSOR_NUM) {
        throw std::runtime_error("inTensor num should be <= 256, but get " + std::to_string(inTensorNum));
    }
    for (const std::string &inTensorName : inTensorNames) {
        inTensorIds_[inTensorName] = id++;
    }
    size_t outTensorNum = outTensorNames.size();
    if (outTensorNum > MAX_TENSOR_NUM) {
        throw std::runtime_error("outTensor num should be <= 256, but get " + std::to_string(outTensorNum));
    }
    for (const std::string &outTensorName : outTensorNames) {
        outTensorIds_[outTensorName] = id++;
    }
    graphParam_.inTensorNum = static_cast<uint32_t>(inTensorNum);
    graphParam_.outTensorNum = static_cast<uint32_t>(outTensorNum);
    return *this;
}

GraphOperationBuilder &GraphOperationBuilder::AddOperation(OperationWrapper &opWrapper,
                                                           const std::vector<std::string> &inTensorNames,
                                                           const std::vector<std::string> &outTensorNames)
{
    atb::Operation *rawPtr = opWrapper.ReleaseOperation();
    if (rawPtr == nullptr) {
        throw std::runtime_error("add operation is null.");
    }
    Node node;
    node.operation = rawPtr;
    node.inTensorIds.resize(0);
    node.outTensorIds.resize(0);
    node.inTensorReshapeFuncs.resize(0);
    for (const std::string &inTensorName : inTensorNames) {
        node.inTensorIds.push_back(GetTensorId(inTensorName));
        if (internalTensorNum_ > MAX_TENSOR_NUM) {
            throw std::runtime_error("internalTensor num should be <= 256, but get " +
                                     std::to_string(internalTensorNum_));
        }
        if (reshapedTensorIds_.find(inTensorName) != reshapedTensorIds_.end()) {
            node.inTensorReshapeFuncs.push_back(reshapedTensorIds_[inTensorName].second);
        } else {
            node.inTensorReshapeFuncs.push_back(nullptr);
        }
    }
    for (const std::string &outTensorName : outTensorNames) {
        node.outTensorIds.push_back(GetTensorId(outTensorName));
        if (internalTensorNum_ > MAX_TENSOR_NUM) {
            throw std::runtime_error("internalTensor num should be <= 256, but get " +
                                     std::to_string(internalTensorNum_));
        }
    }
    graphParam_.nodes.push_back(node);
    return *this;
}

GraphOperationBuilder &GraphOperationBuilder::Reshape(const std::string &srcTensorName,
                                                      const ReshapeHandler &reshapeHandler,
                                                      const std::string &reshapedTensorName)
{
    atb::ReshapeFunc reshapeFunc = [reshapeHandler](const Dims &oldShape, Dims &newShape) {
        std::vector<int64_t> oldShapeVec(oldShape.dimNum);
        for (size_t i = 0; i < oldShape.dimNum; ++i) {
            oldShapeVec[i] = oldShape.dims[i];
        }
        std::vector<int64_t> newShapeVec = reshapeHandler(oldShapeVec);
        ATB_LOG(INFO) << "oldShapeVec: " << oldShapeVec << ", newShapeVec: " << newShapeVec;
        newShape.dimNum = newShapeVec.size();
        for (uint64_t i = 0; i < newShape.dimNum; ++i) {
            newShape.dims[i] = newShapeVec[i];
        }
    };
    reshapedTensorIds_[reshapedTensorName] = {GetTensorId(srcTensorName), reshapeFunc};
    return *this;
}

OperationWrapper GraphOperationBuilder::Build()
{
    graphParam_.internalTensorNum = internalTensorNum_;
    return OperationWrapper(graphParam_);
}

uint32_t GraphOperationBuilder::GetTensorId(const std::string &tensorName)
{
    if (inTensorIds_.find(tensorName) != inTensorIds_.end()) {
        return inTensorIds_[tensorName];
    } else if (outTensorIds_.find(tensorName) != outTensorIds_.end()) {
        return outTensorIds_[tensorName];
    } else if (reshapedTensorIds_.find(tensorName) != reshapedTensorIds_.end()) {
        return reshapedTensorIds_[tensorName].first;
    } else if (internalTensorIds_.find(tensorName) != internalTensorIds_.end()) {
        return internalTensorIds_[tensorName];
    } else {
        uint32_t internalTensorId = inTensorIds_.size() + outTensorIds_.size() + internalTensorNum_++;
        internalTensorIds_[tensorName] = internalTensorId;
        return internalTensorId;
    }
}
} // namespace TorchAtb
