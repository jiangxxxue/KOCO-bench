/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <map>

#include "atb/graph_op_builder.h"
#include "atb/utils/log.h"
#include "atb/operation.h"
#include "atb/types.h"

namespace atb {
static const size_t MAX_TENSOR_NUM = 256;

GraphOpBuilder::GraphOpBuilder() {};

GraphOpBuilder::~GraphOpBuilder() {};

class GraphOpBuilderImpl : public GraphOpBuilder {
public:
    GraphOpBuilderImpl() {};
    ~GraphOpBuilderImpl() override {};
    Status Init(const std::string &opName, const InferShapeFunc &inferShapeFunc,
                const SVector<std::string> &inTensorNames, const SVector<std::string> &outTensorNames) override;
    Status Reshape(const std::string &srcTensorName, const ReshapeFunc &reshapeFunc,
                   const std::string &viewTensorName) override;
    Status AddOperation(Operation *operation, const SVector<std::string> &inTensorNames,
                        const SVector<std::string> &outTensorNames) override;
    Operation *Build() override;

private:
    uint32_t GetTensorId(const std::string &tensorName);

private:
    GraphParam graphParam_;
    uint32_t internalTensorNum_ = 0;
    std::map<std::string, uint32_t> inTensorIds_;
    std::map<std::string, uint32_t> outTensorIds_;
    std::map<std::string, uint32_t> internalTensorIds_;
    std::map<std::string, std::pair<uint32_t, ReshapeFunc>> viewTensorIds_; // key: viewTensorName
};

Status GraphOpBuilderImpl::Init(const std::string &opName, const InferShapeFunc &inferShapeFunc,
                                const SVector<std::string> &inTensorNames, const SVector<std::string> &outTensorNames)
{
    graphParam_.name = opName;
    graphParam_.inferShapeFunc = inferShapeFunc;
    uint32_t id = 0;
    size_t inTensorNum = inTensorNames.size();
    if (inTensorNum > MAX_TENSOR_NUM) {
        ATB_LOG(ERROR) << "inTensor num should be <= 256, but get " << inTensorNum;
        return ERROR_INVALID_IN_TENSOR_NUM;
    }
    for (const std::string &inTensorName : inTensorNames) {
        inTensorIds_[inTensorName] = id++;
    }
    size_t outTensorNum = outTensorNames.size();
    if (outTensorNum > MAX_TENSOR_NUM) {
        ATB_LOG(ERROR) << "outTensor num should be <= 256, but get " << outTensorNum;
        return ERROR_INVALID_IN_TENSOR_NUM;
    }
    for (const std::string &outTensorName : outTensorNames) {
        outTensorIds_[outTensorName] = id++;
    }
    graphParam_.inTensorNum = static_cast<uint32_t>(inTensorNum);
    graphParam_.outTensorNum = static_cast<uint32_t>(outTensorNum);
    return NO_ERROR;
}

Status GraphOpBuilderImpl::Reshape(const std::string &srcTensorName, const ReshapeFunc &reshapeFunc,
                                   const std::string &viewTensorName)
{
    viewTensorIds_[viewTensorName] = {GetTensorId(srcTensorName), reshapeFunc};
    return NO_ERROR;
}

Status GraphOpBuilderImpl::AddOperation(Operation *operation, const SVector<std::string> &inTensorNames,
                                        const SVector<std::string> &outTensorNames)
{
    if (operation == nullptr) {
        ATB_LOG(ERROR) << "operation can not be nullptr!";
        return ERROR_INVALID_PARAM;
    }
    Node node;
    node.operation = operation;
    node.inTensorIds.resize(0);
    node.outTensorIds.resize(0);
    node.inTensorReshapeFuncs.resize(0);
    for (const std::string &inTensorName : inTensorNames) {
        node.inTensorIds.push_back(GetTensorId(inTensorName));
        if (internalTensorNum_ > MAX_TENSOR_NUM) {
            ATB_LOG(ERROR) << "internalTensor num should be <= 256, but get " << internalTensorNum_;
            return ERROR_INVALID_IN_TENSOR_NUM;
        }
        if (viewTensorIds_.find(inTensorName) != viewTensorIds_.end()) {
            node.inTensorReshapeFuncs.push_back(viewTensorIds_[inTensorName].second);
        } else {
            node.inTensorReshapeFuncs.push_back(nullptr);
        }
    }
    for (const std::string &outTensorName : outTensorNames) {
        node.outTensorIds.push_back(GetTensorId(outTensorName));
        if (internalTensorNum_ > MAX_TENSOR_NUM) {
            ATB_LOG(ERROR) << "internalTensor num should be <= 256, but get " << internalTensorNum_;
            return ERROR_INVALID_IN_TENSOR_NUM;
        }
    }
    graphParam_.nodes.push_back(node);
    return NO_ERROR;
}

Operation *GraphOpBuilderImpl::Build()
{
    graphParam_.internalTensorNum = internalTensorNum_;
    Operation *graphOp = nullptr;
    Status st = CreateOperation(graphParam_, &graphOp);
    if (st != NO_ERROR) {
        for (size_t i = 0; i < graphParam_.nodes.size(); i++) {
            if (graphParam_.nodes.at(i).operation != nullptr) {
                DestroyOperation(graphParam_.nodes.at(i).operation);
                graphParam_.nodes.at(i).operation = nullptr;
            }
        }
        ATB_LOG(ERROR) << "create graph op fail";
    }
    return graphOp;
}

uint32_t GraphOpBuilderImpl::GetTensorId(const std::string &tensorName)
{
    if (inTensorIds_.find(tensorName) != inTensorIds_.end()) {
        return inTensorIds_[tensorName];
    } else if (outTensorIds_.find(tensorName) != outTensorIds_.end()) {
        return outTensorIds_[tensorName];
    } else if (viewTensorIds_.find(tensorName) != viewTensorIds_.end()) {
        return viewTensorIds_[tensorName].first;
    } else if (internalTensorIds_.find(tensorName) != internalTensorIds_.end()) {
        return internalTensorIds_[tensorName];
    } else {
        uint32_t internalTensorId = inTensorIds_.size() + outTensorIds_.size() + internalTensorNum_++;
        internalTensorIds_[tensorName] = internalTensorId;
        return internalTensorId;
    }
}

Status CreateGraphOpBuilder(GraphOpBuilder **builder)
{
    if (builder == nullptr) {
        ATB_LOG(ERROR) << "invalid param, builder is null";
        return ERROR_INVALID_PARAM;
    }

    *builder = new GraphOpBuilderImpl();
    return NO_ERROR;
}

Status DestroyGraphOpBuilder(GraphOpBuilder *builder)
{
    if (builder != nullptr) {
        delete builder;
    }

    return NO_ERROR;
}
} // namespace atb