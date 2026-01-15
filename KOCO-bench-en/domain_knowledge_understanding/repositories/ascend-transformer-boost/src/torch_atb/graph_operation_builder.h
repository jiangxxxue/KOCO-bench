/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TORCH_ATB_GRAPH_OPERATION_BUILDER_H
#define TORCH_ATB_GRAPH_OPERATION_BUILDER_H
#include "atb/atb_infer.h"
#include "operation_wrapper.h"

namespace TorchAtb {
using ReshapeHandler = std::function<std::vector<int64_t>(const std::vector<int64_t> &oldShape)>;

class GraphOperationBuilder {
public:
    explicit GraphOperationBuilder(const std::string &graphName);
    GraphOperationBuilder &SetInputOutput(const std::vector<std::string> &inTensorNames,
                                          const std::vector<std::string> &outTensorNames);
    GraphOperationBuilder &AddOperation(OperationWrapper &opWrapper, const std::vector<std::string> &inTensorNames,
                                        const std::vector<std::string> &outTensorNames);
    GraphOperationBuilder &Reshape(const std::string &srcTensorName, const ReshapeHandler &reshapeHandler,
                                   const std::string &reshapedTensorName);
    OperationWrapper Build();

private:
    uint32_t GetTensorId(const std::string &tensorName);

private:
    atb::GraphParam graphParam_;
    uint32_t internalTensorNum_ = 0;
    std::map<std::string, uint32_t> inTensorIds_;
    std::map<std::string, uint32_t> outTensorIds_;
    std::map<std::string, uint32_t> internalTensorIds_;
    std::map<std::string, std::pair<uint32_t, atb::ReshapeFunc>> reshapedTensorIds_;
};
} // namespace TorchAtb
#endif // TORCH_ATB_GRAPH_OPERATION_WRAPPER_H