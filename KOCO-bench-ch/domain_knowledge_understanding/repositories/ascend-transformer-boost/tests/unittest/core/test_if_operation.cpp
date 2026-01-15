/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <atb/utils/log.h>
#include <atb/utils.h>
#include "test_utils/test_common.h"
#include "atb/operation.h"
#include "atb/utils/tensor_util.h"
#include "test_utils/operation_test.h"
#include "atb/utils/operation_util.h"
#include "atb/operation_infra.h"
#include "atb/operation/if_operation.h"
#include "atb/operation/operation_base.h"
#include "plugin_ops/plugin_aclnn_operations/aclnn_gelu_operation.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"

using namespace atb;
using namespace Mki;

bool CondFunction(void *userData)
{
    if (userData != nullptr) {
        int *data = static_cast<int *>(userData);
        return (*data > 10);
    }
    return false;
}

TEST(TestIfOperation, IfOpTest)
{
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        GTEST_SKIP() << "This test case only support 910B";
    }

    atb::Operation *operationA;
    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &operationA);
    EXPECT_EQ(status1, 0);

    atb::Operation *operationB;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &operationB);
    EXPECT_EQ(status2, 0);

    atb::Operation *ifOperation;
    atb::common::IfCondParam opCond;
    std::unique_ptr<int> data = std::make_unique<int>(15);
    opCond.handle = CondFunction;
    opCond.userData = data.get();
    opCond.opA = operationA;
    opCond.opB = operationB;
    atb::Status status3 = CreateOperation(opCond, &ifOperation);
    EXPECT_EQ(status3, 0);

    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {{Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
                                                      {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}}};
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);

    OperationTest opTest;
    atb::Status status4 = opTest.Run(ifOperation, inTensorDescs);
    EXPECT_EQ(status4, 0);

    atb::DestroyOperation(ifOperation);
    atb::DestroyOperation(operationA);
    atb::DestroyOperation(operationB);
}

TEST(TestIfOperation, IfGraphOpTest)
{
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        GTEST_SKIP() << "This test case only support 910B";
    }

    enum InTensorId {
        IN_TENSOR_A = 0,
        IN_TENSOR_B,
        LAYER_OUT,
        ADD_OUT
    };

    // graph with add+sin
    atb::Operation *operationA;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 2;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &addNode = opGraph.nodes.at(nodeId++);
    atb::Node &sinNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status1 = atb::CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status1, 0);
    addNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    addNode.outTensorIds = {ADD_OUT};

    atb::infer::ElewiseParam sinParam;
    sinParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_SIN;
    atb::Status status2 = CreateOperation(sinParam, &sinNode.operation);
    EXPECT_EQ(status2, 0);
    sinNode.inTensorIds = {ADD_OUT};
    sinNode.outTensorIds = {LAYER_OUT};

    atb::Status status3 = atb::CreateOperation(opGraph, &operationA);
    EXPECT_EQ(status3, 0);

    atb::Operation *operationB;
    // reuse addParam defined previously
    atb::Status status4 = CreateOperation(addParam, &operationB);
    EXPECT_EQ(status4, 0);

    atb::Operation *ifOperation;
    atb::common::IfCondParam opCond;
    std::unique_ptr<int> data = std::make_unique<int>(15);
    opCond.handle = CondFunction;
    opCond.userData = data.get();
    opCond.opA = operationA;
    opCond.opB = operationB;
    atb::Status status5 = CreateOperation(opCond, &ifOperation);
    EXPECT_EQ(status5, 0);

    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {{Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
                                                      {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}}};
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);

    OperationTest opTest;
    atb::Status status6 = opTest.Run(ifOperation, inTensorDescs);
    EXPECT_EQ(status6, 0);

    atb::DestroyOperation(ifOperation);
    atb::DestroyOperation(operationA);
    atb::DestroyOperation(operationB);
}

TEST(TestIfOperation, MultiStreamTest)
{
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        GTEST_SKIP() << "This test case only support 910B";
    }

    atb::Operation *graphOp;
    atb::GraphParam graphParam;
    graphParam.inTensorNum = 4;
    graphParam.outTensorNum = 2;
    graphParam.internalTensorNum = 0;
    graphParam.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &ifNode = graphParam.nodes.at(nodeId++);
    atb::Node &addNode = graphParam.nodes.at(nodeId++);

    atb::Operation *operationA;
    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &operationA);
    EXPECT_EQ(status1, 0);

    atb::Operation *operationB;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &operationB);
    EXPECT_EQ(status2, 0);

    atb::common::IfCondParam opCond;
    std::unique_ptr<int> data = std::make_unique<int>(15);
    opCond.handle = CondFunction;
    opCond.userData = data.get();
    opCond.opA = operationA;
    opCond.opB = operationB;
    atb::Status status3 = CreateOperation(opCond, &ifNode.operation);
    EXPECT_EQ(status3, 0);
    ifNode.inTensorIds = {0, 1};
    ifNode.outTensorIds = {4};
    SetExecuteStreamId(ifNode.operation, 1);

    // reuse add param
    atb::Status status4 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status4, 0);
    addNode.inTensorIds = {2, 3};
    addNode.outTensorIds = {5};

    atb::Status status5 = CreateOperation(graphParam, &graphOp);
    EXPECT_EQ(status5, 0);

    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {{Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
                                                      {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
                                                      {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
                                                      {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}}};
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);

    OperationTest opTest;
    atb::Status status6 = opTest.Run(graphOp, inTensorDescs);
    EXPECT_EQ(status6, 0);

    atb::DestroyOperation(graphOp);
    atb::DestroyOperation(operationA);
    atb::DestroyOperation(operationB);
}