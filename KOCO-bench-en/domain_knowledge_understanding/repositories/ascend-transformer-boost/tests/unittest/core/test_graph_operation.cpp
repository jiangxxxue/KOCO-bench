/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "atb/operation.h"
#include "atb/infer_op_params.h"
#include "atb/runner/graph_runner.h"
#include "test_utils/operation_test.h"
#include <vector>
#include <string>
#include <string.h>
#include <cstdlib>
#include <gtest/gtest.h>
#include "atb/utils/tensor_util.h"
#include <atb/utils/log.h>
#include "atb/utils/probe.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cpp-stub/src/stub.h>
#include "atb/utils/singleton.h"
#include "atb/utils/config.h"

using namespace atb;
using namespace Mki;

enum InTensorId {
    IN_TENSOR_A = 0,
    IN_TENSOR_B,
    IN_TENSOR_C,
    LAYER_OUT,
    MUL_OUT
};

void SimpleReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum;
    newShape.dims[0] = oldShape.dims[0];
    newShape.dims[1] = oldShape.dims[1];
}

void WrongReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum;
    newShape.dims[0] = oldShape.dims[0] + 1;
    newShape.dims[1] = oldShape.dims[1];
}

TEST(TestGraphOperation, GraphOpTest)
{
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    mulNode.outTensorIds = {MUL_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {LAYER_OUT};

    atb::Status status3 = atb::CreateOperation(opGraph, &operation);
    EXPECT_EQ(status3, 0);

    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {{Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}}, {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}}};
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);

    OperationTest opTest;
    atb::Status status4 = opTest.Run(operation, inTensorDescs);
    DestroyOperation(operation);
    EXPECT_EQ(status4, 0);
}

TEST(TestGraphOperation, GraphOpTestWithInfershape)
{
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    mulNode.outTensorIds = {MUL_OUT};
    mulNode.inTensorReshapeFuncs = {&SimpleReshapeFunc, &SimpleReshapeFunc};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {LAYER_OUT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::Status status3 = atb::CreateOperation(opGraph, &operation);
    EXPECT_EQ(status3, 0);

    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {{Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}}, {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}}};
    atb::SVector<atb::TensorDesc> inTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);

    OperationTest opTest;
    atb::Status status4 = opTest.Run(operation, inTensorDescs);
    DestroyOperation(operation);
    EXPECT_EQ(status4, 0);
}

TEST(TestGraphOperation, GraphOpTestWrongInfershape1)
{
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    mulNode.outTensorIds = {MUL_OUT};
    mulNode.inTensorReshapeFuncs = {&SimpleReshapeFunc, &SimpleReshapeFunc};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {LAYER_OUT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::ERROR_INVALID_GRAPH;
    };

    atb::Status status3 = atb::CreateOperation(opGraph, &operation);
    DestroyOperation(operation);
    ASSERT_EQ(status3, 0);
}

TEST(TestGraphOperation, GraphOpTestWrongInfershape2)
{
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    mulNode.outTensorIds = {MUL_OUT};
    mulNode.inTensorReshapeFuncs = {&SimpleReshapeFunc, &SimpleReshapeFunc};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {LAYER_OUT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        return atb::NO_ERROR;
    };

    atb::Status status3 = atb::CreateOperation(opGraph, &operation);
    EXPECT_EQ(status3, 0);

    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {{Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, {2, 2}}, {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}}};
    atb::SVector<atb::TensorDesc> inTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);

    OperationTest opTest;
    atb::Status status4 = opTest.Run(operation, inTensorDescs);
    DestroyOperation(operation);
    EXPECT_NE(status4, 0);
}

TEST(TestGraphOperation, WrongNodeInfershape)
{
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    mulNode.outTensorIds = {MUL_OUT};
    mulNode.inTensorReshapeFuncs = {&WrongReshapeFunc, &WrongReshapeFunc};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {LAYER_OUT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::Status status3 = atb::CreateOperation(opGraph, &operation);
    EXPECT_EQ(status3, 0);

    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {{Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, {2, 2}},
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, {2, 2}}, {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, {2, 2}}};
    atb::SVector<atb::TensorDesc> inTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);

    OperationTest opTest;
    atb::Status status4 = opTest.Run(operation, inTensorDescs);
    DestroyOperation(operation);
    EXPECT_NE(status4, 0);
}

TEST(TestGraphOperation, InvalidInputTensors)
{
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    mulNode.outTensorIds = {MUL_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {LAYER_OUT};

    atb::Status status3 = atb::CreateOperation(opGraph, &operation);
    EXPECT_EQ(status3, 0);

    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {{Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, {2, 2}},
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, {2, 2}},
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, {2, 2}},
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, {2, 2}}};
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);

    OperationTest opTest;
    atb::Status status4 = opTest.Run(operation, inTensorDescs);
    DestroyOperation(operation);
    EXPECT_NE(status4, 0);
}

TEST(TestGraphOperation, WrongOpParam1)
{
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.name = "&@!!1123";
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    mulNode.outTensorIds = {MUL_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {LAYER_OUT};

    atb::Status status3 = atb::CreateOperation(opGraph, &operation);
    EXPECT_NE(status3, 0);
}

TEST(TestGraphOperation, WrongOpParam2)
{
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 50;
    opGraph.outTensorNum = 50;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    mulNode.outTensorIds = {MUL_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {LAYER_OUT};

    atb::Status status3 = atb::CreateOperation(opGraph, &operation);
    EXPECT_NE(status3, 0);
}

TEST(TestGraphOperation, WrongOpParam3)
{
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(1025); // too lagre size

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    mulNode.outTensorIds = {MUL_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {LAYER_OUT};

    atb::Status status3 = atb::CreateOperation(opGraph, &operation);
    EXPECT_NE(status3, 0);
}

TEST(TestGraphOperation, WrongOpParam4)
{
    // unused intensor
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 2;
    opGraph.internalTensorNum = 2;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    mulNode.outTensorIds = {MUL_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {LAYER_OUT};

    atb::Status status3 = atb::CreateOperation(opGraph, &operation);
    EXPECT_NE(status3, 0);
}

TEST(TestGraphOperation, nodeWithoutOp)
{
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    mulNode.outTensorIds = {MUL_OUT};
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {LAYER_OUT};

    atb::Status status3 = atb::CreateOperation(opGraph, &operation);
    EXPECT_NE(status3, 0);
}

TEST(TestGraphOperation, InValidNode1)
{
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B, IN_TENSOR_C};
    mulNode.outTensorIds = {MUL_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {LAYER_OUT};

    atb::Status status3 = atb::CreateOperation(opGraph, &operation);
    EXPECT_NE(status3, 0);
}

TEST(TestGraphOperation, InValidNode2)
{
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    mulNode.outTensorIds = {MUL_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, 10}; // outrange id: 4
    addNode.outTensorIds = {LAYER_OUT};

    atb::Status status3 = atb::CreateOperation(opGraph, &operation);
    EXPECT_NE(status3, 0);
}

TEST(TestGraphOperation, InValidNode3)
{
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {}; // empty intensor
    mulNode.outTensorIds = {MUL_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {LAYER_OUT};

    atb::Status status3 = atb::CreateOperation(opGraph, &operation);
    EXPECT_NE(status3, 0);
}

TEST(TestGraphOperation, InValidNode4)
{
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    mulNode.outTensorIds = {MUL_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {LAYER_OUT, LAYER_OUT}; // wrong outTensor size

    atb::Status status3 = atb::CreateOperation(opGraph, &operation);
    EXPECT_NE(status3, 0);
}

TEST(TestGraphOperation, InValidNode5)
{
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    mulNode.outTensorIds = {MUL_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {10}; // wrong outTensor id: 2

    atb::Status status3 = atb::CreateOperation(opGraph, &operation);
    EXPECT_NE(status3, 0);
}

TEST(TestGraphOperation, WrongOp)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    mulNode.outTensorIds = {MUL_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {2}; // wrong outTensor id: 2

    atb::Status status3 = atb::CreateOperation(opGraph, nullptr);
    EXPECT_NE(status3, 0);
}

bool TestGraphOperationReportOperationGraphEnableStub()
{
    return true;
}

void TestGraphOperationReportOperationGraphStub(const std::string &opName, const std::string &graph)
{
    std::string dirPath = "/home/test";
    std::string filePath = dirPath + "/testGraph.json";
    nlohmann::json result;
    result = nlohmann::json::parse(graph);
    std::ofstream ofs(filePath);
    ofs << result;
    ofs.close();
}

TEST(TestGraphOperation, jsonTest)
{
    Stub stub;
    stub.set(ADDR(Probe, ReportOperationGraphEnable), TestGraphOperationReportOperationGraphEnableStub);
    stub.set(ADDR(Probe, ReportOperationGraph), TestGraphOperationReportOperationGraphStub);
    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::Status status1 = atb::CreateOperation(mulParam, &mulNode.operation);
    EXPECT_EQ(status1, 0);
    mulNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    mulNode.outTensorIds = {MUL_OUT};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status2 = CreateOperation(addParam, &addNode.operation);
    EXPECT_EQ(status2, 0);
    addNode.inTensorIds = {MUL_OUT, IN_TENSOR_C};
    addNode.outTensorIds = {LAYER_OUT};

    atb::GraphParam opGraph1;
    opGraph1.inTensorNum = 3;
    opGraph1.outTensorNum = 1;
    opGraph1.internalTensorNum = 1;
    opGraph1.nodes.resize(2);
    nodeId = 0;
    atb::Node &graphNode = opGraph1.nodes.at(nodeId++);
    atb::Node &activateNode = opGraph1.nodes.at(nodeId++);

    atb::Status status3 = atb::CreateOperation(opGraph, &graphNode.operation);
    EXPECT_EQ(status3, 0);
    graphNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B, IN_TENSOR_C};
    graphNode.outTensorIds = {MUL_OUT};

    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    atb::Status status4 = atb::CreateOperation(activationParam, &activateNode.operation);
    EXPECT_EQ(status4, 0);
    activateNode.inTensorIds = {MUL_OUT};
    activateNode.outTensorIds = {LAYER_OUT};

    atb::Status status5 = atb::CreateOperation(opGraph1, &operation);
    EXPECT_EQ(status5, 0);

    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {{Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}}, {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}}};
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);

    if (GetSingleton<atb::Config>().Is310B()) {
        return;
    }
    OperationTest opTest;
    atb::Status status6 = opTest.Run(operation, inTensorDescs);
    DestroyOperation(operation);
    EXPECT_EQ(status6, 0);
}