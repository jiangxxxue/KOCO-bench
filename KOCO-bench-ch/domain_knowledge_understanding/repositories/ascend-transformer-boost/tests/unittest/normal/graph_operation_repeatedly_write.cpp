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
#include "atb/operation/operation_base.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"

using namespace atb;

TEST(TestRepeatWrite, graphTest)
{
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        GTEST_SKIP() << "This test case only support 910B";
    }

    atb::Operation *operation;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 4;
    opGraph.outTensorNum = 2;
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(3);

    size_t nodeId = 0;
    atb::Node &node0 = opGraph.nodes.at(nodeId++);
    atb::Node &node1 = opGraph.nodes.at(nodeId++);
    atb::Node &node2 = opGraph.nodes.at(nodeId++);
    
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status1 = atb::CreateOperation(addParam, &node0.operation);
    EXPECT_EQ(status1, 0);
    node0.inTensorIds = {0, 1};
    node0.outTensorIds = {4};

    atb::infer::LinearParam linearParam;
    linearParam.hasBias = false;
    atb::Status status2 = atb::CreateOperation(linearParam, &node1.operation);
    EXPECT_EQ(status2, 0);
    node1.inTensorIds = {2, 3};
    node1.outTensorIds = {6};

    atb::infer::SplitParam splitParam;
    atb::Status status3 = atb::CreateOperation(splitParam, &node2.operation);
    EXPECT_EQ(status3, 0);
    node2.inTensorIds = {6};
    node2.outTensorIds = {4, 5};

    atb::Status status4 = atb::CreateOperation(opGraph, &operation);
    EXPECT_EQ(status4, 0);

    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {{Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
                                                      {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
                                                      {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}},
                                                      {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}}};
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);

    OperationTest opTest;
    atb::Status status5 = opTest.Run(operation, inTensorDescs);
    DestroyOperation(operation);
    EXPECT_EQ(status5, 0);
}