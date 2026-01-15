/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
#include "atb/operation/operation_base.h"
#include "plugin_ops/plugin_aclnn_operations/aclnn_gelu_operation.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"

using namespace atb;


static void CreateMiniGraphOperation(atb::GraphParam &opGraph, atb::Operation **operation)
{
	// 构子图流程
    opGraph.inTensorNum = 2;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 2;
    opGraph.nodes.resize(3);

    size_t nodeId = 0;
    atb::Node &addNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode2 = opGraph.nodes.at(nodeId++);
    atb::Node &addNode3 = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(addParam, &addNode.operation);
    addNode.inTensorIds = {0, 1};
    addNode.outTensorIds = {3};

    atb::infer::ElewiseParam addParam2;
    addParam2.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(addParam2, &addNode2.operation);
    addNode2.inTensorIds = {3, 1};
    addNode2.outTensorIds = {4};

    atb::infer::ElewiseParam addParam3;
    addParam3.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam3, &addNode3.operation);
    addNode3.inTensorIds = {4, 1};
    addNode3.outTensorIds = {2};

    atb::CreateOperation(opGraph, operation);
}

static void CreateGraphOperationForMultiStream(atb::GraphParam &opGraph, atb::Operation **operation)
{
    // 构大图流程
    opGraph.inTensorNum = 2;
    opGraph.outTensorNum = 2;
    opGraph.internalTensorNum = 2;
    opGraph.nodes.resize(4);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode2 = opGraph.nodes.at(nodeId++);
    atb::Node &graphNode = opGraph.nodes.at(nodeId++);
    atb::Node &mulNode1 = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::CreateOperation(mulParam, &mulNode.operation);
    mulNode.inTensorIds = {0, 1};
    mulNode.outTensorIds = {3};

    atb::infer::ElewiseParam addParam2;
    addParam2.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(addParam2, &addNode2.operation);
    addNode2.inTensorIds = {0, 1};
    addNode2.outTensorIds = {4};
    SetExecuteStreamId(addNode2.operation, 1);

    atb::GraphParam graphParam;
    CreateMiniGraphOperation(graphParam, &graphNode.operation);
    graphNode.inTensorIds = {4, 1};
    graphNode.outTensorIds = {5};
    SetExecuteStreamId(graphNode.operation, 1);

    atb::CreateOperation(mulParam, &mulNode1.operation);
    mulNode1.inTensorIds = {5, 1};
    mulNode1.outTensorIds = {2};
    SetExecuteStreamId(mulNode1.operation, 1);

    atb::CreateOperation(opGraph, operation);
}

static void CreateInTensorDescs(atb::SVector<atb::TensorDesc> &intensorDescs) 
{
    for (size_t i = 0; i < intensorDescs.size(); i++) {
        intensorDescs.at(i).dtype = ACL_FLOAT16;
        intensorDescs.at(i).format = ACL_FORMAT_ND;
        intensorDescs.at(i).shape.dimNum = 2;
        intensorDescs.at(i).shape.dims[0] = 2;
        intensorDescs.at(i).shape.dims[1] = 2;
    }
}

static void CreateInTensors(atb::SVector<atb::Tensor> &inTensors, atb::SVector<atb::TensorDesc> &intensorDescs)
{
    std::vector<char> zeroData(8, 0); // 一段全0的hostBuffer
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensors.at(i).desc = intensorDescs.at(i);
        inTensors.at(i).dataSize = atb::Utils::GetTensorSize(inTensors.at(i));
        int ret = aclrtMalloc(&inTensors.at(i).deviceData, inTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST); // 分配NPU内存
        if (ret != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
        ret = aclrtMemcpy(inTensors.at(i).deviceData, inTensors.at(i).dataSize, zeroData.data(), zeroData.size(), ACL_MEMCPY_HOST_TO_DEVICE); //拷贝CPU内存到NPU侧
    }
}

// 设置各个outtensor并且为outtensor分配内存空间，同intensor设置
static void CreateOutTensors(atb::SVector<atb::Tensor> &outTensors, atb::SVector<atb::TensorDesc> &outtensorDescs)
{
    for (size_t i = 0; i < outTensors.size(); i++) {
        outTensors.at(i).desc = outtensorDescs.at(i);
        outTensors.at(i).dataSize = atb::Utils::GetTensorSize(outTensors.at(i));
        int ret = aclrtMalloc(&outTensors.at(i).deviceData, outTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }
}

static void CreateGraphOperationWithEvent(atb::GraphParam &opGraph, atb::Operation **operation, aclrtEvent event)
{
    opGraph.inTensorNum = 2;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 2;
    opGraph.nodes.resize(5);

    size_t nodeId = 0;
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::Node &waitNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);
    atb::Node &graphNode = opGraph.nodes.at(nodeId++);
    atb::Node &recordNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::CreateOperation(mulParam, &mulNode.operation);
    mulNode.inTensorIds = {0, 1};
    mulNode.outTensorIds = {3};

    atb::common::EventParam waitParam;
    waitParam.event = event;
    waitParam.operatorType = atb::common::EventParam::OperatorType::WAIT;
    atb::CreateOperation(waitParam, &waitNode.operation);

    atb::GraphParam graphParam;
    CreateMiniGraphOperation(graphParam, &graphNode.operation);
    graphNode.inTensorIds = {3, 4};
    graphNode.outTensorIds = {2};

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(addParam, &addNode.operation);
    addNode.inTensorIds = {0, 1};
    addNode.outTensorIds = {4};
    SetExecuteStreamId(addNode.operation, 1);

    atb::common::EventParam recordParam;
    recordParam.event = event;
    recordParam.operatorType = atb::common::EventParam::OperatorType::RECORD;
    atb::CreateOperation(recordParam, &recordNode.operation);
    SetExecuteStreamId(recordNode.operation, 1);

    atb::CreateOperation(opGraph, operation);
}

TEST(TestSingleGraphMultistream, TestNormalGraphOp)
{
    // aclInit(nullptr);
    /*
        特性：单图多流接口SetExecuteStreamId，测试内置算子 + 子图嵌套场景
        测试场景：mul->add->graph 前面一个node执行流为stream0，后两个node执行流为stream1
        预期结果：streamId符合预期
    */
    // 创建Graph1
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream stream1;
    aclrtCreateStream(&stream1);
    aclrtStream stream2;
    aclrtCreateStream(&stream2);
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    std::vector<aclrtStream> streams = {stream1, stream2};
    context->SetExecuteStreams(streams);

    atb::Operation *graphOp = nullptr;
    atb::GraphParam graphParam1;
    CreateGraphOperationForMultiStream(graphParam1, &graphOp);

    // 准备输入输出tensor
    atb::VariantPack pack;
    atb::SVector<atb::TensorDesc> intensorDescs1;
    atb::SVector<atb::TensorDesc> outtensorDescs1;

    uint32_t inTensorNum = 2;
    uint32_t outTensorNum = 2;
    pack.inTensors.resize(inTensorNum);
    intensorDescs1.resize(inTensorNum);

    CreateInTensorDescs(intensorDescs1);
    CreateInTensors(pack.inTensors, intensorDescs1);

    outtensorDescs1.resize(outTensorNum);
    outtensorDescs1.at(0).dtype = ACL_FLOAT16;
    outtensorDescs1.at(0).format = ACL_FORMAT_ND;
    outtensorDescs1.at(0).shape.dimNum = 2;
    outtensorDescs1.at(0).shape.dims[0] = 2;
    outtensorDescs1.at(0).shape.dims[1] = 2;
    outtensorDescs1.at(1).dtype = ACL_FLOAT16;
    outtensorDescs1.at(1).format = ACL_FORMAT_ND;
    outtensorDescs1.at(1).shape.dimNum = 2;
    outtensorDescs1.at(1).shape.dims[0] = 2;
    outtensorDescs1.at(1).shape.dims[1] = 2;
    pack.outTensors.resize(outTensorNum);
    CreateOutTensors(pack.outTensors, outtensorDescs1);

    // Setup
    uint64_t workspaceSize = 0;
    graphOp->Setup(pack, workspaceSize, context);
    void *workSpace = nullptr;
    int ret1 = 0;
    if (workspaceSize != 0) {
        ret1 = aclrtMalloc(&workSpace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret1 != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }

    //Execute
    atb::Status st1 = graphOp->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);

    atb::Status st = (st1 == atb::NO_ERROR) ? atb::NO_ERROR : atb::ERROR_INVALID_GRAPH;

    //流同步
    ret1 = aclrtSynchronizeStream(stream1);
    EXPECT_EQ(ret1, atb::NO_ERROR);
    if (ret1 != 0) {
        std::cout << "sync error!";
        exit(0);
    }

    ret1 = aclrtSynchronizeStream(stream2);
    EXPECT_EQ(ret1, atb::NO_ERROR);
    if (ret1 != 0) {
        std::cout << "sync error!";
        exit(0);
    }

    uint32_t nodeStreamId0 = GetExecuteStreamId(graphParam1.nodes.at(0).operation);
    uint32_t nodeStreamId1 = GetExecuteStreamId(graphParam1.nodes.at(1).operation);
    uint32_t nodeStreamId2 = GetExecuteStreamId(graphParam1.nodes.at(2).operation);
    uint32_t nodeStreamId3 = GetExecuteStreamId(graphParam1.nodes.at(3).operation);

    EXPECT_EQ(nodeStreamId0, 0);
    EXPECT_EQ(nodeStreamId1, 1);
    EXPECT_EQ(nodeStreamId2, 1);
    EXPECT_EQ(nodeStreamId3, 1);

    // 资源释放
    atb::DestroyOperation(graphOp);
    atb::DestroyContext(context);
    for (size_t i = 0; i < pack.inTensors.size(); i++) {
        aclrtFree(pack.inTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack.outTensors.size(); i++) {
        aclrtFree(pack.outTensors.at(i).deviceData);
    }
    aclrtFree(workSpace);
    aclrtDestroyStream(stream1);
    aclrtResetDevice(deviceId);
    
    ASSERT_EQ(st, atb::NO_ERROR);
    // aclFinalize();
}

TEST(TestSingleGraphMultistream, TestGraphOpWithEvent)
{
    // aclInit(nullptr);
    // 创建Graph1
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream stream1;
    aclrtCreateStream(&stream1);
    aclrtStream stream2;
    aclrtCreateStream(&stream2);
    atb::Context *context = nullptr;
    aclrtEvent event;
    aclrtCreateEventWithFlag(&event, ACL_EVENT_SYNC);
    atb::CreateContext(&context);
    std::vector<aclrtStream> streams = {stream1, stream2};
    context->SetExecuteStreams(streams);

    atb::Operation *graphOp = nullptr;
    atb::GraphParam graphParam1;
    CreateGraphOperationWithEvent(graphParam1, &graphOp, event);

    // 准备输入输出tensor
    atb::VariantPack pack;
    atb::SVector<atb::TensorDesc> intensorDescs1;
    atb::SVector<atb::TensorDesc> outtensorDescs1;

    uint32_t inTensorNum = 2;
    uint32_t outTensorNum = 1;
    pack.inTensors.resize(inTensorNum);
    intensorDescs1.resize(inTensorNum);

    CreateInTensorDescs(intensorDescs1);
    CreateInTensors(pack.inTensors, intensorDescs1);

    outtensorDescs1.resize(outTensorNum);
    outtensorDescs1.at(0).dtype = ACL_FLOAT16;
    outtensorDescs1.at(0).format = ACL_FORMAT_ND;
    outtensorDescs1.at(0).shape.dimNum = 2;
    outtensorDescs1.at(0).shape.dims[0] = 2;
    outtensorDescs1.at(0).shape.dims[1] = 2;
    pack.outTensors.resize(outTensorNum);
    CreateOutTensors(pack.outTensors, outtensorDescs1);

    // Setup
    uint64_t workspaceSize = 0;
    graphOp->Setup(pack, workspaceSize, context);
    void *workSpace = nullptr;
    int ret1 = 0;
    if (workspaceSize != 0) {
        ret1 = aclrtMalloc(&workSpace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret1 != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }

    //Execute
    atb::Status st1 = graphOp->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);

    atb::Status st = (st1 == atb::NO_ERROR) ? atb::NO_ERROR : atb::ERROR_INVALID_GRAPH;

    //流同步
    ret1 = aclrtSynchronizeStream(stream1);
    EXPECT_EQ(ret1, atb::NO_ERROR);
    if (ret1 != 0) {
        std::cout << "sync error!";
        exit(0);
    }

    ret1 = aclrtSynchronizeStream(stream2);
    EXPECT_EQ(ret1, atb::NO_ERROR);
    if (ret1 != 0) {
        std::cout << "sync error!";
        exit(0);
    }

    uint32_t nodeStreamId0 = GetExecuteStreamId(graphParam1.nodes.at(0).operation);
    uint32_t nodeStreamId1 = GetExecuteStreamId(graphParam1.nodes.at(1).operation);
    uint32_t nodeStreamId2 = GetExecuteStreamId(graphParam1.nodes.at(2).operation);
    uint32_t nodeStreamId3 = GetExecuteStreamId(graphParam1.nodes.at(3).operation);
    uint32_t nodeStreamId4 = GetExecuteStreamId(graphParam1.nodes.at(4).operation);

    EXPECT_EQ(nodeStreamId0, 0);
    EXPECT_EQ(nodeStreamId1, 0);
    EXPECT_EQ(nodeStreamId2, 1);
    EXPECT_EQ(nodeStreamId3, 0);
    EXPECT_EQ(nodeStreamId4, 1);

    // 资源释放
    atb::DestroyOperation(graphOp);
    atb::DestroyContext(context);
    for (size_t i = 0; i < pack.inTensors.size(); i++) {
        aclrtFree(pack.inTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack.outTensors.size(); i++) {
        aclrtFree(pack.outTensors.at(i).deviceData);
    }
    aclrtFree(workSpace);
    aclrtDestroyEvent(event);
    aclrtDestroyStream(stream1);
    aclrtResetDevice(deviceId);
    
    ASSERT_EQ(st, atb::NO_ERROR);
    // aclFinalize();
}

TEST(TestSingleGraphMultistream, TestGetStreamByStreamId)
{
    // 测试SetExecuteStreamId接口与GetExecuteStreamId接口，确认设置id后取得的stream是否正确
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream stream1;
    aclrtCreateStream(&stream1);
    aclrtStream stream2;
    aclrtCreateStream(&stream2);
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    std::vector<aclrtStream> streams = {stream1, stream2};
    context->SetExecuteStreams(streams);

    // 内置单算子校验
    Operation *mulNode = nullptr;
    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::CreateOperation(mulParam, &mulNode);
    OperationBase *opBase = dynamic_cast<OperationBase*>(mulNode);
    aclrtStream examStream1 = opBase->GetExecuteStream(context);
    EXPECT_EQ(examStream1, stream1);

    // 内置图算子校验
    GraphParam opGraph;
    Operation *graphOperation = nullptr;
    CreateMiniGraphOperation(opGraph, &graphOperation);
    Status ret = SetExecuteStreamId(graphOperation, 1);
    EXPECT_EQ(ret, 0);
    opBase = dynamic_cast<OperationBase*>(graphOperation);
    aclrtStream examStream2 = opBase->GetExecuteStream(context);
    EXPECT_EQ(examStream2, stream2);


    Operation *mulNode1 = nullptr;
    atb::CreateOperation(mulParam, &mulNode1);
    ret = SetExecuteStreamId(mulNode1, 1);
    opBase = dynamic_cast<OperationBase*>(mulNode1);
    aclrtStream examStream3 = opBase->GetExecuteStream(context);
    EXPECT_EQ(examStream3, stream2);
}

TEST(TestSingleGraphMultistream, TestContextInterface)
{
    // 测试context提供的SetExecuteStreams接口与GetExecuteStreams接口
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream stream1;
    aclrtCreateStream(&stream1);
    aclrtStream stream2;
    aclrtCreateStream(&stream2);
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    std::vector<aclrtStream> streams = {stream1, stream2};
    context->SetExecuteStreams(streams);

    std::vector<aclrtStream> getStreams = context->GetExecuteStreams();
    EXPECT_EQ(getStreams.size(), streams.size());
    EXPECT_EQ(getStreams.at(0), streams.at(0));
    EXPECT_EQ(getStreams.at(1), streams.at(1));
}

