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
#include "atb/common_op_params.h"
#include "atb/utils/tensor_util.h"
#include "test_utils/operation_test.h"
#include "atb/utils/operation_util.h"

using namespace atb;

TEST(TestEventOperation, ErrorEvent)
{
    /*
        特性：反例
        测试场景：测试EventOp的Param中的event为空的情况
        预期结果：返回报错ERROR_INVALID_PARAM
    */
    // 设置卡号、创建stream、创建context、设置stream
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    context->SetExecuteStream(stream);

    // 创建UNDEFINED EventOp
    atb::common::EventParam param;
    param.operatorType = atb::common::EventParam::OperatorType::RECORD;
    aclrtEvent event = nullptr;
    param.event = event;
    atb::Operation *operation = nullptr;
    Status st = atb::CreateOperation(param, &operation);
    ASSERT_EQ(st, atb::NO_ERROR);

    // 执行Operation
    atb::VariantPack pack;
    pack.inTensors = {};
    pack.outTensors = {};
    uint64_t workspaceSize = 0;
    // Setup
    operation->Setup(pack, workspaceSize, context);
    //Execute
    void *workSpace = nullptr;
    st = operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);

    // 释放资源
    atb::DestroyOperation(operation);
    atb::DestroyContext(context);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    ASSERT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestEventOperation, RecordOperation)
{
    /*
        特性：EventOp中的Record功能
        测试场景：测试EventOp的Record功能
        预期结果：返回错误码NO_ERROR
    */
    // 设置卡号、创建stream、创建context、设置stream
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    context->SetExecuteStream(stream);

    // // 创建RECORD Operation
    atb::common::EventParam param;
    param.operatorType = atb::common::EventParam::OperatorType::RECORD;
    aclrtEvent event;
    aclrtCreateEventWithFlag(&event, ACL_EVENT_SYNC);
    param.event = event;
    atb::Operation *operation = nullptr;
    atb::CreateOperation(param, &operation);

    // 执行Operation
    atb::VariantPack pack;
    pack.inTensors = {};
    pack.outTensors = {};
    uint64_t workspaceSize = 0;
    // Setup
    operation->Setup(pack, workspaceSize, context);
    //Execute
    void *workSpace = nullptr;
    atb::Status st = operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);

    // 释放资源
    aclrtDestroyEvent(event);
    atb::DestroyOperation(operation);
    atb::DestroyContext(context);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestEventOperation, WaitOperation)
{
    /*
        特性：EventOp中的Wait功能
        测试场景：测试EventOp的Wait功能，在调用之后暂停线程100毫秒后调用aclrtRecordEvent
        预期结果：调用aclrtSynchronizeStream成功
    */
    // 设置卡号、创建stream、创建context、设置stream
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream stream1;
    aclrtCreateStream(&stream1);
    aclrtStream stream2;
    aclrtCreateStream(&stream2);
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    context->SetExecuteStream(stream2);

    // // 创建WAIT Operation
    atb::common::EventParam param;
    param.operatorType = atb::common::EventParam::OperatorType::WAIT;
    aclrtEvent event;
    aclrtCreateEventWithFlag(&event, ACL_EVENT_SYNC);
    // aclrtRecordEvent(event, stream1);
    param.event = event;
    atb::Operation *operation = nullptr;
    atb::CreateOperation(param, &operation);

    // 执行Operation
    atb::VariantPack pack;
    pack.inTensors = {};
    pack.outTensors = {};
    uint64_t workspaceSize = 0;
    // Setup
    operation->Setup(pack, workspaceSize, context);
    //Execute
    void *workSpace = nullptr;
    atb::Status st = operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    aclrtRecordEvent(event, stream1);

    int ret = aclrtSynchronizeStream(stream2);
    
    // 释放资源
    aclrtDestroyEvent(event);
    atb::DestroyOperation(operation);
    atb::DestroyContext(context);
    aclrtDestroyStream(stream2);
    aclrtDestroyStream(stream1);
    aclrtResetDevice(deviceId);
    ASSERT_EQ(ret, 0);
}

void CreateGraphOperationRecord(atb::GraphParam &opGraph, atb::Operation **operation, aclrtEvent event)
{
	// 构图流程
    opGraph.inTensorNum = 2;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 2;
    opGraph.nodes.resize(4);

    size_t nodeId = 0;
    atb::Node &addNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode2 = opGraph.nodes.at(nodeId++);
    atb::Node &recordNode = opGraph.nodes.at(nodeId++);
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

    atb::common::EventParam recordParam;
    recordParam.operatorType = atb::common::EventParam::OperatorType::RECORD;
    recordParam.event = event;
    atb::CreateOperation(recordParam, &recordNode.operation);
    recordNode.inTensorIds = {};
    recordNode.outTensorIds = {};

    atb::infer::ElewiseParam addParam3;
    addParam3.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam3, &addNode3.operation);
    addNode3.inTensorIds = {4, 1};
    addNode3.outTensorIds = {2};

    atb::CreateOperation(opGraph, operation);
}

void CreateGraphOperationWait(atb::GraphParam &opGraph, atb::Operation **operation, aclrtEvent event)
{
	// 构图流程
    opGraph.inTensorNum = 2;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 2;
    opGraph.nodes.resize(4);

    size_t nodeId = 0;
    atb::Node &addNode = opGraph.nodes.at(nodeId++);
    atb::Node &waitNode = opGraph.nodes.at(nodeId++);
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

    atb::common::EventParam waitParam;
    waitParam.operatorType = atb::common::EventParam::OperatorType::WAIT;
    waitParam.event = event;
    atb::CreateOperation(waitParam, &waitNode.operation);
    waitNode.inTensorIds = {};
    waitNode.outTensorIds = {};

    atb::infer::ElewiseParam addParam3;
    addParam3.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam3, &addNode3.operation);
    addNode3.inTensorIds = {4, 1};
    addNode3.outTensorIds = {2};

    atb::CreateOperation(opGraph, operation);
}

void CreateInTensorDescs(atb::SVector<atb::TensorDesc> &intensorDescs) 
{
    for (size_t i = 0; i < intensorDescs.size(); i++) {
        intensorDescs.at(i).dtype = ACL_FLOAT16;
        intensorDescs.at(i).format = ACL_FORMAT_ND;
        intensorDescs.at(i).shape.dimNum = 2;
        intensorDescs.at(i).shape.dims[0] = 2;
        intensorDescs.at(i).shape.dims[1] = 2;
    }
}

void CreateInTensors(atb::SVector<atb::Tensor> &inTensors, atb::SVector<atb::TensorDesc> &intensorDescs)
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
void CreateOutTensors(atb::SVector<atb::Tensor> &outTensors, atb::SVector<atb::TensorDesc> &outtensorDescs)
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

TEST(TestEventOperation, TestRecordWaitGraphOp)
{
    /*
        特性：多图多流，先Record再Wait
        测试场景：测试两图两流，先执行含有OperatorType = Record的EventOp的图，再执行含有Wait功能的图
        预期结果：两个组图Execute的返回错误码为NO_ERROR
    */
    // 创建Graph1
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream stream1;
    aclrtCreateStream(&stream1);
    atb::Context *context1 = nullptr;
    aclrtEvent event;
    aclrtCreateEventWithFlag(&event, ACL_EVENT_SYNC);
    atb::CreateContext(&context1);
    context1->SetExecuteStream(stream1);

    atb::Operation *graphOp1 = nullptr;
    atb::GraphParam graphParam1;
    CreateGraphOperationRecord(graphParam1, &graphOp1, event);

    // 创建Graph2
    aclrtStream stream2;
    aclrtCreateStream(&stream2);
    atb::Context *context2 = nullptr;
    atb::CreateContext(&context2);
    context2->SetExecuteStream(stream2);

    atb::Operation *graphOp2 = nullptr;
    atb::GraphParam graphParam2;
    CreateGraphOperationWait(graphParam2, &graphOp2, event);

    // 准备输入输出tensor
    atb::VariantPack pack1;
    atb::SVector<atb::TensorDesc> intensorDescs1;
    atb::SVector<atb::TensorDesc> outtensorDescs1;

    uint32_t inTensorNum = 2;
    uint32_t outTensorNum = 1;
    pack1.inTensors.resize(inTensorNum);
    intensorDescs1.resize(inTensorNum);

    CreateInTensorDescs(intensorDescs1);
    CreateInTensors(pack1.inTensors, intensorDescs1);

    outtensorDescs1.resize(outTensorNum);
    outtensorDescs1.at(0).dtype = ACL_FLOAT16;
    outtensorDescs1.at(0).format = ACL_FORMAT_ND;
    outtensorDescs1.at(0).shape.dimNum = 2;
    outtensorDescs1.at(0).shape.dims[0] = 2;
    outtensorDescs1.at(0).shape.dims[1] = 2;
    pack1.outTensors.resize(outTensorNum);
    CreateOutTensors(pack1.outTensors, outtensorDescs1);

    atb::VariantPack pack2;
    atb::SVector<atb::TensorDesc> intensorDescs2;
    atb::SVector<atb::TensorDesc> outtensorDescs2;

    pack2.inTensors.resize(inTensorNum);
    intensorDescs2.resize(inTensorNum);

    CreateInTensorDescs(intensorDescs2);
    CreateInTensors(pack2.inTensors, intensorDescs2);

    outtensorDescs2.resize(outTensorNum);
    outtensorDescs2.at(0).dtype = ACL_FLOAT16;
    outtensorDescs2.at(0).format = ACL_FORMAT_ND;
    outtensorDescs2.at(0).shape.dimNum = 2;
    outtensorDescs2.at(0).shape.dims[0] = 2;
    outtensorDescs2.at(0).shape.dims[1] = 2;
    pack2.outTensors.resize(outTensorNum);
    CreateOutTensors(pack2.outTensors, outtensorDescs2);

    // Setup
    uint64_t workspaceSize1 = 0;
    graphOp1->Setup(pack1, workspaceSize1, context1);
    void *workSpace1 = nullptr;
    int ret1 = 0;
    if (workspaceSize1 != 0) {
        ret1 = aclrtMalloc(&workSpace1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret1 != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }
    uint64_t workspaceSize2 = 0;
    graphOp2->Setup(pack2, workspaceSize2, context2);
    void *workSpace2 = nullptr;
    int ret2 = 0;
    if (workspaceSize2 != 0) {
        ret2 = aclrtMalloc(&workSpace2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret2 != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }

    //Execute
    atb::Status st1 = graphOp1->Execute(pack1, (uint8_t*)workSpace1, workspaceSize1, context1);
    atb::Status st2 = graphOp2->Execute(pack2, (uint8_t*)workSpace2, workspaceSize2, context2);

    atb::Status st = (st2 == atb::NO_ERROR && st1 == atb::NO_ERROR) ? atb::NO_ERROR : atb::ERROR_INVALID_GRAPH;

    //流同步
    ret1 = aclrtSynchronizeStream(stream1);
    ret2 = aclrtSynchronizeStream(stream2);
    if (ret1 != 0 || ret2 != 0) {
        std::cout << "sync error!";
        exit(0);
    }

    // 资源释放
    aclrtDestroyEvent(event);
    atb::DestroyOperation(graphOp1);
    atb::DestroyOperation(graphOp2);
    atb::DestroyContext(context1);
    atb::DestroyContext(context2);
    for (size_t i = 0; i < pack1.inTensors.size(); i++) {
        aclrtFree(pack1.inTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack1.outTensors.size(); i++) {
        aclrtFree(pack1.outTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack2.inTensors.size(); i++) {
        aclrtFree(pack2.inTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack2.outTensors.size(); i++) {
        aclrtFree(pack2.outTensors.at(i).deviceData);
    }
    aclrtFree(workSpace1);
    aclrtFree(workSpace2);
    aclrtDestroyStream(stream1);
    aclrtDestroyStream(stream2);
    aclrtResetDevice(deviceId);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestEventOperation, TestWaitRecordGraphOp)
{
    /*
        特性：多图多流，先Wait再Record
        测试场景：测试两图两流，先执行含有OperatorType = Wait的EventOp的图，再执行含有Record功能的图
        在device侧上，含有Wait的图的EventOp会阻塞对应流直到含有Record的图中的EventOp执行完成后再往下运行
        预期结果：两个组图Execute的返回错误码为NO_ERROR
    */
    // 创建Graph1
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream stream1;
    aclrtCreateStream(&stream1);
    atb::Context *context1 = nullptr;
    aclrtEvent event;
    aclrtCreateEventWithFlag(&event, ACL_EVENT_SYNC);
    atb::CreateContext(&context1);
    context1->SetExecuteStream(stream1);

    atb::Operation *graphOp1 = nullptr;
    atb::GraphParam graphParam1;
    CreateGraphOperationRecord(graphParam1, &graphOp1, event);

    // 创建Graph2
    aclrtStream stream2;
    aclrtCreateStream(&stream2);
    atb::Context *context2 = nullptr;
    atb::CreateContext(&context2);
    context2->SetExecuteStream(stream2);

    atb::Operation *graphOp2 = nullptr;
    atb::GraphParam graphParam2;
    CreateGraphOperationWait(graphParam2, &graphOp2, event);

    // 准备输入输出tensor
    atb::VariantPack pack1;
    atb::SVector<atb::TensorDesc> intensorDescs1;
    atb::SVector<atb::TensorDesc> outtensorDescs1;

    uint32_t inTensorNum = 2;
    uint32_t outTensorNum = 1;
    pack1.inTensors.resize(inTensorNum);
    intensorDescs1.resize(inTensorNum);

    CreateInTensorDescs(intensorDescs1);
    CreateInTensors(pack1.inTensors, intensorDescs1);

    outtensorDescs1.resize(outTensorNum);
    outtensorDescs1.at(0).dtype = ACL_FLOAT16;
    outtensorDescs1.at(0).format = ACL_FORMAT_ND;
    outtensorDescs1.at(0).shape.dimNum = 2;
    outtensorDescs1.at(0).shape.dims[0] = 2;
    outtensorDescs1.at(0).shape.dims[1] = 2;
    pack1.outTensors.resize(outTensorNum);
    CreateOutTensors(pack1.outTensors, outtensorDescs1);

    atb::VariantPack pack2;
    atb::SVector<atb::TensorDesc> intensorDescs2;
    atb::SVector<atb::TensorDesc> outtensorDescs2;

    pack2.inTensors.resize(inTensorNum);
    intensorDescs2.resize(inTensorNum);

    CreateInTensorDescs(intensorDescs2);
    CreateInTensors(pack2.inTensors, intensorDescs2);

    outtensorDescs2.resize(outTensorNum);
    outtensorDescs2.at(0).dtype = ACL_FLOAT16;
    outtensorDescs2.at(0).format = ACL_FORMAT_ND;
    outtensorDescs2.at(0).shape.dimNum = 2;
    outtensorDescs2.at(0).shape.dims[0] = 2;
    outtensorDescs2.at(0).shape.dims[1] = 2;
    pack2.outTensors.resize(outTensorNum);
    CreateOutTensors(pack2.outTensors, outtensorDescs2);

    // Setup
    uint64_t workspaceSize1 = 0;
    graphOp1->Setup(pack1, workspaceSize1, context1);
    void *workSpace1 = nullptr;
    int ret1 = 0;
    if (workspaceSize1 != 0) {
        ret1 = aclrtMalloc(&workSpace1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret1 != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }
    uint64_t workspaceSize2 = 0;
    graphOp2->Setup(pack2, workspaceSize2, context2);
    void *workSpace2 = nullptr;
    int ret2 = 0;
    if (workspaceSize2 != 0) {
        ret2 = aclrtMalloc(&workSpace2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret2 != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }

    //Execute
    atb::Status st2 = graphOp2->Execute(pack2, (uint8_t*)workSpace2, workspaceSize2, context2);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    atb::Status st1 = graphOp1->Execute(pack1, (uint8_t*)workSpace1, workspaceSize1, context1);

    atb::Status st = (st2 == atb::NO_ERROR && st1 == atb::NO_ERROR) ? atb::NO_ERROR : atb::ERROR_INVALID_GRAPH;

    //流同步
    ret1 = aclrtSynchronizeStream(stream1);
    ret2 = aclrtSynchronizeStream(stream2);
    if (ret1 != 0 || ret2 != 0) {
        std::cout << "sync error!";
        exit(0);
    }

    // 资源释放
    aclrtDestroyEvent(event);
    atb::DestroyOperation(graphOp1);
    atb::DestroyOperation(graphOp2);
    atb::DestroyContext(context1);
    atb::DestroyContext(context2);
    for (size_t i = 0; i < pack1.inTensors.size(); i++) {
        aclrtFree(pack1.inTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack1.outTensors.size(); i++) {
        aclrtFree(pack1.outTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack2.inTensors.size(); i++) {
        aclrtFree(pack2.inTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack2.outTensors.size(); i++) {
        aclrtFree(pack2.outTensors.at(i).deviceData);
    }
    aclrtFree(workSpace1);
    aclrtFree(workSpace2);
    aclrtDestroyStream(stream1);
    aclrtDestroyStream(stream2);
    aclrtResetDevice(deviceId);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestEventOperation, TestUpdateWaitGraphOp)
{
    /*
        特性：修改图中的EventOp的Param
        测试场景：测试将OperatorType = Wait的EventOp接入到图中，调用Execute之后再修改EventOp的Param
        为OperatorType = UNDEFINED，跳过该EventOp
        预期结果：图中的EventOp的Param修改成功，Execute执行正常
    */
    // 设置前提
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream stream1;
    aclrtCreateStream(&stream1);
    aclrtEvent event;
    aclrtCreateEventWithFlag(&event, ACL_EVENT_SYNC);
    aclrtSetOpWaitTimeout(60);

    // 创建Graph2
    aclrtStream stream2;
    aclrtCreateStream(&stream2);
    atb::Context *context2 = nullptr;
    atb::CreateContext(&context2);
    context2->SetExecuteStream(stream2);

    atb::Operation *graphOp2 = nullptr;
    atb::GraphParam graphParam2;
    CreateGraphOperationWait(graphParam2, &graphOp2, event);

    // 准备输入输出tensor
    atb::VariantPack pack;
    atb::SVector<atb::TensorDesc> intensorDescs;
    atb::SVector<atb::TensorDesc> outtensorDescs;

    uint32_t inTensorNum = 2;
    uint32_t outTensorNum = 1;
    pack.inTensors.resize(inTensorNum);
    intensorDescs.resize(inTensorNum);

    CreateInTensorDescs(intensorDescs);
    CreateInTensors(pack.inTensors, intensorDescs);

    outtensorDescs.resize(outTensorNum);
    outtensorDescs.at(0).dtype = ACL_FLOAT16;
    outtensorDescs.at(0).format = ACL_FORMAT_ND;
    outtensorDescs.at(0).shape.dimNum = 2;
    outtensorDescs.at(0).shape.dims[0] = 2;
    outtensorDescs.at(0).shape.dims[1] = 2;
    pack.outTensors.resize(outTensorNum);
    CreateOutTensors(pack.outTensors, outtensorDescs);

    uint64_t workspaceSize2 = 0;
    graphOp2->Setup(pack, workspaceSize2, context2);
    void *workSpace2 = nullptr;
    int ret2 = 0;
    if (workspaceSize2 != 0) {
        ret2 = aclrtMalloc(&workSpace2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret2 != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }

    // Execute
    graphOp2->Execute(pack, (uint8_t*)workSpace2, workspaceSize2, context2);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    aclrtRecordEvent(event, stream1);

    // 流同步
    ret2 = aclrtSynchronizeStream(stream2);
    ASSERT_EQ(ret2, 0);

    // 修改EventOp成UNDEIFNED
    atb::common::EventParam unParam;
    atb::Status st = atb::CloneOperationParam(graphParam2.nodes.at(1).operation, unParam);
    ASSERT_EQ(st, 0);
    unParam.operatorType = atb::common::EventParam::OperatorType::UNDEFINED;
    st = atb::UpdateOperationParam(graphParam2.nodes.at(1).operation, unParam);

    // 执行
    uint64_t workspaceSize3 = 0;
    graphOp2->Setup(pack, workspaceSize3, context2);
    void *workSpace3 = nullptr;
    int ret3 = 0;
    if (workspaceSize3 != 0) {
        ret3 = aclrtMalloc(&workSpace3, workspaceSize3, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret3 != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }

    // Execute
    graphOp2->Execute(pack, (uint8_t*)workSpace3, workspaceSize3, context2);

    // 资源释放
    aclrtDestroyEvent(event);
    atb::DestroyOperation(graphOp2);
    atb::DestroyContext(context2);
    for (size_t i = 0; i < pack.inTensors.size(); i++) {
        aclrtFree(pack.inTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack.outTensors.size(); i++) {
        aclrtFree(pack.outTensors.at(i).deviceData);
    }
    aclrtFree(workSpace2);
    aclrtFree(workSpace3);
    aclrtDestroyStream(stream1);
    aclrtDestroyStream(stream2);
    aclrtResetDevice(deviceId);
    ASSERT_EQ(ret3, 0);
}

TEST(TestEventOperation, TestUpdateParam)
{
    /*
        特性：修改EventOp的Param
        测试场景：创建完成EventOperation后，修改Param参数从RECORD改成WAIT再改成UNDEFINED
        预期结果：修改成功
    */
    // 设置卡号、创建stream、创建context、设置stream
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    context->SetExecuteStream(stream);

    // 创建RECORD Operation
    atb::common::EventParam param;
    param.operatorType = atb::common::EventParam::OperatorType::RECORD;
    aclrtEvent event;
    aclrtCreateEventWithFlag(&event, ACL_EVENT_SYNC);
    param.event = event;
    atb::Operation *operation = nullptr;
    atb::CreateOperation(param, &operation);

    // 执行Operation
    atb::VariantPack pack;
    pack.inTensors = {};
    pack.outTensors = {};
    uint64_t workspaceSize = 0;
    // Setup
    operation->Setup(pack, workspaceSize, context);
    //Execute
    void *workSpace = nullptr;
    atb::Status st = operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);

    // 修改Param
    atb::common::EventParam waitParam;
    st = atb::CloneOperationParam(operation, waitParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(param, waitParam);

    waitParam.operatorType = atb::common::EventParam::OperatorType::WAIT;
    st = atb::UpdateOperationParam(operation, waitParam);
    EXPECT_EQ(st, atb::NO_ERROR);

    atb::common::EventParam getParamAfter;
    st = atb::CloneOperationParam(operation, getParamAfter);
    ATB_LOG(INFO) << "after update OperatorType: " << getParamAfter.operatorType;
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParamAfter, waitParam);

    // 修改event为nullptr
    waitParam.event = nullptr;
    st = atb::UpdateOperationParam(operation, waitParam);
    EXPECT_EQ(st, atb::NO_ERROR);

    // 执行修改后的Operation
    // Setup
    operation->Setup(pack, workspaceSize, context);
    //Execute
    workSpace = nullptr;
    st = operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
    ASSERT_EQ(st, atb::ERROR_INVALID_PARAM);

    // 再修改 WAIT -> UNDEFINED, UNDEFINED可以不需要传入event
    getParamAfter.operatorType = atb::common::EventParam::OperatorType::UNDEFINED;
    getParamAfter.event = nullptr;
    st = atb::UpdateOperationParam(operation, getParamAfter);
    EXPECT_EQ(st, atb::NO_ERROR);

    // 执行
    // Setup
    operation->Setup(pack, workspaceSize, context);
    //Execute
    workSpace = nullptr;
    st = operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
    ASSERT_EQ(st, atb::NO_ERROR);

    // 释放资源
    aclrtDestroyEvent(event);
    atb::DestroyOperation(operation);
    atb::DestroyContext(context);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
}

TEST(TestEventOperation, TestWaitRecordGraphOpMultiTime)
{
    /*
        特性：重复运行双图重复使用同一个Event
        测试场景：测试两图两流，先执行含有OperatorType = Wait的EventOp的图，再执行含有Record功能的图
        在device侧上，含有Wait的图的EventOp会阻塞对应流直到含有Record的图中的EventOp执行完成后再往下运行
        一次运行完成之后，Reset所使用的的Event再运行下一轮
        预期结果：两个组图Execute的返回错误码为NO_ERROR
    */
    // 创建Graph1
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream stream1;
    aclrtCreateStream(&stream1);
    atb::Context *context1 = nullptr;
    aclrtEvent event;
    aclrtCreateEventWithFlag(&event, ACL_EVENT_SYNC);
    atb::CreateContext(&context1);
    context1->SetExecuteStream(stream1);

    atb::Operation *graphOp1 = nullptr;
    atb::GraphParam graphParam1;
    CreateGraphOperationRecord(graphParam1, &graphOp1, event);

    // 创建Graph2
    aclrtStream stream2;
    aclrtCreateStream(&stream2);
    atb::Context *context2 = nullptr;
    atb::CreateContext(&context2);
    context2->SetExecuteStream(stream2);

    atb::Operation *graphOp2 = nullptr;
    atb::GraphParam graphParam2;
    CreateGraphOperationWait(graphParam2, &graphOp2, event);

    for(int i = 0; i < 5; i++) {
        // 准备输入输出tensor
        atb::VariantPack pack1;
        atb::SVector<atb::TensorDesc> intensorDescs1;
        atb::SVector<atb::TensorDesc> outtensorDescs1;

        uint32_t inTensorNum = 2;
        uint32_t outTensorNum = 1;
        pack1.inTensors.resize(inTensorNum);
        intensorDescs1.resize(inTensorNum);

        CreateInTensorDescs(intensorDescs1);
        CreateInTensors(pack1.inTensors, intensorDescs1);

        outtensorDescs1.resize(outTensorNum);
        outtensorDescs1.at(0).dtype = ACL_FLOAT16;
        outtensorDescs1.at(0).format = ACL_FORMAT_ND;
        outtensorDescs1.at(0).shape.dimNum = 2;
        outtensorDescs1.at(0).shape.dims[0] = 2;
        outtensorDescs1.at(0).shape.dims[1] = 2;
        pack1.outTensors.resize(outTensorNum);
        CreateOutTensors(pack1.outTensors, outtensorDescs1);

        atb::VariantPack pack2;
        atb::SVector<atb::TensorDesc> intensorDescs2;
        atb::SVector<atb::TensorDesc> outtensorDescs2;

        pack2.inTensors.resize(inTensorNum);
        intensorDescs2.resize(inTensorNum);

        CreateInTensorDescs(intensorDescs2);
        CreateInTensors(pack2.inTensors, intensorDescs2);

        outtensorDescs2.resize(outTensorNum);
        outtensorDescs2.at(0).dtype = ACL_FLOAT16;
        outtensorDescs2.at(0).format = ACL_FORMAT_ND;
        outtensorDescs2.at(0).shape.dimNum = 2;
        outtensorDescs2.at(0).shape.dims[0] = 2;
        outtensorDescs2.at(0).shape.dims[1] = 2;
        pack2.outTensors.resize(outTensorNum);
        CreateOutTensors(pack2.outTensors, outtensorDescs2);

        // Setup
        uint64_t workspaceSize1 = 0;
        graphOp1->Setup(pack1, workspaceSize1, context1);
        void *workSpace1 = nullptr;
        int ret1 = 0;
        if (workspaceSize1 != 0) {
            ret1 = aclrtMalloc(&workSpace1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret1 != 0) {
                std::cout << "alloc error!";
                exit(0);
            }
        }
        uint64_t workspaceSize2 = 0;
        graphOp2->Setup(pack2, workspaceSize2, context2);
        void *workSpace2 = nullptr;
        int ret2 = 0;
        if (workspaceSize2 != 0) {
            ret2 = aclrtMalloc(&workSpace2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret2 != 0) {
                std::cout << "alloc error!";
                exit(0);
            }
        }

        //Execute
        atb::Status st2 = graphOp2->Execute(pack2, (uint8_t*)workSpace2, workspaceSize2, context2);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        atb::Status st1 = graphOp1->Execute(pack1, (uint8_t*)workSpace1, workspaceSize1, context1);

        atb::Status st = (st2 == atb::NO_ERROR && st1 == atb::NO_ERROR) ? atb::NO_ERROR : atb::ERROR_INVALID_GRAPH;
        ASSERT_EQ(st, 0);

        // //流同步
        ret1 = aclrtSynchronizeStream(stream1);
        ret2 = aclrtSynchronizeStream(stream2);
        if (ret1 != 0 || ret2 != 0) {
            std::cout << "sync error!";
            exit(0);
        }
        
        // ResetEvent 后再次执行
        int ret3 = aclrtResetEvent(event, stream2);
        ASSERT_EQ(ret3, 0);

        for (size_t i = 0; i < pack1.inTensors.size(); i++) {
            aclrtFree(pack1.inTensors.at(i).deviceData);
        }
        for (size_t i = 0; i < pack1.outTensors.size(); i++) {
            aclrtFree(pack1.outTensors.at(i).deviceData);
        }
        for (size_t i = 0; i < pack2.inTensors.size(); i++) {
            aclrtFree(pack2.inTensors.at(i).deviceData);
        }
        for (size_t i = 0; i < pack2.outTensors.size(); i++) {
            aclrtFree(pack2.outTensors.at(i).deviceData);
        }
        aclrtFree(workSpace1);
        aclrtFree(workSpace2);
    }

    // 资源释放
    aclrtDestroyEvent(event);
    atb::DestroyOperation(graphOp1);
    atb::DestroyOperation(graphOp2);
    atb::DestroyContext(context1);
    atb::DestroyContext(context2);
    aclrtDestroyStream(stream1);
    aclrtDestroyStream(stream2);
    aclrtResetDevice(deviceId);
}