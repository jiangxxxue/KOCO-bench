/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <atb/operation.h>
#include "atb/infer_op_params.h"
#include <atb/types.h>
#include <acl/acl.h>
#include <iostream>
#include <vector>
#include <atb/utils.h>
#include <unistd.h>
#include <acl/acl_mdl.h>
#include <cstdlib>  // 包含 std::getenv
#include <cstring>  // 包含 std::strcmp（用于字符串比较）
#include <gtest/gtest.h>
#include <atb/utils/log.h>
#include <cpp-stub/src/stub.h>
#include <atb/utils/config.h>
#include "atb/utils/singleton.h"
#include <mki/utils/time/timer.h>
 
// 设置各个intensor的属性
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
 
// 设置各个intensor并且为各个intensor分配内存空间，此处的intensor为手动设置，工程实现上可以使用torchTensor转换或者其他简单数据结构转换的方式
static void CreateInTensors(atb::SVector<atb::Tensor> &inTensors, atb::SVector<atb::TensorDesc> &intensorDescs, uint32_t value)
{
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensors.at(i).desc = intensorDescs.at(i);
        inTensors.at(i).dataSize = atb::Utils::GetTensorSize(inTensors.at(i));
        std::vector<uint16_t> hostData(atb::Utils::GetTensorNumel(inTensors.at(i)), value);   // 一段全2的hostBuffer
        int ret = aclrtMalloc(&inTensors.at(i).deviceData, inTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST); // 分配NPU内存
        ASSERT_EQ(ret, 0);
        ret = aclrtMemcpy(inTensors.at(i).deviceData, inTensors.at(i).dataSize, hostData.data(), hostData.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE); //拷贝CPU内存到NPU侧
        ASSERT_EQ(ret, 0);
    }
}
 
// 设置各个outtensor并且为outtensor分配内存空间，同intensor设置
static void CreateOutTensors(atb::SVector<atb::Tensor> &outTensors, atb::SVector<atb::TensorDesc> &outtensorDescs)
{
    for (size_t i = 0; i < outTensors.size(); i++) {
        outTensors.at(i).desc = outtensorDescs.at(i);
        outTensors.at(i).dataSize = atb::Utils::GetTensorSize(outTensors.at(i));
        int ret = aclrtMalloc(&outTensors.at(i).deviceData, outTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ASSERT_EQ(ret, 0);
    }
}
 
static void CreateGraphOperation(atb::GraphParam &opGraph, atb::Operation **operation)
{
    opGraph.inTensorNum = 4;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 2;
    opGraph.nodes.resize(3);
 
    enum InTensorId {
        IN_TENSOR_A = 0,
        IN_TENSOR_B,
        IN_TENSOR_C,
        IN_TENSOR_D,
        ADD3_OUT,
        ADD1_OUT,
        ADD2_OUT
    };
 
    size_t nodeId = 0;
    atb::Node &addNode = opGraph.nodes.at(nodeId++);
    atb::Node &addNode2 = opGraph.nodes.at(nodeId++);
    atb::Node &addNode3 = opGraph.nodes.at(nodeId++);
 
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Status status = atb::CreateOperation(addParam, &addNode.operation);
    addNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
    addNode.outTensorIds = {ADD1_OUT};
 
    atb::infer::ElewiseParam addParam2;
    addParam2.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    status = atb::CreateOperation(addParam2, &addNode2.operation);
    addNode2.inTensorIds = {IN_TENSOR_C, IN_TENSOR_D};
    addNode2.outTensorIds = {ADD2_OUT};
 
    atb::infer::ElewiseParam addParam3;
    addParam3.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    status = CreateOperation(addParam3, &addNode3.operation);
    addNode3.inTensorIds = {ADD1_OUT, ADD2_OUT};
    addNode3.outTensorIds = {ADD3_OUT};
 
    status = atb::CreateOperation(opGraph, operation);
    ASSERT_EQ(status, 0);
}
 
static void CreateMultiGraphOperation(atb::GraphParam &opGraph, atb::Operation **operation)
{
    opGraph.inTensorNum = 4;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 4;
    opGraph.nodes.resize(5);
 
    enum InTensorId {
        IN_TENSOR_A = 0,
        IN_TENSOR_B,
        IN_TENSOR_C,
        IN_TENSOR_D,
        ADD5_OUT,
        ADD1_INTER,
        ADD2_INTER,
        ADD3_INTER,
        ADD4_INTER
    };
 
    size_t nodeId = 0;
    atb::Node &addGraphNode1 = opGraph.nodes.at(nodeId++);
    atb::Node &addGraphNode2 = opGraph.nodes.at(nodeId++);
    atb::Node &addGraphNode3 = opGraph.nodes.at(nodeId++);
    atb::Node &addGraphNode4 = opGraph.nodes.at(nodeId++);
    atb::Node &addGraphNode5 = opGraph.nodes.at(nodeId++);
 
    atb::GraphParam addGraphParam1;
    CreateGraphOperation(addGraphParam1, &addGraphNode1.operation);
    addGraphNode1.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B, IN_TENSOR_C, IN_TENSOR_D};
    addGraphNode1.outTensorIds = {ADD1_INTER};
 
    atb::GraphParam addGraphParam2;
    CreateGraphOperation(addGraphParam2, &addGraphNode2.operation);
    addGraphNode2.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B, IN_TENSOR_C, IN_TENSOR_D};
    addGraphNode2.outTensorIds = {ADD2_INTER};
 
    atb::GraphParam addGraphParam3;
    CreateGraphOperation(addGraphParam3, &addGraphNode3.operation);
    addGraphNode3.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B, IN_TENSOR_C, IN_TENSOR_D};
    addGraphNode3.outTensorIds = {ADD3_INTER};
 
    atb::GraphParam addGraphParam4;
    CreateGraphOperation(addGraphParam4, &addGraphNode4.operation);
    addGraphNode4.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B, IN_TENSOR_C, IN_TENSOR_D};
    addGraphNode4.outTensorIds = {ADD4_INTER};
 
    atb::GraphParam addGraphParam5;
    CreateGraphOperation(addGraphParam5, &addGraphNode5.operation);
    addGraphNode5.inTensorIds = {ADD1_INTER, ADD2_INTER, ADD3_INTER, ADD4_INTER};
    addGraphNode5.outTensorIds = {ADD5_OUT};
 
    atb::Status status = atb::CreateOperation(opGraph, operation);
}
 
static void PrintOutTensorValue(atb::Tensor &outTensor, std::vector<uint16_t> &outBuffer)
{
    int ret = aclrtMemcpy(outBuffer.data(), outBuffer.size() * sizeof(uint16_t), outTensor.deviceData, outTensor.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(ret, 0);
    for (size_t i = 0; i < outBuffer.size(); i = i + 1) {
        ATB_LOG(DEBUG) << "out[" << i << "] = " << (uint32_t)outBuffer.at(i);
    }
}
 
TEST(TestGraphLaunchMode, CapturedByUserAndTestGraphOp)
{
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        return;
    }
	// 设置卡号、创建stream、创建context、设置stream
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream exeStream = nullptr;
    aclrtCreateStream(&exeStream);
    int err = 0;
 
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    context->SetExecuteStream(exeStream);
 
    // 创建图算子
    atb::Operation *operation = nullptr;
    atb::Operation *warmUpOperation = nullptr;
    atb::Operation *kernelLaunchOperation = nullptr;
    atb::GraphParam opGraph;
    CreateMultiGraphOperation(opGraph, &operation);
    CreateMultiGraphOperation(opGraph, &warmUpOperation);
    CreateMultiGraphOperation(opGraph, &kernelLaunchOperation);
 
	// 输入输出tensor准备
    atb::VariantPack pack;
    atb::SVector<atb::TensorDesc> intensorDescs;
    atb::SVector<atb::TensorDesc> outtensorDescs;
 
    uint32_t inTensorNum = opGraph.inTensorNum;
    uint32_t outTensorNum = opGraph.outTensorNum;
    inTensorNum = operation->GetInputNum();
    outTensorNum = operation->GetOutputNum();
 
    pack.inTensors.resize(inTensorNum);
    intensorDescs.resize(inTensorNum);
    CreateInTensorDescs(intensorDescs);
    
    outtensorDescs.resize(outTensorNum);
    pack.outTensors.resize(outTensorNum);
    operation->InferShape(intensorDescs, outtensorDescs);

    uint64_t workspaceSize = 0;
    void *workSpace = nullptr;
    int ret = 0;

    // prof golden
    CreateInTensors(pack.inTensors, intensorDescs, 1);
    CreateOutTensors(pack.outTensors, outtensorDescs);
    context->SetLaunchMode(atb::KERNEL_LAUNCH_MODE);
    uint64_t goldenTime = 100000000;
    for (size_t i = 0; i < 4; i++) {
        Mki::Timer timer;
        kernelLaunchOperation->Setup(pack, workspaceSize, context);
        if (workspaceSize != 0 && workSpace == nullptr) {
            ret = aclrtMalloc(&workSpace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            ASSERT_EQ(ret, 0);
        }
        context->SetExecuteType(atb::EXECUTE_NORMAL);
        kernelLaunchOperation->Execute(pack, (uint8_t *)workSpace, workspaceSize, context);
        goldenTime = std::min(timer.ElapsedMicroSecond(), goldenTime);
        ret = aclrtSynchronizeStream(exeStream);
        ASSERT_EQ(ret, 0);
    }
    aclrtFree(workSpace);
    DestroyOperation(kernelLaunchOperation);
 
    workspaceSize = 0;
    workSpace = nullptr;
    ret = 0;

	// 算子执行
    aclmdlRI model = nullptr;
    context->SetLaunchMode(atb::GRAPH_LAUNCH_MODE);
    uint64_t updateTime = 100000000;
    for (size_t i = 0; i < 10; i++) {
        CreateInTensors(pack.inTensors, intensorDescs, i + 1);
        CreateOutTensors(pack.outTensors, outtensorDescs);
        if (i == 0) {
            aclmdlRICaptureBegin(exeStream, ACL_MODEL_RI_CAPTURE_MODE_RELAXED);
            operation->Setup(pack, workspaceSize, context);
            if (workspaceSize != 0 && workSpace == nullptr) {
                ret = aclrtMalloc(&workSpace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                ASSERT_EQ(ret, 0);
            }
            std::cout << "workspace:" << workSpace << ", workspaceSize:" << workspaceSize << std::endl;
            context->SetExecuteType(atb::EXECUTE_PRELAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_LAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_NORMAL);
            aclmdlRICaptureEnd(exeStream, &model);
            aclmdlRIExecuteAsync(model, exeStream);
        } else {
            // 参数更新
            Mki::Timer timer;
            operation->Setup(pack, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_PRELAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_NORMAL);
            updateTime = std::min(timer.ElapsedMicroSecond(), updateTime);
 
            // 重放
            aclmdlRIExecuteAsync(model, exeStream);
        }
        // 流同步，作用是等待device侧任务计算完成
        ret = aclrtSynchronizeStream(exeStream);
        ASSERT_EQ(ret, 0);
        std::cout << "aclrtSynchronizeStream success!" << std::endl;
 
        // 打印输出Tensor的值
        std::vector<uint16_t> outBuffer(atb::Utils::GetTensorNumel(pack.outTensors.at(0)));
        PrintOutTensorValue(pack.outTensors.at(0), outBuffer);
        for (size_t j = 0; j < outBuffer.size(); j++) {
            EXPECT_EQ((uint32_t)outBuffer.at(j), (i + 1) * 16);
        }
    }
    double timeCompareRate = (double)updateTime / (double)goldenTime;
    ATB_LOG(INFO) << "timeCompareRate:" << timeCompareRate;
    // EXPECT_LT(timeCompareRate, 0.5);
 
	// 资源释放
    atb::DestroyOperation(operation);
    atb::DestroyContext(context);
    aclmdlRIDestroy(model);
    for (size_t i = 0; i < pack.inTensors.size(); i++) {
        aclrtFree(pack.inTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack.outTensors.size(); i++) {
        aclrtFree(pack.outTensors.at(i).deviceData);
    }
    aclrtFree(workSpace);
    aclrtDestroyStream(exeStream);
    aclrtResetDevice(deviceId);
}
 
TEST(TestGraphLaunchMode, CapturedByAtbAndTestGraphOp)
{
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        return;
    }
	// 设置卡号、创建stream、创建context、设置stream
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream exeStream = nullptr;
    aclrtCreateStream(&exeStream);
    int err = 0;
 
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    context->SetExecuteStream(exeStream);
 
    // 创建图算子
    atb::Operation *operation = nullptr;
    atb::Operation *warmUpOperation = nullptr;
    atb::GraphParam opGraph;
    CreateMultiGraphOperation(opGraph, &operation);
    CreateMultiGraphOperation(opGraph, &warmUpOperation);
 
	// 输入输出tensor准备
    atb::VariantPack pack;
    atb::SVector<atb::TensorDesc> intensorDescs;
    atb::SVector<atb::TensorDesc> outtensorDescs;
 
    uint32_t inTensorNum = opGraph.inTensorNum;
    uint32_t outTensorNum = opGraph.outTensorNum;
    inTensorNum = operation->GetInputNum();
    outTensorNum = operation->GetOutputNum();
 
    pack.inTensors.resize(inTensorNum);
    intensorDescs.resize(inTensorNum);
    CreateInTensorDescs(intensorDescs);
    
    outtensorDescs.resize(outTensorNum);
    pack.outTensors.resize(outTensorNum);
    operation->InferShape(intensorDescs, outtensorDescs);

    uint64_t workspaceSize = 0;
    void *workSpace = nullptr;
    int ret = 0;

    // warm up
    CreateInTensors(pack.inTensors, intensorDescs, 1);
    CreateOutTensors(pack.outTensors, outtensorDescs);
    context->SetLaunchMode(atb::KERNEL_LAUNCH_MODE);
    for (size_t i = 0; i < 2; i++) {
        warmUpOperation->Setup(pack, workspaceSize, context);
        if (workspaceSize != 0 && workSpace == nullptr) {
            ret = aclrtMalloc(&workSpace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            ASSERT_EQ(ret, 0);
        }
        context->SetExecuteType(atb::EXECUTE_NORMAL);
        warmUpOperation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
        ret = aclrtSynchronizeStream(exeStream);
        ASSERT_EQ(ret, 0);
    }
    aclrtFree(workSpace);
    DestroyOperation(warmUpOperation);
 
    workspaceSize = 0;
    workSpace = nullptr;
    ret = 0;
	// 算子执行
    context->SetLaunchMode(atb::GRAPH_LAUNCH_MODE);
    for (size_t i = 0; i < 10; i++) {
        CreateInTensors(pack.inTensors, intensorDescs, i + 1);
        CreateOutTensors(pack.outTensors, outtensorDescs);
        operation->Setup(pack, workspaceSize, context);
        if (workspaceSize != 0 && workSpace == nullptr) {
            ret = aclrtMalloc(&workSpace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            ASSERT_EQ(ret, 0);
        }
        std::cout << "workspace:" << workSpace << ", workspaceSize:" << workspaceSize << std::endl;
        context->SetExecuteType(atb::EXECUTE_PRELAUNCH);
        operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
        context->SetExecuteType(atb::EXECUTE_LAUNCH);
        operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
        context->SetExecuteType(atb::EXECUTE_NORMAL);
 
        // 流同步，作用是等待device侧任务计算完成
        ret = aclrtSynchronizeStream(exeStream);
        ASSERT_EQ(ret, 0);
        std::cout << "aclrtSynchronizeStream success!" << std::endl;
 
        // 打印输出Tensor的值
        std::vector<uint16_t> outBuffer(atb::Utils::GetTensorNumel(pack.outTensors.at(0)));
        PrintOutTensorValue(pack.outTensors.at(0), outBuffer);
        for (size_t j = 0; j < outBuffer.size(); j++) {
            EXPECT_EQ((uint32_t)outBuffer.at(j), (i + 1) * 16);
        }
    }
 
	// 资源释放
    atb::DestroyOperation(operation);
    atb::DestroyContext(context);
    for (size_t i = 0; i < pack.inTensors.size(); i++) {
        aclrtFree(pack.inTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack.outTensors.size(); i++) {
        aclrtFree(pack.outTensors.at(i).deviceData);
    }
    aclrtFree(workSpace);
    aclrtDestroyStream(exeStream);
    aclrtResetDevice(deviceId);
}

TEST(TestGraphLaunchMode, CapturedByAtbAndChangeWorkspace)
{
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        return;
    }
	// 设置卡号、创建stream、创建context、设置stream
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream exeStream = nullptr;
    aclrtCreateStream(&exeStream);
    int err = 0;
 
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    context->SetExecuteStream(exeStream);
    context->SetLaunchMode(atb::GRAPH_LAUNCH_MODE);
 
    // 创建图算子
    atb::Operation *operation = nullptr;
    atb::GraphParam opGraph;
    CreateMultiGraphOperation(opGraph, &operation);
 
	// 输入输出tensor准备
    atb::VariantPack pack;
    atb::SVector<atb::TensorDesc> intensorDescs;
    atb::SVector<atb::TensorDesc> outtensorDescs;
 
    uint32_t inTensorNum = opGraph.inTensorNum;
    uint32_t outTensorNum = opGraph.outTensorNum;
    inTensorNum = operation->GetInputNum();
    outTensorNum = operation->GetOutputNum();
 
    pack.inTensors.resize(inTensorNum);
    intensorDescs.resize(inTensorNum);
    CreateInTensorDescs(intensorDescs);
    
    outtensorDescs.resize(outTensorNum);
    pack.outTensors.resize(outTensorNum);
    operation->InferShape(intensorDescs, outtensorDescs);
    CreateOutTensors(pack.outTensors, outtensorDescs);
 
    uint64_t workspaceSize = 0;
    void *workSpace = nullptr;
    void *workSpace1 = nullptr;
    void *workSpace2 = nullptr;
    void *workSpace3 = nullptr;
    int ret = 0;
	// 算子执行
    aclmdlRI model = nullptr;
    for (size_t i = 0; i < 10; i++) {
        CreateInTensors(pack.inTensors, intensorDescs, i + 1);
        CreateOutTensors(pack.outTensors, outtensorDescs);
        if (i == 0) {
            operation->Setup(pack, workspaceSize, context);
            if (workspaceSize != 0 && workSpace == nullptr) {
                ret = aclrtMalloc(&workSpace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                ASSERT_EQ(ret, 0);
            }
            std::cout << "workspace:" << workSpace << ", workspaceSize:" << workspaceSize << std::endl;
            context->SetExecuteType(atb::EXECUTE_PRELAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_LAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_NORMAL);
        } else if (i > 3 && i < 6) {
            // 支持workSpace更新
            operation->Setup(pack, workspaceSize, context);
            if (workspaceSize != 0 && workSpace1 == nullptr) {
                ret = aclrtMalloc(&workSpace1, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                ASSERT_EQ(ret, 0);
            }
            std::cout << "workspace1:" << workSpace1 << ", workspaceSize:" << workspaceSize << std::endl;
            context->SetExecuteType(atb::EXECUTE_PRELAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace1, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_LAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace1, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_NORMAL);
            aclrtFree(workSpace1);
            workSpace1 = nullptr;
        } else if (i >= 6 && i < 8) {
            // 支持workSpace更新
            operation->Setup(pack, workspaceSize, context);
            if (workspaceSize != 0 && workSpace2 == nullptr) {
                ret = aclrtMalloc(&workSpace2, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                ASSERT_EQ(ret, 0);
            }
            std::cout << "workSpace2:" << workSpace2 << ", workspaceSize:" << workspaceSize << std::endl;
            context->SetExecuteType(atb::EXECUTE_PRELAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace2, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_LAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace2, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_NORMAL);
            aclrtFree(workSpace2);
            workSpace2 = nullptr;
        } else if (i >= 8 && i <= 9) {
            // 支持workSpace更新
            operation->Setup(pack, workspaceSize, context);
            if (workspaceSize != 0 && workSpace3 == nullptr) {
                ret = aclrtMalloc(&workSpace3, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                ASSERT_EQ(ret, 0);
            }
            std::cout << "workSpace3:" << workSpace3 << ", workspaceSize:" << workspaceSize << std::endl;
            context->SetExecuteType(atb::EXECUTE_PRELAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace3, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_LAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace3, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_NORMAL);
            aclrtFree(workSpace3);
            workSpace3 = nullptr;
        } else {
            std::cout << "workspace:" << workSpace << ", workspaceSize:" << workspaceSize << std::endl;
            operation->Setup(pack, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_PRELAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_LAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_NORMAL);
        }
 
        // 流同步，作用是等待device侧任务计算完成
        ret = aclrtSynchronizeStream(exeStream);
        ASSERT_EQ(ret, 0);
        std::cout << "aclrtSynchronizeStream success!" << std::endl;
 
        // 打印输出Tensor的值
        std::vector<uint16_t> outBuffer(atb::Utils::GetTensorNumel(pack.outTensors.at(0)));
        PrintOutTensorValue(pack.outTensors.at(0), outBuffer);
        for (size_t j = 0; j < outBuffer.size(); j++) {
            EXPECT_EQ((uint32_t)outBuffer.at(j), (i + 1) * 16);
        }
    }
 
	// 资源释放
    atb::DestroyOperation(operation);
    atb::DestroyContext(context);
    aclmdlRIDestroy(model);
    for (size_t i = 0; i < pack.inTensors.size(); i++) {
        aclrtFree(pack.inTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack.outTensors.size(); i++) {
        aclrtFree(pack.outTensors.at(i).deviceData);
    }
    aclrtFree(workSpace);
    aclrtFree(workSpace1);
    aclrtFree(workSpace2);
    aclrtFree(workSpace3);
    aclrtDestroyStream(exeStream);
    aclrtResetDevice(deviceId);
}

TEST(TestGraphLaunchMode, CapturedByUserAndChangeWorkspace)
{
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        return;
    }
	// 设置卡号、创建stream、创建context、设置stream
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream exeStream = nullptr;
    aclrtCreateStream(&exeStream);
    int err = 0;
 
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    context->SetExecuteStream(exeStream);
    context->SetLaunchMode(atb::GRAPH_LAUNCH_MODE);
 
    // 创建图算子
    atb::Operation *operation = nullptr;
    atb::GraphParam opGraph;
    CreateMultiGraphOperation(opGraph, &operation);
 
	// 输入输出tensor准备
    atb::VariantPack pack;
    atb::SVector<atb::TensorDesc> intensorDescs;
    atb::SVector<atb::TensorDesc> outtensorDescs;
 
    uint32_t inTensorNum = opGraph.inTensorNum;
    uint32_t outTensorNum = opGraph.outTensorNum;
    inTensorNum = operation->GetInputNum();
    outTensorNum = operation->GetOutputNum();
 
    pack.inTensors.resize(inTensorNum);
    intensorDescs.resize(inTensorNum);
    CreateInTensorDescs(intensorDescs);
    
    outtensorDescs.resize(outTensorNum);
    pack.outTensors.resize(outTensorNum);
    operation->InferShape(intensorDescs, outtensorDescs);
    CreateOutTensors(pack.outTensors, outtensorDescs);
 
    uint64_t workspaceSize = 0;
    void *workSpace = nullptr;
    void *workSpace1 = nullptr;
    void *workSpace2 = nullptr;
    void *workSpace3 = nullptr;
    int ret = 0;
	// 算子执行
    aclmdlRI model = nullptr;
    for (size_t i = 0; i < 10; i++) {
        CreateInTensors(pack.inTensors, intensorDescs, i + 1);
        CreateOutTensors(pack.outTensors, outtensorDescs);
        if (i == 0) {
            aclmdlRICaptureBegin(exeStream, ACL_MODEL_RI_CAPTURE_MODE_RELAXED);
            operation->Setup(pack, workspaceSize, context);
            if (workspaceSize != 0 && workSpace == nullptr) {
                ret = aclrtMalloc(&workSpace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                ASSERT_EQ(ret, 0);
            }
            std::cout << "workspace:" << workSpace << ", workspaceSize:" << workspaceSize << std::endl;
            context->SetExecuteType(atb::EXECUTE_PRELAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_LAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_NORMAL);
            aclmdlRICaptureEnd(exeStream, &model);
            aclmdlRIExecuteAsync(model, exeStream);
        } else if (i > 3 && i < 6) {
            // 支持workSpace更新
            operation->Setup(pack, workspaceSize, context);
            if (workspaceSize != 0 && workSpace1 == nullptr) {
                ret = aclrtMalloc(&workSpace1, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                ASSERT_EQ(ret, 0);
            }
            std::cout << "workspace1:" << workSpace1 << ", workspaceSize:" << workspaceSize << std::endl;
            context->SetExecuteType(atb::EXECUTE_PRELAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace1, workspaceSize, context);
            aclmdlRIExecuteAsync(model, exeStream);
            context->SetExecuteType(atb::EXECUTE_NORMAL);
        } else if (i >= 6 && i < 8) {
            // 支持workSpace更新
            operation->Setup(pack, workspaceSize, context);
            if (workspaceSize != 0 && workSpace2 == nullptr) {
                ret = aclrtMalloc(&workSpace2, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                ASSERT_EQ(ret, 0);
            }
            std::cout << "workSpace2:" << workSpace2 << ", workspaceSize:" << workspaceSize << std::endl;
            context->SetExecuteType(atb::EXECUTE_PRELAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace2, workspaceSize, context);
            aclmdlRIExecuteAsync(model, exeStream);
            context->SetExecuteType(atb::EXECUTE_NORMAL);
        } else if (i >= 8 && i <= 9) {
            // 支持workSpace更新
            operation->Setup(pack, workspaceSize, context);
            if (workspaceSize != 0 && workSpace3 == nullptr) {
                ret = aclrtMalloc(&workSpace3, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                ASSERT_EQ(ret, 0);
            }
            std::cout << "workSpace3:" << workSpace3 << ", workspaceSize:" << workspaceSize << std::endl;
            context->SetExecuteType(atb::EXECUTE_PRELAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace3, workspaceSize, context);
            aclmdlRIExecuteAsync(model, exeStream);
            context->SetExecuteType(atb::EXECUTE_NORMAL);
        } else {
            std::cout << "workspace:" << workSpace << ", workspaceSize:" << workspaceSize << std::endl;
            operation->Setup(pack, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_PRELAUNCH);
            operation->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
            context->SetExecuteType(atb::EXECUTE_NORMAL);
 
            // 重放
            aclmdlRIExecuteAsync(model, exeStream);
        }
        // 流同步，作用是等待device侧任务计算完成
        ret = aclrtSynchronizeStream(exeStream);
        ASSERT_EQ(ret, 0);
        std::cout << "aclrtSynchronizeStream success!" << std::endl;
 
        // 打印输出Tensor的值
        std::vector<uint16_t> outBuffer(atb::Utils::GetTensorNumel(pack.outTensors.at(0)));
        PrintOutTensorValue(pack.outTensors.at(0), outBuffer);
        for (size_t j = 0; j < outBuffer.size(); j++) {
            EXPECT_EQ((uint32_t)outBuffer.at(j), (i + 1) * 16);
        }
    }
 
	// 资源释放
    atb::DestroyOperation(operation);
    atb::DestroyContext(context);
    aclmdlRIDestroy(model);
    for (size_t i = 0; i < pack.inTensors.size(); i++) {
        aclrtFree(pack.inTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack.outTensors.size(); i++) {
        aclrtFree(pack.outTensors.at(i).deviceData);
    }
    aclrtFree(workSpace);
    aclrtFree(workSpace1);
    aclrtFree(workSpace2);
    aclrtFree(workSpace3);
    aclrtDestroyStream(exeStream);
    aclrtResetDevice(deviceId);
}