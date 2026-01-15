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
 
static void CreateGraphOperation(atb::GraphParam &opGraph, atb::Operation **operation)
{
 
    opGraph.inTensorNum = 4;
    opGraph.outTensorNum = 1;
    opGraph.internalTensorNum = 3;
    opGraph.nodes.resize(2);
 
    size_t nodeId = 0;
    atb::Node &topkToppSamplingNode = opGraph.nodes.at(nodeId++);
    atb::Node &cumsumNode = opGraph.nodes.at(nodeId++);
 
    atb::infer::TopkToppSamplingParam topkToppSamplingParam;
    topkToppSamplingParam.topkToppSamplingType =
        atb::infer::TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_LOGPROBS_SAMPLING;
    atb::CreateOperation(topkToppSamplingParam, &topkToppSamplingNode.operation);
    topkToppSamplingNode.inTensorIds = {0, 1, 2, 3};
    topkToppSamplingNode.outTensorIds = {5, 6, 7};
 
    atb::infer::CumsumParam cumsumParam;
    SVector<int64_t> axes = {1};
    cumsumParam.axes = axes;
    atb::CreateOperation(cumsumParam, &cumsumNode.operation);
    cumsumNode.inTensorIds = {6};
    cumsumNode.outTensorIds = {4};
 
    atb::CreateOperation(opGraph, operation);
}
 
static void CreateInTensorDescs(atb::SVector<atb::TensorDesc> &intensorDescs) 
{
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {{Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {4, 50}},
                                                      {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, {4, 1}},
                                                      {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {4, 1}},
                                                      {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, {4, 1}}};
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, intensorDescs);
}
 
static void CreateInTensors(atb::SVector<atb::Tensor> &inTensors, atb::SVector<atb::TensorDesc> &intensorDescs)
{
    std::vector<char> zeroData(8, 0);
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
 
TEST(TestGraphNullInternalTensor, TestOne)
{
    if (!atb::GetSingleton<atb::Config>().Is910B() && !atb::GetSingleton<atb::Config>().Is310P()) {
        GTEST_SKIP() << "This test case does not support 910A";
    }
    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    std::vector<aclrtStream> streams = {stream};
    context->SetExecuteStreams(streams);
 
    atb::Operation *graphOp = nullptr;
    atb::GraphParam graphParam;
    CreateGraphOperation(graphParam, &graphOp);
 
    atb::VariantPack pack;
    atb::SVector<atb::TensorDesc> intensorDescs;
    atb::SVector<atb::TensorDesc> outtensorDescs;
 
    uint32_t inTensorNum = 4;
    uint32_t outTensorNum = 1;
    pack.inTensors.resize(inTensorNum);
    intensorDescs.resize(inTensorNum);
 
    CreateInTensorDescs(intensorDescs);
    CreateInTensors(pack.inTensors, intensorDescs);
 
    outtensorDescs.resize(outTensorNum);
    outtensorDescs.at(0).dtype = ACL_FLOAT16;
    outtensorDescs.at(0).format = ACL_FORMAT_ND;
    outtensorDescs.at(0).shape.dimNum = 2;
    outtensorDescs.at(0).shape.dims[0] = 4;
    outtensorDescs.at(0).shape.dims[1] = 1;
    pack.outTensors.resize(outTensorNum);
    CreateOutTensors(pack.outTensors, outtensorDescs);
 
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
 
    atb::Status st1 = graphOp->Execute(pack, (uint8_t*)workSpace, workspaceSize, context);
 
    atb::Status st = (st1 == atb::NO_ERROR) ? atb::NO_ERROR : atb::ERROR_INVALID_GRAPH;
 
    atb::DestroyOperation(graphOp);
    atb::DestroyContext(context);
    for (size_t i = 0; i < pack.inTensors.size(); i++) {
        aclrtFree(pack.inTensors.at(i).deviceData);
    }
    for (size_t i = 0; i < pack.outTensors.size(); i++) {
        aclrtFree(pack.outTensors.at(i).deviceData);
    }
    aclrtFree(workSpace);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    
    ASSERT_EQ(st, atb::NO_ERROR);
}