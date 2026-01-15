/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../demo_util.h"

/**
 * @brief 进行Activation Split示例调用
 * @param context context指针
 * @param stream stream
 * @return atb::Status atb错误码
 */
atb::Status RunSplitDemo(atb::Context *context, void *stream)
{
    // 配置Op参数
    atb::infer::SplitParam opParam;
    opParam.splitDim = 1;        // 设定切分轴为1
    opParam.splitNum = 2;        // 设置切分后得到的块数
    opParam.splitSizes = {2, 3}; // 设置不均匀切分时每块大小

    // 准备VariantPack
    atb::VariantPack variantPack;
    std::vector<int64_t> inputXShape = {1, 5, 2};
    std::vector<int64_t> output1Shape = {1, 2, 2};
    std::vector<int64_t> output2Shape = {1, 3, 2};

    std::vector<float> inTensorXData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> output1RefData = {0, 1, 2, 3};
    std::vector<float> output2RefData = {4, 5, 6, 7, 8, 9};

    atb::Tensor inTensorX;
    CHECK_STATUS(CreateTensorFromVector(context, stream, inTensorXData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                        inputXShape, inTensorX));
    atb::Tensor outTensor1;
    CHECK_STATUS(CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, output1Shape, outTensor1));
    atb::Tensor outTensor2;
    CHECK_STATUS(CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, output2Shape, outTensor2));

    variantPack.inTensors = {inTensorX};
    variantPack.outTensors = {outTensor1, outTensor2};

    // 申请SplitOp
    atb::Operation *splitOp = {nullptr};
    CHECK_STATUS(atb::CreateOperation(opParam, &splitOp));
    uint64_t workspaceSize = 0;
    // ATB Operation 第一阶段接口调用：对输入输出进行检查，并根据需要计算workspace大小
    CHECK_STATUS(splitOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // ATB Operation 第二阶段接口调用：执行算子
    CHECK_STATUS(splitOp->Execute(variantPack, workspacePtr, workspaceSize, context));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成

    // 资源释放
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    for (atb::Tensor &outTensor : variantPack.outTensors) {
        CHECK_STATUS(aclrtFree(outTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    return atb::DestroyOperation(splitOp);
}

int main(int argc, char **argv)
{
    // 设置卡号、创建context、设置stream
    CHECK_STATUS(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_STATUS(aclrtSetDevice(deviceId));
    atb::Context *context = nullptr;
    CHECK_STATUS(atb::CreateContext(&context));
    void *stream = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);
    CHECK_STATUS(RunSplitDemo(context, stream));

    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(atb::DestroyContext(context));
    CHECK_STATUS(aclFinalize());
    std::cout << "Split demo success!" << std::endl;
    return 0;
}
