/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <random>
#include "../demo_util.h"

const uint32_t BATCH_SIZE = 16;    // 批处理大小
const uint32_t SEQ_LEN = 1024;     // 序列长度
const uint32_t HIDDEN_SIZE = 4096; // 隐藏层维度

/**
 * @brief 准备atb::VariantPack中的输入tensor
 * @param atb::SVector<atb::Tensor> 返回{[BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]}的输入tensor
 */
atb::Status PrepareInTensor(atb::SVector<atb::Tensor> &inTensors)
{
    // 创建一个[BATCH_SIZE*SEQ_LEN*HIDDEN_SIZE]的vector，其中各个值为取值范围为[-100,100)的随机数
    std::vector<float> inTensorData(BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
    for (float &val : inTensorData) {
        val = dis(gen);
    }
    // 创建输入tensor
    atb::Tensor inTensor;
    CreateTensor(ACL_FLOAT, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE}, inTensor);
    CHECK_STATUS(aclrtMemcpy(inTensor.deviceData, inTensor.dataSize, inTensorData.data(),
                             sizeof(float) * inTensorData.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    inTensors = {inTensor};
    std::cout << "finished PrepareInTensor!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    return atb::NO_ERROR;
}

/**
 * @brief 创建一个Gelu Activation的Operation，并设置参数
 * @param opPtr 创建一个Operation指针
 * @return atb::Status atb错误码
 */
atb::Status GeluOperation(atb::Operation **opPtr)
{
    atb::infer::ActivationParam opParam;
    opParam.activationType = atb::infer::ActivationType::ACTIVATION_FASTER_GELU_FORWARD;
    CHECK_STATUS(atb::CreateOperation(opParam, opPtr));
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个Swiglu_forward Activation的Operation，并设置参数
 * @param atb::Operation * 创建一个Operation指针
 * @return atb::Status atb错误码
 */
atb::Status SwigluOperation(atb::Operation **opPtr)
{
    atb::infer::ActivationParam opParam;
    opParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
    opParam.dim = -1;
    CHECK_STATUS(atb::CreateOperation(opParam, opPtr));
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 进行Activation Gelu示例调用
 * @param context context指针
 * @param stream stream
 * @return atb::Status atb错误码
 */
atb::Status RunGeluDemo(atb::Context *context, void *stream)
{
    // Activation Gelu示例
    atb::Operation *geluOp = nullptr;
    CHECK_STATUS(GeluOperation(&geluOp));
    // 准备VariantPack
    atb::VariantPack geluVariantPack;
    CHECK_STATUS(PrepareInTensor(geluVariantPack.inTensors)); // 放入输入tensor
    atb::Tensor tensorOut;
    CreateTensor(ACL_FLOAT, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE}, tensorOut);
    geluVariantPack.outTensors.push_back(tensorOut); // 放入输出tensor
    uint64_t geluWorkspaceSize = 0;
    // Gelu Activation准备工作
    CHECK_STATUS(geluOp->Setup(geluVariantPack, geluWorkspaceSize, context));
    uint8_t *geluWorkspacePtr = nullptr;
    if (geluWorkspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&geluWorkspacePtr), geluWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // Gelu Activation执行
    geluOp->Execute(geluVariantPack, geluWorkspacePtr, geluWorkspaceSize, context);
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    for (atb::Tensor &inTensor : geluVariantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    for (atb::Tensor &outTensor : geluVariantPack.outTensors) {
        CHECK_STATUS(aclrtFree(outTensor.deviceData));
    }
    if (geluWorkspaceSize > 0) {
        CHECK_STATUS(aclrtFree(geluWorkspacePtr));
    }
    CHECK_STATUS(atb::DestroyOperation(geluOp)); // operation，对象概念，先释放
    std::cout << "Gelu Activation demo success!" << std::endl;
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 进行Activation Swiglu示例调用
 * @param context context指针
 * @return atb::Status atb错误码
 */
atb::Status RunSwigluDemo(atb::Context *context, void *stream)
{
    // Activation Swiglu_forward示例
    atb::Operation *swigluOp = nullptr;
    CHECK_STATUS(SwigluOperation(&swigluOp));
    // 准备VariantPack
    atb::VariantPack swigluVariantPack;
    CHECK_STATUS(PrepareInTensor(swigluVariantPack.inTensors)); // 放入输入tensor
    atb::Tensor tensorOut;
    CreateTensor(ACL_FLOAT, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE / 2}, tensorOut);
    swigluVariantPack.outTensors.push_back(tensorOut); // 放入输出tensor
    uint64_t swigluWorkspaceSize = 0;
    // Swiglu Activation准备工作
    CHECK_STATUS(swigluOp->Setup(swigluVariantPack, swigluWorkspaceSize, context));
    uint8_t *swigluWorkspacePtr = nullptr;
    if (swigluWorkspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&swigluWorkspacePtr), swigluWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // Swiglu Activation执行
    swigluOp->Execute(swigluVariantPack, swigluWorkspacePtr, swigluWorkspaceSize, context);
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    for (atb::Tensor &inTensor : swigluVariantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    for (atb::Tensor &outTensor : swigluVariantPack.outTensors) {
        CHECK_STATUS(aclrtFree(outTensor.deviceData));
    }
    if (swigluWorkspaceSize > 0) {
        CHECK_STATUS(aclrtFree(swigluWorkspacePtr));
    }
    CHECK_STATUS(atb::DestroyOperation(swigluOp)); // operation，对象概念，先释放
    std::cout << "Swiglu_forward Activation demo success!" << std::endl;
    return atb::ErrorType::NO_ERROR;
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
    RunGeluDemo(context, stream);
    RunSwigluDemo(context, stream);
    // 资源释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(atb::DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS(aclFinalize());
    return 0;
}
