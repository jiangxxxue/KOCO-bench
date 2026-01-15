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

const int32_t DEVICE_ID = 0;
const uint32_t DIM_0 = 128;
const uint32_t DIM_1 = 256;
const int32_t BEGIN_NORM_AXIS = 1;

/**
 * @brief 准备atb::VariantPack中的所有输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors atb::VariantPack中的输入tensor
 * @return atb::Status 错误码
 */
atb::Status PrepareInTensor(atb::Context *contextPtr, aclrtStream stream, atb::SVector<atb::Tensor> &inTensors)
{
    atb::Tensor x;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<float>(DIM_0 * DIM_1, 2.0), ACL_FLOAT16,
                                        aclFormat::ACL_FORMAT_ND, {DIM_0, DIM_1}, x));
    atb::Tensor gamma;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<float>(DIM_1, 2.0), ACL_FLOAT16,
                                        aclFormat::ACL_FORMAT_ND, {DIM_1}, gamma));
    atb::Tensor beta;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<float>(DIM_1, 1.0), ACL_FLOAT16,
                                        aclFormat::ACL_FORMAT_ND, {DIM_1}, beta));
    inTensors = {x, gamma, beta};
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个beginNormAxis为1的layernorm operation
 * @param layerNormOp 创建一个Operation指针
 * @return atb::Status 错误码
 */
atb::Status CreateLayerNormOperation(atb::Operation **layerNormOp)
{
    atb::infer::LayerNormParam param;
    param.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    param.normParam.beginNormAxis = BEGIN_NORM_AXIS;
    return atb::CreateOperation(param, layerNormOp);
}

int main(int argc, char **argv)
{
    // 设置卡号、创建context、设置stream
    atb::Context *context = nullptr;
    void *stream = nullptr;

    CHECK_STATUS(aclInit(nullptr));
    CHECK_STATUS(aclrtSetDevice(DEVICE_ID));
    CHECK_STATUS(atb::CreateContext(&context));
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);

    // 创建op
    atb::Operation *layerNormOp = nullptr;
    CHECK_STATUS(CreateLayerNormOperation(&layerNormOp));
    // 准备输入tensor
    atb::VariantPack variantPack;
    CHECK_STATUS(PrepareInTensor(context, stream,  variantPack.inTensors)); // 放入输入tensor
    atb::Tensor tensorOut;
    CHECK_STATUS(CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {DIM_0, DIM_1}, tensorOut));
    variantPack.outTensors = {tensorOut}; // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspace大小
    CHECK_STATUS(layerNormOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // layernorm执行
    CHECK_STATUS(layerNormOp->Execute(variantPack, workspacePtr, workspaceSize, context));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成

    // 释放资源
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    CHECK_STATUS(aclrtFree(tensorOut.deviceData));
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    CHECK_STATUS(atb::DestroyOperation(layerNormOp)); // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS(aclFinalize());
    std::cout << "Layernorm demo success!" << std::endl;
    return 0;
}
