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
const uint32_t DIM_0 = 4;
const uint32_t DIM_1 = 1024;
const uint32_t DIM_2 = 5120;

/**
 * @brief 准备atb::VariantPack中的所有输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors atb::VariantPack中的输入tensor
 * @return atb::Status 错误码
 */
atb::Status PrepareInTensor(atb::Context *contextPtr, aclrtStream stream, atb::SVector<atb::Tensor> &inTensors)
{
    // 创建shape为[4, 1024, 5120]的tensor
    atb::Tensor x;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<float>(DIM_0 * DIM_1 * DIM_2, 2.0), ACL_FLOAT16,
                                        aclFormat::ACL_FORMAT_ND, {DIM_0, DIM_1, DIM_2}, x));
    atb::Tensor gamma;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<float>(DIM_2, 2.0), ACL_FLOAT16,
                                        aclFormat::ACL_FORMAT_ND, {DIM_2}, gamma));
    inTensors = {x, gamma};
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个非量化rmsnorm operation
 * @param rmsNormOp 创建一个Operation指针
 * @return atb::Status 错误码
 */
atb::Status CreateRmsNormOperation(atb::Operation **rmsNormOp)
{
    atb::infer::RmsNormParam param;
    param.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    param.normParam.quantType = atb::infer::QuantType::QUANT_UNQUANT;
    return atb::CreateOperation(param, rmsNormOp);
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
    CHECK_STATUS(context->SetExecuteStream(stream));

    // 创建op
    atb::Operation *rmsnormOp = nullptr;
    CHECK_STATUS(CreateRmsNormOperation(&rmsnormOp));
    // 准备输入tensor
    atb::VariantPack variantPack;
    CHECK_STATUS(PrepareInTensor(context, stream, variantPack.inTensors)); // 放入输入tensor
    atb::Tensor tensorOut;
    CHECK_STATUS(CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {DIM_0, DIM_1, DIM_2}, tensorOut));
    variantPack.outTensors = {tensorOut}; // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspace大小
    CHECK_STATUS(rmsnormOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // rmsnorm执行
    rmsnormOp->Execute(variantPack, workspacePtr, workspaceSize, context);
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成

    // 释放资源
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    CHECK_STATUS(aclrtFree(tensorOut.deviceData));
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    CHECK_STATUS(atb::DestroyOperation(rmsnormOp)); // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS(aclFinalize());
    std::cout << "Rmsnorm demo success!" << std::endl;
    return 0;
}
