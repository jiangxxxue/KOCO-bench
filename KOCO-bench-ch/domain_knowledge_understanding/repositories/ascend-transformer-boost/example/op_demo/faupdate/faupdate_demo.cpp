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
const uint32_t LES_DIM_0 = 8;
const uint32_t LES_DIM_1 = 16384;
const uint32_t LOCALOUT_DIM_0 = 8;
const uint32_t LOCALOUT_DIM_1 = 16384;
const uint32_t HEAD_SIZE = 128;
const uint32_t SP_PARA_DEGREE = 8;

/**
 * @brief 准备atb::VariantPack
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors atb::VariantPack中的输入tensor
 * @return atb::Status 错误码
 */
atb::Status PrepareInTensor(atb::Context *contextPtr, aclrtStream stream, atb::SVector<atb::Tensor> &inTensors)
{
    // 创建shape为[8, 16384]的输入grad tensor
    atb::Tensor lse;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<float>{1, 2, 3, 4, 5, 6}, ACL_FLOAT,
                                        aclFormat::ACL_FORMAT_ND, {LES_DIM_0, LES_DIM_1}, lse));
    // 创建shape为[8, 16384, 128]的输入weight tensor
    atb::Tensor localout;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<float>{1, 2, 3, 4, 5, 6}, ACL_FLOAT,
                                        aclFormat::ACL_FORMAT_ND, {LOCALOUT_DIM_0, LOCALOUT_DIM_1, HEAD_SIZE},
                                        localout));
    inTensors = {lse, localout};
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个faupdate operation
 * @param faupdateOp 创建一个Operation指针
 * @return atb::Status 错误码
 */
atb::Status CreateFaUpdateOperation(atb::Operation **faupdateOp)
{
    atb::infer::FaUpdateParam param;
    param.faUpdateType = atb::infer::FaUpdateParam::FaUpdateType::DECODE_UPDATE;
    param.sp = SP_PARA_DEGREE;
    CHECK_STATUS(atb::CreateOperation(param, faupdateOp));
    return atb::ErrorType::NO_ERROR;
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
    atb::Operation *faupdateOp = nullptr;
    CHECK_STATUS(CreateFaUpdateOperation(&faupdateOp));
    // 准备输入tensor
    atb::VariantPack variantPack;
    CHECK_STATUS(PrepareInTensor(context, stream, variantPack.inTensors)); // 放入输入tensor
    // 准备输出tensor
    atb::Tensor output;
    CHECK_STATUS(CreateTensor(ACL_FLOAT, aclFormat::ACL_FORMAT_ND, {LOCALOUT_DIM_1, HEAD_SIZE}, output));
    variantPack.outTensors = {output}; // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspaceSize大小
    CHECK_STATUS(faupdateOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // faupdate执行
    CHECK_STATUS(faupdateOp->Execute(variantPack, workspacePtr, workspaceSize, context));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成

    // 释放资源
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    for (atb::Tensor &outTensor : variantPack.outTensors) {
        CHECK_STATUS(aclrtFree(outTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    CHECK_STATUS(atb::DestroyOperation(faupdateOp)); // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS(aclFinalize());
    std::cout << "faupdate demo success!" << std::endl;
    return 0;
}