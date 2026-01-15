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

const uint32_t DIM1 = 2;
const uint32_t DIM2 = 3;

/**
 * @brief 准备atb::VariantPack中的所有输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors 创建输入Tensors
 * @return atb::Status atb错误码
 */
atb::Status PrepareInTensor(atb::Context *contextPtr, aclrtStream stream, atb::SVector<atb::Tensor> &inTensors)
{
    // 创建x tensor
    std::vector<float> xData(DIM1 * DIM2, 1.0);
    std::vector<int64_t> xShape = {DIM1, DIM2};
    atb::Tensor tensorX;
    CHECK_STATUS(
        CreateTensorFromVector(contextPtr, stream, xData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, xShape, tensorX));
    inTensors = {tensorX};
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个ND转NZ的TransdataOperation，并设置参数
 * @param transposeOp 创建一个Operation指针
 * @return atb::Status atb错误码
 */
atb::Status PrepareOperation(atb::Operation **transposeOp)
{
    atb::infer::TransposeParam transposeOpParam;
    atb::SVector<int32_t> perm = {1, 0};
    transposeOpParam.perm = perm;
    return atb::CreateOperation(transposeOpParam, transposeOp);
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
    CHECK_STATUS(context->SetExecuteStream(stream));

    // Transpose示例
    atb::Operation *transposeOp;
    PrepareOperation(&transposeOp);
    // 准备输入tensor
    atb::VariantPack transposeVariantPack;
    PrepareInTensor(context, stream, transposeVariantPack.inTensors); // 放入输入tensor
    atb::Tensor tensorOut;
    CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {DIM2, DIM1}, tensorOut);
    transposeVariantPack.outTensors = {tensorOut}; // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspace大小
    CHECK_STATUS(transposeOp->Setup(transposeVariantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // reduce执行
    CHECK_STATUS(transposeOp->Execute(transposeVariantPack, workspacePtr, workspaceSize, context));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    for (atb::Tensor &inTensor : transposeVariantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    CHECK_STATUS(atb::DestroyOperation(transposeOp)); // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context)); // context，全局资源，后释放
    std::cout << "Transpose demo success!" << std::endl;
    return 0;
}
