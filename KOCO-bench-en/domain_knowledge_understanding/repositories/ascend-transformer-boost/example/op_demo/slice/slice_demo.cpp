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
 * @brief 准备atb::VariantPack中的所有输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors atb::VariantPack中的输入tensor
 * @return atb::Status atb错误码
 */
atb::Status PrepareInTensor(atb::Context *contextPtr, aclrtStream stream, atb::SVector<atb::Tensor> &inTensors)
{
    uint32_t dim0 = 3;
    uint32_t dim1 = 6;
    // 创建tensor0
    std::vector<float> tensorzero{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    atb::Tensor tensorZero;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, tensorzero, ACL_FLOAT, aclFormat::ACL_FORMAT_ND,
                                        {dim0, dim1}, tensorZero));

    // 根据顺序将所有输入tensor放入SVector
    inTensors = {tensorZero};
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个slice的Operation，并设置参数
 * @param 创建一个Operation指针
 * @return atb::Status atb错误码
 */
atb::Status PrepareOperation(atb::Operation **op)
{
    atb::infer::SliceParam sliceParam;
    sliceParam.offsets = {1, 2};
    sliceParam.size = {2, 4};
    return atb::CreateOperation(sliceParam, op);
}

int main(int argc, char **argv)
{
    // 1.设置卡号、创建context、设置stream
    CHECK_STATUS(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_STATUS(aclrtSetDevice(deviceId));
    atb::Context *context = nullptr;
    CHECK_STATUS(atb::CreateContext(&context));
    void *stream = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream));
    CHECK_STATUS(context->SetExecuteStream(stream));

    // Slice示例
    atb::Operation *op = nullptr;
    CHECK_STATUS(PrepareOperation(&op));
    // 准备输入张量
    atb::VariantPack variantPack;
    CHECK_STATUS(PrepareInTensor(context, stream, variantPack.inTensors)); // 放入输入tensor
    atb::Tensor tensorOut;
    CHECK_STATUS(CreateTensor(ACL_FLOAT, aclFormat::ACL_FORMAT_ND, {2, 4}, tensorOut)); // 创建输出tensor
    variantPack.outTensors.push_back(tensorOut);                                        // 放入输出tensor

    // 计算workspace大小
    uint64_t workspaceSize = 0;
    CHECK_STATUS(op->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // execute阶段
    CHECK_STATUS(op->Execute(variantPack, workspacePtr, workspaceSize, context));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成

    // 释放内存
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    // 资源释放
    CHECK_STATUS(atb::DestroyOperation(op)); // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(atb::DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS((aclFinalize()));
    std::cout << "slice demo success!" << std::endl;
    return 0;
}