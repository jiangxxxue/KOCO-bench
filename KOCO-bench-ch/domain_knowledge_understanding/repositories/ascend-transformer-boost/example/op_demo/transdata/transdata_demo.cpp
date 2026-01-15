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

const uint32_t BATCH_SIZE = 8;   // 批处理大小
const uint32_t SEQ_LEN = 100;    // 序列长度
const uint32_t HIDDEN_SIZE = 30; // 隐藏层维度

/**
 * @brief 准备atb::VariantPack中的所有输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors 创建输入Tensors
 * @return atb::Status atb错误码
 */
atb::Status PrepareInTensors(atb::Context *contextPtr, aclrtStream stream, atb::SVector<atb::Tensor> &inTensors)
{
    // 创建一个[BATCH_SIZE*SEQ_LEN*HIDDEN_SIZE]的vector，其中各个值为取值范围为[-100,100)的随机数
    std::vector<float> inTensorData(BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE, 1.0);
    // 创建输入Tensor
    atb::Tensor inTensor;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, inTensorData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                        {BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE}, inTensor));
    inTensors = {inTensor};
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个ND转NZ的TransdataOperation，并设置参数
 * @param atb::Operation * 创建一个Operation指针
 * @return atb::Status atb错误码
 */
atb::Status PrepareOperation(atb::Operation **transdataOp)
{
    atb::infer::TransdataParam opParam;
    opParam.transdataType = atb::infer::TransdataParam::TransdataType::ND_TO_FRACTAL_NZ;
    return atb::CreateOperation(opParam, transdataOp);
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

    // TransdataOp ND to NZ示例
    atb::Operation *transdataOp;
    PrepareOperation(&transdataOp);
    // 准备VariantPack
    uint32_t ALIGN_16 = 16;
    atb::VariantPack variantPack;
    CHECK_STATUS(PrepareInTensors(context, stream, variantPack.inTensors)); // 放入输入tensor
    atb::Tensor outTensor;
    CHECK_STATUS(CreateTensor(
        ACL_FLOAT16, aclFormat::ACL_FORMAT_FRACTAL_NZ,
        {BATCH_SIZE, (HIDDEN_SIZE + ALIGN_16 - 1) / ALIGN_16, (SEQ_LEN + ALIGN_16 - 1) / ALIGN_16 * ALIGN_16, ALIGN_16},
        outTensor));
    variantPack.outTensors.push_back(outTensor); // 放入输出tensor
    uint64_t workspaceSize = 0;
    // Transdata ND to NZ准备工作
    CHECK_STATUS(transdataOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // Transdata ND to NZ执行
    CHECK_STATUS(transdataOp->Execute(variantPack, workspacePtr, workspaceSize, context));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成

    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    for (atb::Tensor &outTensor : variantPack.outTensors) {
        CHECK_STATUS(aclrtFree(outTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    CHECK_STATUS(atb::DestroyOperation(transdataOp)); // operation，对象概念，先释放
    std::cout << "Transdata ND to NZ demo success!" << std::endl;

    // 资源释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(atb::DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS(aclFinalize());
    return 0;
}