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

const uint32_t BATCH_SIZE = 1;   // 批处理大小
const uint32_t NTOKENS = 4;      // TOKEN大小
const uint32_t HIDDENSIZEQ = 16; // Q 隐藏层大小
const uint32_t HIDDENSIZEK = 16; // K 隐藏层大小
const uint32_t HEAD_SIZE = 8;    // 头大小

/**
 * @brief 创建一个rope operation
 * @param rmsNormOp 创建一个Operation指针
 * @return atb::Status 错误码
 */
atb::Status PrepareOperation(atb::Operation **ropeOp)
{
    atb::infer::RopeParam opParam;
    opParam.rotaryCoeff = 4;
    opParam.cosFormat = 0;
    return atb::CreateOperation(opParam, ropeOp);
}

/**
 * @brief 准备atb::VariantPack中的所有输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors atb::VariantPack中的输入tensor
 * @return atb::Status 错误码
 */
atb::Status PrepareInTensor(atb::Context *contextPtr, aclrtStream stream, atb::SVector<atb::Tensor> &inTensors)
{
    std::vector<float> qData(NTOKENS * HIDDENSIZEQ, 1.0);
    atb::Tensor tensorQ;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, qData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                        {NTOKENS, HIDDENSIZEQ}, tensorQ));
    std::vector<float> kData(NTOKENS * HIDDENSIZEK, 1.0);
    atb::Tensor tensorK;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, kData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                        {NTOKENS, HIDDENSIZEK}, tensorK));
    std::vector<float> cos(NTOKENS * HEAD_SIZE, 1.0);
    atb::Tensor tensorCos;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, cos, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                        {NTOKENS, HEAD_SIZE}, tensorCos));
    std::vector<float> sin(NTOKENS * HEAD_SIZE, 1.0);
    atb::Tensor tensorSin;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, sin, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                        {NTOKENS, HEAD_SIZE}, tensorSin));
    std::vector<int32_t> seqLenHost(BATCH_SIZE, 4);
    atb::Tensor tensorSeqLen;
    CHECK_STATUS(CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE}, tensorSeqLen));
    tensorSeqLen.hostData = seqLenHost.data();
    inTensors = {tensorQ, tensorK, tensorCos, tensorSin, tensorSeqLen};
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
    // 算子实例
    atb::Operation *ropeOp = nullptr;
    CHECK_STATUS(PrepareOperation(&ropeOp));
    // 准备输入张量
    atb::VariantPack variantPack;
    CHECK_STATUS(PrepareInTensor(context, stream, variantPack.inTensors));
    // 准备输入张量ropeQ和ropeK
    atb::Tensor ropeQ;
    CHECK_STATUS(CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NTOKENS, HIDDENSIZEQ}, ropeQ));
    atb::Tensor ropeK;
    CHECK_STATUS(CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NTOKENS, HIDDENSIZEK}, ropeK));
    variantPack.outTensors = {ropeQ, ropeK};
    uint64_t workspaceSize = 0;
    CHECK_STATUS(ropeOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // 算子执行
    ropeOp->Execute(variantPack, workspacePtr, workspaceSize, context);
    CHECK_STATUS(aclrtSynchronizeStream(stream));
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    // 资源释放
    CHECK_STATUS(atb::DestroyOperation(ropeOp));
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(atb::DestroyContext(context));
    CHECK_STATUS(aclFinalize());
    std::cout << "Rope demo success!" << std::endl;
    return 0;
}
