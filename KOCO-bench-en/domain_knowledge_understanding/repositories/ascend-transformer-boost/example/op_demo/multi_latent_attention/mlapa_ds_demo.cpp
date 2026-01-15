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

const int32_t DEVICE_ID = 1;
const uint32_t blockSize = 128;
int32_t blockNum = 64;
const uint32_t BATCH = 32;
std::vector<int32_t> contextLensHost;

/**
 * @brief 准备atb::VariantPack
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors atb::VariantPack中的输入tensor
 * @return atb::Status 错误码
 */
atb::Status PrepareInTensor(atb::Context *contextPtr, aclrtStream stream, aclDataType dtype, int tokenNum, int headNum,
                            int kSeqLen, atb::SVector<atb::Tensor> &inTensors)
{
    // 创建shape为[tokenNum, headNum, 512]的输入qNope tensor
    atb::Tensor qNope;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(tokenNum * headNum * 512, 1),
                                        ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {tokenNum, headNum, 512}, qNope));
    // 创建shape为[tokenNum, headNum, 64]的输入qRope tensor
    atb::Tensor qRope;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(tokenNum * headNum * 64, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {tokenNum, headNum, 64}, qRope, dtype));
    int maxBlockNumPerSeq = (kSeqLen + blockSize - 1) / blockSize;
    blockNum = tokenNum * maxBlockNumPerSeq;
    // 创建shape为[blockNum, blockSize, kvHeadNum, 512]的输入ctKV tensor
    atb::Tensor ctKV;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(blockNum * blockSize * 512, 1),
                                        ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {blockNum, blockSize, 1, 512}, ctKV));
    // 创建shape为[blockNum, blockSize, kvHeadNum, 64]的输入kRope tensor
    atb::Tensor kRope;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, std::vector<__fp16>(blockNum * blockSize * 64, 0), dtype,
                                        aclFormat::ACL_FORMAT_ND, {blockNum, blockSize, 1, 64}, kRope, dtype));
    // 创建shape为[BATCH, maxBlockNumPerSeq]的输入blockTables tensor
    auto blockTablesHost = std::vector<int32_t>(BATCH * maxBlockNumPerSeq);
    for (size_t i = 0; i < BATCH; i++) {
        for (size_t j = 0; j < maxBlockNumPerSeq; j++) {
            blockTablesHost[i * maxBlockNumPerSeq + j] = i * maxBlockNumPerSeq + j;
        }
    }
    atb::Tensor blockTables;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, blockTablesHost, ACL_INT32, aclFormat::ACL_FORMAT_ND,
                                        {BATCH, maxBlockNumPerSeq}, blockTables));
    // 创建shape为[BATCH]的输入contextLens hostTensor
    contextLensHost = std::vector<int32_t>(BATCH, kSeqLen);
    atb::Tensor contextLens;
    CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {BATCH}, contextLens);
    contextLens.hostData = contextLensHost.data();

    inTensors = {qNope, qRope, ctKV, kRope, blockTables, contextLens};
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个MultiLatentAttentionOperation operation
 * @param mlaOp 创建一个Operation指针
 * @return atb::Status 错误码
 */
atb::Status CreateMultiLatentAttentionOperation(int headNum, atb::Operation **mlaOp)
{
    atb::infer::MultiLatentAttentionParam param;
    param.headNum = headNum;
    param.qkScale = 0.1352667747812271;
    param.kvHeadNum = 1;
    param.cacheMode = atb::infer::MultiLatentAttentionParam::CacheMode::KROPE_CTKV;
    return atb::CreateOperation(param, mlaOp);
}

/**
 * @brief 进行MlaPreprocessOperation的循环调用
 * @param context context指针
 * @param stream stream
 * @param dtype 指定部分输入/输出vector数据类型
 * @param tokenNum 词元数
 * @param headNum 头数
 * @param kSeqLen key/value 的单个词元长度
 * @return atb::Status 错误码
 */
atb::Status RunDemo(atb::Operation *mlaOp, atb::Context *context, void *stream, aclDataType dtype, int tokenNum,
                    int headNum, int kSeqLen)
{
    // 准备输入tensor
    atb::VariantPack variantPack;
    CHECK_STATUS(
        PrepareInTensor(context, stream, dtype, tokenNum, headNum, kSeqLen, variantPack.inTensors)); // 放入输入tensor
    // 准备输出tensor
    atb::Tensor attenOut;
    CHECK_STATUS(CreateTensor(dtype, aclFormat::ACL_FORMAT_ND, {tokenNum, headNum, 512}, attenOut));
    variantPack.outTensors = {attenOut}; // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspaceSize大小
    CHECK_STATUS(mlaOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    for (size_t i = 0; i < 2; i++) {
        std::cout << "tokenNum: " << tokenNum << " headNum: " << headNum << " loop: " << i << std::endl;
        // mlaPreprocess执行
        mlaOp->Execute(variantPack, workspacePtr, workspaceSize, context);
        CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    }
    // 释放资源
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
        for (atb::Tensor &outTensor : variantPack.outTensors) {
            if (outTensor.deviceData == inTensor.deviceData) {
                outTensor.deviceData = nullptr;
            }
        }
        inTensor.deviceData = nullptr;
    }
    for (atb::Tensor &outTensor : variantPack.outTensors) {
        if (outTensor.deviceData == nullptr)
            continue;
        CHECK_STATUS(aclrtFree(outTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    return atb::ErrorType::NO_ERROR;
}

int main(int argc, char **argv)
{
    std::string dtypeStr;
    int tokenNum = 32;
    int headNum = 128;
    int kSeqLen = 526;
    aclDataType dtype = ACL_FLOAT16;
    if (argc == 5) {
        dtypeStr = argv[1];
        tokenNum = std::stoi(argv[2]);
        headNum = std::stoi(argv[3]);
        kSeqLen = std::stoi(argv[4]);
    }
    if (dtypeStr == "bf16") {
        dtype = ACL_BF16;
    }
    // 设置卡号、创建context、设置stream
    atb::Context *context = nullptr;
    void *stream = nullptr;

    CHECK_STATUS(aclInit(nullptr));
    CHECK_STATUS(aclrtSetDevice(DEVICE_ID));
    CHECK_STATUS(atb::CreateContext(&context));
    CHECK_STATUS(aclrtCreateStream(&stream));
    CHECK_STATUS(context->SetExecuteStream(stream));

    // 创建op
    atb::Operation *mlaOp = nullptr;
    CHECK_STATUS(CreateMultiLatentAttentionOperation(headNum, &mlaOp));

    // 执行demo
    CHECK_STATUS(RunDemo(mlaOp, context, stream, dtype, tokenNum, headNum, kSeqLen));
    CHECK_STATUS(atb::DestroyOperation(mlaOp)); // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS(aclFinalize());
    std::cout << "MultiLatentAttention demo success!" << std::endl;
    return 0;
}
