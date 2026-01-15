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

const uint32_t NTOKENS = 2;                            // token数量
const uint32_t BATCH_SIZE = NTOKENS;                   // batch数量
const uint32_t MAX_SEQ_LEN = 1024;                     // 最大序列长度
const uint32_t HEAD_NUM = 32;                          // 头数
const uint32_t KV_HEAD_NUM = 32;                       // kv头数
const uint32_t HEAD_SIZE = 128;                        // 头大小
const uint32_t BLOCK_NUM = 16;                         // 块数量
const uint32_t BLOCK_SIZE = 128;                       // 块大小
const uint32_t MAX_CONTEXT_LEN = 1024;                 // 上下文最大长度
std::vector<int32_t> contextLensData(BATCH_SIZE, 256); // contextLens的host侧数据

/**
 * @brief 准备atb::VariantPack中的所有输入tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensors atb::VariantPack中的输入tensor
 * @return atb::Status 错误码
 */
atb::Status PrepareInTensor(atb::Context *contextPtr, aclrtStream stream, atb::SVector<atb::Tensor> &inTensors)
{
    // 创建query tensor
    std::vector<float> queryData(NTOKENS * HEAD_NUM * HEAD_SIZE, 1.0);
    atb::Tensor query;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, queryData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                        {NTOKENS, HEAD_NUM, HEAD_SIZE}, query));
    // 创建key，value tensor
    std::vector<float> kvCacheData(BLOCK_NUM * BLOCK_SIZE * KV_HEAD_NUM * HEAD_SIZE, 1.0);
    atb::Tensor kCache;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, kvCacheData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                        {BLOCK_NUM, BLOCK_SIZE, KV_HEAD_NUM, HEAD_SIZE}, kCache));
    atb::Tensor vCache;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, kvCacheData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                        {BLOCK_NUM, BLOCK_SIZE, KV_HEAD_NUM, HEAD_SIZE}, vCache));
    // 创建blockTables
    uint32_t maxNumBlocksPerQuery = (MAX_CONTEXT_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<int32_t> blockTablesData(NTOKENS * maxNumBlocksPerQuery, 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, BLOCK_NUM - 2);
    for (size_t i = 0; i < blockTablesData.size(); i++) {
        blockTablesData[i] = dist(gen);
    }
    atb::Tensor blockTables;
    CHECK_STATUS(CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {NTOKENS, maxNumBlocksPerQuery}, blockTables));
    CHECK_STATUS(aclrtMemcpy(blockTables.deviceData, blockTables.dataSize, blockTablesData.data(),
                             sizeof(int32_t) * blockTablesData.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    // 创建contextLens，host侧tensor
    atb::Tensor contextLens;
    CHECK_STATUS(CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {BATCH_SIZE}, contextLens));
    contextLens.hostData = contextLensData.data();
    // 创建norm mask，值为-inf的上三角mask
    std::vector<float> maskData(BATCH_SIZE * MAX_SEQ_LEN, 0);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < MAX_SEQ_LEN; ++j) {
            maskData[i * MAX_SEQ_LEN + j] = -32768; // 32768 : -inf
        }
    }
    atb::Tensor mask;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, maskData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND,
                                        {BATCH_SIZE, 1, MAX_SEQ_LEN}, mask));
    // 根据顺序将所有输入tensor放入SVector
    inTensors = {query, kCache, vCache, blockTables, contextLens, mask};
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个PA的Operation，并设置参数
 * @param atb::Operation * 返回一个Operation指针
 * @return atb::Status 错误码
 */
atb::Status PrepareOperation(atb::Operation **paOp)
{
    atb::infer::PagedAttentionParam paOpParam;
    paOpParam.maskType = atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM;
    paOpParam.headNum = HEAD_NUM;
    paOpParam.kvHeadNum = KV_HEAD_NUM;
    paOpParam.qkScale = 1 / sqrt(HEAD_SIZE);
    return atb::CreateOperation(paOpParam, paOp);
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

    // PA示例
    atb::Operation *paOp;
    PrepareOperation(&paOp);
    // 准备输入张量
    atb::VariantPack paVariantPack;
    CHECK_STATUS(PrepareInTensor(context, stream, paVariantPack.inTensors)); // 放入输入tensor
    atb::Tensor tensorOut;
    CreateTensor(ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NTOKENS, HEAD_NUM, HEAD_SIZE}, tensorOut);
    paVariantPack.outTensors.push_back(tensorOut); // 放入输出tensor

    uint64_t workspaceSize = 0;
    // 计算workspaceSize大小
    CHECK_STATUS(paOp->Setup(paVariantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // PA执行
    paOp->Execute(paVariantPack, workspacePtr, workspaceSize, context);
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    CHECK_STATUS(aclrtFree(tensorOut.deviceData));
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    CHECK_STATUS(atb::DestroyOperation(paOp)); // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS((aclFinalize()));
    std::cout << "PA demo success!" << std::endl;
    return 0;
}
