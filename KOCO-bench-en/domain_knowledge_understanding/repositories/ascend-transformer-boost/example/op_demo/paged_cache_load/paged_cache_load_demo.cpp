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

const uint32_t NUM_BLOCKS = 4;       // BLOCK数量
const uint32_t BLOCK_SIZE = 128;       // BLOCK大小
const uint32_t NUM_HEADS = 4;       // 头大小
const uint32_t HEAD_SIZE_K = 32;     // K 头大小
const uint32_t HEAD_SIZE_V = 32;     // V 头大小
const uint32_t ELENUM_ALIGNED = 32;  // 32字节对齐 int8为32 其他情况为16
const uint32_t LEN_CONTEXT_LENS = 3; // contextlens的长度
const uint32_t NUM_TOKENS = 384;     // TOKEN 数量

/**
 * @brief 创建一个 paged_cache_load operation
 * @param PagedCacheLoadOperation 创建一个Operation指针
 * @return atb::Status 错误码
 */
atb::Status PrepareOperation(atb::Operation **pagedCacheLoadOp)
{
    atb::infer::PagedCacheLoadParam opParam;
    opParam.kvCacheCfg = atb::infer::PagedCacheLoadParam::K_CACHE_V_CACHE_NZ;
    opParam.isSeqLensCumsumMode = false;
    opParam.hasSeqStarts = false;
    return atb::CreateOperation(opParam, pagedCacheLoadOp);
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
    std::vector<int8_t> kCacheData(NUM_BLOCKS * int32_t(NUM_HEADS * HEAD_SIZE_K / ELENUM_ALIGNED) * BLOCK_SIZE * ELENUM_ALIGNED, 1);
    atb::Tensor tensorKCache;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, kCacheData, ACL_INT8, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                                        {NUM_BLOCKS, int32_t(NUM_HEADS * HEAD_SIZE_K / ELENUM_ALIGNED), BLOCK_SIZE, ELENUM_ALIGNED}, tensorKCache));
    std::vector<int8_t> vCacheData(NUM_BLOCKS * int32_t(NUM_HEADS * HEAD_SIZE_V / ELENUM_ALIGNED) * BLOCK_SIZE * ELENUM_ALIGNED, 1);
    atb::Tensor tensorVCache;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, vCacheData, ACL_INT8, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                                        {NUM_BLOCKS, int32_t(NUM_HEADS * HEAD_SIZE_V / ELENUM_ALIGNED), BLOCK_SIZE, ELENUM_ALIGNED}, tensorVCache));
    std::vector<int32_t> blockTableData = {2, 0, 0};
    atb::Tensor blockTable;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, blockTableData, ACL_INT32, aclFormat::ACL_FORMAT_ND,
                                        {LEN_CONTEXT_LENS, 1}, blockTable));
    std::vector<int32_t> contextLensData = {128, 128, 128};
    atb::Tensor contextLens;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, contextLensData, ACL_INT32, aclFormat::ACL_FORMAT_ND,
                                        {LEN_CONTEXT_LENS}, contextLens));
    std::vector<int8_t> kData(NUM_TOKENS * NUM_HEADS * HEAD_SIZE_K, 1);
    atb::Tensor tensorK;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, kData, ACL_INT8, aclFormat::ACL_FORMAT_ND,
                                        {NUM_TOKENS, NUM_HEADS * HEAD_SIZE_K}, tensorK));
    std::vector<int8_t> vData(NUM_TOKENS * NUM_HEADS * HEAD_SIZE_V, 1);
    atb::Tensor tensorV;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, vData, ACL_INT8, aclFormat::ACL_FORMAT_ND,
                                        {NUM_TOKENS, NUM_HEADS * HEAD_SIZE_V}, tensorV));

    inTensors = {tensorKCache, tensorVCache, blockTable, contextLens, tensorK, tensorV};
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
    // 准备输出张量kCache和vCache
    atb::Tensor kCache;
    CHECK_STATUS(CreateTensor(ACL_INT8, aclFormat::ACL_FORMAT_ND, {NUM_TOKENS, NUM_HEADS * HEAD_SIZE_K}, kCache));
    atb::Tensor vCache;
    CHECK_STATUS(CreateTensor(ACL_INT8, aclFormat::ACL_FORMAT_ND, {NUM_TOKENS, NUM_HEADS * HEAD_SIZE_V}, vCache));
    variantPack.outTensors = {kCache, vCache};
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
    std::cout << "PagedCacheLoad demo success!" << std::endl;
    return 0;
}
