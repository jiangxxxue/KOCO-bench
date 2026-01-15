/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include "c_interface_utils.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"
#include "atb/utils/log.h"

using namespace atb;

const int64_t PCLINOUTPCL = 7;
const int64_t MAXSEQLEN = 1024;

const int64_t elenumAlignedInt8 = 32;
const int64_t elenumAlignedOther = 16;
const int64_t seqLen = 1;

const int64_t batchP1 = 16;
const int64_t numHeadsP1 = 16;
const int64_t headSizeKP1 = 576;
const int64_t headSizeVP1 = 512;
const int64_t blockSizeP1 = 128;
const int64_t numBlocksP1 = 128;

const int64_t batchP2 = 32;
const int64_t numHeadsP2 = 64;
const int64_t headSizeKP2 = 276;
const int64_t headSizeVP2 = 212;
const int64_t blockSizeP2 = 128;
const int64_t numBlocksP2 = 128;

const int64_t batchP3 = 30;
const int64_t numHeadsP3 = 64;
const int64_t headSizeKP3 = 76;
const int64_t headSizeVP3 = 12;
const int64_t blockSizeP3 = 128;
const int64_t numBlocksP3 = 100;

const int64_t batchP4 = 16;
const int64_t numHeadsP4 = 16;
const int64_t headSizeKP4 = 144;
const int64_t headSizeVP4 = 128;
const int64_t blockSizeP4 = 128;
const int64_t numBlocksP4 = 128;

const int64_t batchP5 = 32;
const int64_t numHeadsP5 = 16;
const int64_t headSizeKP5 = 192;
const int64_t headSizeVP5 = 128;
const int64_t blockSizeP5 = 64;
const int64_t numBlocksP5 = 256;

void TestPagedCacheLoadNZ(const int64_t batch, const int64_t numHeads, const int64_t headSizeK, const int64_t headSizeV,
                          const int64_t blockSize, const int64_t numBlocks, const aclDataType dataType)
{
    atb::Context *context = nullptr;
    aclrtStream stream = nullptr;
    int64_t deviceId = 0;
    cinterfaceTest::Init(&context, &stream, &deviceId);
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        ATB_LOG(ERROR) << "Paged Cache Load only supports A2/A3";
        cinterfaceTest::Destroy(&context, &stream);
        GTEST_SKIP();
    }
    uint8_t *inoutHost[PCLINOUTPCL];
    uint8_t *inoutDevice[PCLINOUTPCL];
    aclTensor *tensorList[PCLINOUTPCL];
    int64_t numTokens = batch * seqLen;

    // contextLens
    size_t dataSizeContextLens = numTokens * sizeof(int32_t);
    int32_t *hostDataContextLens = (int32_t *)malloc(dataSizeContextLens);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distribContextLens(1, MAXSEQLEN);
    for (int i = 0; i < numTokens; ++i) {
        hostDataContextLens[i] = distribContextLens(gen);
    }

    // blockTables
    std::vector<int32_t> vec(hostDataContextLens, hostDataContextLens + numTokens);
    int32_t maxVal = *std::max_element(vec.begin(), vec.end());
    int64_t sum = std::accumulate(vec.begin(), vec.end(), int64_t(0));
    size_t maxNumBlocksPerSeq = (maxVal - 1) / blockSize + 1;
    size_t dataSizeBlockTables = numTokens * maxNumBlocksPerSeq * sizeof(int32_t);
    std::uniform_int_distribution<> distribBlockTables(0, numBlocks - 1);
    int32_t *hostDataBlockTables = (int32_t *)malloc(dataSizeBlockTables);
    for (int i = 0; i < numTokens * maxNumBlocksPerSeq; ++i) {
        hostDataBlockTables[i] = distribBlockTables(gen);
    }

    int64_t elenumAligned = dataType == ACL_INT8 ? elenumAlignedInt8 : elenumAlignedOther;
    size_t inoutSize[PCLINOUTPCL] = {
        numBlocks * numHeads * headSizeK / elenumAligned * blockSize * elenumAligned,
        numBlocks * numHeads * headSizeV / elenumAligned * blockSize * elenumAligned,
        numTokens * maxNumBlocksPerSeq,
        numTokens,
        sum * numHeads * headSizeK,
        sum * numHeads * headSizeV,
        numTokens,
    };
    cinterfaceTest::CreateInOutData(PCLINOUTPCL, inoutHost, inoutDevice, inoutSize);
    size_t i = 0;

    // keyCache
    std::vector<int64_t> viewDim = {numBlocks, numHeads * headSizeK / elenumAligned, blockSize, elenumAligned};
    cinterfaceTest::CreateACLTensorInOut(viewDim, dataType, ACL_FORMAT_FRACTAL_NZ, tensorList, i, inoutDevice[i]);

    // valueCache
    viewDim = {numBlocks, numHeads * headSizeV / elenumAligned, blockSize, elenumAligned};
    cinterfaceTest::CreateACLTensorInOut(viewDim, dataType, ACL_FORMAT_FRACTAL_NZ, tensorList, i, inoutDevice[i]);

    // blockTables
    aclrtMemcpy(inoutDevice[i], dataSizeBlockTables, hostDataBlockTables, dataSizeBlockTables,
                ACL_MEMCPY_HOST_TO_DEVICE);
    viewDim = {numTokens, maxNumBlocksPerSeq};
    cinterfaceTest::CreateACLTensorInOut(viewDim, ACL_INT32, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // contextLens
    aclrtMemcpy(inoutDevice[i], dataSizeContextLens, hostDataContextLens, dataSizeContextLens,
                ACL_MEMCPY_HOST_TO_DEVICE);
    viewDim = {numTokens};
    cinterfaceTest::CreateACLTensorInOut(viewDim, ACL_INT32, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // key
    viewDim = {sum, numHeads * headSizeK};
    cinterfaceTest::CreateACLTensorInOut(viewDim, dataType, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // value
    viewDim = {sum, numHeads * headSizeV};
    cinterfaceTest::CreateACLTensorInOut(viewDim, dataType, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // seqStarts
    viewDim = {numTokens};
    cinterfaceTest::CreateACLTensorInOut(viewDim, ACL_INT32, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    uint64_t workspaceSize = 0;
    atb::Operation *op = nullptr;

    Status ret = AtbPagedCacheLoadGetWorkspaceSize(
        tensorList[0], tensorList[1], tensorList[2], tensorList[3], tensorList[4], tensorList[5], tensorList[6],
        atb::infer::PagedCacheLoadParam::KvCacheCfg::K_CACHE_V_CACHE_NZ, false, false, &workspaceSize, &op, context);
    EXPECT_EQ(ret, atb::NO_ERROR);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        EXPECT_EQ(ret, ACL_SUCCESS);
    }
    ret = AtbPagedCacheLoad(workspaceAddr, workspaceSize, op, context);
    EXPECT_EQ(ret, atb::NO_ERROR);
    ret = aclrtSynchronizeStream(stream);

    if (workspaceSize > 0) {
        EXPECT_EQ(aclrtFree(workspaceAddr), ACL_SUCCESS);
    }
    EXPECT_EQ(atb::DestroyOperation(op), atb::NO_ERROR);
    cinterfaceTest::Destroy(&context, &stream);
    for (i = 0; i < PCLINOUTPCL; i++) {
        aclrtFreeHost(inoutHost[i]);
        aclrtFree(inoutDevice[i]);
    }
}

void TestPagedCacheLoadND(const int64_t batch, const int64_t numHeads, const int64_t headSizeK, const int64_t headSizeV,
                          const int64_t blockSize, const int64_t numBlocks, const aclDataType dataType,
                          bool isSeqLensCumsumType, bool hasSeqStarts)
{
    atb::Context *context = nullptr;
    aclrtStream stream = nullptr;
    int64_t deviceId = 0;
    cinterfaceTest::Init(&context, &stream, &deviceId);
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        ATB_LOG(ERROR) << "Paged Cache Load only supports A2/A3";
        cinterfaceTest::Destroy(&context, &stream);
        GTEST_SKIP();
    }
    uint8_t *inoutHost[PCLINOUTPCL];
    uint8_t *inoutDevice[PCLINOUTPCL];
    aclTensor *tensorList[PCLINOUTPCL];
    int64_t numTokens = batch * seqLen;

    // contextLens
    size_t dataSizeContextLens = numTokens * sizeof(int32_t);
    int32_t *hostDataContextLens = (int32_t *)malloc(dataSizeContextLens);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distribContextLens(1, MAXSEQLEN);
    for (int i = 0; i < numTokens; ++i) {
        hostDataContextLens[i] = distribContextLens(gen);
    }

    // 累加和模式
    size_t dataSizeCuContextLens = (numTokens + 1) * sizeof(int32_t);
    int32_t *hostDataCuContextLens = (int32_t *)malloc(dataSizeCuContextLens);
    hostDataCuContextLens[0] = 0;
    for (int i = 1; i < numTokens + 1; ++i) {
        hostDataCuContextLens[i] = hostDataCuContextLens[i - 1] + hostDataContextLens[i];
    }

    // blockTables
    std::vector<int32_t> vec(hostDataContextLens, hostDataContextLens + numTokens);
    int32_t maxVal = *std::max_element(vec.begin(), vec.end());
    size_t maxNumBlocksPerSeq = (maxVal + blockSize - 1) / blockSize + 4;
    size_t dataSizeBlockTables = numTokens * maxNumBlocksPerSeq * sizeof(int32_t);
    std::uniform_int_distribution<> distribBlockTables(0, numBlocks - 1);
    int32_t *hostDataBlockTables = (int32_t *)malloc(dataSizeBlockTables);
    for (int i = 0; i < numTokens * maxNumBlocksPerSeq; ++i) {
        hostDataBlockTables[i] = distribBlockTables(gen);
    }

    size_t inoutSize[PCLINOUTPCL] = {
        numBlocks * blockSize * numHeads * headSizeK,
        numBlocks * blockSize * numHeads * headSizeV,
        numTokens * maxNumBlocksPerSeq,
        numTokens,
        numTokens * numHeads * headSizeK,
        numTokens * numHeads * headSizeV,
        numTokens,
    };
    cinterfaceTest::CreateInOutData(PCLINOUTPCL, inoutHost, inoutDevice, inoutSize);
    size_t i = 0;

    // keyCache
    std::vector<int64_t> viewDim = {numBlocks, blockSize, numHeads, headSizeK};
    cinterfaceTest::CreateACLTensorInOut(viewDim, dataType, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // valueCache
    viewDim = {numBlocks, blockSize, numHeads, headSizeV};
    cinterfaceTest::CreateACLTensorInOut(viewDim, dataType, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // blockTables
    aclrtMemcpy(inoutDevice[i], dataSizeBlockTables, hostDataBlockTables, dataSizeBlockTables,
                ACL_MEMCPY_HOST_TO_DEVICE);
    viewDim = {numTokens, maxNumBlocksPerSeq};
    cinterfaceTest::CreateACLTensorInOut(viewDim, ACL_INT32, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // contextLens
    if (isSeqLensCumsumType) {
        aclrtMemcpy(inoutDevice[i], dataSizeCuContextLens, hostDataCuContextLens, dataSizeCuContextLens,
                    ACL_MEMCPY_HOST_TO_DEVICE);
        viewDim = {numTokens + 1};
        cinterfaceTest::CreateACLTensorInOut(viewDim, ACL_INT32, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);
    } else {
        aclrtMemcpy(inoutDevice[i], dataSizeContextLens, hostDataContextLens, dataSizeContextLens,
                    ACL_MEMCPY_HOST_TO_DEVICE);
        viewDim = {numTokens};
        cinterfaceTest::CreateACLTensorInOut(viewDim, ACL_INT32, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);
    }

    // key
    viewDim = {numTokens, numHeads, headSizeK};
    cinterfaceTest::CreateACLTensorInOut(viewDim, dataType, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // value
    viewDim = {numTokens, numHeads, headSizeV};
    cinterfaceTest::CreateACLTensorInOut(viewDim, dataType, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // seqStarts
    viewDim = {numTokens};
    cinterfaceTest::CreateACLTensorInOut(viewDim, ACL_INT32, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    uint64_t workspaceSize = 0;
    atb::Operation *op = nullptr;

    Status ret = AtbPagedCacheLoadGetWorkspaceSize(tensorList[0], tensorList[1], tensorList[2], tensorList[3],
                                                   tensorList[4], tensorList[5], tensorList[6],
                                                   atb::infer::PagedCacheLoadParam::KvCacheCfg::K_CACHE_V_CACHE_ND,
                                                   isSeqLensCumsumType, hasSeqStarts, &workspaceSize, &op, context);
    EXPECT_EQ(ret, atb::NO_ERROR);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        EXPECT_EQ(ret, ACL_SUCCESS);
    }
    ret = AtbPagedCacheLoad(workspaceAddr, workspaceSize, op, context);
    EXPECT_EQ(ret, atb::NO_ERROR);
    ret = aclrtSynchronizeStream(stream);

    if (workspaceSize > 0) {
        EXPECT_EQ(aclrtFree(workspaceAddr), ACL_SUCCESS);
    }
    EXPECT_EQ(atb::DestroyOperation(op), atb::NO_ERROR);
    cinterfaceTest::Destroy(&context, &stream);
    for (i = 0; i < PCLINOUTPCL; i++) {
        aclrtFreeHost(inoutHost[i]);
        aclrtFree(inoutDevice[i]);
    }
}

TEST(TestATBACL, TestPagedCacheLoadNZP1FP16)
{
    TestPagedCacheLoadNZ(batchP1, numHeadsP1, headSizeKP1, headSizeVP1, blockSizeP1, numBlocksP1, ACL_FLOAT16);
}

TEST(TestATBACL, TestPagedCacheLoadNZP1BF16)
{
    TestPagedCacheLoadNZ(batchP1, numHeadsP1, headSizeKP1, headSizeVP1, blockSizeP1, numBlocksP1, ACL_BF16);
}

TEST(TestATBACL, TestPagedCacheLoadNZP1INT8)
{
    TestPagedCacheLoadNZ(batchP1, numHeadsP1, headSizeKP1, headSizeVP1, blockSizeP1, numBlocksP1, ACL_INT8);
}

TEST(TestATBACL, TestPagedCacheLoadNZP2FP16)
{
    TestPagedCacheLoadNZ(batchP2, numHeadsP2, headSizeKP2, headSizeVP2, blockSizeP2, numBlocksP2, ACL_FLOAT16);
}

TEST(TestATBACL, TestPagedCacheLoadNZP2BF16)
{
    TestPagedCacheLoadNZ(batchP2, numHeadsP2, headSizeKP2, headSizeVP2, blockSizeP2, numBlocksP2, ACL_BF16);
}

TEST(TestATBACL, TestPagedCacheLoadNZP2INT8)
{
    TestPagedCacheLoadNZ(batchP2, numHeadsP2, headSizeKP2, headSizeVP2, blockSizeP2, numBlocksP2, ACL_INT8);
}

TEST(TestATBACL, TestPagedCacheLoadNZP3FP16)
{
    TestPagedCacheLoadNZ(batchP1, numHeadsP3, headSizeKP3, headSizeVP3, blockSizeP3, numBlocksP3, ACL_FLOAT16);
}

TEST(TestATBACL, TestPagedCacheLoadNZP3BF16)
{
    TestPagedCacheLoadNZ(batchP3, numHeadsP3, headSizeKP3, headSizeVP3, blockSizeP3, numBlocksP3, ACL_BF16);
}

TEST(TestATBACL, TestPagedCacheLoadNZP3INT8)
{
    TestPagedCacheLoadNZ(batchP3, numHeadsP3, headSizeKP3, headSizeVP3, blockSizeP3, numBlocksP3, ACL_INT8);
}

TEST(TestATBACL, TestPagedCacheLoadNDP4FP16)
{
    TestPagedCacheLoadND(batchP4, numHeadsP4, headSizeKP4, headSizeVP4, blockSizeP4, numBlocksP4, ACL_FLOAT16, true,
                         true);
}

TEST(TestATBACL, TestPagedCacheLoadNDP4BF16)
{
    TestPagedCacheLoadND(batchP4, numHeadsP4, headSizeKP4, headSizeVP4, blockSizeP4, numBlocksP4, ACL_BF16, true, true);
}

TEST(TestATBACL, TestPagedCacheLoadNDP4INT8)
{
    TestPagedCacheLoadND(batchP4, numHeadsP4, headSizeKP4, headSizeVP4, blockSizeP4, numBlocksP4, ACL_INT8, true, true);
}

TEST(TestATBACL, TestPagedCacheLoadNDP5FP16)
{
    TestPagedCacheLoadND(batchP5, numHeadsP5, headSizeKP5, headSizeVP5, blockSizeP5, numBlocksP5, ACL_FLOAT16, true,
                         true);
}

TEST(TestATBACL, TestPagedCacheLoadNDP5BF16)
{
    TestPagedCacheLoadND(batchP5, numHeadsP5, headSizeKP5, headSizeVP5, blockSizeP5, numBlocksP5, ACL_BF16, true, true);
}

TEST(TestATBACL, TestPagedCacheLoadNDP5INT8)
{
    TestPagedCacheLoadND(batchP5, numHeadsP5, headSizeKP5, headSizeVP5, blockSizeP5, numBlocksP5, ACL_INT8, true, true);
}