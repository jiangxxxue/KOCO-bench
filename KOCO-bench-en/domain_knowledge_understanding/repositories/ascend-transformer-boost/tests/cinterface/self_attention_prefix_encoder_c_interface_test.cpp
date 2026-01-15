/*

Copyright (c) 2025 Huawei Technologies Co., Ltd.
This file is a part of the CANN Open Software.
Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include "c_interface_utils.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"
#include "atb/utils/log.h"

using namespace atb;
using namespace atb::cinterfaceTest;

const int64_t INOUT_TENSOR_NUM = 9;
const int64_t MASK_INDEX = 4;
const int64_t SEQLEN_INDEX = 5;
const int64_t KV_SEQLEN_INDEX = 6;
const int64_t SLOPES_INDEX = 7;

void TestSelfAttentionPrefixEncoder(const int64_t headNum, const int64_t kvHeadNum, const int64_t headSize,
                                    const int64_t batch, const int64_t numBlocks, const int64_t blockSize,
                                    const float qkScale, const int maskType, const aclDataType dtype,
                                    std::vector<int32_t> qSeqLen, std::vector<int32_t> kvSeqLen)
{
    int inputNum = INOUT_TENSOR_NUM;
    atb::Context *context = nullptr;
    aclrtStream stream = nullptr;
    int64_t deviceId = 0;
    Init(&context, &stream, &deviceId);
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        ATB_LOG(ERROR) << "SelfAttention PrefixEncoder only supports A2/A3";
        Destroy(&context, &stream);
        GTEST_SKIP();
    }
    uint8_t *inoutHost[inputNum];
    uint8_t *inoutDevice[inputNum];
    aclTensor *tensorList[inputNum];
    std::vector inputTypes = {dtype, dtype, dtype, ACL_INT32, dtype, ACL_INT32, ACL_INT32, ACL_FLOAT, dtype};
    int32_t qNTokens = 0;
    for (int i = 0; i < batch; ++i) {
        qNTokens += qSeqLen[i];
    }
    int64_t maxBlockNum = 20;
    std::vector<std::vector<int64_t>> tensorDim = {
        {qNTokens, headNum, headSize},             // query
        {numBlocks, blockSize, headNum, headSize}, // key
        {numBlocks, blockSize, headNum, headSize}, // value
        {batch, maxBlockNum},                      // blockTables
        {128, 128},                                // mask
        {batch},                                   // seqLen
        {batch},                                   // kvSeqLen
        {headNum * headSize},                      // slopes
        {qNTokens, headNum, headSize},             // attnOut
    };
    bool isAlibiMask = maskType == 4 || maskType == 5; // 4, 5: alibi compress mask
    if (isAlibiMask) {
        tensorDim[MASK_INDEX] = {256, 256};
    } else {
        tensorDim[SLOPES_INDEX] = {}; // delete slopes
    }
    if (maskType == 9) {            // 9: casual mask
        tensorDim[MASK_INDEX] = {}; // delete mask
    }
    size_t inoutSize[inputNum];
    int total = 0;
    for (int i = 0; i < inputNum; ++i) {
        if (tensorDim[i].size() == 0) {
            inoutSize[i] = 0;
            continue;
        }
        total = 1;
        for (int j = 0; j < tensorDim[i].size(); ++j) {
            total *= tensorDim[i][j];
        }
        inoutSize[i] = total * aclDataTypeSize(inputTypes[i]);
    }
    CreateInOutData(inputNum, inoutHost, inoutDevice, inoutSize);
    size_t i = 0;

    while (i < tensorDim.size()) {
        if (i == SEQLEN_INDEX) {
            CreateACLTensorInOut(tensorDim[i], inputTypes[i], ACL_FORMAT_ND, tensorList, i, (void *)qSeqLen.data());
            continue;
        } else if (i == KV_SEQLEN_INDEX) {
            CreateACLTensorInOut(tensorDim[i], inputTypes[i], ACL_FORMAT_ND, tensorList, i, (void *)kvSeqLen.data());
            continue;
        }
        CreateACLTensorInOut(tensorDim[i], inputTypes[i], ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);
    }
    uint64_t workspaceSize = 0;
    atb::Operation *op = nullptr;

    Status ret = AtbSelfAttentionPrefixEncoderGetWorkspaceSize(
        tensorList[0], tensorList[1], tensorList[2], tensorList[3], tensorList[4], tensorList[5], tensorList[6],
        tensorList[7], maskType, headNum, kvHeadNum, qkScale, tensorList[8], &workspaceSize, &op, context);
    EXPECT_EQ(ret, atb::NO_ERROR);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        EXPECT_EQ(ret, ACL_SUCCESS);
    }
    ret = AtbSelfAttentionPrefixEncoder(workspaceAddr, workspaceSize, op, context);
    EXPECT_EQ(ret, atb::NO_ERROR);

    ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(ret, ACL_SUCCESS);

    if (workspaceSize > 0) {
        EXPECT_EQ(aclrtFree(workspaceAddr), ACL_SUCCESS);
    }
    EXPECT_EQ(atb::DestroyOperation(op), atb::NO_ERROR);
    Destroy(&context, &stream);
    for (i = 0; i < INOUT_TENSOR_NUM; i++) {
        aclrtFreeHost(inoutHost[i]);
        aclrtFree(inoutDevice[i]);
    }
}

TEST(TestATBACL, TestSelfAttentionPrefixEncoderAlibiMask)
{
    const int64_t headNum = 16;
    const int64_t kvHeadNum = 8;
    const int64_t headSize = 128; // max 128
    const int64_t batch = 2;
    const int64_t numBlocks = 10;
    const int64_t blockSize = 128; // max 128
    const float qkScale = 0.0887;
    const int maskType = 4;
    const aclDataType dtype = ACL_FLOAT16;
    std::vector<int32_t> seqLen = {96, 123};
    std::vector<int32_t> KvSeqLen = {224, 123};
    TestSelfAttentionPrefixEncoder(headNum, kvHeadNum, headSize, batch, numBlocks, blockSize, qkScale, maskType, dtype,
                                   seqLen, KvSeqLen);
}

TEST(TestATBACL, TestSelfAttentionPrefixEncoderAlibiSqrtMask)
{
    const int64_t headNum = 16;
    const int64_t kvHeadNum = 8;
    const int64_t headSize = 96; // max 128
    const int64_t batch = 2;
    const int64_t numBlocks = 10;
    const int64_t blockSize = 128; // max 128
    const float qkScale = 0.0887;
    const int maskType = 5;
    const aclDataType dtype = ACL_FLOAT16;
    std::vector<int32_t> seqLen = {96, 123};
    std::vector<int32_t> KvSeqLen = {96, 251};
    TestSelfAttentionPrefixEncoder(headNum, kvHeadNum, headSize, batch, numBlocks, blockSize, qkScale, maskType, dtype,
                                   seqLen, KvSeqLen);
}

TEST(TestATBACL, TestSelfAttentionPrefixEncoderNormMask)
{
    const int64_t headNum = 10;
    const int64_t kvHeadNum = 5;
    const int64_t headSize = 64; // max 128
    const int64_t batch = 2;
    const int64_t numBlocks = 10;
    const int64_t blockSize = 128; // max 128
    const float qkScale = 0.0887;
    const int maskType = 3;
    const aclDataType dtype = ACL_BF16;
    std::vector<int32_t> seqLen = {64, 123};
    std::vector<int32_t> KvSeqLen = {192, 123};
    TestSelfAttentionPrefixEncoder(headNum, kvHeadNum, headSize, batch, numBlocks, blockSize, qkScale, maskType, dtype,
                                   seqLen, KvSeqLen);
}

TEST(TestATBACL, TestSelfAttentionPrefixEncoderCasualMask)
{
    const int64_t headNum = 32;
    const int64_t kvHeadNum = 8;
    const int64_t headSize = 32; // max 128
    const int64_t batch = 2;
    const int64_t numBlocks = 10;
    const int64_t blockSize = 128; // max 128
    const float qkScale = 0.0887;
    const int maskType = 9;
    const aclDataType dtype = ACL_BF16;
    std::vector<int32_t> seqLen = {128, 85};
    std::vector<int32_t> KvSeqLen = {128, 213};
    TestSelfAttentionPrefixEncoder(headNum, kvHeadNum, headSize, batch, numBlocks, blockSize, qkScale, maskType, dtype,
                                   seqLen, KvSeqLen);
}