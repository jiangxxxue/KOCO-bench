/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "c_interface_utils.h"

#include <algorithm>
#include <iostream>
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"
#include "atb/utils/log.h"

using namespace atb;
using namespace atb::cinterfaceTest;

// querySptli1, querySplit2, keySplit1, keySplit2, value, mask, seqLen, (preOut), (preLse), [o]output, [o]softmaxLse
const int64_t INOUT_TENSOR_NUM = 11;
const int64_t NOPE_HEADSIZE = 128;
const int64_t ROPE_HEADSIZE = 64;
const int64_t SEQLEN_INDEX = 6;
const int64_t PREF_OUT_INDEX = 7;

void TestRingMLA(const int64_t headNum, const int64_t kvHeadNum, const int64_t headSizeV, const int64_t batch,
                 const float qkscale, const int kernelType, const int maskType, const int calcType,
                 const aclDataType dtype, std::vector<int32_t> seqLen)
{
    int inputNum = INOUT_TENSOR_NUM;
    atb::Context *context = nullptr;
    aclrtStream stream = nullptr;
    int64_t deviceId = 0;
    Init(&context, &stream, &deviceId);
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        ATB_LOG(ERROR) << "RingMLA only supports A2/A3";
        Destroy(&context, &stream);
        GTEST_SKIP();
    }
    uint8_t *inoutHost[inputNum];
    uint8_t *inoutDevice[inputNum];
    aclTensor *tensorList[inputNum];
    std::vector<aclDataType> inputTypes = {dtype,     dtype, dtype,     dtype, dtype,    dtype,
                                           ACL_INT32, dtype, ACL_FLOAT, dtype, ACL_FLOAT};
    int32_t qNTokens = 0;
    int32_t kvNTokens = 0;
    bool hasKvSeqlen = 2 * batch == seqLen.size();
    if (!hasKvSeqlen && batch != seqLen.size()) {
        std::cout << "wrong seqlen size, expect [batch]/[2 * batch], batch = " << batch
                  << ", but got: " << seqLen.size() << std::endl;
        GTEST_SKIP();
    }
    for (int i = 0; i < batch; ++i) {
        qNTokens += seqLen[i];
        if (hasKvSeqlen) {
            kvNTokens += seqLen[batch + i];
        }
    }
    if (kvNTokens == 0) {
        kvNTokens = qNTokens;
    }
    std::vector<std::vector<int64_t>> tensorDim = {
        {qNTokens, headNum, NOPE_HEADSIZE},    // querySplit1
        {qNTokens, headNum, ROPE_HEADSIZE},    // querySplit2
        {kvNTokens, kvHeadNum, NOPE_HEADSIZE}, // keySplit1
        {kvNTokens, kvHeadNum, ROPE_HEADSIZE}, // keySplit2
        {kvNTokens, kvHeadNum, headSizeV},     // value
        {512, 512},                            // mask
        {batch},                               // seqLen
        {qNTokens, headNum, headSizeV},        // prevOut
        {headNum, qNTokens},                   // prevLse
        {qNTokens, headNum, headSizeV},        // output
        {headNum, qNTokens},                   // softmaxLse
    };
    if (hasKvSeqlen) {
        tensorDim[SEQLEN_INDEX] = {2l, batch};
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
            CreateACLTensorInOut(tensorDim[i], inputTypes[i], ACL_FORMAT_ND, tensorList, i, (void *)seqLen.data());
            continue;
        }
        CreateACLTensorInOut(tensorDim[i], inputTypes[i], ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);
    }
    uint64_t workspaceSize = 0;
    atb::Operation *op = nullptr;

    if (calcType == 1) {                                // CALC_TYPE_FISRT_RING case: Don't take in prevOut, prevLse
        aclrtFreeHost(inoutDevice[PREF_OUT_INDEX]);     // preOut
        aclrtFreeHost(inoutDevice[PREF_OUT_INDEX + 1]); // preLse
        inoutHost[PREF_OUT_INDEX] = nullptr;
        inoutHost[PREF_OUT_INDEX + 1] = nullptr;
    }
    Status ret = AtbRingMLAGetWorkspaceSize(tensorList[0], tensorList[1], tensorList[2], tensorList[3], tensorList[4],
                                            tensorList[5], tensorList[6], tensorList[7], tensorList[8], headNum,
                                            kvHeadNum, qkscale, kernelType, maskType, 0, calcType, tensorList[9],
                                            tensorList[10], &workspaceSize, &op, context);
    EXPECT_EQ(ret, atb::NO_ERROR);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        EXPECT_EQ(ret, ACL_SUCCESS);
    }
    ret = AtbRingMLA(workspaceAddr, workspaceSize, op, context);
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

TEST(TestATBACL, TestRingMLADefault)
{
    std::vector<int32_t> seqLen = {96, 123};
    TestRingMLA(16, 8, 128, 2, 0.0887, 1, 1, 0, ACL_FLOAT16, seqLen);
}

TEST(TestATBACL, TestRingMLAFirstRing)
{
    std::vector<int32_t> seqLen = {100, 100, 200, 200};
    TestRingMLA(16, 8, 128, 2, 0.0887, 1, 0, 1, ACL_BF16, seqLen);
}

TEST(TestATBACL, TestRingMLADefault2)
{
    const int64_t headNum = 64;
    const int64_t kvHeadNum = 64;
    const int64_t headSizeV = 128;
    const int64_t batch = 2;
    const float qkscale = 0.0721688;
    const int kernelType = 1;
    const int maskType = 1;
    const int calcType = 0;
    const aclDataType dtype = ACL_BF16;
    std::vector<int32_t> seqLen = {3, 4, 4, 0};
    TestRingMLA(headNum, kvHeadNum, headSizeV, batch, qkscale, kernelType, maskType, calcType, dtype, seqLen);
}
