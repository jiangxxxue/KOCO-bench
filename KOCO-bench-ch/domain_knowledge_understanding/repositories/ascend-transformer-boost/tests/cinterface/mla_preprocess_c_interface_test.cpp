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
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"
#include "atb/utils/log.h"

using namespace atb;
using namespace atb::cinterfaceTest;

const int64_t MLAINOUTMLAPP = 28;
const int64_t dims = 7168;
const int64_t dimB = 2112;
const int64_t dimC = 1536;
const int64_t dimD = 1;
const int64_t quantScale0 = 0;
const int64_t quantOffset0 = 0;
const int64_t wdqkv = 1 * 224 * 2112 * 32;
const int64_t deScale0 = 2112;
const int64_t bias0 = 2112;
const int64_t beta1 = 1536;
const int64_t gamma1 = 1536;
const int64_t quantScale1 = 1;
const int64_t quantOffset1 = 1;

const int64_t blockSize = 128;
const int64_t numTokens = 32;
const int64_t numHeads = 32;
const int64_t kvHeads = 1;
const int64_t kSeqlen = 256;
const int64_t batch = numTokens * kSeqlen;
const int64_t numBlocks = 64;

const int64_t sizeofFP16 = 2;
const int64_t wuq = 1 * 48 * numHeads * 192 * 32;
const int64_t deScale1 = numHeads * 192;
const int64_t bias1 = numHeads * 192;
const int64_t gamma2 = 512;
const int64_t cosNum = numTokens * 64;
const int64_t sinNum = numTokens * 64;
const int64_t wuk = numHeads * 128 * 512;

const int64_t kvCache = numBlocks * blockSize * 1 * 512;
const int64_t kvCacheC0 = numBlocks * blockSize * 1 * 576;
const int64_t kvCacheC2 = numBlocks * numHeads * 512 / 32 * blockSize * 32;
const int64_t kvCacheC3 = numBlocks * numHeads * 512 / 16 * blockSize * 16;

const int64_t kvCacheRope = numBlocks * blockSize * 1 * 64;
const int64_t kvCacheRopeC2 = numBlocks * numHeads * 64 / 16 * blockSize * 16;
const int64_t kvCacheRopeC3 = numBlocks * numHeads * 64 / 16 * blockSize * 16;
const int64_t slotmapping = numTokens;
const int64_t ctkvScale = 1;
const int64_t qNopeScale = numHeads;

const int64_t outTensor0C0 = numTokens * numHeads * 576;
const int64_t outTensor0C1 = numTokens * numHeads * 512;
const int64_t outTensor0C2 = numTokens * numHeads * 512;

const int64_t outTensor1C0 = numBlocks * blockSize * 576;
const int64_t outTensor1C1 = numBlocks * blockSize * 512;
const int64_t outTensor1C2 = numBlocks * numHeads * 512 / 32 * blockSize * 32;
const int64_t outTensor1C3 = numBlocks * numHeads * 512 / 16 * blockSize * 16;

const int64_t outTensor2 = numTokens * numHeads * 64;

const int64_t outTensor3C1 = numBlocks * blockSize * 1 * 64;
const int64_t outTensor3C2 = numBlocks * numHeads * 64 / 16 * blockSize * 16;


TEST(TestATBACL, TestMLAPreProcesscomb0Q0C0)
{
    atb::Context *context = nullptr;
    aclrtStream stream = nullptr;
    int64_t deviceId = 0;
    Init(&context, &stream, &deviceId);
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        ATB_LOG(ERROR) << "MLA PreProcess only supports A2/A3";
        Destroy(&context, &stream);
        GTEST_SKIP();
    }
    uint8_t *inoutHost[MLAINOUTMLAPP];
    uint8_t *inoutDevice[MLAINOUTMLAPP];
    aclTensor *tensorList[MLAINOUTMLAPP];
    size_t inoutSize[MLAINOUTMLAPP] = {
        numTokens * dims * sizeofFP16,
        dims * sizeofFP16,
        dims * sizeofFP16,
        quantScale0 * sizeofFP16,
        quantOffset0,
        wdqkv,
        deScale0 * sizeof(int64_t),
        bias0 * sizeof(int32_t),
        gamma1 * sizeofFP16,
        beta1 * sizeofFP16,
        quantScale1 * sizeofFP16,
        quantOffset1,
        wuq,
        deScale1 * sizeof(float),
        bias1 * sizeof(int32_t),
        gamma2 * sizeofFP16,
        cosNum * sizeofFP16,
        sinNum * sizeofFP16,
        wuk * sizeofFP16,
        kvCacheC0 * sizeofFP16,
        kvCacheRope * sizeofFP16,
        slotmapping * sizeof(int32_t),
        ctkvScale * sizeofFP16,
        qNopeScale * sizeofFP16,
        outTensor0C0 * sizeofFP16,
        outTensor1C0 * sizeofFP16,
        outTensor2 * sizeofFP16,
        outTensor1C3 * sizeofFP16,
    };
    CreateInOutData(MLAINOUTMLAPP, inoutHost, inoutDevice, inoutSize);
    size_t i = 0;
    aclDataType inputFormat = ACL_FLOAT16;

    // 0
    std::vector<int64_t> viewDim = {numTokens, dims};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 1
    viewDim = {dims};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 2
    viewDim = {dims};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 3
    viewDim = {dimD};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 4
    viewDim = {dimD};
    CreateACLTensorInOut(viewDim, ACL_INT8, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 5
    viewDim = {1, 224, dimB, 32};
    CreateACLTensorInOut(viewDim, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, tensorList, i, inoutDevice[i]);

    // 6
    viewDim = {dimB};
    CreateACLTensorInOut(viewDim, ACL_INT64, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 7
    viewDim = {dimB};
    CreateACLTensorInOut(viewDim, ACL_INT32, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 8
    viewDim = {dimC};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 9
    viewDim = {dimC};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 10
    viewDim = {dimD};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 11
    viewDim = {dimD};
    CreateACLTensorInOut(viewDim, ACL_INT8, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 12
    viewDim = {1, 48, numHeads * 192, 32};
    CreateACLTensorInOut(viewDim, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, tensorList, i, inoutDevice[i]);

    // 13
    viewDim = {numHeads * 192};
    CreateACLTensorInOut(viewDim, ACL_INT64, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 14
    viewDim = {numHeads * 192};
    CreateACLTensorInOut(viewDim, ACL_INT32, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 15
    viewDim = {gamma2};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 16
    viewDim = {numTokens, 64};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 17
    viewDim = {numTokens, 64};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 18
    viewDim = {numHeads, 128, 512};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 19
    viewDim = {numBlocks, blockSize, 1, 576};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 20
    viewDim = {numBlocks, blockSize, 1, 64};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 21
    viewDim = {numTokens};
    CreateACLTensorInOut(viewDim, ACL_INT32, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 22
    viewDim = {ctkvScale};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 23
    viewDim = {qNopeScale};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // out
    // 0
    viewDim = {numTokens, numHeads, 576};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 1
    viewDim = {numBlocks, blockSize, 1, 576};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 2
    viewDim = {numTokens, numHeads, 64};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);

    // 3
    viewDim = {numBlocks, blockSize, 1, 64};
    CreateACLTensorInOut(viewDim, inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);
    uint64_t workspaceSize = 0;
    atb::Operation *op = nullptr;

    atb::Status ret = AtbMLAPreprocessGetWorkspaceSize(
        tensorList[0], tensorList[1], tensorList[2], tensorList[3], tensorList[4], tensorList[5], tensorList[6],
        tensorList[7], tensorList[8], tensorList[9], tensorList[10], tensorList[11], tensorList[12], tensorList[13],
        tensorList[14], tensorList[15], tensorList[16], tensorList[17], tensorList[18], tensorList[19], tensorList[20],
        tensorList[21], tensorList[22], tensorList[23], 0, 0, 0, 1e-5, 2, 3, true, true, true, 0, 0, tensorList[24],
        tensorList[25], tensorList[26], tensorList[27], &workspaceSize, &op, context);

    EXPECT_EQ(ret, atb::NO_ERROR);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        EXPECT_EQ(ret, ACL_SUCCESS);
    }
    ret = AtbMLAPreprocess(workspaceAddr, workspaceSize, op, context);
    EXPECT_EQ(ret, atb::NO_ERROR);
    ret = aclrtSynchronizeStream(stream);

    if (workspaceSize > 0) {
        EXPECT_EQ(aclrtFree(workspaceAddr), ACL_SUCCESS);
    }
    EXPECT_EQ(atb::DestroyOperation(op), atb::NO_ERROR);
    Destroy(&context, &stream);
    for (i = 0; i < MLAINOUTMLAPP; i++) {
        aclrtFreeHost(inoutHost[i]);
        aclrtFree(inoutDevice[i]);
    }
}