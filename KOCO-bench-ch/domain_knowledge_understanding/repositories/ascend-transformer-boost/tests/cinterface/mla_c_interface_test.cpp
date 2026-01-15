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

const int64_t MLAINOUTMLA = 12;
const int64_t headSizeQk = 576;
const int64_t headSizeVo = 512;
const int64_t maxNumBlocksPerQuery = 16;
const int64_t dimD = 1;
const int64_t dimE = 125;
const int64_t dimF = 512;
const int64_t dimG = 2;
const int64_t sizeofFP16 = 2;
const int64_t numHeads = 32;
const int64_t kvHeads = 1;
const int64_t blockSize = 128;
const int64_t kSeqlen = 256;
const int64_t numTokens = 32;
const int64_t batch = numTokens * kSeqlen;
const int64_t numBlocks = 64;

TEST(TestATBACL, TestMLAM0C2C1)
{
    atb::Context *context = nullptr;
    aclrtStream stream = nullptr;
    int64_t deviceId = 0;
    Init(&context, &stream, &deviceId);
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        ATB_LOG(ERROR) << "MLA only supports A2/A3";
        Destroy(&context, &stream);
        GTEST_SKIP();
    }
    uint8_t *inoutHost[MLAINOUTMLA];
    uint8_t *inoutDevice[MLAINOUTMLA];
    aclTensor *tensorList[MLAINOUTMLA];
    size_t inoutSize[MLAINOUTMLA] = {
        numTokens * numHeads * headSizeVo * sizeofFP16,
        numTokens * numHeads * (headSizeQk - headSizeVo) * sizeofFP16,
        numBlocks * blockSize * kvHeads * headSizeVo * sizeofFP16,
        numBlocks * blockSize * kvHeads * (headSizeQk - headSizeVo) * sizeofFP16,
        batch * maxNumBlocksPerQuery * sizeof(int32_t),
        batch * sizeof(int),
        (dimE + dimG * batch) * blockSize * sizeofFP16,
        batch * sizeof(int),
        numHeads * sizeof(float),
        numHeads * sizeof(float),
        numTokens * numHeads * dimF * sizeofFP16,
        numTokens * numHeads * dimD * sizeofFP16,
    };
    CreateInOutData(MLAINOUTMLA, inoutHost, inoutDevice, inoutSize);
    size_t i = 0;
    aclDataType inputFormat = ACL_FLOAT16;
    // 0
    std::vector<std::vector<int64_t>> viewDim = {
        {numTokens, numHeads, headSizeVo},              // 0
        {numTokens, numHeads, headSizeQk - headSizeVo}, // 1
        {numBlocks, blockSize, kvHeads, 512},           // 2
        {numBlocks, blockSize, kvHeads, 64},            // 3
        {batch, maxNumBlocksPerQuery},                  // 4
        {batch},                                        // 5
        {(dimE + dimG * batch), blockSize},             // 6
        {batch},                                        // 7
        {numHeads},                                     // 8
        {numHeads},                                     // 9
        {numTokens, numHeads, 512},                     // 10
        {numTokens, numHeads, 1},                       // 11
    };
    while (i < viewDim.size()) {
        if (i == 4) {
            CreateACLTensorInOut(viewDim[i], ACL_INT32, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);
        } else if (i == 5 || i == 7) {
            CreateACLTensorInOut(viewDim[i], ACL_INT32, ACL_FORMAT_ND, tensorList, i, inoutHost[i]);
        } else if (8 == i || 9 == i) {
            CreateACLTensorInOut(viewDim[i], ACL_FLOAT, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);
        } else {
            CreateACLTensorInOut(viewDim[i], inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);
        }
    }
    uint64_t workspaceSize = 0;
    atb::Operation *op = nullptr;

    atb::Status ret =
        AtbMLAGetWorkspaceSize(tensorList[0], tensorList[1], tensorList[2], tensorList[3], tensorList[4], tensorList[5],
                               tensorList[6], tensorList[7], tensorList[8], tensorList[9], 32, 1.0, 1, 0, 2, 1,
                               tensorList[10], tensorList[11], &workspaceSize, &op, context);
    EXPECT_EQ(ret, atb::NO_ERROR);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        EXPECT_EQ(ret, ACL_SUCCESS);
    }
    ret = AtbMLA(workspaceAddr, workspaceSize, op, context);
    EXPECT_EQ(ret, atb::NO_ERROR);

    ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(ret, ACL_SUCCESS);

    if (workspaceSize > 0) {
        EXPECT_EQ(aclrtFree(workspaceAddr), ACL_SUCCESS);
    }
    EXPECT_EQ(atb::DestroyOperation(op), NO_ERROR);
    Destroy(&context, &stream);
    for (i = 0; i < MLAINOUTMLA; i++) {
        aclrtFreeHost(inoutHost[i]);
        aclrtFree(inoutDevice[i]);
    }
}