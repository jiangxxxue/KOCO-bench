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

const int64_t INOUT_TENSOR_NUM = 6;
const int ACTIVATION_SIGMOID = 8;
const int MAPPING_NUM_INDEX = 2;
const int MAPPING_TABLE_INDEX = 3;

void TestFusedAddTopK(const int64_t batchSize, const int64_t expertNum, const int maxRedundantExpertNum,
                      const uint32_t groupNum, const uint32_t groupTopk, const uint32_t n, const uint32_t k,
                      const int activationType, const bool isNorm, const float scale, const bool enableExpertMapping,
                      const aclDataType dtype)
{
    atb::Context *context = nullptr;
    aclrtStream stream = nullptr;
    int64_t deviceId = 0;
    Init(&context, &stream, &deviceId);
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        ATB_LOG(ERROR) << "FusedAddTopK only supports A2/A3";
        Destroy(&context, &stream);
        GTEST_SKIP();
    }
    uint8_t *inoutHost[INOUT_TENSOR_NUM];
    uint8_t *inoutDevice[INOUT_TENSOR_NUM];
    aclTensor *tensorList[INOUT_TENSOR_NUM];
    std::vector<aclDataType> inputTypes = {dtype, dtype, ACL_INT32, ACL_INT32, ACL_FLOAT, ACL_INT32};
    std::vector<std::vector<int64_t>> tensorDim = {
        {batchSize, expertNum},             // x
        {expertNum},                        // addNum
        {expertNum},                        // mappingNum
        {expertNum, maxRedundantExpertNum}, // mappingTable
        {batchSize, k},                     // y
        {batchSize, k},                     // indices
    };
    size_t inoutSize[INOUT_TENSOR_NUM];
    int total = 0;
    for (int i = 0; i < INOUT_TENSOR_NUM; ++i) {
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
    CreateInOutData(INOUT_TENSOR_NUM, inoutHost, inoutDevice, inoutSize);
    size_t i = 0;


    while (i < tensorDim.size()) {
        if (!enableExpertMapping && (i == MAPPING_NUM_INDEX || i == MAPPING_TABLE_INDEX)) {
            tensorList[i] = nullptr;
            ++i;
            continue;
        }
        CreateACLTensorInOut(tensorDim[i], inputTypes[i], ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);
    }
    uint64_t workspaceSize = 0;
    atb::Operation *op = nullptr;

    Status ret = AtbFusedAddTopkDivGetWorkspaceSize(
        tensorList[0], tensorList[1], tensorList[2], tensorList[3], groupNum, groupTopk, n, k, activationType, isNorm,
        scale, enableExpertMapping, tensorList[4], tensorList[5], &workspaceSize, &op, context);
    EXPECT_EQ(ret, atb::NO_ERROR);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        EXPECT_EQ(ret, ACL_SUCCESS);
    }
    ret = AtbFusedAddTopkDiv(workspaceAddr, workspaceSize, op, context);
    EXPECT_EQ(ret, atb::NO_ERROR);

    ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(ret, ACL_SUCCESS);

    if (workspaceSize > 0) {
        EXPECT_EQ(aclrtFree(workspaceAddr), ACL_SUCCESS);
    }
    EXPECT_EQ(atb::DestroyOperation(op), NO_ERROR);
    Destroy(&context, &stream);
    for (i = 0; i < INOUT_TENSOR_NUM; i++) {
        aclrtFreeHost(inoutHost[i]);
        aclrtFree(inoutDevice[i]);
    }
}

TEST(TestATBACL, TestFusedAddTopK1WithMapping)
{
    TestFusedAddTopK(10, 200, 32, 8, 2, 10, 5, ACTIVATION_SIGMOID, true, 0.0887, true, ACL_FLOAT16);
}

TEST(TestATBACL, TestFusedAddTopK1NoMapping)
{
    TestFusedAddTopK(6, 256, 8, 8, 2, 5, 5, ACTIVATION_SIGMOID, true, 0.0887, false, ACL_FLOAT16);
}
