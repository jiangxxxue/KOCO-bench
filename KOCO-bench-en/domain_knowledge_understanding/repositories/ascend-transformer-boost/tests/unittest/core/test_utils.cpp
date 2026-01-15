/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <cstring>
#include <gtest/gtest.h>
#include "atb/types.h"
#include "atb/utils.h"

using namespace atb;

TEST(UtilsTest, GetTensorSize)
{
    TensorDesc tensorDesc = { ACL_FLOAT16, ACL_FORMAT_ND, { { 1, 2, 3 }, 3 } };
    EXPECT_EQ(Utils::GetTensorSize(tensorDesc), 12);
}

TEST(UtilsTest, GetTensorSizeInvalidDim)
{
    TensorDesc tensorDesc = { ACL_FLOAT16, ACL_FORMAT_ND, { { 1, -1, 3 }, 3 } };
    EXPECT_EQ(Utils::GetTensorSize(tensorDesc), 0);
    tensorDesc = { ACL_FLOAT16, ACL_FORMAT_ND, { { 1, 1, 0 }, 3 } };
    EXPECT_EQ(Utils::GetTensorSize(tensorDesc), 0);
}

TEST(UtilsTest, GetTensorSizeOverFlow)
{
    TensorDesc tensorDesc = { ACL_FLOAT16, ACL_FORMAT_ND, { { 10000, 3000000, 50000000, 50000 }, 4 } };
    EXPECT_EQ(Utils::GetTensorSize(tensorDesc), 0);
}

TEST(UtilsTest, QuantParamConvert)
{
    Utils::QuantParamConvert(nullptr, nullptr, 0);

    float src[5] = {1.0, 2.3, 3.0, 4.0, 5.0};
    uint64_t dest[5] = {};
    Utils::QuantParamConvert(src, dest, 5);
    float result;
    for (uint64_t i = 0; i < 5; i++) {
        uint32_t lower32 = static_cast<uint32_t>(dest[i]);
        std::memcpy(&result, &lower32, sizeof(result));
        EXPECT_EQ(result, src[i]);
    }
}