/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <vector>
#include <string>
#include <string.h>
#include <cstdlib>
#include <gtest/gtest.h>
#include "atb/types.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/utils_internal.h"
#include <atb/utils/log.h>
#include "mki/utils/file_system/file_system.h"

using namespace atb;
using namespace Mki;

TEST(Testequal, TensorDescNOTEqualTest)
{
    Mki::TensorDesc TensorDescA;
    TensorDescA.dtype = TENSOR_DTYPE_INT32;
    TensorDescA.format = TENSOR_FORMAT_ND;
    Mki::SVector<int64_t> dimsA = { 3, 5 };
    TensorDescA.dims = dimsA;
    Mki::TensorDesc TensorDescB;
    TensorDescB.dtype = TENSOR_DTYPE_INT32;
    TensorDescB.format = TENSOR_FORMAT_ND;
    Mki::SVector<int64_t> dimsB = { 4, 6 };
    TensorDescB.dims = dimsB;
    EXPECT_EQ(TensorUtil::AsdOpsTensorDescEqual(TensorDescA, TensorDescB), false);
}

TEST(Testequal, TensorDescEqualTest)
{
    Mki::TensorDesc TensorDescA;
    TensorDescA.dtype = TENSOR_DTYPE_INT32;
    TensorDescA.format = TENSOR_FORMAT_ND;
    Mki::SVector<int64_t> dimsA = { 2, 5 };
    TensorDescA.dims = dimsA;
    Mki::TensorDesc TensorDescB;
    TensorDescB.dtype = TENSOR_DTYPE_INT32;
    TensorDescB.format = TENSOR_FORMAT_ND;
    Mki::SVector<int64_t> dimsB = { 2, 5 };
    TensorDescB.dims = dimsB;
    EXPECT_EQ(TensorUtil::AsdOpsTensorDescEqual(TensorDescA, TensorDescB), true);
}

TEST(Testequal, AtbTensorDescEqualTest)
{
    atb::TensorDesc TensorDescA;
    atb::TensorDesc TensorDescB;
    TensorDescA.dtype = ACL_INT32;
    TensorDescA.format = ACL_FORMAT_ND;
    TensorDescA.shape.dimNum = 2;
    TensorDescA.shape.dims[0] = 2;
    TensorDescA.shape.dims[1] = 5;
    TensorDescB.dtype = ACL_INT32;
    TensorDescB.format = ACL_FORMAT_ND;
    TensorDescB.shape.dimNum = 2;
    TensorDescB.shape.dims[0] = 2;
    TensorDescB.shape.dims[1] = 5;
    TensorUtil::ShapeToString(TensorDescA.shape);
    TensorUtil::TensorDescToString(TensorDescA);
    EXPECT_EQ(TensorUtil::TensorDescEqual(TensorDescA, TensorDescB), true);
}

TEST(Testequal, AtbTensorDescNotEqualTest)
{
    atb::TensorDesc TensorDescA;
    atb::TensorDesc TensorDescB;
    TensorDescA.dtype = ACL_INT32;
    TensorDescA.format = ACL_FORMAT_ND;
    TensorDescA.shape.dimNum = 2;
    TensorDescA.shape.dims[0] = 8;
    TensorDescA.shape.dims[1] = 8;
    TensorDescB.dtype = ACL_INT32;
    TensorDescB.format = ACL_FORMAT_ND;
    TensorDescB.shape.dimNum = 2;
    TensorDescB.shape.dims[0] = 2;
    TensorDescB.shape.dims[1] = 5;
    EXPECT_EQ(TensorUtil::TensorDescEqual(TensorDescA, TensorDescB), false);
}

TEST(CalTest, CalcAtbTensorDataSizeTest1)
{
    atb::Tensor tensor;
    tensor.desc.dtype = ACL_INT16;
    tensor.desc.shape.dimNum = 2;
    tensor.desc.shape.dims[0] = 2;
    tensor.desc.shape.dims[1] = 5;
    TensorUtil::TensorToString(tensor);
    EXPECT_EQ(TensorUtil::CalcTensorDataSize(tensor), 20);
}

TEST(CalTest, CalcAtbTensorDataSizeTest2)
{
    atb::Tensor tensor;
    tensor.desc.dtype = ACL_UINT16;
    tensor.desc.shape.dimNum = 2;
    tensor.desc.shape.dims[0] = 2;
    tensor.desc.shape.dims[1] = 5;
    EXPECT_EQ(TensorUtil::CalcTensorDataSize(tensor), 20);
}

TEST(CalTest, CalcAtbTensorDataSizeTest3)
{
    atb::Tensor tensor;
    tensor.desc.dtype = ACL_UINT64;
    tensor.desc.shape.dimNum = 2;
    tensor.desc.shape.dims[0] = 2;
    tensor.desc.shape.dims[1] = 5;
    EXPECT_EQ(TensorUtil::CalcTensorDataSize(tensor), 80);
}

TEST(CalTest, CalcTensorDataSizeTest1)
{
    Mki::Tensor tensor;
    tensor.desc.dtype = TENSOR_DTYPE_FLOAT16;
    Mki::SVector<int64_t> dims = { 3, 4, 7 };
    tensor.desc.dims = dims;
    EXPECT_EQ(TensorUtil::CalcTensorDataSize(tensor), 168);
}

TEST(CalTest, CalcTensorDataSizeTest2)
{
    Mki::Tensor tensor;
    tensor.desc.dtype = TENSOR_DTYPE_DOUBLE;
    Mki::SVector<int64_t> dims = { 3, 4, 7 };
    tensor.desc.dims = dims;
    EXPECT_EQ(TensorUtil::CalcTensorDataSize(tensor), 672);
}

TEST(CalTest, CalcTensorDataSizeTest3)
{
    Mki::Tensor tensor;
    tensor.desc.dtype = TENSOR_DTYPE_FLOAT16;
    Mki::SVector<int64_t> dims = {};
    tensor.desc.dims = dims;
    EXPECT_EQ(TensorUtil::CalcTensorDataSize(tensor), 0);
}

TEST(TestTensorUtil, AlignIntDivZero)
{
    EXPECT_EQ(TensorUtil::AlignInt(12, 0), -1);
}

TEST(CalTest, CalcTensorDataSizeTest4)
{
    Mki::Tensor tensor;
    tensor.desc.dtype = TENSOR_DTYPE_INT16;
    Mki::SVector<int64_t> dims = { 2, 5 };
    tensor.desc.dims = dims;
    TensorUtil::AsdOpsTensorToString(tensor);
    EXPECT_EQ(TensorUtil::CalcTensorDataSize(tensor), 20);
}

TEST(CalTest, CalcTensorDataSizeTest5)
{
    Mki::Tensor tensor;
    tensor.desc.dtype = TENSOR_DTYPE_INT32;
    Mki::SVector<int64_t> dims = { 2, 5 };
    tensor.desc.dims = dims;
    EXPECT_EQ(TensorUtil::CalcTensorDataSize(tensor), 40);
}


TEST(CalTest, CalcTensorDataSizeTest6)
{
    Mki::Tensor tensor;
    tensor.desc.dtype = TENSOR_DTYPE_INT64;
    Mki::SVector<int64_t> dims = { 2, 5 };
    tensor.desc.dims = dims;
    EXPECT_EQ(TensorUtil::CalcTensorDataSize(tensor), 80);
}

TEST(CalTest, CalcTensorDataSizeTest7)
{
    Mki::Tensor tensor;
    tensor.desc.dtype = TENSOR_DTYPE_UINT16;
    Mki::SVector<int64_t> dims = { 2, 5 };
    tensor.desc.dims = dims;
    EXPECT_EQ(TensorUtil::CalcTensorDataSize(tensor), 20);
}

TEST(CalTest, CalcTensorDataSizeTest8)
{
    Mki::Tensor tensor;
    tensor.desc.dtype = TENSOR_DTYPE_UINT32;
    Mki::SVector<int64_t> dims = { 2, 5 };
    tensor.desc.dims = dims;
    EXPECT_EQ(TensorUtil::CalcTensorDataSize(tensor), 40);
}

TEST(CalTest, CalcTensorDataSizeTest9)
{
    Mki::Tensor tensor;
    tensor.desc.dtype = TENSOR_DTYPE_UINT64;
    Mki::SVector<int64_t> dims = { 2, 5 };
    tensor.desc.dims = dims;
    EXPECT_EQ(TensorUtil::CalcTensorDataSize(tensor), 80);
}

TEST(CalTest, CalcTensorDataSizeOverFlow)
{
    Mki::Tensor tensor;
    tensor.desc.dtype = TENSOR_DTYPE_UINT64;
    Mki::SVector<int64_t> dims = { 10000, 3000000, 50000000, 50000 };
    tensor.desc.dims = dims;
    EXPECT_EQ(TensorUtil::CalcTensorDataSize(tensor), 0);
}

TEST(Testequal, OtherTest)
{
    Mki::TensorDesc TensorDescA;
    TensorDescA.dtype = TENSOR_DTYPE_INT32;
    TensorDescA.format = TENSOR_FORMAT_ND;
    Mki::SVector<int64_t> dimsA = { 2, 5 };
    TensorDescA.dims = dimsA;
    TensorUtil::AsdOpsDimsToString(dimsA);
    TensorUtil::AsdOpsTensorDescToString(TensorDescA);
    const double num1 = 3.0;
    const double num2 = 3.0;
    EXPECT_EQ(UtilsInternal::IsDoubleEqual(num1, num2), true);
}