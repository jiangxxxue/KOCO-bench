/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <securec.h>
#include <gtest/gtest.h>
#include <iostream>
#include <atb/utils/log.h>
#include <atb/utils/runner_variant_pack.h>
#include "atb/utils/tensor_util.h"
#include "atb/utils/config.h"

using namespace atb;
using namespace Mki;

TEST(TestSVector, SVectorHeap)
{
    atb::SVector<atb::Tensor> inTensors;
    inTensors.reserve(100);

    atb::Tensor firstTensor;
    firstTensor.desc.dtype = ACL_INT16;
    firstTensor.desc.shape.dimNum = 2;
    firstTensor.desc.shape.dims[0] = 1;
    firstTensor.desc.shape.dims[1] = 1;

    atb::Tensor tensorItem;
    tensorItem.desc.dtype = ACL_INT16;
    tensorItem.desc.shape.dimNum = 2;
    tensorItem.desc.shape.dims[0] = 2;
    tensorItem.desc.shape.dims[1] = 5;

    atb::Tensor endTensor;
    endTensor.desc.dtype = ACL_INT16;
    endTensor.desc.shape.dimNum = 2;
    endTensor.desc.shape.dims[0] = 10;
    endTensor.desc.shape.dims[1] = 10;

    inTensors.insert(0, firstTensor);
    EXPECT_EQ(inTensors.at(0).desc.shape.dims[0], 1);

    for (int i = 1; i < 50; i++) {
        inTensors.insert(i, tensorItem);
    }

    inTensors.push_back(endTensor);
    EXPECT_EQ(inTensors.at(50).desc.shape.dims[0], 10);
    // test SVector.insert and SVector.size()
    EXPECT_EQ(inTensors.size(), 51);

    EXPECT_EQ(inTensors.begin()->desc.shape.dims[0], 1);
    inTensors.clear();
    EXPECT_EQ(inTensors.size(), 0);
    EXPECT_EQ(inTensors.empty(), true);
}

TEST(TestSVector, HeapInitializerlist)
{
    atb::SVector<uint64_t> inTensors;
    inTensors.reserve(120);
    inTensors = {11, 1234, 12345, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1,  1,    1,     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1,  1,    1,     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1,  1,    1,     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100};
    EXPECT_EQ(inTensors.at(0), 11);

    inTensors.push_back(101);
    EXPECT_EQ(inTensors.at(100), 101);
    // test SVector.insert and SVector.size()
    EXPECT_EQ(inTensors.size(), 101);
    inTensors.clear();
    EXPECT_EQ(inTensors.size(), 0);
    EXPECT_EQ(inTensors.empty(), true);

    atb::SVector<bool> inTensorPerms;
    inTensorPerms.reserve(133);
    inTensorPerms.resize(133);
    for (size_t i = 0; i < inTensorPerms.size(); ++i) {
        inTensorPerms.at(i) = false;
    }
    EXPECT_EQ(inTensorPerms.at(132), false);
}

TEST(TestSVector, HeapInitializerlistTwo)
{
    atb::SVector<bool> inTensorPerms;
    inTensorPerms.reserve(133);
    inTensorPerms.resize(133);
    for (size_t i = 0; i < inTensorPerms.size(); ++i) {
        inTensorPerms.at(i) = false;
    }
    EXPECT_EQ(inTensorPerms.at(132), false);
}

TEST(TestSVector, StackInitializerlist)
{
    atb::SVector<int16_t> inTensors;
    inTensors = {11, 1234, 12345, 1, 7, 100};
    EXPECT_EQ(inTensors.at(0), 11);
    EXPECT_EQ(inTensors.at(1), 1234);
    EXPECT_EQ(inTensors.at(2), 12345);

    inTensors.push_back(101);
    EXPECT_EQ(inTensors.at(6), 101);
    // test SVector.insert and SVector.size()
    EXPECT_EQ(inTensors.size(), 7);
    inTensors.clear();
    EXPECT_EQ(inTensors.size(), 0);
    EXPECT_EQ(inTensors.empty(), true);
}

TEST(TestSVector, SVectorStack)
{
    atb::SVector<atb::Tensor> inTensors;
    atb::Tensor tensorItem;
    tensorItem.desc.dtype = ACL_INT16;
    tensorItem.desc.shape.dimNum = 2;
    tensorItem.desc.shape.dims[0] = 2;
    tensorItem.desc.shape.dims[1] = 5;

    atb::Tensor firstTensor;
    firstTensor.desc.dtype = ACL_INT16;
    firstTensor.desc.shape.dimNum = 2;
    firstTensor.desc.shape.dims[0] = 1;
    firstTensor.desc.shape.dims[1] = 1;

    atb::Tensor endTensor;
    endTensor.desc.dtype = ACL_INT16;
    endTensor.desc.shape.dimNum = 2;
    endTensor.desc.shape.dims[0] = 10;
    endTensor.desc.shape.dims[1] = 10;

    inTensors.insert(0, firstTensor);
    EXPECT_EQ(inTensors.at(0).desc.shape.dims[0], 1);

    for (int i = 1; i < 63; i++) {
        inTensors.insert(i, tensorItem);
    }

    inTensors.push_back(endTensor);
    EXPECT_EQ(inTensors.at(63).desc.shape.dims[0], 10);
    // test SVector.insert and SVector.size()
    EXPECT_EQ(inTensors.size(), 64);
    EXPECT_EQ(inTensors.begin()->desc.shape.dims[0], 1);
    EXPECT_EQ(inTensors.size(), 64);
    inTensors.clear();
    EXPECT_EQ(inTensors.size(), 0);
    EXPECT_EQ(inTensors.empty(), true);
}

TEST(TestSVector, HeapCopyConstructor)
{
    atb::SVector<uint64_t> inTensorSrc;
    inTensorSrc.reserve(100);
    inTensorSrc = {11, 1234, 12345, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1,  1,    1,     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1,  1,    1,     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1,  1,    1,     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100};
    atb::SVector<uint64_t> inTensorDest;
    inTensorDest.reserve(100);
    inTensorDest.resize(100);
    inTensorDest = inTensorSrc;
    EXPECT_EQ(inTensorDest.at(0), 11);
    EXPECT_EQ(inTensorDest.at(1), 1234);
    EXPECT_EQ(inTensorDest.at(2), 12345);

    // test SVector.insert and SVector.size()
    EXPECT_EQ(inTensorDest.size(), 100);
    inTensorDest.clear();
    EXPECT_EQ(inTensorDest.size(), 0);
    EXPECT_EQ(inTensorDest.empty(), true);
}

TEST(TestSVector, EqualityOperator)
{
    atb::SVector<int> vec1 = {1, 2, 3};
    atb::SVector<int> vec2 = {1, 2, 3};
    atb::SVector<int> vec3 = {1, 2, 4};

    EXPECT_EQ(vec1, vec2);
    EXPECT_NE(vec1, vec3);
}

TEST(TestSVector, LessThanOperator)
{
    atb::SVector<int> vec1 = {1, 2, 3};
    atb::SVector<int> vec2 = {1, 2, 4};
    atb::SVector<int> vec3 = {1, 2};

    EXPECT_LT(vec1, vec2);
    EXPECT_NE(vec3, vec1);
    EXPECT_FALSE(vec2 < vec1);
}

TEST(TestSVector, ReserveExceedsMaxSize)
{
    atb::SVector<int> vec;
    EXPECT_THROW(vec.reserve(atb::MAX_SVECTOR_SIZE + 1), atb::MaxSizeExceeded);
}

TEST(TestSVector, InsertExceedsDefaultSize)
{
    atb::SVector<int> vec;
    for (int i = 0; i < atb::DEFAULT_SVECTOR_SIZE; ++i) {
        vec.insert(i, i); // 先填满 stack 存储
    }
    for (int i = atb::DEFAULT_SVECTOR_SIZE; i < atb::DEFAULT_SVECTOR_SIZE + 5; ++i) {
        EXPECT_NO_THROW(vec.insert(i, i)); // 自动转换heap
    }
    EXPECT_EQ(vec.at(atb::DEFAULT_SVECTOR_SIZE), atb::DEFAULT_SVECTOR_SIZE);
    EXPECT_EQ(vec.size(), atb::DEFAULT_SVECTOR_SIZE + 5);
    for (int i = 0; i <= atb::DEFAULT_SVECTOR_SIZE; i += 7) {
        EXPECT_EQ(vec.at(i), i); // 原先stack值不改变
    }
}

TEST(TestSVector, PushBackExceedsDefaultSize)
{
    atb::SVector<int> vec;
    for (int i = 0; i < atb::DEFAULT_SVECTOR_SIZE; ++i) {
        vec.insert(i, i); // 先填满 stack 存储
    }
    for (int i = atb::DEFAULT_SVECTOR_SIZE; i < atb::DEFAULT_SVECTOR_SIZE + 5; ++i) {
        EXPECT_NO_THROW(vec.push_back(i)); // 自动转换heap
    }
    EXPECT_EQ(vec.at(atb::DEFAULT_SVECTOR_SIZE), atb::DEFAULT_SVECTOR_SIZE);
    EXPECT_EQ(vec.size(), atb::DEFAULT_SVECTOR_SIZE + 5);
    for (int i = 0; i < atb::DEFAULT_SVECTOR_SIZE + 5; ++i) {
        EXPECT_EQ(vec.at(i), i); // 原先stack值在heap中顺序不改变
    }
}

TEST(TestSVector, ClearAndReuse)
{
    atb::SVector<int> vec;
    vec.push_back(1);
    vec.push_back(2);
    EXPECT_EQ(vec.size(), 2);

    vec.clear();
    EXPECT_EQ(vec.size(), 0);
    EXPECT_TRUE(vec.empty());

    vec.push_back(3);
    EXPECT_EQ(vec.at(0), 3);
}

TEST(TestSVector, ResizeAndAccess)
{
    atb::SVector<int> vec = {1, 2, 3, 4, 5};
    vec.resize(3); // 缩小到 3 个元素
    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec.at(2), 3); // 最后一个有效元素

    EXPECT_THROW(vec.at(3), std::out_of_range); // 越界访问
}

TEST(TestSVector, IteratorTraversal)
{
    atb::SVector<int> vec = {10, 20, 30};
    int sum = 0;
    for (const auto &val : vec) {
        sum += val;
    }
    EXPECT_EQ(sum, 60);
}

TEST(TestSVector, OutOfRangeAccess)
{
    atb::SVector<int> vec = {1, 2, 3};
    EXPECT_THROW(vec.at(3), std::out_of_range); // 越界
    EXPECT_THROW(vec.at(100), std::out_of_range);

    EXPECT_THROW(vec[3], std::out_of_range); // 如果 operator[] 也检查边界
}

TEST(TestSVector, PushBackExceedsMaxSvectorSize)
{
    atb::SVector<int> vec;
    for (int i = 0; i < atb::MAX_SVECTOR_SIZE; ++i) {
        EXPECT_NO_THROW(vec.push_back(i)); // 自动转换heap
    }
    EXPECT_EQ(vec.at(atb::DEFAULT_SVECTOR_SIZE), atb::DEFAULT_SVECTOR_SIZE);
    EXPECT_THROW(vec.push_back(atb::DEFAULT_SVECTOR_SIZE), atb::MaxSizeExceeded);
}

TEST(TestSVector, InsertMiddle)
{
    // stack
    atb::SVector<int> vec;
    for (int i = 0; i < 10; ++i) {
        vec.insert(i, i);
    }
    for (int j = 0; j <= 15; j += 3) {
        EXPECT_NO_THROW(vec.insert(j, j * j));
    }
    EXPECT_EQ(vec.size(), 16);
    for (int i = -1, j = -1, idx = 0; idx <= 15; ++idx) {
        ++j;
        if (j % 3 == 0) {
            EXPECT_EQ(j * j, vec.at(j));
            continue;
        }
        ++i;
        EXPECT_EQ(i, vec[idx]);
    }

    // heap
    atb::SVector<int> vecHeap;
    vecHeap.reserve(100);
    for (int i = 0; i < 10; ++i) {
        vecHeap.insert(i, i);
    }
    for (int j = 0; j <= 15; j += 3) {
        EXPECT_NO_THROW(vecHeap.insert(j, j * j));
    }
    EXPECT_EQ(vecHeap.size(), 16);
    for (int i = -1, j = -1, idx = 0; idx <= 15; ++idx) {
        ++j;
        if (j % 3 == 0) {
            EXPECT_EQ(j * j, vecHeap.at(j));
            continue;
        }
        ++i;
        EXPECT_EQ(i, vecHeap[idx]);
    }
}

TEST(TestSVector, ResizeSmall)
{
    // stack
    atb::SVector<int> vec;
    for (int i = 0; i < 40; ++i) {
        vec.push_back(i);
    }
    vec.resize(15);
    for (int i = 0; i < 10; ++i) {
        vec.push_back(i + i);
    }
    for (int i = 0; i < 15; ++i) {
        EXPECT_EQ(i, vec.at(i));
    }
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(i + i, vec[15 + i]);
    }
    vec.clear();
    vec.reserve(128);
    for (int i = 0; i < 40; ++i) {
        vec.push_back(i);
    }
    vec.resize(22);
    for (int i = 22; i < 56; ++i) {
        vec.insert(i, i * i);
    }
    for (int i = 0; i < 22; ++i) {
        EXPECT_EQ(i, vec.at(i));
    }
    for (int i = 22; i < 56; ++i) {
        EXPECT_EQ(i * i, vec[i]);
    }
}

TEST(TestSVector, ThrowOperationAtAndBracket)
{
    atb::SVector<int> vec;
    vec.resize(80);
    EXPECT_THROW(vec.at(100), std::out_of_range);
    EXPECT_THROW(vec[100], std::out_of_range);
}
