/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include "atb/utils/config.h"
#include <atb/utils/log.h>
#include "atb/train_op_params.h"
#include "atb/operation.h"
#include "test_utils/operation_test.h"
#include "atb/utils/singleton.h"

using namespace atb;
using namespace Mki;

/// @brief 测试CloneOperationParam接口的基础功能
/// @param
/// @param
TEST(TestOpParamFuncs, CloneOperationParam)
{
    if (!GetSingleton<atb::Config>().Is910B()) {
        return;
    }
    train::FastSoftMaxParam param;
    param.headNum = 3;
    param.qSeqLen = { 10, 20, 30 };
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::FastSoftMaxParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(cloneParam, param);
    ATB_LOG(INFO) << "cloneParam headNum: " << cloneParam.headNum;
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

/// @brief 测试UpdateOperationParam接口的基础功能
/// @param
/// @param
TEST(TestOpParamFuncs, UpdateOperationParam)
{
    if (!GetSingleton<atb::Config>().Is910B()) {
        return;
    }
    train::FastSoftMaxParam param;
    param.headNum = 3;
    param.qSeqLen = { 10, 20, 30 };
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::FastSoftMaxParam getParam;
    st = atb::CloneOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParam, param);

    getParam.headNum = 66;
    st = atb::UpdateOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);

    train::FastSoftMaxParam getParamAfter;
    st = atb::CloneOperationParam(op, getParamAfter);
    ATB_LOG(INFO) << "after update headNum: " << getParamAfter.headNum;
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParamAfter, getParam);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

/// @brief 用非法的param属性值，调用UpdateOperationParam，返回错误
/// @param
/// @param
TEST(TestOpParamFuncs, UpdateOperationParamCheckParamFailed)
{
    if (!GetSingleton<atb::Config>().Is910B()) {
        return;
    }
    train::FastSoftMaxParam param;
    param.headNum = 3;
    param.qSeqLen = { 10, 20, 30 };
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.qSeqLen[0] = -1;
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

/// @brief 用非法的param属性值，调用CreateOperation，返回错误
/// @param
/// @param
TEST(TestOpParamFuncs, CreateOperationCheckParam)
{
    if (!GetSingleton<atb::Config>().Is910B()) {
        return;
    }
    train::FastSoftMaxParam param;
    param.headNum = 3;
    param.qSeqLen = { -1 };
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}