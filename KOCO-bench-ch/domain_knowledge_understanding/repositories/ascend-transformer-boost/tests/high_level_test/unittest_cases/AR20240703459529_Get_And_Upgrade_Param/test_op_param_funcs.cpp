/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendTransformerBoost is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 * http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include <gtest/gtest.h>
#include "atb/utils/log.h"
#include "atb/train_op_params.h"
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "test_utils/operation_test.h"

using namespace atb;
using namespace Mki;

TEST(TestOpParamFuncs, CloneOperationParam)
{
    train::FastSoftMaxParam param;
    param.qSeqLen = { 10, 20, 30 };
    param.headNum = 3;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::FastSoftMaxParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(cloneParam, param);

    for (auto len : cloneParam.qSeqLen) {
        ATB_LOG(INFO) << "cloneParam qSeqLen: " << len;
    }
    ATB_LOG(INFO) << "cloneParam headNum: " << cloneParam.headNum;
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParam)
{
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

TEST(TestOpParamFuncs, UpdateOperationParamCheckParamFailed)
{
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

TEST(TestOpParamFuncs, CreateOperationCheckParam)
{
    train::FastSoftMaxParam param;
    param.headNum = 3;
    param.qSeqLen = { -1 };
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam1)
{
    train::FastSoftMaxParam param;
    param.headNum = 3;
    param.qSeqLen = {};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam2)
{
    train::FastSoftMaxParam param;
    param.headNum = 3;
    param.qSeqLen = {0};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam3)
{
    train::FastSoftMaxParam param;
    param.headNum = 8;
    param.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam4)
{
    train::FastSoftMaxParam param;
    param.headNum = 8;
    param.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam5)
{
    train::FastSoftMaxParam param;
    param.headNum = 8;
    param.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::FastSoftMaxGradParam param1;
    param1.headNum = 8;
    param1.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    atb::Operation *op1 = nullptr;

    atb::Status st1 = atb::CreateOperation(param1, &op1);
    EXPECT_EQ(st1, atb::NO_ERROR);
    EXPECT_NE(op1, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam7)
{
    train::RmsNormBackwardParam param;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam8)
{
    train::FastSoftMaxParam param;
    param.headNum = 3;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam9)
{
    train::StridedBatchMatmulParam param;
    param.transposeA = false;
    param.transposeB = false;
    param.batch = 1;
    param.headNum = 1;
    param.m = {16};
    param.n = {16};
    param.k = {16};
    param.lda = {16};
    param.ldb = {16};
    param.ldc = {16};
    param.strideA = {16};
    param.strideB = {16};
    param.strideC = {256};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam10)
{
    infer::AsStridedParam param;
    param.size = {3,3};
    param.stride = {1,0};
    param.offset = {0};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam11)
{
    train::FastSoftMaxGradParam param;
    param.headNum = 8;
    param.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam33)
{
    train::FastSoftMaxGradParam param;
    param.headNum = 8;
    param.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam12)
{
    train::FastSoftMaxGradParam param;
    param.headNum = 8;
    param.qSeqLen = {};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam13)
{
    train::FastSoftMaxGradParam param;
    param.headNum = 8;
    param.qSeqLen = {-1};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam14)
{
    train::FastSoftMaxGradParam param;
    param.headNum = 8;
    param.qSeqLen = {0};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam15)
{
    train::GenAttentionMaskParam param;
    param.headNum = 8;
    param.seqLen = {0};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam16)
{
    train::GenAttentionMaskParam param;
    param.headNum = 8;
    param.seqLen = {-1};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam17)
{
    train::GenAttentionMaskParam param;
    param.headNum = 8;
    param.seqLen = {};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam18)
{
    train::PadWithHiddenStateParam param;
    param.qSeqLen = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam19)
{
    train::PadWithHiddenStateParam param;
    param.qSeqLen = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam20)
{
    train::PadWithHiddenStateParam param;
    param.qSeqLen = {};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam21)
{
    train::PadWithHiddenStateParam param;
    param.qSeqLen = {-1};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam22)
{
    train::PadWithHiddenStateParam param;
    param.qSeqLen = {0};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam23)
{
    train::RopeGradParam param;
    param.qSeqLen = {1,1};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam24)
{
    train::RopeGradParam param;
    param.qSeqLen = {};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam25)
{
    train::RopeGradParam param;
    param.qSeqLen = {-1};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam26)
{
    train::RopeGradParam param;
    param.qSeqLen = {0};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam27)
{
    train::UnpadWithHiddenStateParam param;
    param.qSeqLen = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam28)
{
    train::UnpadWithHiddenStateParam param;
    param.qSeqLen = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8};
    param.maxSeqLen = 4097;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam29)
{
    train::UnpadWithHiddenStateParam param;
    param.qSeqLen = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam30)
{
    train::UnpadWithHiddenStateParam param;
    param.qSeqLen = {};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam31)
{
    train::UnpadWithHiddenStateParam param;
    param.qSeqLen = {-1};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CreateOperationCheckParam32)
{
    train::UnpadWithHiddenStateParam param;
    param.qSeqLen = {0};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);
}

TEST(TestOpParamFuncs, CloneOperationParam1) {
    train::FastSoftMaxParam param;
    param.headNum = 3;
    param.qSeqLen = { 10, 20, 30 };
    atb::Operation *op = nullptr;
    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::FastSoftMaxGradParam param1;
    param1.headNum = 8;
    param1.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    atb::Operation *op1 = nullptr;
    atb::Status st1 = atb::CreateOperation(param1, &op1);
    EXPECT_EQ(st1, atb::NO_ERROR);
    EXPECT_NE(op1, nullptr);

    train::FastSoftMaxParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(cloneParam, param);
    ATB_LOG(INFO) << "cloneParam headNum: " << cloneParam.headNum;
    for (auto len : cloneParam.qSeqLen) {
        ATB_LOG(INFO) << "cloneParam qSeqLen: " << len;
    }
    st = atb::DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);

    train::FastSoftMaxGradParam cloneParam1;
    st = atb::CloneOperationParam(op1, cloneParam1);
    EXPECT_EQ(st1, atb::NO_ERROR);
    EXPECT_EQ(cloneParam1, param1);
    ATB_LOG(INFO) << "cloneParam1 headNum: " << cloneParam1.headNum;
    for (auto len : cloneParam1.qSeqLen) {
        ATB_LOG(INFO) << "cloneParam1 qSeqLen: " << len;
    }
    EXPECT_EQ(st1, atb::NO_ERROR);
    st1 = atb::DestroyOperation(op1);
    ASSERT_EQ(st1, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, CloneOperationParam3)
{
    train::FastSoftMaxParam param;
    param.headNum = 3;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);

    train::FastSoftMaxParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestOpParamFuncs, CloneOperationParam4)
{
    train::FastSoftMaxGradParam param;
    param.headNum = 8;
    param.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::FastSoftMaxGradParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(cloneParam, param);
    ATB_LOG(INFO) << "cloneParam headNum: " << cloneParam.headNum;
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, CloneOperationParam5)
{
    train::FastSoftMaxGradParam param;
    param.headNum = 8;
    param.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);

    train::FastSoftMaxGradParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestOpParamFuncs, CloneOperationParam50)
{
    train::FastSoftMaxGradParam param;
    param.headNum = 8;
    param.qSeqLen = {};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);

    train::FastSoftMaxGradParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestOpParamFuncs, CloneOperationParam6)
{
    train::GenAttentionMaskParam param;
    param.headNum = 2;
    param.seqLen = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::GenAttentionMaskParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(cloneParam, param);

    for (auto len : cloneParam.seqLen) {
        ATB_LOG(INFO) << "cloneParam seqLen: " << len;
    }
    ATB_LOG(INFO) << "cloneParam headNum: " << cloneParam.headNum;
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, CloneOperationParam7)
{
    train::GenAttentionMaskParam param;
    param.headNum = 2;
    param.seqLen = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);

    train::GenAttentionMaskParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestOpParamFuncs, CloneOperationParam70)
{
    train::GenAttentionMaskParam param;
    param.headNum = 2;
    param.seqLen = {};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);

    train::GenAttentionMaskParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestOpParamFuncs, CloneOperationParam71)
{
    train::GenAttentionMaskParam param;
    param.headNum = 2;
    param.seqLen = {0};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);

    train::GenAttentionMaskParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestOpParamFuncs, CloneOperationParam72)
{
    train::GenAttentionMaskParam param;
    param.headNum = 2;
    param.seqLen = {-1};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);

    train::GenAttentionMaskParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestOpParamFuncs, CloneOperationParam8)
{
    train::PadWithHiddenStateParam param;
    param.qSeqLen = {8,8};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::PadWithHiddenStateParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(cloneParam, param);

    for (auto len : cloneParam.qSeqLen) {
        ATB_LOG(INFO) << "cloneParam qSeqLen: " << len;
    }
    ATB_LOG(INFO) << "cloneParam maxSeqLen: " << cloneParam.maxSeqLen;
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, CloneOperationParam9)
{
    train::PadWithHiddenStateParam param;
    param.qSeqLen = {8,8,8,8,8};
    param.maxSeqLen = 4097;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);

    train::PadWithHiddenStateParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestOpParamFuncs, CloneOperationParam90)
{
    train::PadWithHiddenStateParam param;
    param.qSeqLen = {};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);

    train::PadWithHiddenStateParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestOpParamFuncs, CloneOperationParam10)
{
    train::FastSoftMaxParam param;
    param.headNum = 3;
    param.qSeqLen = { -1 };
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);

    train::FastSoftMaxParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestOpParamFuncs, CloneOperationParam100)
{
    train::FastSoftMaxParam param;
    param.headNum = 3;
    param.qSeqLen = {};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);

    train::FastSoftMaxParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestOpParamFuncs, CloneOperationParam11)
{
    train::RopeGradParam param;
    param.qSeqLen = {8,8};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::RopeGradParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(cloneParam, param);

    for (auto len : cloneParam.qSeqLen) {
        ATB_LOG(INFO) << "cloneParam qSeqLen: " << len;
    }
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, CloneOperationParam13)
{
    train::RopeGradParam param;
    param.qSeqLen = {0,-1};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);

    train::RopeGradParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestOpParamFuncs, CloneOperationParam130)
{
    train::RopeGradParam param;
    param.qSeqLen = {};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);

    train::RopeGradParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestOpParamFuncs, CloneOperationParam14)
{
    train::StridedBatchMatmulParam param;
    param.transposeA = false;
    param.transposeB = true;
    param.batch = 1;
    param.headNum = 1;
    param.m = {16};
    param.n = {16};
    param.k = {16};
    param.lda = {16};
    param.ldb = {16};
    param.ldc = {16};
    param.strideA = {16};
    param.strideB = {16};
    param.strideC = {256};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::StridedBatchMatmulParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(cloneParam, param);
    for (auto len : cloneParam.m) {
        ATB_LOG(INFO) << "cloneParam m: " << len;
    }
    for (auto len : cloneParam.n) {
        ATB_LOG(INFO) << "cloneParam n: " << len;
    }
    for (auto len : cloneParam.k) {
        ATB_LOG(INFO) << "cloneParam k: " << len;
    }
    for (auto len : cloneParam.lda) {
        ATB_LOG(INFO) << "cloneParam lda: " << len;
    }
    for (auto len : cloneParam.ldb) {
        ATB_LOG(INFO) << "cloneParam ldb: " << len;
    }
    for (auto len : cloneParam.ldc) {
        ATB_LOG(INFO) << "cloneParam ldc: " << len;
    }
    for (auto len : cloneParam.strideA) {
        ATB_LOG(INFO) << "cloneParam strideA: " << len;
    }
    for (auto len : cloneParam.strideB) {
        ATB_LOG(INFO) << "cloneParam strideB: " << len;
    }
    for (auto len : cloneParam.strideC) {
        ATB_LOG(INFO) << "cloneParam strideC: " << len;
    }
    ATB_LOG(INFO) << "cloneParam transposeA: " << cloneParam.transposeA;
    ATB_LOG(INFO) << "cloneParam transposeB: " << cloneParam.transposeB;
    ATB_LOG(INFO) << "cloneParam batch: " << cloneParam.batch;
    ATB_LOG(INFO) << "cloneParam headNum: " << cloneParam.headNum;
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, CloneOperationParam16)
{
    train::UnpadWithHiddenStateParam param;
    param.qSeqLen = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::UnpadWithHiddenStateParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(cloneParam, param);

    for (auto len : cloneParam.qSeqLen) {
        ATB_LOG(INFO) << "cloneParam qSeqLen: " << len;
    }
    ATB_LOG(INFO) << "cloneParam maxSeqLen: " << cloneParam.maxSeqLen;
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, CloneOperationParam17)
{
    train::UnpadWithHiddenStateParam param;
    param.qSeqLen = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8};
    param.maxSeqLen = 4097;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);

    train::UnpadWithHiddenStateParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);

}

TEST(TestOpParamFuncs, CloneOperationParam170)
{
    train::UnpadWithHiddenStateParam param;
    param.qSeqLen = {};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    EXPECT_EQ(op, nullptr);

    train::UnpadWithHiddenStateParam cloneParam;
    st = atb::CloneOperationParam(op, cloneParam);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);

}

TEST(TestOpParamFuncs, UpdateOperationParam1)
{
    train::FastSoftMaxParam param;
    param.headNum = 8;
    param.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
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

TEST(TestOpParamFuncs, UpdateOperationParam2)
{
    train::FastSoftMaxParam param;
    param.headNum = 8;
    param.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
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

    getParam.headNum = 77;
    st = atb::UpdateOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);

    getParam.headNum = 88;
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

TEST(TestOpParamFuncs, UpdateOperationParam15)
{
    train::FastSoftMaxParam param;
    param.headNum = 8;
    param.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::FastSoftMaxParam getParam;
    st = atb::CloneOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParam, param);

    getParam.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    st = atb::UpdateOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);

    getParam.qSeqLen[0] = {3};
    st = atb::UpdateOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);

    train::FastSoftMaxParam getParamAfter;
    st = atb::CloneOperationParam(op, getParamAfter);
    for (auto len : getParam.qSeqLen) {
        ATB_LOG(INFO) << "getParam qSeqLen: " << len;
    }
    ATB_LOG(INFO) << "after update headNum: " << getParamAfter.headNum;
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParamAfter, getParam);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParamCheckParamFailed4)
{
    train::FastSoftMaxParam param;
    param.headNum = 8;
    param.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParamCheckParamFailed40)
{
    train::FastSoftMaxParam param;
    param.headNum = 8;
    param.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.qSeqLen = {};
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParam3)
{
    train::FastSoftMaxGradParam param;
    param.headNum = 8;
    param.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::FastSoftMaxGradParam getParam;
    st = atb::CloneOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParam, param);

    getParam.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    st = atb::UpdateOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);

    getParam.qSeqLen[0] = {3};
    st = atb::UpdateOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);

    train::FastSoftMaxGradParam getParamAfter;
    st = atb::CloneOperationParam(op, getParamAfter);
    for (auto len : getParam.qSeqLen) {
        ATB_LOG(INFO) << "getParam qSeqLen: " << len;
    }
    ATB_LOG(INFO) << "after update headNum: " << getParamAfter.headNum;
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParamAfter, getParam);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParam0)
{
    train::FastSoftMaxGradParam param;
    param.headNum = 8;
    param.qSeqLen = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::FastSoftMaxGradParam getParam;
    st = atb::CloneOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParam, param);

    getParam.qSeqLen = {8,8};
    st = atb::UpdateOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);

    getParam.qSeqLen = {9,9};
    st = atb::UpdateOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);

    train::FastSoftMaxGradParam getParamAfter;
    st = atb::CloneOperationParam(op, getParamAfter);
    for (auto len : getParam.qSeqLen) {
        ATB_LOG(INFO) << "getParam qSeqLen: " << len;
    }
    ATB_LOG(INFO) << "after update headNum: " << getParamAfter.headNum;
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParamAfter, getParam);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParamCheckParamFailed2)
{
    train::FastSoftMaxGradParam param;
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

TEST(TestOpParamFuncs, UpdateOperationParamCheckParamFailed20)
{
    train::FastSoftMaxGradParam param;
    param.headNum = 3;
    param.qSeqLen = { 10, 20, 30 };
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.qSeqLen = {};
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParam8)
{
    train::GenAttentionMaskParam param;
    param.headNum = 2;
    param.seqLen = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::GenAttentionMaskParam getParam;
    st = atb::CloneOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParam, param);

    getParam.headNum = 66;
    st = atb::UpdateOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);

    param.seqLen = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    st = atb::UpdateOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);

    train::GenAttentionMaskParam getParamAfter;
    st = atb::CloneOperationParam(op, getParamAfter);
    ATB_LOG(INFO) << "after update headNum: " << getParamAfter.headNum;
    for (auto len : getParamAfter.seqLen) {
        ATB_LOG(INFO) << "getParamAfter seqLen: " << len;
    }
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParamAfter, getParam);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParamCheckParamFailed1)
{
    train::GenAttentionMaskParam param;
    param.headNum = 2;
    param.seqLen = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.seqLen ={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParamCheckParamFailed10)
{
    train::GenAttentionMaskParam param;
    param.headNum = 2;
    param.seqLen = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.seqLen ={};
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParamCheckParamFailed11)
{
    train::GenAttentionMaskParam param;
    param.headNum = 2;
    param.seqLen = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.seqLen ={-1};
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParamCheckParamFailed12)
{
    train::GenAttentionMaskParam param;
    param.headNum = 2;
    param.seqLen = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.seqLen ={0};
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParam9)
{
    train::PadWithHiddenStateParam param;
    param.qSeqLen = {8,8};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::PadWithHiddenStateParam getParam;
    st = atb::CloneOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParam, param);

    getParam.maxSeqLen = 4095;
    getParam.qSeqLen = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    st = atb::UpdateOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);

    train::PadWithHiddenStateParam getParamAfter;
    st = atb::CloneOperationParam(op, getParamAfter);
    ATB_LOG(INFO) << "after update maxSeqLen: " << getParamAfter.maxSeqLen;
    for (auto len : getParamAfter.qSeqLen) {
        ATB_LOG(INFO) << "getParamAfter qSeqLen: " << len;
    }
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParamAfter, getParam);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParamCheckParamFailed3)
{
    train::PadWithHiddenStateParam param;
    param.qSeqLen = {8,8};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.maxSeqLen =4097;
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParamCheckParamFailed30)
{
    train::PadWithHiddenStateParam param;
    param.qSeqLen = {8,8};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.qSeqLen = {};
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParamCheckParamFailed5)
{
    train::PadWithHiddenStateParam param;
    param.qSeqLen = {8,8};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.qSeqLen ={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33};
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParam10)
{
    train::StridedBatchMatmulParam param;
    param.transposeA = false;
    param.transposeB = false;
    param.batch = 1;
    param.headNum = 1;
    param.m = {16};
    param.n = {16};
    param.k = {16};
    param.lda = {16};
    param.ldb = {16};
    param.ldc = {16};
    param.strideA = {16};
    param.strideB = {16};
    param.strideC = {256};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::StridedBatchMatmulParam getParam;
    st = atb::CloneOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParam, param);

    getParam.transposeA = true;
    getParam.transposeB = true;
    st = atb::UpdateOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);

    train::StridedBatchMatmulParam getParamAfter;
    st = atb::CloneOperationParam(op, getParamAfter);
    ATB_LOG(INFO) << "after update transposeA: " << getParamAfter.transposeA;
    ATB_LOG(INFO) << "after update transposeB: " << getParamAfter.transposeB;
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParamAfter, getParam);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParam11)
{
    train::UnpadWithHiddenStateParam param;
    param.qSeqLen = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    train::UnpadWithHiddenStateParam getParam;
    st = atb::CloneOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParam, param);

    getParam.maxSeqLen = 4095;
    getParam.qSeqLen = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8};
    st = atb::UpdateOperationParam(op, getParam);
    EXPECT_EQ(st, atb::NO_ERROR);

    train::UnpadWithHiddenStateParam getParamAfter;
    st = atb::CloneOperationParam(op, getParamAfter);
    ATB_LOG(INFO) << "after update maxSeqLen: " << getParamAfter.maxSeqLen;
    for (auto len : getParamAfter.qSeqLen) {
        ATB_LOG(INFO) << "getParamAfter qSeqLen: " << len;
    }
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_EQ(getParamAfter, getParam);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParam12)
{
    train::UnpadWithHiddenStateParam param;
    param.qSeqLen = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.maxSeqLen = 4097;
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParam13)
{
    train::UnpadWithHiddenStateParam param;
    param.qSeqLen = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8};
    param.maxSeqLen = 4096;
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.qSeqLen ={8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8};
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    st = DestroyOperation(op);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestOpParamFuncs, UpdateOperationParam16)
{
    train::RopeGradParam param;
    param.qSeqLen = {8,8};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.qSeqLen = {};
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestOpParamFuncs, UpdateOperationParam17)
{
    train::RopeGradParam param;
    param.qSeqLen = {8,8};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.qSeqLen ={-1};
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestOpParamFuncs, UpdateOperationParam18)
{
    train::RopeGradParam param;
    param.qSeqLen = {8,8};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.qSeqLen ={};
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}

TEST(TestOpParamFuncs, UpdateOperationParam19)
{
    train::RopeGradParam param;
    param.qSeqLen = {8,8};
    atb::Operation *op = nullptr;

    atb::Status st = atb::CreateOperation(param, &op);
    EXPECT_EQ(st, atb::NO_ERROR);
    EXPECT_NE(op, nullptr);

    param.qSeqLen ={-1};
    st = atb::UpdateOperationParam(op, param);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
}



