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
#include <iostream>
#include <torch/torch.h>
#include <atb/utils/log.h>
#include <acl/acl.h>
#include <cpp-stub/src/stub.h>
#include "test_utils/test_common.h"
#include "atb/operation.h"
#include "atb/infer_op_params.h"
#include "atb/utils/tensor_util.h"
#include "test_utils/operation_test.h"
#include <atb/utils/probe.h>
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"

using namespace atb;
using namespace Mki;

bool g_isOverflowedVar = false;

bool IsOverflowCheckStub()
{
    return true;
}

void ReportOverflowKernelStub(const std::string &kernelPath)
{
    g_isOverflowedVar = true;
    std::cout << "----------------" << kernelPath << " overflowed!\n";
}

TEST(TestOverflow, TestNotOverflowByAdd)
{
    aclrtSetDeviceSatMode(ACL_RT_OVERFLOW_MODE_SATURATION);
    bool is910B = GetSingleton<Config>().Is910B();
    if (!is910B) {
        return;
    }
    Stub stub;
    stub.set(ADDR(Probe, IsOverflowCheck), IsOverflowCheckStub);
    stub.set(ADDR(Probe, ReportOverflowKernel), ReportOverflowKernelStub);
    atb::infer::ElewiseParam param;
    param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Operation *op = nullptr;
    atb::CreateOperation<atb::infer::ElewiseParam>(param, &op);
    Mki::SVector<Mki::TensorDesc> inTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, {1}},
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, {1}}};
    ASSERT_NE(op, nullptr);
    OperationTest opTest;
    opTest.FloatRand(FLT_MIN, FLT_MIN);
    Mki::Status status = opTest.Run(op, inTensorDescs);
    atb::DestroyOperation(op);
    ASSERT_EQ(status.Ok(), true);
    ASSERT_EQ(g_isOverflowedVar, false);
    (void)aclFinalize();
}

TEST(TestOverflow, TestOverflowByAdd)
{
    bool is910B = GetSingleton<Config>().Is910B();
    if (!is910B) {
        return;
    }
    Stub stub;
    stub.set(ADDR(Probe, IsOverflowCheck), IsOverflowCheckStub);
    stub.set(ADDR(Probe, ReportOverflowKernel), ReportOverflowKernelStub);
    atb::infer::ElewiseParam param;
    param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Operation *op = nullptr;
    atb::CreateOperation<atb::infer::ElewiseParam>(param, &op);
    Mki::SVector<Mki::TensorDesc> inTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {1}},
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {1}}};
    ASSERT_NE(op, nullptr);
    OperationTest opTest;
    opTest.FloatRand(FLT_MAX, FLT_MAX);
    Mki::Status status = opTest.Run(op, inTensorDescs);
    atb::DestroyOperation(op);
    ASSERT_EQ(status.Ok(), true);
    ASSERT_EQ(g_isOverflowedVar, true);
    (void)aclFinalize();
}

TEST(TestOverflow, TestOverflowByAdd2)
{
    aclrtSetDeviceSatMode(ACL_RT_OVERFLOW_MODE_INFNAN);
    bool is910B = GetSingleton<Config>().Is910B();
    if (!is910B) {
        return;
    }
    Stub stub;
    stub.set(ADDR(Probe, IsOverflowCheck), IsOverflowCheckStub);
    stub.set(ADDR(Probe, ReportOverflowKernel), ReportOverflowKernelStub);
    atb::infer::ElewiseParam param;
    param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Operation *op = nullptr;
    atb::CreateOperation<atb::infer::ElewiseParam>(param, &op);
    Mki::SVector<Mki::TensorDesc> inTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, {1}},
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, {1}}};
    ASSERT_NE(op, nullptr);
    OperationTest opTest;
    opTest.FloatRand(FLT_MAX, FLT_MAX);
    Mki::Status status = opTest.Run(op, inTensorDescs);
    atb::DestroyOperation(op);
    ASSERT_EQ(status.Ok(), true);
    ASSERT_EQ(g_isOverflowedVar, true);
    (void)aclFinalize();
}