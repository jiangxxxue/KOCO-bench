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
#include <torch/torch.h>
#include "test_common.h"
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float HALF_FLOAT_MIN = 1;
constexpr float HALF_FLOAT_MAX = 65500;

template <at::ScalarType ATT = at::kFloat, typename T = float>
Status RealDivGolden(const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor1 = context.hostInTensors.at(0);
    at::Tensor atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), ATT);
    const Tensor &inTensor2 = context.hostInTensors.at(1);
    at::Tensor atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), ATT);

    const Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), ATT);
    at::Tensor refOutTensor = atInRefTensor1.div(atInRefTensor2);
    T *atOutArray = (T *)atOutTensor.storage().data_ptr().get();
    T *atRefOutArray = (T *)refOutTensor.storage().data_ptr().get(); // golden

    for (int i = 0; i < outTensor.Numel(); i++) {
        float expect = static_cast<float>(atRefOutArray[i]);
        float actual = static_cast<float>(atOutArray[i]);
        bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
        if (!judge) {
            MKI_LOG(ERROR) << "index[" << i << "], result: " << actual << ", expect: " << expect;
        }
        EXPECT_EQ(judge, true);
    }
    return Status::OkStatus();
}

TEST(TestOpElewiseRealDiv, realdivTest1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(RealDivGolden);
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 32;
    int64_t n = 32;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n}},
                                        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseRealDiv, realdivTest2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(RealDivGolden);
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 10;
    int64_t n = 70;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n}},
                                        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseRealDiv, realdivTest3)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(RealDivGolden);
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, k, k, b}},
                                        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, 1, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseRealDiv, realdivTest4)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(RealDivGolden);
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, 1, 1, b}},
                                        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, k, k, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseRealDiv, realdivTest5)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(RealDivGolden);
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, k, n, b}},
                                        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1, n, k, n, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseRealDiv, realdivTest6)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(RealDivGolden);
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, k, n, b}},
                                        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, 1, k, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseRealDiv, realdivF16Test1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(RealDivGolden<at::kHalf, fp16_t>);
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 32;
    int64_t n = 32;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseRealDiv, realdivF16Test2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(RealDivGolden<at::kHalf, fp16_t>);
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 10;
    int64_t n = 70;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseRealDiv, realdivF16Test3)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(RealDivGolden<at::kHalf, fp16_t>);
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, k, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, 1, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseRealDiv, realdivF16Test5)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(RealDivGolden<at::kHalf, fp16_t>);
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, n, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, n, k, n, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseRealDiv, realdivF16Test6)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(RealDivGolden<at::kHalf, fp16_t>);
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, n, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, 1, k, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseRealDiv, TestGetBestKernel)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_BOOL, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_BOOL, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_BOOL, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}

/**
 * @brief ok
 */
TEST(TestOpElewiseRealDiv, TestCanSupport0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("RealDivF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief elewiseType wrong
 */
TEST(TestOpElewiseRealDiv, TestCanSupport1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_ADD;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("RealDivF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestOpElewiseRealDiv, TestCanSupport2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("RealDivF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestOpElewiseRealDiv, TestCanSupport3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("RealDivF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpElewiseRealDiv, TestCanSupport4)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("RealDivF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor dtype wrong
 */
TEST(TestOpElewiseRealDiv, TestCanSupport5)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_REALDIV;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("RealDivF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}