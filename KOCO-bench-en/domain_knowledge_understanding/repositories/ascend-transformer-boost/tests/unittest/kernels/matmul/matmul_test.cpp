/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include <mki/launch_param.h>
#include "asdops/params/params.h"
#include "kernels/matmul/tiling/pp_matmul_i8_nz_tiling.h"
#include "kernels/matmul/tiling/pp_matmul_tiling.h"
#include "kernels/matmul/tiling/pp_matmul_mix_tiling.h"
#include "kernels/matmul/tiling/matmul_nd_tiling.h"
#include "kernels/matmul/tiling/matmul_nz_tiling.h"
#include "device_version_check.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float HALF_FLOAT_MIN = -5.0;
constexpr float HALF_FLOAT_MAX = 5.0;

namespace AsdOps {
Status MatMulNzGolden(float atol, float rtol, OpParam::MatMul opParam, const Mki::Test::GoldenContext &context)
{
    const Tensor &tensorA = context.hostInTensors.at(0);
    const Tensor &tensorB = context.hostInTensors.at(1);
    const Tensor &tensorOut = context.hostOutTensors.at(0);
    auto &dimsA = tensorA.desc.dims;
    auto &dimsB = tensorB.desc.dims;
    auto &dimsOut = tensorOut.desc.dims;

    auto attensorA = at::from_blob(
        tensorA.data, {dimsA[0], dimsA[1], dimsA[2], dimsA[3]}, at::kHalf); // b, k1, m1m0, k0
    attensorA = at::transpose(attensorA, 1, 2).contiguous(); // b, m1m0, k1, k0
    attensorA = at::reshape(attensorA, {dimsA[0], dimsA[2], dimsA[1] * dimsA[3]}).to(at::kFloat); // b, m1m0, k1k0
    if (opParam.transposeA) {
        attensorA = at::transpose(attensorA, 1, 2).contiguous();
    }

    auto attensorB = at::from_blob(
        tensorB.data, {dimsB[0], dimsB[1], dimsB[2], dimsB[3]}, at::kHalf); // b, n1, k1k0, n0
    attensorB = at::transpose(attensorB, 1, 2).contiguous(); // b, k1k0, n1, n0
    attensorB = at::reshape(attensorB, {dimsB[0], dimsB[2], dimsB[1] * dimsB[3]}).to(at::kFloat); // b, k1k0, n1n0
    if (opParam.transposeB) {
        attensorB = at::transpose(attensorB, 1, 2).contiguous();
    }

    auto attensorOut = at::from_blob(
        tensorOut.data, {dimsOut[0], dimsOut[1], dimsOut[2], dimsOut[3]}, at::kHalf); // b, n1, m1m0, n0
    attensorOut = at::transpose(attensorOut, 1, 2).contiguous(); // b, m1m0, n1, n0
    attensorOut = at::reshape(attensorOut, {dimsOut[0], dimsOut[2], dimsOut[1] * dimsOut[3]}); // b, m1m0, n1n0

    auto attensorGt = at::matmul(attensorA, attensorB).to(at::kHalf);

    return Mki::Test::FloatUtil::MatchAtTensorFloat<fp16_t>(attensorOut, attensorGt, atol, rtol);
}
} // namespace AsdOps

/**
 * @brief ok
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz0)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, m1, k1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {true, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz1)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, m1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {true, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz2)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, m1, k1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {true, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor0 format wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz3)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {b, m1, k1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024}}});
    OpParam::MatMul opParam = {true, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor0 dtype wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz4)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_FRACTAL_NZ, {b, m1, k1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024}}});
    OpParam::MatMul opParam = {true, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor0 dim size wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz5)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, m1, k1 * mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024}}});
    OpParam::MatMul opParam = {true, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor1 format wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz6)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, m1, k1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024}}});
    OpParam::MatMul opParam = {true, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor1 dtype wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz7)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, m1, k1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024}}});
    OpParam::MatMul opParam = {true, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor1 dim size wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz8)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, m1, k1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024}}});
    OpParam::MatMul opParam = {true, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor dim size wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz9)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, m1, k1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024}}});
    OpParam::MatMul opParam = {true, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor format wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz10)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, m1, k1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024, 2, 4}}});
    OpParam::MatMul opParam = {true, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor dtype wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz11)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, m1, k1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024, 2, 4}}});
    OpParam::MatMul opParam = {true, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief opDesc wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz12)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, m1, k1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024, 2, 4}}});
    OpParam::Gather opParam;
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief opParam.transposeA wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz13)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, m1, k1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024, 2, 4}}});
    OpParam::MatMul opParam = {false, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief opParam.transposeB wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz14)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, m1, k1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024, 2, 4}}});
    OpParam::MatMul opParam = {true, true, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
/**
 * @brief opParam.oriShape.size() == 3 wrong
 */
TEST(TestMatMulNz, TestInitRunInfoImpltMatMulNz110)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, m1, k1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024, 2, 4}}});
    OpParam::MatMul opParam = {true, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0, 1}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    Status status = kernel->Init(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief ok
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz15)
{
    int64_t b = 2;
    int64_t m1 = 1;
    int64_t k1 = 256;
    int64_t n1 = 688;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, k1, m1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024, 2, 4}}});
    OpParam::MatMul opParam = {false, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief opParam.transposeA wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz16)
{
    int64_t b = 2;
    int64_t m1 = 1;
    int64_t k1 = 256;
    int64_t n1 = 688;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, k1, m1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024, 2, 4}}});
    OpParam::MatMul opParam = {true, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief opParam.transposeB wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz17)
{
    int64_t b = 2;
    int64_t m1 = 1;
    int64_t k1 = 256;
    int64_t n1 = 688;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, k1, m1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024, 2, 4}}});
    OpParam::MatMul opParam = {false, true, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
/**
 * @brief opParam.transposeA && opParam.transposeB wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz18)
{
    int64_t b = 2;
    int64_t m1 = 1;
    int64_t k1 = 256;
    int64_t n1 = 688;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, k1, m1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024, 2, 4}}});
    //opParam.transposeA true opParam.transposeB false
    OpParam::MatMul opParam = {true, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TATBKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //opParam.transposeA false opParam.transposeB false
    OpParam::MatMul opParam0 = {false, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc0 = {"MatMulOperation", opParam0};
    launchParam.SetParam(opDesc0.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //opParam.transposeA false opParam.transposeB true
    OpParam::MatMul opParam1 = {false, true, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc1 = {"MatMulOperation", opParam1};
    launchParam.SetParam(opDesc1.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz21)
{
    int64_t b = 8;
    int64_t m1 = 28;
    int64_t k1 = 15;
    int64_t n1 = 31;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, k1, m1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, k1, n1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024, 2, 4}}});
    OpParam::MatMul opParam = {false, true, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TBKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief opParam.transposeA wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz22)
{
    int64_t b = 8;
    int64_t m1 = 28;
    int64_t k1 = 15;
    int64_t n1 = 31;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, k1, m1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, k1, n1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024, 2, 4}}});
    OpParam::MatMul opParam = {true, true, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TBKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief opParam.transposeB wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz23)
{
    int64_t b = 8;
    int64_t m1 = 28;
    int64_t k1 = 15;
    int64_t n1 = 31;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, k1, m1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, k1, n1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 1024, 2, 4}}});
    OpParam::MatMul opParam = {false, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("BatchMatMulNzF16TBKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz24)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    OpParam::MatMul opParam = {true, true, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MatMulNzF16TATBKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief ok
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz25)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    OpParam::MatMul opParam = {true, false, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief ok
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz26)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    OpParam::MatMul opParam = {false, false, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MatMulNzF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief ok
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz27)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    OpParam::MatMul opParam = {false, true, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MatMulNzF16TBKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}
/**
 * @brief !opParam.transposeA && opParam.transposeB wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz28)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    //opParam.transposeA false opParam.transposeB false wrong
    OpParam::MatMul opParam = {false, false, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MatMulNzF16TBKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //opParam.transposeA true opParam.transposeB true wrong
    OpParam::MatMul opParam0 = {true, true, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc0 = {"MatMulOperation", opParam0};
    launchParam.SetParam(opDesc0.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //opParam.transposeA true opParam.transposeB false wrong
    OpParam::MatMul opParam1 = {true, false, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc1 = {"MatMulOperation", opParam1};
    launchParam.SetParam(opDesc1.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief opParam.transposeA && (!opParam.transposeB) A=false B=false wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz31)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    //opParam.transposeA false opParam.transposeB false wrong
    OpParam::MatMul opParam = {false, false, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MatMulNzF16TAKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //opParam.transposeA true opParam.transposeB true wrong
    OpParam::MatMul opParam0 = {true, true, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc0 = {"MatMulOperation", opParam0};
    launchParam.SetParam(opDesc0.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //opParam.transposeA false opParam.transposeB true wrong
    OpParam::MatMul opParam1 = {false, true, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc1 = {"MatMulOperation", opParam1};
    launchParam.SetParam(opDesc1.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief opParam.transposeA && opParam.transposeB wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz34)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    //opParam.transposeA false opParam.transposeB false wrong
    OpParam::MatMul opParam = {false, false, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MatMulNzF16TATBKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //opParam.transposeA false opParam.transposeB true wrong
    OpParam::MatMul opParam0 = {false, true, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc0 = {"MatMulOperation", opParam0};
    launchParam.SetParam(opDesc0.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //opParam.transposeA true opParam.transposeB false wrong
    OpParam::MatMul opParam1 = {true, false, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc1 = {"MatMulOperation", opParam1};
    launchParam.SetParam(opDesc1.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
/**
 * @brief (!opParam.transposeA) && (!opParam.transposeB) 
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz37)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    //opParam.transposeA false opParam.transposeB true wrong
    OpParam::MatMul opParam = {false, true, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MatMulNzF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //opParam.transposeA true opParam.transposeB true wrong
    OpParam::MatMul opParam0 = {true, true, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc0 = {"MatMulOperation", opParam0};
    launchParam.SetParam(opDesc0.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //opParam.transposeA true opParam.transposeB false wrong
    OpParam::MatMul opParam1 = {true, false, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc1 = {"MatMulOperation", opParam1};
    launchParam.SetParam(opDesc1.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
/**
 * @brief runinfo.GetInTensorCount() == 2 wrong
 */
TEST(TestMatMulNz, TestCanSupportMatMulNz40)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    OpParam::MatMul opParam = {false, false, {1, 1, 1}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MatMulNzF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //runinfo.GetOutTensorCount() == 1 wrong
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //mmNzInTensor0.desc.format == TENSOR_FORMAT_FRACTAL_NZ wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //mmNzInTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //mmNzInTensor0.desc.dims.size() == 4 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //mmNzInTensor1.desc.format == TENSOR_FORMAT_FRACTAL_NZ wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NC1HWC0_C04, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //mmNzInTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //mmNzInTensor1.desc.dims.size() == 4 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //mmNzIOutTensor.desc.format == TENSOR_FORMAT_FRACTAL_NZ wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NC1HWC0_C04, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //mmNzIOutTensor.desc.dtype == TENSOR_DTYPE_FLOAT16 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    //mmNzIOutTensor.desc.dims.size() == 4 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
/**
 * @brief ok
 */
TEST(TestMatMulNz, TestInferShapeMatMul0)
{
    int64_t b = 8;
    int64_t m1 = 15;
    int64_t k1 = 31;
    int64_t n1 = 28;
    int64_t mkn0 = 16;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, m1, k1 * mkn0, mkn0}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {b, n1, k1 * mkn0, mkn0}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {true, false, {m1 * mkn0, k1 * mkn0, n1 * mkn0}};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
}