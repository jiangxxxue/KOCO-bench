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
#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "kernels/matmul/tiling/pp_matmul_i8_nz_tiling.h"
#include "kernels/matmul/tiling/pp_matmul_tiling.h"
#include "kernels/matmul/tiling/pp_matmul_mix_tiling.h"
#include "kernels/matmul/tiling/matmul_nd_tiling.h"
#include "device_version_check.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float HALF_FLOAT_MIN = -1.0;
constexpr float HALF_FLOAT_MAX = 1.0;

namespace AsdOps {
Status MatchAtTensorHalf(float atol, float rtol, at::Tensor &out, at::Tensor &gt)
{
    fp16_t *result = static_cast<fp16_t *>(out.storage().data_ptr().get());
    fp16_t *expect = static_cast<fp16_t *>(gt.storage().data_ptr().get());
    for (int i = 0; i < out.numel(); i++) {
        if (!Mki::Test::FloatUtil::FloatJudgeEqual(expect[i], result[i], atol, rtol)) {
            std::string msg = "pos " + std::to_string(i) +
                              ", expect: " + std::to_string(static_cast<float>(expect[i])) +
                              ", result: " + std::to_string(static_cast<float>(result[i]));
            return Status::FailStatus(-1, msg);
        }
    }
    return Status::OkStatus();
}

Status MatMulNdGolden(float atol, float rtol, OpParam::MatMul opParam, const Mki::Test::GoldenContext &context)
{
    const Tensor &tensorA = context.hostInTensors.at(0);
    const Tensor &tensorB = context.hostInTensors.at(1);
    const Tensor &tensorOut = context.hostOutTensors.at(0);

    auto attensorA =
        at::from_blob(tensorA.data, {tensorA.desc.dims[0], tensorA.desc.dims[1]}, at::kHalf).to(at::kFloat);
    auto attensorB =
        at::from_blob(tensorB.data, {tensorB.desc.dims[0], tensorB.desc.dims[1]}, at::kHalf).to(at::kFloat);
    if (opParam.transposeA) {
        attensorA = at::transpose(attensorA, 0, 1).contiguous();
    }
    if (opParam.transposeB) {
        attensorB = at::transpose(attensorB, 0, 1).contiguous();
    }
    auto attensorOut = at::from_blob(tensorOut.data, {tensorOut.desc.dims[0], tensorOut.desc.dims[1]}, at::kHalf);
    auto attensorGt = at::matmul(attensorA, attensorB).to(at::kHalf);

    return MatchAtTensorHalf(atol, rtol, attensorOut, attensorGt);
}

Status MatMulNdMixGolden(float atol, float rtol, OpParam::MatMul opParam, const Mki::Test::GoldenContext &context)
{
    const Tensor &tensorA = context.hostInTensors.at(0);
    const Tensor &tensorB = context.hostInTensors.at(1);
    const Tensor &tensorbias = context.hostInTensors.at(2);
    const Tensor &tensorOut = context.hostOutTensors.at(0);

    auto attensorA =
        at::from_blob(tensorA.data, {tensorA.desc.dims[0], tensorA.desc.dims[1]}, at::kHalf).to(at::kFloat);
    auto attensorB =
        at::from_blob(tensorB.data, {tensorB.desc.dims[0], tensorB.desc.dims[1]}, at::kHalf).to(at::kFloat);
    auto attensorbias =
        at::from_blob(tensorbias.data, {tensorbias.desc.dims[0], tensorbias.desc.dims[1]}, at::kHalf).to(at::kFloat);
    if (opParam.transposeA) {
        attensorA = at::transpose(attensorA, 0, 1).contiguous();
    }
    if (opParam.transposeB) {
        attensorB = at::transpose(attensorB, 0, 1).contiguous();
    }
    auto attensorOut = at::from_blob(tensorOut.data, {tensorOut.desc.dims[0], tensorOut.desc.dims[1]}, at::kHalf);
    auto matmulRes = at::matmul(attensorA, attensorB).to(at::kHalf);
    auto attensorGt = at::add(matmulRes, attensorbias).to(at::kHalf);
    return MatchAtTensorHalf(atol, rtol, attensorOut, attensorGt);
}

Status BatchMatMulNdGolden(float atol, float rtol, OpParam::MatMul opParam, const Mki::Test::GoldenContext &context)
{
    const Tensor &tensorA = context.hostInTensors.at(0);
    const Tensor &tensorB = context.hostInTensors.at(1);
    const Tensor &tensorOut = context.hostOutTensors.at(0);

    auto attensorA =
        at::from_blob(tensorA.data, {tensorA.desc.dims[0], tensorA.desc.dims[1], tensorA.desc.dims[2]}, at::kHalf)
            .to(at::kFloat);
    auto attensorB =
        at::from_blob(tensorB.data, {tensorA.desc.dims[0], tensorB.desc.dims[1], tensorB.desc.dims[2]}, at::kHalf)
            .to(at::kFloat);
    if (opParam.transposeA) {
        attensorA = at::transpose(attensorA, 1, 2).contiguous();
    }
    if (opParam.transposeB) {
        attensorB = at::transpose(attensorB, 1, 2).contiguous();
    }
    auto attensorOut = at::from_blob(tensorOut.data, {tensorOut.desc.dims[0], tensorOut.desc.dims[1]}, at::kHalf);
    auto attensorGt = at::matmul(attensorA, attensorB).to(at::kHalf);

    return MatchAtTensorHalf(atol, rtol, attensorOut, attensorGt);
}
Status MatMulNdIntGolden(float atol, float rtol, OpParam::MatMul opParam, const Mki::Test::GoldenContext &context)
{
    const Tensor &tensorA = context.hostInTensors.at(0);
    const Tensor &tensorB = context.hostInTensors.at(1);
    const Tensor &tensorBias = context.hostInTensors.at(2);
    const Tensor &tensorDescale = context.hostInTensors.at(3);

    const Tensor &tensorOut = context.hostOutTensors.at(0);
    std::vector<int64_t> shape = {tensorA.desc.dims[0], tensorB.desc.dims[1]};
    auto attensorA = at::from_blob(tensorA.data, {tensorA.desc.dims[0], tensorA.desc.dims[1]}, at::kChar).to(at::kInt);
    auto attensorB = at::from_blob(tensorB.data, {tensorB.desc.dims[0], tensorB.desc.dims[1]}, at::kChar).to(at::kInt);
    auto attensorBias = at::from_blob(tensorBias.data, {tensorBias.desc.dims[0], tensorBias.desc.dims[1]}, at::kInt);
    auto attensorDescale =
        at::from_blob(tensorDescale.data, {tensorDescale.desc.dims[0], tensorDescale.desc.dims[1]}, at::kFloat);
    if (opParam.transposeA) {
        attensorA = at::transpose(attensorA, 0, 1).contiguous();
    }
    if (opParam.transposeB) {
        attensorB = at::transpose(attensorB, 0, 1).contiguous();
        shape = {tensorA.desc.dims[0], tensorB.desc.dims[0]};
    }
    auto attensorOut = at::from_blob(tensorOut.data, {tensorOut.desc.dims[0], tensorOut.desc.dims[1]}, at::kHalf);
    // 计算golden
    auto matmulRes = at::matmul(attensorA, attensorB).to(at::kInt);
    auto addRes = at::add(matmulRes, at::broadcast_to(attensorBias, shape)).to(at::kInt);
    auto attensorGt = at::mul(addRes, at::broadcast_to(attensorDescale, shape)).to(at::kHalf);

    return MatchAtTensorHalf(atol, rtol, attensorOut, attensorGt);
}

Status BatchMatMulNdIntGolden(float atol, float rtol, OpParam::MatMul opParam, const Mki::Test::GoldenContext &context)
{
    const Tensor &tensorA = context.hostInTensors.at(0);
    const Tensor &tensorB = context.hostInTensors.at(1);
    const Tensor &tensorBias = context.hostInTensors.at(2);
    const Tensor &tensorDescale = context.hostInTensors.at(3);
    const Tensor &tensorOut = context.hostOutTensors.at(0);
    std::vector<int64_t> shape = {tensorA.desc.dims[0], tensorA.desc.dims[1], tensorB.desc.dims[2]};
    auto attensorA =
        at::from_blob(tensorA.data, {tensorA.desc.dims[0], tensorA.desc.dims[1], tensorA.desc.dims[2]}, at::kChar)
            .to(at::kInt);
    auto attensorB =
        at::from_blob(tensorB.data, {tensorB.desc.dims[0], tensorB.desc.dims[1], tensorB.desc.dims[2]}, at::kChar)
            .to(at::kInt);
    auto attensorBias = at::from_blob(tensorBias.data, {tensorBias.desc.dims[0], tensorBias.desc.dims[1]}, at::kInt);
    auto attensorDescale =
        at::from_blob(tensorDescale.data, {tensorDescale.desc.dims[0], tensorDescale.desc.dims[1]}, at::kFloat);
    if (opParam.transposeA) {
        attensorA = at::transpose(attensorA, 1, 2).contiguous();
    }
    if (opParam.transposeB) {
        attensorB = at::transpose(attensorB, 1, 2).contiguous();
        shape = {tensorA.desc.dims[0], tensorA.desc.dims[1], tensorB.desc.dims[1]};
    }
    auto attensorOut = at::from_blob(
        tensorOut.data, {tensorOut.desc.dims[0], tensorOut.desc.dims[1], tensorOut.desc.dims[2]}, at::kHalf);
    // 计算golden
    auto matmulRes = at::matmul(attensorA, attensorB).to(at::kInt);
    auto addRes = at::add(matmulRes, at::broadcast_to(attensorBias, shape)).to(at::kInt);
    auto attensorGt = at::mul(addRes, at::broadcast_to(attensorDescale, shape)).to(at::kHalf);
    return MatchAtTensorHalf(atol, rtol, attensorOut, attensorGt);
}

Kernel *GetKernelByName(const std::string &kernelName)
{
    Kernel *kernel = nullptr;
    auto &kernelCreators = KernelRegister::GetKernelCreators();
    for (const auto &iter : kernelCreators) {
        if (iter.kernelName == kernelName) {
            const std::string &opName = iter.opName;
            Mki::Operation *op = Mki::AutoGen::GetOpByName(opName);
            kernel = op->GetKernelByName(kernelName);
        }
    }
    return kernel;
}

} // namespace AsdOps

TEST(TestMatMulNd, MatMulSize1)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, false};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {300, 500}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {500, 700}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, MatMulSize2)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, false};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4096, 11008}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, MatMulSize3)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, true};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {11008, 4096}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, MatMulSize4)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, false};
    opTest.Golden(std::bind(BatchMatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 4096, 11008}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, MatMulSize5)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, true};
    opTest.Golden(std::bind(BatchMatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, MatMulSize6)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, false};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {16384, 512}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {512, 16384}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 4096, 11008}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief inPutNumT wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor0 format wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NHWC, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor0 dim size wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd4)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor0 dtype wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd5)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor1 format wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd6)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NHWC, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor1 dim size wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd7)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor1 dtype wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd8)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor dtype wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd9)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 1024, 5}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor format wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd10)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NDHWC, {7, 1024, 5}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor dim size wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd11)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief opDesc wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd12)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024, 5}}});
    OpParam::Gather opParam;
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief opParam.transpose wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd13)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {true, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief opParam.transpose wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd14)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
/**
 * @brief batchMMNdInTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16 wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
/**
 * @brief ok
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd15)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4096, 11008}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief opParam.transposeA wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd16)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4096, 11008}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {true, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief opParam.transposeB wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd17)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4096, 11008}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd18)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd19)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4096, 11008}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor0 format wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd20)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NHWC, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4096, 11008}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor0 dtype wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd21)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4096, 11008}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor1 format wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd22)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NHWC, {4096, 11008}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor1 dtype wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd23)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {4096, 11008}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor format wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd24)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4096, 11008}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NHWC, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor dtype wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd25)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4096, 11008}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief opDesc wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd26)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4096, 11008}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::Gather opParam;
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor0 dims size wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd27)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4096, 11008}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor1 dims size wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd28)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4096, 11008, 100}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor dims size wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNd29)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4096, 11008}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024, 15}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestMatMulNd, TestCanSupportMatMulNdF16Tb0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16TbKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief opParam.transpose wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNdF16Tb1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16TbKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief opParam.transpose wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNdF16Tb2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {true, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("BatchMatMulNdF16TbKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestMatMulNd, TestCanSupportMatMulNdF16Tb3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024, 5}}});
    OpParam::MatMul opParam = {false, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16TbKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief opParam.transposeA wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNdF16Tb4)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024, 5}}});
    OpParam::MatMul opParam = {true, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16TbKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief opParam.transposeB wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNdF16Tb5)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024, 5}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16TbKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor0 dims size wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNdF16Tb6)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024, 5}}});
    OpParam::MatMul opParam = {false, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16TbKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor1 dims size wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNdF16Tb7)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024, 5}}});
    OpParam::MatMul opParam = {false, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16TbKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor dims size wrong
 */
TEST(TestMatMulNd, TestCanSupportMatMulNdF16Tb8)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7}}});
    OpParam::MatMul opParam = {false, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("MatMulNdF16TbKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul0)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul1)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul2)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor0 format wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul3)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NHWC, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor0 dtype wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul4)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor0 dims size wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul5)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor1 format wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul6)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NHWC, {1024, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor1 dtype wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul7)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1024, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor1 dims size wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul8)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor format wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul9)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NHWC, {7, 1024}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor dtype wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul10)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor dims size wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul11)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
/**
 * @brief opDesc wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul12)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::Gather opParam = {};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
/**
 * @brief mmInfo.biasFlag = (launchParam.GetInTensorCount() > CONST_2)
 */
TEST(TestMatMulNd, TestCanSupportPpMatMulInitlaunchParamImpl0)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
    uint32_t launchBufferSize = kernel->GetTilingSize(launchParam);
    if (launchBufferSize == 0) {
        MKI_LOG(ERROR) << "empty tiling size";
    }
    KernelInfo &kernelInfo = const_cast<KernelInfo &>(kernel->GetKernelInfo());
    kernelInfo.AllocTilingHost(launchBufferSize);
    Status status = PpMatmulMixTiling(launchParam, kernelInfo);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief mmInfo.transB true
 */
TEST(TestMatMulNd, TestCanSupportPpMatMulInitlaunchParamImpl2)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 1024, 1}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, true, {0, 0, 0}, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
    uint32_t launchBufferSize = kernel->GetTilingSize(launchParam);
    if (launchBufferSize == 0) {
        MKI_LOG(ERROR) << "empty tiling size";
    }
    KernelInfo &kernelInfo = const_cast<KernelInfo &>(kernel->GetKernelInfo());
    kernelInfo.AllocTilingHost(launchBufferSize);

    Status status = PpMatmulMixTiling(launchParam, kernelInfo);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(TestMatMulNd, TestCanSupportPpMatMulInitlaunchParamImpl3)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 1024, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);

    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
    uint32_t launchBufferSize = kernel->GetTilingSize(launchParam);
    if (launchBufferSize == 0) {
        MKI_LOG(ERROR) << "empty tiling size";
    }
    KernelInfo &kernelInfo = const_cast<KernelInfo &>(kernel->GetKernelInfo());
    kernelInfo.AllocTilingHost(launchBufferSize);

    status = PpMatmulMixTiling(launchParam, kernelInfo);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief CanSupportPpMatMul
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 1024}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 1024}}});
    OpParam::Gather opParam = {};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16MixKernel"));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief ok
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul15)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 8}}});
    OpParam::MatMul opParam = {false, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul16)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 8}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul17)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 8}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 8}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor0 format wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul18)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NHWC, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 8}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor0 dtype wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul19)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 8}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor0 dims size wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul20)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 8}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor1 format wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul21)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NHWC, {11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 8}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor1 dtype wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul22)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 8}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor1 dims size wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul23)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {11008}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 8}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor format wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul24)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NHWC, {1024, 8}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor dtype wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul25)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1024, 8}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor dims size wrong
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul26)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestMatMulNd, TestCanSupportPpMatMul29)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulI8Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}
/**
 * @brief ok
 */
TEST(TestMatMulNd, PpMatMulI8KernelInitHostLaunchBuffer)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulI8Kernel"));
    ASSERT_NE(kernel, nullptr);
    kernel->Init(launchParam);
    //(runInfo.GetInTensorCount() > 2) wrong
    kernel->Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    kernel->Init(launchParam);
    // formatA == TENSOR_FORMAT_ND && formatB == TENSOR_FORMAT_FRACTAL_NZ wrong
    kernel->Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    kernel->Init(launchParam);
    // inputADim.size() == 2
    kernel->Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    kernel->Init(launchParam);
    // inputADim.size() == 2
    kernel->Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    kernel->Init(launchParam);
}

/**
 * @brief ok
 */
TEST(TestMatMulNz, TestCanSupportPpMatMulF16OptKernel)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {6, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {6, 1, 16, 16}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {6, 7, 10}}});
    OpParam::MatMul opParam = {false, false};
    opParam.oriShape = {7, 7, 10};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName("MatMulOperation");
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
    // launchParam.GetInTensorCount() == 2 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {6, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // descA.format == TENSOR_FORMAT_ND wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // descA.dtype == TENSOR_DTYPE_FLOAT16 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // descB.format == TENSOR_FORMAT_FRACTAL_NZ wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // descB.dims.size() == 4 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // descB.dtype == TENSOR_DTYPE_FLOAT16 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // descA.dims.size() == 2  true
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {1, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
    // descA.dims.size() != 2  descA.dims.size() == 3 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {1, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // descA.dims[0] == descB.dims[0]  wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {6, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestMatMulNz, TestCanSupportPpMatMulNzF16Kernel)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulNzF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
    // GetInTensorCount() == 2 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // GetOutTensorCount() == 1 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor0.desc.format == TENSOR_FORMAT_FRACTAL_NZ wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NDHWC, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor0.desc.dims.size() == 3||inTensor0.desc.dims.size() == 4 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor1.desc.format == TENSOR_FORMAT_FRACTAL_NZ wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NDHWC, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor1.desc.dims.size() == 3 || inTensor0.desc.dims.size() == 4 ok
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
    // outTensor.desc.format == TENSOR_FORMAT_FRACTAL_NZ wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NDHWC, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // outTensor.desc.dims.size() == 3 || inTensor0.desc.dims.size() == 4 true
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
    // outTensor.desc.dims.size() == 3 || inTensor0.desc.dims.size() == 4 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // outTensor.desc.dims.size() == 3 || inTensor0.desc.dims.size() == 4 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief launchParam.GetOpDesc().specificParam.Type() == typeid(OpParam::MatMul) wrong
 */
TEST(TestMatMulNz, TestCanSupportPpMatMulNzF16Kernel15)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_MULS};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulNzF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestMatMulNz, TestCanSupportPpMatMulI8NzKernel)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    OpParam::MatMul opParam = {false, false, {0, 0, 0}, true, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulI8NzKernel"));
    ASSERT_EQ(status.Ok(), true);
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
    // launchParam.GetParam().Type() == typeid(OpParam::MatMul) wrong
    OpParam::Activation opParam0 = {};
    opParam0.activationType = OpParam::Activation::ACTIVATION_FAST_GELU;
    Mki::Test::UtOpDesc opDesc0 = {"ActivationOperation", opParam0};
    launchParam.SetParam(opDesc0.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // launchParam.GetInTensorCount() == 4 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // launchParam.GetInTensor(0).desc.format == TENSOR_FORMAT_FRACTAL_NZ wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_NDHWC, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT8 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT16, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // launchParam.GetInTensor(0).desc.dims.size() == 4 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestMatMulNz, TestInitRunInfoImplPpMatMulI8NzKernel1)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    OpParam::MatMul opParam = {false, false, {1, 1, 1}, true, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulI8NzKernel"));
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);

    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
    uint32_t launchBufferSize = kernel->GetTilingSize(launchParam);
    if (launchBufferSize == 0) {
        MKI_LOG(ERROR) << "empty tiling size";
    }
    KernelInfo &kernelInfo = const_cast<KernelInfo &>(kernel->GetKernelInfo());
    kernelInfo.AllocTilingHost(launchBufferSize);
    status = PpTiling310P(launchParam, kernelInfo);
    ASSERT_EQ(status.Ok(), true);
    kernelInfo.Reset();
    OpParam::MatMul opParam0 = {false, false, {1, 1, 1, 1}, true, true};
    Mki::Test::UtOpDesc opDesc0 = {"MatMulOperation", opParam0};
    launchParam.SetParam(opDesc0.specificParam);
    kernelInfo.AllocTilingHost(launchBufferSize);
    status = PpTiling310P(launchParam, kernelInfo);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief opParam.oriShape[i] > 0 && opParam.oriShape[i] <= UINT32_MAX wrong
 */
TEST(TestMatMulNz, TestInitlaunchParamImplPpMatMulI8NzKernel2)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {7, 7, 7, 7}}});
    OpParam::MatMul opParam = {false, false, {1, 1, 4294}, true, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulI8NzKernel"));
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    ASSERT_NE(kernel, nullptr);
    uint32_t launchBufferSize = kernel->GetTilingSize(launchParam);
    if (launchBufferSize == 0) {
        MKI_LOG(ERROR) << "empty tiling size";
    }
    KernelInfo &kernelInfo = const_cast<KernelInfo &>(kernel->GetKernelInfo());
    kernelInfo.AllocTilingHost(launchBufferSize);
    status = PpTiling310P(launchParam, kernelInfo);
    ASSERT_EQ(status.Ok(), true);
    // opParam.oriShape[i] > 0 && opParam.oriShape[i] <= UINT32_MAX
    OpParam::MatMul opParam0 = {false, false, {0, 0, 0}, true, true};
    Mki::Test::UtOpDesc opDesc0 = {"MatMulOperation", opParam0};
    launchParam.SetParam(opDesc0.specificParam);
    kernelInfo.AllocTilingHost(launchBufferSize);
    status = PpTiling310P(launchParam, kernelInfo);
    ASSERT_EQ(status.Ok(), false);
    // attrs.tilingK > 0 && attrs.tilingN > 0
    OpParam::MatMul opParam1 = {false, false, {1, 1, 1}, true, true, 1, 0};
    Mki::Test::UtOpDesc opDesc1 = {"MatMulOperation", opParam1};
    launchParam.SetParam(opDesc1.specificParam);
    kernelInfo.AllocTilingHost(launchBufferSize);
    status = PpTiling310P(launchParam, kernelInfo);
    ASSERT_EQ(status.Ok(), true);
    // attrs.tilingK > 0 && attrs.tilingN > 0
    OpParam::MatMul opParam2 = {false, false, {1, 1, 1}, true, true, 0, 1};
    Mki::Test::UtOpDesc opDesc2 = {"MatMulOperation", opParam2};
    launchParam.SetParam(opDesc2.specificParam);
    kernelInfo.AllocTilingHost(launchBufferSize);
    status = PpTiling310P(launchParam, kernelInfo);
    ASSERT_EQ(status.Ok(), true);
    // attrs.tilingK > 0 && attrs.tilingN > 0
    OpParam::MatMul opParam3 = {false, false, {1, 1, 1}, true, true, 0, 0};
    Mki::Test::UtOpDesc opDesc3 = {"MatMulOperation", opParam3};
    launchParam.SetParam(opDesc3.specificParam);
    kernelInfo.AllocTilingHost(launchBufferSize);
    status = PpTiling310P(launchParam, kernelInfo);
    ASSERT_EQ(status.Ok(), true);
    OpParam::MatMul opParam4 = {false, false, {1, 1, 52}, true, true};
    Mki::Test::UtOpDesc opDesc4 = {"MatMulOperation", opParam4};
    launchParam.SetParam(opDesc4.specificParam);
    kernelInfo.AllocTilingHost(launchBufferSize);
    status = PpTiling310P(launchParam, kernelInfo);
    ASSERT_EQ(status.Ok(), true);
    OpParam::MatMul opParam5 = {false, false, {52, 1, 1}, true, true};
    Mki::Test::UtOpDesc opDesc5 = {"MatMulOperation", opParam5};
    launchParam.SetParam(opDesc5.specificParam);
    kernelInfo.AllocTilingHost(launchBufferSize);
    status = PpTiling310P(launchParam, kernelInfo);
    ASSERT_EQ(status.Ok(), true);
}
/**
 * @brief ok
 */
TEST(TestMatMulNd, TestInferShapeMatMulNd0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 8}}});
    OpParam::MatMul opParam = {false, true};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief inTensor format wrong
 */
TEST(TestMatMulNd, TestInferShapeMatMulNd1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_HWCN, {7, 4096}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {11008, 4096}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1024, 8}}});
    OpParam::MatMul opParam = {false, false};
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestMatMulNd, PPMatMulSize1)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, false};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {300, 500}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {500, 700}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, PPMatMulSize2)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, false};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4096, 11008}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, PPMatMulSize3)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, true};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4096}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {11008, 4096}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, PPMatMulSize4)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, false};
    opTest.Golden(std::bind(BatchMatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 4096, 11008}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, PPMatMulSize5)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, true};
    opTest.Golden(std::bind(BatchMatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 7, 4096}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 11008, 4096}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, MatMulGemvCase1)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, true};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 4096}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {12288, 4096}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, MatMulGemvCase2)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, true};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 4096}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4096, 4096}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, MatMulGemvCase3)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, true};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 4096}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {16384, 4096}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, MatMulGemvCase4)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, true};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 16384}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4096, 16384}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief  ok
 */
TEST(TestMatMulNz, TestCanSupportPpMatMulI8NzCompressKernel)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    OpParam::MatMul opParam = {false, false, {1, 1, 1}};
    opParam.tilingN = 1;
    opParam.tilingK = 1;
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulI8NzCompressKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // launchParam.GetOpDesc().specificParam.Type() == typeid(OpParam::MatMul)
    OpParam::Elewise opParam0 = {};
    opParam0.elewiseType = OpParam::Elewise::ELEWISE_ADD;
    Mki::Test::UtOpDesc opDesc0 = {"ElewiseOperation", opParam0};
    launchParam.SetParam(opDesc0.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
/**
 * @brief  launchParam.GetInTensorCount() == DIM_5 wrong
 */
TEST(TestMatMulNz, TestCanSupportPpMatMulI8NzCompressKernel1)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    OpParam::MatMul opParam = {false, false, {1, 1, 1}};
    opParam.tilingN = 1;
    opParam.tilingK = 1;
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulI8NzCompressKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // launchParam.GetOutTensorCount() == DIM_1 wrong
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor0.desc.dtype == TENSOR_DTYPE_INT8 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor0.desc.format == TENSOR_FORMAT_FRACTAL_NZ wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor0.desc.dims.size() == DIM_4 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor1.desc.dtype == TENSOR_DTYPE_INT8 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor1.desc.format == TENSOR_FORMAT_FRACTAL_NZ wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor1.desc.dims.size() == DIM_4 || inTensor1.desc.dims.size() == DIM_2 wrong
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor1.desc.dims.size() == DIM_4 || inTensor1.desc.dims.size() == DIM_2 true
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor1.desc.dims.size() == DIM_4 || inTensor1.desc.dims.size() == DIM_2 false
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor2.desc.dtype == TENSOR_DTYPE_INT32 false
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor2.desc.format == TENSOR_FORMAT_ND false
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor2.desc.dims.size() == DIM_2 false
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor3.desc.dtype == TENSOR_DTYPE_UINT64 false
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor3.desc.format == TENSOR_FORMAT_ND false
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor3.desc.dims.size() == DIM_2 false
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor4.desc.dtype == TENSOR_DTYPE_INT8 false
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor4.desc.format == TENSOR_FORMAT_ND false
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // inTensor4.desc.dims.size() == DIM_2 false
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16 false
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // outTensor.desc.format == TENSOR_FORMAT_FRACTAL_NZ false
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
    // outTensor.desc.dims.size() == 4 false
    launchParam.Reset();
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5, 5}}});
    launchParam.SetParam(opDesc.specificParam);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
/**
 * @brief  InitlaunchParamImpl ok
 */
TEST(TestMatMulNz, TestInitlaunchParamImplPpMatMulI8NzCompressKernel22)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_FRACTAL_NZ, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {5, 5, 5, 5}}});
    OpParam::MatMul opParam = {false, false, {1, 1, 1}};
    opParam.tilingN = 1;
    opParam.tilingK = 1;

    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    auto kernel = std::unique_ptr<Mki::Kernel>(GetKernelByName("PpMatMulI8NzCompressKernel"));
    ASSERT_NE(kernel, nullptr);
    uint32_t launchBufferSize = kernel->GetTilingSize(launchParam);
    if (launchBufferSize == 0) {
        MKI_LOG(ERROR) << "empty tiling size";
    }
    KernelInfo &kernelInfo = const_cast<KernelInfo &>(kernel->GetKernelInfo());
    kernelInfo.AllocTilingHost(launchBufferSize);
    Status status = PpTiling310P(launchParam, kernelInfo);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, BdCase0)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, true};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {20, 8192}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {12544, 8192}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, BdCase1)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, true};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {248, 8192}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3072, 8192}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, BdCase2)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, true};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {248, 1024}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {8192, 1024}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, BdCase3)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, true};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {248, 8192}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5504, 8192}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, BdCase4)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, true};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {248, 2752}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {8192, 2752}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestMatMulNd, BdCase5)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    OpParam::MatMul opParam = {false, true};
    opTest.Golden(std::bind(MatMulNdGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    Mki::Test::UtOpDesc opDesc = {"MatMulOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {248, 8192}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {12544, 8192}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}