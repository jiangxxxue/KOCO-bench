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
#include <torch/torch.h>
#include <gtest/gtest.h>
#include <future>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include <mki/utils/rt/rt.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/params.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "kernels/sort/tiling/sort_tiling.h"
#include "device_version_check.h"
#include "test_common.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;

static constexpr float HALF_FLOAT_MIN = -65504;
static constexpr float HALF_FLOAT_MAX = 65504;
constexpr float EXTENT_OF_ERROR = 0.001;

Status TopKV2DescGolden(const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor0 = context.hostInTensors.at(0);
    auto param = AnyCast<OpParam::Sort>(context.opDesc.specificParam);
    const Tensor outTensor0 = context.hostOutTensors.at(0);
    const Tensor outTensor1 = context.hostOutTensors.at(1);

    at::Tensor atInRefTensor0 = at::from_blob(inTensor0.data, ToIntArrayRef(inTensor0.desc.dims), at::kHalf);
    at::Tensor atOutTensor0 = at::from_blob(outTensor0.data, ToIntArrayRef(outTensor0.desc.dims), at::kHalf);
    at::Tensor atOutTensor1 = at::from_blob(outTensor1.data, ToIntArrayRef(outTensor1.desc.dims), at::kInt);

    at::Tensor atInRefTensor0_f32 = atInRefTensor0.toType(at::kFloat);
    at::Tensor atOutRefTensor0_f32, atOutRefTensor1_f32;
    std::tie(atOutRefTensor0_f32, atOutRefTensor1_f32) = at::topk(atInRefTensor0_f32, param.num[0]);

    at::Tensor atOutRefTensor0 = atOutRefTensor0_f32.type_as(atOutTensor0);
    if (!at::allclose(atOutRefTensor0, atOutTensor0, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        return Status::FailStatus(-1, "judge not equal");
    }

    // 由于golden在value相同时，idx顺序不定，outTensor1不能与golden严格比较。上面已保证outTensor0正确，保证outTensor0和outTensor1对应关系正确即可。
    const SVector<int64_t> &dims = inTensor0.desc.dims;
    fp16_t *inArray = (fp16_t *)atInRefTensor0.storage().data_ptr().get();
    fp16_t *OutArray0 = (fp16_t *)atOutTensor0.storage().data_ptr().get();
    int32_t *OutArray1 = (int32_t *)atOutTensor1.storage().data_ptr().get();
    for (int i = 0; i < outTensor1.Numel(); i++) {
        if (OutArray0[i] != inArray[i / param.num[0] * dims[dims.size() - 1] + OutArray1[i]]) {
            return Status::FailStatus(-1, "judge not equal");
        }
    }

    return Status::OkStatus();
}

TEST(TestOpSort, TopKV2DescCase0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&TopKV2DescGolden);

    OpParam::Sort opParam;
    opParam.num.push_back(1);

    Mki::Test::UtOpDesc opDesc = {"SortOperation", opParam};
    int64_t m = 7;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m}}};

    Status status = opTest.Run(opDesc, inTensorDesc);

    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSort, TopKV2DescCase1)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&TopKV2DescGolden);

    OpParam::Sort opParam;
    opParam.num.push_back(2);

    Mki::Test::UtOpDesc opDesc = {"SortOperation", opParam};
    int64_t m = 8;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m}}};

    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSort, TopKV2DescCase2)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&TopKV2DescGolden);

    OpParam::Sort opParam;
    opParam.num.push_back(1);

    Mki::Test::UtOpDesc opDesc = {"SortOperation", opParam};
    int64_t m = 8, n = 6;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}}};

    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSort, TopKV2DescCase3)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&TopKV2DescGolden);

    OpParam::Sort opParam;
    opParam.num.push_back(2);

    Mki::Test::UtOpDesc opDesc = {"SortOperation", opParam};
    int64_t m = 8, n = 6;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}}};

    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSort, TopKV2DescCase4)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&TopKV2DescGolden);

    OpParam::Sort opParam;
    opParam.num.push_back(1);

    Mki::Test::UtOpDesc opDesc = {"SortOperation", opParam};
    int64_t m = 8, n = 6, k = 20, q = 8;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}};

    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(TestOpSort, TestGetBestKernelTopK0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    int64_t m = 8, n = 6, k = 20, q = 8;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    OpParam::Sort opParam;
    opParam.num.push_back(1);
    Mki::Test::UtOpDesc opDesc = {"SortOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpSort, TestGetBestKernelTopK1)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    int64_t m = 8, n = 6, k = 20, q = 8;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_BOOL, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_BOOL, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_BOOL, TENSOR_FORMAT_ND, {m, n, k, q}}});
    OpParam::Sort opParam;
    opParam.num.push_back(1);
    Mki::Test::UtOpDesc opDesc = {"SortOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}

/**
 * @brief ok
 */
TEST(TestOpSort, TestInferShapeTopK0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    int64_t m = 8, n = 6, k = 20, q = 8;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    OpParam::Sort opParam;
    opParam.num.push_back(1);
    Mki::Test::UtOpDesc opDesc = {"SortOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief InTensor0.dims.size = 0
 */
TEST(TestOpSort, TestInferShapeTopK1)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    int64_t m = 8, n = 6, k = 20, q = 8;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    OpParam::Sort opParam;
    opParam.num.push_back(1);
    Mki::Test::UtOpDesc opDesc = {"SortOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief num.size = 0
 */
TEST(TestOpSort, TestInferShapeTopK2)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    int64_t m = 8, n = 6, k = 20, q = 8;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    OpParam::Sort opParam;
    Mki::Test::UtOpDesc opDesc = {"SortOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief TopK input tensor last dim at least k
 */
TEST(TestOpSort, TestInferShapeTopK3)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    int64_t m = 8, n = 6, k = 20, q = 1;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    OpParam::Sort opParam;
    opParam.num.push_back(2);
    Mki::Test::UtOpDesc opDesc = {"SortOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief ok
 */
TEST(TestOpSort, TestCanSupportTopK0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    int64_t m = 8, n = 6, k = 20, q = 8;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    OpParam::Sort opParam;
    opParam.num.push_back(1);
    Mki::Test::UtOpDesc opDesc = {"SortOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("TopKDescF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestOpSort, TestCanSupportTopK1)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    int64_t m = 8, n = 6, k = 20, q = 8;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    OpParam::Sort opParam;
    opParam.num.push_back(1);
    Mki::Test::UtOpDesc opDesc = {"SortOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("TopKDescF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestOpSort, TestCanSupportTopK2)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    int64_t m = 8, n = 6, k = 20, q = 8;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, q}}});
    OpParam::Sort opParam;
    opParam.num.push_back(1);
    Mki::Test::UtOpDesc opDesc = {"SortOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("TopKDescF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief multi-thread auto tiling test
 */
TEST(TestOpSort, TestSortCaseAutoTiling)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&TopKV2DescGolden);

    Mki::Test::MkiOpTest opTest1;
    opTest1.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest1.Golden(&TopKV2DescGolden);

    Mki::Test::MkiOpTest opTest2;
    opTest2.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest2.Golden(&TopKV2DescGolden);

    OpParam::Sort opParam;
    opParam.num.push_back(1);

    Mki::Test::UtOpDesc opDesc = {"SortOperation", opParam};
    Mki::Test::UtOpDesc opDesc1 = {"SortOperation", opParam};
    Mki::Test::UtOpDesc opDesc2 = {"SortOperation", opParam};
    int64_t m = 7;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m}}};
    SVector<TensorDesc> inTensorDesc1 = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m}}};
    SVector<TensorDesc> inTensorDesc2 = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m}}};

    auto test1 = [&]() {
        return opTest.Run(opDesc, inTensorDesc);
    };

    auto test2 = [&]() {
        return opTest1.Run(opDesc1, inTensorDesc1);
    };

    auto test3 = [&]() {
        return opTest2.Run(opDesc2, inTensorDesc2);
    };

    for (uint32_t i = 0; i < 100; i++) {
        std::future<Status> result1 = std::async(test1);
        std::future<Status> result2 = std::async(test2);
        std::future<Status> result3 = std::async(test3);
        Status status1 = result1.get();
        Status status2 = result2.get();
        Status status3 = result3.get();
        ASSERT_EQ(status1.Ok() && status2.Ok() && status3.Ok() , true);
    }
}
