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
#include <atb/utils/log.h>
#include "test_utils/test_common.h"
#include "atb/operation.h"
#include "atb/infer_op_params.h"
#include "atb/utils/tensor_util.h"
#include "test_utils/operation_test.h"
#include "atb/utils/operation_util.h"
#include "mki/utils/fp16/fp16_t.h"
#include "atb/utils/singleton.h"
#include "atb/utils/config.h"

using namespace atb;
using namespace Mki;
constexpr size_t IDX_0 = 0;
constexpr size_t IDX_1 = 1;
constexpr size_t IDX_2 = 2;
constexpr size_t IDX_3 = 3;
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;

TEST(TestTransdataOperation, InferShapeNdToNz3dInt8)
{
    if (GetSingleton<Config>().Is310B()) {
        ATB_LOG(INFO) << "Platform Atlas 200I/500 A2 inference product only support fp16 as input dtype!";
        return;
    }
    atb::infer::TransdataParam param;
    param.transdataType = atb::infer::TransdataParam::TransdataType::ND_TO_FRACTAL_NZ;
    param.outCrops = { 0, 0 };
    atb::Operation *op = nullptr;
    atb::Status st = atb::CreateOperation<atb::infer::TransdataParam>(param, &op);
    ASSERT_EQ(st, atb::NO_ERROR);

    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        { Mki::TENSOR_DTYPE_INT8, Mki::TENSOR_FORMAT_ND, { 1, 23, 257 } }
    };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    st = op->InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(st, atb::NO_ERROR);
    ASSERT_EQ(outTensorDescs.size(), static_cast<std::size_t>(1));
    EXPECT_EQ(outTensorDescs.at(0).dtype, ACL_INT8);
    EXPECT_EQ(outTensorDescs.at(0).format, ACL_FORMAT_FRACTAL_NZ);
    Mki::SVector<int64_t> expectDims = { 1, 9, 32, 32 };
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).shape.dimNum);
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).shape.dims[0]);
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).shape.dims[1]);
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(0).shape.dims[2]);
    EXPECT_EQ(expectDims.at(3), outTensorDescs.at(0).shape.dims[3]);
    DestroyOperation(op);
}

/// @brief golden NdToNz2dInt8
/// @param context
/// @return
Mki::Status TransdataGoldenNdToNz2dInt8(const OpsGoldenContext &context)
{
    // get constructed input/output tensors
    constexpr int64_t ALIGN_INT8 = 32;
    constexpr int64_t DEFAULT_ALIGN = 16;
    const Mki::Tensor inTensor = context.hostInTensors.at(0);
    const Mki::Tensor outTensor = context.hostOutTensors.at(0);
    std::vector<int64_t> auxDims{ 0, 0, 0, 0 };
    auxDims[IDX_0] = 1;
    auxDims[IDX_1] = OperationUtil::RoundUp(inTensor.desc.dims[IDX_0], DEFAULT_ALIGN);
    auxDims[IDX_2] = OperationUtil::RoundUp(inTensor.desc.dims[IDX_1], ALIGN_INT8) / ALIGN_INT8;
    auxDims[IDX_3] = ALIGN_INT8;
    std::vector<int64_t> padDims{ 0, 0, 0, 0 };
    padDims[IDX_1] = OperationUtil::RoundUp(inTensor.desc.dims[IDX_1], ALIGN_INT8) - inTensor.desc.dims[IDX_1];
    padDims[IDX_3] = OperationUtil::RoundUp(inTensor.desc.dims[IDX_0], DEFAULT_ALIGN) - inTensor.desc.dims[IDX_0];
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kChar).to(at::kInt);
    // construct ref input tensors
    at::Tensor atInRefTensor =
        at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kChar).to(at::kInt);
    // get ref output tensor
    at::Tensor atOutRefTensor = at::transpose(
        at::reshape(at::constant_pad_nd(atInRefTensor, ToIntArrayRef(padDims)), ToIntArrayRef(auxDims)), IDX_0, IDX_1)
                                    .to(at::kInt)
                                    .contiguous();
    // compare
    int8_t *atOutArray = static_cast<int8_t *>(atOutTensor.storage().data_ptr().get());
    int8_t *atRefOutArray = static_cast<int8_t *>(atOutRefTensor.storage().data_ptr().get()); // golden
    for (int i = 0; i < outTensor.Numel(); i++) {
        float expect = atRefOutArray[i];
        float actual = atOutArray[i];
        bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
        EXPECT_EQ(judge, true);
        if (!judge) {
            return Mki::Status::FailStatus(1, "unequal");
        }
    }
    return Mki::Status::OkStatus();
}

/// @brief golden NdToNz3dInt8
/// @param context
/// @return
Mki::Status TransdataGoldenNdToNz3dInt8(const OpsGoldenContext &context)
{
    // get constructed input/output tensors
    constexpr int64_t ALIGN_INT8 = 32;
    constexpr int64_t DEFAULT_ALIGN = 16;
    const Mki::Tensor inTensor = context.hostInTensors.at(0);
    const Mki::Tensor outTensor = context.hostOutTensors.at(0);
    std::vector<int64_t> auxDims{ 0, 0, 0, 0 };
    auxDims[IDX_0] = inTensor.desc.dims[IDX_0];
    auxDims[IDX_1] = OperationUtil::RoundUp(inTensor.desc.dims[IDX_1], DEFAULT_ALIGN);
    auxDims[IDX_2] = OperationUtil::RoundUp(inTensor.desc.dims[IDX_2], ALIGN_INT8) / ALIGN_INT8;
    auxDims[IDX_3] = ALIGN_INT8;
    std::vector<int64_t> padDims{ 0, 0, 0, 0 };
    padDims[IDX_1] = OperationUtil::RoundUp(inTensor.desc.dims[IDX_2], ALIGN_INT8) - inTensor.desc.dims[IDX_2];
    padDims[IDX_3] = OperationUtil::RoundUp(inTensor.desc.dims[IDX_1], DEFAULT_ALIGN) - inTensor.desc.dims[IDX_1];
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kChar).to(at::kInt);
    // construct ref input tensors
    at::Tensor atInRefTensor =
        at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kChar).to(at::kInt);
    // get ref output tensor
    at::Tensor atOutRefTensor = at::transpose(
        at::reshape(at::constant_pad_nd(atInRefTensor, ToIntArrayRef(padDims)), ToIntArrayRef(auxDims)), IDX_1, IDX_2)
                                    .to(at::kInt)
                                    .contiguous();
    // compare
    int8_t *atOutArray = static_cast<int8_t *>(atOutTensor.storage().data_ptr().get());
    int8_t *atRefOutArray = static_cast<int8_t *>(atOutRefTensor.storage().data_ptr().get()); // golden
    for (int i = 0; i < outTensor.Numel(); i++) {
        float expect = atRefOutArray[i];
        float actual = atOutArray[i];
        bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
        EXPECT_EQ(judge, true);
        if (!judge) {
            return Mki::Status::FailStatus(1, "unequal");
        }
    }
    return Mki::Status::OkStatus();
}

/// @brief test NdToNz2dInt8
/// @param
/// @param
TEST(TestTransdataOperation, TestTransdataNdToNz2dInt8)
{
    if (GetSingleton<Config>().Is310B()) {
        ATB_LOG(INFO) << "Platform Atlas 200I/500 A2 inference product only support fp16 as input dtype!";
        return;
    }
    atb::infer::TransdataParam param;
    param.transdataType = atb::infer::TransdataParam::TransdataType::ND_TO_FRACTAL_NZ;
    param.outCrops = { 0, 0 };
    atb::Operation *op = nullptr;
    atb::Status st = atb::CreateOperation<atb::infer::TransdataParam>(param, &op);
    ASSERT_EQ(st, atb::NO_ERROR);

    Mki::SVector<Mki::TensorDesc> inTensorDescs = {
        { Mki::TENSOR_DTYPE_INT8, Mki::TENSOR_FORMAT_ND, { 6144, 4096 } }
    };
    OperationTest opTest;
    opTest.FloatRand(1, 10);
    opTest.Golden(&TransdataGoldenNdToNz2dInt8);
    Mki::Status status = opTest.Run(op, inTensorDescs);
    DestroyOperation(op);
    ASSERT_EQ(status.Ok(), true);
}

/// @brief test NdToNz3dInt8
/// @param
/// @param
TEST(TestTransdataOperation, TestTransdataNdToNz3dInt8)
{
    if (GetSingleton<Config>().Is310B()) {
        ATB_LOG(INFO) << "Platform Atlas 200I/500 A2 inference product only support fp16 as input dtype!";
        return;
    }
    atb::infer::TransdataParam param;
    param.transdataType = atb::infer::TransdataParam::TransdataType::ND_TO_FRACTAL_NZ;
    param.outCrops = { 0, 0 };
    atb::Operation *op = nullptr;
    atb::Status st = atb::CreateOperation<atb::infer::TransdataParam>(param, &op);
    ASSERT_EQ(st, atb::NO_ERROR);

    Mki::SVector<Mki::TensorDesc> inTensorDescs = {
        { Mki::TENSOR_DTYPE_INT8, Mki::TENSOR_FORMAT_ND, { 1, 8, 32 } }
    };
    OperationTest opTest;
    opTest.Int8Rand(1, 10);
    opTest.Golden(&TransdataGoldenNdToNz3dInt8);
    Mki::Status status = opTest.Run(op, inTensorDescs);
    DestroyOperation(op);
    ASSERT_EQ(status.Ok(), true);
}
