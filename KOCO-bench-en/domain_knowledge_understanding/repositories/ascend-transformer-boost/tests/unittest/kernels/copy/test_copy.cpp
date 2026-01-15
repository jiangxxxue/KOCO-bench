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
#include "asdops/params/params.h"
#include <mki/utils/compare/compare.h>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;

template <typename T>
static Status GoldenAndCompare(OpParam::Copy &param, const Tensor &inTensor2, const Tensor outTensor,
    const at::Tensor &atOutTensor, const at::Tensor atInRefTensor2, at::Tensor atOutRefTensor)
{
    T *srcArray = (T *)atInRefTensor2.storage().data_ptr().get();
    T *dstRefArray = (T *)atOutRefTensor.storage().data_ptr().get();
    const uint32_t len = param.dstSize.size();
    for (uint32_t i = 0; i < inTensor2.Numel(); i++) {
        uint32_t tmp = i;
        uint32_t pos = 0;
        uint32_t val[len];
        for (uint32_t j = 0; j < len; j++) {
            val[j] = tmp % param.dstSize.at(len - j - 1);
            pos += val[j] * param.dstStride.at(len - j - 1);
            tmp /= param.dstSize.at(len - j - 1);
        }

        dstRefArray[pos] = srcArray[i];
    }

    T *dstArray = (T *)atOutTensor.storage().data_ptr().get();
    for (int i = 0; i < outTensor.Numel(); i++) {
        if (!Utils::Compare<T>::IsEqual(dstArray[i], dstRefArray[i])) {
            return Status::FailStatus(-1, "view judge not equal");
        }
    }

    return Status::OkStatus();
}

Status ViewCopyGolden(const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor1 = context.hostInTensors.at(0);
    const Tensor &inTensor2 = context.hostInTensors.at(1);
    const Tensor &outTensor = context.hostOutTensors.at(0);

    const Mki::Test::UtOpDesc &opDesc = context.opDesc;
    OpParam::Copy param = AnyCast<OpParam::Copy>(opDesc.specificParam);

    if (inTensor1.desc.dtype == TENSOR_DTYPE_INT64) {
        at::Tensor atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kLong);
        at::Tensor atOutRefTensor = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kLong);
        at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kLong);

        return GoldenAndCompare<int64_t>(param, inTensor2, outTensor, atOutTensor, atInRefTensor2, atOutRefTensor);
    } else if (inTensor1.desc.dtype == TENSOR_DTYPE_INT32) {
        at::Tensor atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kInt);
        at::Tensor atOutRefTensor = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kInt);
        at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kInt);

        return GoldenAndCompare<int32_t>(param, inTensor2, outTensor, atOutTensor, atInRefTensor2, atOutRefTensor);
    } else if (inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT) {
        at::Tensor atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kFloat);
        at::Tensor atOutRefTensor = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kFloat);
        at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);

        return GoldenAndCompare<float>(param, inTensor2, outTensor, atOutTensor, atInRefTensor2, atOutRefTensor);
    } else if (inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16) {
        at::Tensor atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kHalf);
        at::Tensor atOutRefTensor = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kHalf);
        at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);

        return GoldenAndCompare<fp16_t>(param, inTensor2, outTensor, atOutTensor, atInRefTensor2, atOutRefTensor);
    }
    return Status::OkStatus();
}

TEST(TestOpCopy, ViewCopyInt64Case0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(0, 20);
    opTest.SetOutdataUseInputData(0, 0);
    opTest.Golden(&ViewCopyGolden);
    OpParam::Copy opParam;
    opParam.dstSize = {2, 3};
    opParam.dstStride = {4, 1};
    opParam.dstOffset = {0};
    Mki::Test::UtOpDesc opDesc = {"CopyOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {2, 4}},
                                        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {2, 3}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpCopy, ViewCopyInt64Case1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(0, 20);
    opTest.SetOutdataUseInputData(0, 0);
    opTest.Golden(&ViewCopyGolden);
    OpParam::Copy opParam;
    opParam.dstSize = {2, 2, 2};
    opParam.dstStride = {12, 6, 1};
    opParam.dstOffset = {0};
    Mki::Test::UtOpDesc opDesc = {"CopyOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {3, 4, 2, 3}},
                                        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {2, 2, 2}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpCopy, ViewCopyInt64Case2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(0, 20);
    opTest.SetOutdataUseInputData(0, 0);
    opTest.Golden(&ViewCopyGolden);
    OpParam::Copy opParam;
    opParam.dstSize = {28, 2, 4096};
    opParam.dstStride = {10 * 4096, 4096, 1};
    opParam.dstOffset = {0};
    Mki::Test::UtOpDesc opDesc = {"CopyOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {28, 10, 4096}},
                                        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {28, 2, 4096}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpCopy, ViewCopyInt32Case0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.IntRand(0, 20);
    opTest.SetOutdataUseInputData(0, 0);
    opTest.Golden(&ViewCopyGolden);
    OpParam::Copy opParam;
    opParam.dstSize = {2, 3};
    opParam.dstStride = {4, 1};
    opParam.dstOffset = {0};
    Mki::Test::UtOpDesc opDesc = {"CopyOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {2, 4}},
                                        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {2, 3}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpCopy, ViewCopyInt32Case1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.IntRand(0, 20);
    opTest.SetOutdataUseInputData(0, 0);
    opTest.Golden(&ViewCopyGolden);
    OpParam::Copy opParam;
    opParam.dstSize = {2, 2, 2};
    opParam.dstStride = {12, 6, 1};
    opParam.dstOffset = {0};
    Mki::Test::UtOpDesc opDesc = {"CopyOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {3, 4, 2, 3}},
                                        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {2, 2, 2}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpCopy, ViewCopyInt32Case2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.IntRand(0, 20);
    opTest.SetOutdataUseInputData(0, 0);
    opTest.Golden(&ViewCopyGolden);
    OpParam::Copy opParam;
    opParam.dstSize = {28, 2, 4096};
    opParam.dstStride = {10 * 4096, 4096, 1};
    opParam.dstOffset = {0};
    Mki::Test::UtOpDesc opDesc = {"CopyOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {28, 10, 4096}},
                                        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {28, 2, 4096}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpCopy, ViewCopyF32Case0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(0, 20);
    opTest.SetOutdataUseInputData(0, 0);
    opTest.Golden(&ViewCopyGolden);
    OpParam::Copy opParam;
    opParam.dstSize = {2, 3};
    opParam.dstStride = {4, 1};
    opParam.dstOffset = {0};
    Mki::Test::UtOpDesc opDesc = {"CopyOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {2, 4}},
                                        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {2, 3}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpCopy, ViewCopyF32Case1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(0, 20);
    opTest.SetOutdataUseInputData(0, 0);
    opTest.Golden(&ViewCopyGolden);
    OpParam::Copy opParam;
    opParam.dstSize = {2, 2, 2};
    opParam.dstStride = {12, 6, 1};
    opParam.dstOffset = {0};
    Mki::Test::UtOpDesc opDesc = {"CopyOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {3, 4, 2, 3}},
                                        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {2, 2, 2}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpCopy, ViewCopyF32Case2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(0, 20);
    opTest.SetOutdataUseInputData(0, 0);
    opTest.Golden(&ViewCopyGolden);
    OpParam::Copy opParam;
    opParam.dstSize = {28, 2, 4096};
    opParam.dstStride = {10 * 4096, 4096, 1};
    opParam.dstOffset = {0};
    Mki::Test::UtOpDesc opDesc = {"CopyOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {28, 10, 4096}},
                                        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {28, 2, 4096}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpCopy, ViewCopyF16Case0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(0, 20);
    opTest.SetOutdataUseInputData(0, 0);
    opTest.Golden(&ViewCopyGolden);
    OpParam::Copy opParam;
    opParam.dstSize = {2, 3};
    opParam.dstStride = {4, 1};
    opParam.dstOffset = {0};
    Mki::Test::UtOpDesc opDesc = {"CopyOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {2, 4}},
                                        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {2, 3}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpCopy, ViewCopyF16Case1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(0, 20);
    opTest.SetOutdataUseInputData(0, 0);
    opTest.Golden(&ViewCopyGolden);
    OpParam::Copy opParam;
    opParam.dstSize = {2, 2, 2};
    opParam.dstStride = {12, 6, 1};
    opParam.dstOffset = {0};
    Mki::Test::UtOpDesc opDesc = {"CopyOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {3, 4, 2, 3}},
                                        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {2, 2, 2}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpCopy, ViewCopyF16Case2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(0, 20);
    opTest.SetOutdataUseInputData(0, 0);
    opTest.Golden(&ViewCopyGolden);
    OpParam::Copy opParam;
    opParam.dstSize = {28, 2, 4096};
    opParam.dstStride = {10 * 4096, 4096, 1};
    opParam.dstOffset = {0};
    Mki::Test::UtOpDesc opDesc = {"CopyOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {28, 10, 4096}},
                                        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {28, 2, 4096}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpCopy, ViewCopyFLOATCase3)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(0, 20);
    opTest.SetOutdataUseInputData(0, 0);
    opTest.Golden(&ViewCopyGolden);
    OpParam::Copy opParam;
    opParam.dstSize = {28, 2, 4096};
    opParam.dstStride = {10 * 4096, 4096, 1};
    opParam.dstOffset = {0};
    Mki::Test::UtOpDesc opDesc = {"CopyOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {28, 10, 4096}},
                                        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {28, 2, 4096}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}