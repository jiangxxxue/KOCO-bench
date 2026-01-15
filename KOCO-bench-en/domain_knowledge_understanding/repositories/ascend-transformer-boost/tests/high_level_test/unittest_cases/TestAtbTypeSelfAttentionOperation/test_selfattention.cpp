/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendTransformerBoost is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <gtest/gtest.h>
#include <atb/utils/log.h>
#include <atb/utils/singleton.h>
#include "atb/utils/config.h"
#include "test_utils/test_common.h"
#include "atb/operation.h"
#include "atb/infer_op_params.h"
#include "atb/utils/tensor_util.h"
#include "test_utils/operation_test.h"
#include "test_utils/test_utils.h"

using namespace atb;
using namespace Mki;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;

/// @brief golden
/// @param context
/// @return

Mki::Status SelfAttentionGolden(const OpsGoldenContext &context)
{
    // define param

    // get constructed input/output tensors
    const Mki::Tensor inTensor = context.hostInTensors.at(0);
    const Mki::Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    // construct ref input tensors
    at::Tensor atInRefTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kFloat);
    // get ref output tensor
    at::Tensor atOutRefTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    // compare
    float *atOutArray = (float *)atOutTensor.storage().data_ptr().get();
    float *atRefOutArray = (float *)atOutRefTensor.storage().data_ptr().get(); // golden
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

TEST(TestSelfAttentionOperation, InferShapeIntensor0Pass0)
{
    if (GetSingleton<Config>().Is310P()) {
        return;
    }
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Pass1BF)
{
    if (GetSingleton<Config>().Is310P()) {
        return;
    }
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_BF16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_BF16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_BF16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_BF16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_BF16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_BF16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
//    EXPECT_EQ(st, atb::NO_ERROR);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Pass2)
{
    if (GetSingleton<Config>().Is310P()) {
        return;
    }
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_UINT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_UINT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_UINT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail1)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_UNDEFINED, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail2)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail3)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_INT8, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail4)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail5)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_UINT8, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail6)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_INT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail7)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_INT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail8)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_UINT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail9)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_UINT32, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail10)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_INT64, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail11)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_UINT64, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail12)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_DOUBLE, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail13)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_BOOL, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail14)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_STRING, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail15)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_COMPLEX64, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail16)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_COMPLEX128, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail17)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_UNDEFINED, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail18)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NCHW, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail19)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NHWC, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail20)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NC1HWC0, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail21)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_Z, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail22)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NC1HWC0_C04, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail23)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_HWCN, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail24)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NDHWC, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail25)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_NZ, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail26)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NCDHW, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail27)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NDC1HWC0, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor0Fail28)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_Z_3D, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail1)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_UNDEFINED, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail2)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail3)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_INT8, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail4)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail5)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_UINT8, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail6)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_INT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail7)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_INT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail8)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_UINT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail9)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_UINT32, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail10)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_INT64, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail11)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_UINT64, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail12)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_DOUBLE, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail613)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_BOOL, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail14)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_STRING, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail15)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_COMPLEX64, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail16)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_COMPLEX128, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail17)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_UNDEFINED, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail18)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NCHW, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail19)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NHWC, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail20)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NC1HWC0, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail21)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_Z, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail22)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NC1HWC0_C04, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail23)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_HWCN, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail24)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NDHWC, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail25)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_NZ, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail26)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NCDHW, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail27)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NDC1HWC0, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor1Fail28)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_Z_3D, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail1)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_UNDEFINED, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail2)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail3)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_INT8, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail4)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail5)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_UINT8, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail6)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_INT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail7)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_INT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail8)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_UINT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail9)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_UINT32, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail10)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_INT64, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail11)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_UINT64, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail12)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_DOUBLE, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail613)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_BOOL, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail14)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_STRING, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail15)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_COMPLEX64, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail16)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_COMPLEX128, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail17)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_UNDEFINED, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail18)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NCHW, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail19)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NHWC, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail20)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NC1HWC0, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail21)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_Z, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail22)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NC1HWC0_C04, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail23)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_HWCN, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail24)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NDHWC, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail25)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_NZ, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_NZ, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail26)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NCDHW, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail27)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NDC1HWC0, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor2Fail28)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_Z_3D, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail1)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_UNDEFINED, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail2)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail3)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_INT8, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail4)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail5)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_UINT8, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail6)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_INT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail7)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_INT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail8)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_UINT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail9)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_UINT32, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail10)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_INT64, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail11)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_UINT64, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail12)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_DOUBLE, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail13)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_BOOL, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail14)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_STRING, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail15)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_COMPLEX64, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail16)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_COMPLEX128, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail17)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_UNDEFINED, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail18)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NCHW, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail19)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NHWC, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail20)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NC1HWC0, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail21)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_Z, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail22)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NC1HWC0_C04, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail23)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_HWCN, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail24)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NDHWC, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail25)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_NZ, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail26)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NCDHW, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail27)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NDC1HWC0, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor3Fail28)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_Z_3D, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail1)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_UNDEFINED, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail2)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail3)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_INT8, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail4)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail5)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_UINT8, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail6)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_INT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail7)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_INT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail8)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_UINT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail9)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_UINT32, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail10)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_INT64, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail11)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_UINT64, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail12)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_DOUBLE, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail13)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_BOOL, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail14)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_STRING, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail15)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_COMPLEX64, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail16)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_COMPLEX128, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail17)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_UNDEFINED, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail18)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NCHW, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail19)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NHWC, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail20)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NC1HWC0, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail21)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_Z, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail22)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NC1HWC0_C04, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail23)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_HWCN, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail24)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NDHWC, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail25)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_NZ, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail26)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NCDHW, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail27)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NDC1HWC0, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor4Fail28)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_Z_3D, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail1)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_UNDEFINED, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail2)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail3)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_INT8, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail4)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail5)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_UINT8, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail6)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_INT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail7)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_INT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail8)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_UINT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail9)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_UINT32, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail10)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_INT64, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail11)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_UINT64, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail12)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_DOUBLE, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail13)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_BOOL, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail14)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_STRING, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail15)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_COMPLEX64, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail16)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_COMPLEX128, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail17)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_UNDEFINED, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail18)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NCHW, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail19)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NHWC, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail20)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NC1HWC0, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail21)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_Z, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail22)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NC1HWC0_C04, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail23)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_HWCN, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail24)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NDHWC, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail25)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_NZ, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail26)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NCDHW, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail27)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_NDC1HWC0, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5Fail28)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_Z_3D, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5_310PFail30)
{
    if (GetSingleton<Config>().Is910B()) {
        return;
    }
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM;
    param.calcType = atb::infer::SelfAttentionParam::DECODER;
    param.kvcacheCfg = atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS;
    param.inputLayout = atb::infer::TYPE_BNSD;
    param.headNum = 32;
    param.qScale = 1;
    param.qkScale = 0.0883883;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 2;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2, 32, 16, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2, 64, 8, 2048, 16 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2, 64, 8, 2048, 16 } },
//        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_NZ, { 2, 128, 2048, 16 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2, 128, 2048, 16 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 2 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 2 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_DTYPE);
//    ASSERT_EQ(st, atb::NO_ERROR);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor5_310PFail31)
{
    if (GetSingleton<Config>().Is910B()) {
        return;
    }
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM;
    param.calcType = atb::infer::SelfAttentionParam::DECODER;
    param.kvcacheCfg = atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS;
    param.inputLayout = atb::infer::TYPE_BNSD;
    param.headNum = 32;
    param.qScale = 1;
    param.qkScale = 0.0883883;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 2;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2, 32, 16, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2, 64, 8, 2048, 16 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2, 64, 8, 2048, 16 } },
//        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_NZ, { 2, 128, 2048, 16 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_FRACTAL_NZ, { 2, 128, 2048, 16 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 2 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 2 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail1)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_UNDEFINED, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail2)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail3)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT8, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail4)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail5)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_UINT8, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail6)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT16, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail7)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_UINT16, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail8)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_UINT16, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail9)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_UINT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail10)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT64, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail11)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_UINT64, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail12)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_DOUBLE, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail13)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_BOOL, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail14)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_STRING, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail15)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_COMPLEX64, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail16)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_COMPLEX128, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail17)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_UNDEFINED, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail18)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NCHW, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail19)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NHWC, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail20)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NC1HWC0, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail21)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_FRACTAL_Z, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail22)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NC1HWC0_C04, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail23)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_HWCN, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail24)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NDHWC, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail25)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_FRACTAL_NZ, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail26)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NCDHW, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail27)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NDC1HWC0, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail28)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_FRACTAL_Z_3D, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor6Fail29)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_BF16, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail1)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_UNDEFINED, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail2)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail3)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT8, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail4)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail5)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_UINT8, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail6)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT16, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail7)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_UINT16, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail8)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_UINT16, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail9)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_UINT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail10)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT64, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail11)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_UINT64, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail12)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_DOUBLE, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail13)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_BOOL, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail14)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_STRING, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail15)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_COMPLEX64, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail16)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_COMPLEX128, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail17)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_UNDEFINED, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail18)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NCHW, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail19)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NHWC, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail20)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NC1HWC0, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail21)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_FRACTAL_Z, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail22)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NC1HWC0_C04, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail23)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_HWCN, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail24)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NDHWC, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail25)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_FRACTAL_NZ, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail26)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NCDHW, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail27)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NDC1HWC0, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail28)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_FRACTAL_Z_3D, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor7Fail29)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_BF16, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail1)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_UNDEFINED, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail2)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail3)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT8, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail4)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail5)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_UINT8, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail6)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT16, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail7)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_UINT16, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail8)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_UINT16, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail9)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_UINT32, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail10)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT64, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail11)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_UINT64, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail12)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_DOUBLE, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail13)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_BOOL, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail14)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_STRING, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail15)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_COMPLEX64, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail16)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_COMPLEX128, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail17)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_UNDEFINED, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail18)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NCHW, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail19)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NHWC, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail20)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NC1HWC0, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail21)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_FRACTAL_Z, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail22)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NC1HWC0_C04, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail23)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_HWCN, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail24)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NDHWC, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail25)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_FRACTAL_NZ, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail26)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NCDHW, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail27)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_NDC1HWC0, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail28)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_FRACTAL_Z_3D, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail29)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_BF16, Mki::TENSOR_FORMAT_ND, { 1 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_INI_MATCH);
}

TEST(TestSelfAttentionOperation, InferShapeIntensor8Fail30)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    param.headNum = 4;
    param.qScale = 0.2;
    param.qkScale = 0.2;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 4, 3, 32, 128 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 28, 3, 2048, 4096 } },
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 2048, 2048 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 3 } },
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, { 0 } } };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status st = op->InferShape(inTensorDescs, outTensorDescs);
    EXPECT_EQ(st, ERROR_INVALID_TENSOR_SIZE);
}

TEST(TestSelfAttentionOperation, isSupportAlibiInferShape)
{
    atb::infer::SelfAttentionParam param;
    param.headNum = 32;
    param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI;
    atb::Operation *op;
    atb::Status st = atb::CreateOperation<atb::infer::SelfAttentionParam>(param, &op);
    ASSERT_EQ(st, atb::NO_ERROR);
    const size_t seqLen = 4;
    const size_t batch = 3;
    const size_t headNum = 32;
    const size_t headSize = 128;
    const size_t layerNum = 28;
    const size_t maxSeqLen = 2048;
    const size_t hiddenSize = headNum * headSize;
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {seqLen, batch, headNum, headSize}},
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {seqLen, batch, headNum, headSize}},
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {seqLen, batch, headNum, headSize}},
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {layerNum, batch, maxSeqLen, hiddenSize}},
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {layerNum, batch, maxSeqLen, hiddenSize}},
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {maxSeqLen, maxSeqLen}},
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, {batch}},
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, {batch}},
        {Mki::TENSOR_DTYPE_INT32, Mki::TENSOR_FORMAT_ND, {1}}};
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Status status = op->InferShape(inTensorDescs, outTensorDescs);
    ASSERT_EQ(status, atb::NO_ERROR);
    ASSERT_EQ(outTensorDescs.size(), static_cast<std::size_t>(1));
    EXPECT_EQ(outTensorDescs.at(0).dtype, ACL_FLOAT16);
    Mki::SVector<int64_t> expectDims = {seqLen, batch, hiddenSize};
    ASSERT_EQ(expectDims.size(), outTensorDescs.at(0).shape.dimNum);
    EXPECT_EQ(expectDims.at(0), outTensorDescs.at(0).shape.dims[0]);
    EXPECT_EQ(expectDims.at(1), outTensorDescs.at(0).shape.dims[1]);
    EXPECT_EQ(expectDims.at(2), outTensorDescs.at(0).shape.dims[2]);
}