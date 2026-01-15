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
#include "mki/utils/fp16/fp16_t.h"
#include "mki/utils/log/log.h"
#include "atbops/params/params.h"
#include "device_version_check.h"
#include "test_common.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"

using namespace AtbOps;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
}  // namespace

namespace AtbOps {
Status MatchAtTensorHalfKVCache(at::Tensor &out, at::Tensor &gt)
{
    fp16_t *result = static_cast<fp16_t *>(out.storage().data_ptr().get());
    fp16_t *expect = static_cast<fp16_t *>(gt.storage().data_ptr().get());
    for (int i = 0; i < out.numel(); i++) {
        if (!Mki::Test::FloatUtil::FloatJudgeEqual(expect[i], result[i], ATOL, RTOL)) {
            std::string msg = "pos " + std::to_string(i) +
                              ", expect: " + std::to_string(static_cast<float>(expect[i])) +
                              ", result: " + std::to_string(static_cast<float>(result[i]));
            return Status::FailStatus(-1, msg);
        }
    }
    return Status::OkStatus();
}

Status KVCacheGolden(void *gt, const Mki::Test::GoldenContext &context)
{
    const Tensor &outTensor = context.hostOutTensors.at(0);
    at::Tensor atGtTensor = at::from_blob(gt, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    return MatchAtTensorHalfKVCache(atOutTensor, atGtTensor);
}
}  // namespace AtbOps

TEST(TestKVCache, KVCacheTest0)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    int64_t batch = 16;
    int64_t maxSeqLen = 384;
    int64_t hiddenSize = 1024;
    int64_t layer = 28;
    int64_t ntokens = 1331;

    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);

    Mki::Test::MkiOpTest opTest;
    opTest.SetOutdataUseInputData(0, 2);
    std::vector<uint16_t> expect(layer * batch * maxSeqLen * hiddenSize, 0);
    opTest.ReadFile(expect.data(),
        expect.size() * sizeof(uint16_t),
        atbHomeDir + "/../../tests/unittest/mix/kvcache/data/cache_Out.bin");
    opTest.Golden(std::bind(KVCacheGolden, expect.data(), std::placeholders::_1));
    OpParam::KVCache opParam;
    opParam.type = OpParam::KVCache::KVCACHE_ND;
    Mki::Test::UtOpDesc opDesc = {"KVCacheOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {1}},
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {layer, batch, maxSeqLen, hiddenSize}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {batch}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {batch}},
    };
    // tokenoffset must be greater than or equal to seqlen
    SVector<std::string> inDataFiles = {atbHomeDir + "/../../tests/unittest/mix/kvcache/data/newKV.bin",
        atbHomeDir + "/../../tests/unittest/mix/kvcache/data/layerid.bin",
        atbHomeDir + "/../../tests/unittest/mix/kvcache/data/cacheIn.bin",
        atbHomeDir + "/../../tests/unittest/mix/kvcache/data/tokenoffset.bin",
        atbHomeDir + "/../../tests/unittest/mix/kvcache/data/seqlen.bin"};

    Status status = opTest.RunWithDataFile(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestKVCache, KVCacheTest1)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    int64_t batch = 16;
    int64_t maxSeqLen = 384;
    int64_t hiddenSize = 2048;
    int64_t layer = 1;
    int64_t ntokens = 1024;

    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);

    Mki::Test::MkiOpTest opTest;
    opTest.SetOutdataUseInputData(0, 2);
    std::vector<uint16_t> expect(layer * batch * ((maxSeqLen + 15) / 16) * ((hiddenSize + 15) / 16) * 16 * 16, 0);
    opTest.ReadFile(expect.data(),
        expect.size() * sizeof(uint16_t),
        atbHomeDir + "/../../tests/unittest/mix/kvcache/data_nz/cache_Out.bin");

    opTest.Golden(std::bind(KVCacheGolden, expect.data(), std::placeholders::_1));
    OpParam::KVCache opParam;
    opParam.type = OpParam::KVCache::KVCACHE_NZ;
    Mki::Test::UtOpDesc opDesc = {"KVCacheOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {1, (hiddenSize + 15) / 16, (ntokens + 15) / 16 * 16, 16}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {1}},
        {TENSOR_DTYPE_FLOAT16,
            TENSOR_FORMAT_FRACTAL_NZ,
            {layer, batch, (hiddenSize + 15) / 16, (maxSeqLen + 15) / 16 * 16, 16}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {batch}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {batch}},
    };
    // tokenoffset must be greater than or equal to seqlen
    SVector<std::string> inDataFiles = {atbHomeDir + "/../../tests/unittest/mix/kvcache/data_nz/newKV.bin",
        atbHomeDir + "/../../tests/unittest/mix/kvcache/data_nz/layerid.bin",
        atbHomeDir + "/../../tests/unittest/mix/kvcache/data_nz/cacheIn.bin",
        atbHomeDir + "/../../tests/unittest/mix/kvcache/data_nz/tokenoffset.bin",
        atbHomeDir + "/../../tests/unittest/mix/kvcache/data_nz/seqlen.bin"};

    Status status = opTest.RunWithDataFile(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestKVCache, KVCacheTest2)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    int64_t batch = 16;
    int64_t maxSeqLen = 384;
    int64_t hiddenSize = 1024;
    int64_t ntokens = 1331;

    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);

    Mki::Test::MkiOpTest opTest;
    opTest.SetOutdataUseInputData(0, 2);
    std::vector<uint16_t> expect(batch * maxSeqLen * hiddenSize, 0);
    opTest.ReadFile(expect.data(),
        expect.size() * sizeof(uint16_t),
        atbHomeDir + "/../../tests/unittest/mix/kvcache/data_nd_dync_batch/cache_Out.bin");
    opTest.Golden(std::bind(KVCacheGolden, expect.data(), std::placeholders::_1));
    OpParam::KVCache opParam;
    opParam.type = OpParam::KVCache::KVCACHE_ND;
    Mki::Test::UtOpDesc opDesc = {"KVCacheOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {1}},
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch, maxSeqLen, hiddenSize}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {batch}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {batch}},
    };
    // tokenoffset must be greater than or equal to seqlen
    SVector<std::string> inDataFiles = {atbHomeDir + "/../../tests/unittest/mix/kvcache/data_nd_dync_batch/newKV.bin",
        atbHomeDir + "/../../tests/unittest/mix/kvcache/data_nd_dync_batch/layerid.bin",
        atbHomeDir + "/../../tests/unittest/mix/kvcache/data_nd_dync_batch/cacheIn.bin",
        atbHomeDir + "/../../tests/unittest/mix/kvcache/data_nd_dync_batch/tokenoffset.bin",
        atbHomeDir + "/../../tests/unittest/mix/kvcache/data_nd_dync_batch/seqlen.bin"};

    Status status = opTest.RunWithDataFile(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}