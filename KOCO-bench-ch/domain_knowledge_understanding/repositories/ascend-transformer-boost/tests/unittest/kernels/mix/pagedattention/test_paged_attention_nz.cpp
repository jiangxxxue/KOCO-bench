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
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "device_version_check.h"
#include "../../test_common.h"

using namespace AtbOps;
using namespace Mki;
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float HALF_FLOAT_MIN = -1.0;
constexpr float HALF_FLOAT_MAX = 1.0;

namespace AtbOps {
Status MatchAtTensorCompareHalf1(at::Tensor &out, at::Tensor &gt)
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

Status PagedAttentionNzGolden(void *gt, const Mki::Test::GoldenContext &context)
{
    const Tensor &outTensor = context.hostOutTensors.at(0);
    at::Tensor atGtTensor = at::from_blob(gt, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    return MatchAtTensorCompareHalf1(atOutTensor, atGtTensor);
}
} // namespace AtbOps


TEST(TestPagedAttentionNz, PagedAttentionNzMaskTest1)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    int32_t numTokens = 1;
    int32_t numHeads = 32;
    int32_t kvHeads = 32;
    int32_t embeddingSize = 128;
    int32_t numBlocks = 128;
    int32_t blockSize = 128;

    float tor = 1.0 / sqrt(embeddingSize);
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);

    OpParam::PagedAttention opParam;
    opParam.type = OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK;
    opParam.headSize = numHeads;
    opParam.tor = tor;
    opParam.kvHead = 0;  // opParam.kvHead 默认是0;
    Mki::Test::UtOpDesc opDesc = {"PagedAttentionOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    int32_t numTokensPad = (numTokens + 15) / 16 * 16;
    std::vector<uint16_t> expect(numTokensPad * numHeads * embeddingSize, 0);
    opTest.ReadFile(expect.data(), expect.size() * sizeof(uint16_t),
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/expect_nz.bin");
    opTest.Golden(std::bind(PagedAttentionNzGolden, expect.data(), std::placeholders::_1));

    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {1, numHeads * embeddingSize / 16, numTokensPad, 16}}, // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {numBlocks, kvHeads * embeddingSize / 16, blockSize, 16}}, // k
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {numBlocks, kvHeads * embeddingSize / 16, blockSize, 16}}, // v
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {numTokens, 1}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {numTokens}},
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {numTokens * numHeads, 128/16, 16, 16}}, // alibi mask
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {0}},
    };
    SVector<std::string> inDataFiles = {
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/query_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/key_cache_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/value_cache_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/block_tables.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/context_lens.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/alibi_mask_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/logn_list.bin",
    };

    Status status = opTest.RunWithDataFile(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestPagedAttentionNz, PagedAttentionNzMaskTest2)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    int32_t numTokens = 2;
    int32_t numHeads = 32;
    int32_t kvHeads = 32;
    int32_t embeddingSize = 128;
    int32_t numBlocks = 128;
    int32_t blockSize = 128;

    float tor = 1.0 / sqrt(embeddingSize);
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);

    OpParam::PagedAttention opParam;
    opParam.type = OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK;
    opParam.headSize = numHeads;

    opParam.tor = tor;
    opParam.kvHead = 0;  // opParam.kvHead 默认是0;
    Mki::Test::UtOpDesc opDesc = {"PagedAttentionOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    int32_t numTokensPad = (numTokens + 15) / 16 * 16;
    std::vector<uint16_t> expect(numTokensPad * numHeads * embeddingSize, 0);
    opTest.ReadFile(expect.data(), expect.size() * sizeof(uint16_t),
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/expect_nz.bin");
    opTest.Golden(std::bind(PagedAttentionNzGolden, expect.data(), std::placeholders::_1));

    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {1, numHeads * embeddingSize / 16, numTokensPad, 16}}, // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {numBlocks, kvHeads * embeddingSize / 16, blockSize, 16}}, // k
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {numBlocks, kvHeads * embeddingSize / 16, blockSize, 16}}, // v
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {numTokens, 1}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {numTokens}},
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {numTokens * numHeads, 128/16, 16, 16}}, // alibi mask
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {0}} // logn
    };
    SVector<std::string> inDataFiles = {
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/query_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/key_cache_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/value_cache_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/block_tables.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/context_lens.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/alibi_mask_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/logn_list.bin",
    };

    Status status = opTest.RunWithDataFile(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestPagedAttentionNz, PagedAttentionNzMaskTest3)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    int32_t numTokens = 20;
    int32_t numHeads = 32;
    int32_t kvHeads = 16;
    int32_t embeddingSize = 128;
    int32_t numBlocks = 128;
    int32_t blockSize = 128;

    float tor = 1.0 / sqrt(embeddingSize);
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);

    OpParam::PagedAttention opParam;
    opParam.type = OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK;

    opParam.headSize = numHeads;
    opParam.tor = tor;
    opParam.kvHead = kvHeads;  // opParam.kvHead 默认是0;
    Mki::Test::UtOpDesc opDesc = {"PagedAttentionOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    int32_t numTokensPad = (numTokens + 15) / 16 * 16;
    std::vector<uint16_t> expect(numTokensPad * numHeads * embeddingSize, 0);
    opTest.ReadFile(expect.data(), expect.size() * sizeof(uint16_t),
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/expect_nz.bin");
    opTest.Golden(std::bind(PagedAttentionNzGolden, expect.data(), std::placeholders::_1));

    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {1, numHeads * embeddingSize / 16, numTokensPad, 16}}, // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {numBlocks, kvHeads * embeddingSize / 16, blockSize, 16}}, // k
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {numBlocks, kvHeads * embeddingSize / 16, blockSize, 16}}, // v
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {numTokens, 1}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {numTokens}},
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {numTokens * numHeads, 128/16, 16, 16}}, // alibi mask
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {0}} // logn
    };
    SVector<std::string> inDataFiles = {
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/query_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/key_cache_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/value_cache_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/block_tables.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/context_lens.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/alibi_mask_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/logn_list.bin"
    };

    Status status = opTest.RunWithDataFile(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestPagedAttentionNz, PagedAttentionNzMaskTest4)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    int32_t numTokens = 20;
    int32_t numHeads = 32;
    int32_t kvHeads = 16;
    int32_t embeddingSize = 128;
    int32_t numBlocks = 128;
    int32_t blockSize = 128;

    int32_t kvSeqLen = 513;
    int32_t kvBlockLen = (kvSeqLen + 128 - 1) / 128;
    int32_t kvMaskLen = (kvSeqLen + 16 - 1) / 16;

    float tor = 1.0 / sqrt(embeddingSize);
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);

    OpParam::PagedAttention opParam;
    opParam.type = OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK;
    opParam.headSize = numHeads;
    opParam.tor = tor;
    opParam.kvHead = kvHeads;  // opParam.kvHead 默认是0;
    Mki::Test::UtOpDesc opDesc = {"PagedAttentionOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    int32_t numTokensPad = (numTokens + 15) / 16 * 16;
    std::vector<uint16_t> expect(numTokensPad * numHeads * embeddingSize, 0);
    opTest.ReadFile(expect.data(), expect.size() * sizeof(uint16_t),
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/expect_nz.bin");
    opTest.Golden(std::bind(PagedAttentionNzGolden, expect.data(), std::placeholders::_1));

    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {1, numHeads * embeddingSize / 16, numTokensPad, 16}}, // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {numBlocks, kvHeads * embeddingSize / 16, blockSize, 16}}, // k
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {numBlocks, kvHeads * embeddingSize / 16, blockSize, 16}}, // v
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {numTokens, kvBlockLen}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {numTokens}},
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {numTokens * numHeads, kvMaskLen, 16, 16}}, // alibi mask
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {0}} // logn
    };

    SVector<std::string> inDataFiles = {
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/query_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/key_cache_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/value_cache_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/block_tables.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/context_lens.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/alibi_mask_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/logn_list.bin",
    };

    Status status = opTest.RunWithDataFile(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestPagedAttentionNz, PagedAttentionNzMaskLogNTest5)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    int32_t numTokens = 20;
    int32_t numHeads = 32;
    int32_t kvHeads = 32;
    int32_t embeddingSize = 128;
    int32_t numBlocks = 128;
    int32_t blockSize = 128;

    int32_t kvSeqLen = 513;
    int32_t kvBlockLen = (kvSeqLen + 128 - 1) / 128;
    int32_t kvMaskLen = (kvSeqLen + 16 - 1) / 16;

    float tor = 1.0 / sqrt(embeddingSize);
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);

    OpParam::PagedAttention opParam;
    opParam.type = OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK;

    opParam.headSize = numHeads;
    opParam.tor = tor;
    opParam.kvHead = kvHeads;  // opParam.kvHead 默认是0;
    opParam.scaleType = OpParam::PagedAttention::SCALE_LOGN;
    Mki::Test::UtOpDesc opDesc = {"PagedAttentionOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    int32_t numTokensPad = (numTokens + 15) / 16 * 16;
    std::vector<uint16_t> expect(numTokensPad * numHeads * embeddingSize, 0);
    opTest.ReadFile(expect.data(), expect.size() * sizeof(uint16_t),
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/expect_nz.bin");
    opTest.Golden(std::bind(PagedAttentionNzGolden, expect.data(), std::placeholders::_1));

    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {1, numHeads * embeddingSize / 16, numTokensPad, 16}}, // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {numBlocks, kvHeads * embeddingSize / 16, blockSize, 16}}, // k
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {numBlocks, kvHeads * embeddingSize / 16, blockSize, 16}}, // v
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {numTokens, kvBlockLen}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {numTokens}},
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {numTokens * numHeads, kvMaskLen, 16, 16}}, // alibi mask
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {numTokens}} // logn
    };

    SVector<std::string> inDataFiles = {
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/query_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/key_cache_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/value_cache_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/block_tables.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/context_lens.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/alibi_mask_nz.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/logn_list.bin",
    };

    Status status = opTest.RunWithDataFile(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}