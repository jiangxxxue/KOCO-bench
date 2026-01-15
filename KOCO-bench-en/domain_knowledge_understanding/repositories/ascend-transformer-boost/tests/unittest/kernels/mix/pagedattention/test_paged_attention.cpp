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
#include "atbops/params/params.h"
#include "mki/utils/fp16/fp16_t.h"
#include "mki/utils/log/log.h"
#include "device_version_check.h"
#include "test_common.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"

using namespace AtbOps;
using namespace Mki;
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float HALF_FLOAT_MIN = -1.0;
constexpr float HALF_FLOAT_MAX = 1.0;

namespace AtbOps {
Status MatchAtTensorCompareHalf(at::Tensor &out, at::Tensor &gt)
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

Status PagedAttentionNdGolden(void *gt, const Mki::Test::GoldenContext &context)
{
    const Tensor &outTensor = context.hostOutTensors.at(0);
    at::Tensor atGtTensor = at::from_blob(gt, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    return MatchAtTensorCompareHalf(atOutTensor, atGtTensor);
}
} // namespace AtbOps


TEST(TestPagedAttentionNd, FlashAttentionEncoderTest)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    int64_t q_tokens = 512;
    int64_t headnum = 32;
    int64_t edim = 128;
    int64_t hiddenSize = 4096;
    int64_t max_s = 256;
    int32_t kvHead = 32;
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);

    Mki::Test::MkiOpTest opTest;
    std::vector<uint16_t> expect(q_tokens * hiddenSize, 0);
    opTest.ReadFile(expect.data(), expect.size() * sizeof(uint16_t),
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/expect.bin");
    opTest.Golden(std::bind(PagedAttentionNdGolden, expect.data(), std::placeholders::_1));
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_ND;
    opParam.qSeqLen = {256, 256};
    opParam.kvSeqLen = {256, 256};
    opParam.headSize = headnum;
    opParam.tor = 1.0 / sqrt(1.0 * edim);
    opParam.kvHead = 0; // kvHead
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_NORM;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {q_tokens, headnum, edim}}, // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {q_tokens, kvHead, edim}},  // k
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {q_tokens, kvHead, edim}},  // v
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}},                        // layerId
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {max_s, max_s}},            // mask
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {q_tokens}},
    };
    SVector<std::string> inDataFiles = {
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/input1.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/input2.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/input3.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/input4.bin"
    };

    Status status = opTest.RunWithDataFile(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestPagedAttentionNd, FlashAttentionEncoderTestGQA)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    int64_t q_tokens = 512;
    int64_t headnum = 32;
    int64_t edim = 128;
    int64_t hiddenSize = 4096;
    int64_t max_s = 256;
    int32_t kvHead = 16;
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);

    Mki::Test::MkiOpTest opTest;
    std::vector<uint16_t> expect(q_tokens * hiddenSize, 0);
    opTest.ReadFile(expect.data(), expect.size() * sizeof(uint16_t),
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/expect.bin");
    opTest.Golden(std::bind(PagedAttentionNdGolden, expect.data(), std::placeholders::_1));
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_ND;
    opParam.qSeqLen = {256, 256};
    opParam.kvSeqLen = {256, 256};
    opParam.headSize = headnum;
    opParam.tor = 1.0 / sqrt(1.0 * edim);
    opParam.kvHead = kvHead;
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_NORM;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {q_tokens, headnum, edim}}, // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {q_tokens, kvHead, edim}},  // k
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {q_tokens, kvHead, edim}},  // v
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}},                        // layerId
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {max_s, max_s}},            // mask
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {q_tokens}}
    };
    SVector<std::string> inDataFiles = {
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/input1.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/input2.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/input3.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/input4.bin"
    };

    Status status = opTest.RunWithDataFile(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestPagedAttentionNd, FlashAttentionEncoderTestBigBatch)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    int64_t q_tokens = 32 * 128;
    int64_t max_s = 128;

    int64_t headnum = 32;
    int64_t edim = 128;
    int64_t hiddenSize = 4096;
    int32_t kvHead = 32;
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);

    Mki::Test::MkiOpTest opTest;
    std::vector<uint16_t> expect(q_tokens * hiddenSize, 0);
    opTest.ReadFile(expect.data(), expect.size() * sizeof(uint16_t),
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/expect.bin");
    opTest.Golden(std::bind(PagedAttentionNdGolden, expect.data(), std::placeholders::_1));
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_ND;
    opParam.qSeqLen = {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                       128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    opParam.kvSeqLen = {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                        128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};

    opParam.headSize = headnum;
    opParam.tor = 1.0 / sqrt(1.0 * edim);
    opParam.kvHead = kvHead;
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_NORM;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {q_tokens, headnum, edim}}, // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {q_tokens, kvHead, edim}},  // k
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {q_tokens, kvHead, edim}},  // v
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}},                        // layerId
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {max_s, max_s}},            // mask
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                          // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {q_tokens}}
    };
    SVector<std::string> inDataFiles = {
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/input1.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/input2.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/input3.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/input4.bin"
    };

    Status status = opTest.RunWithDataFile(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestPagedAttentionNd, PagedAttentionTestAlibi)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    int32_t numTokens = 2;
    int32_t numHeads = 32;
    int32_t kvHeads = 32;
    int32_t embeddingSize = 128;
    int32_t numBlocks = 64;
    int32_t blockSize = 128;
    int32_t kSeqLen = 500;

    float tor = 1.0 / sqrt(embeddingSize);
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);

    OpParam::PagedAttention opParam;
    opParam.type = OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND;
    opParam.headSize = numHeads;
    opParam.tor = tor;
    opParam.kvHead = kvHeads;
    opParam.maskType = OpParam::PagedAttention::MASK_TYPE_ALIBI;
    opParam.kvSeqLen = {kSeqLen, kSeqLen};
    Mki::Test::UtOpDesc opDesc = {"PagedAttentionOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    std::vector<uint16_t> expect(numTokens * numHeads * embeddingSize, 0);
    opTest.ReadFile(expect.data(), expect.size() * sizeof(uint16_t),
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/expect.bin");
    opTest.Golden(std::bind(PagedAttentionNdGolden, expect.data(), std::placeholders::_1));

    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {numTokens, numHeads, embeddingSize}}, // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {numBlocks, blockSize, kvHeads, embeddingSize}}, // k
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {numBlocks, blockSize, kvHeads, embeddingSize}}, // v
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {numTokens, 4}},
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {numTokens, numHeads, 1, kSeqLen}},
        {TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},
        {TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {numTokens}}
    };
    SVector<std::string> inDataFiles = {
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/query.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/key_cache.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/value_cache.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/block_tables.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/alibi_mask.bin"
    };

    Status status = opTest.RunWithDataFile(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestPagedAttentionNd, PagedAttentionTestAlibi_dim3)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    int32_t numTokens = 2;
    int32_t numHeads = 32;
    int32_t kvHeads = 32;
    int32_t embeddingSize = 128;
    int32_t numBlocks = 64;
    int32_t blockSize = 128;
    int32_t kSeqLen = 500;

    float tor = 1.0 / sqrt(embeddingSize);
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);

    OpParam::PagedAttention opParam;
    opParam.type = OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND;
    opParam.headSize = numHeads;
    opParam.tor = tor;
    opParam.kvHead = kvHeads;
    opParam.maskType = OpParam::PagedAttention::MASK_TYPE_ALIBI;
    opParam.kvSeqLen = {kSeqLen, kSeqLen};
    Mki::Test::UtOpDesc opDesc = {"PagedAttentionOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    std::vector<uint16_t> expect(numTokens * numHeads * embeddingSize, 0);
    opTest.ReadFile(expect.data(), expect.size() * sizeof(uint16_t),
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/expect.bin");
    opTest.Golden(std::bind(PagedAttentionNdGolden, expect.data(), std::placeholders::_1));

    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {numTokens, numHeads, embeddingSize}}, // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {numBlocks, blockSize, kvHeads, embeddingSize}}, // k
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {numBlocks, blockSize, kvHeads, embeddingSize}}, // v
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {numTokens, 4}},
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {numTokens, numHeads, 1, kSeqLen}},
        {TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},
        {TENSOR_DTYPE_UINT64, TENSOR_FORMAT_ND, {}},
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {numTokens}}
        
    };
    SVector<std::string> inDataFiles = {
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/query.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/key_cache.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/value_cache.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/block_tables.bin",
        atbHomeDir + "/../../tests/unittest/mix/pagedattention/data/alibi_mask.bin"
    };

    Status status = opTest.RunWithDataFile(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}
