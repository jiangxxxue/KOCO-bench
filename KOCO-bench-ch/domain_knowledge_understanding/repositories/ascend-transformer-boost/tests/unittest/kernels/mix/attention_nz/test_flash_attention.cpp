/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <cmath>
#include <cstdlib>
#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include "mki_loader/op_register.h"
#include "mki/utils/fp16/fp16_t.h"
#include "mki/utils/log/log.h"
#include "atbops/params/params.h"
#include "mixkernels/unpad_flash_attention_nz/tiling/flash_attention_nz_tiling.h"
#include "device_version_check.h"
#include "test_common.h"
#include "test_utils/op_test.h"
#include "test_utils/op_test_tools.h"


using namespace AtbOps;
using namespace Mki;
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float HALF_FLOAT_MIN = -1.0;
constexpr float HALF_FLOAT_MAX = 1.0;
constexpr int32_t DEFAULT_BATCH = 1;

namespace AtbOps {
Status FlashAttentionNzGolden(void *gt, void *gt_high, const Mki::Test::GoldenContext &context)
{
    const Tensor &outTensor = context.hostOutTensors.at(0);
    at::Tensor atGtTensor = at::from_blob(gt, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    at::Tensor atGtHighTensor = at::from_blob(gt_high, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    at::Tensor atGtTensor_float = atGtTensor.to(at::kFloat);
    at::Tensor atOutTensor_float = atOutTensor.to(at::kFloat);
    std::string dtype = "fp16";
    return Mki::Test::MatchAtTensor(atOutTensor, atGtTensor, atGtHighTensor, dtype);
}
} // namespace AtbOps

TEST(TestFlashAttentionNz, FlashAttentionTestKVPtr_normmask_dim3)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND310P();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention_nz";
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_nz_data_gen_renew.py 2 513 8 1 32 128 2048 0 0 3 0 0";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    PrepareScalarParams(scalars, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += (it + 15) / 16 * 16;
    }
    std::vector<uint16_t> expect(((qHead * edim + 15) / 16) * qnTokens * 16, 0);
    std::vector<uint32_t> expect_high(((qHead * edim + 15) / 16) * qnTokens * 16, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    opTest.Golden(std::bind(FlashAttentionNzGolden, expect.data(), expect_high.data(), std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {(edim * kvHead + 15)/ 16, (maxSeq + 15) / 16 * 16, 16}};
    std::vector<std::string> kvFiles;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles);
    
    OpParam::UnpadFlashAttentionNz opParam;
    opParam.type = OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_DECODER;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionNzOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {DEFAULT_BATCH, (qHead * edim + 15) / 16,
                                                  qnTokens, 16}},       // q
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // k
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // v
        {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}},                                     // layer
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (maxSeq + 15) / 16,
                                                  (maxSeq + 15) / 16 * 16, 16}},                       // mask
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}}, // optional logn for scale attention score
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestFlashAttentionNz, FlashAttentionTestKVPtr_normmask_dim2)
{
        // should generate data first
    CHECK_DEVICE_VERSION_ASCEND310P();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention_nz";
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_nz_data_gen_renew.py 5 1025 8 1 8 128 2048 0 0 2 0 0";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    PrepareScalarParams(scalars, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += (it + 15) / 16 * 16;
    }
    std::vector<uint16_t> expect(((qHead * edim + 15) / 16) * qnTokens * 16, 0);
    std::vector<uint32_t> expect_high(((qHead * edim + 15) / 16) * qnTokens * 16, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    opTest.Golden(std::bind(FlashAttentionNzGolden, expect.data(), expect_high.data(), std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {(edim * kvHead + 15)/ 16, (maxSeq + 15) / 16 * 16, 16}};
    std::vector<std::string> kvFiles;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles);

    OpParam::UnpadFlashAttentionNz opParam;
    opParam.type = OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_DECODER;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionNzOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {DEFAULT_BATCH, (qHead * edim + 15) / 16,
                                                  qnTokens, 16}},       // q
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // k
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // v
        {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}},                                     // layer
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {1, (maxSeq + 15) / 16,
                                                  (maxSeq + 15) / 16 * 16, 16}},                       // mask
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}}, // optional logn for scale attention score
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestFlashAttentionNz, FlashAttentionTestKVPtr_alibimask_dim4)
{
        // should generate data first
    CHECK_DEVICE_VERSION_ASCEND310P();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention_nz";
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_nz_data_gen_renew.py 16 1024 12 1 12 128 2048 0 1 4 0 0";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    PrepareScalarParams(scalars, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += (it + 15) / 16 * 16;
    }
    std::vector<uint16_t> expect(((qHead * edim + 15) / 16) * qnTokens * 16, 0);
    std::vector<uint32_t> expect_high(((qHead * edim + 15) / 16) * qnTokens * 16, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    opTest.Golden(std::bind(FlashAttentionNzGolden, expect.data(), expect_high.data(), std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {(edim * kvHead + 15)/ 16, (maxSeq + 15) / 16 * 16, 16}};
    std::vector<std::string> kvFiles;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles);
    
    OpParam::UnpadFlashAttentionNz opParam;
    opParam.type = OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_DECODER;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.maskType = OpParam::UnpadFlashAttentionNz::MASK_TYPE_ALIBI;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionNzOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {DEFAULT_BATCH, (qHead * edim + 15) / 16,
                                                  qnTokens, 16}},       // q
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // k
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // v
        {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}},                                     // layer
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch * kvHead, (maxSeq + 15) / 16,
                                                  (maxSeq + 15) / 16 * 16, 16}},                       // mask
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}}, // optional logn for scale attention score
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestFlashAttentionNz, FlashAttentionTestKVPtr_alibimask_dim3)
{
        // should generate data first
    CHECK_DEVICE_VERSION_ASCEND310P();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention_nz";
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_nz_data_gen_renew.py 28 88 32 1 32 128 256 0 1 3 0 0";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    PrepareScalarParams(scalars, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += (it + 15) / 16 * 16;
    }
    std::vector<uint16_t> expect(((qHead * edim + 15) / 16) * qnTokens * 16, 0);
    std::vector<uint32_t> expect_high(((qHead * edim + 15) / 16) * qnTokens * 16, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    opTest.Golden(std::bind(FlashAttentionNzGolden, expect.data(), expect_high.data(), std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {(edim * kvHead + 15)/ 16, (maxSeq + 15) / 16 * 16, 16}};
    std::vector<std::string> kvFiles;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles);
    
    OpParam::UnpadFlashAttentionNz opParam;
    opParam.type = OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_DECODER;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.maskType = OpParam::UnpadFlashAttentionNz::MASK_TYPE_ALIBI;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionNzOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {DEFAULT_BATCH, (qHead * edim + 15) / 16,
                                                  qnTokens, 16}},       // q
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // k
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // v
        {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}},                                     // layer
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {kvHead, (maxSeq + 15) / 16,
                                                  (maxSeq + 15) / 16 * 16, 16}},                       // mask
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}}, // optional logn for scale attention score
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}
/**
 * @brief param.headSize > 0, return Status::FailStatus(ERROR_INVALID_VALUE, "headSize is invalid"
 */
TEST(TestFlashAttentionNz, TestFlashAttentionNzCanSupport)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    int64_t batch = 1;
    int64_t maxSeqlen = 2048;
    int64_t headnum = 32;
    int64_t edim = 128;
    int64_t layer = 2;
    int64_t kvHeads = 32;
    int64_t kvRealHeads = kvHeads > 0 ? kvHeads : headnum;
    int64_t qntokens = 256;
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (headnum * edim + 15) / 16,
                                                                         (qntokens + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {layer, batch, (edim * kvRealHeads + 15)/ 16,
                                                                         (maxSeqlen + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {layer, batch, (edim * kvRealHeads + 15)/ 16,
                                                                         (maxSeqlen + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (maxSeqlen + 15) / 16,
                                                                          (maxSeqlen + 15) / 16 * 16, 16}}); // mask
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}}); // optional logn for scale attention score
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (headnum * edim + 15) / 16,
                                                                          (qntokens + 15) / 16 * 16, 16}});
    OpParam::UnpadFlashAttentionNz opParam;
    opParam.type = OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ;
    opParam.kvSeqLen = {maxSeqlen};
    opParam.qSeqLen = {maxSeqlen};
    opParam.headSize = headnum;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionNzOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Operation *operation = AutoGen::GetOpByName(opDesc.opName);
    Status status = operation->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    std::unique_ptr<Kernel> kernel{nullptr};
    auto &kernelCreators = AtbOps::KernelRegister::GetKernelCreators();
    for (const auto &iter : kernelCreators) {
        if (iter.kernelName == "UnpadFlashAttentionNzEncoderKernel") {
            const std::string &opName = iter.opName;
            Operation *op = AutoGen::GetOpByName(opName);
            kernel.reset(op->GetKernelByName("UnpadFlashAttentionNzEncoderKernel"));
            break;
        }
    }
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->CanSupport(launchParam), true);
}

TEST(TestFlashAttentionNz, TestFlashAttentionNzInitHostLaunchBuffer)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    int64_t batch = 1;
    int64_t maxSeqlen = 2048;
    int64_t headnum = 32;
    int64_t edim = 128;
    int64_t layer = 2;
    int64_t kvHeads = 32;
    int64_t kvRealHeads = kvHeads > 0 ? kvHeads : headnum;
    int64_t qntokens = 256;
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (headnum * edim + 15) / 16,
                                                                         (qntokens + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {layer, batch, (edim * kvRealHeads + 15)/ 16,
                                                                         (maxSeqlen + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {layer, batch, (edim * kvRealHeads + 15)/ 16,
                                                                         (maxSeqlen + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (maxSeqlen + 15) / 16,
                                                                          (maxSeqlen + 15) / 16 * 16, 16}}); // mask
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}}); // optional logn for scale attention score                                        
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (headnum * edim + 15) / 16,
                                                                          (qntokens + 15) / 16 * 16, 16}});
    OpParam::UnpadFlashAttentionNz opParam;
    opParam.type = OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionNzOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Operation *op = AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    EXPECT_EQ(status.Ok(), true);
    std::unique_ptr<Kernel> kernel(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->GetTilingSize(launchParam), 0);
}
/**
 * @brief ASDOPS_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 4, return false)
 */
TEST(TestFlashAttentionNz, TestFlashAttentionNzCanSupport0)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    int64_t batch = 1;
    int64_t seqlen = 2;
    int64_t headnum = 32;
    int64_t edim = 128;
    int64_t layer = 1;
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch * seqlen, headnum * edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {layer, batch, seqlen, headnum * edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {layer, batch, seqlen, headnum * edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {seqlen, seqlen}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}}); // optional logn for scale attention score
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch * seqlen, headnum * edim}});
    OpParam::UnpadFlashAttentionNz opParam;
    opParam.type = OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionNzOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    std::unique_ptr<Kernel> kernel{nullptr};
    auto &kernelCreators = AtbOps::KernelRegister::GetKernelCreators();
    for (const auto &iter : kernelCreators) {
        if (iter.kernelName == "UnpadFlashAttentionNzEncoderKernel") {
            const std::string &opName = iter.opName;
            Operation *op = AutoGen::GetOpByName(opName);
            kernel.reset(op->GetKernelByName("UnpadFlashAttentionNzEncoderKernel"));
            break;
        }
    }
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ASDOPS_CHECK(qSeqLenAligned > 0, return Status::FailStatus(ERROR_INVALID_VALUE, "qSeqLenAligned is invalid"));
 */
TEST(TestFlashAttentionNz, TestFlashAttentionNzTiling)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    int64_t batch = 1;
    int64_t maxSeqlen = 2048;
    int64_t headnum = 32;
    int64_t edim = 128;
    int64_t layer = 2;
    int64_t kvHeads = 32;
    int64_t kvRealHeads = kvHeads > 0 ? kvHeads : headnum;
    int64_t qntokens = 256;
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (headnum * edim + 15) / 16,
                                                                         (qntokens + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {layer, batch, (edim * kvRealHeads + 15)/ 16,
                                                                         (maxSeqlen + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {layer, batch, (edim * kvRealHeads + 15)/ 16,
                                                                         (maxSeqlen + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (maxSeqlen + 15) / 16,
                                                                          (maxSeqlen + 15) / 16 * 16, 16}}); // mask
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}}); // optional logn for scale attention score
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (headnum * edim + 15) / 16,
                                                                          (qntokens + 15) / 16 * 16, 16}});
    OpParam::UnpadFlashAttentionNz opParam;
    opParam.type = OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ;
    opParam.headSize = headnum;
    opParam.qSeqLen = {0};
    opParam.kvSeqLen = {0};
    opParam.kvHead = kvHeads;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionNzOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    std::unique_ptr<Kernel> kernel{nullptr};
    auto &kernelCreators = AtbOps::KernelRegister::GetKernelCreators();
    for (const auto &iter : kernelCreators) {
        if (iter.kernelName == "UnpadFlashAttentionNzEncoderKernel") {
            const std::string &opName = iter.opName;
            Operation *op = AutoGen::GetOpByName(opName);
            kernel.reset(op->GetKernelByName("UnpadFlashAttentionNzEncoderKernel"));
            break;
        }
    }

    ASSERT_NE(kernel, nullptr);
    kernel->GetTilingSize(launchParam);
}

TEST(TestFlashAttentionNz, TestFlashAttentionNzGetBestKernel)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    int64_t batch = 1;
    int64_t maxSeqlen = 2048;
    int64_t headnum = 32;
    int64_t edim = 128;
    int64_t layer = 2;
    int64_t kvHeads = 32;
    int64_t kvRealHeads = kvHeads > 0 ? kvHeads : headnum;
    int64_t qntokens = 256;
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (headnum * edim + 15) / 16,
                                                                         (qntokens + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {layer, batch, (edim * kvRealHeads + 15)/ 16,
                                                                         (maxSeqlen + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {layer, batch, (edim * kvRealHeads + 15)/ 16,
                                                                         (maxSeqlen + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (maxSeqlen + 15) / 16,
                                                                          (maxSeqlen + 15) / 16 * 16, 16}}); // mask
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}}); // optional logn for scale attention score
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (headnum * edim + 15) / 16,
                                                                          (qntokens + 15) / 16 * 16, 16}});
    OpParam::UnpadFlashAttentionNz opParam;
    opParam.type = OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionNzOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    Operation *operation = AutoGen::GetOpByName(opDesc.opName);
    ASSERT_NE(operation, nullptr);
    std::unique_ptr<Kernel> kernel(operation->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

TEST(TestFlashAttentionNz, TestFlashAttentionNzInferShape)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    int64_t batch = 1;
    int64_t maxSeqlen = 2048;
    int64_t headnum = 32;
    int64_t edim = 128;
    int64_t layer = 2;
    int64_t kvHeads = 32;
    int64_t kvRealHeads = kvHeads > 0 ? kvHeads : headnum;
    int64_t qntokens = 256;
    AtbOps::SVector<AtbOps::Tensor> outTensor;
    outTensor.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (headnum * edim + 15) / 16,
                                                                          (qntokens + 15) / 16 * 16, 16}});
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (headnum * edim + 15) / 16,
                                                                         (qntokens + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {layer, batch, (edim * kvRealHeads + 15)/ 16,
                                                                         (maxSeqlen + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {layer, batch, (edim * kvRealHeads + 15)/ 16,
                                                                         (maxSeqlen + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (maxSeqlen + 15) / 16,
                                                                          (maxSeqlen + 15) / 16 * 16, 16}}); // mask
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}}); // optional logn for scale attention score
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (headnum * edim + 15) / 16,
                                                                          (qntokens + 15) / 16 * 16, 16}});
    OpParam::UnpadFlashAttentionNz opParam;
    opParam.type = OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ;
    opParam.kvSeqLen = {maxSeqlen};
    opParam.qSeqLen = {maxSeqlen};
    opParam.headSize = headnum;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionNzOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    Operation *operation = AutoGen::GetOpByName(opDesc.opName);
    ASSERT_EQ(operation != nullptr, true);
    Status status = operation->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);

    bool flag = true;
    for (size_t i = 0; i < launchParam.GetOutTensorCount(); ++i) {
        flag &= outTensor[i].desc.dtype == launchParam.GetOutTensor(i).desc.dtype;
        flag &= outTensor[i].desc.format == launchParam.GetOutTensor(i).desc.format;
        flag &= outTensor[i].desc.dims.size() == launchParam.GetOutTensor(i).desc.dims.size();
        for (size_t j = 0; j < outTensor[i].desc.dims.size(); ++j) {
            flag &= outTensor[i].desc.dims[j] == launchParam.GetOutTensor(i).desc.dims[j];
        }
    }
    ASSERT_EQ(flag, true);
}

TEST(TestFlashAttentionNz, TestFlashAttentionNzInferShapeFailed)
{
    CHECK_DEVICE_VERSION_ASCEND310P();
    int64_t batch = 1;
    int64_t maxSeqlen = 2048;
    int64_t headnum = 32;
    int64_t edim = 128;
    int64_t layer = 2;
    int64_t kvHeads = 32;
    int64_t kvRealHeads = kvHeads > 0 ? kvHeads : headnum;
    int64_t qntokens = 256;
    AtbOps::SVector<AtbOps::Tensor> outTensor;
    outTensor.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 1}});
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (headnum * edim + 15) / 16,
                                                                         (qntokens + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {layer, batch, (edim * kvRealHeads + 15)/ 16,
                                                                         (maxSeqlen + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {layer, batch, (edim * kvRealHeads + 15)/ 16,
                                                                         (maxSeqlen + 15) / 16 * 16, 16}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (maxSeqlen + 15) / 16,
                                                                          (maxSeqlen + 15) / 16 * 16, 16}}); // mask
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}}); // optional logn for scale attention score
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch, (headnum * edim + 15) / 16,
                                                                          (qntokens + 15) / 16 * 16, 16}});
    OpParam::UnpadFlashAttentionNz opParam;
    opParam.type = OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ;
    opParam.kvSeqLen = {maxSeqlen};
    opParam.qSeqLen = {maxSeqlen};
    opParam.headSize = headnum;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionNzOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    Operation *operation = AutoGen::GetOpByName(opDesc.opName);
    ASSERT_EQ(operation != nullptr, true);
    Status status = operation->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);

    bool flag = true;
    for (size_t i = 0; i < launchParam.GetOutTensorCount(); ++i) {
        flag &= outTensor[i].desc.dtype == launchParam.GetOutTensor(i).desc.dtype;
        flag &= outTensor[i].desc.format == launchParam.GetOutTensor(i).desc.format;
        flag &= outTensor[i].desc.dims.size() == launchParam.GetOutTensor(i).desc.dims.size();
    }
    ASSERT_EQ(flag, false);
}

TEST(TestFlashAttentionNz, FlashAttentionTestKVPtr_alibimask_dim4_BNSD)
{
        // should generate data first
    CHECK_DEVICE_VERSION_ASCEND310P();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention_nz";
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_nz_data_gen_renew.py 16 1024 12 1 12 128 2048 0 1 4 0 1";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    PrepareScalarParams(scalars, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += (it + 15) / 16 * 16;
    }
    std::vector<uint16_t> expect(((qHead * edim + 15) / 16) * qnTokens * 16, 0);
    std::vector<uint32_t> expect_high(((qHead * edim + 15) / 16) * qnTokens * 16, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    opTest.Golden(std::bind(FlashAttentionNzGolden, expect.data(), expect_high.data(), std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {(edim * kvHead + 15)/ 16, (maxSeq + 15) / 16 * 16, 16}};
    std::vector<std::string> kvFiles;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles);
    
    OpParam::UnpadFlashAttentionNz opParam;
    opParam.type = OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_DECODER;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.maskType = OpParam::UnpadFlashAttentionNz::MASK_TYPE_ALIBI;
    opParam.dataDimOrder = OpParam::UnpadFlashAttentionNz::TYPE_BNSD;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionNzOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {DEFAULT_BATCH, (qHead * edim + 15) / 16,
                                                  qnTokens, 16}},       // q
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // k
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // v
        {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}},                                     // layer
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {batch * kvHead, (maxSeq + 15) / 16,
                                                  (maxSeq + 15) / 16 * 16, 16}},                       // mask
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}}, // optional logn for scale attention score
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestFlashAttentionNz, FlashAttentionTestKVPtr_alibimask_dim3_BNSD)
{
        // should generate data first
    CHECK_DEVICE_VERSION_ASCEND310P();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention_nz";
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_nz_data_gen_renew.py 28 88 32 1 32 128 256 0 1 3 0 1";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    PrepareScalarParams(scalars, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += (it + 15) / 16 * 16;
    }
    std::vector<uint16_t> expect(((qHead * edim + 15) / 16) * qnTokens * 16, 0);
    std::vector<uint32_t> expect_high(((qHead * edim + 15) / 16) * qnTokens * 16, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    opTest.Golden(std::bind(FlashAttentionNzGolden, expect.data(), expect_high.data(), std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {(edim * kvHead + 15)/ 16, (maxSeq + 15) / 16 * 16, 16}};
    std::vector<std::string> kvFiles;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles);
    
    OpParam::UnpadFlashAttentionNz opParam;
    opParam.type = OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_DECODER;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.maskType = OpParam::UnpadFlashAttentionNz::MASK_TYPE_ALIBI;
    opParam.dataDimOrder = OpParam::UnpadFlashAttentionNz::TYPE_BNSD;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionNzOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {DEFAULT_BATCH, (qHead * edim + 15) / 16,
                                                  qnTokens, 16}},       // q
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // k
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // v
        {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}},                                     // layer
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, {kvHead, (maxSeq + 15) / 16,
                                                  (maxSeq + 15) / 16 * 16, 16}},                       // mask
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}}, // optional logn for scale attention score
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}