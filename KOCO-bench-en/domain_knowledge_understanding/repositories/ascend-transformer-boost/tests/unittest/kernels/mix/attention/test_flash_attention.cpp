/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <cstdlib>
#include <fstream>
#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include "mki_loader/op_register.h"
#include "mki/utils/fp16/fp16_t.h"
#include "mki/utils/log/log.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/op_test_tools.h"
#include "atbops/params/params.h"
#include "mixkernels/unpad_flash_attention/tiling/flash_attention_tiling.h"
#include "op_desc_json.h"
#include "device_version_check.h"
#include "test_common.h"

constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float HALF_FLOAT_MIN = -1.0;
constexpr float HALF_FLOAT_MAX = 1.0;

using namespace Mki;
using namespace AtbOps;
using namespace Mki;

namespace AtbOps {
Status FlashAttentionNdGolden(void *gt, void *gt_high, std::string &dtype, const Mki::Test::GoldenContext &context)
{
    const Tensor &outTensor = context.hostOutTensors.at(0);
    at::Tensor atGtTensor;
    at::Tensor atOutTensor;
    at::Tensor atGtHighTensor = at::from_blob(gt_high, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    if (dtype == "fp16") {
        atGtTensor = at::from_blob(gt, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
        atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    } else {
        atGtTensor = at::from_blob(gt, ToIntArrayRef(outTensor.desc.dims), at::ScalarType::BFloat16);
        atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::ScalarType::BFloat16);
    }
    at::Tensor atGtTensor_float = atGtTensor.to(at::kFloat);
    at::Tensor atOutTensor_float = atOutTensor.to(at::kFloat);
    return Mki::Test::MatchAtTensor(atOutTensor_float, atGtTensor_float, atGtHighTensor, dtype);
}
}

TEST(TestFlashAttentionNd, TestFlashAttentionNdCanSupport)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
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
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {seqlen}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch * seqlen, headnum * edim}});
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ND;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    std::unique_ptr<Kernel> kernel{nullptr};
    auto &kernelCreators = AtbOps::KernelRegister::GetKernelCreators();
    for (const auto &iter : kernelCreators) {
        if (iter.kernelName == "UnpadFlashAttentionNdKernel") {
            const std::string &opName = iter.opName;
            Operation *op = AutoGen::GetOpByName(opName);
            kernel.reset(op->GetKernelByName("UnpadFlashAttentionNdKernel"));
            break;
        }
    }
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

TEST(TestFlashAttentionNd, TestFlashAttentionNdGetBestKernel)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
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
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {seqlen}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch * seqlen, headnum * edim}});
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ND;
    opParam.kvSeqLen = {seqlen};
    opParam.qSeqLen = {seqlen};
    opParam.headSize = headnum;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    Operation *operation = AutoGen::GetOpByName(opDesc.opName);
    ASSERT_NE(operation, nullptr);
    std::unique_ptr<Kernel> kernel(operation->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

TEST(TestFlashAttentionNd, TestFlashAttentionNdInferShape)
{
    int64_t batch = 1;
    int64_t seqlen = 2;
    int64_t headnum = 32;
    int64_t edim = 128;
    int64_t layer = 1;
    SVector<Tensor> outTensor;
    outTensor.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch * seqlen, headnum * edim}});
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch * seqlen, headnum * edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {layer, batch, seqlen, headnum * edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {layer, batch, seqlen, headnum * edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {seqlen, seqlen}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {seqlen}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch * seqlen, headnum * edim}});
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ND;
    opParam.kvSeqLen = {seqlen};
    opParam.qSeqLen = {seqlen};
    opParam.headSize = headnum;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
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

TEST(TestFlashAttentionNd, TestFlashAttentionNdInferShapeFailed)
{
    int64_t batch = 1;
    int64_t seqlen = 2;
    int64_t headnum = 32;
    int64_t edim = 128;
    int64_t layer = 1;
    SVector<Tensor> outTensor;
    outTensor.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 1}});
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch * seqlen, headnum * edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {layer, batch, seqlen, headnum * edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {layer, batch, seqlen, headnum * edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {seqlen, seqlen}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {seqlen}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 1}});
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ND;
    opParam.kvSeqLen = {seqlen};
    opParam.qSeqLen = {seqlen};
    opParam.headSize = headnum;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
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
    ASSERT_EQ(flag, false);
}

TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_mask1_fp16)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" +
                          subDir + "/data/unpad_FA_data_gen_renew.py 16 1024 12 1 12 128 2048 0 0 1 1 0";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles);
    
    OpParam::UnpadFlashAttention opParam;
    opParam.type = batchDynamic ? OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION_DECODER
                                : OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.batchRunStatus = batch_state;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // k
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // v
        {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}},                                     // layer
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch, maxSeq, maxSeq}},                       // mask
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {batch}},
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_mask2_fp16)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_data_gen_renew.py 2 513 8 1 32 128 2048 0 0 2 1 0";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles);
    OpParam::UnpadFlashAttention opParam;
    opParam.type = batchDynamic ? OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION_DECODER
                                : OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.batchRunStatus = batch_state;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // k
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // v
        {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}},                                     // layer
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, maxSeq}},                       // mask
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {batch}},
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_mask3_fp16)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_data_gen_renew.py 128 1367 16 1 16 128 2048 0 0 3 1 0";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles);
    
    OpParam::UnpadFlashAttention opParam;
    opParam.type = batchDynamic ? OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION_DECODER
                                : OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.batchRunStatus = batch_state;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // k
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // v
        {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}},                                     // layer
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch, qHead, maxSeq, maxSeq}},                       // mask
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {batch}},
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_mask4_fp16)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_data_gen_renew.py 128 1634 8 1 8 128 2048 0 0 4 1 0";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles);
    
    OpParam::UnpadFlashAttention opParam;
    opParam.type = batchDynamic ? OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION_DECODER
                                : OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.batchRunStatus = batch_state;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // k
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // v
        {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}},                                     // layer
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qHead, maxSeq, maxSeq}},                       // mask
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {batch}},
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_mask5_fp16)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_data_gen_renew.py 16 114 32 1 32 128 256 0 0 5 1 0";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles);
    
    OpParam::UnpadFlashAttention opParam;
    opParam.type = batchDynamic ? OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION_DECODER
                                : OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.batchRunStatus = batch_state;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // k
        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {}}, // v
        {TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}},                                     // layer
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch, 1, maxSeq}},                       // mask
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}},                                       // optional alibi coeff
        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {batch}},
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_relayAttention_case1)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    bool isRelay = true;
    std::vector<int> shareIdx;
    for(int i = 0; i < 60; i++) {
        if(i % 3 == 0) {
            shareIdx.push_back(0);
        } else if (i % 2 == 0){
            shareIdx.push_back(1);
        } else if (i % 5 == 0){
            shareIdx.push_back(-1);
        } else {
            shareIdx.push_back(2);
        }
    }
    uint32_t shareLength = 10;
    std::vector<int> shareLen;
    for(int i = 0; i < shareLength; i++){
        shareLen.push_back(1024);
    }
    std::ofstream shareIdxBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_idx.bin", std::ios::binary);
    shareIdxBin.write((const char *)shareIdx.data(), shareIdx.size() * sizeof(int));
    shareIdxBin.close();
    std::ofstream shareLenBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_len.bin", std::ios::binary);
    shareLenBin.write((const char *)shareLen.data(), shareLen.size() * sizeof(int));
    shareLenBin.close();
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_data_gen_renew.py 60 2048 8 1 32 128 2048 0 0 5 1 0 1";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    opTest.shareIdx = shareIdx;
    opTest.shareLen = shareLen;
    opTest.isRelay = isRelay;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir, isRelay, shareLength);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles, shareLength);
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kShareTensorList = opTest.kSepCacheBatchWise;
    opParam.vShareTensorList = opTest.vSepCacheBatchWise;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.kvShareMap = shareIdx;
    opParam.kvShareLen = shareLen;
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_NONE;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}},                       // mask
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_relayAttention_case2)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    bool isRelay = true;
    std::vector<int> shareIdx;
    for(int i = 0; i < 20; i++) {
        if(i % 3 == 0) {
            shareIdx.push_back(0);
        } else if (i % 2 == 0){
            shareIdx.push_back(1);
        } else if (i % 5 == 0){
            shareIdx.push_back(-1);
        } else {
            shareIdx.push_back(2);
        }
    }
    uint32_t shareLength = 10;
    std::vector<int> shareLen;
    for(int i = 0; i < shareLength; i++){
        shareLen.push_back(1024);
    }
    std::ofstream shareIdxBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_idx.bin", std::ios::binary);
    shareIdxBin.write((const char *)shareIdx.data(), shareIdx.size() * sizeof(int));
    shareIdxBin.close();
    std::ofstream shareLenBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_len.bin", std::ios::binary);
    shareLenBin.write((const char *)shareLen.data(), shareLen.size() * sizeof(int));
    shareLenBin.close();
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_data_gen_renew.py 20 2048 1 1 8 128 2048 0 0 5 1 0 1";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    opTest.shareIdx = shareIdx;
    opTest.shareLen = shareLen;
    opTest.isRelay = isRelay;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir, isRelay, shareLength);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles, shareLength);
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kShareTensorList = opTest.kSepCacheBatchWise;
    opParam.vShareTensorList = opTest.vSepCacheBatchWise;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.kvShareMap = shareIdx;
    opParam.kvShareLen = shareLen;
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_NONE;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}},                       // mask
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_datatype_int8_relayAttention)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    bool isRelay = true;
    std::vector<int> shareIdx = {0, 0};
    uint32_t shareLength = 4;
    std::vector<int> shareLen = {100, 100, 100, 100};
    std::ofstream shareIdxBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_idx.bin", std::ios::binary);
    shareIdxBin.write((const char *)shareIdx.data(), shareIdx.size() * sizeof(int));
    shareIdxBin.close();
    std::ofstream shareLenBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_len.bin", std::ios::binary);
    shareLenBin.write((const char *)shareLen.data(), shareLen.size() * sizeof(int));
    shareLenBin.close();
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_data_gen_renew.py 2 128 2 1 2 128 128 0 0 5 1 0 1";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    opTest.shareIdx = shareIdx;
    opTest.shareLen = shareLen;
    opTest.isRelay = isRelay;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir, isRelay, shareLength);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles, shareLength);
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kShareTensorList = opTest.kSepCacheBatchWise;
    opParam.vShareTensorList = opTest.vSepCacheBatchWise;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.kvShareMap = shareIdx;
    opParam.kvShareLen = shareLen;
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_NONE;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}},                       // mask
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), false);
}


TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_masktype_alibi_relayAttention)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    bool isRelay = true;
    std::vector<int> shareIdx = {0, 0};
    uint32_t shareLength = 4;
    std::vector<int> shareLen = {100, 100, 100, 100};
    std::ofstream shareIdxBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_idx.bin", std::ios::binary);
    shareIdxBin.write((const char *)shareIdx.data(), shareIdx.size() * sizeof(int));
    shareIdxBin.close();
    std::ofstream shareLenBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_len.bin", std::ios::binary);
    shareLenBin.write((const char *)shareLen.data(), shareLen.size() * sizeof(int));
    shareLenBin.close();
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_data_gen_renew.py 2 128 2 1 2 128 128 0 0 5 1 0 1";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    opTest.shareIdx = shareIdx;
    opTest.shareLen = shareLen;
    opTest.isRelay = isRelay;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir, isRelay, shareLength);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles, shareLength);
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kShareTensorList = opTest.kSepCacheBatchWise;
    opParam.vShareTensorList = opTest.vSepCacheBatchWise;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.kvShareMap = shareIdx;
    opParam.kvShareLen = shareLen;
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}},                       // mask
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_dimbiggerthan256_relayAttention)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    bool isRelay = true;
    std::vector<int> shareIdx = {0, 0};
    uint32_t shareLength = 4;
    std::vector<int> shareLen = {100, 100, 100, 100};
    std::ofstream shareIdxBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_idx.bin", std::ios::binary);
    shareIdxBin.write((const char *)shareIdx.data(), shareIdx.size() * sizeof(int));
    shareIdxBin.close();
    std::ofstream shareLenBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_len.bin", std::ios::binary);
    shareLenBin.write((const char *)shareLen.data(), shareLen.size() * sizeof(int));
    shareLenBin.close();
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_data_gen_renew.py 2 128 2 1 2 257 128 0 0 5 1 0 1";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    opTest.shareIdx = shareIdx;
    opTest.shareLen = shareLen;
    opTest.isRelay = isRelay;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir, isRelay, shareLength);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles, shareLength);
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kShareTensorList = opTest.kSepCacheBatchWise;
    opParam.vShareTensorList = opTest.vSepCacheBatchWise;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.kvShareMap = shareIdx;
    opParam.kvShareLen = shareLen;
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_NONE;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}},                       // mask
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), false);
}


TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_head0_relayAttention)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    bool isRelay = true;
    std::vector<int> shareIdx = {0, 0};
    uint32_t shareLength = 4;
    std::vector<int> shareLen = {100, 100, 100, 100};
    std::ofstream shareIdxBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_idx.bin", std::ios::binary);
    shareIdxBin.write((const char *)shareIdx.data(), shareIdx.size() * sizeof(int));
    shareIdxBin.close();
    std::ofstream shareLenBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_len.bin", std::ios::binary);
    shareLenBin.write((const char *)shareLen.data(), shareLen.size() * sizeof(int));
    shareLenBin.close();
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_data_gen_renew.py 2 128 2 1 0 128 128 0 0 5 1 0 1";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    opTest.shareIdx = shareIdx;
    opTest.shareLen = shareLen;
    opTest.isRelay = isRelay;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir, isRelay, shareLength);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles, shareLength);
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kShareTensorList = opTest.kSepCacheBatchWise;
    opParam.vShareTensorList = opTest.vSepCacheBatchWise;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.kvShareMap = shareIdx;
    opParam.kvShareLen = shareLen;
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_NONE;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}},                       // mask
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), false);
}


TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_kvtensornullptr_relayAttention)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    bool isRelay = true;
    std::vector<int> shareIdx = {0, 0};
    uint32_t shareLength = 4;
    std::vector<int> shareLen = {100, 100, 100, 100};
    std::ofstream shareIdxBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_idx.bin", std::ios::binary);
    shareIdxBin.write((const char *)shareIdx.data(), shareIdx.size() * sizeof(int));
    shareIdxBin.close();
    std::ofstream shareLenBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_len.bin", std::ios::binary);
    shareLenBin.write((const char *)shareLen.data(), shareLen.size() * sizeof(int));
    shareLenBin.close();
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_data_gen_renew.py 2 128 2 1 2 128 128 0 0 5 1 0 1";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    opTest.shareIdx = shareIdx;
    opTest.shareLen = shareLen;
    opTest.isRelay = isRelay;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir, isRelay, shareLength);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles, shareLength);
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kShareTensorList = opTest.kSepCacheBatchWise;
    opParam.kShareTensorList[0].data = nullptr;
    opParam.vShareTensorList = opTest.vSepCacheBatchWise;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.kvShareMap = shareIdx;
    opParam.kvShareLen = shareLen;
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_NONE;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}},                       // mask
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), false);
}
TEST(TestFlashAttentionNd, TestFlashAttentionNdInferShapeKVHeadMissingDIM2)
{
    int64_t batch = 1;
    int64_t seqlen = 2;
    int64_t headnum = 32;
    int64_t kvhead = 4;
    int64_t edim = 128;
    int64_t layer = 1;
    SVector<Tensor> outTensor;
    outTensor.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch * seqlen, headnum * edim}});
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch * seqlen, headnum * edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {layer, batch, seqlen, kvhead * edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {layer, batch, seqlen, kvhead * edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {seqlen, seqlen}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {seqlen}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch * seqlen, headnum * edim}});
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ND;
    opParam.kvSeqLen = {seqlen};
    opParam.qSeqLen = {seqlen};
    opParam.headSize = headnum;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
 
    Operation *operation = AutoGen::GetOpByName(opDesc.opName);
    ASSERT_EQ(operation != nullptr, true);
    Status status = operation->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}
 
TEST(TestFlashAttentionNd, TestFlashAttentionNdInferShapeKVHeadMissingDIM3)
{
    int64_t batch = 1;
    int64_t seqlen = 2;
    int64_t headnum = 32;
    int64_t kvhead = 4;
    int64_t edim = 128;
    int64_t layer = 1;
    SVector<Tensor> outTensor;
    outTensor.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch * seqlen, headnum, edim}});
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch * seqlen, headnum, edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {layer, batch, seqlen, kvhead, edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {layer, batch, seqlen, kvhead, edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {seqlen, seqlen}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {seqlen}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch * seqlen, headnum, edim}});
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ND;
    opParam.kvSeqLen = {seqlen};
    opParam.qSeqLen = {seqlen};
    opParam.headSize = headnum;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
 
    Operation *operation = AutoGen::GetOpByName(opDesc.opName);
    ASSERT_EQ(operation != nullptr, true);
    Status status = operation->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}
 
TEST(TestFlashAttentionNd, TestFlashAttentionNdInferShapeKVHeadMissingDIM4)
{
    int64_t batch = 1;
    int64_t seqlen = 2;
    int64_t headnum = 32;
    int64_t kvhead = 4;
    int64_t edim = 128;
    int64_t layer = 1;
    SVector<Tensor> outTensor;
    outTensor.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch, headnum, seqlen, edim}});
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch, headnum, seqlen, edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {layer, batch, kvhead, seqlen, edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {layer, batch, kvhead, seqlen, edim}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {1}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {seqlen, seqlen}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {seqlen}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {batch, headnum, seqlen, edim}});
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ND;
    opParam.kvSeqLen = {seqlen};
    opParam.qSeqLen = {seqlen};
    opParam.headSize = headnum;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
 
    Operation *operation = AutoGen::GetOpByName(opDesc.opName);
    ASSERT_EQ(operation != nullptr, true);
    Status status = operation->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_relayAttention_bigbatchkvheadcase1_batch61)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    bool isRelay = true;
    std::vector<int> shareIdx;
    for(int i = 0; i < 61; i++) {
        if(i % 3 == 0) {
            shareIdx.push_back(0);
        } else if (i % 2 == 0){
            shareIdx.push_back(1);
        } else if (i % 5 == 0){
            shareIdx.push_back(-1);
        } else {
            shareIdx.push_back(2);
        }
    }
    uint32_t shareLength = 10;
    std::vector<int> shareLen;
    for(int i = 0; i < shareLength; i++){
        shareLen.push_back(1024);
    }
    std::ofstream shareIdxBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_idx.bin", std::ios::binary);
    shareIdxBin.write((const char *)shareIdx.data(), shareIdx.size() * sizeof(int));
    shareIdxBin.close();
    std::ofstream shareLenBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_len.bin", std::ios::binary);
    shareLenBin.write((const char *)shareLen.data(), shareLen.size() * sizeof(int));
    shareLenBin.close();
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_data_gen_renew.py 61 2048 8 1 32 128 2048 0 0 5 1 0 1";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    opTest.shareIdx = shareIdx;
    opTest.shareLen = shareLen;
    opTest.isRelay = isRelay;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir, isRelay, shareLength);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles, shareLength);
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kShareTensorList = opTest.kSepCacheBatchWise;
    opParam.vShareTensorList = opTest.vSepCacheBatchWise;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.kvShareMap = shareIdx;
    opParam.kvShareLen = shareLen;
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_NONE;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}},                       // mask
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_relayAttention_bigbatchkvheadcase2_kvhead9)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    bool isRelay = true;
    std::vector<int> shareIdx;
    for(int i = 0; i < 60; i++) {
        if(i % 3 == 0) {
            shareIdx.push_back(0);
        } else if (i % 2 == 0){
            shareIdx.push_back(1);
        } else if (i % 5 == 0){
            shareIdx.push_back(-1);
        } else {
            shareIdx.push_back(2);
        }
    }
    uint32_t shareLength = 10;
    std::vector<int> shareLen;
    for(int i = 0; i < shareLength; i++){
        shareLen.push_back(1024);
    }
    std::ofstream shareIdxBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_idx.bin", std::ios::binary);
    shareIdxBin.write((const char *)shareIdx.data(), shareIdx.size() * sizeof(int));
    shareIdxBin.close();
    std::ofstream shareLenBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_len.bin", std::ios::binary);
    shareLenBin.write((const char *)shareLen.data(), shareLen.size() * sizeof(int));
    shareLenBin.close();
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_data_gen_renew.py 60 2048 9 1 36 128 2048 0 0 5 1 0 1";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    opTest.shareIdx = shareIdx;
    opTest.shareLen = shareLen;
    opTest.isRelay = isRelay;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir, isRelay, shareLength);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles, shareLength);
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kShareTensorList = opTest.kSepCacheBatchWise;
    opParam.vShareTensorList = opTest.vSepCacheBatchWise;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.kvShareMap = shareIdx;
    opParam.kvShareLen = shareLen;
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_NONE;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}},                       // mask
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_relayAttention_bigbatchkvheadcase3_kvhead16)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    bool isRelay = true;
    std::vector<int> shareIdx;
    for(int i = 0; i < 60; i++) {
        if(i % 3 == 0) {
            shareIdx.push_back(0);
        } else if (i % 2 == 0){
            shareIdx.push_back(1);
        } else if (i % 5 == 0){
            shareIdx.push_back(-1);
        } else {
            shareIdx.push_back(2);
        }
    }
    uint32_t shareLength = 10;
    std::vector<int> shareLen;
    for(int i = 0; i < shareLength; i++){
        shareLen.push_back(1024);
    }
    std::ofstream shareIdxBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_idx.bin", std::ios::binary);
    shareIdxBin.write((const char *)shareIdx.data(), shareIdx.size() * sizeof(int));
    shareIdxBin.close();
    std::ofstream shareLenBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_len.bin", std::ios::binary);
    shareLenBin.write((const char *)shareLen.data(), shareLen.size() * sizeof(int));
    shareLenBin.close();
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_data_gen_renew.py 60 2048 16 1 32 128 2048 0 0 5 1 0 1";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    opTest.shareIdx = shareIdx;
    opTest.shareLen = shareLen;
    opTest.isRelay = isRelay;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir, isRelay, shareLength);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles, shareLength);
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kShareTensorList = opTest.kSepCacheBatchWise;
    opParam.vShareTensorList = opTest.vSepCacheBatchWise;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.kvShareMap = shareIdx;
    opParam.kvShareLen = shareLen;
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_NONE;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}},                       // mask
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_relayAttention_bigbatchkvheadcase4_batch100_kvhead16)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    bool isRelay = true;
    std::vector<int> shareIdx;
    for(int i = 0; i < 100; i++) {
        if(i % 3 == 0) {
            shareIdx.push_back(0);
        } else if (i % 2 == 0){
            shareIdx.push_back(1);
        } else if (i % 5 == 0){
            shareIdx.push_back(-1);
        } else {
            shareIdx.push_back(2);
        }
    }
    uint32_t shareLength = 10;
    std::vector<int> shareLen;
    for(int i = 0; i < shareLength; i++){
        shareLen.push_back(1024);
    }
    std::ofstream shareIdxBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_idx.bin", std::ios::binary);
    shareIdxBin.write((const char *)shareIdx.data(), shareIdx.size() * sizeof(int));
    shareIdxBin.close();
    std::ofstream shareLenBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_len.bin", std::ios::binary);
    shareLenBin.write((const char *)shareLen.data(), shareLen.size() * sizeof(int));
    shareLenBin.close();
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_data_gen_renew.py 100 2048 16 1 32 128 2048 0 0 5 1 0 1";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    opTest.shareIdx = shareIdx;
    opTest.shareLen = shareLen;
    opTest.isRelay = isRelay;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir, isRelay, shareLength);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles, shareLength);
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kShareTensorList = opTest.kSepCacheBatchWise;
    opParam.vShareTensorList = opTest.vSepCacheBatchWise;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.kvShareMap = shareIdx;
    opParam.kvShareLen = shareLen;
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_NONE;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}},                       // mask
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestFlashAttentionNd, FlashAttentionTestKVPtr_relayAttention_bigbatchkvheadcase5_batch60_kvhead8)
{
    // should generate data first
    CHECK_DEVICE_VERSION_ASCEND910B();
    const char *atbHome = std::getenv("ATB_HOME_PATH");
    ASSERT_NE(atbHome, nullptr);
    std::string atbHomeDir(atbHome);
    std::string subDir = "attention";
    bool isRelay = true;
    std::vector<int> shareIdx;
    for(int i = 0; i < 100; i++) {
        if(i % 3 == 0) {
            shareIdx.push_back(0);
        } else if (i % 2 == 0){
            shareIdx.push_back(1);
        } else if (i % 5 == 0){
            shareIdx.push_back(-1);
        } else {
            shareIdx.push_back(2);
        }
    }
    uint32_t shareLength = 10;
    std::vector<int> shareLen;
    for(int i = 0; i < shareLength; i++){
        shareLen.push_back(1024);
    }
    std::ofstream shareIdxBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_idx.bin", std::ios::binary);
    shareIdxBin.write((const char *)shareIdx.data(), shareIdx.size() * sizeof(int));
    shareIdxBin.close();
    std::ofstream shareLenBin(atbHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/share_len.bin", std::ios::binary);
    shareLenBin.write((const char *)shareLen.data(), shareLen.size() * sizeof(int));
    shareLenBin.close();
    std::string dataGen = "python3 " + atbHomeDir + "/../../../tests/unittest/kernels/mix/" + 
                          subDir + "/data/unpad_FA_data_gen_renew.py 60 2048 8 1 32 128 2048 0 0 5 1 0 1";
    int ret = system(dataGen.c_str());
    ASSERT_EQ(WEXITSTATUS(ret), 0);
    Mki::Test::MkiOpTest opTest;
    std::vector<uint32_t> scalars; // {batch, embd, qHead, kvHead, maxSeq}
    bool batchDynamic = false;
    PrepareScalarParams(scalars, batchDynamic, opTest, atbHomeDir, subDir);
    uint32_t batch = scalars[0];
    uint32_t edim = scalars[1];
    uint32_t qHead = scalars[2];
    uint32_t kvHead = scalars[3];
    uint32_t maxSeq = scalars[4];  
    std::vector<int32_t> q_seqlen(batch, 0);
    std::vector<int32_t> kv_seqlen(batch, 0);
    std::vector<int32_t> batch_state(batch, 0);
    PrepareVectors(q_seqlen, kv_seqlen, batch_state, opTest, atbHomeDir, subDir);
    uint64_t qnTokens = 0;
    for (auto &it : q_seqlen) {
        qnTokens += it;
    }
    std::vector<uint16_t> expect(qnTokens * edim * qHead, 0);
    std::vector<uint32_t> expect_high(qnTokens * edim * qHead, 0);
    PrepareExpect(expect, expect_high, opTest, atbHomeDir, subDir);
    std::string dtype = "fp16";
    opTest.Golden(std::bind(FlashAttentionNdGolden, expect.data(), expect_high.data(), dtype, std::placeholders::_1));
    
    TensorDesc kvTensorDesc = {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {maxSeq, kvHead * edim}};
    std::vector<std::string> kvFiles;
    opTest.shareIdx = shareIdx;
    opTest.shareLen = shareLen;
    opTest.isRelay = isRelay;
    PrepareKvFiles(batch, kvFiles, opTest, atbHomeDir, subDir, isRelay, shareLength);
    opTest.PrepareKVcacheBatchwiseTensors(batch, kvTensorDesc, kvFiles, shareLength);
    OpParam::UnpadFlashAttention opParam;
    opParam.type = OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND;
    opParam.qSeqLen = q_seqlen;
    opParam.kvSeqLen = kv_seqlen;
    opParam.headSize = qHead;
    opParam.kvHead = kvHead;
    opParam.tor = 1;
    opParam.kShareTensorList = opTest.kSepCacheBatchWise;
    opParam.vShareTensorList = opTest.vSepCacheBatchWise;
    opParam.kTensorList = opTest.kCacheBatchWise;
    opParam.vTensorList = opTest.vCacheBatchWise;
    opParam.kvShareMap = shareIdx;
    opParam.kvShareLen = shareLen;
    opParam.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_NONE;
    Mki::Test::UtOpDesc opDesc = {"UnpadFlashAttentionOperation", opParam};
    
    SVector<TensorDesc> inTensorDesc = {
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {qnTokens, qHead * edim}},       // q
        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {}},                       // mask
    };
    SVector<std::string> inDataFiles = Mki::Test::GetInDataFiles(atbHomeDir, subDir);
    Status status = opTest.RunWithDataFileKVPtr(opDesc, inTensorDesc, inDataFiles);
    ASSERT_EQ(status.Ok(), true);
}