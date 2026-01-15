/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <ATen/ATen.h>
#include <mki/utils/status/status.h>
#include "op_test_tools.h"

namespace Mki {
namespace Test {
float GetMare(float *actual, float *golden, uint32_t dataLen) 
{
    float mare = 0;
    for (uint32_t i = 0; i < dataLen; i++) {
        float absErr = std::abs(actual[i] - golden[i]) / (std::abs(golden[i]) + pow(10, -7));
        mare = std::max(absErr, mare);
    }
    return mare;
}

float GetMere(float *actual, float *golden, uint32_t dataLen)
{
    float mere = 0;
    for (uint32_t i = 0; i < dataLen; i++) {
        float absErr = std::abs(actual[i] - golden[i]) / (std::abs(golden[i]) + pow(10, -7));
        mere += absErr;
    }
    mere /= dataLen;
    return mere;
}

float GetRmse(float *actual, float *golden, uint32_t dataLen)
{
    float sum = 0;
    for (uint32_t i = 0; i < dataLen; i++) {
        float sqrErr = pow(std::abs(actual[i] - golden[i]), 2);
        sum += sqrErr;
    }
    float rmse = std::sqrt(sum / dataLen);
    return rmse;
}

float GetEb(float *actual, float *golden, uint32_t dataLen)
{
    float eb = 0;
    for (uint32_t i = 0; i < dataLen; i++) {
        float actErr = actual[i] - golden[i];
        float goldenNMax = std::max(golden[i], (float)1.0);
        eb += actErr / goldenNMax;
    }
    eb /= dataLen;
    return eb;
}

Status MatchAtTensor(at::Tensor &out, at::Tensor &gt, at::Tensor &gt_high, std::string &dtype)
{
    float *result = static_cast<float *>(out.storage().data_ptr().get());
    float *expect = static_cast<float *>(gt.storage().data_ptr().get());
    float *expect_high = static_cast<float *>(gt_high.storage().data_ptr().get());
    float errThreshold = dtype == "fp16" ? pow(2, -11) : pow(2, -8);
    float ebThreshold = dtype == "fp16" ? pow(2, -10) : pow(2, -7);
    uint32_t length = out.numel(); 
    float mareNpu = GetMare(result, expect_high, length);
    float mareCpu = GetMare(expect, expect_high, length);
    float mereNpu = GetMere(result, expect_high, length);
    float mereCpu = GetMere(expect, expect_high, length);
    float rmseNpu = GetRmse(result, expect_high, length);
    float rmseCpu = GetRmse(expect, expect_high, length);
    float ebNpu = GetEb(result, expect_high, length);

    float mareRate = mareNpu / std::max(mareCpu, errThreshold);
    float mereRate = mereNpu / std::max(mereCpu, errThreshold);
    float rmseRate = rmseNpu / std::max(rmseCpu, errThreshold);

    if (mareRate < 10 && mereRate < 2 && rmseRate < 2 && ebNpu < ebThreshold) {
        return Status::OkStatus();
    }
    std::string msg = "acc is wrong, mareRate is:" + std::to_string(mareRate) 
                    + " mereRate is:" + std::to_string(mereRate)
                    + " rmseRate is:" + std::to_string(rmseRate)
                    + "eb is:" + std::to_string(ebNpu);
    return Status::FailStatus(-1, msg);
}

void PrepareScalarParams(std::vector<uint32_t> &scalars, bool &dynamicBatch, MkiOpTest &opTest,
                         const std::string &atbopsHomeDir, const std::string &subDir)
{
    std::vector<uint32_t> batch_num(1, 0);
    std::vector<uint32_t> embd_dim(1, 0);
    std::vector<uint32_t> q_heads(1, 0);
    std::vector<uint32_t> kv_heads(1, 0);
    std::vector<uint32_t> max_seqlen(1, 0);
    std::vector<uint32_t> is_batch_dynamic(1, 0);
    opTest.ReadFile(batch_num.data(), batch_num.size() * sizeof(uint32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/batch_num.bin");
    opTest.ReadFile(embd_dim.data(), embd_dim.size() * sizeof(uint32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/embd_dim.bin");
    opTest.ReadFile(q_heads.data(), q_heads.size() * sizeof(uint32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/q_heads.bin");
    opTest.ReadFile(kv_heads.data(), kv_heads.size() * sizeof(uint32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/kv_heads.bin");
    opTest.ReadFile(max_seqlen.data(), max_seqlen.size() * sizeof(uint32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/max_seqlen.bin");
    opTest.ReadFile(is_batch_dynamic.data(), is_batch_dynamic.size() * sizeof(uint32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/is_batch_dynamic.bin");
    scalars.push_back(batch_num[0]);
    scalars.push_back(embd_dim[0]);
    scalars.push_back(q_heads[0]);
    scalars.push_back(kv_heads[0]);
    scalars.push_back(max_seqlen[0]);
    dynamicBatch = (bool)is_batch_dynamic[0];
}

void PrepareScalarParams(std::vector<uint32_t> &scalars, MkiOpTest &opTest, const std::string &atbopsHomeDir,
                         const std::string &subDir)
{
    std::vector<uint32_t> batch_num(1, 0);
    std::vector<uint32_t> embd_dim(1, 0);
    std::vector<uint32_t> q_heads(1, 0);
    std::vector<uint32_t> kv_heads(1, 0);
    std::vector<uint32_t> max_seqlen(1, 0);
    std::vector<uint32_t> is_batch_dynamic(1, 0);
    opTest.ReadFile(batch_num.data(), batch_num.size() * sizeof(uint32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/batch_num.bin");
    opTest.ReadFile(embd_dim.data(), embd_dim.size() * sizeof(uint32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/embd_dim.bin");
    opTest.ReadFile(q_heads.data(), q_heads.size() * sizeof(uint32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/q_heads.bin");
    opTest.ReadFile(kv_heads.data(), kv_heads.size() * sizeof(uint32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/kv_heads.bin");
    opTest.ReadFile(max_seqlen.data(), max_seqlen.size() * sizeof(uint32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/max_seqlen.bin");
    scalars.push_back(batch_num[0]);
    scalars.push_back(embd_dim[0]);
    scalars.push_back(q_heads[0]);
    scalars.push_back(kv_heads[0]);
    scalars.push_back(max_seqlen[0]);
}

void PrepareVectors(std::vector<int32_t> &q_seqlen, std::vector<int32_t> &kv_seqlen, std::vector<int32_t> &batch_state,
                    MkiOpTest &opTest, const std::string &atbopsHomeDir, const std::string &subDir)
{
    opTest.ReadFile(q_seqlen.data(), q_seqlen.size() * sizeof(int32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/q_seqlen.bin");
    opTest.ReadFile(kv_seqlen.data(), kv_seqlen.size() * sizeof(int32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/kv_seqlen.bin");
    opTest.ReadFile(batch_state.data(), batch_state.size() * sizeof(int32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/batchRunStatus.bin");
}

void PrepareVectors(std::vector<int32_t> &q_seqlen, std::vector<int32_t> &kv_seqlen, MkiOpTest &opTest,
                    const std::string &atbopsHomeDir, const std::string &subDir)
{
    opTest.ReadFile(q_seqlen.data(), q_seqlen.size() * sizeof(int32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/q_seqlen.bin");
    opTest.ReadFile(kv_seqlen.data(), kv_seqlen.size() * sizeof(int32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/kv_seqlen.bin");
}

void PrepareExpect(std::vector<uint16_t> &expect, std::vector<uint32_t> &expect_high, MkiOpTest &opTest,
                   const std::string &atbopsHomeDir, const std::string &subDir)
{
    opTest.ReadFile(expect.data(), expect.size() * sizeof(uint16_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/expect.bin");
    opTest.ReadFile(expect_high.data(), expect_high.size() * sizeof(uint32_t),
                    atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/expect_high.bin");
}

void PrepareKvFiles(uint32_t batch, std::vector<std::string> &kvFiles, MkiOpTest &opTest, const std::string &atbopsHomeDir,
                    const std::string &subDir, bool isRelay, uint32_t share_len)
{
    for (size_t i = 0; i < batch; ++i) {
        std::string file = atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/kvBatches/input2_" +
                           std::to_string(i) + ".bin";
        kvFiles.push_back(file);
    }
    for (size_t i = 0; i < batch; ++i) {
        std::string file = atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/kvBatches/input3_" +
                           std::to_string(i) + ".bin";
        kvFiles.push_back(file);
    }
    if (isRelay) {
        for (size_t i = 0; i < share_len; ++i) {
            std::string file = atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/kvBatches/share_input2_" +
                               std::to_string(i) + ".bin";
            kvFiles.push_back(file);
        }
        for (size_t i = 0; i < share_len; ++i) {
            std::string file = atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/kvBatches/share_input3_" +
                               std::to_string(i) + ".bin";
            kvFiles.push_back(file);
        }
    }
}

SVector<std::string> GetInDataFiles(const std::string &atbopsHomeDir, const std::string &subDir, bool isRelay)
{
    SVector<std::string> inDataFiles;
    if (isRelay) {
        inDataFiles = {
            atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/input1.bin",
            atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/input4.bin",
        };
    } else {
        inDataFiles = {
            atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/input1.bin",
            atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/input2.bin",
            atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/input3.bin",
            atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/layerId.bin",
            atbopsHomeDir + "/../../../tests/unittest/kernels/mix/" + subDir + "/data/input4.bin",
        };
    }
    
    return inDataFiles;
}
} // namespace Test
} // namespace Mki