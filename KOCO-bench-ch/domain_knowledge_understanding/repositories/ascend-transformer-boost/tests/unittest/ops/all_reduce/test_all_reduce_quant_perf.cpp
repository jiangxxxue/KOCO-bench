/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <thread>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <acl/acl.h>
#include <atb/utils/log.h>
#include "atb/utils.h"
#include <limits.h>
#include "test_utils/test_common.h"
#include "atb/utils/config.h"
#include "atb/operation.h"
#include "atb/infer_op_params.h"
#include "atb/utils/tensor_util.h"
#include "test_utils/operation_test.h"
#include "mki/utils/fp16/fp16_t.h"
#include "atb/utils/singleton.h"
#include <chrono>
#include "atb/svector.h"
#include <random>

const uint64_t MAX_NUMEL = 300000000000;

using namespace atb;
using namespace Mki;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;

atb::Status CreateOffsetTensor(const atb::TensorDesc &tensorDesc, atb::Tensor &tensor)
{
    tensor.desc = tensorDesc;

    std::random_device rd;
    std::default_random_engine eng(rd());
    uint64_t tensorNumel = atb::Utils::GetTensorNumel(tensor);
    ATB_LOG(DEBUG) << "Create Tensor, TensorSize:" << tensorNumel;
    if (tensorNumel == 0 || tensorNumel > MAX_NUMEL) {
        ATB_LOG(ERROR) << "tensorNumel is inValid!";
        return ERROR_INVALID_TENSOR_SIZE;
    }
    tensor.dataSize = tensorNumel * sizeof(Mki::fp16_t);
    tensor.hostData = malloc(tensor.dataSize);
    tensor.deviceData = malloc(tensor.dataSize);
    std::uniform_real_distribution<float> distr(0.0, 0.0);
    Mki::fp16_t *tensorData = static_cast<Mki::fp16_t *>(tensor.hostData);
    for (uint64_t i = 0; i < tensorNumel; i++) {
        tensorData[i] = static_cast<Mki::fp16_t>(distr(eng));
    }
    tensorData = static_cast<Mki::fp16_t *>(tensor.deviceData);
    for (uint64_t i = 0; i < tensorNumel; i++) {
        tensorData[i] = static_cast<Mki::fp16_t>(distr(eng));
    }
    return atb::NO_ERROR;
}

atb::Status CreateOtherTensor(const atb::TensorDesc &tensorDesc, atb::Tensor &tensor)
{
    tensor.desc = tensorDesc;

    std::random_device rd;
    std::default_random_engine eng(rd());
    uint64_t tensorNumel = atb::Utils::GetTensorNumel(tensor);
    ATB_LOG(DEBUG) << "Create Tensor, TensorSize:" << tensorNumel;
    if (tensorNumel == 0 || tensorNumel > MAX_NUMEL) {
        ATB_LOG(ERROR) << "tensorNumel is inValid!";
        return ERROR_INVALID_TENSOR_SIZE;
    }
    if (tensorDesc.dtype == ACL_INT8) {
        tensor.dataSize = tensorNumel * sizeof(int8_t);
        tensor.hostData = malloc(tensor.dataSize);
        tensor.deviceData = malloc(tensor.dataSize);
        std::uniform_int_distribution<int8_t> distr(1, 1);
        int8_t *tensorData = static_cast<int8_t *>(tensor.hostData);
        for (uint64_t i = 0; i < tensorNumel; i++) {
            tensorData[i] = static_cast<int8_t>(distr(eng));
        }
        tensorData = static_cast<int8_t *>(tensor.deviceData);
        for (uint64_t i = 0; i < tensorNumel; i++) {
            tensorData[i] = static_cast<int8_t>(distr(eng));
        }
    } else if (tensorDesc.dtype == ACL_FLOAT16) {
        tensor.dataSize = tensorNumel * sizeof(Mki::fp16_t);
        tensor.hostData = malloc(tensor.dataSize);
        tensor.deviceData = malloc(tensor.dataSize);
        std::uniform_real_distribution<float> distr(1.0, 1.0);
        Mki::fp16_t *tensorData = static_cast<Mki::fp16_t *>(tensor.hostData);
        for (uint64_t i = 0; i < tensorNumel; i++) {
            tensorData[i] = static_cast<Mki::fp16_t>(distr(eng));
        }
        tensorData = static_cast<Mki::fp16_t *>(tensor.deviceData);
        for (uint64_t i = 0; i < tensorNumel; i++) {
            tensorData[i] = static_cast<Mki::fp16_t>(distr(eng));
        }
    }

    return atb::NO_ERROR;
}

atb::Status GenerateQuantRandomTensors(const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::Tensor> &inTensors)
{
    if (inTensorDescs.size() != inTensors.size()) {
        ATB_LOG(ERROR) << "TensorDescs Num not equal Tensors Num!";
        return ERROR_INVALID_TENSOR_SIZE;
    }
    atb::Status st;
    for (size_t i = 0; i < inTensorDescs.size(); i++) {
        if (i == 2) {
            st = CreateOffsetTensor(inTensorDescs.at(i), inTensors.at(i));
        } else {
            st = CreateOtherTensor(inTensorDescs.at(i), inTensors.at(i));
        }
        if (st != atb::NO_ERROR) {
            ATB_LOG(ERROR) << "CreateHostRandTensor failed!";
            return st;
        }
        ATB_LOG(DEBUG) << "inTensor dataSize " << i << ":" << inTensors.at(i).dataSize;
    }
    return atb::NO_ERROR;
}

void FreeInTensorList(atb::SVector<atb::Tensor> &hostTensors)
{
    for (size_t i = 0; i < hostTensors.size(); i++) {
        if (hostTensors.at(i).hostData) {
            free(hostTensors.at(i).hostData);
        }
        if (hostTensors.at(i).deviceData) {
            free(hostTensors.at(i).deviceData);
        }
    }
}

void RunAllReduceOpQuant(int rank, int rankSize, const atb::SVector<atb::TensorDesc> &inTensorDesc, int *result)
{
    *(result) = -1;
    atb::infer::AllReduceParam param;
    param.rank = rank;
    param.rankSize = rankSize;
    param.backend = "lccl";
    param.quantType = atb::infer::AllReduceParam::QUANT_TYPE_PER_TENSOR;
    param.outDataType = ACL_FLOAT16;
    param.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
    atb::Operation *op = nullptr;
    atb::Status st = atb::CreateOperation(param, &op);
    if (st != atb::NO_ERROR || !op) {
        return;
    };
    OperationTest opTest;
    opTest.SetDeviceId(rank);
    opTest.IntRand(1, 1);
    opTest.FloatRand(1.0, 1.0);
    atb::SVector<atb::Tensor> hostInTensors;
    hostInTensors.resize(inTensorDesc.size());
    atb::Status tensorStatus = GenerateQuantRandomTensors(inTensorDesc, hostInTensors);
    if (tensorStatus != atb::NO_ERROR) {
        ATB_LOG(ERROR) << "GenerateQuantRandomTensor failed!";
        return;
    }
    atb::Status status = opTest.RunPerf(op, {hostInTensors});
    if (status != 0) {
        return;
    }
    *(result) = 0;
    FreeInTensorList(hostInTensors);
    DestroyOperation(op);
}

void TestAllReduceThreadQuant(const atb::SVector<atb::TensorDesc> &inTensorDesc, int rankSize)
{
    std::vector<std::thread> threads;
    int results[2] = {-1, -1};
    for (int i = 0; i < rankSize; ++i) {
        threads.push_back(std::thread(RunAllReduceOpQuant, i, rankSize, inTensorDesc, results + i));
    }
    for (auto &t : threads) {
        t.join();
    }
    threads.clear();
    for (int i = 0; i < rankSize; ++i) {
        EXPECT_EQ(results[i], 0);
    }
}

TEST(TestAllReduceOperationPerfQuant, HcclMulitThreadHalf)
{
    if (GetSingleton<Config>().Is310P() || GetSingleton<Config>().Is910A() ||
        GetSingleton<Config>().Is310B()) {
        return;
    }
    const int32_t n = 1;
    atb::SVector<atb::TensorDesc> inTensorDesc = {{ACL_INT8, ACL_FORMAT_ND, {{n, 1024}, 2}},
        {ACL_FLOAT16, ACL_FORMAT_ND, {{1}, 1}},
        {ACL_FLOAT16, ACL_FORMAT_ND, {{1}, 1}}};
    int rankSize = 2;

    TestAllReduceThreadQuant(inTensorDesc, rankSize);
    inTensorDesc = {{ACL_INT8, ACL_FORMAT_ND, {{n, 1024}, 2}},
        {ACL_FLOAT16, ACL_FORMAT_ND, {{1}, 1}},
        {ACL_FLOAT16, ACL_FORMAT_ND, {{1}, 1}}};

    TestAllReduceThreadQuant(inTensorDesc, rankSize);
    inTensorDesc = {{ACL_INT8, ACL_FORMAT_ND, {{n, 1024}, 2}},
        {ACL_FLOAT16, ACL_FORMAT_ND, {{1}, 1}},
        {ACL_FLOAT16, ACL_FORMAT_ND, {{1}, 1}}};
    TestAllReduceThreadQuant(inTensorDesc, rankSize);
}