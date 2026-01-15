/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include <thread>
#include <random>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include "acl/acl.h"
#include "atb/types.h"
#include "atb/atb_infer.h"
#include "atb/operation.h"


#define CHECK_STATUS(status)                                                                                           \
    do {                                                                                                               \
        if ((status) != 0) {                                                                                           \
            std::cout << __FILE__ << ":" << __LINE__ << " [error]: " << (status) << std::endl;                         \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

const int32_t DEV_NUM = 2;

const int32_t M = 2;
const int32_t K = 256;
const int32_t N = 2;

const aclDataType DATA_TYPE = aclDataType::ACL_FLOAT16;

typedef uint16_t float16;
typedef uint16_t bfloat16;

float16 FloatToFloat16(float fp32)
{
    if (fp32 == 0.0f) {
        return (std::signbit(fp32) ? 0x8000 : 0x0000);
    }

    uint32_t float_bits;
    static_assert(sizeof(float) == sizeof(uint32_t), "Float size mismatch");
    std::memcpy(&float_bits, &fp32, sizeof(float));

    const uint32_t sign = (float_bits >> 31) & 0x1;
    const uint32_t exp = (float_bits >> 23) & 0xFF;
    const uint32_t mant = float_bits & 0x7FFFFF;
    if (exp == 0xFF) {
        if (mant == 0) {
            return (sign << 15) | 0x7C00;
        } else {
            return (sign << 15) | 0x7C00 | (mant >> 13);
        }
    }

    int32_t exp_fp16 = static_cast<int32_t>(exp) - 127 + 15;
    if (exp_fp16 <= 0) {
        return (sign << 15);
    }

    if (exp_fp16 >= 0x1F) {
        return (sign < 15) | 0x7C00;
    }

    uint32_t mant24 = (1 << 23) | mant;
    uint32_t round_bits = mant24 & 0x1FFF;
    uint32_t base = (mant24 >> 13) & 0x3FF;

    if (round_bits > 0x1000 || (round_bits == 0x1000 && (base & 1))) {
        base++;
        if (base > 0xFF) {
            base = 0;
            exp_fp16++;
            if (exp_fp16 >= 0x1F) {
                return (sign << 15) | 0x7C00;
            }
        }
    }

    return (sign << 15) | (exp_fp16 << 10) | base;
}

bfloat16 FloatToBfloat16(float fp32)
{
    if (fp32 == 0.0f) {
        return (std::signbit(fp32) ? 0x8000 : 0x0000);
    }

    uint32_t float_bits;
    static_assert(sizeof(float) == sizeof(uint32_t), "Float size mismatch");
    std::memcpy(&float_bits, &fp32, sizeof(float));

    bfloat16 bfloat16_bits = static_cast<bfloat16>(float_bits >> 16);

    const uint32_t exp = (float_bits >> 23) & 0xFF;
    const uint32_t mant = float_bits & 0x7FFFFF;
    if (exp == 0xFF && mant != 0) {
        bfloat16_bits |= 0x01;
    }

    return bfloat16_bits;
}

size_t GetDataItemSize(aclDataType dtype)
{
    switch (dtype) {
        case ACL_DT_UNDEFINED:
            return sizeof(bool);
        case ACL_FLOAT16:
            return sizeof(uint16_t);
        case ACL_BF16:
            return sizeof(uint16_t);
        default:
            return 0;
    }
}

static std::mt19937 gen(0);

template <typename T> T random_float(float min, float max)
{
    std::uniform_real_distribution<T> dist(min, max);
    return dist(gen);
}

atb::Tensor FillTensorDataRandomly(const atb::TensorDesc &desc, float range_min, float range_max)
{
    atb::Tensor tensor{desc, nullptr, nullptr, 0};
    tensor.dataSize = atb::Utils::GetTensorSize(desc);
    aclrtMallocHost((void **)&tensor.hostData, tensor.dataSize);
    {
        size_t dataItemSize = GetDataItemSize(desc.dtype);
        uint64_t tensorNumel = atb::Utils::GetTensorNumel(desc);
        void *basePtr = static_cast<void *>(tensor.hostData);
        for (uint64_t i = 0; i < tensorNumel; ++i) {
            void *elementPtr = static_cast<char *>(basePtr) + i * dataItemSize;
            switch (desc.dtype) {
                case ACL_FLOAT16:
                    *static_cast<uint16_t *>(elementPtr) = FloatToFloat16(random_float<float>(range_min, range_max));
                    break;
                case ACL_BF16:
                    *static_cast<uint16_t *>(elementPtr) = FloatToBfloat16(random_float<float>(range_min, range_max));
                    break;
                default:
                    break;
            }
        }
    }
    aclrtMalloc((void **)&tensor.deviceData, tensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(tensor.deviceData, tensor.dataSize, tensor.hostData, tensor.dataSize, ACL_MEMCPY_HOST_TO_DEVICE);

    return tensor;
}

atb::Status saveTensor(atb::Tensor tensor, std::string path)
{
    if (tensor.deviceData == nullptr) {}
    void *hostData = nullptr;
    aclrtMallocHost((void **)&hostData, tensor.dataSize);
    aclrtMemcpy(hostData, tensor.dataSize, tensor.deviceData, tensor.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    std::ofstream file(path, std::ios::binary);
    file.write(static_cast<char *>(hostData), tensor.dataSize);
    file.close();
    aclrtFreeHost(hostData);
    return atb::ErrorType::NO_ERROR;
}

atb::Status ExcuteImpl(atb::Operation *op, atb::VariantPack variantPack, atb::Context *context, aclrtStream &stream)
{
    uint64_t workspaceSize = 0;
    CHECK_STATUS(op->Setup(variantPack, workspaceSize, context));
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_STATUS(op->Execute(variantPack, (uint8_t *)workspace, workspaceSize, context));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成

    if (workspace) {
        CHECK_STATUS(aclrtFree(workspace)); // 销毁workspace
    }
    return atb::ErrorType::NO_ERROR;
}

atb::Status LinearParallelOneThread(int rank, int rankSize)
{
    int deviceId = rank;
    CHECK_STATUS(aclrtSetDevice(deviceId));
    atb::Context *context = nullptr;
    CHECK_STATUS(atb::CreateContext(&context));
    aclrtStream stream = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);
    
    atb::TensorDesc inputTensorDesc{
        .dtype = DATA_TYPE, .format = aclFormat::ACL_FORMAT_ND, .shape{.dims = {M, K}, .dimNum = 2}};
    atb::Tensor input = FillTensorDataRandomly(inputTensorDesc, -10, 10);

    atb::TensorDesc weightTensorDesc{
        .dtype = DATA_TYPE, .format = aclFormat::ACL_FORMAT_ND, .shape{.dims = {K, N}, .dimNum = 2}};
    atb::Tensor weight = FillTensorDataRandomly(weightTensorDesc, -10, 10);

    atb::Tensor output;
    output.desc.dtype = DATA_TYPE;
    output.desc.format = ACL_FORMAT_ND;
    output.desc.shape.dimNum = 2;
    output.desc.shape.dims[0] = M / DEV_NUM;
    output.desc.shape.dims[1] = N;
    output.dataSize = atb::Utils::GetTensorSize(output);
    CHECK_STATUS(aclrtMalloc(&output.deviceData, output.dataSize, ACL_MEM_MALLOC_HUGE_FIRST));

    atb::infer::LinearParallelParam param;
    param.transWeight = false;
    param.rank = rank;
    param.rankRoot = 0;
    param.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
    param.rankSize = rankSize;
    param.backend = "mc2";
    param.type = atb::infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER;
    atb::Operation *op = nullptr;
    CHECK_STATUS(atb::CreateOperation(param, &op));

    atb::VariantPack variantPack;
    variantPack.inTensors = {input, weight};
    variantPack.outTensors = {output};
    ExcuteImpl(op, variantPack, context, stream);
    std::cout << "rank: " << rank << " executed END." << std::endl;
    saveTensor(input, "rank" + std::to_string(rank) + "_inTensor0.bin");
    saveTensor(weight, "rank" + std::to_string(rank) + "_inTensor1.bin");
    saveTensor(output, "rank" + std::to_string(rank) + "_outTensor0.bin");
    // 资源释放
    CHECK_STATUS(atb::DestroyOperation(op));    // 销毁op对象
    CHECK_STATUS(aclrtDestroyStream(stream));   // 销毁stream
    CHECK_STATUS(atb::DestroyContext(context)); // 销毁context
    return atb::ErrorType::NO_ERROR;
}

int main(int argc, const char *argv[])
{
    int ret = aclInit(nullptr);

    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (size_t i = 0; i < DEV_NUM; i++) {
        threads[i].reset(new (std::nothrow) std::thread(LinearParallelOneThread, i, DEV_NUM));
    }
    for (size_t i = 0; i < DEV_NUM; ++i) {
        threads[i]->join();
    }

    CHECK_STATUS(aclFinalize());
    return 0;
}
