/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <random>

#include "../demo_util.h"

uint32_t NUM_TOKENS = 1;
uint32_t NUM_HEAD = 1;
uint32_t K_HEAD_SIZE = 128;
uint32_t V_HEAD_SIZE = K_HEAD_SIZE;
uint32_t NUM_BLOCKS = 9;
uint32_t BLOCK_SIZE = 128;

/**
 * @brief 准备随机输入tensorK或输入tensorV的内容
 * @param kvflag 0:key; 1:value
 * @return 输入tensorK或输入tensorV
 */
std::vector<float> KvGeneration(bool kvflag)
{
    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    // 定义随机数分布范围
    std::uniform_real_distribution<> dis(-100.0, 100.0);
    // 定义要生成的随机数的个数
    size_t num_elements = kvflag ? NUM_TOKENS * NUM_HEAD * V_HEAD_SIZE : NUM_TOKENS * NUM_HEAD * K_HEAD_SIZE;
    // 创建一个 vector 并填充随机数
    std::vector<float> intensorKV;
    for (size_t i = 0; i < num_elements; ++i) {
        intensorKV.push_back(dis(gen));
    }
    return intensorKV;
}

/**
 * @brief 准备随机输入tensorSlotmapping的内容
 * @param slotRange 数据生成范围
 * @param num_tokens Slotmapping length
 * @return 输入tensorSlotmapping
 */
std::vector<int32_t> SlotmappingGeneration(int slotRange, size_t num_tokens)
{
    // 创建一个包含范围 [-slotRange, slotRange] 的所有整数的 vector
    std::vector<int32_t> all_numbers;
    for (int i = -slotRange; i <= slotRange; ++i) {
        all_numbers.push_back(i);
    }
    // 检查 num_tokens 是否超过范围内的整数数量
    if (num_tokens > all_numbers.size()) {
        throw std::invalid_argument("num_tokens exceeds the range of unique numbers available");
    }
    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    // 打乱所有整数
    std::shuffle(all_numbers.begin(), all_numbers.end(), gen);
    // 选取前 num_tokens 个元素
    std::vector<int32_t> slotmapping(all_numbers.begin(), all_numbers.begin() + num_tokens);
    return slotmapping;
}

/**
 * @brief 准备atb::VariantPack
 * @param contextPtr context指针
 * @param stream stream
 * @param variantPack atb::VariantPack
 * @return atb::Status 错误码
 */
atb::Status PrepareVariantPack(atb::Context *contextPtr, aclrtStream stream, atb::VariantPack &variantPack)
{
    // 创建key，value tensor
    std::vector<float> keyData = KvGeneration(false);
    atb::Tensor tensorKey;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, keyData, ACL_BF16, aclFormat::ACL_FORMAT_ND,
                                        {NUM_TOKENS, NUM_HEAD, K_HEAD_SIZE}, tensorKey));
    std::vector<float> valueData = KvGeneration(true);
    atb::Tensor tensorValue;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, valueData, ACL_BF16, aclFormat::ACL_FORMAT_ND,
                                        {NUM_TOKENS, NUM_HEAD, V_HEAD_SIZE}, tensorValue));
    // 创建kvCache tensor
    std::vector<float> kCacheData(NUM_BLOCKS * BLOCK_SIZE * NUM_HEAD * K_HEAD_SIZE, 0);
    atb::Tensor tensorKCache;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, kCacheData, ACL_BF16, aclFormat::ACL_FORMAT_ND,
                                        {NUM_BLOCKS, BLOCK_SIZE, NUM_HEAD, K_HEAD_SIZE}, tensorKCache));
    std::vector<float> vCacheData(NUM_BLOCKS * BLOCK_SIZE * NUM_HEAD * V_HEAD_SIZE, 0);
    atb::Tensor tensorVCache;
    CHECK_STATUS(CreateTensorFromVector(contextPtr, stream, vCacheData, ACL_BF16, aclFormat::ACL_FORMAT_ND,
                                        {NUM_BLOCKS, BLOCK_SIZE, NUM_HEAD, V_HEAD_SIZE}, tensorVCache));
    // 创建SlotMapping
    std::vector<int32_t> slotMappingData = SlotmappingGeneration(NUM_BLOCKS * BLOCK_SIZE, NUM_TOKENS);
    atb::Tensor tensorSlotMapping;
    CHECK_STATUS(CreateTensor(ACL_INT32, aclFormat::ACL_FORMAT_ND, {NUM_TOKENS}, tensorSlotMapping));
    CHECK_STATUS(aclrtMemcpy(tensorSlotMapping.deviceData, tensorSlotMapping.dataSize, slotMappingData.data(),
                             sizeof(int32_t) * slotMappingData.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    // 根据顺序将所有输入tensor放入SVector
    atb::SVector<atb::Tensor> inTensors = {tensorKey, tensorValue, tensorKCache, tensorVCache, tensorSlotMapping};
    // 准备输入张量
    variantPack.inTensors = inTensors;                     // 放入输入tensor
    variantPack.outTensors = {tensorKCache, tensorVCache}; // 放入输出tensor
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 创建一个Reshape and Cache的Operation，并设置参数
 * @param reshapeAndCacheOp 创建一个Operation指针
 * @return atb::Status 错误码
 */
atb::Status PrepareOperation(atb::Operation **reshapeAndCacheOp)
{
    atb::infer::ReshapeAndCacheParam opParam;
    opParam.compressType = atb::infer::ReshapeAndCacheParam::CompressType::COMPRESS_TYPE_UNDEFINED;
    opParam.kvCacheCfg = atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE;
    return atb::CreateOperation(opParam, reshapeAndCacheOp);
}

int main(int argc, char **argv)
{
    // 设置卡号、创建context、设置stream
    CHECK_STATUS(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_STATUS(aclrtSetDevice(deviceId));
    atb::Context *context = nullptr;
    CHECK_STATUS(atb::CreateContext(&context));
    void *stream = nullptr;
    CHECK_STATUS(aclrtCreateStream(&stream));
    context->SetExecuteStream(stream);

    // RAC示例
    atb::Operation *reshapeAndCacheOp = nullptr;
    PrepareOperation(&reshapeAndCacheOp);
    // 准备variantPack
    atb::VariantPack variantPack;
    CHECK_STATUS(PrepareVariantPack(context, stream, variantPack));
    uint64_t workspaceSize = 0;
    // 对输入tensor和输出tensor进行校验
    CHECK_STATUS(reshapeAndCacheOp->Setup(variantPack, workspaceSize, context));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // RAC执行
    CHECK_STATUS(reshapeAndCacheOp->Execute(variantPack, workspacePtr, workspaceSize, context));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    for (atb::Tensor &inTensor : variantPack.inTensors) {
        CHECK_STATUS(aclrtFree(inTensor.deviceData));
    }
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspacePtr));
    }
    // 资源释放
    CHECK_STATUS(atb::DestroyOperation(reshapeAndCacheOp)); // operation，对象概念，先释放
    CHECK_STATUS(aclrtDestroyStream(stream));
    CHECK_STATUS(atb::DestroyContext(context)); // context，全局资源，后释放
    CHECK_STATUS((aclFinalize()));
    std::cout << "Reshape and Cache demo success!" << std::endl;
    return 0;
}
