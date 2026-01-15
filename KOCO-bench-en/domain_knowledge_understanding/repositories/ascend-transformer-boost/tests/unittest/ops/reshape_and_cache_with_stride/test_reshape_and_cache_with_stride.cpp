/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <iostream>
#include <vector>
#include <numeric>
#include "acl/acl.h"
#include "atb/operation.h"
#include "atb/types.h"
#include "atb/atb_infer.h"
#include <random>
#include <algorithm>
#include <unordered_set>
#include <cmath>
#include <gtest/gtest.h>
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"

const int NUM_TOKENS = 2;
const int NUM_HEAD = 32;
const int K_HEAD_SIZE = 128;
const int V_HEAD_SIZE = K_HEAD_SIZE;
const int NUM_BLOCKS = 2;
const int BLOCK_SIZE = 2;
std::vector<int64_t> const keyStrides_{NUM_HEAD * K_HEAD_SIZE, K_HEAD_SIZE, 1};
std::vector<int64_t> const valueStrides_{NUM_HEAD * V_HEAD_SIZE, K_HEAD_SIZE, 1};
const int64_t keyOffset_{5};
const int64_t valueOffset_{5};
bool NO_NEED_CAL = false;
// 全局变量key_expect_siso便于golden compare
std::vector<std::vector<std::vector<std::vector<float>>>> key_expect_siso(
    NUM_BLOCKS, std::vector<std::vector<std::vector<float>>>(
        BLOCK_SIZE, std::vector<std::vector<float>>(
            NUM_HEAD, std::vector<float>(K_HEAD_SIZE)
        )
    )
);
// 全局变量key_expect_nd便于golden compare
std::vector<std::vector<std::vector<std::vector<float>>>> key_expect_nd(
    NUM_BLOCKS, std::vector<std::vector<std::vector<float>>>(
        BLOCK_SIZE, std::vector<std::vector<float>>(
            NUM_HEAD, std::vector<float>(K_HEAD_SIZE)
        )
    )
);
// 全局变量value_expect便于golden compare
std::vector<std::vector<std::vector<std::vector<float>>>> value_expect(
    NUM_BLOCKS, std::vector<std::vector<std::vector<float>>>(
        BLOCK_SIZE, std::vector<std::vector<float>>(
            NUM_HEAD, std::vector<float>(V_HEAD_SIZE)
        )
    )
);

/**
 * @brief 创建一个Tensor对象
 * @param  dataType 数据类型
 * @param  format 数据格式
 * @param  shape 数据shape
 * @return atb::Tensor 返回创建的Tensor对象
 */
atb::Tensor CreateTensor(const aclDataType dataType, const aclFormat format, std::vector<int64_t> bigShape, const std::vector<int64_t> smallShape)
{
    atb::Tensor tensor;
    tensor.desc.dtype = dataType;
    tensor.desc.format = format;
    tensor.desc.shape.dimNum = bigShape.size();
    // tensor的dim依次设置为shape中元素
    for (size_t i = 0; i < bigShape.size(); i++) {
        tensor.desc.shape.dims[i] = bigShape.at(i);
    }
    tensor.dataSize = atb::Utils::GetTensorSize(tensor);  // 计算Tensor的数据大小

    aclrtMalloc(&tensor.deviceData, tensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    return tensor;
}

/**
 * @brief 进行数据类型转换，调用Elewise的cast Op
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensor 输入tensor
 * @param outTensorType 输出tensor的数据类型
 * @param shape 输出tensor的shape
 * @return atb::Tensor 转换后的tensor
 */
atb::Tensor CastOp(atb::Context *contextPtr, aclrtStream stream, const atb::Tensor inTensor,
    const aclDataType outTensorType, const std::vector<int64_t> shape)
{
    uint64_t workspaceSize = 0;
    void *workspace = nullptr;
    // 创建Elewise的ELEWISE_CAST
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ELEWISE_CAST;
    castParam.outTensorType = outTensorType;
    atb::Operation *castOp = nullptr;
    CreateOperation(castParam, &castOp);
    atb::Tensor outTensor = CreateTensor(outTensorType, ACL_FORMAT_ND, shape, shape);  // cast输出tensor
    atb::VariantPack castVariantPack;                                           // 参数包
    castVariantPack.inTensors = {inTensor};
    castVariantPack.outTensors = {outTensor};
    // 在Setup接口调用时对输入Tensor和输出Tensor进行校验。
    castOp->Setup(castVariantPack, workspaceSize, contextPtr);
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    // ELEWISE_CAST执行
    castOp->Execute(castVariantPack, (uint8_t *)workspace, workspaceSize, contextPtr);
    aclrtSynchronizeStream(stream);  // 流同步，等待device侧任务计算完成


    if (workspaceSize > 0) {
        aclrtFree(workspace);  // 清理工作空间
    }
    return outTensor;
}

/**
 * @brief 简单封装，拷贝vector data中数据以创建tensor
 * @details 用于创建ACL_FLOAT16类型tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param data 输入vector数据
 * @param outTensorType 期望输出tensor数据类型
 * @param format 输出tensor的格式，即NZ，ND等
 * @param shape 输出tensor的shape
 * @return atb::Tensor 返回创建的tensor
 */
template <typename T> atb::Tensor CreateTensorFromVector(atb::Context *contextPtr, aclrtStream stream, std::vector<T> data,
    const aclDataType outTensorType, const aclFormat format, const std::vector<int64_t> bigShape, const std::vector<int64_t> smallShape)
{
    if (outTensorType == ACL_FLOAT16 || outTensorType == ACL_BF16) {
        atb::Tensor tensor = CreateTensor(ACL_FLOAT, format, bigShape, bigShape);
        std::cout << "tensor.dataSize:" << tensor.dataSize << std::endl;
        std::cout << "sizeof(T):" << sizeof(T) << std::endl;
        std::cout << " data.size():" <<  data.size() << std::endl;
        aclrtMemcpy(
            tensor.deviceData, tensor.dataSize, data.data(), sizeof(T) * data.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        // return tensor;

        atb::Tensor out = CastOp(contextPtr, stream, tensor, outTensorType, bigShape);
        //改变dims
        for (size_t i = 0; i < smallShape.size(); i++) {
            out.desc.shape.dims[i] = smallShape.at(i);
        }
        return out;
    } 
    return atb::Tensor{};
}

/**
 * @brief 创建一个Tensor对象
 * @param  dataType 数据类型
 * @param  format 数据格式
 * @param  shape 数据shape
 * @return atb::Tensor 返回创建的Tensor对象
 */
atb::Tensor CreateTensorOrigin(const aclDataType dataType, const aclFormat format, const std::vector<int64_t> shape)
{
    atb::Tensor tensor;
    tensor.desc.dtype = dataType;
    tensor.desc.format = format;
    tensor.desc.shape.dimNum = shape.size();
    
    // tensor的dim依次设置为shape中元素
    for (size_t i = 0; i < shape.size(); i++) {
        tensor.desc.shape.dims[i] = shape.at(i);
    }
    tensor.dataSize = atb::Utils::GetTensorSize(tensor);  // 计算Tensor的数据大小
    aclrtMalloc(&tensor.deviceData, tensor.dataSize, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
    return tensor;
}

/**
 * @brief 简单封装，拷贝vector data中数据以创建tensor
 * @details 用于创建outTensorType类型的tensor
 * @param contextPtr context指针
 * @param stream stream
 * @param data 输入vector数据
 * @param outTensorType 期望输出tensor数据类型
 * @param format 输出tensor的格式，即NZ，ND等
 * @param shape 输出tensor的shape
 * @return atb::Tensor 返回创建的tensor
 */
template <typename T> atb::Tensor CreateTensorFromVectorOrigin(atb::Context *contextPtr, aclrtStream stream, std::vector<T> data,
    const aclDataType outTensorType, const aclFormat format, const std::vector<int64_t> shape)
{
    atb::Tensor tensor;
    aclDataType intermediateType;
    switch (outTensorType) {
        case aclDataType::ACL_FLOAT:
        case aclDataType::ACL_FLOAT16:
        case aclDataType::ACL_BF16:
            intermediateType = aclDataType::ACL_FLOAT;
            break;
        case aclDataType::ACL_INT32:
        case aclDataType::ACL_INT64:
            intermediateType = aclDataType::ACL_INT32;
            break;
        default:
            intermediateType = outTensorType;
    }
    tensor = CreateTensorOrigin(intermediateType, format, shape);
    aclrtMemcpy(
        tensor.deviceData, tensor.dataSize, data.data(), sizeof(T) * data.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    if (intermediateType == outTensorType) {
        // 原始创建的tensor类型，不需要转换
        return tensor;
    }
    return CastOp(contextPtr, stream, tensor, outTensorType, shape);
}

std::vector<float> create_non_contiguous_tensor(const std::vector<float>& data, const std::vector<int>& shape, const std::vector<int64_t>& strides, const int64_t offset) {
    // 计算目标张量的大小
    int target_size = 1;
    for (int dim : shape) {
        target_size *= dim;
    }
    std::vector<float> result(target_size);
    // 计算并提取非连续索引元素
    for (int idx = 0; idx < target_size; ++idx) {
        // 将 idx 转换为目标形状的多维索引
        std::vector<int> multi_idx(shape.size());
        int temp_idx = idx;
        for (int i = shape.size() - 1; i >= 0; --i) {
            multi_idx[i] = temp_idx % shape[i];
            temp_idx /= shape[i];
        }
        // 计算从原始一维数组中获取的元素的索引
        int64_t linear_idx = offset;
        // 按顺序通过每个维度的 stride 计算 linear_idx
        for (size_t i = 0; i < multi_idx.size(); ++i) {
            linear_idx += multi_idx[i] * strides[i];
        }
        // 打印当前的 linear_idx 值
        // 从原始数据中取出这些元素
        result[idx] = data[linear_idx];
    }
    return result;
}

/**
 * @brief 从长tensor中按最后两维reshape，类似于view(-1, NUM_HEAD, HEAD_SIZE)，并按index从中取vector
 * @param tensor 输入tensor
 * @param tensor_shape 输入tensor_shape
 * @param sub_tensor_shape 目标sub_tensor_shape
 * @param index 第index个目标维度的tensor
 * @return std::vector<std::vector<float>> 返回一个2维的vector
 */
std::vector<std::vector<float>> ExtractTensor(const std::vector<float>& tensor,
                                               const std::vector<int>& tensor_shape,
                                               const std::vector<int>& sub_tensor_shape,
                                               int index)
{
    std::vector<std::vector<float>> sub_tensor;
    int sub_tensor_size = std::accumulate(sub_tensor_shape.begin(), sub_tensor_shape.end(), 1, std::multiplies<int>());
    int rows = sub_tensor_shape[0];
    int cols = sub_tensor_shape[1];
    // 计算提取子张量的起始位置
    int start_idx = index * rows * cols;
    // 逐行提取数据，填充到二维子张量
    for (int i = 0; i < rows; ++i) {
        std::vector<float> row;
        for (int j = 0; j < cols; ++j) {
            row.push_back(tensor[start_idx + i * cols + j]);
        }
        sub_tensor.push_back(row);
    }
    return sub_tensor;
}

/**
 * @brief 将1个4维vector展成1维
 * @return std::vector<float> 返回一个1维vector
 */
std::vector<float> Flatten4dTo1d(const std::vector<std::vector<std::vector<std::vector<float>>>>& tensor)
{
    std::vector<float> flattened;
    for (const auto& block : tensor) {
        for (const auto& block_row : block) {
            for (const auto& head : block_row) {
                for (const auto& val : head) {
                    flattened.push_back(val);
                }
            }
        }
    }
    return flattened;
}
/*=======================================================SISO=================================================*/
/**
 * @brief 准备随机intesorK或intensorV的内容
 * @param 1. SISO: kORv-true && siso-true;2.ND 2.1 key:kORv-true && siso-false
 * @return intesorK或intensorV
 */
std::vector<float> KvGeneration(bool siso, bool kORv)
{
    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    // 定义随机数分布范围
    std::uniform_real_distribution<> dis(-100.0, 100.0);
    // 定义要生成的随机数的个数
    size_t num_elements = 0;
    if (siso || kORv) {
        num_elements = NUM_TOKENS * NUM_HEAD * K_HEAD_SIZE * 2;
    } else {
        num_elements = NUM_TOKENS * NUM_HEAD * V_HEAD_SIZE * 2;
    }
    // 创建一个 vector 并填充随机数
    std::vector<float> intesorK;
    for (size_t i = 0; i < num_elements; ++i) {
        intesorK.push_back(dis(gen));
    }
    return intesorK;
}

/**
 * @brief 准备随机intesorSlotmapping的内容
 * @param slotRange 数据生成范围
 * @param num_tokens Slotmapping length
 * @return intesorSlotmapping
 */
std::vector<int32_t> SlotmappingGeneration(int slotRange, size_t num_tokens)
{
    // 创建一个包含范围 [-slotRange, slotRange] 的所有整数的 vector
    std::vector<int32_t> all_numbers;
    for (int i = -slotRange+1; i < slotRange; ++i) {
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
    int minus_slot_counter = 0;
    std::vector<int32_t> slotmapping(all_numbers.begin(), all_numbers.begin() + num_tokens);
    for (const auto &element : slotmapping) {
        if (element < 0) {
            minus_slot_counter++;
        }
        std::cout << "slotmapping: " << element << std::endl;
    }
    if (minus_slot_counter == num_tokens) {
        NO_NEED_CAL = true;
    }
    return slotmapping;
}

/**
 * @brief 准备atb::VariantPack
 * @param contextPtr context指针
 * @param stream stream
 * @param keyStrides_ 
 * @param keyOffset_ 
 * @return atb::VariantPack
 */
atb::VariantPack PrepareVariantPackSISO(atb::Context *contextPtr, aclrtStream stream, const std::vector<int64_t> &keyStrides_,
    const int64_t keyOffset_)
{
    // 创建key tensor
    std::vector<float> keyData = KvGeneration(true, false);
    atb::Tensor tensorKey = CreateTensorFromVector(
        contextPtr, stream, keyData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NUM_TOKENS, NUM_HEAD, K_HEAD_SIZE*2}, {NUM_TOKENS, NUM_HEAD, K_HEAD_SIZE});
    // 创建kvCache tensor
    std::vector<float> kCacheData(NUM_BLOCKS * BLOCK_SIZE * NUM_HEAD * K_HEAD_SIZE, 0);
    atb::Tensor tensorKCache = CreateTensorFromVectorOrigin(
        contextPtr, stream, kCacheData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NUM_BLOCKS, BLOCK_SIZE, NUM_HEAD, K_HEAD_SIZE});
    // 创建SlotMapping
    std::vector<int32_t> slotMappingData = SlotmappingGeneration(NUM_BLOCKS * BLOCK_SIZE, NUM_TOKENS);
    atb::Tensor tensorSlotMapping = CreateTensorOrigin(ACL_INT32, aclFormat::ACL_FORMAT_ND, {NUM_TOKENS});
    aclrtMemcpy(tensorSlotMapping.deviceData,
        tensorSlotMapping.dataSize,
        slotMappingData.data(),
        sizeof(int32_t) * slotMappingData.size(),
        ACL_MEMCPY_HOST_TO_DEVICE);
    // 创建keyStrides
    atb::Tensor keyStrides = CreateTensor(ACL_INT64, aclFormat::ACL_FORMAT_ND, {3}, {3});
    aclrtMallocHost(&keyStrides.hostData, keyStrides.dataSize);
    for (int i = 0; i < 3; i++) {
        static_cast<int64_t*>(keyStrides.hostData)[i] = keyStrides_[i];
    }
    // 创建keyOffset\valueOffset
    atb::Tensor keyOffset = CreateTensor(ACL_INT64, aclFormat::ACL_FORMAT_ND, {1}, {1});
    aclrtMallocHost(&keyOffset.hostData, keyOffset.dataSize);
    static_cast<int64_t*>(keyOffset.hostData)[0] = keyOffset_;
    // 根据strides和offset获取expect key、value
    std::vector<float> target_key = create_non_contiguous_tensor(keyData, {NUM_TOKENS, NUM_HEAD, K_HEAD_SIZE}, keyStrides_, keyOffset_);

    for (int i = 0; i < NUM_TOKENS; ++i) {
        int slot = slotMappingData[i];
        if (slot < 0) {
            continue;
        }
        int block_index = slot / BLOCK_SIZE;// 计算 block 索引
        int block_offset = slot % BLOCK_SIZE;// 计算 block 内偏移量
        // Get token key and value
        std::vector<std::vector<float>> token_key = ExtractTensor(target_key, {NUM_TOKENS, NUM_HEAD, K_HEAD_SIZE}, {NUM_HEAD, K_HEAD_SIZE}, i);
        // Assign token_key and token_value to the appropriate location in key_expect and value_expect
        for (int h = 0; h < NUM_HEAD; ++h) {
            for (int s = 0; s < K_HEAD_SIZE; ++s) {
                key_expect_siso[block_index][block_offset][h][s] = token_key[h][s];
            }
        }
    }
    // 根据顺序将所有输入tensor放入SVector
    atb::SVector<atb::Tensor> inTensors = {tensorKey,
        tensorKCache,
        tensorSlotMapping,
        keyStrides,
        keyOffset,
        };
    // 准备输入张量
    atb::VariantPack variantPack;
    variantPack.inTensors = inTensors;  // 放入输入tensor
    variantPack.outTensors = {tensorKCache};  // 放入输出tensor
    return variantPack;
}

/**
 * @brief 创建一个Reshape and Cache的Operation，并设置参数
 * @return atb::Operation * 返回一个Operation指针
 */
atb::Operation *PrepareOperationSISO()
{
    atb::infer::ReshapeAndCacheWithStrideParam opParam;
    opParam.compressType = atb::infer::ReshapeAndCacheWithStrideParam::CompressType::COMPRESS_TYPE_UNDEFINED;
    opParam.kvCacheCfg = atb::infer::ReshapeAndCacheWithStrideParam::KvCacheCfg::K_CACHE_V_BYPASS;
    atb::Operation *reshapeAndCacheOp = nullptr;
    atb::CreateOperation(opParam, &reshapeAndCacheOp);
    return reshapeAndCacheOp;
}

/**
 * @brief 准备atb::VariantPack
 * @param contextPtr context指针
 * @param stream stream
 * @param keyStrides_
 * @param valueStrides_
 * @param keyOffset_
 * @param valueOffset_
 * @return atb::VariantPack
 */
atb::VariantPack PrepareVariantPackND(atb::Context *contextPtr, aclrtStream stream, const std::vector<int64_t> &keyStrides_, const std::vector<int64_t> &valueStrides_, const int64_t keyOffset_, const int64_t valueOffset_)
{
    // 创建key，value tensor
    std::vector<float> keyData = KvGeneration(false, true);
    atb::Tensor tensorKey = CreateTensorFromVector(
        contextPtr, stream, keyData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NUM_TOKENS, NUM_HEAD, K_HEAD_SIZE*2}, {NUM_TOKENS, NUM_HEAD, K_HEAD_SIZE});
    std::vector<float> valueData = KvGeneration(false, false);
    atb::Tensor tensorValue = CreateTensorFromVector(
        contextPtr, stream, valueData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NUM_TOKENS, NUM_HEAD, K_HEAD_SIZE*2},{NUM_TOKENS, NUM_HEAD, V_HEAD_SIZE});
    // 创建kvCache tensor
    std::vector<float> kCacheData(NUM_BLOCKS * BLOCK_SIZE * NUM_HEAD * K_HEAD_SIZE, 0);
    atb::Tensor tensorKCache = CreateTensorFromVectorOrigin(
        contextPtr, stream, kCacheData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NUM_BLOCKS, BLOCK_SIZE, NUM_HEAD, K_HEAD_SIZE});
    std::vector<float> vCacheData(NUM_BLOCKS * BLOCK_SIZE * NUM_HEAD * V_HEAD_SIZE, 0);
    atb::Tensor tensorVCache = CreateTensorFromVectorOrigin(
        contextPtr, stream, vCacheData, ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, {NUM_BLOCKS, BLOCK_SIZE, NUM_HEAD, V_HEAD_SIZE});
    // 创建SlotMapping
    std::vector<int32_t> slotMappingData = SlotmappingGeneration(NUM_BLOCKS * BLOCK_SIZE, NUM_TOKENS);
    atb::Tensor tensorSlotMapping = CreateTensorOrigin(ACL_INT32, aclFormat::ACL_FORMAT_ND, {NUM_TOKENS});
    aclrtMemcpy(tensorSlotMapping.deviceData,
        tensorSlotMapping.dataSize,
        slotMappingData.data(),
        sizeof(int32_t) * slotMappingData.size(),
        ACL_MEMCPY_HOST_TO_DEVICE);
    // 创建keyStrides\valueStrides
    atb::Tensor keyStrides = CreateTensor(ACL_INT64, aclFormat::ACL_FORMAT_ND, {3}, {3});
    atb::Tensor valueStrides = CreateTensor(ACL_INT64, aclFormat::ACL_FORMAT_ND, {3}, {3});
    aclrtMallocHost(&keyStrides.hostData, keyStrides.dataSize);
    aclrtMallocHost(&valueStrides.hostData, valueStrides.dataSize);
    for (int i = 0; i < 3; i++) {
        static_cast<int64_t*>(keyStrides.hostData)[i] = keyStrides_[i];
        static_cast<int64_t*>(valueStrides.hostData)[i] = valueStrides_[i];
    }
    // 创建keyOffset\valueOffset
    atb::Tensor keyOffset = CreateTensor(ACL_INT64, aclFormat::ACL_FORMAT_ND, {1}, {1});
    atb::Tensor valueOffset = CreateTensor(ACL_INT64, aclFormat::ACL_FORMAT_ND, {1}, {1});
    aclrtMallocHost(&keyOffset.hostData, keyOffset.dataSize);
    aclrtMallocHost(&valueOffset.hostData, valueOffset.dataSize);

    static_cast<int64_t*>(keyOffset.hostData)[0] = keyOffset_;
    static_cast<int64_t*>(valueOffset.hostData)[0] = valueOffset_;
    // =========================================GOLDEN=======================================
    // 根据strides和offset获取expect key、value
    std::vector<float> target_key = create_non_contiguous_tensor(keyData, {NUM_TOKENS, NUM_HEAD, K_HEAD_SIZE}, keyStrides_, keyOffset_);
    std::vector<float> target_value = create_non_contiguous_tensor(valueData, {NUM_TOKENS, NUM_HEAD, V_HEAD_SIZE}, valueStrides_, valueOffset_);

    for (int i = 0; i < NUM_TOKENS; ++i) {
        int slot = slotMappingData[i];
        if (slot < 0) {
            continue;
        }
        int block_index = slot / BLOCK_SIZE;// 计算 block 索引
        int block_offset = slot % BLOCK_SIZE;// 计算 block 内偏移量
        // Get token key and value
        std::vector<std::vector<float>> token_key = ExtractTensor(target_key, {NUM_TOKENS, NUM_HEAD, K_HEAD_SIZE}, {NUM_HEAD, K_HEAD_SIZE}, i);
        std::vector<std::vector<float>> token_value = ExtractTensor(target_value, {NUM_TOKENS, NUM_HEAD, V_HEAD_SIZE}, {NUM_HEAD, V_HEAD_SIZE}, i);
        // Assign token_key and token_value to the appropriate location in key_expect and value_expect
        for (int h = 0; h < NUM_HEAD; ++h) {
            for (int s = 0; s < K_HEAD_SIZE; ++s) {
                key_expect_nd[block_index][block_offset][h][s] = token_key[h][s];
            }
        }
        for (int h = 0; h < NUM_HEAD; ++h) {
            for (int s = 0; s < V_HEAD_SIZE; ++s) {
                value_expect[block_index][block_offset][h][s] = token_value[h][s];
            }
        }
    }
    // 根据顺序将所有输入tensor放入SVector
    atb::SVector<atb::Tensor> inTensors = {tensorKey,
        tensorValue,
        tensorKCache,
        tensorVCache,
        tensorSlotMapping,
        keyStrides,
        valueStrides,
        keyOffset,
        valueOffset,
        };
    // 准备输入张量
    atb::VariantPack variantPack;
    variantPack.inTensors = inTensors;  // 放入输入tensor
    variantPack.outTensors = {tensorKCache, tensorVCache};  // 放入输出tensor
    return variantPack;
}

/**
 * @brief 创建一个Reshape and Cache的Operation，并设置参数
 * @return atb::Operation * 返回一个Operation指针
 */
atb::Operation *PrepareOperationND()
{
    atb::infer::ReshapeAndCacheWithStrideParam opParam;
    opParam.compressType = atb::infer::ReshapeAndCacheWithStrideParam::CompressType::COMPRESS_TYPE_UNDEFINED;
    opParam.kvCacheCfg = atb::infer::ReshapeAndCacheWithStrideParam::KvCacheCfg::K_CACHE_V_CACHE;
    atb::Operation *reshapeAndCacheOp = nullptr;
    atb::CreateOperation(opParam, &reshapeAndCacheOp);
    return reshapeAndCacheOp;
}

TEST(TestReshapeAndCacheWithStrideOperation, TestReshapeAndCacheWithStrideSISO)
{
    bool is910B = atb::GetSingleton<atb::Config>().Is910B();
    if (!is910B) {
        return;
    }
    // 设置卡号、创建context、设置stream
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    void *stream = nullptr;
    aclrtCreateStream(&stream);
    context->SetExecuteStream(stream);

    // RAC示例
    atb::Operation *reshapeAndCacheOp = PrepareOperationSISO();
    // 准备variantPack
    atb::VariantPack variantPack = PrepareVariantPackSISO(context, stream, keyStrides_, keyOffset_);
    uint64_t workspaceSize = 0;
    // 对输入Tensor和输出Tensor进行校验
    reshapeAndCacheOp->Setup(variantPack, workspaceSize, context);
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    // RAC执行
    atb::Status resultSt = reshapeAndCacheOp->Execute(variantPack, workspacePtr, workspaceSize, context);
    aclrtSynchronizeStream(stream);  // 流同步，等待device侧任务计算完成
    // ========================================输入信息==========================================;
    atb::Tensor &keyTensor = variantPack.inTensors.at(0);
    std::cout<<"keyTensor.dataSize:"<<keyTensor.dataSize<<std::endl;
    std::cout<<"keyTensor.desc.dtype:"<<keyTensor.desc.dtype<<std::endl;
    aclFloat16* deviceDataPtrKeyIn = static_cast<aclFloat16*>(malloc(keyTensor.dataSize));
    aclrtMemcpy(deviceDataPtrKeyIn, keyTensor.dataSize, keyTensor.deviceData, keyTensor.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    std::cout << "deviceDataPtrKeyIn: " << deviceDataPtrKeyIn << std::endl;
    // ========================================输出信息==========================================;
    atb::Tensor &outTensorKey = variantPack.outTensors.at(0);
    std::cout<<"outTensorKey.dataSize:"<<outTensorKey.dataSize<<std::endl;
    std::cout<<"outTensorKey.desc.dtype:"<<outTensorKey.desc.dtype<<std::endl;
    aclFloat16* deviceDataPtrKey = static_cast<aclFloat16*>(malloc(outTensorKey.dataSize));
    aclrtMemcpy(deviceDataPtrKey, outTensorKey.dataSize, outTensorKey.deviceData, outTensorKey.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    //========================================GOLDEN COMPARE====================================;
    // 将4维vector展成1维便于计算
    double total_error_key = 0.0;
    double total_absolute_key = 0.0;
    double eb_percent_Key = 0.0;
    std::vector<float> flattenedKeyExpect = Flatten4dTo1d(key_expect_siso);
    if (!NO_NEED_CAL) {
        for (uint64_t i = 0; i < NUM_BLOCKS * BLOCK_SIZE * NUM_HEAD * K_HEAD_SIZE; i++) {
            total_error_key += std::abs(flattenedKeyExpect[i] - aclFloat16ToFloat(deviceDataPtrKey[i]));
            total_absolute_key += std::abs(flattenedKeyExpect[i]);
            eb_percent_Key = (total_error_key / total_absolute_key) * 100.0;
        }
        std::cout << "KeyTensor EBPercent: " << eb_percent_Key << "%" << std::endl;
        double eb_target = 1.0 / pow(2.0, 10);
        if (eb_percent_Key/100.0 <= eb_target) {
            std::cout << "Reshape and Cache With Stride Run success!" << std::endl;
        } else {
            std::cout << "Reshape and Cache With Stride Run failed!" << std::endl;
        }
    } else {
        std::cout << "Reshape and Cache With Stride Run success!" << std::endl;
    }

    for (atb::Tensor &inTensor : variantPack.inTensors) {
        aclrtFree(inTensor.deviceData);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspacePtr);
    }
    // 资源释放
    atb::DestroyOperation(reshapeAndCacheOp);  // operation，对象概念，先释放
    aclrtDestroyStream(stream);
    atb::DestroyContext(context);  // context，全局资源，后释放
    ASSERT_EQ(resultSt, atb::NO_ERROR);
}

TEST(TestReshapeAndCacheWithStrideOperation, TestReshapeAndCacheWithStrideND)
{
    bool is910B = atb::GetSingleton<atb::Config>().Is910B();
    if (!is910B) {
        return;
    }
    // 设置卡号、创建context、设置stream
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    void *stream = nullptr;
    aclrtCreateStream(&stream);
    context->SetExecuteStream(stream);

    // RAC示例
    atb::Operation *reshapeAndCacheOp = PrepareOperationND();
    // 准备variantPack
    atb::VariantPack variantPack = PrepareVariantPackND(context, stream, keyStrides_, valueStrides_, keyOffset_, valueOffset_);
    uint64_t workspaceSize = 0;
    // 对输入Tensor和输出Tensor进行校验
    reshapeAndCacheOp->Setup(variantPack, workspaceSize, context);
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    // RAC执行
    atb::Status resultSt = reshapeAndCacheOp->Execute(variantPack, workspacePtr, workspaceSize, context);
    aclrtSynchronizeStream(stream);  // 流同步，等待device侧任务计算完成
    // ========================================输入信息==========================================;
    atb::Tensor &keyTensor = variantPack.inTensors.at(0);
    atb::Tensor &valueTensor = variantPack.inTensors.at(1);
    std::cout<<"keyTensor.dataSize:"<<keyTensor.dataSize<<std::endl;
    std::cout<<"keyTensor.desc.dtype:"<<keyTensor.desc.dtype<<std::endl;
    std::cout<<"valueTensor.dataSize:"<<valueTensor.dataSize<<std::endl;
    std::cout<<"valueTensor.desc.dtype:"<<valueTensor.desc.dtype<<std::endl;
    aclFloat16* deviceDataPtrKeyIn = static_cast<aclFloat16*>(malloc(keyTensor.dataSize));
    aclFloat16* deviceDataPtrValueIn = static_cast<aclFloat16*>(malloc(valueTensor.dataSize));
    aclrtMemcpy(deviceDataPtrKeyIn, keyTensor.dataSize, keyTensor.deviceData, keyTensor.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(deviceDataPtrValueIn, valueTensor.dataSize, valueTensor.deviceData, valueTensor.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    std::cout << "deviceDataPtrKeyIn: " << deviceDataPtrKeyIn << std::endl;
    std::cout << "deviceDataPtrValueIn: " << deviceDataPtrValueIn << std::endl;
    // ========================================输出信息==========================================;
    atb::Tensor &outTensorKey = variantPack.outTensors.at(0);
    atb::Tensor &outTensorValue = variantPack.outTensors.at(1);
    std::cout<<"outTensorKey.dataSize:"<<outTensorKey.dataSize<<std::endl;
    std::cout<<"outTensorKey.desc.dtype:"<<outTensorKey.desc.dtype<<std::endl;
    std::cout<<"outTensorValue.dataSize:"<<outTensorValue.dataSize<<std::endl;
    std::cout<<"outTensorValue.desc.dtype:"<<outTensorValue.desc.dtype<<std::endl;
    aclFloat16* deviceDataPtrKey = static_cast<aclFloat16*>(malloc(outTensorKey.dataSize));
    aclFloat16* deviceDataPtrValue = static_cast<aclFloat16*>(malloc(outTensorValue.dataSize));
    aclrtMemcpy(deviceDataPtrKey, outTensorKey.dataSize, outTensorKey.deviceData, outTensorKey.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(deviceDataPtrValue, outTensorValue.dataSize, outTensorValue.deviceData, outTensorValue.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    // ========================================GOLDEN COMPARE====================================;
    double total_error_key = 0.0;
    double total_error_value = 0.0;
    double total_absolute_key = 0.0;
    double total_absolute_value = 0.0;
    double eb_percent_Key = 0.0;
    double eb_percent_Value = 0.0;
    // 将4维vector展成1维便于计算
    std::vector<float> flattenedKeyExpect = Flatten4dTo1d(key_expect_nd);
    std::vector<float> flattenedValueExpect = Flatten4dTo1d(value_expect);
    if (!NO_NEED_CAL) {
        for (uint64_t i = 0; i < NUM_BLOCKS * BLOCK_SIZE * NUM_HEAD * K_HEAD_SIZE; i++) {
            total_error_key += std::abs(flattenedKeyExpect[i] - aclFloat16ToFloat(deviceDataPtrKey[i]));
            total_absolute_key += std::abs(flattenedKeyExpect[i]);
            eb_percent_Key = (total_error_key / total_absolute_key) * 100.0;
        }
        std::cout << "KeyTensor EBPercent: " << eb_percent_Key << "%" << std::endl;
        for (uint64_t i = 0; i < NUM_BLOCKS * BLOCK_SIZE * NUM_HEAD * V_HEAD_SIZE; i++) {
            total_error_value += std::abs(flattenedValueExpect[i] - aclFloat16ToFloat(deviceDataPtrValue[i]));
            total_absolute_value += std::abs(flattenedValueExpect[i]);
            eb_percent_Value = (total_error_value / total_absolute_value) * 100.0;
        }
        std::cout << "ValueTensor EBPercent: " << eb_percent_Value << "%" << std::endl;
        double eb_target = 1.0 / pow(2.0, 10);
        if (eb_percent_Key/100.0 <= eb_target && eb_percent_Value/100.0 <= eb_target) {
            std::cout << "Reshape and Cache With Stride Run success!" << std::endl;
        } else {
            std::cout << "Reshape and Cache With Stride Run failed!" << std::endl;
            EXPECT_FALSE(true);
        }
    } else {
        std::cout << "Reshape and Cache With Stride Run success!" << std::endl;
    }

    for (atb::Tensor &inTensor : variantPack.inTensors) {
        aclrtFree(inTensor.deviceData);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspacePtr);
    }
    // 资源释放
    atb::DestroyOperation(reshapeAndCacheOp);  // operation，对象概念，先释放
    aclrtDestroyStream(stream);
    atb::DestroyContext(context);  // context，全局资源，后释放
    ASSERT_EQ(resultSt, atb::NO_ERROR);
}