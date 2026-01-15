/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DEMO_UTIL_H
#define DEMO_UTIL_H

#include <iostream>
#include <numeric>
#include <vector>

#include "acl/acl.h"
#include "atb/atb_infer.h"
#include "atb/operation.h"
#include "atb/types.h"

#define CHECK_STATUS(status)                                                                                           \
    do {                                                                                                               \
        if ((status) != 0) {                                                                                           \
            std::cout << __FILE__ << ":" << __LINE__ << " [error]: " << (status) << std::endl;                         \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

#define CHECK_STATUS_EXPR(status, expr)                                                                                \
    do {                                                                                                               \
        if ((status) != 0) {                                                                                           \
            std::cout << __FILE__ << ":" << __LINE__ << " [error]: " << (status) << std::endl;                         \
            expr;                                                                                                      \
        }                                                                                                              \
    } while (0)

/**
 * @brief 创建一个Tensor对象
 * @param  dataType 数据类型
 * @param  format 数据格式
 * @param  shape 数据shape
 * @param atb::Tensor 返回创建的Tensor对象
 * @return atb::Status atb错误码
 */
atb::Status CreateTensor(const aclDataType dataType, const aclFormat format, std::vector<int64_t> shape,
                         atb::Tensor &tensor)
{
    tensor.desc.dtype = dataType;
    tensor.desc.format = format;
    tensor.desc.shape.dimNum = shape.size();
    // tensor的dim依次设置为shape中元素
    for (size_t i = 0; i < shape.size(); i++) {
        tensor.desc.shape.dims[i] = shape.at(i);
    }
    tensor.dataSize = atb::Utils::GetTensorSize(tensor); // 计算Tensor的数据大小
    CHECK_STATUS(aclrtMalloc(&tensor.deviceData, tensor.dataSize, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST));
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 进行数据类型转换，调用Elewise的cast Op
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensor 输入tensor
 * @param outTensorType 输出tensor的数据类型
 * @param shape 输出tensor的shape
 * @param atb::Tensor 转换后的tensor
 * @return atb::Status atb错误码
 */
atb::Status CastOp(atb::Context *contextPtr, aclrtStream stream, const atb::Tensor inTensor,
                   const aclDataType outTensorType, atb::Tensor &outTensor)
{
    uint64_t workspaceSize = 0;
    void *workspace = nullptr;
    // 创建Elewise的ELEWISE_CAST
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ELEWISE_CAST;
    castParam.outTensorType = outTensorType;
    atb::Operation *castOp = nullptr;
    CHECK_STATUS(CreateOperation(castParam, &castOp));
    // atb::Tensor outTensor;
    // CreateTensor(outTensorType, aclFormat::ACL_FORMAT_ND, shape, outTensor); // cast输出tensor
    atb::VariantPack castVariantPack; // 参数包
    castVariantPack.inTensors = {inTensor};
    castVariantPack.outTensors = {outTensor};
    // 在Setup接口调用时对输入tensor和输出tensor进行校验。
    CHECK_STATUS(castOp->Setup(castVariantPack, workspaceSize, contextPtr));
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc(&workspace, workspaceSize, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // ELEWISE_CAST执行
    CHECK_STATUS(castOp->Execute(castVariantPack, (uint8_t *)workspace, workspaceSize, contextPtr));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspace)); // 清理工作空间
    }
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 根据输入inShape，判断是NZ格式的shape还是ND格式的shape，并得到两种格式下的shape
 * @param tensorType tensor的数据类型
 * @param inShape tensor的shape
 * @param ndShape ND格式tensor的shape
 * @param nzShape NZ格式tensor的shape
 * @return atb::Status atb错误码
 */
atb::Status GetShape(const aclDataType tensorType, const std::vector<int64_t> &inShape, std::vector<int64_t> &ndShape,
                     std::vector<int64_t> &nzShape)
{
    constexpr int64_t DIM_TRAN_PARAM_INT8 = 32;
    constexpr int64_t DIM_TRAN_PARAM_BF16_FP16 = 16;
    int64_t n0 = DIM_TRAN_PARAM_BF16_FP16; // 维度转换参数
    // 输入tensor数据类型为ACL_INT8时，维度转换参数取32
    if (tensorType == ACL_INT8) {
        n0 = DIM_TRAN_PARAM_INT8;
    }
    if (inShape.size() == 4) { // inShape是NZ格式tensor的shape
        nzShape.assign(inShape.begin(), inShape.end());
        ndShape = {inShape[0], inShape[2], inShape[1] * inShape[3]};
    } else { // inShape是ND格式tensor的shape
        ndShape.assign(inShape.begin(), inShape.end());
        if (inShape.size() == 3) { // 该shape包含batch参数
            nzShape = {inShape[0], inShape[2] / n0, inShape[1], n0};
        } else if (inShape.size() == 2) {
            nzShape = {1, inShape[1] / n0, inShape[0], n0};
        } else {
            std::cout << "shape dimensions invalid!" << std::endl;
            return atb::ErrorType::ERROR_INVALID_TENSOR_DIM_NUM;
        }
    }
    return atb::ErrorType::NO_ERROR;
}

/**
 * @brief 进行ND到NZ的数据格式转换，调用transdata Op
 * @param contextPtr context指针
 * @param stream stream
 * @param inTensor 输入tensor
 * @param outTensorType tensor的数据类型
 * @param outTensor 输出tensor
 * @param shape tensor的shape
 * @return atb::Status atb错误码
 */
atb::Status TransdataOp(atb::Context *contextPtr, aclrtStream stream, const atb::Tensor inTensor,
                        const aclDataType tensorType, atb::Tensor &outTensor, std::vector<int64_t> shape)
{
    uint64_t workspaceSize = 0;
    void *workspace = nullptr;

    atb::infer::TransdataParam opParam;
    opParam.transdataType = atb::infer::TransdataParam::TransdataType::ND_TO_FRACTAL_NZ;

    atb::Operation *transdataOp = nullptr;
    CHECK_STATUS(atb::CreateOperation(opParam, &transdataOp));
    atb::Tensor tensor;
    CHECK_STATUS(CreateTensor(tensorType, aclFormat::ACL_FORMAT_FRACTAL_NZ, shape, tensor));
    atb::VariantPack transdataVariantPack;
    transdataVariantPack.inTensors = {inTensor};
    transdataVariantPack.outTensors = {tensor};
    // 在Setup接口调用时对输入tensor和输出tensor进行校验。
    CHECK_STATUS(transdataOp->Setup(transdataVariantPack, workspaceSize, contextPtr));
    uint8_t *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtMalloc(&workspace, workspaceSize, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // Transdata ND to NZ执行
    CHECK_STATUS(transdataOp->Execute(transdataVariantPack, (uint8_t *)workspace, workspaceSize, contextPtr));
    CHECK_STATUS(aclrtSynchronizeStream(stream)); // 流同步，等待device侧任务计算完成
    if (workspaceSize > 0) {
        CHECK_STATUS(aclrtFree(workspace)); // 清理工作空间
    }
    outTensor = tensor;
    return atb::ErrorType::NO_ERROR;
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
 * @param outTensor 返回创建的tensor
 */
template <typename T>
atb::Status CreateTensorFromVector(atb::Context *contextPtr, aclrtStream stream, std::vector<T> data,
                                   const aclDataType outTensorType, const aclFormat format, std::vector<int64_t> shape,
                                   atb::Tensor &outTensor, const aclDataType inTensorType = ACL_DT_UNDEFINED)
{
    atb::Tensor tensor;
    aclDataType intermediateType;
    switch (outTensorType) {
        case aclDataType::ACL_FLOAT16:
        case aclDataType::ACL_BF16:
        case aclDataType::ACL_DOUBLE:
            intermediateType = aclDataType::ACL_FLOAT;
            break;
        default:
            intermediateType = outTensorType;
    }
    if (inTensorType == outTensorType && inTensorType != ACL_DT_UNDEFINED) {
        intermediateType = outTensorType;
    }
    aclFormat tensorFormat = format;
    if (intermediateType != outTensorType && format == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        tensorFormat = aclFormat::ACL_FORMAT_ND;
    }
    CHECK_STATUS(CreateTensor(intermediateType, tensorFormat, shape, tensor));
    CHECK_STATUS(aclrtMemcpy(tensor.deviceData, tensor.dataSize, data.data(), sizeof(T) * data.size(),
                             ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_STATUS(CreateTensor(outTensorType, aclFormat::ACL_FORMAT_ND, shape, outTensor));
    if (intermediateType == outTensorType) {
        // 原始创建的tensor类型，不需要转换
        outTensor = tensor;
        return atb::ErrorType::NO_ERROR;
    }
    CHECK_STATUS(CastOp(contextPtr, stream, tensor, outTensorType, outTensor));
    // 直接赋值将tensor转成需要的数据格式，或者使用提供的TransdataOp函数进行数据格式转换，详细用法见同级目录README.md
    outTensor.desc.format = format;
    return atb::ErrorType::NO_ERROR;
}

#endif
