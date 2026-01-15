/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/atb_acl.h"
#include "atb/utils/log.h"

#ifdef __cplusplus
extern "C" {
#endif

// 256GB
const int64_t MAX_TENSOR_SIZE = 256uLL * 1024uLL * 1024uLL * 1024uLL;

int64_t GetTensorSize(const aclTensor *input)
{
    const op::Shape shape = input->GetViewShape();
    const size_t dims = shape.GetDimNum();
    int64_t size = 1;
    for (size_t i = 0; i < dims; ++i) {
        size *= shape.GetDim(i);
    }
    return size;
}

atb::Status aclTensorToAtbTensor(const aclTensor *aclTensorSrc, atb::Tensor *atbTensorDst)
{
    if (aclTensorSrc == nullptr) {
        atbTensorDst->hostData = nullptr;
        atbTensorDst->deviceData = nullptr;
        return atb::NO_ERROR;
    }
    ATB_CHECK(atbTensorDst != nullptr, "expect dst tensor address not to be null!",
              return atb::ERROR_INVALID_TENSOR_DIM);
    int64_t *dims = nullptr;
    uint64_t dimCount;
    aclDataType dataType;
    aclFormat format;
    auto status = aclGetViewShape(aclTensorSrc, &dims, &dimCount);
    ATB_CHECK(status == ACL_SUCCESS, "aclGetViewShape failed!", return atb::ERROR_INVALID_TENSOR_DIM);
    status = aclGetDataType(aclTensorSrc, &dataType);
    ATB_CHECK(status == ACL_SUCCESS, "aclGetDataType failed!", return atb::ERROR_INVALID_TENSOR_DTYPE);
    status = aclGetFormat(aclTensorSrc, &format);
    ATB_CHECK(status == ACL_SUCCESS, "aclGetFormat failed!", return atb::ERROR_INVALID_TENSOR_FORMAT);
    atb::TensorDesc desc;
    desc.shape.dimNum = dimCount;
    for (size_t i = 0; i < dimCount; i++) {
        desc.shape.dims[i] = (static_cast<int64_t *>(dims))[i];
    }
    desc.format = format;
    desc.dtype = dataType;
    atbTensorDst->desc = desc;
    atbTensorDst->deviceData = aclTensorSrc->GetData();
    atbTensorDst->hostData = nullptr;
    int64_t tensorSize = GetTensorSize(aclTensorSrc);
    int64_t dataTypeSize = static_cast<int64_t>(aclDataTypeSize(dataType));
    if (tensorSize > MAX_TENSOR_SIZE / dataTypeSize) {
        ATB_LOG(ERROR) << "The size of a tensor * dataTypeSize should be no more than 256GB, but got tensor size: "
                       << tensorSize;
    }
    atbTensorDst->dataSize = tensorSize * dataTypeSize;
    delete[] dims;
    return atb::NO_ERROR;
}

atb::Status aclTensorToAtbTensorHost(const aclTensor *aclTensorSrc, atb::Tensor *atbTensorDst)
{
    if (aclTensorSrc == nullptr) {
        atbTensorDst->hostData = nullptr;
        atbTensorDst->deviceData = nullptr;
        return atb::NO_ERROR;
    }
    ATB_CHECK(atbTensorDst != nullptr, "expect dst tensor address not to be null!",
              return atb::ERROR_INVALID_TENSOR_DIM);
    int64_t *dims = nullptr;
    uint64_t dimCount;
    aclDataType dataType;
    aclFormat format;
    auto status = aclGetViewShape(aclTensorSrc, &dims, &dimCount);
    ATB_CHECK(status == ACL_SUCCESS, "aclGetViewShape failed!", return atb::ERROR_INVALID_TENSOR_DIM);
    status = aclGetDataType(aclTensorSrc, &dataType);
    ATB_CHECK(status == ACL_SUCCESS, "aclGetDataType failed!", return atb::ERROR_INVALID_TENSOR_DTYPE);
    status = aclGetFormat(aclTensorSrc, &format);
    ATB_CHECK(status == ACL_SUCCESS, "aclGetFormat failed!", return atb::ERROR_INVALID_TENSOR_FORMAT);
    atb::TensorDesc desc;
    desc.shape.dimNum = dimCount;
    for (size_t i = 0; i < dimCount; i++) {
        desc.shape.dims[i] = (static_cast<int64_t *>(dims))[i];
    }
    desc.format = format;
    desc.dtype = dataType;
    atbTensorDst->desc = desc;
    atbTensorDst->deviceData = nullptr;
    atbTensorDst->hostData = aclTensorSrc->GetData();
    int64_t tensorSize = GetTensorSize(aclTensorSrc);
    int64_t dataTypeSize = static_cast<int64_t>(aclDataTypeSize(dataType));
    if (tensorSize > MAX_TENSOR_SIZE / dataTypeSize) {
        ATB_LOG(ERROR) << "The size of a tensor * dataTypeSize should be no more than 256GB, but got tensor size: "
                       << tensorSize;
    }
    atbTensorDst->dataSize = tensorSize * dataTypeSize;
    delete[] dims;
    return atb::NO_ERROR;
}

#ifdef __cplusplus
}
#endif
