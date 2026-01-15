/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils.h"
#include <securec.h>
#include <limits>
#include <complex>
#include <string>
#include "atb/utils/log.h"

namespace atb {
std::string Utils::GetAtbVersion()
{
    // ATBVERSION is a placeholder and will be replaced by real version number when CI build runs
    return "ATBVERSION";
}

uint64_t Utils::GetTensorSize(const Tensor &tensor)
{
    return GetTensorSize(tensor.desc);
}

uint64_t Utils::GetTensorSize(const TensorDesc &tensorDesc)
{
    if (tensorDesc.shape.dimNum == 0) {
        return 0;
    }

    uint64_t dataItemSize = 0;
    switch (tensorDesc.dtype) {
        case ACL_DT_UNDEFINED:
            dataItemSize = sizeof(bool);
            break;
        case ACL_BOOL:
            dataItemSize = sizeof(bool);
            break;
        case ACL_FLOAT:
            dataItemSize = sizeof(float);
            break;
        case ACL_FLOAT16:
            dataItemSize = sizeof(int16_t);
            break;
        case ACL_INT8:
            dataItemSize = sizeof(int8_t);
            break;
        case ACL_INT16:
            dataItemSize = sizeof(int16_t);
            break;
        case ACL_INT32:
            dataItemSize = sizeof(int32_t);
            break;
        case ACL_INT64:
            dataItemSize = sizeof(int64_t);
            break;
        case ACL_UINT8:
            dataItemSize = sizeof(uint8_t);
            break;
        case ACL_UINT16:
            dataItemSize = sizeof(uint16_t);
            break;
        case ACL_UINT32:
            dataItemSize = sizeof(uint32_t);
            break;
        case ACL_UINT64:
            dataItemSize = sizeof(uint64_t);
            break;
        case ACL_BF16:
            dataItemSize = sizeof(int16_t);
            break;
        case ACL_DOUBLE:
            dataItemSize = sizeof(double);
            break;
        case ACL_STRING:
            dataItemSize = sizeof(std::string);
            break;
        case ACL_COMPLEX64:
            dataItemSize = sizeof(std::complex<float>);
            break;
        case ACL_COMPLEX128:
            dataItemSize = sizeof(std::complex<double>);
            break;
        default:
            ATB_LOG(ERROR)
                << "Tensor not support dtype:" << tensorDesc.dtype
                << ". Only support below dtype now: ACL_DT_UNDEFINED, ACL_BOOL, ACL_FLOAT, ACL_FLOAT16, ACL_INT8, "
                << "ACL_INT16, ACL_INT32, ACL_INT64, ACL_UINT8, ACL_UINT16, ACL_UINT32, ACL_UINT64, ACL_BF16, "
                   "ACL_DOUBLE,"
                << " ACL_STRING, ACL_COMPLEX64, ACL_COMPLEX128.";
            return 0;
    }

    uint64_t elementCount = GetTensorNumel(tensorDesc);
    if (elementCount == 0) {
        ATB_LOG(ERROR) << "GetTensorSize result is zero!";
        return 0;
    }
    if (std::numeric_limits<uint64_t>::max() / elementCount < dataItemSize) {
        ATB_LOG(ERROR) << "GetTensorSize Overflow!";
        return 0;
    }
    return dataItemSize * elementCount;
}

uint64_t Utils::GetTensorNumel(const Tensor &tensor)
{
    return GetTensorNumel(tensor.desc);
}

uint64_t Utils::GetTensorNumel(const TensorDesc &tensorDesc)
{
    if (tensorDesc.shape.dimNum == 0) {
        return 0;
    }
    uint64_t elementCount = 1;
    uint64_t maxVal = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < tensorDesc.shape.dimNum; i++) {
        if (tensorDesc.shape.dims[i] <= 0) {
            ATB_LOG(ERROR) << "dims[" << i << "] is <= 0!";
            return 0;
        }
        if (maxVal / tensorDesc.shape.dims[i] < elementCount) {
            ATB_LOG(ERROR) << "GetTensorNumel Overflow!";
            return 0;
        }
        elementCount *= tensorDesc.shape.dims[i];
    }

    return elementCount;
}

void Utils::QuantParamConvert(const float *src, uint64_t *dest, uint64_t itemCount)
{
    if (src == nullptr || dest == nullptr) {
        return;
    }
    for (uint64_t i = 0; i < itemCount; i++) {
        uint32_t temp;
        int ret = memcpy_s(&temp, sizeof(temp), &src[i], sizeof(temp));
        ATB_LOG_IF(ret != EOK, ERROR) << "memcpy_s Error! Error Code: " << ret;
        dest[i] = static_cast<uint64_t>(temp);
    }
}
} // namespace atb