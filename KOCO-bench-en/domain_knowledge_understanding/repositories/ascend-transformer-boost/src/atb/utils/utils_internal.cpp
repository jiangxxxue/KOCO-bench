/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/utils_internal.h"
#include <unistd.h>
#include <syscall.h>
#include <cmath>
#include <limits>
#include <complex>
#include <string>
#include "atb/utils/log.h"

namespace atb {
bool UtilsInternal::IsFloatEqual(float lh, float rh)
{
    return (std::fabs(lh - rh) <= std::numeric_limits<float>::epsilon());
}

bool UtilsInternal::IsDoubleEqual(double lh, double rh)
{
    return (std::fabs(lh - rh) <= std::numeric_limits<double>::epsilon());
}

int32_t UtilsInternal::GetCurrentThreadId()
{
    int32_t tid = static_cast<int32_t>(syscall(SYS_gettid));
    if (tid == -1) {
        ATB_LOG(ERROR) << "get tid failed, errno: " << errno;
    }
    return tid;
}

int32_t UtilsInternal::GetCurrentProcessId()
{
    int32_t pid = static_cast<int32_t>(syscall(SYS_getpid));
    if (pid == -1) {
        ATB_LOG(ERROR) << "get pid failed, errno: " << errno;
    }
    return pid;
}

int64_t UtilsInternal::AlignUp(int64_t dim, int64_t align)
{
    if (align == 0) {
        return -1;
    }
    return (dim + align - 1) / align * align;
}

uint64_t UtilsInternal::GetDataTypeSize(const aclDataType &dType)
{
    switch (dType) {
        case ACL_DT_UNDEFINED:
        case ACL_BOOL:
            return sizeof(bool);
        case ACL_INT8:
            return sizeof(int8_t);
        case ACL_UINT8:
            return sizeof(uint8_t);
        case ACL_FLOAT16:
        case ACL_BF16:
        case ACL_INT16:
            return sizeof(int16_t);
        case ACL_UINT16:
            return sizeof(uint16_t);
        case ACL_FLOAT:
            return sizeof(float);
        case ACL_INT32:
            return sizeof(int32_t);
        case ACL_UINT32:
            return sizeof(uint32_t);
        case ACL_INT64:
            return sizeof(int64_t);
        case ACL_UINT64:
            return sizeof(uint64_t);
        case ACL_DOUBLE:
            return sizeof(double);
        case ACL_STRING:
            return sizeof(std::string);
        case ACL_COMPLEX64:
            return sizeof(std::complex<float>);
        case ACL_COMPLEX128:
            return sizeof(std::complex<double>);
        default:
            ATB_LOG(ERROR) << "Tensor does not support dtype: " << dType
                           << " . Only support following dtype now: ACL_DT_UNDEFINED, ACL_BOOL, ACL_FLOAT, ACL_FLOAT16,"
                              "ACL_INT8, ACL_INT16, ACL_INT32, ACL_INT64, ACL_UINT8, ACL_UINT16, ACL_UINT32, "
                              "ACL_UINT64, ACL_BF16, ACL_DOUBLE, ACL_STRING, ACL_COMPLEX64, ACL_COMPLEX128.";
    }
    return 0;
}
} // namespace atb