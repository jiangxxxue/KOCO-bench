/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file host_log.h
 * \brief
 */

#ifndef IMPL_HOST_LOG_H
#define IMPL_HOST_LOG_H
#include <cassert>
#include <cstdint>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>

#define ASCENDC_HOST_ASSERT(cond, ret, format, ...)                            \
    do {                                                                         \
        if (!(cond)) {                                                             \
            MKI_FLOG_ERROR("[%s] " #format, __FUNCTION__, ##__VA_ARGS__);            \
            ret;                                                                     \
        }                                                                          \
    } while (0)

// 0 debug, 1 info, 2 warning, 3 error
#define TILING_LOG_ERROR(format, ...)  MKI_FLOG_ERROR("[%s] " #format, __FUNCTION__, ##__VA_ARGS__)

#define TILING_LOG_INFO(format, ...) MKI_FLOG_INFO("[%s] " #format, __FUNCTION__, ##__VA_ARGS__)

#define TILING_LOG_WARNING(format, ...) MKI_FLOG_WARN("[%s] " #format, __FUNCTION__, ##__VA_ARGS__)

#define TILING_LOG_DEBUG(format, ...) MKI_FLOG_DEBUG("[%s] " #format, __FUNCTION__, ##__VA_ARGS__)
#endif // IMPL_HOST_LOG_H
