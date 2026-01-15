/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_STUB_GE_LOG_H
#define ASCEND_OPS_STUB_GE_LOG_H

#include <mki/utils/log/log.h>
#include "ge_error_codes.h"
#include "common/util/error_manager/error_manager.h"
#include "external/ge_common/ge_api_error_codes.h"
#include "external/base/err_msg.h"

using string = std::string;

#define INTERNAL_ERROR 4

#define GELOGE(ERROR_CODE, fmt, ...)                                                                                   \
    MKI_FLOG_ERROR("[%s] error code: %d, " #fmt, __FUNCTION__, ERROR_CODE, ##__VA_ARGS__)

#define GELOGW(fmt, ...) MKI_FLOG_WARN("[%s] " #fmt, __FUNCTION__, ##__VA_ARGS__)
#define GELOGI(fmt, ...) MKI_FLOG_INFO("[%s] " #fmt, __FUNCTION__, ##__VA_ARGS__)
#define GELOGD(fmt, ...) MKI_FLOG_DEBUG("[%s] " #fmt, __FUNCTION__, ##__VA_ARGS__)

#define GE_LOG_ERROR(...)
#define GE_CHECK_NOTNULL_JUST_RETURN(...)
#define GE_CHECK_NOTNULL(...)

#define REPORT_CALL_ERROR REPORT_INNER_ERROR

#define GE_IF_BOOL_EXEC(...)

// If expr is not true, print the log and return the specified status
#define GE_CHK_BOOL_RET_STATUS(expr, _status, ...)       \
    do {                                                 \
        const bool b = (expr);                           \
        if (!b) {                                        \
            REPORT_INNER_ERROR("E19999", __VA_ARGS__);   \
            GELOGE((_status), __VA_ARGS__);              \
            return (_status);                            \
        }                                                \
    } while (false)

using Status = uint32_t;

#endif // ASCEND_OPS_STUB_GE_LOG_H
