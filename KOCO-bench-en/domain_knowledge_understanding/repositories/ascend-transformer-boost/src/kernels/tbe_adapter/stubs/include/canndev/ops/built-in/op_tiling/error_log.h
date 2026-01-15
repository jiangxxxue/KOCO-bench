/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file error_log.h
 * \brief
 */
#ifndef ASCEND_OPS_STUB_ERROR_LOG_H
#define ASCEND_OPS_STUB_ERROR_LOG_H

#include <mki/utils/log/log.h>
#include "op_log.h"

using namespace Mki;
using namespace std;

namespace optiling {
// ADD FORMAT LOG
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...)                                                         \
    do {                                                                                                               \
        OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);                                                       \
        REPORT_INNER_ERROR("E89999", "op[%s], " err_msg, get_cstr(get_op_info(op_name)), ##__VA_ARGS__);               \
    } while (0)

#define OP_TILING_CHECK(cond, log_func, expr)                                                                          \
    do {                                                                                                               \
        if (cond) {                                                                                                    \
            log_func;                                                                                                  \
            expr;                                                                                                      \
        }                                                                                                              \
    } while (0)
} // namespace optiling

#endif // ASCEND_OPS_STUB_ERROR_LOG_H
