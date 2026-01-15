/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_TBE_STUB_OP_LOG_H
#define ASCEND_OPS_TBE_STUB_OP_LOG_H

#include <list>
#include <cstdio>

#include <mki/utils/log/log.h>

#include "ascend_string.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/operator.h"
#include "graph/node.h"
#include "toolchain/slog.h"
#include "base/err_msg.h"


#define OPPROTO_SUBMOD_NAME "OP_PROTO"

inline const char *get_cstr(const std::string &str) { return str.c_str(); }

inline const char *get_cstr(const char *str) { return str; }

inline const std::string &get_op_info(const std::string &str) { return str; }

inline const char *get_op_info(const char *str) { return str; }

inline std::string get_op_info(const ge::NodePtr &node)
{
    return node != nullptr ? node->GetType() + ":" + node->GetName() : "nil";
}

inline std::string get_op_info(const ge::OpDescPtr &node)
{
    return node != nullptr ? node->GetType() + ":" + node->GetName() : "nil";
}

template <class T> constexpr bool is_ge_operator_type()
{
    return std::is_base_of<ge::Operator, typename std::decay<T>::type>::value;
}

template <class T> typename std::enable_if<is_ge_operator_type<T>(), std::string>::type get_op_info(const T &op)
{
    ge::AscendString name;
    ge::AscendString type;
    auto get_name_ret = op.GetName(name);
    auto get_type_ret = op.GetOpType(type);
    std::string op_info = get_type_ret == ge::GRAPH_SUCCESS ? type.GetString() : "nil";
    op_info += ":";
    op_info += get_name_ret == ge::GRAPH_SUCCESS ? name.GetString() : "nil";
    return op_info;
}

template <class T> constexpr bool is_context_type()
{
    return !std::is_base_of<ge::Operator, typename std::decay<T>::type>::value &&
           !std::is_same<const char *, typename std::decay<T>::type>::value &&
           !std::is_same<char *, typename std::decay<T>::type>::value &&
           !std::is_same<std::string, typename std::decay<T>::type>::value &&
           !std::is_same<ge::NodePtr, typename std::decay<T>::type>::value &&
           !std::is_same<ge::OpDescPtr, typename std::decay<T>::type>::value;
}

template <class T> typename std::enable_if<is_context_type<T>(), std::string>::type get_op_info(T context)
{
    if (context == nullptr) {
        return "nil:nil";
    }
    std::string op_info = context->GetNodeType() != nullptr ? context->GetNodeType() : "nil";
    op_info += ":";
    op_info += context->GetNodeName() != nullptr ? context->GetNodeName() : "nil";
    return op_info;
}

template <typename T> std::string TbeGetName(const T &op)
{
    ge::AscendString op_ascend_name;
    ge::graphStatus ret = op.GetName(op_ascend_name);
    if (ret != ge::GRAPH_SUCCESS) {
        std::string op_name = "None";
        return op_name;
    }
    return op_ascend_name.GetString();
}

template <typename T> std::string TbeGetOpType(const T &op)
{
    ge::AscendString op_ascend_name;
    ge::graphStatus ret = op.GetOpType(op_ascend_name);
    if (ret != ge::GRAPH_SUCCESS) {
        std::string op_name = "None";
        return op_name;
    }
    return op_ascend_name.GetString();
}

#define CHECK_DIVISOR_ZERO(divisor)                                                                                   \
    if ((divisor) == 0) {                                                                                             \
        return;                                                                                                       \
    }

#define CHECK_DIVISOR_ZERO_RET(divisor, ret)                                                                          \
    if ((divisor) == 0) {                                                                                             \
        return ret;                                                                                                   \
    }

#define OP_CHECK(cond, log_func, return_expr)                                                                         \
    if (cond) {                                                                                                       \
        log_func;                                                                                                     \
        return_expr;                                                                                                  \
    }

#define OP_LOGI(opname, ...) D_OP_LOGI(get_op_info(opname), __VA_ARGS__)
#define OP_LOGW(opname, ...) D_OP_LOGW(get_op_info(opname), __VA_ARGS__)

#define OP_LOGE_WITHOUT_REPORT(opname, ...) D_OP_LOGE(get_op_info(opname), __VA_ARGS__)
#define OP_LOGE(op_name, ...)                                                                                          \
    do {                                                                                                               \
        OP_LOGE_WITHOUT_REPORT(op_name, ##__VA_ARGS__);                                                                \
        REPORT_INNER_ERROR("EZ9999", ##__VA_ARGS__);                                                                   \
    } while (0)

#define OP_LOGD(opname, ...) D_OP_LOGD(get_op_info(opname), __VA_ARGS__)
#define OP_EVENT(opname, ...) D_OP_EVENT(get_op_info(opname), __VA_ARGS__)

#define OP_LOG_SUB_DEBUG(op_info, fmt, ...)                                                                           \
    MKI_FLOG_DEBUG("[%s] OpName:[%s] " #fmt, __FUNCTION__, get_cstr(op_info), ##__VA_ARGS__)
#define OP_LOG_SUB_INFO(op_info, fmt, ...)                                                                            \
    MKI_FLOG_INFO("[%s] OpName:[%s] " #fmt, __FUNCTION__, get_cstr(op_info), ##__VA_ARGS__)
#define OP_LOG_SUB_WARN(op_info, fmt, ...)                                                                            \
    MKI_FLOG_WARN("[%s] OpName:[%s] " #fmt, __FUNCTION__, get_cstr(op_info), ##__VA_ARGS__)
#define OP_LOG_SUB_ERROR(op_info, fmt, ...)                                                                           \
    MKI_FLOG_ERROR("[%s] OpName:[%s] " #fmt, __FUNCTION__, get_cstr(op_info), ##__VA_ARGS__)

#define D_OP_LOGI(opname, fmt, ...) OP_LOG_SUB_INFO(opname, fmt, ##__VA_ARGS__)
#define D_OP_LOGW(opname, fmt, ...) OP_LOG_SUB_WARN(opname, fmt, ##__VA_ARGS__)
#define D_OP_LOGE(opname, fmt, ...) OP_LOG_SUB_ERROR(opname, fmt, ##__VA_ARGS__)
#define D_OP_LOGD(opname, fmt, ...) OP_LOG_SUB_DEBUG(opname, fmt, ##__VA_ARGS__)
#define D_OP_EVENT(opname, fmt, ...) OP_LOG_SUB_INFO(opname, fmt, ##__VA_ARGS__)

#define UNLIKELY(x) __builtin_expect((x), 0)
#define LIKELY(x) __builtin_expect((x), 1)

#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                         \
    static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool");       \
    do {                                                                                                               \
        if (UNLIKELY(condition)) {                                                                                     \
            OP_LOGE(op_name, fmt, ##__VA_ARGS__);                                                                      \
            return return_value;                                                                                       \
        }                                                                                                              \
    } while (0)

#define OP_LOGW_IF(condition, op_name, fmt, ...)                                                                       \
    static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool");       \
    do {                                                                                                               \
        if (UNLIKELY(condition)) {                                                                                     \
            OP_LOGW(op_name, fmt, ##__VA_ARGS__);                                                                      \
        }                                                                                                              \
    } while (0)

#define OP_LOGI_IF_RETURN(condition, return_value, op_name, fmt, ...)                                                  \
    static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool");       \
    do {                                                                                                               \
        if (UNLIKELY(condition)) {                                                                                     \
            OP_LOGI(op_name, fmt, ##__VA_ARGS__);                                                                      \
            return return_value;                                                                                       \
        }                                                                                                              \
    } while (0)

constexpr const int OP_MAX_LOG_SIZE = 16000;
constexpr const int OP_MSG_HEADER_LEN = 200;
// print very long log. long line will be split to multipile lines
#define OP_LOG_FULL(opname, format, ...)                                                                               \
    do {                                                                                                               \
        if (Mki::LogLevel::DEBUG < Mki::LogCore::Instance().GetLogLevel()) {                                     \
            break;                                                                                                     \
        }                                                                                                              \
        char msgbufxyz[OP_MAX_LOG_SIZE];                                                                               \
        size_t msgmaxlen = (MSG_LENGTH - OP_MSG_HEADER_LEN);                                                           \
        int rettmp = snprintf_s(msgbufxyz, sizeof(msgbufxyz), sizeof(msgbufxyz) - 1, format, ##__VA_ARGS__);           \
        if (rettmp == -1) {                                                                                            \
            msgbufxyz[sizeof(msgbufxyz) - 1] = '\0';                                                                   \
        }                                                                                                              \
        size_t msglength = std::strlen(msgbufxyz);                                                                     \
        if (msglength < msgmaxlen) {                                                                                   \
            OP_LOG_SUB_DEBUG(opname, "%s", msgbufxyz);                                                                 \
            break;                                                                                                     \
        }                                                                                                              \
        char *msgchunkbegin = msgbufxyz;                                                                               \
        char *msgchunkend = nullptr;                                                                                   \
        while (msgchunkbegin < msgbufxyz + msglength) {                                                                \
            if (msgchunkbegin[0] == '\n') {                                                                            \
                OP_LOG_SUB_DEBUG(opname, "");                                                                          \
                msgchunkbegin += 1;                                                                                    \
                continue;                                                                                              \
            }                                                                                                          \
            msgchunkend = std::strchr(msgchunkbegin, '\n');                                                            \
            if (msgchunkend == nullptr) {                                                                              \
                msgchunkend = msgchunkbegin + std::strlen(msgchunkbegin);                                              \
            }                                                                                                          \
            while (msgchunkend > msgchunkbegin) {                                                                      \
                std::string msgchunk(msgchunkbegin,                                                                    \
                                     std::min(msgmaxlen, static_cast<size_t>(msgchunkend - msgchunkbegin)));           \
                OP_LOG_SUB_DEBUG(opname, "%s", msgchunk.c_str());                                                      \
                msgchunkbegin += msgchunk.size();                                                                      \
            }                                                                                                          \
            msgchunkbegin += 1;                                                                                        \
        }                                                                                                              \
    } while (0)

#define OP_LOGD_FULL(opname, ...) OP_LOG_FULL(get_op_info(opname), __VA_ARGS__)

int CheckLogLevel(int moduleId, int logLevel);

#endif