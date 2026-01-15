/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_LOG_H
#define ATB_LOG_H
#include <mki/utils/log/log.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log_stream.h>
#include <mki/utils/log/log_core.h>
#include <mki/utils/log/log_sink.h>
#include <mki/utils/log/log_entity.h>
#include "atb/types.h"

#define ATB_CHECK(condition, logExpr, handleExpr)                                                                      \
    MKI_CHECK(condition, logExpr, handleExpr)
        
#define ATB_LOG(level) ATB_LOG_##level
#define ATB_FLOG(level, format, ...) ATB_FLOG_##level(format, __VA_ARGS__)

#define ATB_LOG_IF(condition, level)                                                                                   \
    if (condition)                                                                                                     \
    ATB_LOG(level)

#define ATB_LOG_TRACE                                                                                                  \
    if (Mki::LogLevel::TRACE >= Mki::LogCore::Instance().GetLogLevel())                                                \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::TRACE)
#define ATB_LOG_DEBUG                                                                                                  \
    if (Mki::LogLevel::DEBUG >= Mki::LogCore::Instance().GetLogLevel())                                                \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::DEBUG)
#define ATB_LOG_INFO                                                                                                   \
    if (Mki::LogLevel::INFO >= Mki::LogCore::Instance().GetLogLevel())                                                 \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::INFO)
#define ATB_LOG_WARN                                                                                                   \
    if (Mki::LogLevel::WARN >= Mki::LogCore::Instance().GetLogLevel())                                                 \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::WARN)
#define ATB_LOG_ERROR                                                                                                  \
    if (Mki::LogLevel::ERROR >= Mki::LogCore::Instance().GetLogLevel())                                                \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::ERROR)
#define ATB_LOG_FATAL                                                                                                  \
    if (Mki::LogLevel::FATAL >= Mki::LogCore::Instance().GetLogLevel())                                                \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::FATAL)

#define ATB_FLOG_TRACE(format, ...)                                                                                    \
    if (Mki::LogLevel::TRACE >= Mki::LogCore::Instance().GetLogLevel())                                                \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::TRACE).Format(format, __VA_ARGS__)
#define ATB_FLOG_DEBUG(format, ...)                                                                                    \
    if (Mki::LogLevel::DEBUG >= Mki::LogCore::Instance().GetLogLevel())                                                \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::DEBUG).Format(format, __VA_ARGS__)
#define ATB_FLOG_INFO(format, ...)                                                                                     \
    if (Mki::LogLevel::INFO >= Mki::LogCore::Instance().GetLogLevel())                                                 \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::INFO).Format(format, __VA_ARGS__)
#define ATB_FLOG_WARN(format, ...)                                                                                     \
    if (Mki::LogLevel::WARN >= Mki::LogCore::Instance().GetLogLevel())                                                 \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::WARN).Format(format, __VA_ARGS__)
#define ATB_FLOG_ERROR(format, ...)                                                                                    \
    if (Mki::LogLevel::ERROR >= Mki::LogCore::Instance().GetLogLevel())                                                \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::ERROR).Format(format, __VA_ARGS__)
#define ATB_FLOG_FATAL(format, ...)                                                                                    \
    if (Mki::LogLevel::FATAL >= Mki::LogCore::Instance().GetLogLevel())                                                \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::FATAL).Format(format, __VA_ARGS__)

namespace atb {
struct ExternalError {
    ErrorType errorType = NO_ERROR;
    std::string errorDesc;
    std::string errorData;
    std::string solutionDesc;
};

std::ostream &operator<<(std::ostream &os, const ExternalError &externalError);
} // namespace atb
#endif