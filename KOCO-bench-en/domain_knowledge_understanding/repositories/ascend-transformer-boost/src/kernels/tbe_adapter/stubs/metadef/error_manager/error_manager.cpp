/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "common/util/error_manager/error_manager.h"

#define UNUSED_VALUE(x) (void)(x)
using char_t = char;

namespace error_message {
int32_t FormatErrorMessage(char_t *str_dst, size_t dst_max, const char_t *format, ...)
{
    UNUSED_VALUE(str_dst);
    UNUSED_VALUE(dst_max);
    UNUSED_VALUE(format);
    return 0;
}

void ReportInnerError(const char_t *file_name, const char_t *func, uint32_t line, const std::string error_code,
                      const char_t *format, ...)
{
    UNUSED_VALUE(file_name);
    UNUSED_VALUE(func);
    UNUSED_VALUE(line);
    UNUSED_VALUE(error_code);
    UNUSED_VALUE(format);
}
}

ErrorManager &ErrorManager::GetInstance()
{
    static ErrorManager errorManager;
    return errorManager;
}

int32_t ErrorManager::ReportInterErrMessage(const std::string error_code, const std::string &error_msg)
{
    UNUSED_VALUE(error_code);
    UNUSED_VALUE(error_msg);
    return 0;
}

int32_t ErrorManager::ReportErrMessage(const std::string error_code,
                                       const std::map<std::string, std::string> &args_map)
{
    UNUSED_VALUE(error_code);
    UNUSED_VALUE(args_map);
    return 0;
}

std::string ErrorManager::GetErrorMessage()
{
    return "";
}

std::string ErrorManager::GetWarningMessage()
{
    return "";
}

int32_t ErrorManager::OutputErrMessage(int32_t handle)
{
    UNUSED_VALUE(handle);
    return 0;
}

int32_t ErrorManager::OutputMessage(int32_t handle)
{
    UNUSED_VALUE(handle);
    return 0;
}

namespace ge {
int32_t ReportInnerErrMsg(const char *file_name, const char *func, uint32_t line, const char *error_code,
                          const char *format, ...)
{
    UNUSED_VALUE(file_name);
    UNUSED_VALUE(func);
    UNUSED_VALUE(line);
    UNUSED_VALUE(error_code);
    UNUSED_VALUE(format);
    return 0;
}

int32_t ReportUserDefinedErrMsg(const char *error_code, const char *format, ...)
{
    UNUSED_VALUE(error_code);
    UNUSED_VALUE(format);
    return 0;
}

int32_t ReportPredefinedErrMsg(const char *error_code, const std::vector<const char *> &key,
                               const std::vector<const char *> &value)
{
    UNUSED_VALUE(error_code);
    UNUSED_VALUE(key);
    UNUSED_VALUE(value);
    return 0;
}
}