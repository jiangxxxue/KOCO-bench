/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_STUB_OP_TILING_UTIL_H
#define ASCEND_OPS_STUB_OP_TILING_UTIL_H

#include <nlohmann/json.hpp>
#include "op_attr.h"
#include "op_util.h"
#include "op_tiling.h"
#include "error_util.h"

using namespace ge;

namespace optiling {
ge::DataType GetGeTypeFromStr(const std::string &dtype_str);

template <typename T>
bool GetCompileValue(const nlohmann::json& all_vars, const std::string& name, T& value)
{
    if (all_vars.empty()) {
        return false;
    }

    if (all_vars.count(name) == 0) {
        return false;
    }

    value = all_vars[name].get<T>();
    return true;
}

template <typename T1, typename T2>
bool GetCompileValue(const nlohmann::json& all_vars, const std::string& name, T1& value, const T2 default_value)
{
    if (!GetCompileValue(all_vars, name, value)) {
        value = static_cast<T1>(default_value);
    }
    return true;
}
}

#endif