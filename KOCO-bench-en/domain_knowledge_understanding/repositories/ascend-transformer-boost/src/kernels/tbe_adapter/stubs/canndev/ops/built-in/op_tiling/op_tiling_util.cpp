/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "op_tiling_util.h"

namespace optiling {
const std::map<std::string, DataType> STR_TO_DATATYPE = {{"float", DT_FLOAT},
                                                         {"float32", DT_FLOAT},
                                                         {"float16", DT_FLOAT16},
                                                         {"int8", DT_INT8},
                                                         {"int16", DT_INT16},
                                                         {"int32", DT_INT32},
                                                         {"int64", DT_INT64},
                                                         {"uint8", DT_UINT8},
                                                         {"uint16", DT_UINT16},
                                                         {"uint32", DT_UINT32},
                                                         {"uint64", DT_UINT64},
                                                         {"bool", DT_BOOL},
                                                         {"double", DT_DOUBLE},
                                                         {"dual", DT_DUAL},
                                                         {"dual_sub_int8", DT_DUAL_SUB_INT8},
                                                         {"dual_sub_uint8", DT_DUAL_SUB_UINT8},
                                                         {"int4", DT_INT4},
                                                         {"bfloat16", DT_BF16}};

ge::DataType GetGeTypeFromStr(const std::string &dtype_str)
{
    auto it = STR_TO_DATATYPE.find(dtype_str);
    if (it != STR_TO_DATATYPE.end()) {
        return it->second;
    }
    OP_LOGW("GetGeTypeFromStr", "con not get the dtype[%s] in ge::DataType list. will return DT_MAX",
            dtype_str.c_str());
    return DT_MAX;
}
} // namespace optiling