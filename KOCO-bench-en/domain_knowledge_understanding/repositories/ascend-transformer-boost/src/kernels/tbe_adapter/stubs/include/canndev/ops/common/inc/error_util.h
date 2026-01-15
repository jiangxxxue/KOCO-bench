/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_STUB_ERROR_UTIL_H
#define ASCEND_OPS_STUB_ERROR_UTIL_H

#include <sstream>
#include <vector>

#include "error_code.h"
#include "graph/operator.h"
#include "op_log.h"

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg)                                                        \
    do {                                                                                                             \
        OP_LOGE(op_name, "%s", get_cstr(err_msg));                                                                   \
    } while (0)

#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...)                                                                 \
    do {                                                                                                             \
        OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);                                                     \
        REPORT_INNER_ERROR("E69999", "op[%s], " err_msg, get_cstr(get_op_info(op_name)), ##__VA_ARGS__);             \
    } while (0)

namespace ge {
template <typename T> std::string DebugString(const std::vector<T> &v)
{
    std::ostringstream oss;
    oss << "[";
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    oss << "]";
    return oss.str();
}

template <typename T> std::string DebugString(const std::vector<std::pair<T, T>> &v)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << "(" << v[i].first << ", " << v[i].second << ")";
    }
    oss << "]";
    return oss.str();
}

inline std::ostream &operator<<(std::ostream &os, const ge::Operator &op) { return os << get_op_info(op); }

/*
 * str cat util function
 * param[in] params need concat to string
 * return concatted string
 */
template <typename T> std::string ConcatString(const T &arg)
{
    std::ostringstream oss;
    oss << arg;
    return oss.str();
}

template <typename T, typename... Ts> std::string ConcatString(const T &arg, const Ts &...arg_left)
{
    std::ostringstream oss;
    oss << arg;
    oss << ConcatString(arg_left...);
    return oss.str();
}

template <typename T> std::string Shape2String(const T &shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

std::string GetViewErrorCodeStr(ge::ViewErrorCode errCode);

std::string GetShapeErrMsg(uint32_t index, const std::string &wrong_shape, const std::string &correct_shape);

std::string GetAttrValueErrMsg(const std::string &attr_name, const std::string &wrong_val,
                               const std::string &correct_val);

std::string GetAttrSizeErrMsg(const std::string &attr_name, const std::string &wrong_size,
                              const std::string &correct_size);

std::string GetInputInvalidErrMsg(const std::string &param_name);
std::string GetShapeSizeErrMsg(uint32_t index, const std::string &wrong_shape_size,
                               const std::string &correct_shape_size);

std::string GetInputFormatNotSupportErrMsg(const std::string &param_name, const std::string &expected_format_list,
                                           const std::string &data_format);

std::string GetInputDtypeNotSupportErrMsg(const std::string &param_name, const std::string &expected_dtype_list,
                                          const std::string &data_dtype);

std::string GetInputDTypeErrMsg(const std::string &param_name, const std::string &expected_dtype,
                                const std::string &data_dtype);

std::string GetInputFormatErrMsg(const std::string &param_name, const std::string &expected_format,
                                 const std::string &data_format);

std::string SetAttrErrMsg(const std::string &param_name);
std::string UpdateParamErrMsg(const std::string &param_name);

template <typename T>
std::string GetParamOutRangeErrMsg(const std::string &param_name, const T &real_value, const T &min, const T &max);

std::string OtherErrMsg(const std::string &error_detail);

void TbeInputDataTypeErrReport(const std::string &op_name, const std::string &param_name,
                               const std::string &expected_dtype_list, const std::string &dtype);

void GeInfershapeErrReport(const std::string &op_name, const std::string &op_type, const std::string &value,
                           const std::string &reason);
/*
 * log common runtime error
 * param[in] opname op name
 * param[in] error description
 * return void
 */
void CommonRuntimeErrLog(const std::string &opname, const std::string &description);
} // namespace ge

#endif