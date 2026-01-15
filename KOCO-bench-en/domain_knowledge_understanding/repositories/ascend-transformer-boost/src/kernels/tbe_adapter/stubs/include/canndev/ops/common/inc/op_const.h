/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_STUB_OP_CONST_H
#define ASCEND_OPS_STUB_OP_CONST_H

#include <vector>
#include "external/graph/operator.h"
#include "graph/utils/op_desc_utils.h"
#include "runtime/tiling_context.h"
#include "runtime/infer_shape_context.h"
#include "op_util.h"
#include "context_util.h"

namespace ops {
using namespace ge;

template <typename T1, typename T2>
static void GetDataToVector(const uint8_t *const_data, size_t data_size, std::vector<T1> &result)
{
    if (const_data == nullptr || data_size == 0) {
        return;
    }

    size_t size = data_size / sizeof(T2);
    result.resize(size);
    const T2 *data = reinterpret_cast<const T2 *>(const_data);
    for (size_t i = 0; i < size; i++) {
        result[i] = *(data + i);
    }
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] const_input_idx: constvalue axes index
 * @param [out] values: vector to store return values.
 * @return bool: flag of success or not
 */
template <typename T>
bool GetConstIntData(const ge::Operator &paras, const int64_t const_input_idx, std::vector<T> &values)
{
    return true;
}

/*
 * @brief: read constvalue from paras store into value
 * @param [in] paras: ge::Operator
 * @param [in] const_input_idx: constvalue axes index
 * @param [out] value: integer to store return value.
 * @return bool: flag of success or not
 */
template <typename T> bool GetConstInt(const ge::Operator &paras, const int64_t const_input_idx, T &value)
{
    return true;
}

/*
 * @brief: read constvalue from paras store into value
 * @param [in] context: gert::InferShapeContext
 * @param [in] const_input_idx: constvalue axes index
 * @param [out] value: integer to store return value.
 * @return bool: flag of success or not
 */
template <typename T> bool GetConstInt(gert::InferShapeContext *context, const int64_t const_input_idx, T &value)
{
    return true;
}

/*
 * @brief: read constvalue from paras store into value
 * @param [in] context: gert::TilingContext
 * @param [in] const_input_idx: constvalue axes index
 * @param [out] value: integer to store return value.
 * @return bool: flag of success or not
 */
template <typename T> bool GetConstInt(gert::TilingContext *context, const int64_t const_input_idx, T &value)
{
    if (context == nullptr) {
        return false;
    }

    const gert::Tensor* const_tensor = context->GetInputTensor(const_input_idx);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, const_tensor, false);
    if (!IsConstTensor(const_tensor)) {
        OP_LOGW(context->GetNodeName(), "the input[%ld] is not const tensor, will return failed.", const_input_idx);
        return false;
    }

    ge::DataType dtype = const_tensor->GetDataType();
    switch (dtype) {
        case ge::DT_UINT64:
            value = static_cast<T>(const_tensor->GetData<uint64_t>()[0]);
            break;
        case ge::DT_INT64:
            value = static_cast<T>(const_tensor->GetData<int64_t>()[0]);
            break;
        case ge::DT_UINT32:
            value = static_cast<T>(const_tensor->GetData<uint32_t>()[0]);
            break;
        case ge::DT_INT32:
            value = static_cast<T>(const_tensor->GetData<int32_t>()[0]);
            break;
        default: {
            OP_LOGW(context->GetNodeName(), "GetConstInt only support [int32, int64, uint64, uint32]. but is %s",
                    ops::ToString(dtype).c_str());
            return false;
        }
    }
    OP_LOGD("GetConstInt", "GetConstInt of value is %d", static_cast<int>(value));
    return true;
}

template <typename T> static void GetConstValueToShape(const gert::Tensor *tensor, size_t size, gert::Shape *shape)
{
    if (tensor == nullptr) {
        return;
    }

    const T *value = tensor->GetData<T>();
    if (value == nullptr) {
        return;
    }

    shape->SetDimNum(size);
    for (size_t i = 0; i < size; i++) {
        shape->SetDim(i, value[i]);
    }
}

template <typename T> void GetValueToShape(const gert::Tensor *const_tensor, gert::Shape *const_shape)
{
    if (const_tensor == nullptr || const_shape == nullptr) {
        return;
    }

    const T *const_value = const_tensor->GetData<T>();
    if (const_value == nullptr) {
        return;
    }

    const size_t const_num = const_tensor->GetShapeSize();
    const_shape->SetDimNum(0);
    for (size_t i = 0; i < const_num; ++i) {
        const_shape->AppendDim(const_value[i]);
    }
}

template <typename T> void GetValueToShape(const gert::Tensor *const_tensor, gert::Shape &const_shape)
{
    if (const_tensor == nullptr) {
        return;
    }

    const T *const_value = const_tensor->GetData<T>();
    if (const_value == nullptr) {
        return;
    }

    const size_t const_num = const_tensor->GetShapeSize();
    const_shape.SetDimNum(0);
    for (size_t i = 0; i < const_num; ++i) {
        const_shape.AppendDim(const_value[i]);
    }
}

template <typename T> bool GetConstIntToShape(T *context, const int64_t const_idx, gert::Shape &const_shape)
{
    if (context == nullptr) {
        return false;
    }

    const gert::Tensor *const_tensor = context->GetInputTensor(const_idx);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, const_tensor, false);
    if (!IsConstTensor(const_tensor)) {
        OP_LOGW(context->GetNodeName(), "the input[%lld] is not const tensor, will return failed.", const_idx);
        return false;
    }

    ge::DataType const_dtype = const_tensor->GetDataType();

    switch (const_dtype) {
        case ge::DT_INT32: {
            GetValueToShape<int32_t>(const_tensor, const_shape);
            break;
        }
        case ge::DT_INT64: {
            GetValueToShape<int64_t>(const_tensor, const_shape);
            break;
        }
        case ge::DT_UINT64: {
            GetValueToShape<uint64_t>(const_tensor, const_shape);
            break;
        }
        case ge::DT_UINT32: {
            GetValueToShape<uint32_t>(const_tensor, const_shape);
            break;
        }
        default:
            OP_LOGW(context->GetNodeName(), "GetConstIntToShape only support [int32, int64, uint64, uint32]. but is %s",
                    ops::ToString(const_dtype).c_str());
            return false;
    }

    OP_LOGI(context->GetNodeName(), "GetConstIntToShape: output shape is %s", ToString(const_shape).c_str());
    return true;
}
} // namespace ops
#endif // CANN_OPS_BUILT_IN_OPS_CONST_H_
