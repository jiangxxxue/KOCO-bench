/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_OP_PARAM_FUNC_H
#define ATB_OP_PARAM_FUNC_H

#define OPERATION_PARAM_FUNCS(OpName, OpParamType)                                                                     \
    template <> Status CreateOperation(const OpParamType &opParam, Operation **operation)                              \
    {                                                                                                                  \
        if (operation == nullptr) {                                                                                    \
            return ERROR_INVALID_PARAM;                                                                                \
        }                                                                                                              \
        for (uint8_t i : opParam.rsv) {                                                                                \
            if (i != 0) {                                                                                              \
                ATB_LOG(ERROR) << "param rsv has a non-zero value, please check the compilation version.";             \
                return ERROR_INVALID_PARAM;                                                                            \
            }                                                                                                          \
        }                                                                                                              \
        if (!ParamCheck(opParam)) {                                                                                    \
            ATB_LOG(ERROR) << "ParamCheck failed!";                                                                    \
            return ERROR_INVALID_PARAM;                                                                                \
        }                                                                                                              \
        *operation = new (std::nothrow) OpName(opParam);                                                               \
        if (*operation == nullptr) {                                                                                   \
            return ERROR_OUT_OF_HOST_MEMORY;                                                                           \
        }                                                                                                              \
        return NO_ERROR;                                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    template <> Status CloneOperationParam(const Operation *operation, OpParamType &opParam)                           \
    {                                                                                                                  \
        if (operation == nullptr) {                                                                                    \
            return ERROR_INVALID_PARAM;                                                                                \
        }                                                                                                              \
        const OpName *op = dynamic_cast<const OpName *>(operation);                                                    \
        if (op) {                                                                                                      \
            opParam = op->GetParam();                                                                                  \
            return NO_ERROR;                                                                                           \
        }                                                                                                              \
        ATB_LOG(ERROR) << "Operation type does not match " #OpName " type.";                                           \
        return ERROR_INVALID_PARAM;                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    template <> Status UpdateOperationParam(Operation *operation, const OpParamType &opParam)                          \
    {                                                                                                                  \
        if (operation == nullptr) {                                                                                    \
            return ERROR_INVALID_PARAM;                                                                                \
        }                                                                                                              \
        OpName *op = dynamic_cast<OpName *>(operation);                                                                \
        if (op == nullptr) {                                                                                           \
            ATB_LOG(ERROR) << "Operation type does not match " #OpName " type.";                                       \
            return ERROR_INVALID_PARAM;                                                                                \
        }                                                                                                              \
        if (!ParamCheck(opParam)) {                                                                                    \
            ATB_LOG(ERROR) << "ParamCheck failed!";                                                                    \
            return ERROR_INVALID_PARAM;                                                                                \
        }                                                                                                              \
        if (opParam == op->GetParam()) {                                                                               \
            return NO_ERROR;                                                                                           \
        }                                                                                                              \
        ATB_LOG(DEBUG) << "Param Changed!";                                                                            \
        op->SetParam(opParam);                                                                                         \
        return NO_ERROR;                                                                                               \
    }

#define OP_PARAM_RSV_CHECK(opParam)                                                                                    \
    do {                                                                                                               \
        for (uint8_t i : (opParam).rsv) {                                                                              \
            if (i != 0) {                                                                                              \
                ATB_LOG(ERROR) << "param rsv has a non-zero value, please check the compilation version.";             \
                return ERROR_INVALID_PARAM;                                                                            \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

#endif