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
 * \file auto_tiling_register.h
 * \brief
 */
#ifndef ASCEND_OPS_STUB_AUTO_TILING_REGISTER_H
#define ASCEND_OPS_STUB_AUTO_TILING_REGISTER_H

#include "vector_tiling_rt2.h"

#include <vector>

using AutoTilingFunc = bool (*)(gert::TilingContext *, const optiling::OpInfoImpl *);
using AutoTilingParseFunc = optiling::AutoTilingCompileInfo *(*)(const char *op_type,
                                                                 const nlohmann::json &json_compile_info);

#define REGISTER_AUTO_TILING(pattern, tilingfunc, parsefunc)                                                           \
    static AutoTilingRegister g_auto_tiling_register_##__COUNTER__(pattern, tilingfunc, parsefunc)

constexpr size_t PATTERN_BASE = 0x10;
constexpr size_t PATTERN_SIZE = static_cast<size_t>(optiling::SchPattern::DEFAULT) - PATTERN_BASE;

inline size_t PatternIndex(optiling::SchPattern _pattern) { return static_cast<size_t>(_pattern) - PATTERN_BASE; }

class AutoTilingRegister {
public:
    AutoTilingRegister(optiling::SchPattern _pattern, AutoTilingFunc _tiling_func, AutoTilingParseFunc _parser)
    {
        size_t index = PatternIndex(_pattern);
        auto &register_parser = RegisterParser();
        register_parser[index] = _parser;
        auto &register_tiling = RegisterTiling();
        register_tiling[index] = _tiling_func;
    };
    ~AutoTilingRegister() = default;
    static std::array<AutoTilingParseFunc, PATTERN_SIZE> &RegisterParser();
    static std::array<AutoTilingFunc, PATTERN_SIZE> &RegisterTiling();
};

#endif // ASCEND_OPS_STUB_AUTO_TILING_REGISTER_H
