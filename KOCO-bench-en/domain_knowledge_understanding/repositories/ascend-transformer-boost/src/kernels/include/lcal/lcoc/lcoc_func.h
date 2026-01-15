/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCAL_LCOC_FUNC_H
#define LCAL_LCOC_FUNC_H

#include <vector>
#include <string>
#include <lcal_types.h>
#include <lcoc_args.h>
#include <lcoc_base.h>

#pragma once
namespace Lcal {
    // 校验参数取值范围在[min, max]内,当max=-1时，表示参数取值范围在[min, +∞)
    bool CheckParamScope(const std::string &name, const int &value, const int &min, const int &max);
    bool CheckParamScopeList(std::vector<std::tuple<std::string, int, int, int>> paramCheckList);
    bool CheckParamAlign(const std::string &name, const int &value, const int &align);
    void PrintErrorLog(LcalType lcalType, const std::string &log);
    bool CheckParamPowerOfTwo(const std::string &name, int value);
}

#endif // LCAL_LCOC_FUNC_H