/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_COMMON_H
#define ASCEND_OPS_COMMON_H

#include <mki/utils/SVector/SVector.h>

namespace AsdOps {
template<class T>
bool CheckParamAllPositive(Mki::SVector<T> &param)
{
    for (size_t i = 0; i < param.size(); i++) {
        if (param[i] <= 0) {
            return false;
        }
    }
    return true;
}
template<class T>
bool CheckParamAnyNegative(Mki::SVector<T> &param)
{
    for (size_t i = 0; i < param.size(); i++) {
        if (param[i] < 0) {
            return true;
        }
    }
    return false;
}
} // namespace AsdOps

#endif