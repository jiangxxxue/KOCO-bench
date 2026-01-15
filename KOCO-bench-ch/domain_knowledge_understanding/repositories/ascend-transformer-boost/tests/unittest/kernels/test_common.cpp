/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "test_common.h"

at::IntArrayRef ToIntArrayRef(const Mki::SVector<int64_t> &src) { return at::IntArrayRef(src.data(), src.size()); }

at::IntArrayRef ToIntArrayRef(const std::vector<int64_t> &src) { return at::IntArrayRef(src.data(), src.size()); }

int64_t Prod(const Mki::SVector<int64_t> &vec)
{
    int64_t ret = 1;
    for (auto x : vec) {
        ret *= x;
    }
    return ret;
}