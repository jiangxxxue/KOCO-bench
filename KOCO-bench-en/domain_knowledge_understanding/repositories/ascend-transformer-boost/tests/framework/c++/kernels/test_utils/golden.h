/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MKIOPTESTS_GOLDEN_H
#define MKIOPTESTS_GOLDEN_H
#include "op_test.h"
#include <mki/utils/status/status.h>

namespace Mki {
namespace Test {
class Golden {
public:
    static Status InOutTensorEqual(float atol, float rtol, const GoldenContext &context);
};
} // namespace Test
} // namespace Mki
#endif