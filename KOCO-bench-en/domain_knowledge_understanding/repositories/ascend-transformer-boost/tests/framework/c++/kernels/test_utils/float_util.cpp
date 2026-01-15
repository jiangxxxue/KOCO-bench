/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "float_util.h"
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>

namespace Mki {
namespace Test {
bool FloatUtil::FloatJudgeEqual(float expected, float actual, float atol, float rtol)
{
    bool judge = std::abs(expected - actual) <= (atol + rtol * std::abs(actual));
    MKI_LOG_IF(!judge, ERROR) << "judge float expected: " << expected << ", actual: " << actual;
    return judge;
}
} // namespace Test
} // namespace Mki