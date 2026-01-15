/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "device_version_check.h"
#include <mki/utils/platform/platform_info.h>

bool IsAscend310P()
{
    return Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_310P;
}

bool IsAscend910A()
{
    return Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910A;
}

bool IsAscend910B()
{
    return Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B;
}

bool IsAscend310B() {
    return Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_310B;
}