/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <gtest/gtest.h>
#include <atb/utils/log.h>
#include "atb/utils/disk_util.h"
#include "atb/utils/config.h"

TEST(TestDiskUtil, IsDiskAvailable)
{
    std::string filePath = "/tmp";
    bool isAvailable = atb::DiskUtil::IsDiskAvailable(filePath);
}


TEST(TestDiskUtil, IsDiskAvailableFail)
{
    std::string filePath = "/djewoo/dwwdw"; // random input
    bool isAvailable = atb::DiskUtil::IsDiskAvailable(filePath);
    EXPECT_EQ(isAvailable, false);
}