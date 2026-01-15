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
#include "atb/utils/tensor_util.h"
#include "atb/utils/config.h"


TEST(TestConfig, IsStreamSyncEveryRunnerEnable)
{
    setenv("ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE", "0", true);
    atb::Config config;
    bool IsEnable = config.IsStreamSyncEveryRunnerEnable();
    EXPECT_EQ(IsEnable, false);
}

TEST(TestConfig, InitKernelCacheZero)
{
    setenv("ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT", "0", true);
    setenv("ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT", "0", true);
    atb::Config config;
    EXPECT_EQ(config.GetLocalKernelCacheCount(), 1);
    EXPECT_EQ(config.GetLocalKernelCacheCount(), 1);
}

TEST(TestConfig, InitKernelCacheMax)
{
    setenv("ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT", "1025", true);
    setenv("ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT", "1025", true);
    atb::Config config;
    EXPECT_EQ(config.GetLocalKernelCacheCount(), 1024);
    EXPECT_EQ(config.GetLocalKernelCacheCount(), 1024);
}