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
#include <mki/utils/share_memory/share_memory.h>
#include "atb/utils/tensor_util.h"
#include "atb/utils/config.h"
#include "atb/runner/hccl_runner.h"
#include "atb/utils.h"

TEST(TestShareMemory, ShareMemory)
{
    HcclRootInfo hcclCommId = {};
    ATB_LOG(INFO) << "HCCL Runner Init Begin";
    std::string shmName = "hcclShareMem";
    Mki::ShareMemory shm(shmName, sizeof(atb::CommInitInfo));
    auto *shmInfo = (atb::CommInitInfo *)shm.GetShm();
    ATB_LOG(INFO) << "create share memory success";
    EXPECT_NE(shmInfo, nullptr);
}