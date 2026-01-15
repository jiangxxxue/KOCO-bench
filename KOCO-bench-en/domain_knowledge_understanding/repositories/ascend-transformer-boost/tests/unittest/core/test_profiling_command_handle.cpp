/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include <iostream>
#include <torch/torch.h>
#include <atb/utils/log.h>
#include <acl/acl.h>
#include <cpp-stub/src/stub.h>
#include "test_utils/test_common.h"
#include "atb/operation.h"
#include "atb/infer_op_params.h"
#include "atb/utils/tensor_util.h"
#include "test_utils/operation_test.h"
#include <atb/utils/probe.h>
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"
#include <mki/utils/log/log.h>
#include "mki/utils/profiling/prof_api.h"
#include "mki/utils/profiling/profiling_funcs.h"

using namespace atb;
using namespace Mki;


TEST(TestProfiling, TestCommandHandle)
{
    uint32_t type = PROF_CTRL_SWITCH;
    uint32_t len = sizeof(MsprofCommandHandle);
    MsprofCommandHandle command;
    command.type = PROF_COMMANDHANDLE_TYPE_START;

    command.profSwitch = PROF_TASK_TIME_L0;
    GetSingleton<Mki::ProfilingFuncs>().MkiProfCommandHandle(type, &command, len);
    ASSERT_EQ(GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel0Status(),true);
    ASSERT_EQ(GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel1Status(),false);

    command.profSwitch = PROF_TASK_TIME_L0 | PROF_TASK_TIME_L1;
    GetSingleton<Mki::ProfilingFuncs>().MkiProfCommandHandle(type, &command, len);
    ASSERT_EQ(GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel0Status(),true);
    ASSERT_EQ(GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel1Status(),true);

    command.type = PROF_COMMANDHANDLE_TYPE_INIT;
    command.profSwitch = 0;
    GetSingleton<Mki::ProfilingFuncs>().MkiProfCommandHandle(type, &command, len);
    ASSERT_EQ(GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel0Status(),true);
    ASSERT_EQ(GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel1Status(),true);

    command.type = PROF_COMMANDHANDLE_TYPE_INIT;
    command.profSwitch = PROF_TASK_TIME_L0;
    GetSingleton<Mki::ProfilingFuncs>().MkiProfCommandHandle(type, &command, len);
    ASSERT_EQ(GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel0Status(),true);
    ASSERT_EQ(GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel1Status(),true);

    command.type = PROF_COMMANDHANDLE_TYPE_INIT;
    command.profSwitch = PROF_TASK_TIME_L0 | PROF_TASK_TIME_L1;
    GetSingleton<Mki::ProfilingFuncs>().MkiProfCommandHandle(type, &command, len);
    ASSERT_EQ(GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel0Status(),true);
    ASSERT_EQ(GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel1Status(),true);

    command.type = PROF_COMMANDHANDLE_TYPE_STOP;
    command.profSwitch = PROF_TASK_TIME_L0 | PROF_TASK_TIME_L1;
    GetSingleton<Mki::ProfilingFuncs>().MkiProfCommandHandle(type, &command, len);
    ASSERT_EQ(GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel0Status(),false);
    ASSERT_EQ(GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel1Status(),false);
}

