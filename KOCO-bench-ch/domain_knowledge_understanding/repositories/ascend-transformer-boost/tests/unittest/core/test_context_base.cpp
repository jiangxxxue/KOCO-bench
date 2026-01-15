/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
// #include <vector>
// #include <string>
// #include <cstring>
// #include <cstdlib>
// #include <gtest/gtest.h>
// #include <atb/utils/log.h>
// #include "atb/utils/tensor_util.h"
// #include "tests/test_utils/operation_test.h"
// #include "atb/utils/config.h"
// #include "tests/pythontest/pytorch/adapter/utils/utils.h"
// #include "test_utils/context/memory_context.h"
// #include "atb/types.h"
// #include "atb/context/context_base.h"
// #include "atb/operation/operation_base.h"
// #include "core/ops/split/split_operation.h"

// TEST(TestContextBase, ForwardTilingCopyEvent)
// {
//     atb::Context *context = nullptr;
//     atb::Status st = atb::CreateContext(&context, MULTI_STREAM_MASK);
//     atb::ContextBase *contextBase = dynamic_cast<atb::ContextBase *>(context);
//     ASSERT_EQ(st, atb::NO_ERROR);
//     aclrtStream stream = nullptr;
//     st = aclrtCreateStream(&stream);
//     ASSERT_EQ(st, atb::NO_ERROR);
//     contextBase->SetExecuteStream(stream);
//     void *tilingCopyStream = contextBase->GetTilingCopyStream();
//     void *executeStream = contextBase->GetExecuteStream();

//     atb::Status ret = atb::NO_ERROR;
//     ret = aclrtRecordEvent(contextBase->GetTilingCopyEvent(), tilingCopyStream) + st;
//     ret = aclrtStreamWaitEvent(executeStream, contextBase->GetTilingCopyEvent()) + st;
//     ret = aclrtResetEvent(contextBase->GetTilingCopyEvent(), executeStream) + st;
//     contextBase->ForwardTilingCopyEvent();
//     EXPECT_NE(ret, atb::NO_ERROR);
//     atb::DestroyContext(context);
// }
