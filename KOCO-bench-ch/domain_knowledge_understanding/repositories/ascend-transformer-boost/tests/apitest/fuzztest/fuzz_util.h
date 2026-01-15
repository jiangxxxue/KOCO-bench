/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef FUZZ_UTIL_H
#define FUZZ_UTIL_H
#include <cstdlib>
#include <ctime>
#include <acl/acl.h>
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/operation.h"
#include "test_utils/operation_test.h"

namespace atb {
namespace FuzzUtil {
    extern const std::vector<std::string> errorType_;
    aclDataType GetRandomAclDataType(int input);
    aclFormat GetRandomAclFormat(int input);
    bool GetRandomBool(uint32_t &fuzzIndex);
    TensorDesc GetRandomTensorDesc(uint32_t &fuzzIndex);
    Status SetupAndExecute(Operation *operation, SVector<TensorDesc> inTensorDescs, SVector<TensorDesc> outTensorDescs);
    uint64_t GetRandomDimNum(uint32_t input);
}
}
#endif
