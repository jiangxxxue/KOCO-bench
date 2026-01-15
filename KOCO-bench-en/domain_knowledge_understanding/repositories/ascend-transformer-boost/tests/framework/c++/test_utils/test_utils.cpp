/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "test_utils.h"
#include <string>
#include <acl/acl_rt.h>
#include <atb/utils/log.h>
#include "atb/context/context_base.h"
#include "atb/types.h"

namespace atb {
bool IsNot910B(void *obj)
{
    const char *socName = aclrtGetSocName();
    if (!socName) {
        ATB_LOG(ERROR) << "aclrtGetSocName failed!";
        return false;
    }
    ATB_LOG(INFO) << "SocVersion:" << std::string(socName);
    return std::string(socName).find("Ascend910B") == std::string::npos &&
        std::string(socName).find("Ascend910_93") == std::string::npos;
}

bool IsNot310P(void *obj)
{
    const char *socName = aclrtGetSocName();
    if (!socName) {
        ATB_LOG(ERROR) << "aclrtGetSocName failed!";
        return false;
    }
    ATB_LOG(INFO) << "SocVersion:" << std::string(socName);
    return std::string(socName).find("Ascend310P") == std::string::npos;
}

Context *CreateContextAndExecuteStream()
{
    int deviceId = -1;
    ATB_LOG(INFO) << "aclrtGetDevice start";
    int ret = aclrtGetDevice(&deviceId);
    ATB_LOG(INFO) << "aclrtGetDevice ret:" << ret << ", deviceId:" << deviceId;
    if (ret != 0 || deviceId < 0) {
        const char *envStr = std::getenv("SET_NPU_DEVICE");
        deviceId = (envStr != nullptr) ? atoi(envStr) : 0;
        ATB_LOG(INFO) << "aclrtSetDevice deviceId:" << deviceId;
        ret = aclrtSetDevice(deviceId);
        ATB_LOG_IF(ret != 0, ERROR) << "aclrtSetDevice fail, ret:" << ret << ", deviceId:" << deviceId;
    }

    atb::Context *context = nullptr;
    Status st = atb::CreateContext(&context);
    ATB_LOG_IF(st != 0, ERROR) << "CreateContext fail";
    aclrtStream stream = nullptr;
    st = aclrtCreateStream(&stream);
    ATB_LOG_IF(st != 0, ERROR) << "aclrtCreateStream fail";
    context->SetExecuteStream(stream);
    return context;
}
} // namespace atb