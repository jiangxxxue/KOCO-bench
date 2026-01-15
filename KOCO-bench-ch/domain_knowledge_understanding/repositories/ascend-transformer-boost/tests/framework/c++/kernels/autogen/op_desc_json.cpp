
/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <functional>
#include <map>
#include "mki/utils/log/log.h"
#include "asdops/params/params.h"
#include "op_desc_json.h"

namespace Mki {
namespace AutoGen {

Mki::Operation *GetOpByNameAtbOps(const std::string &operationName);
Mki::Operation *GetOpByNameAsdOps(const std::string &operationName);
bool JsonToOpParamAtbOps(const nlohmann::json &opDescJson, Mki::LaunchParam &launchParam);
bool JsonToOpParamAsdOps(const nlohmann::json &opDescJson, Mki::LaunchParam &launchParam);

void JsonToOpParam(const nlohmann::json &opDescJson, Mki::LaunchParam &launchParam)
{
    if (JsonToOpParamAtbOps(opDescJson, launchParam)) {
        return;
    }
    if (!JsonToOpParamAsdOps(opDescJson, launchParam)) {
        MKI_LOG(ERROR) << "no opName: " << opDescJson["opName"];
    }
    return;
}

Mki::Operation *GetOpByName(const std::string &operationName)
{
    Mki::Operation *op = GetOpByNameAtbOps(operationName);
    if (op == nullptr) {
        op = GetOpByNameAsdOps(operationName);
    }
    return op;
}
}
}