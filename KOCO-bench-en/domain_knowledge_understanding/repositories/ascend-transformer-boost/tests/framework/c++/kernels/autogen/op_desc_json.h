/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TESTS_AUTOGEN_OP_DESC_JSON_H
#define TESTS_AUTOGEN_OP_DESC_JSON_H

#include <nlohmann/json.hpp>
#include <string>
#include "mki/launch_param.h"
#include "mki/operation.h"

namespace Mki {
namespace AutoGen {
void JsonToOpParam(const nlohmann::json &opDescJson, Mki::LaunchParam &launchParam);
Mki::Operation *GetOpByName(const std::string &operationName);
}
}

#endif
