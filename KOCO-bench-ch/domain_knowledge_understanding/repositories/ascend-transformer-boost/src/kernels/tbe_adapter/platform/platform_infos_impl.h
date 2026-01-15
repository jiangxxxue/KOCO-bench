/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UTILS_PLATFORM_PLATFORM_INFOS_IMPL_H
#define UTILS_PLATFORM_PLATFORM_INFOS_IMPL_H

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "platform/platform_infos_def.h"

namespace fe {
class PlatFormInfosImpl {
using PlatInfoMapType = std::map<std::string, std::vector<std::string>>;

public:
    PlatInfoMapType GetAICoreIntrinsicDtype();
    PlatInfoMapType GetVectorCoreIntrinsicDtype();
    PlatInfoMapType GetFixPipeDtypeMap();

    void SetPlatformRes(const std::string &label, std::map<std::string, std::string> &result);
    bool GetPlatformRes(const std::string &label, const std::string &key, std::string &value);
    bool GetPlatformRes(const std::string &label, std::map<std::string, std::string> &result);

    void SetFixPipeDtypeMap(const PlatInfoMapType &dtypeMap);
    void SetAICoreIntrinsicDtype(PlatInfoMapType &dtypes);
    void SetVectorCoreIntrinsicDtype(PlatInfoMapType &dtypes);

private:
    PlatInfoMapType aiCoreIntrinsicDtypeMap_;
    PlatInfoMapType vectorCoreIntrinsicDtypeMap_;
    std::map<std::string, std::map<std::string, std::string>> platformResMap_;
    PlatInfoMapType fixpipeDtypeMap_;
};
} // namespace fe

#endif
