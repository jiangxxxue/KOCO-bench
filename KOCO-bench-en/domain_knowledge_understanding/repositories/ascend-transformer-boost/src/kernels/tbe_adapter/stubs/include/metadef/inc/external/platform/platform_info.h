/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PLATFORM_INFO_H
#define PLATFORM_INFO_H

#include <cstdint>
#include <map>
#include <string>

using namespace std;
#define UNUSED_VALUE(x) (void)(x)

namespace fe {
class PlatformInfo {};
class PlatFormInfos {
public:
    bool GetPlatformResWithLock(const std::string &label, const std::string &key, std::string &val);

    bool GetPlatformResWithLock(const std::string &label, std::map<std::string, std::string> &res);
};
class OptionalInfo {};
class OptionalInfos {};
class PlatformInfoManager {
public:
    PlatformInfoManager(const PlatformInfoManager &) = delete;
    PlatformInfoManager &operator=(const PlatformInfoManager &) = delete;

    static PlatformInfoManager &Instance();
    static PlatformInfoManager &GeInstance();
    uint32_t InitializePlatformInfo();

    uint32_t GetPlatformInfoWithOutSocVersion(PlatformInfo &platformInfo, OptionalInfo &optiCompilationInfo);

    uint32_t GetPlatformInfoWithOutSocVersion(PlatFormInfos &platformInfo, OptionalInfos &optiCompilationInfo);

    uint32_t InitRuntimePlatformInfos(const std::string &socVersion);

private:
    PlatformInfoManager();
};
} // namespace fe

#endif
