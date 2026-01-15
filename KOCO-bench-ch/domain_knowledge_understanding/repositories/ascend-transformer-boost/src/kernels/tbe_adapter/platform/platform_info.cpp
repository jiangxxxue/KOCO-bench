/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "platform/platform_info.h"

namespace fe {
PlatformInfoManager& PlatformInfoManager::Instance()
{
    static PlatformInfoManager manager;
    return manager;
}

PlatformInfoManager& PlatformInfoManager::GeInstance()
{
    return Instance();
}

uint32_t PlatformInfoManager::InitializePlatformInfo()
{
    return 1;
}

uint32_t PlatformInfoManager::GetPlatformInfoWithOutSocVersion(
    PlatformInfo &platformInfo, OptionalInfo &optiCompilationInfo)
{
    (void)platformInfo;
    (void)optiCompilationInfo;
    return 1;
}

uint32_t PlatformInfoManager::GetPlatformInfoWithOutSocVersion(
    PlatFormInfos &platformInfo, OptionalInfos &optiCompilationInfo)
{
    (void)platformInfo;
    (void)optiCompilationInfo;
    return 1;
}

uint32_t PlatformInfoManager::InitRuntimePlatformInfos(const std::string &socVersion)
{
    (void)socVersion;
    return 1;
}

uint32_t PlatformInfoManager::GetPlatformInfos(
    const string SoCVersion, PlatFormInfos &platform_info, OptionalInfos &opti_compilation_info)
{
    (void)SoCVersion;
    (void)platform_info;
    (void)opti_compilation_info;
    return 1;
}

PlatformInfoManager::PlatformInfoManager() {}
PlatformInfoManager::~PlatformInfoManager() {}
}