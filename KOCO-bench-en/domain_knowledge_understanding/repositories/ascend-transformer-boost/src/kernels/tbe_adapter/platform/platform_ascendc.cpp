/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mutex>
#include <csignal>
#include <tiling/platform/platform_ascendc.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/rt/rt.h>
#include "securec.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"

namespace platform_ascendc {
const static uint64_t LOCAL_RESERV_SIZE = 256;
const static uint32_t WORKSPACE_SIZE_910B = 16 * 1024 * 1024;
const static uint32_t WORKSPACE_SIZE = 2 * 1024 * 1024;
const static uint32_t MIX_AIC_AIV_RATION_910B1 = 2;
const static uint32_t CUBE_GROUP_WORKSPACE_SIZE_910B = 1 * 1024 * 1024;
const static uint32_t GROUP_BARRIER_WORKSPACE_SIZE_910B = 1 * 1024 * 1024;
const static std::string STR_VERSION = "version";
const static std::string STR_SOC_INFO = "SoCInfo";
const static std::string SHORT_SOC_VERSION = "Short_SoC_version";
const static std::string STR_SPLIT_KEY = "core_type_list";
const static std::string STR_SPLIT_VAL = "CubeCore,VectorCore";
const static std::string STR_CORE_CNT_CUB = "cube_core_cnt";
const static std::string STR_CORE_CNT_VEC = "vector_core_cnt";
const static std::string STR_CORE_CNT_AICORE = "ai_core_cnt";
const static std::map<std::string, SocVersion> CONVERT_MAP = {
    {"Ascend310P", SocVersion::ASCEND310P},
    {"Ascend310B", SocVersion::ASCEND310B},
    {"Ascend910", SocVersion::ASCEND910},
    {"Ascend910B", SocVersion::ASCEND910B},
    {"Ascend910_93", SocVersion::ASCEND910B},
};

static inline uint32_t GetCoreNumByType(fe::PlatFormInfos *platformInfo, bool isAiv)
{
    std::string key;
    std::string val;
    bool ret = platformInfo->GetPlatformResWithLock(STR_SOC_INFO, STR_SPLIT_KEY, val);
    MKI_LOG_IF(!ret, ERROR) << "get platform failed, val is " << val;

    if (STR_SPLIT_VAL.compare(val) != 0) {
        key = STR_CORE_CNT_AICORE;
    } else if (isAiv) {
        key = STR_CORE_CNT_VEC;
    } else {
        key = STR_CORE_CNT_CUB;
    }
    ret = platformInfo->GetPlatformResWithLock(STR_SOC_INFO, key, val);
    MKI_LOG_IF(!ret, ERROR) << "get platform failed, key is " << key << ", val is" << val;
    return val.empty() ? 0 : static_cast<uint32_t>(std::atoi(val.c_str()));
}

uint32_t PlatformAscendC::GetCoreNumVector(void) const
{
    if (GetSocVersion() == SocVersion::ASCEND310P) {
        std::string val;
        bool ret = GetPlatFormInfo()->GetPlatformResWithLock(STR_SOC_INFO, STR_CORE_CNT_VEC, val);
        MKI_LOG_IF(!ret, ERROR) << "get platform vector num failed, val is " << val;
        return val.empty() ? 0 : std::atoi(val.c_str());
    }
    return 0;
}

uint32_t PlatformAscendC::GetCoreNumAic(void) const
{
    return GetCoreNumByType(GetPlatFormInfo(), false);
}

uint32_t PlatformAscendC::GetCoreNumAiv(void) const
{
    return GetCoreNumByType(GetPlatFormInfo(), true);
}

uint32_t PlatformAscendC::GetCoreNum(void) const
{
    return GetPlatFormInfo()->GetCoreNum();
}

void PlatformAscendC::GetCoreMemSize(const CoreMemType &memType, uint64_t &size) const
{
    const fe::LocalMemType localType = static_cast<fe::LocalMemType>(memType);
    GetPlatFormInfo()->GetLocalMemSize(localType, size);
    // only ascend910B need UB/L1 local reserved buf for kfc
    if ((memType == CoreMemType::UB || memType == CoreMemType::L1)
         && GetSocVersion() == SocVersion::ASCEND910B) {
        size -= LOCAL_RESERV_SIZE;
    }
}

SocVersion PlatformAscendC::GetSocVersion(void) const
{
    std::string socVersionStr;
    const auto ret = GetPlatFormInfo()->GetPlatformResWithLock(STR_VERSION, SHORT_SOC_VERSION, socVersionStr);
    MKI_CHECK(ret, "get platform failed, socVersionStr is " << socVersionStr,
              return SocVersion::RESERVED_VERSION);

    const auto &it = CONVERT_MAP.find(socVersionStr);
    if (it != CONVERT_MAP.cend()) {
        return it->second;
    }
    MKI_LOG(ERROR) << "get platform failed, convertMap do not find soc " << socVersionStr << " version";
    return SocVersion::RESERVED_VERSION;
}
void PlatformAscendC::GetCoreMemBw(const CoreMemType &memType, uint64_t &bwSize) const
{
    const fe::LocalMemType localType = static_cast<fe::LocalMemType>(memType);
    GetPlatFormInfo()->GetLocalMemBw(localType, bwSize);
}

fe::PlatFormInfos* PlatformAscendC::GetPlatFormInfo(void) const
{
    MKI_CHECK(platformInfo_, "PlatformInfo cannot be initialized to nulltpr!!", raise(SIGABRT));
    return platformInfo_;
}

uint32_t PlatformAscendC::CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum) const
{
    if (aicCoreNum == 0 || aivCoreNum == 0 || aicCoreNum > aivCoreNum) {
        return sliceNum;
    }
    uint32_t ration = aivCoreNum / aicCoreNum;
    uint32_t blockDim = (sliceNum + (ration - 1)) / ration;
    // in mix case: 910B1(ration = 2), blockDim should not be greater than physical aic core num
    if ((ration == MIX_AIC_AIV_RATION_910B1) && (blockDim > aicCoreNum)) {
        MKI_LOG(ERROR) << "CalcTschBlockDim failed, calc blockDim " << blockDim
                       << " should not be greater than aicCoreNum " << aicCoreNum;
        return 0;
    }
    return blockDim;
}

uint32_t PlatformAscendC::GetLibApiWorkSpaceSize(void) const
{
    auto socVersion = GetSocVersion();
    if (socVersion == SocVersion::RESERVED_VERSION) {
        MKI_LOG(ERROR) << "get platform failed, socVersionStr is " << static_cast<int32_t>(socVersion);
        return -1;
    } else if (socVersion == SocVersion::ASCEND910B) {
        return WORKSPACE_SIZE_910B;
    }
    return WORKSPACE_SIZE;
}

uint32_t PlatformAscendC::GetResCubeGroupWorkSpaceSize(void) const
{
    auto socVersion = GetSocVersion();
    if (socVersion == SocVersion::ASCEND910B) {
        return CUBE_GROUP_WORKSPACE_SIZE_910B;
    } else {
        MKI_LOG(ERROR) << "get platform failed, socVersionStr is " << static_cast<int32_t>(socVersion);
        return -1;
    }
}

uint32_t PlatformAscendC::GetResGroupBarrierWorkSpaceSize(void) const
{
    auto socVersion = GetSocVersion();
    if (socVersion == SocVersion::ASCEND910B) {
        return GROUP_BARRIER_WORKSPACE_SIZE_910B;
    } else {
        MKI_LOG(ERROR) << "get platform failed, socVersionStr is " << static_cast<int32_t>(socVersion);
        return -1;
    }
}

PlatformAscendC* PlatformAscendCManager::platformInfo = nullptr;
std::mutex PlatformAscendCManager::platformInitMtx;
SocVersion PlatformAscendCManager::SocVersionMap(const char *socVersionStr)
{
    const auto &iter = CONVERT_MAP.find(socVersionStr);
    if (iter != CONVERT_MAP.cend()) {
        return iter->second;
    }
    MKI_LOG(ERROR) << "get platform failed, convertMap do not find soc " << socVersionStr << " version";
    return SocVersion::RESERVED_VERSION;
}
fe::PlatFormInfos* PlatformAscendCManager::PlatformAscendCInit(const char *customSocVersion)
{
    static fe::PlatFormInfos gPlatformInfo;
    std::string socVersion;

    if (customSocVersion == nullptr) {
        const uint32_t maxLen = 50;
        MKI_CHECK(Mki::MkiRtDeviceGetSocVersion(socVersion, maxLen) == MKIRT_SUCCESS,
            "failed to get soc version", return nullptr);
    } else {
        socVersion = customSocVersion;
    }

    fe::PlatformInfoManager::GeInstance().InitRuntimePlatformInfos(socVersion);
    fe::OptionalInfos optionalInfos;
    fe::PlatformInfoManager::GeInstance().GetPlatformInfos(socVersion,
        gPlatformInfo, optionalInfos);
    std::string socVersionStr;
    const auto ret = gPlatformInfo.GetPlatformResWithLock(STR_VERSION, SHORT_SOC_VERSION, socVersionStr);
    MKI_LOG_IF(!ret, ERROR) << "get platform short version failed, socVersion is " << socVersion;
    SocVersion version = SocVersionMap(socVersionStr.c_str());
    if (version == SocVersion::RESERVED_VERSION) {
        MKI_LOG(ERROR) << "Invalid SocVersion.";
        return nullptr;
    } else if ((version == SocVersion::ASCEND310P) || (version == SocVersion::ASCEND910)) {
        gPlatformInfo.SetCoreNumByCoreType("AiCore");
    } else {
        gPlatformInfo.SetCoreNumByCoreType("VectorCore");
    }
    return &gPlatformInfo;
}
PlatformAscendC* PlatformAscendCManager::PlatformAscendCManagerInit(const char *customSocVersion)
{
    static fe::PlatFormInfos* gPlatformAscendCInfo = PlatformAscendCInit(customSocVersion);
    MKI_CHECK(gPlatformAscendCInfo, "failed to get platformInfo", return nullptr);

    static PlatformAscendC tmp(gPlatformAscendCInfo);
    platformInfo = &tmp;
    return platformInfo;
}
}
