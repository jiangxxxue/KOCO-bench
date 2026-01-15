/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform/platform_infos_def.h"
#include <mutex>
#include "mki/utils/dl/dl.h"
#include "mki/utils/env/env.h"
#include "mki/utils/log/log.h"
#include "platform_infos_impl.h"

namespace fe {
constexpr uint32_t MAX_CORE_NUM = 128;
std::mutex g_asdopsFePlatMutex;

bool PlatFormInfos::Init()
{
    platform_infos_impl_ = std::make_shared<PlatFormInfosImpl>();
    if (platform_infos_impl_ == nullptr) {
        return false;
    }
    return true;
}

bool PlatFormInfos::GetPlatformResWithLock(const std::string &label, const std::string &key, std::string &val)
{
    std::lock_guard<std::mutex> lockGuard(g_asdopsFePlatMutex);
    if (platform_infos_impl_ == nullptr) {
        return false;
    }
    return platform_infos_impl_->GetPlatformRes(label, key, val);
}

bool PlatFormInfos::GetPlatformResWithLock(const std::string &label, std::map<std::string, std::string> &res)
{
    std::lock_guard<std::mutex> lockGuard(g_asdopsFePlatMutex);
    if (platform_infos_impl_ == nullptr) {
        return false;
    }
    return platform_infos_impl_->GetPlatformRes(label, res);
}

std::map<std::string, std::vector<std::string>> PlatFormInfos::GetAICoreIntrinsicDtype()
{
    if (platform_infos_impl_ == nullptr) {
        return {};
    }
    return platform_infos_impl_->GetAICoreIntrinsicDtype();
}

std::map<std::string, std::vector<std::string>> PlatFormInfos::GetVectorCoreIntrinsicDtype()
{
    if (platform_infos_impl_ == nullptr) {
        return {};
    }
    return platform_infos_impl_->GetVectorCoreIntrinsicDtype();
}

bool PlatFormInfos::GetPlatformRes(const std::string &label, const std::string &key, std::string &val)
{
    if (platform_infos_impl_ == nullptr) {
        return false;
    }
    return platform_infos_impl_->GetPlatformRes(label, key, val);
}

bool PlatFormInfos::GetPlatformRes(const std::string &label, std::map<std::string, std::string> &res)
{
    if (platform_infos_impl_ == nullptr) {
        return false;
    }
    return platform_infos_impl_->GetPlatformRes(label, res);
}

void PlatFormInfos::SetAICoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsicDtypes)
{
    if (platform_infos_impl_ == nullptr) {
        return;
    }
    platform_infos_impl_->SetAICoreIntrinsicDtype(intrinsicDtypes);
}

void PlatFormInfos::SetVectorCoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsicDtypes)
{
    if (platform_infos_impl_ == nullptr) {
        return;
    }
    platform_infos_impl_->SetVectorCoreIntrinsicDtype(intrinsicDtypes);
}

void PlatFormInfos::SetFixPipeDtypeMap(const std::map<std::string, std::vector<std::string>> &fixpipeDtypeMap)
{
    if (platform_infos_impl_ == nullptr) {
        return;
    }
    platform_infos_impl_->SetFixPipeDtypeMap(fixpipeDtypeMap);
}

using AclrtGetResInCurrentThreadFunc = int(*)(int, uint32_t*);

void PlatFormInfos::SetCoreNumByCoreType(const std::string &core_type)
{
    uint32_t coreNum = 0;
    Mki::Dl dl = Mki::Dl(std::string(Mki::GetEnv("ASCEND_HOME_PATH")) + "/runtime/lib64/libascendcl.so", false);
    AclrtGetResInCurrentThreadFunc aclrtGetResInCurrentThread =
        (AclrtGetResInCurrentThreadFunc)dl.GetSymbol("aclrtGetResInCurrentThread");
    if (aclrtGetResInCurrentThread != nullptr) {
        int8_t resType = core_type == "VectorCore" ? 1 : 0;
        int getResRet = aclrtGetResInCurrentThread(resType, &coreNum);
        if (getResRet == 0) {
            core_num_ = coreNum;
            MKI_LOG(DEBUG) << "Get ThreadResource::core_num_ to " << core_type << ": " << coreNum;
            if (core_num_ == 0 || core_num_ > MAX_CORE_NUM) {
                MKI_LOG(ERROR) << "core_num is out of range : " << core_num_;
                core_num_ = 1;
            }
            return;
        } else {
            MKI_LOG(WARN) << "Failed to get thread core num!";
        }
    } else {
        MKI_LOG(WARN) << "Failed to acl function!";
    }
    std::string coreNumStr = "";
    std::string coreTypeStr = "";
    if (core_type == "VectorCore") {
        coreTypeStr = "vector_core_cnt";
    } else {
        coreTypeStr = "ai_core_cnt";
    }
    std::lock_guard<std::mutex> lockGuard(g_asdopsFePlatMutex);
    (void)GetPlatformRes("SoCInfo", coreTypeStr, coreNumStr);
    MKI_LOG(DEBUG) << "Set PlatFormInfos::core_num_ to " << coreTypeStr << ": " << coreNumStr;
    if (coreNumStr.empty()) {
        core_num_ = 1;
        MKI_LOG(ERROR) << "CoreNumStr is empty!";
    } else {
        core_num_ = std::strtoul(coreNumStr.c_str(), nullptr, 10); // 10 进制
    }
    if (core_num_ == 0 || core_num_ > MAX_CORE_NUM) {
        MKI_LOG(ERROR) << "core_num is out of range : " << core_num_;
        core_num_ = 1;
    }
}

uint32_t PlatFormInfos::GetCoreNumByType(const std::string &core_type)
{
    uint32_t coreNum = 0;
    Mki::Dl dl = Mki::Dl(std::string(Mki::GetEnv("ASCEND_HOME_PATH")) + "/runtime/lib64/libascendcl.so", false);
    AclrtGetResInCurrentThreadFunc aclrtGetResInCurrentThread = (AclrtGetResInCurrentThreadFunc)dl.GetSymbol("aclrtGetResInCurrentThread");
    if (aclrtGetResInCurrentThread != nullptr) {
        int resType = core_type == "VectorCore" ? 1 : 0;
        int getResRet = aclrtGetResInCurrentThread(resType, &coreNum);
        if (getResRet == 0) {
            MKI_LOG(DEBUG) << "Get ThreadResource::core_num_ to " << core_type << ": " << coreNum;
            if (coreNum > MAX_CORE_NUM) {
                MKI_LOG(ERROR) << "core_num is out of range : " << coreNum;
                return 1;
            }
            return coreNum;
        } else {
            MKI_LOG(WARN) << "Failed to get thread resource! ";
        }
    } else {
        MKI_LOG(WARN) << "Failed to load acl Function!";
    }
    std::string coreNumStr = "";
    std::string coreTypeStr = core_type == "VectorCore" ? "vector_core_cnt" : "ai_core_cnt";
    std::lock_guard<std::mutex> lockGuard(g_asdopsFePlatMutex);
    (void)GetPlatformRes("SoCInfo", coreTypeStr, coreNumStr);
    MKI_LOG(DEBUG) << "Get PlatFormInfos::core_num_ to " << coreTypeStr << ": " << coreNumStr;
    if (coreNumStr.empty()) {
        MKI_LOG(ERROR) << "CoreNumStr is empty!";
        return 1;
    } else {
        coreNum = std::strtoul(coreNumStr.c_str(), nullptr, 10); // 10 进制
    }
    if (coreNum > MAX_CORE_NUM) {
        MKI_LOG(ERROR) << "core_num is out of range : " << coreNum;
        return 1;
    }
    return coreNum;
}

void PlatFormInfos::SetCoreNum(const uint32_t &coreNum)
{
    MKI_LOG(DEBUG) << "Set PlatFormInfos::core_num_: " << coreNum;
    core_num_ = coreNum;
}

uint32_t PlatFormInfos::GetCoreNum() const
{
    MKI_LOG(DEBUG) << "Get PlatFormInfos::core_num_: " << core_num_;
    return core_num_;
}

void PlatFormInfos::GetLocalMemSize(const LocalMemType &memType, uint64_t &size)
{
    std::string sizeStr;
    switch (memType) {
        case LocalMemType::L0_A: {
            (void)GetPlatformRes("AICoreSpec", "l0_a_size", sizeStr);
            break;
        }
        case LocalMemType::L0_B: {
            (void)GetPlatformRes("AICoreSpec", "l0_b_size", sizeStr);
            break;
        }
        case LocalMemType::L0_C: {
            (void)GetPlatformRes("AICoreSpec", "l0_c_size", sizeStr);
            break;
        }
        case LocalMemType::L1: {
            (void)GetPlatformRes("AICoreSpec", "l1_size", sizeStr);
            break;
        }
        case LocalMemType::L2: {
            (void)GetPlatformRes("SoCInfo", "l2_size", sizeStr);
            break;
        }
        case LocalMemType::UB: {
            (void)GetPlatformRes("AICoreSpec", "ub_size", sizeStr);
            break;
        }
        case LocalMemType::HBM: {
            (void)GetPlatformRes("SoCInfo", "memory_size", sizeStr);
            break;
        }
        default: {
            break;
        }
    }

    if (sizeStr.empty()) {
        size = 0;
    } else {
        try {
            size = static_cast<uint64_t>(std::stoll(sizeStr.c_str()));
        } catch (const std::invalid_argument &e) {
            size = 0;
        } catch (const std::out_of_range &e) {
            size = 0;
        }
    }
}

void PlatFormInfos::GetLocalMemBw(const LocalMemType &memType, uint64_t &bwSize)
{
    std::string bwSizeStr;
    switch (memType) {
        case LocalMemType::L2: {
            (void)GetPlatformRes("AICoreMemoryRates", "l2_rate", bwSizeStr);
            break;
        }
        case LocalMemType::HBM: {
            (void)GetPlatformRes("AICoreMemoryRates", "ddr_rate", bwSizeStr);
            break;
        }
        default: {
            break;
        }
    }

    if (bwSizeStr.empty()) {
        bwSize = 0;
    } else {
        try {
            bwSize = static_cast<uint64_t>(std::stoll(bwSizeStr.c_str()));
        } catch (const std::invalid_argument &e) {
            bwSize = 0;
        } catch (const std::out_of_range &e) {
            bwSize = 0;
        }
    }
}

std::map<std::string, std::vector<std::string>> PlatFormInfos::GetFixPipeDtypeMap()
{
    if (platform_infos_impl_ == nullptr) {
        return {};
    }
    return platform_infos_impl_->GetFixPipeDtypeMap();
}

void PlatFormInfos::SetPlatformRes(const std::string &label, std::map<std::string, std::string> &res)
{
    if (platform_infos_impl_ == nullptr) {
        return;
    }
    platform_infos_impl_->SetPlatformRes(label, res);
}
} // namespace fe
