/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/config.h"
#include <string>
#include <cstring>
#include <iostream>
#include <thread>
#include <acl/acl_rt.h>
#include <atb/utils/log.h>
#include <mki/utils/profiling/profiling_funcs.h>
#include <mki/utils/strings/match.h>
#include <mki/utils/strings/str_split.h>
#include "atb/utils/singleton.h"

namespace atb {
constexpr int32_t DECIMAL = 10;
constexpr uint32_t DEFAULT_WORKSPACE_MEM_ALLOC_ALG_TYPE = 1;
const size_t MAX_ENV_STRING_LEN = 12800;

Config::Config()
{
    ATB_LOG(INFO) << "Config:";
    InitAtbHomePath();
    InitSocVersion();
    InitKernelCache();
    InitTilingBuffer();
    InitShareMemoryNameSuffix();
    isStreamSyncEveryKernelEnable_ = IsEnable("ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE");
    isStreamSyncEveryRunnerEnable_ = IsEnable("ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE");
    isStreamSyncEveryOperationEnable_ = IsEnable("ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE");
    isOpsRunnerSetupCacheEnable_ = IsEnable("ATB_OPSRUNNER_SETUP_CACHE_ENABLE", true);
    const char *envStr = std::getenv("ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE");
    workspaceMemAllocAlgType_ = envStr != nullptr ? static_cast<uint32_t>(strtol(envStr, nullptr, DECIMAL)) :
                                                    DEFAULT_WORKSPACE_MEM_ALLOC_ALG_TYPE;
    isCompareTilingEveryKernelEnable_ = IsEnable("ATB_COMPARE_TILING_EVERY_KERNEL");
    isworkspaceMemAllocGlobal_ = IsEnable("ATB_WORKSPACE_MEM_ALLOC_GLOBAL");
    isMatmulShuffleKEnable_ = IsEnable("ATB_MATMUL_SHUFFLE_K_ENABLE", true);
    ATB_LOG(INFO) << "AtbHomePath: " << atbHomePath_
                  << ", IsStreamSyncEveryRunnerEnable: " << isStreamSyncEveryRunnerEnable_
                  << ", IsStreamSyncEveryKernelEnable: " << isStreamSyncEveryKernelEnable_
                  << ", IsStreamSyncEveryOperationEnable: " << isStreamSyncEveryOperationEnable_;
    ATB_LOG(INFO) << "IsOpsRunnerSetupCacheEnable: " << isOpsRunnerSetupCacheEnable_
                  << ", LocalKernelCacheCount: " << localKernelCacheCount_
                  << ", GlobalKernelCacheCount: " << globalKernelCacheCount_;
    ATB_LOG(INFO) << "ProfilingLevel0Status: " << GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel0Status()
                  << ", ProfilingLevel1Status: " << GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel1Status()
                  << ", IsCompareTilingEveryKernelEnable: " << isCompareTilingEveryKernelEnable_;
    ATB_LOG(INFO) << "WorkspaceMemAllocAlgType: " << workspaceMemAllocAlgType_
                  << ", IsworkspaceMemAllocGlobal: " << isworkspaceMemAllocGlobal_
                  << ", HostTilingBufferBlockNum:" << hostTilingBlockNum_
                  << ", DeviceTilingBufferBlockNum:" << deviceTilingBlockNum_
                  << ", ShareMemoryNameSuffix:" << shareMemoryNameSuffix_
                  << ", IsMatmulShuffleKEnable:" << isMatmulShuffleKEnable_;
}

Config::~Config() {}

bool Config::IsEnable(const char *env, bool enable)
{
    const char *saveTensor = std::getenv(env);
    if (!saveTensor) {
        return enable;
    }
    return std::string(saveTensor) == "1";
}

bool Config::IsStreamSyncEveryKernelEnable() const
{
    return isStreamSyncEveryKernelEnable_;
}

bool Config::IsStreamSyncEveryRunnerEnable() const
{
    return isStreamSyncEveryRunnerEnable_;
}

bool Config::IsStreamSyncEveryOperationEnable() const
{
    return isStreamSyncEveryOperationEnable_;
}

void Config::InitAtbHomePath()
{
    const char *envStr = std::getenv("ATB_HOME_PATH");
    if (!envStr) {
        return;
    }
    if (strlen(envStr) > MAX_ENV_STRING_LEN) {
        ATB_LOG(ERROR) << "ATB_HOME_PATH length is more than " << MAX_ENV_STRING_LEN;
        return;
    }
    atbHomePath_ = std::string(envStr);
}

void Config::InitTilingBuffer()
{
    const uint32_t hostMinBlockNum = 128;
    const uint32_t hostMaxBlockNum = 1024;
    InitVariable("ATB_HOST_TILING_BUFFER_BLOCK_NUM", hostMinBlockNum, hostMaxBlockNum, hostTilingBlockNum_);

    const uint32_t deviceMinBlockNum = 32;
    const uint32_t deviceMaxBlockNum = 1024;
    InitVariable("ATB_DEVICE_TILING_BUFFER_BLOCK_NUM", deviceMinBlockNum, deviceMaxBlockNum, deviceTilingBlockNum_);
}

void Config::InitShareMemoryNameSuffix()
{
    const char *envStr = std::getenv("ATB_SHARE_MEMORY_NAME_SUFFIX");
    if (!envStr) {
        return;
    }
    if (strlen(envStr) > MAX_ENV_STRING_LEN) {
        ATB_LOG(ERROR) << "$ATB_SHARE_MEMORY_NAME_SUFFIX length is more than " << MAX_ENV_STRING_LEN;
        return;
    }
    shareMemoryNameSuffix_ = std::string(envStr);
}

bool Config::Is910B() const
{
    return is910B_;
}

bool Config::Is310P() const
{
    return is310P_;
}

bool Config::Is910A() const
{
    return is910A_;
}

bool Config::Is310B() const
{
    return is310B_;
}

void Config::InitSocVersion()
{
    const char *socName = aclrtGetSocName();
    if (!socName) {
        ATB_LOG(ERROR) << "aclrtGetSocName failed!";
        return;
    }
    const uint32_t LEN_OF_ASCEND_910B = 10;
    ATB_LOG(INFO) << "SocVersion:" << std::string(socName);
    is910B_ = (std::string(socName).find("Ascend910B") != std::string::npos &&
               std::string(socName).length() > LEN_OF_ASCEND_910B) ||
              std::string(socName).find("Ascend910_93") != std::string::npos;
    is310B_ = std::string(socName).find("Ascend310B") != std::string::npos;
    is310P_ = std::string(socName).find("Ascend310P") != std::string::npos;
    is910A_ = std::string(socName).find("Ascend910A") != std::string::npos ||
              std::string(socName).find("Ascend910ProA") != std::string::npos ||
              (std::string(socName).find("Ascend910B") != std::string::npos &&
               std::string(socName).length() == LEN_OF_ASCEND_910B) ||
              std::string(socName).find("Ascend910ProB") != std::string::npos ||
              std::string(socName).find("Ascend910PremiumA") != std::string::npos;
}

std::string Config::GetAtbHomePath() const
{
    return atbHomePath_;
}

bool Config::IsOpsRunnerSetupCacheEnable() const
{
    return isOpsRunnerSetupCacheEnable_;
}

uint32_t Config::GetWorkspaceMemAllocAlgType() const
{
    return workspaceMemAllocAlgType_;
}


void Config::InitKernelCache()
{
    const uint32_t maxKernelCacheCount = 1024;

    const char *envStr = nullptr;

    envStr = std::getenv("ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT");
    localKernelCacheCount_ = envStr != nullptr ? static_cast<uint32_t>(strtol(envStr, nullptr, DECIMAL)) : 1;
    if (localKernelCacheCount_ == 0) {
        localKernelCacheCount_ = 1;
    }
    if (localKernelCacheCount_ > maxKernelCacheCount) {
        localKernelCacheCount_ = maxKernelCacheCount;
    }

    envStr = std::getenv("ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT");
    globalKernelCacheCount_ = envStr != nullptr ? static_cast<uint32_t>(strtol(envStr, nullptr, DECIMAL)) : 1;
    if (globalKernelCacheCount_ == 0) {
        globalKernelCacheCount_ = 1;
    }
    if (globalKernelCacheCount_ > maxKernelCacheCount) {
        globalKernelCacheCount_ = maxKernelCacheCount;
    }
}

uint32_t Config::GetLocalKernelCacheCount() const
{
    return localKernelCacheCount_;
}

uint32_t Config::GetGlobalKernelCacheCount() const
{
    return globalKernelCacheCount_;
}

bool Config::IsCompareTilingEveryKernelEnable() const
{
    return isCompareTilingEveryKernelEnable_;
}

bool Config::IsworkspaceMemAllocGlobal() const
{
    return isworkspaceMemAllocGlobal_;
}

uint32_t Config::GetHostTilingBlockNum() const
{
    return hostTilingBlockNum_;
}

uint32_t Config::GetDeviceTilingBlockNum() const
{
    return deviceTilingBlockNum_;
}

void Config::InitVariable(const char *envName, uint32_t min, uint32_t max, uint32_t &value) const
{
    const char *env = std::getenv(envName);

    if (env) {
        uint32_t tmpValue = static_cast<uint32_t>(strtoll(env, nullptr, DECIMAL));
        ATB_LOG(INFO) << "env:" << envName << " value:" << tmpValue;
        if (tmpValue < min) {
            ATB_LOG(WARN) << "env:" << envName << " value:" << tmpValue << " must >= " << min;
            tmpValue = min;
        }
        if (tmpValue > max) {
            ATB_LOG(WARN) << "env:" << envName << " value:" << tmpValue << " must <= " << max;
            tmpValue = max;
        }
        value = tmpValue;
    }
}

std::string Config::GetShareMemoryNameSuffix() const
{
    return shareMemoryNameSuffix_;
}

bool Config::IsMatmulShuffleKEnable() const
{
    return isMatmulShuffleKEnable_;
}

} // namespace atb
