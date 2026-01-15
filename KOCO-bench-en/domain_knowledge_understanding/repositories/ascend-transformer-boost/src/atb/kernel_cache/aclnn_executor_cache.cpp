/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_executor_cache.h"
#include "atb/utils/log.h"
#include "atb/utils/aclnn_util.h"
#include "atb/utils/tensor_util.h"

namespace atb {
AclnnExecutorCache::AclnnExecutorCache()
{
    ATB_LOG(INFO) << "ATB aclnn cache init";
    cachePool_ = {};
}

AclnnExecutorCache::~AclnnExecutorCache() {}

Status AclnnExecutorCache::FetchCacheSlot(const std::string &opNameStr, const RunnerVariantPack &aclnnCacheKey,
                                          AclnnCacheSlot &outAclnnCacheSlot)
{
#ifdef _DEBUG
    ATB_LOG(INFO) << "ATB aclnn op: " << opNameStr
                  << ", try to fetch cache slot with RunnerVariantPack: " << aclnnCacheKey.ToString();
#else
    ATB_LOG(INFO) << "ATB aclnn op: " << opNameStr << ", try to fetch cache slot";
#endif
    std::map<std::string, std::vector<std::pair<RunnerVariantPack, AclnnCacheSlot>>>::iterator it =
        cachePool_.find(opNameStr);
    if (it == cachePool_.end()) {
        ATB_LOG(INFO) << "ATB aclnn op: " << opNameStr << " not found in executor cache, consider add the cache first";
        return ERROR_INVALID_PARAM;
    }
    const std::vector<std::pair<RunnerVariantPack, AclnnCacheSlot>> &slotVec = it->second;
    int slotVecSize = static_cast<int>(slotVec.size());
    for (int i = 0; i < slotVecSize; ++i) {
        if (slotVec[i].second.executor == nullptr) {
            ATB_LOG(INFO) << "ATB aclnn op: " << opNameStr << " cache index [" << i << "]'s executor is nullptr";
            continue;
        }
        if (TensorUtil::IsRunnerVariantPackInputEqual(aclnnCacheKey, slotVec[i].first)) {
            ATB_LOG(INFO) << "ATB aclnn op: " << opNameStr << ", fetched slot at index [" << i << "]";
            outAclnnCacheSlot = slotVec[i].second;
            return NO_ERROR;
        }
    }
    ATB_LOG(INFO) << "ATB aclnn op: " << opNameStr
                  << ", cache slot was not found with variantPack: " << aclnnCacheKey.ToString();
    return ERROR_INVALID_PARAM;
}

Status AclnnExecutorCache::AddCacheSlot(const std::string &opNameStr, const RunnerVariantPack &aclnnCacheKey,
                                        AclnnCacheSlot &inAclnnCacheSlot)
{
#ifdef _DEBUG
    ATB_LOG(INFO) << "ATB aclnn op: " << opNameStr
                  << ", try to add cache slot with RunnerVariantPack: " << aclnnCacheKey.ToString();
#else
    ATB_LOG(INFO) << "ATB aclnn op: " << opNameStr << ", add to fetch cache slot";
#endif
    if (inAclnnCacheSlot.executor == nullptr) {
        ATB_LOG(ERROR) << "ATB aclnn AddCacheSlot with op: " << opNameStr << " got null aclOpExecutor";
        return ERROR_INVALID_PARAM;
    }
    std::map<std::string, std::vector<std::pair<RunnerVariantPack, AclnnCacheSlot>>>::iterator it =
        cachePool_.find(opNameStr);
    if (it == cachePool_.end()) {
        ATB_LOG(INFO) << "ATB aclnn executor cache add op: " << opNameStr;
        cachePool_[opNameStr].emplace_back(std::make_pair(aclnnCacheKey, inAclnnCacheSlot));
        return NO_ERROR;
    }

    std::vector<std::pair<RunnerVariantPack, AclnnCacheSlot>> slotVec = it->second;
    size_t slotVecSize = slotVec.size();
    if (slotVecSize < cacheCapacity_) {
        slotVec.emplace_back(aclnnCacheKey, inAclnnCacheSlot);
        ATB_LOG(INFO) << "ATB aclnn executor cache add op: " << opNameStr << " at index[" << slotVecSize << "]";
    }

    // 淘汰方式：使用等长vector+FIFO
    ATB_LOG(INFO) << "ATB aclnn executor cache full for op: " << opNameStr << "update index [" << nextUpdateIndex_
                  << "]";
    cachePool_[opNameStr][nextUpdateIndex_] = std::make_pair(aclnnCacheKey, inAclnnCacheSlot);
    nextUpdateIndex_ = (nextUpdateIndex_ + 1) % cacheCapacity_;
    return NO_ERROR;
}

} // namespace atb
