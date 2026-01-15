/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_ACLNN_EXECUTOR_CACHE_H
#define ATB_ACLNN_EXECUTOR_CACHE_H
#include <map>
#include "atb/types.h"
#include "atb/utils/runner_variant_pack.h"
namespace atb {

struct AclnnCacheSlot {
    uint64_t workspaceSize;
    std::shared_ptr<aclOpExecutor> executor;
};

class AclnnExecutorCache {
public:
    AclnnExecutorCache();
    ~AclnnExecutorCache();
    Status FetchCacheSlot(
        const std::string &opNameStr, const RunnerVariantPack &aclnnCacheKey, AclnnCacheSlot &outAclnnCacheSlot);
    Status AddCacheSlot(
        const std::string &opNameStr, const RunnerVariantPack &aclnnCacheKey, AclnnCacheSlot &inAclnnCacheSlot);

private:
    std::map<std::string, std::vector<std::pair<RunnerVariantPack, AclnnCacheSlot>>> cachePool_;
    int nextUpdateIndex_ = 0;
    uint32_t cacheCapacity_ = 16;
};
}  // namespace atb
#endif
