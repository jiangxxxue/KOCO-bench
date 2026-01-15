/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/disk_util.h"
#include <sys/statfs.h>
#include <mki/utils/file_system/file_system.h>
#include "atb/utils/log.h"

namespace atb {
constexpr uint64_t DISK_AVAILABEL_LIMIT = 10737418240;

bool DiskUtil::IsDiskAvailable(const std::string &filePath)
{
    std::string dirPath = Mki::FileSystem::DirName(filePath);
    uint64_t availableSize = GetDiskAvailable(dirPath);
    if (availableSize < DISK_AVAILABEL_LIMIT) {
        ATB_LOG(WARN) << "disk available space it too low, not write file:" << filePath
                      << ", availableSize:" << availableSize << ", DISK_AVAILABEL_LIMIT:" << DISK_AVAILABEL_LIMIT;
        return false;
    }

    return true;
}

uint64_t DiskUtil::GetDiskAvailable(const std::string &filePath)
{
    std::string resolvePath = Mki::FileSystem::PathCheckAndRegular(filePath);
    if (resolvePath == "") {
        ATB_LOG(ERROR) << "realpath fail, filePath:" << filePath;
        return 0;
    }

    ATB_LOG(INFO) << "statfs resolvePath:" << resolvePath;
    struct statfs diskInfo;
    int ret = statfs(resolvePath.c_str(), &diskInfo);
    if (ret != 0) {
        ATB_LOG(WARN) << "statfs fail, errno:" << errno;
        return 0;
    }

    uint64_t availableSize = static_cast<uint64_t>(diskInfo.f_bavail * static_cast<uint64_t>(diskInfo.f_bsize));

    ATB_LOG(INFO) << "statfs availableSize:" << availableSize;
    return availableSize;
}
} // namespace atb