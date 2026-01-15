/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "atb/utils/dl_manager.h"
#include <dlfcn.h>

namespace atb {
DlManager::DlManager(std::string path) : path_(path)
{
    // 在构造函数中加载动态库
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle_ == nullptr) {
        ATB_LOG(ERROR) << "Dynamic library handle is null, please check the path: " << path_;
    }
}

DlManager::~DlManager()
{
    // 在析构函数中卸载动态库
    if (handle_ != nullptr) {
        dlclose(handle_);
    }
}

Status DlManager::getSymbol(const std::string &symbol, void *&symbolPtr) const
{
    if (handle_ == nullptr) {
        ATB_LOG(ERROR) << "Dynamic library handle is null, please check the path: " << path_;
        return ERROR_CANN_ERROR;
    }
    symbolPtr = dlsym(handle_, symbol.c_str());
    char *errorInfo = dlerror();
    if (symbolPtr == nullptr || errorInfo != nullptr) {
        ATB_LOG(ERROR) << "Failed to find symbol " << symbol << " from path: " << path_ << ", error: " << errorInfo;
        return ERROR_CANN_ERROR;
    }
    return NO_ERROR;
}

} // namespace atb
