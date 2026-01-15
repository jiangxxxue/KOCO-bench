/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/context.h"
#include "atb/context/context_base.h"
#include "atb/types.h"
#include "atb/utils/log.h"

namespace atb {
Status CreateContext(Context **context)
{
    if (!context) {
        ATB_LOG(ERROR) << "param context is null, CreateContext fail";
        return ERROR_INVALID_PARAM;
    }

    ContextBase *contextBase = new (std::nothrow) ContextBase();
    if (!contextBase) {
        ATB_LOG(ERROR) << "new ContextBase fail, CreateContext fail";
        return ERROR_OUT_OF_HOST_MEMORY;
    }

    Status st = contextBase->Init();
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "ContextBase init fail, CreateContext fail";
        delete contextBase;
        contextBase = nullptr;
        return st;
    }

    *context = contextBase;
    return NO_ERROR;
}

Status CreateContext(Context **context, const std::function<void*(size_t)>& alloc, const std::function<void(void*)>& dealloc)
{
    if (!context) {
        ATB_LOG(ERROR) << "param context is null, CreateContext fail";
        return ERROR_INVALID_PARAM;
    }

    ContextBase *contextBase = new (std::nothrow) ContextBase();
    if (!contextBase) {
        ATB_LOG(ERROR) << "new ContextBase fail, CreateContext fail";
        return ERROR_OUT_OF_HOST_MEMORY;
    }

    Status st = contextBase->Init(alloc, dealloc);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "ContextBase init fail, CreateContext fail";
        delete contextBase;
        contextBase = nullptr;
        return st;
    }

    *context = contextBase;
    return NO_ERROR;
}

Status DestroyContext(Context *context)
{
    if (context) {
        ContextBase *contextBase = dynamic_cast<ContextBase *>(context);
        if (contextBase) {
            contextBase->Destroy();
            delete contextBase;
            contextBase = nullptr;
        }
    }

    return NO_ERROR;
}
} // namespace atb