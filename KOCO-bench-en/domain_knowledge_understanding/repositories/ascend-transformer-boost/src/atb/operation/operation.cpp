/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/operation.h"
#include "atb/operation_infra.h"
#include "atb/operation/operation_base.h"

namespace atb {
Status DestroyOperation(Operation *operation)
{
    if (operation) {
        delete operation;
        operation = nullptr;
    }
    return NO_ERROR;
}

Status SetExecuteStreamId(Operation *operation, uint32_t streamId)
{
    OperationBase *opBase = dynamic_cast<OperationBase *>(operation);
    if (opBase) {
        opBase->SetExecuteStreamId(streamId);
        return NO_ERROR;
    }

    OperationInfra *opInfra = dynamic_cast<OperationInfra *>(operation);
    if (opInfra) {
        opInfra->SetExecuteStreamId(streamId);
        return NO_ERROR;
    }

    return ERROR_INVALID_PARAM;
}

uint32_t GetExecuteStreamId(Operation *operation)
{
    OperationBase *opBase = dynamic_cast<OperationBase *>(operation);
    if (opBase) {
        return opBase->GetExecuteStreamId();
    }

    OperationInfra *opInfra = dynamic_cast<OperationInfra *>(operation);
    if (opInfra) {
        return opInfra->GetExecuteStreamId();
    }

    ATB_LOG(ERROR) << "GetExecuteStreamId failed! operation invalid!";
    return 0;
}
} // namespace atb