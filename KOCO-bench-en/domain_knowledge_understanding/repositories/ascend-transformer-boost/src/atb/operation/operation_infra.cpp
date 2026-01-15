/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/operation_infra.h"
#include "atb/operation/operation_impl.h"
#include "atb/utils/log.h"

namespace atb {
OperationInfra::OperationInfra()
{
    impl_ = std::make_unique<OperationImpl>();
}

OperationInfra::OperationInfra(const OperationInfra &other)
{
    impl_ = std::make_unique<OperationImpl>();
    *(impl_) = *(other.impl_);
}

OperationInfra& OperationInfra::operator = (const OperationInfra &other)
{
    impl_ = std::make_unique<OperationImpl>();
    *(impl_) = *(other.impl_);
    return *this;
}

OperationInfra::~OperationInfra() {}

void OperationInfra::SetExecuteStreamId(uint32_t streamId)
{
    impl_->SetExecuteStreamId(streamId);
}
uint32_t OperationInfra::GetExecuteStreamId() const
{
    return impl_->GetExecuteStreamId();
}

aclrtStream OperationInfra::GetExecuteStream(Context *context)
{
    if (context == nullptr) {
        ATB_LOG(ERROR) << "GetExecuteStream interface failed. Please check context is not nullptr.";
        return nullptr;
    }
    std::vector<aclrtStream> streams = context->GetExecuteStreams();
    uint32_t streamId = impl_->GetExecuteStreamId();
    if (streamId >= streams.size()) {
        ATB_LOG(ERROR) << "stream id is " << streamId << ", which exceed stream number " << streams.size();
        return nullptr;
    }
    return streams.at(streamId);
}
}