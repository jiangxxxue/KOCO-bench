/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atbops/ops.h"
#include "mki_loader/op_schedule_base.h"
#include "mki_loader/op_register.h"

namespace AtbOps {
Ops::Ops() : opSchedule_(std::make_unique<OpSchedule>()) {}

Ops::~Ops() {}

Ops &Ops::Instance()
{
    static Ops instance;
    return instance;
}

std::vector<Mki::Operation *> Ops::GetAllOperations() const
{
    return opSchedule_->GetAllOperations();
}

Mki::Operation *Ops::GetOperationByName(const std::string &opName) const
{
    return opSchedule_->GetOperationByName(opName);
}

Mki::Kernel *Ops::GetKernelInstance(const std::string &kernelName) const
{
    return opSchedule_->GetKernelInstance(kernelName);
}

void Ops::UpdateSchedule()
{
    opSchedule_->UpdateLoader();
}

} // namespace AtbOps
