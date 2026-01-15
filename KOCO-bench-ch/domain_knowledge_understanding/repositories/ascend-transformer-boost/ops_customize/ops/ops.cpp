/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief 实现单例以及 operation 和 Kernel 的注册、查询、调度功能。
 *
 *  - 构造并持有唯一的 OpSchedule 对象
 *  - 通过宏注册（REG_OPERATION、REG_KERNEL_BASE 等）把所有算子/Kernel 加入调度器
 *  - 提供查询接口，运行时获取已注册的 Operation 或 Kernel
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
