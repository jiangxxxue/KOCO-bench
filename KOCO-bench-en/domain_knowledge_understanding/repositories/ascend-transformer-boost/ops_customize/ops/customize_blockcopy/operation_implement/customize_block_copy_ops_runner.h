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
 * @file customize_block_copy_ops_runner.h
 * @brief 定义 CustomizeBlockCopyOpsRunner 类，实现 ATB 对底层 Mki 计算图节点的封装。
 */
#ifndef CUSTOMIZE_BLOCK_COPY_BLOCK_COPYOPSRUNNER_H
#define CUSTOMIZE_BLOCK_COPY_BLOCK_COPYOPSRUNNER_H
#include "atb/runner/ops_runner.h"
#include "customize_op_params.h"

namespace atb {
/**
 * @class CustomizeBlockCopyOpsRunner
 * @brief 用于构建并执行 Mki 层的 CustomizeBlockCopyOperation。
 */
class CustomizeBlockCopyOpsRunner : public OpsRunner {
public:
    explicit CustomizeBlockCopyOpsRunner(const customize::BlockCopyParam &param);
    ~CustomizeBlockCopyOpsRunner() override;

private:
    customize::BlockCopyParam param_;
};
} // namespace atb
#endif