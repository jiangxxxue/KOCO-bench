/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CUSTOMIZE_OPS_BLOCKCOPY_TILING_H
#define CUSTOMIZE_OPS_BLOCKCOPY_TILING_H

#include <mki/launch_param.h>
#include <mki/kernel_info.h>
#include <mki/utils/status/status.h>

namespace AtbOps {
using namespace Mki;
/**
 * @brief 计算 CustomizeBlockCopy 所需的 tiling 信息，并填充到 kernelInfo 中。
 *
 * 本函数会根据 launchParam 中的 input 张量和参数类型，
 * 选择调用 Nd 或 Nz 版 tiling 计算，并在 310P 平台上做对齐检查。
 *
 * @param[in]  launchParam   包含输入张量、参数类型等运行时信息
 * @param[out] kernelInfo    输出的 kernel 调度信息（blockDim、tilingId 等）
 * @return Status            返回执行状态，成功为 0，失败则为错误码
 */
Status CustomizeBlockCopyTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo);
} // namespace AtbOps
#endif