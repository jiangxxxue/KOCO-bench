/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_SCATTER_ELEMENTS_V2_TILING_H
#define ASCEND_OPS_SCATTER_ELEMENTS_V2_TILING_H

#include <mki/bin_handle.h>
#include <mki/kernel_info.h>
#include <mki/launch_param.h>

namespace AsdOps {
using namespace Mki;

/**
 * @brief ScatterElementsV2 算子的通用 Tiling 函数
 *
 * @param kernelName 内核名称
 * @param launchParam 启动参数
 * @param kernelInfo 内核信息
 * @param binHandle 二进制句柄
 * @return Status 返回状态
 */
Status ScatterElementsV2CommonTiling(const std::string &kernelName, const LaunchParam &launchParam,
                                     KernelInfo &kernelInfo, const BinHandle &binHandle);

} // namespace AsdOps
#endif // ASCEND_OPS_SCATTER_ELEMENTS_TILING_H
