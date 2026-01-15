/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_PAD_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_PAD_H_

#include <mki/launch_param.h>
#include <mki/utils/status/status.h>
#include <mki/kernel_info.h>

namespace AtbOps {
using namespace Mki;
Status PadTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo);
} // namespace AtbOps
#endif
