/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_SIMPLE_BROADCAST_TILING_H
#define ASCEND_OPS_SIMPLE_BROADCAST_TILING_H

#include <mki/launch_param.h>
#include <mki/kernel_info.h>
#include <mki/utils/status/status.h>
#include "simple_broadcast_tiling_data.h"

namespace AsdOps {
using namespace Mki;
constexpr uint32_t TENSOR_DST_IDX = 0;
constexpr uint32_t TENSOR_SRC_0_IDX = 1;

constexpr uint32_t TENSOR_SCALE_IDX = 1;
constexpr uint32_t TENSOR_OFFSET_IDX = 2;

constexpr uint32_t TILING_ID_BASE = 2000000000;
constexpr uint32_t TILING_ID_CUT = 100000000; // 最高位表示不同切分策略，0: scalar, 1: cutN, 2: cutB
constexpr uint32_t TILING_ID_DTYPE = 10000000; // 0:fp16, 1:bf16

// quant/dequant
constexpr uint32_t TILING_ID_HAS_OFFSET = 1;
constexpr uint32_t TILING_ID_MIN_NEG_127 = 10;


struct BroadcastInfo {
    size_t broadcastBytesPerElem; // UB中，计算一个元素，broadcast buffer占用的字节(包括输入输出，以及中间缓存)
    size_t normalBytesPerElem; // normal  buffer占用的字节(包括输入输出，以及中间缓存)
};

Status SimplyBroadcastTiling(BroadcastInfo &broadcastInfo, const LaunchParam &launchParam, KernelInfo &kernelInfo);

Status QuantPerChannelTiling(BroadcastInfo &broadcastInfo, const LaunchParam &launchParam, KernelInfo &kernelInfo);

Status DequantPerChannelTiling(BroadcastInfo &broadcastInfo, const LaunchParam &launchParam, KernelInfo &kernelInfo);
} // namespace AsdOps
#endif