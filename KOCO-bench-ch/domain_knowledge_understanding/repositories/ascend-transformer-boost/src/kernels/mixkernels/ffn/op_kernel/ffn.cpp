/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
/*!
 * \file ffn.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "ffn_quant.h"
#include "mixkernels/ffn/tiling/tiling_data.h"

template <class T>
__aicore__ inline void InitPrivateTilingData(const __gm__ uint8_t *tiling, T *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(T) / 4; i++) {
        *(dst + i) = *(src + i);
    }
}

extern "C" __global__ __aicore__ void ffn(GM_ADDR x, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR scale_quant, GM_ADDR scale_dequant1, GM_ADDR scale_dequant2, GM_ADDR bias_quant, GM_ADDR bias_dequant1,
    GM_ADDR bias_dequant2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::SetLoadDataPaddingValue<uint64_t>((uint64_t)0x0);
    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    AscendC::SetAtomicNone();

    AtbOps::FFNTilingData tilingData;
    InitPrivateTilingData<AtbOps::FFNTilingData>(tiling, &tilingData);
    if (TILING_KEY_IS(0)) {
        FFN::DefaultFFN<ArchType::ASCEND_V200, int8_t, uint64_t, int32_t, half, half, CubeFormat::ND, CubeFormat::ND,
            false, true, true, false, false> kernel_int8_nd_prec;
        kernel_int8_nd_prec.Init(x, weight1, weight2, scale_quant, scale_dequant1, scale_dequant2,
            bias_quant, bias_dequant1, bias_dequant2, y, workspace, &tilingData);
        kernel_int8_nd_prec.Process();
    } else if (TILING_KEY_IS(1)) {
        FFN::DefaultFFN<ArchType::ASCEND_V200, int8_t, uint64_t, int32_t, half, half, CubeFormat::ND, CubeFormat::ND,
            false, true, true, false, true> kernel_int8_nd_perf;
        kernel_int8_nd_perf.Init(x, weight1, weight2, scale_quant, scale_dequant1, scale_dequant2,
            bias_quant, bias_dequant1, bias_dequant2, y, workspace, &tilingData);
        kernel_int8_nd_perf.Process();
    }
}