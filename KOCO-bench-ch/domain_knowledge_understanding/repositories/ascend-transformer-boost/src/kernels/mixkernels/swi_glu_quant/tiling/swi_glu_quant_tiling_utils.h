/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SWI_GLU_QUANT_TILING_UTILS_H
#define SWI_GLU_QUANT_TILING_UTILS_H

#include <mki/utils/log/log.h>
#include <mki/launch_param.h>
#include <mki/utils/status/status.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/const/op_const.h>
#include <iostream>
#include "atbops/params/params.h"
#include "swi_glu_quant_tiling.h"
#include "tiling_data.h"

namespace AtbOps {
using namespace Mki;
constexpr uint32_t INPUT_X_INDEX = 0;
constexpr uint32_t INPUT_SMOOTH_SCALES_INDEX = 1;
constexpr uint32_t INPUT_OFFSETS_INDEX = 2;
constexpr uint32_t INPUT_GROUPIDX_INDEX = 3;
constexpr uint32_t OUTPUT_Y_INDEX = 0;
constexpr uint32_t OUTPUT_SCALE_INDEX = 1;

constexpr uint32_t EVEN_FACTOR = 2;
constexpr uint32_t MAX_EXPERT_NUM = 1024;

constexpr uint32_t ALIGEN_EIGHT = 8;

constexpr uint32_t ONE = 1;
constexpr uint32_t COMPARE_INT = 255;

const std::string QUANT_MODE_STATIC = "static";
const std::string QUANT_MODE_DYNAMIC = "dynamic";
const std::string QUANT_MODE_DYNAMIC_MSD = "dynamic_msd";


template<typename T> T AlignUp(T num, T div) { return (div == 0) ? 0 : (num + div - 1) / div * div; }

template<typename T> T AlignDown(T num, T div) { return (div == 0) ? 0 : num / div * div; }

template<typename T> T CeilDiv(T num, T div) { return div == 0 ? 0 : (num + div - 1) / div; }

template<typename T> T Min(T num, T div) { return num < div ? num : div; }

template<typename T> T Max(T num, T div) { return num < div ? div : num; }

inline bool SetTotalShape(const Mki::SVector<int64_t> &inShape, SwiGluQuantTilingData& tilingData)
    {
    int64_t shapeBefore = 1;
    int64_t shapeAfter = 1;
    int32_t dimNum = static_cast<int32_t>(inShape.size());
    int32_t inDim = -1;
    int32_t splitDim = inDim < 0 ? dimNum + inDim : inDim;
    for (int32_t i = 0; i < splitDim; i++) {
        shapeBefore *= inShape[i];
    }
    for (int32_t i = splitDim; i < dimNum; i++) {
        shapeAfter *= inShape[i];
    }
    MKI_CHECK(shapeAfter % EVEN_FACTOR == 0, "shapeAfter % 2 != 0", return false);
    MKI_CHECK(shapeAfter != 0, "shapeAfter == 0", return false);
    tilingData.rowLen = static_cast<uint32_t>(shapeBefore);
    tilingData.colLen = static_cast<uint32_t>(shapeAfter / EVEN_FACTOR);
    return true;
}

inline bool CalculateMaxUbSizePerRow(SwiGluQuantTilingData& tilingData)
{
    uint32_t colLen = tilingData.colLen;
    uint32_t alignedColLen = AlignUp<uint32_t>(colLen, tilingData.blockNum);
    MKI_CHECK(alignedColLen != 0, "CalculateMaxUbSizePerRow Unsupported alignedColLen  == 0", return false);
    MKI_LOG(INFO) << "alignedColLen:" << alignedColLen << "\n";
    uint32_t ubAvail = tilingData.dataNumSingleUb / alignedColLen;
    MKI_LOG(INFO) << "tilingData.dataNumSingleUb:" << tilingData.dataNumSingleUb << "\n";
    MKI_LOG(INFO) << "ubAvail:" << ubAvail << "\n";
    MKI_CHECK(ubAvail != 0, "The input vector is too large. It is not supported currently.", return false);

    tilingData.optBaseColLen = colLen;
    ubAvail = Max(ubAvail, ONE);

    tilingData.optBaseRowLenHeadCore = Min(Min(ubAvail, tilingData.rowLenPerHeadCore), COMPARE_INT);
    tilingData.optBaseRowLenTailCore = Min(Min(ubAvail, tilingData.rowLenPerTailCore), COMPARE_INT);
    return true;
}

bool CalTilingData(SwiGluQuantTilingData &tilingData);

void SetTilingData(SwiGluQuantTilingData &tilingData);

} // namespace AsdOps
#endif // OPS_SWI_GLU_QUANT_TILING_H