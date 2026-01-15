/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>
#include "tiling_args.h"

namespace Lcal {
    const char* CoCTilingData::ToString() const
    {
        std::string str =
            "[CoCTilingData]: \nm=" + std::to_string(m) + ", k=" + std::to_string(k) + ", n=" + std::to_string(n) +
            ", batchSize=" + std::to_string(batchSize) +
            ", \nblockDim=" + std::to_string(blockDim) + ", rank=" + std::to_string(rank) +
            ", rankSize=" + std::to_string(rankSize) + ", tag=" + std::to_string(tag) +
            ", \nmLoop=" + std::to_string(mLoop) + ", kLoop=" + std::to_string(kLoop) +
            ", nLoop=" + std::to_string(nLoop) + ", coreLoop=" + std::to_string(coreLoop) +
            ", tilingKey=" + std::to_string(tilingKey) +
            ", \nm0=" + std::to_string(m0) + ", k0=" + std::to_string(k0) + ", n0=" + std::to_string(n0) +
            ", swizzlCount=" + std::to_string(swizzlCount) + ", swizzlDirect=" + std::to_string(swizzlDirect) +
            ", pValue=" + std::to_string(pValue) + ", ubMoveNum=" + std::to_string(ubMoveNum) +
            ", commNpuSplit=" + std::to_string(commNpuSplit) + ", commDataSplit=" + std::to_string(commDataSplit) +
            ", commDirect=" + std::to_string(commDirect) + ", lenPerLoop=" + std::to_string(lenPerLoop) +
            ", \nextraUbMoveNum=" + std::to_string(extraUbMoveNum) +
            ", extraCommNpuSplit=" + std::to_string(extraCommNpuSplit) +
            ", extraCommDataSplit=" + std::to_string(extraCommDataSplit) +
            ", extraCommDirect=" + std::to_string(extraCommDirect) +
            ", extraLenPerLoop=" + std::to_string(extraLenPerLoop) + ", \nsplitK=" + std::to_string(splitK) +
            ", write2OtherRank=" + std::to_string(write2OtherRank) +
            ", withSerialMode=" + std::to_string(withSerialMode) + ", \nis_91093=" + std::to_string(is91093);
        return str.data();
    }

    void CoCTilingData::SetDefaultValue()
    {
        m0 = m < n ? DEFAULT_COL : DEFAULT_ROW;
        k0 = DEFAULT_COL;
        n0 = m0 == DEFAULT_COL ? DEFAULT_ROW : DEFAULT_COL;
        swizzlCount = DEFAULT_SWIZZLE_COUNT;
        swizzlDirect = m > n ? SWIZZLE_DIRECT_ZERO : SWIZZLE_DIRECT_ONE;
        pValue = DEFAULT_P_VALUE;
        ubMoveNum = MAX_UB_NUM;
        commNpuSplit = COMMNPUSPLIT_ONE;
        commDataSplit = rankSize;
        commDirect = COMM_DATA_DIRECT;
        lenPerLoop = LENPERLOOP_DEFAULT;
        extraUbMoveNum = ubMoveNum;
        extraCommNpuSplit = commNpuSplit;
        extraCommDataSplit = commDataSplit;
        extraCommDirect = commDirect;
        extraLenPerLoop = lenPerLoop;
        splitK = DEFAULT_SPLIT_K;
        write2OtherRank = false;
        withSerialMode = false;
        is91093 = false;
        tag = 0;
    }
}