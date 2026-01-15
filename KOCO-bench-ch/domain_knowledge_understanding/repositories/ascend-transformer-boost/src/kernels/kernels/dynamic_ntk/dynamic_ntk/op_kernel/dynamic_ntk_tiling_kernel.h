/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __DYNAMIC_NTK_TILING_KERNEL_H__
#define __DYNAMIC_NTK_TILING_KERNEL_H__

struct DynamicNTKTilingKernel {
    uint32_t numTokens;
    int32_t headDim;
    int32_t batchNum;
    int32_t halfHeadDim;
    uint32_t freqTileLen;
    uint32_t freqTileNum;
    uint32_t freqTailLen;
    uint64_t posGmOff;
    uint32_t posTotalLen;
    uint32_t posTileLen;

    __aicore__ inline void copyTiling(const AsdOps::DynamicNTKTilingData *tilingData)
    {
        auto blkId = AscendC::GetBlockIdx();
        numTokens = tilingData->numTokens;
        headDim = tilingData->headDim;
        batchNum = tilingData->batchNum;
        halfHeadDim = headDim >> 1;
        freqTileLen = tilingData->freqTileLen;
        freqTileNum = tilingData->freqTileNum;
        freqTailLen = tilingData->freqTailLen;
        posTileLen = tilingData->posTileLen;
        auto posLongCores = tilingData->posLongCores;
        auto posShortCores = tilingData->posShortCores;
        auto posLongLen = tilingData->posLongLen;
        auto posShortLen = tilingData->posShortLen;
        auto posTailCoreLen = tilingData->posTailCoreLen;
        if (blkId < posLongCores) {
            posGmOff = posLongLen * blkId;
            posTotalLen = posLongLen;
        } else if (blkId < (posLongCores + posShortCores)) {
            posGmOff = posLongCores * posLongLen + posShortLen * (blkId - posLongCores);
            posTotalLen = posShortLen;
        } else {
            posGmOff = posLongCores * posLongLen + posShortLen * posShortCores;
            posTotalLen = posTailCoreLen;
        }
    }
};

#endif // __DYNAMIC_NTK_TILING_KERNEL_H__