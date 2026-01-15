/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef SWI_GLU_QUANT_BASE_H
#define SWI_GLU_QUANT_BASE_H

#include "mixkernels/swi_glu_quant/tiling/tiling_data.h"
#include "kernel_operator.h"
namespace AtbOps {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t SPLIT_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t SWI_GLU_QUANT_EIGHT = 8;
constexpr uint32_t SWI_GLU_QUANT_THIRTY_TWO = 32;

constexpr float SWI_GLU_QUANT_INT8_SYM_SCALE = 127.0;
constexpr uint32_t MAX_VALUE_NUM = 8;
constexpr uint32_t SMOOTH_INDEX_UPBOUND = 65536;
constexpr int32_t ELEM_PER_REP_FP32 = 64;   // ONE_REPEAT_BYTE_SIZE / sizeof(float)

// 单输入场景，一个tile需要的偏置参数
struct XxGluSingleTileOffsetParam {
    uint64_t splitVecGmOffset1; // 拼接的vector，第一个vector gm上的偏移
    uint64_t splitVecGmOffset2; // 拼接的vector，第er个vector gm上的偏移
    uint64_t tmpVecGmOffset;
};

class SwiGluQuantBase {
public:
    __aicore__ inline SwiGluQuantBase()
    {}

    __aicore__ inline void ParseTilingData(const SwiGluQuantTilingData *tilingData)
    {
        tilingData_.groupLen = tilingData->groupLen;            // group的长度
        tilingData_.rowLen = tilingData->rowLen;                // 多少行数据
        tilingData_.colLen = tilingData->colLen;                // 列数，对输入x的一半
        tilingData_.rowLenPerHeadCore = tilingData->rowLenPerHeadCore;  // 每核处理的行数
        tilingData_.rowLenPerTailCore = tilingData->rowLenPerTailCore;  // 每核处理的行数
        tilingData_.basicRowLenHeadCore = tilingData->basicRowLenHeadCore;      // 每次计算的行数
        tilingData_.basicRowLenTailCore = tilingData->basicRowLenTailCore;      // 每次计算的行数
        tilingData_.basicColLen = tilingData->basicColLen;      // 每次计算的列数
        tilingData_.headCoreNum = tilingData->headCoreNum;      // 使用的head核数
        tilingData_.realCoreNum = tilingData->realCoreNum;      // 使用的核数
    }

    __aicore__ inline void InitBaseBuffer()
    {
        pPipe->InitBuffer(tmpConstBuffer, MAX_VALUE_NUM * sizeof(float));
    }

    __aicore__ inline void DuplicateConst()
    {
        constScale = tmpConstBuffer.Get<float>();
        Duplicate<float>(constScale, SWI_GLU_QUANT_INT8_SYM_SCALE, MAX_VALUE_NUM);
    }

    template <typename T>
    __aicore__ inline T CeilDiv(T x, T y)
    {
        return y == 0 ? 0 : (x + y - 1) / y;
    }

    __aicore__ inline float GetMax(float a, float b)
    {
        return a > b ? a : b;
    }

    template<typename T>
    __aicore__ inline T AlignUp(T num, T div)
    {
        return (div == 0) ? 0 : (num + div - 1) / div * div;
    }

protected:
    TPipe *pPipe = nullptr;
    /* tiling data */
    SwiGluQuantTilingData tilingData_;

    /* variable */
    uint32_t rowLen, colLen, groupLen, alignedGroupLen;
    uint32_t rowLenPerHeadCore, rowLenPerTailCore;
    uint32_t basicRowLen, rowLenPerCore;
    uint32_t basicRowLenHeadCore, basicRowLenTailCore, basicColLen;
    uint32_t headCoreNum, realCoreNum;
    uint32_t outAlignLen;
    uint32_t sizeHalfLen, smoothSizeFloatLen;
    uint32_t outLen;
    uint8_t rightPadding = 0;
    uint8_t smoothRightPadding = 0;
    bool isPad = false;
    bool smoothIsPad = false;
    uint16_t blockUnit;

    uint32_t coreIdx;
    uint32_t rowLoop = 1;
    uint32_t baseRow = 0;     // 记录开始处理的行数
    uint16_t basicRowLenCal;
    uint32_t mergedColLen;
    uint64_t tileLength;

    XxGluSingleTileOffsetParam offsetParam;

    TBuf<TPosition::VECCALC> tmpConstBuffer, tmpMaxBuffer;

    /* local memory */
    LocalTensor<float> constScale;
};
}  // namespace SwiGluQuantOpt
#endif  // SwiGluQuantBase