/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include "tiling_910B.h"
#include "tiling_func.h"
#include "lcal_types.h"

namespace Lcal {
    constexpr int32_t ALLGATHERV2_EIGHT_RANK_FP16_PVALUE_DEFAULT = 6;
    constexpr int32_t ALLGATHERV2_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT = 3;
    constexpr int32_t ALLGATHERV2_EIGHT_RANK_FP16_M0_DEFAULT = 128;
    constexpr int32_t ALLGATHERV2_EIGHT_RANK_FP16_COMMDATASPLIT_DEFAULT = 8;
    constexpr int32_t ALLGATHERV2_EIGHT_RANK_FP16_CORE16_PVALUE_DEFAULT = 6;
    constexpr int32_t ALLGATHERV2_EIGHT_RANK_FP16_CORE16_UBMOVENUM_DEFAULT = 8;
    constexpr int32_t ALLGATHERV2_EIGHT_RANK_FP16_CORE16_COMMDIRECT_DEFAULT = 1;
    constexpr int32_t ALLGATHERV2_EIGHT_RANK_FP16_CORE16_COMMDATASPLIT_DEFAULT = 16;
    constexpr int32_t ALLGATHERV2_EIGHT_RANK_FP16_CORE16_M0_DEFAULT = 128;

    static std::map<int, std::vector<std::vector<int>>> g_allgatherV2EightRankFP16CorE16M0Map = {
        {128,
         {{-1, 3798, -1, 2147483647, -1, 1200}, {-1, 3798, -1, 10720, 1720, 2274},
          {-1, 3798, 10720, 2147483647, 1200, 2274}, {3798, 4298, -1, 2786, -1, 2274},
          {4810, 2147483647, -1, 2786, -1, 2274}, {3798, 2147483647, 2786, 2147483647, -1, 2274},
          {-1, 2147483647, -1, 6950, 2912, 5360}, {-1, 2147483647, 6450, 6950, 5360, 6934},
          {-1, 2147483647, -1, 6950, 6934, 2147483647}, {-1, 2147483647, 9950, 2147483647, 2274, 2626},
          {-1, 2147483647, 6950, 2147483647, 2626, 2147483647}}},
        {256,
         {{-1, 3798, -1, 10720, 1200, 1720}, {4298, 4810, -1, 2786, -1, 2274},
          {-1, 2147483647, -1, 6950, 2274, 2912}, {-1, 2147483647, -1, 6450, 5360, 6934},
          {-1, 2147483647, 6950, 9950, 2274, 2626}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherV2EightRankFP16CorE16CommdatasplitMap = {
        {16,
         {{-1, 2147483647, -1, 2147483647, -1, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherV2EightRankFP16CorE16CommdirectMap = {
        {1,
         {{-1, 2147483647, -1, 2147483647, -1, 3248}, {-1, 2147483647, -1, 768, 3248, 5660},
          {-1, 2147483647, -1, 768, 8704, 2147483647}, {-1, 2147483647, 768, 2147483647, 3248, 2147483647}}},
        {0,
         {{-1, 2147483647, -1, 768, 5660, 8704}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherV2EightRankFP16CorE16UbmovenumMap = {
        {2,
         {{-1, 6450, -1, 2147483647, -1, 2912}, {-1, 2786, -1, 2147483647, 2912, 7434},
          {2786, 6450, -1, 2147483647, 2912, 6934}, {2786, 6450, 768, 2147483647, 6934, 7434},
          {6450, 7850, 1262, 1774, 768, 7434}, {7850, 2147483647, -1, 1774, 1536, 7434},
          {6450, 8958, 1774, 10720, -1, 7434}, {6450, 8958, 10720, 2147483647, -1, 6150},
          {8958, 2147483647, 1774, 2147483647, -1, 7434}, {-1, 1262, -1, 2147483647, 7434, 8958},
          {1262, 1774, 1792, 2147483647, 7434, 8958}, {1774, 2147483647, -1, 2147483647, 7434, 8958},
          {1774, 2147483647, -1, 768, 8958, 2147483647}, {-1, 2147483647, 768, 2147483647, 8958, 2147483647}}},
        {3,
         {{2786, 6450, -1, 768, 6934, 7434}, {6450, 7850, -1, 1262, -1, 7434},
          {7850, 2147483647, -1, 1774, 768, 1536}, {1262, 1774, -1, 1792, 7434, 8958},
          {1262, 1774, -1, 768, 8958, 2147483647}}},
        {6,
         {{6450, 7850, 1262, 1774, -1, 768}}},
        {8,
         {{7850, 2147483647, -1, 1774, -1, 768}}},
        {4,
         {{6450, 8958, 10720, 2147483647, 6150, 7434}, {-1, 1262, -1, 768, 8958, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherV2EightRankFP16CorE16PvalueMap = {
        {2,
         {{-1, 3798, -1, 2147483647, -1, 832}, {768, 3798, 768, 2147483647, 832, 1200},
          {2024, 2560, -1, 1262, 1200, 2274}, {2024, 3798, 1262, 2147483647, 1200, 2274},
          {3798, 4608, 7946, 8446, -1, 2274}, {3798, 2147483647, 9728, 2147483647, -1, 768},
          {-1, 1262, -1, 768, 2274, 5660}, {768, 1262, -1, 7696, 7680, 10752},
          {1262, 2147483647, -1, 2147483647, 2912, 3248}, {8958, 9728, -1, 2147483647, 7680, 2147483647}}},
        {4,
         {{-1, 3798, -1, 768, 832, 1200}, {2560, 3798, -1, 1262, 1200, 2274},
          {3798, 2147483647, -1, 1774, -1, 2274}, {3798, 2147483647, 8446, 9728, -1, 1262},
          {3798, 2147483647, 9728, 2147483647, 768, 1262}, {3798, 2147483647, 8446, 2147483647, 1262, 2274},
          {-1, 1262, 7696, 2147483647, 8704, 10752}, {-1, 1262, -1, 2147483647, 10752, 2147483647},
          {1262, 2147483647, -1, 2274, 3248, 6934}, {1262, 2147483647, 8958, 2147483647, 4298, 6934},
          {1262, 1774, 6700, 2147483647, 6934, 2147483647}, {1774, 8958, 6450, 2147483647, 6934, 2147483647},
          {8958, 2147483647, 6700, 2147483647, 6934, 7680}, {9728, 2147483647, -1, 2147483647, 7680, 2147483647}}},
        {1,
         {{-1, 768, 768, 2147483647, 832, 1200}, {-1, 2024, -1, 2147483647, 1200, 2274},
          {-1, 1262, -1, 768, 5660, 7680}, {-1, 1262, 768, 2147483647, 2274, 7680},
          {-1, 768, -1, 7696, 7680, 10752}, {-1, 1262, 7696, 2147483647, 7680, 8704},
          {1262, 2147483647, -1, 2147483647, 2274, 2912}, {1262, 2147483647, 2274, 8958, 3248, 6934},
          {1262, 2147483647, 8958, 2147483647, 3248, 4298}, {1262, 1774, -1, 6700, 6934, 2147483647},
          {1774, 8958, -1, 6450, 6934, 2147483647}, {8958, 2147483647, -1, 6700, 6934, 7680}}},
        {6,
         {{3798, 2147483647, 1774, 7946, -1, 2274}, {4608, 2147483647, 7946, 8446, -1, 2274}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherV2EightRankFP16CommdatasplitMap = {
        {8,
         {{-1, 2274, -1, 2147483647, -1, 5312}, {2274, 2147483647, -1, 2147483647, -1, 4810},
          {5148, 2147483647, -1, 768, 4810, 5312}, {2274, 2147483647, 768, 2147483647, 4810, 5312},
          {-1, 2147483647, -1, 2147483647, 5312, 2147483647}}},
        {4,
         {{2274, 5148, -1, 768, 4810, 5312}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherV2EightRankFP16M0Map = {
        {128,
         {{-1, 2274, -1, 2147483647, -1, 6172}, {-1, 2274, 6700, 2147483647, 6172, 6934},
          {2274, 2786, 8200, 2147483647, -1, 6934}, {2786, 2147483647, -1, 6950, -1, 5360},
          {2786, 2147483647, 6950, 2147483647, -1, 6934}, {-1, 2147483647, -1, 2274, 6934, 2147483647},
          {-1, 2147483647, 2274, 4810, 6934, 7434}, {-1, 2147483647, 2274, 4810, 7946, 2147483647},
          {-1, 2147483647, 4810, 2147483647, 6934, 2147483647}}},
        {256,
         {{-1, 2274, -1, 6700, 6172, 6934}, {2274, 2786, -1, 8200, -1, 6934},
          {2786, 2147483647, -1, 6950, 5360, 6934}, {-1, 2147483647, 2274, 4810, 7434, 7946}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherV2EightRankFP16UbmovenumMap = {
        {2.0,
         {{-1, 768, -1, 2560, -1, 576}, {-1, 768, -1, 3584, 832, 2274},
          {768, 2147483647, -1, 1774, -1, 2274}, {-1, 2147483647, -1, 8200, 2274, 2626},
          {-1, 2147483647, -1, 2147483647, 2626, 2147483647}}},
        {3.0,
         {{-1, 768, 2560, 2147483647, -1, 576}, {-1, 768, -1, 2147483647, 576, 832},
          {-1, 768, 3584, 2147483647, 832, 2274}, {768, 2147483647, 1774, 2147483647, -1, 2274},
          {-1, 2147483647, 8200, 2147483647, 2274, 2626}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherV2EightRankFP16PvalueMap = {
        {2,
         {{-1, 2786, -1, 2560, -1, 576}, {1280, 2786, -1, 2147483647, 576, 832},
          {-1, 2786, -1, 4608, 832, 1200}, {1262, 2786, -1, 2147483647, 1720, 2274},
          {2786, 3286, -1, 7708, -1, 768}, {2786, 3286, -1, 2147483647, 768, 2274},
          {3286, 4298, -1, 2147483647, -1, 1262}, {4298, 7450, 4810, 2147483647, -1, 768},
          {-1, 2147483647, -1, 768, 2274, 3584}, {4608, 2147483647, -1, 768, 5660, 7680},
          {-1, 2147483647, 768, 5900, 2912, 3248}, {-1, 2147483647, 6450, 8446, 8446, 2147483647},
          {-1, 768, 11264, 2147483647, 2274, 7680}, {-1, 768, 8446, 2147483647, 7680, 2147483647},
          {768, 1262, 8446, 9728, 4636, 2147483647}, {768, 1262, 9728, 2147483647, 6684, 2147483647},
          {1262, 2147483647, 8446, 9728, 6450, 2147483647}, {1262, 2147483647, 9728, 10720, 6684, 2147483647},
          {1262, 2147483647, 10720, 2147483647, 5104, 2147483647}}},
        {1,
         {{-1, 2786, 2560, 2147483647, -1, 576}, {-1, 1280, -1, 2147483647, 576, 832},
          {-1, 2786, 4608, 2147483647, 832, 1200}, {-1, 2786, -1, 2147483647, 1200, 1720},
          {-1, 1262, -1, 2147483647, 1720, 2274}, {2786, 3286, 7708, 2147483647, -1, 768},
          {3286, 4298, -1, 2147483647, 1774, 2274}, {7450, 2147483647, 4810, 2147483647, 1774, 2274},
          {-1, 4608, -1, 768, 5660, 7680}, {-1, 2147483647, -1, 768, 7680, 2147483647},
          {-1, 2147483647, 768, 5900, 2274, 2912}, {-1, 2147483647, 768, 5900, 3248, 2147483647},
          {-1, 2147483647, 5900, 8446, 2274, 8446}, {-1, 2147483647, 5900, 6450, 8446, 2147483647},
          {-1, 768, 8446, 11264, 2274, 7680}, {768, 1262, 8446, 9728, 2274, 4636},
          {768, 1262, 9728, 2147483647, 2274, 6684}, {1262, 2147483647, 8446, 9728, 2274, 6450},
          {1262, 2147483647, 9728, 10720, 2274, 6684}, {1262, 2147483647, 10720, 2147483647, 2274, 5104}}},
        {4,
         {{3286, 4298, -1, 2147483647, 1262, 1774}, {4298, 8958, -1, 4298, -1, 2274},
          {8958, 2147483647, -1, 4810, -1, 1536}, {4298, 7450, 4810, 2147483647, 768, 2274},
          {7450, 2147483647, 4810, 2147483647, -1, 1774}, {-1, 2147483647, -1, 768, 3584, 5660}}},
        {6,
         {{4298, 8958, 4298, 4810, -1, 2274}, {8958, 2147483647, -1, 4810, 1536, 2274}}}
    };

    void AllGatherV2EightRankFP16GetDefaultTiling(CoCTilingData &cocTilingData)
    {
        std::map<int*, TilingValue> tilingParamMap = {
            {&cocTilingData.commDataSplit,
             {ALLGATHERV2_EIGHT_RANK_FP16_COMMDATASPLIT_DEFAULT,
              g_allgatherV2EightRankFP16CommdatasplitMap}},
            {&cocTilingData.m0,
             {ALLGATHERV2_EIGHT_RANK_FP16_M0_DEFAULT,
              g_allgatherV2EightRankFP16M0Map}},
            {&cocTilingData.ubMoveNum,
             {ALLGATHERV2_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT,
              g_allgatherV2EightRankFP16UbmovenumMap}},
            {&cocTilingData.pValue,
             {ALLGATHERV2_EIGHT_RANK_FP16_PVALUE_DEFAULT,
              g_allgatherV2EightRankFP16PvalueMap}},
            {&cocTilingData.swizzlDirect, {SWIZZLE_DIRECT_ONE}},
            {&cocTilingData.swizzlCount, {SWIZZLE_COUNT_FOUR}},
            {&cocTilingData.commDirect, {COMM_NPU_DIRECT}}
        };
        SetTilingParam(cocTilingData, tilingParamMap);

        int32_t coreNum = cocTilingData.blockDim - cocTilingData.rankSize;
        cocTilingData.commNpuSplit =
                cocTilingData.commDataSplit >= COMMDATASPLIT_EIGHT ? COMMNPUSPLIT_ONE : COMMNPUSPLIT_THREE;
        cocTilingData.commNpuSplit = std::min(cocTilingData.commNpuSplit, cocTilingData.rankSize);
        cocTilingData.commDataSplit =
                ClampValue(cocTilingData.commDataSplit, COMMDATASPLIT_ONE, coreNum / cocTilingData.commNpuSplit);
        cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;

        DealTilingParamByBuffSize(cocTilingData);
    }

    void AllGatherV2EightRankFP16Core16GetDefaultTiling(CoCTilingData &cocTilingData)
    {
        std::map<int*, TilingValue> tilingParamMap = {
            {&cocTilingData.m0,
             {ALLGATHERV2_EIGHT_RANK_FP16_CORE16_M0_DEFAULT,
              g_allgatherV2EightRankFP16CorE16M0Map}},
            {&cocTilingData.commDataSplit,
             {ALLGATHERV2_EIGHT_RANK_FP16_CORE16_COMMDATASPLIT_DEFAULT,
              g_allgatherV2EightRankFP16CorE16CommdatasplitMap}},
            {&cocTilingData.commDirect,
             {ALLGATHERV2_EIGHT_RANK_FP16_CORE16_COMMDIRECT_DEFAULT,
              g_allgatherV2EightRankFP16CorE16CommdirectMap}},
            {&cocTilingData.ubMoveNum,
             {ALLGATHERV2_EIGHT_RANK_FP16_CORE16_UBMOVENUM_DEFAULT,
              g_allgatherV2EightRankFP16CorE16UbmovenumMap}},
            {&cocTilingData.pValue,
             {ALLGATHERV2_EIGHT_RANK_FP16_CORE16_PVALUE_DEFAULT,
              g_allgatherV2EightRankFP16CorE16PvalueMap}},
            {&cocTilingData.swizzlDirect, {SWIZZLE_DIRECT_ONE}},
            {&cocTilingData.swizzlCount, {SWIZZLE_COUNT_FOUR}}
        };
        SetTilingParam(cocTilingData, tilingParamMap);

        int32_t coreNum = cocTilingData.blockDim - cocTilingData.rankSize;
        cocTilingData.commNpuSplit =
                cocTilingData.commDataSplit >= COMMDATASPLIT_EIGHT ? COMMNPUSPLIT_ONE : cocTilingData.rankSize;
        cocTilingData.commDataSplit =
                ClampValue(cocTilingData.commDataSplit, COMMDATASPLIT_ONE, coreNum / cocTilingData.commNpuSplit);
        cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;

        DealTilingParamByBuffSize(cocTilingData);
    }
}