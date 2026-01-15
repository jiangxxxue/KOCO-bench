/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tiling_910B.h"
#include "tiling_func.h"
namespace Lcal {
    constexpr int32_t REDUCESCATTER_FOUR_RANK_INT8_PVALUE_DEFAULT = 14;
    constexpr int32_t REDUCESCATTER_FOUR_RANK_INT8_UBMOVENUM_DEFAULT = 8;
    constexpr int32_t REDUCESCATTER_FOUR_RANK_INT8_M0_DEFAULT = 128;
    constexpr int32_t REDUCESCATTER_EIGHT_RANK_FP16_M0_DEFAULT = 128;
    constexpr int32_t REDUCESCATTER_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT = 20;
    constexpr int32_t REDUCESCATTER_EIGHT_RANK_FP16_COMMDATASPLIT_DEFAULT = 16;
    constexpr int32_t REDUCESCATTER_EIGHT_RANK_FP16_PVALUE_DEFAULT = 12;

    static std::map<int, std::vector<std::vector<int>>> g_reducescatterFourRankINT8M0Map = {
        {128,
         {{-1, 2560, -1, 7680, -1, 1536}, {-1, 1536, 7680, 2147483647, -1, 1536},
          {1536, 2560, 8704, 2147483647, -1, 1536}, {3584, 2147483647, -1, 4608, -1, 1536},
          {8704, 2147483647, 4608, 5632, -1, 1536}, {2560, 3584, 5632, 2147483647, -1, 1536},
          {-1, 2147483647, -1, 2147483647, 1536, 2147483647}}},
        {256,
         {{1536, 2560, 7680, 8704, -1, 1536}, {2560, 3584, -1, 4608, -1, 1536},
          {2560, 8704, 4608, 5632, -1, 1536}, {3584, 2147483647, 5632, 2147483647, -1, 1536}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_reducescatterFourRankINT8UbmovenumMap = {
        {8.0,
         {{-1, 1536, -1, 7168, -1, 1536}, {-1, 1536, -1, 2560, 1536, 3584},
          {1536, 2147483647, -1, 1536, -1, 1536}, {1536, 2560, 1536, 4608, -1, 1536}}},
        {6.0,
         {{-1, 1536, 7168, 2147483647, -1, 1536}, {-1, 1536, 2560, 2147483647, 1536, 3584},
          {-1, 1536, -1, 4608, 3584, 13312}, {1536, 2147483647, -1, 1536, 1536, 13312},
          {1536, 2560, 1536, 4608, 1536, 13312}, {2560, 2147483647, 1536, 4608, -1, 13312},
          {1536, 2560, 4608, 5632, -1, 6144}, {-1, 2147483647, -1, 4608, 13312, 2147483647},
          {5632, 6656, 9728, 2147483647, 13312, 2147483647}}},
        {4.0,
         {{-1, 1536, 4608, 2147483647, 3584, 13312}, {1536, 2560, 4608, 5632, 6144, 13312},
          {2560, 2147483647, 4608, 5632, -1, 13312}, {1536, 2147483647, 5632, 2147483647, -1, 13312},
          {-1, 5632, 4608, 2147483647, 13312, 2147483647}, {5632, 2147483647, 4608, 9728, 13312, 2147483647},
          {6656, 2147483647, 9728, 2147483647, 13312, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_reducescatterFourRankINT8PvalueMap = {
        {12,
         {{-1, 1536, -1, 4096, -1, 1536}, {5632, 2147483647, -1, 2560, 3584, 5632}}},
        {1,
         {{-1, 3584, 4096, 2147483647, -1, 1536}, {-1, 3584, 6656, 2147483647, 1536, 3584},
          {4608, 7680, 7680, 2147483647, -1, 3584}, {9728, 2147483647, 8192, 2147483647, -1, 1536},
          {-1, 1536, 6656, 9728, 3584, 2147483647}, {-1, 1536, 9728, 2147483647, 9728, 2147483647},
          {1536, 2560, 7680, 2147483647, 3584, 11264}}},
        {2,
         {{1536, 3584, -1, 4096, -1, 1536}, {-1, 3584, -1, 6656, 1536, 3584},
          {3584, 4608, -1, 2147483647, -1, 2560}, {4608, 7680, 4608, 7680, -1, 3584},
          {7680, 9728, -1, 2147483647, -1, 1536}, {9728, 2147483647, -1, 8192, -1, 1536},
          {-1, 1536, 4608, 6656, 3584, 2147483647}, {-1, 1536, 9728, 2147483647, 3584, 9728},
          {1536, 2560, 5632, 7680, 3584, 2147483647}, {1536, 2560, 7680, 2147483647, 11264, 2147483647}}},
        {4,
         {{3584, 4608, -1, 6144, 2560, 3584}, {4608, 7680, 1536, 4608, -1, 3584},
          {-1, 1536, 1536, 4608, 3584, 2147483647}, {1536, 2560, -1, 4608, 4608, 7680},
          {5632, 6656, 4608, 5632, 3584, 2147483647}, {6656, 8704, 4608, 2147483647, 6656, 2147483647}}},
        {3,
         {{3584, 4608, 6144, 2147483647, 2560, 3584}, {7680, 8704, 4608, 2147483647, 1536, 3584},
          {8704, 2147483647, 5632, 2147483647, 1536, 3584}, {1536, 2560, -1, 4608, 3584, 4608},
          {1536, 2560, 4608, 5632, 3584, 2147483647}, {2560, 5632, 4608, 2147483647, 3584, 2147483647},
          {5632, 6656, 5632, 2147483647, 3584, 2147483647}, {6656, 8704, 4608, 2147483647, 3584, 6656},
          {8704, 2147483647, 4608, 2147483647, 3584, 2147483647}}},
        {8,
         {{4608, 7680, -1, 1536, -1, 3584}, {2560, 5632, -1, 2560, 3584, 7680},
          {4608, 5632, 2560, 4608, 3584, 2147483647}, {5632, 2147483647, 2560, 4608, 3584, 9728}}},
        {6,
         {{7680, 8704, -1, 4608, 1536, 3584}, {8704, 2147483647, -1, 5632, 1536, 3584},
          {-1, 1536, -1, 1536, 3584, 2147483647}, {1536, 2560, 1536, 4608, 7680, 2147483647},
          {2560, 4608, 2560, 4608, 3584, 2147483647}}},
        {10,
         {{1536, 2560, -1, 1536, 7680, 2147483647}}},
        {14,
         {{2560, 5632, -1, 2560, 7680, 2147483647}, {5632, 2147483647, -1, 2560, 5632, 2147483647},
          {5632, 2147483647, 2560, 4608, 9728, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_reducescatterEightRankFP16PvalueMap = {
        {2,
         {{-1, 1536, -1, 2147483647, -1, 1536}, {1536, 5632, 1536, 2147483647, -1, 1536},
          {-1, 1536, -1, 2147483647, 1536, 2560}, {1536, 5632, 1536, 2147483647, 1536, 2560},
          {5632, 6656, 1536, 2560, -1, 1536}, {5632, 2147483647, 2560, 2147483647, -1, 2560},
          {-1, 4608, 1536, 2560, 2560, 4608}, {-1, 2147483647, 2560, 2147483647, 2560, 2147483647}}},
        {4,
         {{1536, 6656, -1, 1536, -1, 2560}, {5632, 6656, 1536, 2560, 1536, 2560},
          {6656, 2147483647, 1536, 2560, -1, 2560}, {-1, 4608, -1, 1536, 2560, 5632},
          {-1, 4608, 1536, 2560, 4608, 5632}, {4608, 8704, -1, 2560, 2560, 3584},
          {8704, 2147483647, 1536, 2560, 2560, 5632}, {-1, 2560, -1, 2560, 5632, 2147483647},
          {2560, 2147483647, 1536, 2560, 5632, 2147483647}}},
        {6,
         {{6656, 8704, -1, 1536, -1, 2560}}},
        {8,
         {{8704, 2147483647, -1, 1536, -1, 2560}, {4608, 8704, -1, 2560, 3584, 5632},
          {2560, 6656, -1, 1536, 5632, 2147483647}}},
        {10,
         {{8704, 2147483647, -1, 1536, 2560, 5632}}},
        {12,
         {{6656, 2147483647, -1, 1536, 5632, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_reducescatterEightRankFP16CommdatasplitMap = {
        {16,
         {{-1, 9728, -1, 2147483647, -1, 1536}, {9728, 2147483647, -1, 9728, -1, 1536},
          {-1, 2147483647, -1, 2147483647, 1536, 2147483647}}},
        {8,
         {{9728, 2147483647, 9728, 2147483647, -1, 1536}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_reducescatterEightRankFP16UbmovenumMap = {
        {8.0,
         {{-1, 1536, -1, 4096, -1, 1536}, {-1, 1536, 7168, 8704, -1, 1536},
          {1536, 2560, -1, 7680, -1, 1536}, {-1, 2560, 8704, 2147483647, -1, 1536},
          {2560, 2147483647, -1, 1536, -1, 1536}, {3584, 2147483647, 7680, 8704, -1, 1536},
          {6144, 2147483647, 8704, 9728, -1, 1536}, {2560, 3584, 9728, 2147483647, -1, 1536},
          {-1, 1536, -1, 3584, 1536, 2560}, {-1, 1536, -1, 5120, 5632, 7680},
          {1536, 2560, -1, 1536, 1536, 2147483647}, {1536, 2560, 9728, 2147483647, 11264, 2147483647}}},
        {10.0,
         {{-1, 1536, 4096, 7168, -1, 1536}, {1536, 2560, 7680, 8704, -1, 1536},
          {2560, 2147483647, 1536, 7680, -1, 1536}, {2560, 3584, 7680, 8704, -1, 1536},
          {2560, 6144, 8704, 9728, -1, 1536}, {3584, 9728, 9728, 2147483647, -1, 1536},
          {-1, 1536, -1, 3584, 2560, 5632}, {-1, 1536, 3584, 2147483647, 1536, 5632},
          {-1, 1536, -1, 5120, 7680, 13312}, {-1, 1536, 5120, 2147483647, 5632, 13312},
          {-1, 1536, -1, 5120, 13312, 2147483647}, {-1, 1536, 7680, 2147483647, 13312, 2147483647},
          {2560, 2147483647, -1, 1536, 1536, 2147483647}, {1536, 2147483647, 1536, 9728, 1536, 2147483647},
          {1536, 2147483647, 9728, 2147483647, 1536, 11264}, {2560, 2147483647, 9728, 2147483647, 11264, 2147483647}}},
        {20.0,
         {{9728, 2147483647, 9728, 2147483647, -1, 1536}}},
        {6.0,
         {{-1, 1536, 5120, 7680, 13312, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_reducescatterEightRankFP16M0Map = {
        {128,
         {{-1, 5632, -1, 2147483647, -1, 1536}, {5632, 8704, -1, 3584, -1, 1536},
          {-1, 8704, -1, 2147483647, 1536, 7680}, {8704, 2147483647, -1, 2147483647, -1, 7680},
          {-1, 2147483647, -1, 3584, 7680, 2147483647}, {-1, 1536, 3584, 2147483647, 7680, 2147483647},
          {2560, 2147483647, 3584, 2147483647, 7680, 2147483647}}},
        {256,
         {{5632, 8704, 3584, 2147483647, -1, 1536}, {1536, 2560, 3584, 2147483647, 7680, 2147483647}}}
    };

    void ReduceScatterFourRankINT8Tiling(CoCTilingData &cocTilingData)
    {
        std::map<int*, TilingValue> TilingParamMap = {
            {&cocTilingData.m0,
             {REDUCESCATTER_FOUR_RANK_INT8_M0_DEFAULT,
              g_reducescatterFourRankINT8M0Map}},
            {&cocTilingData.ubMoveNum,
             {REDUCESCATTER_FOUR_RANK_INT8_UBMOVENUM_DEFAULT,
              g_reducescatterFourRankINT8UbmovenumMap}},
            {&cocTilingData.pValue,
             {REDUCESCATTER_FOUR_RANK_INT8_PVALUE_DEFAULT,
              g_reducescatterFourRankINT8PvalueMap}},
            {&cocTilingData.swizzlDirect, {SWIZZLE_DIRECT_ONE}},
            {&cocTilingData.swizzlCount, {DEFAULT_SWIZZLE_COUNT}},
            {&cocTilingData.commDirect, {COMM_DATA_DIRECT}},
            {&cocTilingData.commNpuSplit, {COMMNPUSPLIT_ONE}},
            {&cocTilingData.commDataSplit, {COMMDATASPLIT_SIXTEEN}},
        };
        SetTilingParam(cocTilingData, TilingParamMap);

        cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
    }

    void ReduceScatterEightRankFP16GetDefaultTiling(CoCTilingData &cocTilingData)
    {
        std::map<int*, TilingValue> TilingParamMap = {
            {&cocTilingData.pValue,
             {REDUCESCATTER_EIGHT_RANK_FP16_PVALUE_DEFAULT,
              g_reducescatterEightRankFP16PvalueMap}},
            {&cocTilingData.commDataSplit,
             {REDUCESCATTER_EIGHT_RANK_FP16_COMMDATASPLIT_DEFAULT,
              g_reducescatterEightRankFP16CommdatasplitMap}},
            {&cocTilingData.ubMoveNum,
             {REDUCESCATTER_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT,
              g_reducescatterEightRankFP16UbmovenumMap}},
            {&cocTilingData.m0,
             {REDUCESCATTER_EIGHT_RANK_FP16_M0_DEFAULT,
              g_reducescatterEightRankFP16M0Map}},
            {&cocTilingData.swizzlDirect, {SWIZZLE_DIRECT_ONE}},
            {&cocTilingData.swizzlCount, {SWIZZLE_COUNT_FOUR}},
            {&cocTilingData.commDirect, {COMM_DATA_DIRECT}},
            {&cocTilingData.commNpuSplit, {COMMNPUSPLIT_ONE}},
        };
        SetTilingParam(cocTilingData, TilingParamMap);

        cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
    }
}