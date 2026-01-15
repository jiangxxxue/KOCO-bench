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
    constexpr int32_t ALLGATHER_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT = 10;
    constexpr int32_t ALLGATHER_EIGHT_RANK_FP16_PVALUE_DEFAULT = 8;
    constexpr int32_t ALLGATHER_EIGHT_RANK_FP16_COMMDIRECT_DEFAULT = 1;
    constexpr int32_t ALLGATHER_EIGHT_RANK_FP16_COMMDATASPLIT_DEFAULT = 16;
    constexpr int32_t ALLGATHER_EIGHT_RANK_FP16_M0_DEFAULT = 128;

    constexpr int32_t ALLGATHER_FOUR_RANK_INT8_UBMOVENUM_DEFAULT = 4;
    constexpr int32_t ALLGATHER_FOUR_RANK_INT8_COMMDATASPLIT_DEFAULT = 16;
    constexpr int32_t ALLGATHER_FOUR_RANK_INT8_PVALUE_DEFAULT = 14;
    constexpr int32_t ALLGATHER_FOUR_RANK_INT8_M0_DEFAULT = 128;

    const int UBMOVE_MAX_M = 10096;
    const int UBMOVE_MAX_K = 6656;
    const int UBMOVE_MAX_N = 2336;
    const int UBMOVE_MAX_M_HIGH = 14144;
    const int UBMOVE_DEFAULT = 20;
    const int UBMOVE_SMALL_M_SMALL_K_SMALL_N = 50;
    const int UBMOVE_SMALL_M_LARGE_K_SMALL_N = 60;
    const int UBMOVE_LARGE_M_SMALL_N = 80;
    const int SCALE_FACTOR = 512;
    const int PVALUE_M_SMALL = 6144;
    const int PVALUE_K_SMALL = 10240;
    const int PVALUE_N_MEDIUM = 9216;
    const int PVALUE_M_MEDIUM = 14144;
    const int PVALUE_M_LARGE = 10096;
    const int PVALUE_ONE = 1;
    const int PVALUE_TWO = 2;
    const int PVALUE_THREE = 3;
    const int PVALUE_FOUR = 4;

    static std::vector<double> g_allgatherSwizzldirectCoef = {
        { -2.462e-04, -7.154e-06, 6.700e-05,  1.416e-06,  1.747e-04,  1.513e-07,  2.296e-02,  -3.022e-04, -6.992e-03,
          -1.865e-03, 8.685e-03,  -2.039e-03, -1.701e-02, 1.805e-03,  1.174e-03,  -5.262e-03, 3.752e-05,  -1.539e-05,
          -2.508e-02, -9.660e-05, 2.489e-03,  -7.638e-03, -1.360e-03, -3.614e-04, -1.150e-03 }
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherEightRankFP16M0Map = {
        {128,
         {{-1, 1262, -1, 2147483647, -1, 576}, {-1, 1262, 5660, 2147483647, 576, 1200},
          {-1, 1262, -1, 2147483647, 1200, 5552}, {1262, 8958, -1, 2274, 1888, 5552},
          {8958, 2147483647, -1, 2147483647, 1536, 5552}, {-1, 2147483647, -1, 2274, 6684, 2147483647},
          {9728, 2147483647, 2786, 3286, 5552, 2147483647}, {-1, 768, 3286, 7696, 5552, 2147483647}}},
        {256,
         {{-1, 1262, -1, 5660, 576, 1200}, {1262, 8958, -1, 2147483647, -1, 1888},
          {1262, 8958, 2274, 2147483647, 1888, 5552}, {8958, 2147483647, -1, 2147483647, -1, 1536},
          {-1, 2147483647, -1, 2274, 5552, 6684}, {-1, 2147483647, 2274, 2786, 5552, 2147483647},
          {-1, 9728, 2786, 3286, 5552, 2147483647}, {-1, 768, 7696, 2147483647, 5552, 2147483647},
          {768, 2147483647, 3286, 2147483647, 5552, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherEightRankFP16CommdatasplitMap = {
        {8,
         {{-1, 1262, -1, 768, -1, 576}, {-1, 2274, 768, 2147483647, -1, 576},
          {-1, 2274, -1, 2147483647, 576, 1624}, {-1, 1512, -1, 2147483647, 1624, 1720},
          {-1, 2274, -1, 2147483647, 1720, 2274}, {2274, 9728, -1, 2147483647, -1, 768},
          {9728, 2147483647, 768, 2147483647, -1, 768}, {2786, 2147483647, -1, 4298, 768, 1774},
          {2274, 2147483647, 4298, 2147483647, 768, 1774}, {2274, 3584, -1, 1262, 1774, 2274},
          {3584, 2147483647, 1774, 2147483647, 1774, 2274}, {-1, 768, 6700, 2147483647, 2626, 3248}}},
        {16,
         {{1262, 2274, -1, 768, -1, 576}, {1512, 2274, -1, 2147483647, 1624, 1720},
          {9728, 2147483647, -1, 768, -1, 768}, {2274, 2786, -1, 4298, 768, 1774},
          {2274, 3584, 2798, 2147483647, 1774, 2274}, {768, 2147483647, 6700, 2147483647, 2626, 2912}}},
        {2,
         {{2274, 3584, 1262, 2798, 1774, 2274}, {3584, 2147483647, -1, 1774, 1774, 2274},
          {-1, 2147483647, -1, 2147483647, 2274, 2626}, {-1, 2147483647, -1, 6700, 2626, 3248},
          {768, 2147483647, 6700, 2147483647, 2912, 3248}, {-1, 2147483647, -1, 2147483647, 3248, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherEightRankFP16CommdirectMap = {
        {1,
         {{-1, 768, -1, 2147483647, -1, 4608}, {-1, 768, 768, 2147483647, 4608, 5824},
          {768, 2147483647, -1, 2147483647, -1, 2912}, {1774, 2147483647, -1, 768, 2912, 3584},
          {768, 2147483647, -1, 768, 3584, 6172}, {768, 2147483647, 768, 2147483647, 2912, 6172},
          {-1, 6950, -1, 2147483647, 6172, 2147483647}, {6950, 7450, -1, 2560, 6172, 2147483647},
          {6950, 7450, 2560, 3584, 6172, 8704}, {6950, 7450, 3584, 2147483647, 6172, 2147483647},
          {7450, 2147483647, -1, 2147483647, 6172, 2147483647}}},
        {0,
         {{-1, 768, -1, 768, 4608, 5824}, {-1, 768, -1, 2147483647, 5824, 6172},
          {768, 1774, -1, 768, 2912, 3584}, {6950, 7450, 2560, 3584, 8704, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherEightRankFP16PvalueMap = {
        {1,
         {{-1, 768, -1, 2147483647, -1, 576}, {768, 1262, 7196, 2147483647, -1, 576},
          {1262, 4298, -1, 2147483647, -1, 576}, {-1, 4298, 768, 2147483647, 576, 1518},
          {-1, 4298, -1, 2147483647, 1518, 1984}, {-1, 2560, -1, 5660, 1984, 2274},
          {-1, 4298, 5660, 2147483647, 1984, 2274}, {4810, 2147483647, 7946, 10720, -1, 2274},
          {-1, 8958, -1, 2147483647, 2274, 2912}, {-1, 8958, 2560, 2147483647, 2912, 3248},
          {-1, 8958, -1, 1262, 3248, 5660}, {-1, 8958, 1262, 2147483647, 3248, 6450},
          {-1, 8958, -1, 2147483647, 6450, 2147483647}, {8958, 9728, -1, 2147483647, 2274, 5660},
          {8958, 9728, 1536, 2147483647, 5660, 6684}, {8958, 9728, -1, 2147483647, 6684, 2147483647},
          {9728, 2147483647, -1, 768, 6684, 2147483647}, {9728, 2147483647, 768, 3584, 4608, 2147483647},
          {9728, 2147483647, 3584, 2147483647, 2274, 2147483647}}},
        {6,
         {{768, 1262, -1, 7196, -1, 576}, {-1, 4298, -1, 768, 576, 1518},
          {4810, 7850, -1, 1262, -1, 2274}, {4810, 7450, 1262, 3286, -1, 2274}}},
        {4,
         {{2560, 4298, -1, 5660, 1984, 2274}, {7450, 2147483647, 1262, 3286, -1, 2274},
          {9728, 2147483647, -1, 768, 2274, 3584}}},
        {2,
         {{4298, 4810, -1, 2147483647, -1, 2274}, {4810, 2147483647, 3286, 7946, -1, 2274},
          {4810, 2147483647, 10720, 2147483647, -1, 2274}, {-1, 8958, -1, 2560, 2912, 3248},
          {-1, 8958, -1, 1262, 5660, 6450}, {8958, 9728, -1, 1536, 5660, 6684},
          {9728, 2147483647, -1, 768, 3584, 6684}, {9728, 2147483647, 768, 3584, 2274, 4608}}},
        {8,
         {{7850, 2147483647, -1, 1262, -1, 2274}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherEightRankFP16UbmovenumMap = {
        {3.0,
         {{-1, 1262, -1, 2147483647, -1, 832}, {-1, 1262, 768, 2147483647, 832, 2400},
          {1262, 2147483647, 768, 2147483647, -1, 1624}, {1262, 2147483647, 1774, 2147483647, 1624, 2274},
          {1262, 2147483647, -1, 1262, 6684, 7434}, {7850, 2147483647, 1262, 1774, 5552, 7434},
          {1262, 2147483647, 1774, 2147483647, 5552, 7434}, {1262, 2147483647, -1, 768, 7434, 8704},
          {-1, 768, 1262, 3286, 7434, 9728}, {768, 1262, 3286, 2147483647, 7434, 9728}}},
        {2.0,
         {{-1, 1262, -1, 768, 832, 2400}, {-1, 1262, 7696, 2147483647, 6684, 7434},
          {1262, 2147483647, -1, 768, -1, 1624}, {1262, 2147483647, -1, 1262, 5552, 6684},
          {1262, 7850, 1262, 1774, 5552, 7434}, {-1, 1262, -1, 768, 7434, 8704},
          {-1, 2147483647, -1, 768, 8704, 9728}, {768, 2147483647, 768, 3286, 7434, 9728},
          {1262, 2147483647, 3286, 2147483647, 7434, 9728}, {-1, 7200, -1, 2147483647, 9728, 2147483647},
          {7200, 2147483647, -1, 2147483647, 9728, 11744}, {7200, 2147483647, -1, 11744, 11744, 2147483647}}},
        {8.0,
         {{-1, 1262, -1, 2147483647, 2400, 6172}, {1262, 2147483647, -1, 2147483647, 2274, 3248},
          {-1, 768, 3286, 2147483647, 7434, 9728}}},
        {6.0,
         {{-1, 768, -1, 2147483647, 6172, 6684}, {-1, 1262, -1, 7696, 6684, 7434},
          {1262, 2147483647, -1, 2147483647, 3248, 4298}, {1262, 2147483647, -1, 768, 4298, 5552}}},
        {4.0,
         {{768, 1262, -1, 2147483647, 6172, 6684}, {1262, 2147483647, 768, 2147483647, 4298, 5552},
          {-1, 768, 768, 1262, 7434, 9728}, {7200, 2147483647, 11744, 2147483647, 11744, 2147483647}}},
        {10.0,
         {{1262, 2147483647, -1, 1774, 1624, 2274}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherFourRankINT8M0Map = {
        {128,
         {{-1, 2147483647, -1, 2147483647, -1, 3584}, {-1, 2147483647, -1, 4608, 3584, 8704},
          {-1, 2560, 4608, 2147483647, 3584, 8704}, {-1, 2147483647, -1, 2560, 8704, 2147483647}}},
        {256,
         {{2560, 2147483647, 4608, 2147483647, 3584, 8704}, {-1, 2147483647, 2560, 2147483647, 8704, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherFourRankINT8PvalueMap = {
        {12,
         {{-1, 5632, -1, 1792, -1, 1536}, {9728, 2147483647, -1, 8704, -1, 2560}}},
        {10,
         {{-1, 5632, 1792, 3584, -1, 1536}, {2560, 5632, -1, 1280, 1536, 2560},
          {5632, 7680, -1, 2147483647, -1, 4608}, {7680, 9728, -1, 2147483647, -1, 2560},
          {7680, 9728, -1, 5632, 3584, 4608}, {9728, 2147483647, 8704, 2147483647, 1536, 2560},
          {9728, 2147483647, 1280, 2147483647, 3584, 4608}}},
        {6,
         {{-1, 5632, 3584, 8704, -1, 1536}, {-1, 2560, -1, 1280, 1536, 2560},
          {3584, 5632, -1, 2147483647, 2560, 3584}, {1536, 5632, -1, 6656, 3584, 4608},
          {9728, 2147483647, 8704, 2147483647, -1, 1536}}},
        {3,
         {{-1, 5632, 8704, 2147483647, -1, 1536}, {-1, 5632, 1280, 2147483647, 1536, 2560},
          {768, 2560, -1, 2147483647, 2560, 3584}, {1536, 5632, 6656, 2147483647, 3584, 4608},
          {7680, 9728, 5632, 2147483647, 3584, 4608}, {1536, 9728, -1, 1280, 4608, 8704},
          {9728, 2147483647, -1, 2560, 4608, 8704}, {1536, 2147483647, 2560, 6656, 4608, 5632}}},
        {1,
         {{-1, 768, -1, 2147483647, 2560, 3584}, {-1, 1536, 4096, 2147483647, 3584, 4608},
          {-1, 768, -1, 2560, 4608, 8704}, {-1, 1536, 2560, 2147483647, 4608, 5632},
          {1536, 2147483647, 6656, 2147483647, 4608, 5632}, {-1, 2147483647, 2560, 2147483647, 5632, 8704},
          {-1, 768, -1, 2147483647, 8704, 2147483647}, {768, 2147483647, -1, 768, 9728, 11264},
          {768, 2147483647, 768, 2147483647, 8704, 11264}, {768, 1536, 4096, 2147483647, 11264, 2147483647},
          {1536, 2147483647, -1, 2147483647, 11264, 2147483647}}},
        {4,
         {{2560, 3584, -1, 2147483647, 2560, 3584}}},
        {2,
         {{-1, 1536, -1, 4096, 3584, 4608}, {768, 1536, -1, 2560, 4608, 8704},
          {1536, 9728, 1280, 2560, 4608, 8704}, {768, 2147483647, -1, 768, 8704, 9728},
          {768, 1536, -1, 4096, 11264, 2147483647}}},
        {14,
         {{7680, 9728, -1, 2147483647, 2560, 3584}, {9728, 2147483647, -1, 1280, 2560, 4608},
          {9728, 2147483647, 1280, 2147483647, 2560, 3584}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherFourRankINT8CommdatasplitMap = {
        {16,
         {{-1, 2147483647, -1, 2147483647, -1, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allgatherFourRankINT8UbmovenumMap = {
        {4.0,
         {{-1, 2560, -1, 2147483647, -1, 3584}, {-1, 2560, 1792, 2147483647, 3584, 4608},
          {2560, 2147483647, -1, 1792, -1, 2560}, {2560, 4608, -1, 1792, 2560, 4608},
          {2560, 2147483647, 1792, 2147483647, -1, 3584}, {9728, 2147483647, 1792, 2147483647, 3584, 4608},
          {-1, 768, -1, 4096, 4608, 5632}}},
        {2.0,
         {{-1, 2560, -1, 1792, 3584, 4608}, {-1, 768, 4096, 2147483647, 4608, 5632},
          {-1, 768, -1, 2147483647, 5632, 2147483647}, {768, 2147483647, -1, 2147483647, 4608, 2147483647}}},
        {3.0,
         {{4608, 2147483647, -1, 1792, 2560, 4608}, {2560, 9728, 1792, 2147483647, 3584, 4608}}}
    };

    void AllGatherFourRankINT8Tiling(CoCTilingData &cocTilingData)
    {
        std::map<int*, TilingValue> TilingParamMap = {
            {&cocTilingData.m0,
             {ALLGATHER_FOUR_RANK_INT8_M0_DEFAULT,
              g_allgatherFourRankINT8M0Map}},
            {&cocTilingData.pValue,
             {ALLGATHER_FOUR_RANK_INT8_PVALUE_DEFAULT,
              g_allgatherFourRankINT8PvalueMap}},
            {&cocTilingData.commDataSplit,
             {ALLGATHER_FOUR_RANK_INT8_COMMDATASPLIT_DEFAULT,
              g_allgatherFourRankINT8CommdatasplitMap}},
            {&cocTilingData.ubMoveNum,
             {ALLGATHER_FOUR_RANK_INT8_UBMOVENUM_DEFAULT,
              g_allgatherFourRankINT8UbmovenumMap}},
            {&cocTilingData.swizzlDirect, {SWIZZLE_DIRECT_ZERO}},
            {&cocTilingData.swizzlCount, {DEFAULT_SWIZZLE_COUNT}},
            {&cocTilingData.commDirect, {COMM_NPU_DIRECT}},
            {&cocTilingData.commNpuSplit, {COMMNPUSPLIT_ONE}},
        };
        SetTilingParam(cocTilingData, TilingParamMap);

        cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;
        DealTilingParamByBuffSize(cocTilingData);
    }

    void AllGatherEightRankFP16GetDefaultTiling(CoCTilingData &cocTilingData)
    {
        std::map<int*, TilingValue> TilingParamMap = {
            {&cocTilingData.m0,
             {ALLGATHER_EIGHT_RANK_FP16_M0_DEFAULT,
              g_allgatherEightRankFP16M0Map}},
            {&cocTilingData.commDataSplit,
             {ALLGATHER_EIGHT_RANK_FP16_COMMDATASPLIT_DEFAULT,
              g_allgatherEightRankFP16CommdatasplitMap}},
            {&cocTilingData.commDirect,
             {ALLGATHER_EIGHT_RANK_FP16_COMMDIRECT_DEFAULT,
              g_allgatherEightRankFP16CommdirectMap}},
            {&cocTilingData.pValue,
             {ALLGATHER_EIGHT_RANK_FP16_PVALUE_DEFAULT,
              g_allgatherEightRankFP16PvalueMap}},
            {&cocTilingData.ubMoveNum,
             {ALLGATHER_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT,
              g_allgatherEightRankFP16UbmovenumMap}},
            {&cocTilingData.swizzlDirect, {SWIZZLE_DIRECT_ZERO}},
            {&cocTilingData.swizzlCount, {SWIZZLE_COUNT_FOUR}}
        };
        SetTilingParam(cocTilingData, TilingParamMap);

        cocTilingData.commNpuSplit =
                cocTilingData.commDataSplit >= COMMDATASPLIT_EIGHT ? COMMNPUSPLIT_ONE : cocTilingData.rankSize;
        cocTilingData.commDataSplit = ClampValue(cocTilingData.commDataSplit, COMMDATASPLIT_ONE,
                                                 cocTilingData.blockDim / cocTilingData.commNpuSplit);
        cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;

        DealTilingParamByBuffSize(cocTilingData);
    }

    int AllGatherUbMoveNum(int m, int k, int n)
    {
        if (m <= UBMOVE_MAX_M) {
            if (k <= UBMOVE_MAX_K) {
                if (n <= UBMOVE_MAX_N) {
                    return UBMOVE_SMALL_M_SMALL_K_SMALL_N * SCALE_FACTOR;
                } else {
                    return UBMOVE_DEFAULT * SCALE_FACTOR;
                }
            } else {
                return UBMOVE_SMALL_M_LARGE_K_SMALL_N * SCALE_FACTOR;
            }
        } else {
            if (n <= UBMOVE_MAX_N) {
                if (m <= UBMOVE_MAX_M_HIGH) {
                    return UBMOVE_LARGE_M_SMALL_N * SCALE_FACTOR;
                } else {
                    return UBMOVE_SMALL_M_LARGE_K_SMALL_N * SCALE_FACTOR;
                }
            } else {
                return UBMOVE_DEFAULT * SCALE_FACTOR;
            }
        }
        return UBMOVE_DEFAULT * SCALE_FACTOR;
    }

    int AllGatherPValue(int m, int k, int n)
    {
        if (m <= PVALUE_M_SMALL) {
            if (k <= PVALUE_K_SMALL) {
                return PVALUE_ONE;
            } else {
                if (n <= PVALUE_N_MEDIUM) {
                    return PVALUE_ONE;
                } else {
                    return PVALUE_TWO;
                }
            }
        } else {
            if (n <= PVALUE_N_MEDIUM) {
                if (m <= PVALUE_M_MEDIUM) {
                    return PVALUE_ONE;
                } else {
                    return PVALUE_THREE;
                }
            } else {
                if (m <= PVALUE_M_LARGE) {
                    return PVALUE_THREE;
                } else {
                    return PVALUE_FOUR;
                }
            }
        }
    }

    void AllGatherGetDefaultTiling(CoCTilingData &cocTilingData)
    {
        int32_t m = cocTilingData.m;
        int32_t k = cocTilingData.k;
        int32_t n = cocTilingData.n;
        double mknGB = (1.0 * m / ONE_K) * (1.0 * k / ONE_K) * (1.0 * n / ONE_K);
        double mkGB = (1.0 * m / ONE_K) * (1.0 * k / ONE_K);
        double mnGB = (1.0 * m / ONE_K) * (1.0 * n / ONE_K);
        double knGB = (1.0 * k / ONE_K) * (1.0 * n / ONE_K);
        double c0 = sqrt(1.0 * m / k);
        double c1 = 1.0 * m * k / n;
        double c2 = sqrt(c1);
        double c3 = sqrt(m * k) / n;
        double c4 = sqrt(1.0 * k / n);
        double swizzlDirectDouble = 0;
        std::vector<double> feats = { 1.0 * m,     1.0 / m,  1.0 * k,    1.0 / k,  1.0 * n,    1.0 / n,  mknGB,
                                      1.0 / mknGB, mkGB,     1.0 / mkGB, mnGB,     1.0 / mnGB, knGB,     1.0 / knGB,
                                      c0,          1.0 / c0, c1,         1.0 / c1, c2,         1.0 / c2, c3,
                                      1.0 / c3,    c4,       1.0 / c4,   1 };
        for (uint32_t i = 0; i < feats.size(); i++) {
            swizzlDirectDouble += feats[i] * g_allgatherSwizzldirectCoef[i];
        }
        swizzlDirectDouble = 1.0 / (1.0 + exp(-swizzlDirectDouble));
        if (swizzlDirectDouble >= HALF_PROB) {
            cocTilingData.swizzlDirect = 1;
        } else {
            cocTilingData.swizzlDirect = 0;
        }

        cocTilingData.pValue = AllGatherPValue(m, k, n);
        cocTilingData.ubMoveNum = AllGatherUbMoveNum(m, k, n);
        cocTilingData.m0 = DEFAULT_ROW;
        cocTilingData.n0 = DEFAULT_COL;
        cocTilingData.k0 = DEFAULT_COL;
        cocTilingData.kLoop = CeilDev(k, cocTilingData.k0);

        cocTilingData.write2OtherRank = 1;
        cocTilingData.commDirect = COMM_DATA_DIRECT;
        cocTilingData.commNpuSplit = cocTilingData.rankSize;
        cocTilingData.commDataSplit = COMMDATASPLIT_ONE;
        DealTilingParamByBuffSize(cocTilingData);
        cocTilingData.lenPerLoop = cocTilingData.m0 * cocTilingData.k0 * cocTilingData.kLoop * cocTilingData.pValue;
    }
}