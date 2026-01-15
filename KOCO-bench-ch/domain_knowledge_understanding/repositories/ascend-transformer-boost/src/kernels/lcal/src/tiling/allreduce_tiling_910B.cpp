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
    const int32_t ALLREDUCE_SERIAL_MODE_K_SIZE = 8192;
    const int64_t ALLREDUCE_SERIAL_MODE_MN_SIZE = 256 * 256 * 12;

    constexpr int32_t ALLREDUCE_FOUR_RANK_FP16_DATASPLIT_DEFAULT = 32;
    constexpr int32_t ALLREDUCE_FOUR_RANK_FP16_PVALUE_DEFAULT = 8;
    constexpr int32_t ALLREDUCE_FOUR_RANK_FP16_UBMOVENUM_DEFAULT = 30;
    constexpr int32_t ALLREDUCE_FOUR_RANK_FP16_M0_DEFAULT = 128;
    constexpr int32_t ALLREDUCE_FOUR_RANK_INT8_UBMOVENUM_DEFAULT = 40;
    constexpr int32_t ALLREDUCE_FOUR_RANK_INT8_PVALUE_DEFAULT = 8;
    constexpr int32_t ALLREDUCE_FOUR_RANK_INT8_DATASPLIT_DEFAULT = 32;
    constexpr int32_t ALLREDUCE_FOUR_RANK_INT8_M0_DEFAULT = 128;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_FP16_PVALUE_DEFAULT = 14;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT = 100;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_FP16_DATASPLIT_DEFAULT = 16;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_FP16_M0_DEFAULT = 128;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_INT8_UBMOVENUM_DEFAULT = 100;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_INT8_PVALUE_DEFAULT = 14;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_INT8_DATASPLIT_DEFAULT = 8;
    constexpr int32_t ALLREDUCE_EIGHT_RANK_INT8_M0_DEFAULT = 128;
    constexpr int32_t ALLREDUCE_TWO_RANK_FP16_PVALUE_DEFAULT = 6;
    constexpr int32_t ALLREDUCE_TWO_RANK_FP16_M0_DEFAULT = 128;
    constexpr int32_t ALLREDUCE_TWO_RANK_FP16_SWIZZLCOUNT_DEFAULT = 8;
    constexpr int32_t ALLREDUCE_TWO_RANK_FP16_SWIZZLDIRECT_DEFAULT = 0;
    constexpr int32_t ALLREDUCE_TWO_RANK_FP16_UBMOVENUM_DEFAULT = 6;
    constexpr int32_t ALLREDUCE_TWO_RANK_FP16_COMMDATASPLIT_DEFAULT = 16;

    static std::vector<double> g_allreduceUbmovenumCoef = {
        { -1.72352427e+01, 2.56887672e-03, -8.21819480e+00, 8.70965589e+01, -3.63853858e-01, 1.27789264e+01,
          1.29782183e+02,  1.90250023e-02, -3.48175441e+00, 6.18921914e+03, 3.77072171e+03, -5.86895290e+01,
          -8.70740991e-01, -1.40262280e-04, -2.81910331e-08, 3.22795486e-05, -4.84522320e-03, 2.94839177e-01,
          2.97260958e-03, 9.08844709e+01, -5.80426209e-10, 38.183465184603484 }
    };
    static std::vector<double> g_allreducePvalueCoef = {
        { -4.23166350e+00, 6.71137487e-04, -1.33434156e+00, 1.12915884e+01, -7.85892737e-02, 2.59059897e+00,
          3.22129881e+01, -5.15776887e-02, 9.15542742e-01, 1.56322201e+03, 3.61977421e+01, -5.49544589e-01,
          -2.66903417e-01, -3.68521920e-05, -6.40666333e-09, 6.77406054e-06, -9.92992099e-04, 5.60658043e-02,
          2.69372863e-04, 2.17222337e+01, -1.17749660e-10, 6.100544547671263 }
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankInT8M0Map = {
        {256,
         {{-1, 3072, -1, 2147483647, -1, 768}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankInT8DatasplitMap = {
        {1,
         {{-1, 768, -1, 2147483647, -1, 768}}},
        {2,
         {{-1, 1536, -1, 2147483647, 768, 1536}}},
        {4,
         {{768, 10010, -1, 2147483647, -1, 768}}},
        {8,
         {{1536, 10010, -1, 2147483647, 768, 1536}, {10010, 2147483647, 3072, 2147483647, -1, 768},
          {-1, 19680, 7170, 2147483647, 1536, 7424}, {-1, 7350, 7170, 2147483647, 7424, 2147483647}}},
        {16,
         {{10010, 2147483647, 3072, 2147483647, 768, 1536}, {19680, 2147483647, 7170, 2147483647, 1536, 7424},
          {7350, 2147483647, 7170, 2147483647, 7424, 2147483647}}},
        {32,
         {{10010, 2147483647, -1, 3072, -1, 1536}, {-1, 2147483647, -1, 7170, 1536, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankInT8PvalueMap = {
        {1,
         {{-1, 10010, -1, 2147483647, -1, 768}, {-1, 5148, -1, 2147483647, 768, 1536}}},
        {2,
         {{5148, 10010, -1, 2147483647, 768, 1536}, {10010, 2147483647, 3072, 2147483647, -1, 768},
          {-1, 19680, 7170, 2147483647, 1536, 7424}, {-1, 7350, 7170, 2147483647, 7424, 2147483647}}},
        {4,
         {{19680, 2147483647, 7170, 2147483647, 1536, 7424}, {7350, 2147483647, 7170, 2147483647, 7424, 2147483647},
          {10010, 2147483647, 3072, 2147483647, 768, 1536}}},
        {6,
         {{10010, 36980, -1, 3072, -1, 1536}}},
        {8,
         {{36980, 2147483647, -1, 3072, -1, 1536}, {-1, 2147483647, -1, 7170, 1536, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankInT8UbmovenumMap = {
        {20,
         {{53340, 2147483647, 7170, 2147483647, -1, 3072}}},
        {30,
         {{-1, 53340, -1, 2147483647, -1, 3072}, {53340, 2147483647, -1, 4608, -1, 768},
          {53340, 2147483647, 4608, 7170, -1, 3072}, {-1, 7196, -1, 2147483647, 3072, 2147483647},
          {10010, 15412, -1, 2147483647, 3072, 2147483647}, {15412, 2147483647, 5634, 2147483647, 3072, 2147483647}}},
        {40,
         {{53340, 2147483647, -1, 4608, 768, 3072}, {7196, 10010, -1, 2147483647, 3072, 2147483647},
          {15412, 2147483647, -1, 5634, 3072, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankFP16M0Map = {
        {256,
         {{-1, 12980, -1, 2147483647, -1, 768}, {12980, 2147483647, -1, 5634, -1, 768},
          {-1, 63360, -1, 4608, 768, 2147483647}, {63360, 2147483647, -1, 4000, 768, 2147483647},
          {63360, 2147483647, 4000, 4608, 1536, 2147483647}, {-1, 5148, 4608, 11264, 768, 2147483647},
          {-1, 2560, 11264, 2147483647, 768, 2147483647}, {5148, 19680, 4608, 2147483647, 13312, 2147483647}}},
        {128,
         {{12980, 2147483647, 5634, 2147483647, -1, 768}, {63360, 2147483647, 4000, 4608, 768, 1536},
          {2560, 5148, 11264, 2147483647, 768, 2147483647}, {5148, 19680, 4608, 2147483647, 768, 13312},
          {19680, 2147483647, 4608, 2147483647, 768, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankFP16UbmovenumMap = {
        {20,
         {{-1, 1536, -1, 2147483647, -1, 1536}, {-1, 1536, 7170, 2147483647, 1536, 19456},
          {1536, 2147483647, 5634, 2147483647, -1, 19456}, {-1, 2147483647, -1, 2147483647, 19456, 2147483647}}},
        {30.0,
         {{-1, 1536, -1, 7170, 1536, 19456}, {1536, 2147483647, -1, 5634, -1, 19456}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankFP16PvalueMap = {
        {2,
         {{-1, 5148, -1, 1536, -1, 1536}, {-1, 5148, 1152, 4608, 3072, 5376},
          {5148, 31220, 3072, 2147483647, -1, 1536}, {-1, 3072, -1, 2147483647, 10240, 2147483647},
          {3072, 5148, 7170, 2147483647, 5376, 2147483647}, {13364, 142040, 7170, 2147483647, 5376, 7424},
          {142040, 2147483647, 7170, 2147483647, 5376, 2147483647}}},
        {1,
         {{-1, 5148, 1536, 2147483647, -1, 3072}, {-1, 5148, 4608, 2147483647, 3072, 5376},
          {-1, 3072, -1, 2147483647, 5376, 10240}}},
        {8,
         {{-1, 5148, -1, 1536, 1536, 3072}, {68160, 2147483647, -1, 3072, -1, 768},
          {16340, 2147483647, -1, 3072, 768, 5376}, {5148, 13364, -1, 2976, 5376, 2147483647},
          {13364, 2147483647, -1, 5634, 5376, 2147483647}, {13364, 2147483647, 5634, 7170, 10240, 2147483647}}},
        {4,
         {{-1, 5148, -1, 1152, 3072, 5376}, {5148, 68160, -1, 3072, -1, 768},
          {5148, 16340, -1, 3072, 768, 5376}, {5148, 31220, 3072, 2147483647, 1536, 5376},
          {31220, 2147483647, 3072, 2147483647, -1, 5376}, {3072, 5148, -1, 7170, 5376, 2147483647},
          {5148, 13364, 2976, 2147483647, 5376, 2147483647}, {13364, 2147483647, 5634, 7170, 5376, 10240},
          {13364, 142040, 7170, 2147483647, 7424, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceFourRankFP16DatasplitMap = {
        {8,
         {{-1, 5148, -1, 3072, -1, 1536}, {-1, 5148, 1152, 4608, 3072, 5376},
          {5148, 68160, 3072, 2147483647, -1, 1536}, {-1, 3072, -1, 2147483647, 10240, 2147483647},
          {3072, 5148, 7170, 2147483647, 5376, 2147483647}, {13364, 142040, 7170, 2147483647, 5376, 7424},
          {142040, 2147483647, 7170, 2147483647, 5376, 2147483647}}},
        {4,
         {{-1, 5148, 3072, 2147483647, -1, 1536}, {-1, 5148, 1536, 2147483647, 1536, 3072},
          {-1, 5148, 4608, 2147483647, 3072, 5376}, {-1, 3072, -1, 2147483647, 5376, 10240}}},
        {32,
         {{-1, 5148, -1, 1536, 1536, 3072}, {68160, 2147483647, -1, 3072, -1, 768},
          {16340, 2147483647, -1, 3072, 768, 5376}, {5148, 13364, -1, 2976, 5376, 2147483647},
          {13364, 2147483647, -1, 5634, 5376, 2147483647}, {13364, 2147483647, 5634, 7170, 10240, 2147483647}}},
        {16,
         {{-1, 5148, -1, 1152, 3072, 5376}, {5148, 68160, -1, 3072, -1, 768},
          {5148, 16340, -1, 3072, 768, 5376}, {5148, 68160, 3072, 2147483647, 1536, 5376},
          {68160, 2147483647, 3072, 2147483647, -1, 5376}, {3072, 5148, -1, 7170, 5376, 2147483647},
          {5148, 13364, 2976, 2147483647, 5376, 2147483647}, {13364, 2147483647, 5634, 7170, 5376, 10240},
          {13364, 142040, 7170, 2147483647, 7424, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankFP16M0Map = {
        {128,
         {{-1, 31220, -1, 2147483647, -1, 768}, {31220, 36980, 1280, 2147483647, -1, 768},
          {36980, 2147483647, -1, 2147483647, -1, 768}, {-1, 2147483647, -1, 2147483647, 768, 2147483647}}},
        {256,
         {{31220, 36980, -1, 1280, -1, 768}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankFP16DatasplitMap = {
        {1,
         {{-1, 3072, -1, 2147483647, -1, 768}, {3072, 26880, 3072, 2147483647, -1, 768},
          {-1, 1536, -1, 2147483647, 768, 1536}, {1536, 26880, 4608, 2147483647, 768, 1536},
          {26880, 53340, 4608, 2147483647, -1, 768}, {26880, 53340, 3072, 2147483647, 768, 1536},
          {53340, 2147483647, 3072, 2147483647, -1, 1536}, {-1, 768, 4608, 2147483647, 1536, 2147483647},
          {768, 5148, 4608, 2147483647, 1536, 7424}}},
        {4,
         {{3072, 26880, -1, 3072, -1, 768}, {-1, 22848, 2976, 4608, 1536, 2147483647},
          {23040, 2147483647, 4608, 7170, 1536, 2147483647}}},
        {8,
         {{1536, 26880, -1, 4608, 768, 1536}, {26880, 53340, -1, 3072, 768, 1536},
          {53340, 2147483647, -1, 3072, -1, 1536}, {-1, 2147483647, -1, 384, 3072, 10240},
          {3072, 2147483647, 384, 2976, 1536, 2147483647}, {22848, 2147483647, 2976, 4608, 1536, 2147483647}}},
        {2,
         {{26880, 53340, -1, 4608, -1, 768}, {-1, 3072, 384, 2976, 1536, 2147483647},
          {768, 5148, 4608, 2147483647, 7424, 2147483647}, {5148, 23040, 4608, 7170, 1536, 2147483647},
          {5148, 2147483647, 7170, 2147483647, 1536, 2147483647}}},
        {16,
         {{-1, 2147483647, -1, 384, 1536, 3072}, {-1, 2147483647, -1, 384, 10240, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankFP16UbmovenumMap = {
        {100,
         {{-1, 3072, -1, 2147483647, -1, 768}, {3072, 19680, -1, 3072, -1, 768},
          {-1, 3072, -1, 2147483647, 768, 1536}, {3072, 19680, -1, 3072, 768, 1536},
          {-1, 2147483647, 1792, 2976, 1536, 13312}}},
        {30,
         {{3072, 19680, 3072, 2147483647, -1, 768}, {19680, 2147483647, -1, 3072, -1, 1536},
          {-1, 2147483647, -1, 1792, 1536, 13312}, {-1, 768, 2976, 2147483647, 5376, 13312},
          {-1, 768, -1, 2147483647, 13312, 2147483647}, {26880, 2147483647, -1, 3072, 13312, 2147483647}}},
        {20,
         {{3072, 19680, 3072, 2147483647, 768, 1536}, {19680, 2147483647, 3072, 2147483647, -1, 1536},
          {-1, 2147483647, 2976, 2147483647, 1536, 5376}, {768, 2147483647, 2976, 2147483647, 5376, 13312},
          {768, 26880, -1, 2147483647, 13312, 2147483647}, {26880, 2147483647, 3072, 2147483647, 13312, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankFP16PvalueMap = {
        {4,
         {{-1, 768, -1, 2147483647, -1, 768}, {12980, 26880, -1, 3072, -1, 768},
          {-1, 15412, 2976, 4608, 1536, 2147483647}, {23040, 2147483647, 4608, 7170, 1536, 2147483647}}},
        {1,
         {{768, 12980, -1, 2147483647, -1, 768}, {12980, 26880, 3072, 2147483647, -1, 768},
          {-1, 1536, -1, 2147483647, 768, 1536}, {1536, 26880, 4608, 2147483647, 768, 1536},
          {26880, 53340, 4608, 2147483647, -1, 768}, {26880, 53340, 3072, 2147483647, 768, 1536},
          {53340, 2147483647, 3072, 2147483647, -1, 1536}, {-1, 768, 4608, 2147483647, 1536, 2147483647},
          {768, 5148, 4608, 2147483647, 1536, 7424}}},
        {8,
         {{1536, 26880, -1, 4608, 768, 1536}, {26880, 53340, -1, 3072, 768, 1536},
          {53340, 2147483647, -1, 3072, -1, 1536}, {-1, 2147483647, -1, 384, 3072, 10240},
          {3072, 2147483647, 384, 2976, 1536, 2147483647}, {15412, 2147483647, 2976, 4608, 1536, 2147483647}}},
        {2,
         {{26880, 53340, -1, 4608, -1, 768}, {-1, 3072, 384, 2976, 1536, 2147483647},
          {768, 5148, 4608, 2147483647, 7424, 2147483647}, {5148, 23040, 4608, 7170, 1536, 2147483647},
          {5148, 2147483647, 7170, 2147483647, 1536, 2147483647}}},
        {14,
         {{-1, 2147483647, -1, 384, 1536, 3072}, {-1, 2147483647, -1, 384, 10240, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankInT8M0Map = {
        {128,
         {{-1, 31220, -1, 2147483647, -1, 768}, {31220, 36980, 1280, 2147483647, -1, 768},
          {-1, 36980, -1, 2147483647, 768, 3072}, {36980, 2147483647, -1, 2147483647, -1, 3072},
          {-1, 2147483647, -1, 2147483647, 3072, 13312}, {-1, 1536, -1, 384, 13312, 2147483647},
          {5274, 2147483647, -1, 384, 13312, 2147483647}, {-1, 2147483647, 384, 2147483647, 13312, 2147483647}}},
        {256,
         {{31220, 36980, -1, 1280, -1, 768}, {1536, 5274, -1, 384, 13312, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankInT8DatasplitMap = {
        {1,
         {{-1, 3072, -1, 2147483647, -1, 768}, {3072, 5148, 4608, 2147483647, -1, 768},
          {-1, 1536, -1, 2147483647, 768, 1536}, {3072, 5148, -1, 2147483647, 768, 1536},
          {5148, 2147483647, 5634, 2147483647, -1, 1536}, {-1, 2147483647, 11264, 2147483647, 1536, 5376}}},
        {4,
         {{3072, 5148, -1, 4608, -1, 768}, {5148, 31220, -1, 3072, -1, 768},
          {5148, 2147483647, 3072, 4608, -1, 1536}, {-1, 2147483647, 5634, 11264, 1536, 5376},
          {34560, 2147483647, 5634, 2147483647, 5376, 7424}, {7196, 2147483647, 5634, 2147483647, 7424, 13312}}},
        {2,
         {{1536, 3072, -1, 2147483647, 768, 1536}, {5148, 2147483647, 4608, 5634, -1, 1536},
          {-1, 34560, 5634, 2147483647, 5376, 7424}, {-1, 3072, -1, 2147483647, 7424, 2147483647},
          {3072, 7196, 5634, 2147483647, 7424, 2147483647}}},
        {8,
         {{5148, 31220, -1, 3072, 768, 1536}, {31220, 2147483647, -1, 3072, -1, 1536},
          {-1, 2147483647, -1, 5634, 1536, 7424}, {3072, 7196, -1, 5634, 7424, 2147483647},
          {7196, 2147483647, -1, 5634, 7424, 13312}, {7196, 2147483647, -1, 2147483647, 13312, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankInT8PvalueMap = {
        {14,
         {{-1, 1536, -1, 2147483647, -1, 768}, {10010, 12980, -1, 2147483647, -1, 768},
          {-1, 7350, -1, 1536, 768, 1536}, {-1, 768, 1536, 2147483647, 768, 1536},
          {-1, 768, -1, 1536, 7424, 2147483647}}},
        {1,
         {{1536, 10010, -1, 2147483647, -1, 768}, {12980, 2147483647, 5634, 2147483647, -1, 1536},
          {-1, 2147483647, 11264, 2147483647, 1536, 5376}}},
        {10,
         {{7350, 12980, -1, 1536, 768, 1536}}},
        {2,
         {{768, 12980, 1536, 2147483647, 768, 1536}, {12980, 2147483647, 4608, 5634, -1, 1536},
          {-1, 34560, 5634, 2147483647, 5376, 7424}, {-1, 768, 1536, 2147483647, 7424, 2147483647},
          {768, 3072, -1, 2147483647, 7424, 19456}}},
        {4,
         {{12980, 36980, -1, 3072, -1, 768}, {12980, 2147483647, 3072, 4608, -1, 1536},
          {-1, 2147483647, 5634, 11264, 1536, 5376}, {34560, 2147483647, 5634, 2147483647, 5376, 7424},
          {768, 3072, -1, 2147483647, 19456, 2147483647}, {3072, 2147483647, 5634, 2147483647, 7424, 2147483647}}},
        {8,
         {{12980, 36980, -1, 3072, 768, 1536}, {36980, 2147483647, -1, 3072, -1, 1536},
          {-1, 2147483647, -1, 5634, 1536, 7424}, {3072, 2147483647, -1, 5634, 7424, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceEightRankInT8UbmovenumMap = {
        {80,
         {{-1, 7350, -1, 3072, -1, 768}}},
        {100,
         {{-1, 7350, 3072, 2147483647, -1, 768}, {-1, 7350, -1, 7170, 768, 3072},
          {-1, 7350, -1, 4608, 3072, 5376}, {7350, 2147483647, -1, 3072, -1, 5376},
          {-1, 768, -1, 2147483647, 5376, 10240}, {768, 1536, -1, 4608, 5376, 2147483647},
          {3072, 2147483647, -1, 2976, 5376, 2147483647}}},
        {30,
         {{-1, 7350, 7170, 2147483647, 768, 3072}, {-1, 3072, 7170, 2147483647, 3072, 5376},
          {7350, 23040, 3072, 2147483647, 768, 5376}, {23040, 2147483647, 3072, 2147483647, -1, 5376},
          {-1, 768, -1, 2147483647, 13312, 2147483647}, {768, 1536, 4608, 2147483647, 5376, 2147483647},
          {3072, 120832, 2976, 2147483647, 5376, 13312}, {3072, 2147483647, 2976, 4608, 13312, 2147483647}}},
        {50,
         {{-1, 7350, 4608, 7170, 3072, 5376}, {-1, 768, -1, 2147483647, 10240, 13312}}},
        {20,
         {{3072, 7350, 7170, 2147483647, 3072, 5376}, {1536, 3072, 7170, 2147483647, 5376, 2147483647},
          {120832, 2147483647, 2976, 2147483647, 5376, 13312},
          {3072, 2147483647, 4608, 2147483647, 13312, 2147483647}}},
        {40,
         {{7350, 23040, 3072, 2147483647, -1, 768}, {1536, 3072, -1, 7170, 5376, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceTwoRankFP16CommdatasplitMap = {
        {16,
         {{-1, 6656, -1, 2147483647, -1, 1536}, {6656, 2147483647, -1, 19456, -1, 1536},
          {7680, 2147483647, 19456, 2147483647, -1, 1536}, {-1, 2147483647, -1, 2147483647, 1536, 2147483647}}},
        {4,
         {{6656, 7680, 19456, 2147483647, -1, 1536}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceTwoRankFP16UbmovenumMap = {
        {2,
         {{-1, 1536, -1, 3072, -1, 1536}, {-1, 1536, 15360, 2147483647, -1, 1536},
          {1536, 6656, -1, 2147483647, -1, 1536}, {6656, 2147483647, -1, 19456, -1, 1536},
          {7680, 2147483647, 19456, 2147483647, -1, 1536}, {-1, 2147483647, -1, 2147483647, 1536, 2147483647}}},
        {3,
         {{-1, 1536, 3072, 15360, -1, 1536}}},
        {6,
         {{6656, 7680, 19456, 2147483647, -1, 1536}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceTwoRankFP16SwizzldirectMap = {
        {1,
         {{-1, 6656, -1, 2147483647, -1, 7680}, {6656, 35840, -1, 13312, -1, 7680},
          {35840, 2147483647, -1, 2147483647, -1, 7680}, {-1, 25600, -1, 2147483647, 7680, 2147483647},
          {25600, 2147483647, -1, 2147483647, 7680, 9216}, {25600, 2147483647, -1, 15360, 9216, 11264},
          {25600, 2147483647, -1, 2147483647, 11264, 2147483647}}},
        {0,
         {{6656, 35840, 13312, 2147483647, -1, 7680}, {25600, 2147483647, 15360, 2147483647, 9216, 11264}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceTwoRankFP16SwizzlcountMap = {
        {4,
         {{-1, 5632, -1, 2147483647, -1, 1536}, {5632, 7680, -1, 17408, -1, 1536},
          {7680, 9216, -1, 11264, -1, 1536}, {9216, 2147483647, -1, 19456, -1, 1536},
          {19456, 2147483647, 19456, 2147483647, -1, 1536}, {-1, 2147483647, -1, 11264, 1536, 13312},
          {-1, 2147483647, 11264, 15360, 4608, 13312}, {-1, 2147483647, 17408, 2147483647, 1536, 13312},
          {-1, 9216, -1, 15360, 13312, 2147483647}, {-1, 9216, 17408, 2147483647, 13312, 2147483647},
          {9216, 25600, -1, 11264, 13312, 2147483647}, {25600, 35840, -1, 13312, 13312, 2147483647},
          {35840, 2147483647, -1, 2147483647, 13312, 2147483647}}},
        {8,
         {{5632, 7680, 17408, 19456, -1, 1536}, {5632, 19456, 19456, 2147483647, -1, 1536},
          {-1, 2147483647, 11264, 17408, 1536, 4608}, {-1, 2147483647, 15360, 17408, 4608, 13312},
          {-1, 9216, 15360, 17408, 13312, 2147483647}, {9216, 25600, 11264, 2147483647, 13312, 2147483647}}},
        {16,
         {{7680, 9216, 11264, 19456, -1, 1536}, {25600, 35840, 13312, 2147483647, 13312, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceTwoRankFP16M0Map = {
        {128,
         {{-1, 6656, -1, 2147483647, -1, 7680}, {6656, 2147483647, -1, 13312, -1, 7680},
          {-1, 1536, -1, 7680, 7680, 11264}, {-1, 1536, -1, 6656, 11264, 2147483647},
          {1536, 2147483647, -1, 2147483647, 7680, 2147483647}}},
        {256,
         {{6656, 2147483647, 13312, 2147483647, -1, 7680}, {-1, 1536, 7680, 2147483647, 7680, 11264},
          {-1, 1536, 6656, 2147483647, 11264, 2147483647}}}
    };

    static std::map<int, std::vector<std::vector<int>>> g_allreduceTwoRankFP16PvalueMap = {
        {2,
         {{-1, 2560, -1, 3584, -1, 1536}, {4608, 7680, -1, 7680, -1, 1536},
          {7680, 9216, -1, 2147483647, -1, 1536}, {-1, 15360, 4608, 13312, 1536, 2560},
          {-1, 7680, -1, 13312, 2560, 3584}, {6656, 15360, 13312, 2147483647, 2560, 3584},
          {15360, 25600, 4608, 15360, -1, 2560}, {25600, 2147483647, 19456, 2147483647, -1, 2560},
          {15360, 25600, 11264, 2147483647, 2560, 3584}, {-1, 15360, 9216, 17408, 3584, 9216},
          {-1, 6656, 13312, 2147483647, 11264, 2147483647}, {15360, 35840, 13312, 2147483647, 11264, 2147483647}}},
        {1,
         {{-1, 2560, 3584, 2147483647, -1, 1536}, {2560, 4608, -1, 2147483647, -1, 1536},
          {4608, 7680, 7680, 2147483647, -1, 1536}, {9216, 15360, -1, 2147483647, -1, 1536},
          {-1, 15360, 13312, 2147483647, 1536, 2560}, {-1, 6656, 13312, 2147483647, 2560, 3584},
          {15360, 25600, 15360, 2147483647, -1, 2560}, {-1, 15360, 17408, 2147483647, 3584, 9216},
          {-1, 6656, 13312, 2147483647, 9216, 11264}, {9216, 15360, 13312, 2147483647, 9216, 2147483647}}},
        {3,
         {{-1, 15360, -1, 4608, 1536, 2560}, {7680, 15360, -1, 13312, 2560, 3584},
          {15360, 2147483647, 2560, 4608, -1, 1536}, {25600, 2147483647, 4608, 19456, -1, 2560},
          {15360, 25600, 4608, 11264, 2560, 3584}, {-1, 15360, 1536, 9216, 3584, 9216},
          {-1, 6656, -1, 13312, 9216, 2147483647}, {15360, 25600, 11264, 2147483647, 3584, 7680}}},
        {4,
         {{15360, 30720, -1, 1536, -1, 3584}, {15360, 2147483647, 1536, 2560, -1, 2560},
          {15360, 2147483647, 2560, 4608, 1536, 3584}, {25600, 2147483647, 4608, 2147483647, 2560, 3584},
          {-1, 15360, -1, 1536, 3584, 9216}, {6656, 9216, -1, 2147483647, 9216, 2147483647},
          {9216, 15360, -1, 13312, 9216, 2147483647}, {15360, 25600, -1, 11264, 3584, 7680},
          {25600, 35840, -1, 2147483647, 3584, 6656}, {15360, 35840, 5632, 2147483647, 7680, 11264}}},
        {6,
         {{30720, 2147483647, -1, 1536, -1, 3584}, {15360, 2147483647, 1536, 2560, 2560, 3584},
          {25600, 35840, -1, 2147483647, 6656, 7680}, {15360, 35840, -1, 5632, 7680, 11264},
          {15360, 35840, -1, 13312, 11264, 2147483647}, {35840, 2147483647, -1, 2147483647, 3584, 2147483647}}}
    };

    int32_t AllReduceUbMoveNum(int m, int k, int n)
    {
        double commPredict = 1.0 * (m / ONE_K) * (n / ONE_K) * (SECOND_TO_MS / ONE_K) / 40;
        double cubePredict = DOUBLE * m * k / B1_FLOP_PER_MS * n;
        double mknGB = (m / ONE_K) * (k / ONE_K) * (n / ONE_K);
        double mteTimePredict1 = GetMTETime(mknGB, DEFAULT_ROW, DEFAULT_COL);
        double mteTimePredict2 = GetMTETime(mknGB, DEFAULT_COL, DEFAULT_ROW);
        double mteTimePredict = std::min(mteTimePredict1, mteTimePredict2);
        double matmulPredict = std::max(cubePredict, mteTimePredict);
        double c0 = matmulPredict / commPredict;
        double c1 = 1.0 * m * n / k;
        double c2 = sqrt(c1);
        double c3 = sqrt(1.0 * m * n) / k;
        double c4 = c3 * c3;
        double c5 = matmulPredict;
        double c6 = commPredict;
        double c7 = 1.0 * n / m;
        double c8 = 1.0 * m * n / sqrt(k);
        double c9 = 1.0 * m * n * sqrt(k);
        double c10 = sqrt(1.0 * m * n) * k;
        double c11 = sqrt(1.0 * m * n * k);
        double c12 = sqrt(1.0 * m * n);
        double c13 = 1.0 * k * k / sqrt(1.0 * m * n);
        double c14 = 1.0 * k * k * sqrt(1.0 * m * n);
        double ubMoveNumDouble = 0;
        std::vector<double> featsUpdate = { c0, c1, c2, c3, c4, c5, c6, c7, 1.0 / c0, 1.0 / c1, 1.0 / c2, 1.0 / c3,
                                            1.0 / c4, c8, c9, c10, c11, c12, c13, 1.0 / c13, c14, 1 };
        for (uint32_t i = 0; i < featsUpdate.size(); i++) {
            ubMoveNumDouble += featsUpdate[i] * g_allreduceUbmovenumCoef[i];
        }

        return std::min(std::max(static_cast<int32_t>(ubMoveNumDouble) * HALF_KBYTE, MIN_UB_MOVE_NUM), MAX_UB_NUM);
    }

    int32_t AllReducePValue(int m, int k, int n)
    {
        double commPredict = 1.0 * (m / ONE_K) * (n / ONE_K) * (SECOND_TO_MS / ONE_K) / 40;
        double cubePredict = DOUBLE * m * k / B1_FLOP_PER_MS * n;
        double mknGB = (m / ONE_K) * (k / ONE_K) * (n / ONE_K);
        double mteTimePredict1 = GetMTETime(mknGB, DEFAULT_ROW, DEFAULT_COL);
        double mteTimePredict2 = GetMTETime(mknGB, DEFAULT_COL, DEFAULT_ROW);
        double mteTimePredict = std::min(mteTimePredict1, mteTimePredict2);
        double matmulPredict = std::max(cubePredict, mteTimePredict);
        double c0 = matmulPredict / commPredict;
        double c1 = 1.0 * m * n / k;
        double c2 = sqrt(c1);
        double c3 = sqrt(1.0 * m * n) / k;
        double c4 = c3 * c3;
        double c5 = matmulPredict;
        double c6 = commPredict;
        double c7 = 1.0 * n / m;
        double c8 = 1.0 * m * n / sqrt(k);
        double c9 = 1.0 * m * n * sqrt(k);
        double c10 = sqrt(1.0 * m * n) * k;
        double c11 = sqrt(1.0 * m * n * k);
        double c12 = sqrt(1.0 * m * n);
        double c13 = 1.0 * k * k / sqrt(1.0 * m * n);
        double c14 = 1.0 * k * k * sqrt(1.0 * m * n);
        double pValueDouble = 0;
        std::vector<double> featsUpdate = { c0, c1, c2, c3, c4, c5, c6, c7, 1.0 / c0, 1.0 / c1, 1.0 / c2, 1.0 / c3,
                                            1.0 / c4, c8, c9, c10, c11, c12, c13, 1.0 / c13, c14, 1 };
        for (uint32_t i = 0; i < featsUpdate.size(); i++) {
            pValueDouble += featsUpdate[i] * g_allreducePvalueCoef[i];
        }

        return std::min(std::max(static_cast<int32_t>(pValueDouble), 1), MAX_P_VALUE);
    }

    void AllReduceSetWithSerialMode(CoCTilingData &cocTilingData)
    {
        int32_t m = cocTilingData.m;
        int32_t k = cocTilingData.k;
        int32_t n = cocTilingData.n;

        int64_t batchSize = cocTilingData.batchSize;
        int64_t commSize = static_cast<int64_t>(batchSize) * m * n;
        if (commSize <= ALLREDUCE_SERIAL_MODE_MN_SIZE && k <= ALLREDUCE_SERIAL_MODE_K_SIZE) {
            cocTilingData.withSerialMode = 1;
            cocTilingData.ubMoveNum = MAX_UB_NUM;
            cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
        } else {
            cocTilingData.withSerialMode = 0;
        }
    }

    void AllReduceGetDefaultTiling(CoCTilingData &cocTilingData)
    {
        int64_t batchSize = cocTilingData.batchSize;
        int32_t m = cocTilingData.m;
        int32_t k = cocTilingData.k;
        int32_t n = cocTilingData.n;

        cocTilingData.swizzlDirect = SWIZZLE_DIRECT_ONE;
        cocTilingData.ubMoveNum = AllReduceUbMoveNum(m, k, n);
        cocTilingData.pValue = AllReducePValue(m, k, n);

        int64_t cubeSize = static_cast<int64_t>(batchSize) * m * k * n;
        int64_t commSize = static_cast<int64_t>(batchSize) * m * n;
        constexpr int32_t bufferUnit = 1024;
        if ((cubeSize <= MATMUL_BASE_100US &&
            commSize < (cocTilingData.bufferSize * bufferUnit * bufferUnit) / INPUT_DTYPE / MAX_BLOCK_COUNT) ||
            commSize <= ALLREDUCE_BASE_100US) {
            cocTilingData.withSerialMode = 1;
            cocTilingData.ubMoveNum = MAX_UB_NUM;
        } else {
            cocTilingData.withSerialMode = 0;
        }
        cocTilingData.commDirect = COMM_DATA_DIRECT;
        cocTilingData.commNpuSplit = cocTilingData.rankSize;
        cocTilingData.commDataSplit = COMMDATASPLIT_ONE;
        cocTilingData.lenPerLoop = cocTilingData.m0 * cocTilingData.n0 * cocTilingData.pValue * cocTilingData.blockDim;
        cocTilingData.lenPerLoop = cocTilingData.lenPerLoop / cocTilingData.rankSize;
        cocTilingData.lenPerLoop = RoundNum(cocTilingData.lenPerLoop, HALF_KBYTE);
        SetSecondCoreSplitTling(cocTilingData);
    }

    void AllReduceFourRankInt8GetDefaultTiling(CoCTilingData &cocTilingData)
    {
        std::map<int*, TilingValue> tilingParamMap = {
            {&cocTilingData.m0,
             {ALLREDUCE_FOUR_RANK_INT8_M0_DEFAULT,
              g_allreduceFourRankInT8M0Map}},
            {&cocTilingData.ubMoveNum,
             {ALLREDUCE_FOUR_RANK_INT8_UBMOVENUM_DEFAULT,
              g_allreduceFourRankInT8UbmovenumMap}},
            {&cocTilingData.pValue,
             {ALLREDUCE_FOUR_RANK_INT8_PVALUE_DEFAULT,
              g_allreduceFourRankInT8PvalueMap}},
            {&cocTilingData.swizzlDirect, {SWIZZLE_DIRECT_ONE}},
            {&cocTilingData.swizzlCount, {DEFAULT_SWIZZLE_COUNT}},
            {&cocTilingData.commDirect, {COMM_DATA_DIRECT}},
            {&cocTilingData.commNpuSplit, {COMMNPUSPLIT_ONE}},
            {&cocTilingData.commDataSplit, {COMMDATASPLIT_SIXTEEN}}
        };
        SetTilingParam(cocTilingData, tilingParamMap);

        cocTilingData.lenPerLoop = ALLREDUCE_LENPERLOOP_DEFAULT / RANKSIZE_EIGHT * cocTilingData.rankSize;
        cocTilingData.lenPerLoop = RoundNum(cocTilingData.lenPerLoop, HALF_KBYTE);
        cocTilingData.ubMoveNum = cocTilingData.lenPerLoop;

        AllReduceSetWithSerialMode(cocTilingData);
        SetSecondCoreSplitTling(cocTilingData);
    }

    void AllReduceFourRankFP16GetDefaultTiling(CoCTilingData &cocTilingData)
    {
        std::map<int*, TilingValue> tilingParamMap = {
            {&cocTilingData.m0,
             {ALLREDUCE_FOUR_RANK_FP16_M0_DEFAULT,
              g_allreduceFourRankFP16M0Map}},
            {&cocTilingData.ubMoveNum,
             {ALLREDUCE_FOUR_RANK_FP16_UBMOVENUM_DEFAULT,
              g_allreduceFourRankFP16UbmovenumMap}},
            {&cocTilingData.pValue,
             {ALLREDUCE_FOUR_RANK_FP16_PVALUE_DEFAULT,
              g_allreduceFourRankFP16PvalueMap}},
            {&cocTilingData.swizzlDirect, {SWIZZLE_DIRECT_ONE}},
            {&cocTilingData.swizzlCount, {DEFAULT_SWIZZLE_COUNT}},
            {&cocTilingData.commDirect, {COMM_DATA_DIRECT}},
            {&cocTilingData.commNpuSplit, {COMMNPUSPLIT_ONE}},
            {&cocTilingData.commDataSplit, {COMMDATASPLIT_SIXTEEN}}
        };
        SetTilingParam(cocTilingData, tilingParamMap);

        cocTilingData.lenPerLoop = ALLREDUCE_LENPERLOOP_DEFAULT / RANKSIZE_EIGHT * cocTilingData.rankSize;
        cocTilingData.lenPerLoop = RoundNum(cocTilingData.lenPerLoop, HALF_KBYTE);
        cocTilingData.ubMoveNum = cocTilingData.lenPerLoop;

        AllReduceSetWithSerialMode(cocTilingData);
        SetSecondCoreSplitTling(cocTilingData);
    }

    void AllReduceEightRankFP16GetDefaultTiling(CoCTilingData &cocTilingData)
    {
        int dataSplit = 0;
        std::map<int*, TilingValue> tilingParamMap = {
            {&cocTilingData.m0,
             {ALLREDUCE_EIGHT_RANK_FP16_M0_DEFAULT,
              g_allreduceEightRankFP16M0Map}},
            {&cocTilingData.ubMoveNum,
             {ALLREDUCE_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT,
              g_allreduceEightRankFP16UbmovenumMap}},
            {&cocTilingData.pValue,
             {ALLREDUCE_EIGHT_RANK_FP16_PVALUE_DEFAULT,
              g_allreduceEightRankFP16PvalueMap}},
            {&dataSplit,
             {ALLREDUCE_EIGHT_RANK_FP16_DATASPLIT_DEFAULT,
              g_allreduceEightRankFP16DatasplitMap}},
            {&cocTilingData.swizzlDirect, {SWIZZLE_DIRECT_ONE}},
            {&cocTilingData.swizzlCount, {DEFAULT_SWIZZLE_COUNT}},
            {&cocTilingData.commDirect, {COMM_DATA_DIRECT}},
            {&cocTilingData.commNpuSplit, {COMMNPUSPLIT_ONE}},
            {&cocTilingData.commDataSplit, {COMMDATASPLIT_SIXTEEN}}
        };
        SetTilingParam(cocTilingData, tilingParamMap);

        cocTilingData.lenPerLoop = cocTilingData.m0 * cocTilingData.n0 * cocTilingData.pValue * cocTilingData.blockDim;
        cocTilingData.lenPerLoop = cocTilingData.lenPerLoop / cocTilingData.rankSize / cocTilingData.commDataSplit;
        cocTilingData.lenPerLoop = cocTilingData.lenPerLoop / dataSplit;
        cocTilingData.lenPerLoop = RoundNum(cocTilingData.lenPerLoop, HALF_KBYTE);
        cocTilingData.lenPerLoop = std::min(cocTilingData.lenPerLoop, TREE_LEN_PER_LOOP);

        AllReduceSetWithSerialMode(cocTilingData);
        SetSecondCoreSplitTling(cocTilingData);
    }

    void AllReduceEightRankINT8GetDefaultTiling(CoCTilingData &cocTilingData)
    {
        int dataSplit = 0;
        std::map<int*, TilingValue> tilingParamMap = {
            {&cocTilingData.m0,
             {ALLREDUCE_EIGHT_RANK_INT8_M0_DEFAULT,
              g_allreduceEightRankInT8M0Map}},
            {&cocTilingData.ubMoveNum,
             {ALLREDUCE_EIGHT_RANK_INT8_UBMOVENUM_DEFAULT,
              g_allreduceEightRankInT8UbmovenumMap}},
            {&cocTilingData.pValue,
             {ALLREDUCE_EIGHT_RANK_INT8_PVALUE_DEFAULT,
              g_allreduceEightRankInT8PvalueMap}},
            {&dataSplit,
             {ALLREDUCE_EIGHT_RANK_INT8_DATASPLIT_DEFAULT,
              g_allreduceEightRankInT8DatasplitMap}},
            {&cocTilingData.swizzlDirect, {SWIZZLE_DIRECT_ONE}},
            {&cocTilingData.swizzlCount, {DEFAULT_SWIZZLE_COUNT}},
            {&cocTilingData.commDirect, {COMM_DATA_DIRECT}},
            {&cocTilingData.commNpuSplit, {COMMNPUSPLIT_ONE}},
            {&cocTilingData.commDataSplit, {COMMDATASPLIT_SIXTEEN}}
        };
        SetTilingParam(cocTilingData, tilingParamMap);

        cocTilingData.lenPerLoop = cocTilingData.m0 * cocTilingData.n0 * cocTilingData.pValue * cocTilingData.blockDim;
        cocTilingData.lenPerLoop = cocTilingData.lenPerLoop / cocTilingData.rankSize / cocTilingData.commDataSplit;
        cocTilingData.lenPerLoop = cocTilingData.lenPerLoop / dataSplit;
        cocTilingData.lenPerLoop = RoundNum(cocTilingData.lenPerLoop, HALF_KBYTE);
        cocTilingData.lenPerLoop = std::min(cocTilingData.lenPerLoop, TREE_LEN_PER_LOOP);

        AllReduceSetWithSerialMode(cocTilingData);
        SetSecondCoreSplitTling(cocTilingData);
    }

    void AllReduceTwoRankFP16Tiling(CoCTilingData &cocTilingData)
    {
        std::map<int*, TilingValue> tilingParamMap = {
            {&cocTilingData.commDataSplit,
             {ALLREDUCE_TWO_RANK_FP16_COMMDATASPLIT_DEFAULT,
              g_allreduceTwoRankFP16CommdatasplitMap}},
            {&cocTilingData.ubMoveNum,
             {ALLREDUCE_TWO_RANK_FP16_UBMOVENUM_DEFAULT,
              g_allreduceTwoRankFP16UbmovenumMap}},
            {&cocTilingData.swizzlDirect,
             {ALLREDUCE_TWO_RANK_FP16_SWIZZLDIRECT_DEFAULT,
              g_allreduceTwoRankFP16SwizzldirectMap}},
            {&cocTilingData.swizzlCount,
             {ALLREDUCE_TWO_RANK_FP16_SWIZZLCOUNT_DEFAULT,
              g_allreduceTwoRankFP16SwizzlcountMap}},
            {&cocTilingData.m0,
             {ALLREDUCE_TWO_RANK_FP16_M0_DEFAULT,
              g_allreduceTwoRankFP16M0Map}},
            {&cocTilingData.pValue,
             {ALLREDUCE_TWO_RANK_FP16_PVALUE_DEFAULT,
              g_allreduceTwoRankFP16PvalueMap}},
            {&cocTilingData.commDirect, {COMM_DATA_DIRECT}},
            {&cocTilingData.commNpuSplit, {COMMNPUSPLIT_ONE}}
        };
        SetTilingParam(cocTilingData, tilingParamMap);

        cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
        AllReduceSetWithSerialMode(cocTilingData);
    }
}