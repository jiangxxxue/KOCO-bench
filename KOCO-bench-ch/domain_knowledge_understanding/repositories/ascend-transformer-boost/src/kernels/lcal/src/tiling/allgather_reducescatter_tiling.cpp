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
#include <iostream>
#include "tiling.h"
#include "tiling_func.h"
#include "lcoc_func.h"

#define TILING_MAP std::map<int, std::vector<std::vector<int>>>
namespace Lcal {
constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_SWIZZLECOUNT_DEFAULT = 11;
static TILING_MAP g_allgatherEightReducescatterTwoFalseFP16SwizzlecountMap = {
    {9,
        {{768, 1536, -1, 2147483647, -1, 7168},
         {1536, 3072, -1, 5120, -1, 14848},
         {1536, 2147483647, 5120, 2147483647, -1, 10752}}},
    {14, {{768, 1536, -1, 5120, 10752, 2147483647}, {1536, 2147483647, -1, 5120, 14848, 2147483647}}}};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_UBMOVENUM_DEFAULT = 40;
static TILING_MAP g_allgatherEightReducescatterTwoFalseFP16UbmovenumMap = {
    {24,
        {{768, 1536, -1, 2147483647, 3072, 10752},
         {1536, 3072, -1, 7168, 3072, 2147483647},
         {3072, 2147483647, -1, 7168, 3072, 2147483647}}},
    {30, {{3072, 2147483647, 7168, 2147483647, 3072, 2147483647}}}};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_LENPERLOOPMULT_DEFAULT = 400;
static TILING_MAP g_allgatherEightReducescatterTwoFalseFP16LenperloopmultMap = {
    {2, {{768, 1536, -1, 5120, -1, 3072}}},
    {4, {{3072, 2147483647, -1, 3072, -1, 2147483647}, {3072, 2147483647, 14848, 2147483647, 3072, 2147483647}}}};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_COMMNPUSPLIT_DEFAULT = 8;
static TILING_MAP g_allgatherEightReducescatterTwoFalseFP16CommnpusplitMap = {
    {1,
        {{768, 1536, 5120, 2147483647, -1, 3072},
         {1536, 3072, 14848, 2147483647, -1, 7168},
         {3072, 2147483647, 14848, 2147483647, -1, 2147483647}}}};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_COMMDATASPLIT_DEFAULT = 1;
static TILING_MAP g_allgatherEightReducescatterTwoFalseFP16CommdatasplitMap = {
    {8,
        {{768, 1536, 5120, 2147483647, -1, 3072},
         {1536, 3072, 14848, 2147483647, -1, 7168},
         {3072, 2147483647, 14848, 2147483647, -1, 2147483647}}}};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_EXTRAUBMOVENUM_DEFAULT = 12;
static TILING_MAP g_allgatherEightReducescatterTwoFalseFP16ExtraubmovenumMap = {
    {10,
        {{-1, 768, -1, 2147483647, 5120, 10752},
         {768, 1536, -1, 2147483647, 5120, 2147483647},
         {1536, 2147483647, -1, 10752, 5120, 2147483647},
         {1536, 2147483647, 10752, 14848, -1, 10752}}},
    {20, {{1536, 2147483647, 14848, 2147483647, 10752, 2147483647}}}};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_EXTRALENPERLOOPMULT_DEFAULT = 4;
static TILING_MAP g_allgatherEightReducescatterTwoFalseFP16ExtralenperloopmultMap = {
    {1, {{3072, 2147483647, -1, 3072, 10752, 2147483647}, {1536, 2147483647, 3072, 7168, -1, 2147483647}}},
    {2, {{768, 1536, 5120, 2147483647, 5120, 2147483647}, {1536, 2147483647, 7168, 10752, -1, 2147483647}}},
    {400, {{3072, 2147483647, 10752, 2147483647, -1, 2147483647}}}};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_EXTRACOMMNPUSPLIT_DEFAULT = 1;

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_EXTRACOMMDATASPLIT_DEFAULT = 8;

// 821
constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_SWIZZLECOUNT_DEFAULT = 5;
static TILING_MAP g_allgatherEightReducescatterTwoTrueFP16SwizzlecountMap = {
    {9, {{192, 768, -1, 2147483647, -1, 12800}}},
    {17, {{768, 2147483647, 7168, 2147483647, -1, 7936}}},
    {13,
        {{-1, 192, 5120, 2147483647, -1, 12800},
         {-1, 192, -1, 2147483647, 15360, 2147483647},
         {768, 2147483647, -1, 7168, -1, 9088}}}};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_UBMOVENUM_DEFAULT = 60;
static TILING_MAP g_allgatherEightReducescatterTwoTrueFP16UbmovenumMap = {
    {30, {{384, 2147483647, -1, 5120, 3968, 2147483647}, {768, 2147483647, 7168, 2147483647, -1, 2147483647}}}};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_LENPERLOOPMULT_DEFAULT = 400;

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_COMMNPUSPLIT_DEFAULT = 8;
static TILING_MAP g_allgatherEightReducescatterTwoTrueFP16CommnpusplitMap = {
    {2, {{-1, 192, 7168, 2147483647, -1, 4608}, {192, 2147483647, 5120, 7168, -1, 4544}}}};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_COMMDATASPLIT_DEFAULT = 1;
static TILING_MAP g_allgatherEightReducescatterTwoTrueFP16CommdatasplitMap = {
    {4, {{-1, 192, 7168, 2147483647, -1, 4608}, {192, 2147483647, 5120, 7168, -1, 4544}}}};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_COMMDIRECT_DEFAULT = 1;

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_EXTRAUBMOVENUM_DEFAULT = 20;
static TILING_MAP g_allgatherEightReducescatterTwoTrueFP16ExtraubmovenumMap = {
    {60, {{-1, 192, -1, 5120, -1, 6912}, {-1, 192, -1, 2147483647, 10368, 2147483647}}},
    {40,
        {{-1, 192, 5120, 2147483647, -1, 6912},
         {-1, 192, -1, 2147483647, 6912, 10368},
         {192, 384, -1, 2147483647, 1600, 4608}}},
    {30, {{192, 384, -1, 2147483647, 4608, 2147483647}, {768, 2147483647, -1, 5120, -1, 3968}}}};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_EXTRALENPERLOOPMULT_DEFAULT = 2;
static TILING_MAP g_allgatherEightReducescatterTwoTrueFP16ExtralenperloopmultMap = {
    {4, {{384, 2147483647, -1, 5120, -1, 3968}, {384, 2147483647, 7168, 2147483647, -1, 2147483647}}}};

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_EXTRACOMMNPUSPLIT_DEFAULT = 1;

constexpr int32_t ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_EXTRACOMMDATASPLIT_DEFAULT = 8;

// 281
constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_SWIZZLECOUNT_DEFAULT = 11;
static TILING_MAP g_allgatherTwoReducescatterEightTrueFP16SwizzlecountMap = {
    {9,
        {{3072, 6144, -1, 2147483647, -1, 10752},
         {12288, 2147483647, -1, 7168, -1, 10752},
         {12288, 2147483647, 10752, 2147483647, -1, 5120}}},
    {14,
        {{-1, 3072, -1, 7168, -1, 14848},
         {-1, 3072, -1, 10752, 14848, 2147483647},
         {12288, 2147483647, 7168, 10752, -1, 5120}}}};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_UBMOVENUM_DEFAULT = 10;
static TILING_MAP g_allgatherTwoReducescatterEightTrueFP16UbmovenumMap = {
    {14, {{3072, 6144, 14848, 2147483647, -1, 3072}, {6144, 12288, 14848, 2147483647, 3072, 2147483647}}},
    {24, {{12288, 2147483647, 10752, 14848, -1, 2147483647}}},
    {32, {{-1, 6144, -1, 2147483647, 10752, 2147483647}, {12288, 2147483647, -1, 10752, -1, 2147483647}}},
    {40, {{3072, 6144, -1, 2147483647, 10752, 2147483647}, {6144, 12288, -1, 14848, 3072, 2147483647}}}};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_LENPERLOOPMULT_DEFAULT = 400;
static TILING_MAP g_allgatherTwoReducescatterEightTrueFP16LenperloopmultMap = {
    {4,
        {{3072, 6144, -1, 2147483647, 3072, 10752},
         {12288, 2147483647, -1, 10752, -1, 3072},
         {6144, 2147483647, -1, 2147483647, 3072, 2147483647}}}};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_COMMNPUSPLIT_DEFAULT = 1;

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_COMMDATASPLIT_DEFAULT = 8;

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_EXTRAUBMOVENUM_DEFAULT = 20;
static TILING_MAP g_allgatherTwoReducescatterEightTrueFP16ExtraubmovenumMap = {
    {8, {{6144, 2147483647, 3072, 5120, 7168, 2147483647}}},
    {10, {{6144, 2147483647, 3072, 5120, -1, 7168}, {6144, 2147483647, 5120, 14848, -1, 2147483647}}},
    {12, {{3072, 6144, 3072, 2147483647, 14848, 2147483647}}},
    {15, {{-1, 3072, 3072, 2147483647, -1, 10752}, {3072, 6144, -1, 2147483647, -1, 5120}}}};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_EXTRALENPERLOOPMULT_DEFAULT = 2;
static TILING_MAP g_allgatherTwoReducescatterEightTrueFP16ExtralenperloopmultMap = {
    {4,
        {{-1, 3072, -1, 10752, 14848, 2147483647},
         {12288, 2147483647, 3072, 5120, -1, 2147483647},
         {6144, 2147483647, 5120, 7168, -1, 2147483647}}}};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_EXTRACOMMNPUSPLIT_DEFAULT = 8;
static TILING_MAP g_allgatherTwoReducescatterEightTrueFP16ExtracommnpusplitMap = {
    {1, {{12288, 2147483647, 14848, 2147483647, -1, 3072}, {12288, 2147483647, 14848, 2147483647, 5120, 2147483647}}}};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_EXTRACOMMDATASPLIT_DEFAULT = 2;
static TILING_MAP g_allgatherTwoReducescatterEightTrueFP16ExtracommdatasplitMap = {
    {8, {{12288, 2147483647, 14848, 2147483647, -1, 3072}, {12288, 2147483647, 14848, 2147483647, 5120, 2147483647}}}};

// 280
constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_SWIZZLECOUNT_DEFAULT = 9;
static TILING_MAP g_allgatherTwoReducescatterEightFalseFP16SwizzlecountMap = {
    {13,
        {{-1, 768, 1280, 2147483647, -1, 7168},
         {1536, 3072, -1, 2147483647, -1, 7168},
         {3072, 2147483647, 5184, 2147483647, -1, 2147483647}}},
    {17,
        {{-1, 768, -1, 2147483647, 7168, 2147483647},
         {3072, 2147483647, -1, 4544, 7168, 2147483647},
         {3072, 2147483647, 4544, 5184, -1, 2147483647}}},
    {5,
        {{768, 1536, -1, 2147483647, 5120, 2147483647},
         {3072, 2147483647, -1, 4544, -1, 7168},
         {3072, 2147483647, 7680, 2147483647, -1, 2147483647}}}};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_UBMOVENUM_DEFAULT = 40;
static TILING_MAP g_allgatherTwoReducescatterEightFalseFP16UbmovenumMap = {
    {30,
        {{-1, 768, 2176, 3840, -1, 5120}, {-1, 768, 2560, 2147483647, 5120, 7168},
         {-1, 768, -1, 7680, 7168, 2147483647}, {1536, 3072, -1, 6400, -1, 2147483647}}},
    {60, {{-1, 768, 7680, 2147483647, 7168, 2147483647}, {768, 1536, -1, 1280, -1, 7168}}},
    {20, {{768, 1536, -1, 4352, 7168, 2147483647}, {3072, 2147483647, -1, 6400, -1, 2147483647}}}};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_LENPERLOOPMULT_DEFAULT = 400;

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_COMMNPUSPLIT_DEFAULT = 1;

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_COMMDATASPLIT_DEFAULT = 8;

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_COMMDIRECT_DEFAULT = 0;
static TILING_MAP g_allgatherTwoReducescatterEightFalseFP16CommdirectMap = {
    {1,
        {{-1, 768, 3456, 2147483647, -1, 5120},
         {-1, 768, 2560, 2147483647, 5120, 7168},
         {-1, 768, 4352, 7680, 7168, 2147483647},
         {768, 1536, -1, 2147483647, -1, 7168},
         {1536, 3072, 1280, 2147483647, -1, 2147483647},
         {3072, 2147483647, -1, 7680, 5120, 2147483647}}}};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_EXTRAUBMOVENUM_DEFAULT = 60;
static TILING_MAP g_allgatherTwoReducescatterEightFalseFP16ExtraubmovenumMap = {
    {40, {{768, 2147483647, -1, 2176, -1, 5120}}},
    {30,
        {{768, 1536, 2176, 2147483647, -1, 5120},
         {768, 1536, -1, 2147483647, 5120, 2147483647},
         {1536, 2147483647, -1, 1792, 5120, 2147483647}}},
    {20, {{1536, 2147483647, 2176, 2147483647, -1, 5120}, {1536, 2147483647, 1792, 2147483647, 5120, 2147483647}}}};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_EXTRALENPERLOOPMULT_DEFAULT = 2;

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_EXTRACOMMNPUSPLIT_DEFAULT = 8;
static TILING_MAP g_allgatherTwoReducescatterEightFalseFP16ExtracommnpusplitMap = {
    {1, {{3072, 2147483647, 2176, 2147483647, -1, 5120}, {768, 2147483647, -1, 2147483647, 5120, 2147483647}}}};

constexpr int32_t ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_EXTRACOMMDATASPLIT_DEFAULT = 1;
static TILING_MAP g_allgatherTwoReducescatterEightFalseFP16ExtracommdatasplitMap = {
    {8, {{3072, 2147483647, 2176, 2147483647, -1, 5120}, {768, 2147483647, -1, 2147483647, 5120, 2147483647}}}};

const int PVALUE_ONE = 1;
const int M0_DEFAULT = 128;
const int K0_DEFAULT = 256;
const int N0_DEFAULT = 256;
const int SWIZZLEDIRECT_ONE = 1;

void AG8RS2FalseFP16Tiling(CoCTilingData &cocTilingData)
{
    std::map<int *, TilingValue> tilingParamMap = {
        {&cocTilingData.swizzlCount,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_SWIZZLECOUNT_DEFAULT,
             g_allgatherEightReducescatterTwoFalseFP16SwizzlecountMap}},
        {&cocTilingData.ubMoveNum,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_UBMOVENUM_DEFAULT,
             g_allgatherEightReducescatterTwoFalseFP16UbmovenumMap}},
        {&cocTilingData.lenPerLoop,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_LENPERLOOPMULT_DEFAULT,
             g_allgatherEightReducescatterTwoFalseFP16LenperloopmultMap}},
        {&cocTilingData.commNpuSplit,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_COMMNPUSPLIT_DEFAULT,
             g_allgatherEightReducescatterTwoFalseFP16CommnpusplitMap}},
        {&cocTilingData.commDataSplit,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_COMMDATASPLIT_DEFAULT,
             g_allgatherEightReducescatterTwoFalseFP16CommdatasplitMap}},
        {&cocTilingData.extraUbMoveNum,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_EXTRAUBMOVENUM_DEFAULT,
             g_allgatherEightReducescatterTwoFalseFP16ExtraubmovenumMap}},
        {&cocTilingData.extraLenPerLoop,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_EXTRALENPERLOOPMULT_DEFAULT,
             g_allgatherEightReducescatterTwoFalseFP16ExtralenperloopmultMap}},
        {&cocTilingData.extraCommNpuSplit, {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_EXTRACOMMNPUSPLIT_DEFAULT}},
        {&cocTilingData.extraCommDataSplit,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_FALSE_FP16_EXTRACOMMDATASPLIT_DEFAULT}}};
    SetTilingParam2D(cocTilingData, tilingParamMap);
    return;
}

void AG8RS2TrueFP16Tiling(CoCTilingData &cocTilingData)
{
    std::map<int *, TilingValue> tilingParamMap = {
        {&cocTilingData.swizzlCount,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_SWIZZLECOUNT_DEFAULT,
             g_allgatherEightReducescatterTwoTrueFP16SwizzlecountMap}},
        {&cocTilingData.ubMoveNum,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_UBMOVENUM_DEFAULT,
             g_allgatherEightReducescatterTwoTrueFP16UbmovenumMap}},
        {&cocTilingData.lenPerLoop,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_LENPERLOOPMULT_DEFAULT}},
        {&cocTilingData.commNpuSplit,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_COMMNPUSPLIT_DEFAULT,
             g_allgatherEightReducescatterTwoTrueFP16CommnpusplitMap}},
        {&cocTilingData.commDataSplit,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_COMMDATASPLIT_DEFAULT,
             g_allgatherEightReducescatterTwoTrueFP16CommdatasplitMap}},
        {&cocTilingData.commDirect, {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_COMMDIRECT_DEFAULT}},
        {&cocTilingData.extraUbMoveNum,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_EXTRAUBMOVENUM_DEFAULT,
             g_allgatherEightReducescatterTwoTrueFP16ExtraubmovenumMap}},
        {&cocTilingData.extraLenPerLoop,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_EXTRALENPERLOOPMULT_DEFAULT,
             g_allgatherEightReducescatterTwoTrueFP16ExtralenperloopmultMap}},
        {&cocTilingData.extraCommNpuSplit, {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_EXTRACOMMNPUSPLIT_DEFAULT}},
        {&cocTilingData.extraCommDataSplit,
            {ALLGATHER_EIGHT_REDUCESCATTER_TWO_TRUE_FP16_EXTRACOMMDATASPLIT_DEFAULT}}};
    SetTilingParam2D(cocTilingData, tilingParamMap);
    return;
}

void AG2RS8TrueFP16Tiling(CoCTilingData &cocTilingData)
{
    std::map<int *, TilingValue> tilingParamMap = {
        {&cocTilingData.swizzlCount,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_SWIZZLECOUNT_DEFAULT,
             g_allgatherTwoReducescatterEightTrueFP16SwizzlecountMap}},
        {&cocTilingData.ubMoveNum,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_UBMOVENUM_DEFAULT,
             g_allgatherTwoReducescatterEightTrueFP16UbmovenumMap}},
        {&cocTilingData.lenPerLoop,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_LENPERLOOPMULT_DEFAULT,
             g_allgatherTwoReducescatterEightTrueFP16LenperloopmultMap}},
        {&cocTilingData.commNpuSplit,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_COMMNPUSPLIT_DEFAULT}},
        {&cocTilingData.commDataSplit,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_COMMDATASPLIT_DEFAULT}},
        {&cocTilingData.extraUbMoveNum,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_EXTRAUBMOVENUM_DEFAULT,
             g_allgatherTwoReducescatterEightTrueFP16ExtraubmovenumMap}},
        {&cocTilingData.extraLenPerLoop,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_EXTRALENPERLOOPMULT_DEFAULT,
             g_allgatherTwoReducescatterEightTrueFP16ExtralenperloopmultMap}},
        {&cocTilingData.extraCommNpuSplit,
            {DIM_EIGHT, g_allgatherTwoReducescatterEightTrueFP16ExtracommnpusplitMap}},
        {&cocTilingData.extraCommDataSplit,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_TRUE_FP16_EXTRACOMMDATASPLIT_DEFAULT,
             g_allgatherTwoReducescatterEightTrueFP16ExtracommdatasplitMap}}};
    SetTilingParam2D(cocTilingData, tilingParamMap);
    return;
}

void AG2RS8FalseFP16Tiling(CoCTilingData &cocTilingData)
{
    std::map<int *, TilingValue> tilingParamMap = {
        {&cocTilingData.swizzlCount,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_SWIZZLECOUNT_DEFAULT,
             g_allgatherTwoReducescatterEightFalseFP16SwizzlecountMap}},
        {&cocTilingData.ubMoveNum,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_UBMOVENUM_DEFAULT,
             g_allgatherTwoReducescatterEightFalseFP16UbmovenumMap}},
        {&cocTilingData.lenPerLoop, {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_LENPERLOOPMULT_DEFAULT}},
        {&cocTilingData.commNpuSplit, {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_COMMNPUSPLIT_DEFAULT}},
        {&cocTilingData.commDataSplit, {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_COMMDATASPLIT_DEFAULT}},
        {&cocTilingData.commDirect,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_COMMDIRECT_DEFAULT,
             g_allgatherTwoReducescatterEightFalseFP16CommdirectMap}},
        {&cocTilingData.extraUbMoveNum,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_EXTRAUBMOVENUM_DEFAULT,
             g_allgatherTwoReducescatterEightFalseFP16ExtraubmovenumMap}},
        {&cocTilingData.extraLenPerLoop, {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_EXTRALENPERLOOPMULT_DEFAULT}},
        {&cocTilingData.extraCommNpuSplit,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_EXTRACOMMNPUSPLIT_DEFAULT,
             g_allgatherTwoReducescatterEightFalseFP16ExtracommnpusplitMap}},
        {&cocTilingData.extraCommDataSplit,
            {ALLGATHER_TWO_REDUCESCATTER_EIGHT_FALSE_FP16_EXTRACOMMDATASPLIT_DEFAULT,
             g_allgatherTwoReducescatterEightFalseFP16ExtracommdatasplitMap}}};
    SetTilingParam2D(cocTilingData, tilingParamMap);
    return;
}

void CoCAllgatherMatmulReduceScatterTilingFunc::GetDefaultTiling(const TaskParam &taskParam)
{
    CoCTilingFunc::GetDefaultTiling(taskParam);

    cocTilingData.swizzlDirect = SWIZZLEDIRECT_ONE;

    cocTilingData.m0 = M0_DEFAULT;
    cocTilingData.k0 = K0_DEFAULT;
    cocTilingData.n0 = N0_DEFAULT;

    cocTilingData.withSerialMode = 0;
    cocTilingData.is91093 = 0;
    cocTilingData.pValue = PVALUE_ONE;
    cocTilingData.commDirect = 0;

    auto rsDim = taskParam.cocParamDesc.twoDimTPInfo.rsDim;
    auto agDim = taskParam.cocParamDesc.twoDimTPInfo.agDim;
    auto innerDimIsAg = taskParam.cocParamDesc.twoDimTPInfo.innerDimIsAg;
    if (agDim == DIM_EIGHT && rsDim == DIM_TWO && !innerDimIsAg) {
        AG8RS2FalseFP16Tiling(cocTilingData);
    } else if (agDim == DIM_EIGHT && rsDim == DIM_TWO && innerDimIsAg) {
        AG8RS2TrueFP16Tiling(cocTilingData);
    } else if (agDim == DIM_TWO && rsDim == DIM_EIGHT && innerDimIsAg) {
        AG2RS8TrueFP16Tiling(cocTilingData);
    } else {
        AG2RS8FalseFP16Tiling(cocTilingData);
    }
    cocTilingData.commNpuSplit = std::min(cocTilingData.commNpuSplit, agDim);
    cocTilingData.extraCommNpuSplit = std::min(cocTilingData.extraCommNpuSplit, rsDim);
}

bool CoCAllgatherMatmulReduceScatterTilingFunc::CheckTiling(const TaskParam &taskParam)
{
    if (!CoCTilingFunc::CheckTiling(taskParam)) {
        return false;
    }

    auto commNpuSplit = cocTilingData.commNpuSplit;
    auto commDataSplit = cocTilingData.commDataSplit;
    auto extraCommNpuSplit = cocTilingData.extraCommNpuSplit;
    auto extraCommDataSplit = cocTilingData.extraCommDataSplit;
    auto coreNum = cocTilingData.blockDim;
    auto useCoreCount = commNpuSplit * commDataSplit + extraCommNpuSplit * extraCommDataSplit;

    const int maxMValue = 200000;
    const int maxNValue = 32768;
    const int maxKValue = 32768;
    std::vector<std::tuple<std::string, int, int, int>> paramCheckList = {
        {"m", cocTilingData.m, PARAM_CHECK_MIN_VALUE_ONE, maxMValue},
        {"k", cocTilingData.k, PARAM_CHECK_MIN_VALUE_ONE, maxKValue},
        {"n", cocTilingData.n, PARAM_CHECK_MIN_VALUE_ONE, maxNValue},
        {"commNpuSplit * commDataSplit + extraCommNpuSplit * extraCommDataSplit",
         useCoreCount, PARAM_CHECK_MIN_VALUE_ONE, coreNum},
    };
    return CheckParamScopeList(paramCheckList);
}
}