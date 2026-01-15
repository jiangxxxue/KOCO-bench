/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lcoc_func.h"
#include "lcoc_args.h"
#include "tiling_args.h"
#include "tiling_func.h"

namespace Lcal {
    int32_t CeilDev(int32_t num, int32_t div)
    {
        if (div == 0) {
            return 0;
        }
        return (num + div - 1) / div;
    }

    int32_t RoundNum(int32_t num, int32_t rnd)
    {
        if (rnd == 0) {
            return 0;
        }
        return (num + rnd - 1) / rnd * rnd;
    }

    void UpdateTilingValue(const int32_t &tilingParam, int32_t &tilingDataParam)
    {
        if (tilingParam != INPUT_PARAM_DEFAULT_VALUE) {
            tilingDataParam = tilingParam;
        }
    }

    double GetMTETime(double mknGB, int32_t m0, int32_t n0, double aBindWidth, double bBindWidth)
    {
        // 预估Matmul计算的MTE2搬运时间
        return DOUBLE * mknGB * (SECOND_TO_MS / ONE_K) * (1.0 / (n0 * aBindWidth) + 1.0 / (m0 * bBindWidth));
    }

    int32_t GetValueFromMKNConditionMap(int32_t m, int32_t k, int32_t n,
                                        int32_t defaultValue,
                                        std::map<int, std::vector<std::vector<int>>> conditionMap)
    {
        int32_t value = defaultValue;
        for (auto iter = conditionMap.cbegin(); iter != conditionMap.cend(); ++iter) {
            for (auto &condition : iter->second) {
                bool inRange =
                        m > condition[CONDITION_M_ST] && m <= condition[CONDITION_M_END] &&
                        k > condition[CONDITION_K_ST] && k <= condition[CONDITION_K_END] &&
                        n > condition[CONDITION_N_ST] && n <= condition[CONDITION_N_END];
                if (inRange) {
                    return iter->first;
                }
            }
        }
        return value;
    }

    bool Is910B(const ChipName &chipName)
    {
        return chipName >= ChipName::CHIP_910B1 && chipName <= ChipName::CHIP_910B41;
    }

    bool Is91093(const ChipName &chipName)
    {
        return chipName >= ChipName::CHIP_910_9391 && chipName <= ChipName::CHIP_910_9362;
    }

    uint32_t GetTilingKey(const MatMulInfo &mmInfo, CoCTilingData &tilingData)
    {
        uint32_t tilingKey = static_cast<uint32_t>(tilingData.swizzlDirect);  // 32
        tilingKey = (static_cast<uint32_t>(tilingKey) << 1) + static_cast<uint32_t>(mmInfo.transA);   // 16
        tilingKey = (static_cast<uint32_t>(tilingKey) << 1) + static_cast<uint32_t>(mmInfo.transB);   // 8
        tilingKey = (static_cast<uint32_t>(tilingKey) << 1) + static_cast<uint32_t>(mmInfo.isInt8);   // 4
        tilingKey = (static_cast<uint32_t>(tilingKey) << 1) + static_cast<uint32_t>(mmInfo.withBias); // 2
        tilingKey = (static_cast<uint32_t>(tilingKey) << 1) + static_cast<uint32_t>(tilingData.splitK);    // 1
        return tilingKey;
    }

    void DealTilingParamByBuffSize(CoCTilingData &cocTilingData)
    {
        auto blockCount = (cocTilingData.is91093 != 0) ? BLOCK_COUNT_3 : MAX_BLOCK_COUNT;
        int maxPeerMemPerRank =
                (cocTilingData.bufferSize * 1024 * 1024) / INPUT_DTYPE / cocTilingData.rankSize / blockCount;
        int maxPValue = maxPeerMemPerRank / cocTilingData.m0 / cocTilingData.k0 / cocTilingData.kLoop;
        cocTilingData.pValue = ClampValue(cocTilingData.pValue, MIN_P_VALUE, maxPValue);

        if (cocTilingData.m0  == DEFAULT_COL
        && cocTilingData.pValue * cocTilingData.m0 * cocTilingData.k0 * cocTilingData.kLoop >= maxPeerMemPerRank) {
            cocTilingData.m0 = DEFAULT_ROW;
            cocTilingData.n0 = DEFAULT_COL;
            cocTilingData.mLoop = CeilDev(cocTilingData.m, cocTilingData.m0);
            cocTilingData.nLoop = CeilDev(cocTilingData.n, cocTilingData.n0);
        }
    }

    int ClampValue(int32_t value, int32_t min, int32_t max)
    {
        return std::max(min, std::min(value, max));
    }

    void SetTilingParam(CoCTilingData &cocTilingData, const std::map<int*, TilingValue>& tilingParamMap)
    {
        int32_t m = cocTilingData.m;
        int32_t k = cocTilingData.k;
        int32_t n = cocTilingData.n;

        for (auto &item : tilingParamMap) {
            auto value = item.second.value;
            auto conditionMap = item.second.conditionMap;
            if (!conditionMap.empty()) {
                *item.first = GetValueFromMKNConditionMap(m, k, n, value, conditionMap);
            } else if (value != -1) {
                *item.first = value;
            }
        }

        cocTilingData.ubMoveNum = cocTilingData.ubMoveNum * HALF_KBYTE;
        if (cocTilingData.m0 >= DEFAULT_ROW) {
            cocTilingData.k0 = DEFAULT_COL;
            cocTilingData.n0 = cocTilingData.m0 == DEFAULT_ROW ? DEFAULT_COL : DEFAULT_ROW;
            cocTilingData.mLoop = CeilDev(cocTilingData.m, cocTilingData.m0);
            cocTilingData.nLoop = CeilDev(cocTilingData.n, cocTilingData.n0);
            cocTilingData.kLoop = CeilDev(cocTilingData.k, cocTilingData.k0);
        }
    }

    void SetSecondCoreSplitTling(CoCTilingData &cocTilingData)
    {
        cocTilingData.extraCommDirect = cocTilingData.commDirect;
        cocTilingData.extraCommNpuSplit = cocTilingData.commNpuSplit;
        cocTilingData.extraCommDataSplit = cocTilingData.commDataSplit;
        cocTilingData.extraLenPerLoop = cocTilingData.lenPerLoop;
        cocTilingData.extraUbMoveNum = cocTilingData.ubMoveNum;
    }

    void SetTilingParam2D(CoCTilingData &cocTilingData, const std::map<int*, TilingValue>& tilingParamMap)
    {
        SetTilingParam(cocTilingData, tilingParamMap);

        cocTilingData.extraUbMoveNum = cocTilingData.extraUbMoveNum * HALF_KBYTE;
        cocTilingData.lenPerLoop = cocTilingData.lenPerLoop * cocTilingData.ubMoveNum / DIV_TWO;
        cocTilingData.extraLenPerLoop = cocTilingData.extraLenPerLoop * cocTilingData.extraUbMoveNum / DIV_TWO;
    }

    std::map<std::string, bool> GetCoCTilingPowerOfTwoParamMap()
    {
        std::map<std::string, bool> powerOfTwoParamMap = {
            {"commDataSplit", true},
            {"extraCommDataSplit", true}
        };
        return powerOfTwoParamMap;
    }

    std::map<std::string, int> GetCoCTilingAlignParamMap()
    {
        std::map<std::string, int> alignParamMap = {
            {"m0", BLOCK_SIZE},
            {"n0", BLOCK_SIZE},
            {"k0", BLOCK_SIZE},
            {"ubMoveNum", HALF_KBYTE},
            {"lenPerLoop", HALF_KBYTE},
            {"extraUbMoveNum", HALF_KBYTE},
            {"extraLenPerLoop", HALF_KBYTE}
        };
        return alignParamMap;
    }

    std::vector<std::tuple<std::string, int, int, int>> GetCoCTilingParamCheckList(const CoCTiling &tiling)
    {
        std::vector<std::tuple<std::string, int, int, int>> paramCheckList = {
            {"m0", tiling.m0, BLOCK_SIZE, CUBE_BLOCK_SIZE},
            {"n0", tiling.n0, BLOCK_SIZE, CUBE_BLOCK_SIZE},
            {"k0", tiling.k0, CUBE_BLOCK_SIZE, AXES_ALIGN_SIZE},
            {"swizzlCount", tiling.swizzlCount, PARAM_CHECK_MIN_VALUE_ONE, PARAM_CHECK_MAX_VALUE},
            {"swizzlDirect", tiling.swizzlDirect, SWIZZLE_DIRECT_ZERO, SWIZZLE_DIRECT_ONE},
            {"ubMoveNum", tiling.ubMoveNum, HALF_KBYTE, MAX_UB_NUM},
            {"commNpuSplit", tiling.commNpuSplit, PARAM_CHECK_MIN_VALUE_ONE, PARAM_CHECK_MAX_VALUE},
            {"commDataSplit", tiling.commDataSplit, PARAM_CHECK_MIN_VALUE_ONE, PARAM_CHECK_MAX_VALUE},
            {"commDirect", tiling.commDirect, COMM_DATA_DIRECT, COMM_NPU_DIRECT},
            {"lenPerLoop", tiling.lenPerLoop, HALF_KBYTE, PARAM_CHECK_MAX_VALUE},
            {"extraUbMoveNum", tiling.extraUbMoveNum, HALF_KBYTE, MAX_UB_NUM},
            {"extraCommNpuSplit", tiling.extraCommNpuSplit, PARAM_CHECK_MIN_VALUE_ONE, PARAM_CHECK_MAX_VALUE},
            {"extraCommDataSplit", tiling.extraCommDataSplit, PARAM_CHECK_MIN_VALUE_ONE, PARAM_CHECK_MAX_VALUE},
            {"extraCommDirect", tiling.extraCommDirect, COMM_DATA_DIRECT, COMM_NPU_DIRECT},
            {"extraLenPerLoop", tiling.extraLenPerLoop, HALF_KBYTE, PARAM_CHECK_MAX_VALUE},
            {"splitK", tiling.splitK, PARAM_CHECK_MIN_VALUE_ZERO, PARAM_CHECK_MAX_VALUE},
            {"write2OtherRank", tiling.write2OtherRank, PARAM_CHECK_MIN_VALUE_ZERO, PARAM_CHECK_MIN_VALUE_ONE},
            {"withSerialMode", tiling.withSerialMode, PARAM_CHECK_MIN_VALUE_ZERO, PARAM_CHECK_MIN_VALUE_ONE},
            {"is91093", tiling.is91093, PARAM_CHECK_MIN_VALUE_ZERO, PARAM_CHECK_MIN_VALUE_ONE}
        };
        return paramCheckList;
    }

    bool CheckCoCTiling(const CoCTiling &tiling)
    {
        auto powerOfTwoParamMap = GetCoCTilingPowerOfTwoParamMap();
        auto alignParamMap = GetCoCTilingAlignParamMap();
        auto paramCheckList = GetCoCTilingParamCheckList(tiling);
        for (auto &param : paramCheckList) {
            auto name = std::get<0>(param);
            auto value = std::get<1>(param);
            auto min = std::get<2>(param);
            auto max = std::get<3>(param);
            if (value == INPUT_PARAM_DEFAULT_VALUE) {
                continue;
            }
            if (!CheckParamScope(name, value, min, max)) {
                return false;
            }
            if (alignParamMap.find(name) != alignParamMap.end()
            && !CheckParamAlign(name, value, alignParamMap[name])) {
                return false;
            }
            if (powerOfTwoParamMap.find(name) != powerOfTwoParamMap.end()
            && !CheckParamPowerOfTwo(name, value)) {
                return false;
            }
        }
        return true;
    }

    bool CheckCoCTilingData(const CoCTilingData &tilingData)
    {
        if (!CheckCoCTiling(tilingData)) {
            return false;
        }
        std::vector<std::tuple<std::string, int, int, int>> paramCheckList = {
            {"mLoop", tilingData.mLoop, PARAM_CHECK_MIN_VALUE_ONE, PARAM_CHECK_MAX_VALUE},
            {"kLoop", tilingData.kLoop, PARAM_CHECK_MIN_VALUE_ONE, PARAM_CHECK_MAX_VALUE},
            {"nLoop", tilingData.nLoop, PARAM_CHECK_MIN_VALUE_ONE, PARAM_CHECK_MAX_VALUE},
            {"coreLoop", tilingData.coreLoop, PARAM_CHECK_MIN_VALUE_ONE, PARAM_CHECK_MAX_VALUE},
            {"tilingKey", tilingData.tilingKey, PARAM_CHECK_MIN_VALUE_ZERO, PARAM_CHECK_MAX_VALUE},
        };
        return CheckParamScopeList(paramCheckList);
    }

    void TransformCoCTiling(const CoCTiling &tiling, CoCTilingData &tilingData)
    {
        int* tilingPtr = reinterpret_cast<int*>(const_cast<Lcal::CoCTiling*>(&tiling));
        int* tilingDataPtr = reinterpret_cast<int*>(&tilingData);
        int length = sizeof(tiling) / sizeof(int32_t);
        for (int i = 0; i < length; i++) {
            UpdateTilingValue(tilingPtr[i], tilingDataPtr[i]);
        }
    }

    void CalTilingParam(const MatMulInfo &mmInfo, CoCTilingData &tilingData)
    {
        // 计算
        tilingData.mLoop = CeilDev(tilingData.m, tilingData.m0);
        tilingData.kLoop = CeilDev(tilingData.k, tilingData.k0);
        tilingData.nLoop = CeilDev(tilingData.n, tilingData.n0);
        tilingData.coreLoop = tilingData.batchSize * tilingData.mLoop * tilingData.nLoop;
        tilingData.tilingKey = GetTilingKey(mmInfo, tilingData);
        // 对齐
        tilingData.ubMoveNum = RoundNum(tilingData.ubMoveNum, HALF_KBYTE);
        tilingData.lenPerLoop = RoundNum(tilingData.lenPerLoop, HALF_KBYTE);
        tilingData.extraUbMoveNum = RoundNum(tilingData.extraUbMoveNum, HALF_KBYTE);
        tilingData.extraLenPerLoop = RoundNum(tilingData.extraLenPerLoop, HALF_KBYTE);
    }

    void SetTilingInputParam(const TaskParam &taskParam, CoCTilingData &tilingData)
    {
        tilingData.m = taskParam.cocParamDesc.mmInfo.m;
        tilingData.n = taskParam.cocParamDesc.mmInfo.n;
        tilingData.k = taskParam.cocParamDesc.mmInfo.k;
        tilingData.batchSize = taskParam.cocParamDesc.mmInfo.batchSize;
        tilingData.blockDim = taskParam.blockDim;
        tilingData.rank = taskParam.rank;
        tilingData.rankSize = taskParam.rankSize;
        tilingData.bufferSize = taskParam.bufferSize;
    }

    void SetTilingData(const TaskParam &taskParam, const CoCTiling &tiling, CoCTilingData &tilingData)
    {
        // 输入Tiling赋值给Tiling策略的参数
        TransformCoCTiling(tiling, tilingData);
        // 根据最终的Tiling策略参数，计算mLoop等参数
        CalTilingParam(taskParam.cocParamDesc.mmInfo, tilingData);
    }
}