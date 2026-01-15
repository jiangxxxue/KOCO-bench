/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <limits>
#include <set>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "tiling_data.h"
#include "gating_tiling.h"

static constexpr uint32_t CORE_NUM = 1;
static constexpr uint64_t GATING_GLOBAL_SORT_WORKSPACE_TENSOR_IDX = 6;
static constexpr uint64_t GATING_CUMSUM_WORKSPACE_TENSOR_IDX = 7;
static constexpr uint64_t GATING_SYNC_WORKSPACE_TENSOR_IDX = 8;
static constexpr int32_t BYTES_32 = 32;

constexpr uint64_t INT64_SIZE = sizeof(int64_t);
constexpr uint64_t INT32_SIZE = sizeof(int32_t);
static constexpr int32_t GATING_TILING_SIZE_310P = 512;
static constexpr int32_t GATING_TILING_SIZE_910B = 2048;
static constexpr uint16_t WORKSPACE_SIZE_FACTOR_310P = 16;
static constexpr uint16_t WORKSPACE_SIZE_FACTOR_910B = 8;


static constexpr uint64_t GATING_TILING_ID_BASE = 2000000000;
static constexpr uint64_t GATING_TILING_ID_TYPE = 1; // 1:cumSum int64

namespace AtbOps {
Status LaunchParamCheck(const LaunchParam &launchParam)
{
    if (launchParam.GetParam().Type() != typeid(OpParam::Gating)) {
        return Status::FailStatus(
            ERROR_ATTR_INVALID_TYPE,
            "Failed to check gating param, type of specificParam is not equals to OpParam::Gating");
    }
    auto param = AnyCast<OpParam::Gating>(launchParam.GetParam());

    const int32_t gatingTilingSize = PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B ?
        GATING_TILING_SIZE_910B : GATING_TILING_SIZE_310P;
    const int64_t topKNum = launchParam.GetInTensor(0).desc.dims.at(0);
    std::set<int32_t> sortedExpert(param.deviceExpert.begin(), param.deviceExpert.end());
    MKI_CHECK(topKNum <= (std::numeric_limits<int32_t>::max() - gatingTilingSize),
              "topKNum is larger than INT32_MAX - GATING_TILING_SIZE, tiling calculation is failed",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(param.deviceExpert.size() <= static_cast<uint32_t>(param.headSize),
              "deviceExpert.size() must less than or equal cumSumNum",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(sortedExpert.size() == param.deviceExpert.size(),
              "Expert elements should be unique",
              return Status::FailStatus(ERROR_INVALID_VALUE));

    return Status::OkStatus();
}

void SetBaseTilingData(const LaunchParam &launchParam, GatingTilingData *tilingDataPointer)
{
    const int64_t topKNum = launchParam.GetInTensor(0).desc.dims.at(0); // topK*tokenNum, token数 * 专家数
    const int32_t gatingTilingSize = PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B ?
        GATING_TILING_SIZE_910B : GATING_TILING_SIZE_310P;
    auto param = AnyCast<OpParam::Gating>(launchParam.GetParam());
    const int64_t cumSumNum = param.headSize; // cumSum的长度
    tilingDataPointer->cumSumInt64 = static_cast<uint32_t>(param.cumSumInt64);
    tilingDataPointer->topkExpertNum = param.headNum;
    tilingDataPointer->topKNum = static_cast<int32_t>(topKNum);
    tilingDataPointer->topKNumPadded = tilingDataPointer->topKNum % gatingTilingSize == 0 ? tilingDataPointer->topKNum :
        (tilingDataPointer->topKNum + (gatingTilingSize - tilingDataPointer->topKNum % gatingTilingSize));
    tilingDataPointer->cumSumNum = static_cast<int32_t>(cumSumNum);
    tilingDataPointer->cumSumNum32BytesPadded = static_cast<int32_t>((cumSumNum * INT32_SIZE) % BYTES_32 == 0 ?
        cumSumNum : (cumSumNum + (BYTES_32 - (cumSumNum * INT32_SIZE) % BYTES_32) / INT32_SIZE));
    if (tilingDataPointer->cumSumNum > 0) {
        tilingDataPointer->deviceExpertNum = static_cast<int32_t>(param.deviceExpert.size());
    }
    if (tilingDataPointer->deviceExpertNum > 0) {
        std::set<int32_t> sortedExpert(param.deviceExpert.begin(), param.deviceExpert.end());
        int index = 0;
        for (auto iter = sortedExpert.begin(); iter != sortedExpert.end(); iter++) {
            tilingDataPointer->deviceExpert[index] = *iter;
            index++;
        }
    }
}

// 计算实际tiling所需参数
void DoTiling(GatingTilingData *tilingDataPointer, const int32_t coreNum)
{
    const int32_t gatingTilingSize = PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B ?
        GATING_TILING_SIZE_910B : GATING_TILING_SIZE_310P;
    // tiling 逻辑
    int32_t blockNum =
        (tilingDataPointer->topKNum + gatingTilingSize - 1) / gatingTilingSize;  // 输入tensor数据分块数量
    int32_t avgBlockNumPerCore = blockNum / coreNum;                             // 每个核平均分到的块数量
    int32_t tailBlockNum = blockNum % coreNum;                                   // 全部块平均分到每个核后，剩余块数量
    int32_t actualCoreNum = blockNum > coreNum ? coreNum : blockNum;             // 实际用到的核数量
    tilingDataPointer->blockNum = blockNum;
    tilingDataPointer->actualCoreNum = actualCoreNum;
    tilingDataPointer->tailBlockDataSize = tilingDataPointer->topKNum % gatingTilingSize == 0 ?
        gatingTilingSize : tilingDataPointer->topKNum % gatingTilingSize;  // 尾块有效数据长度，单位为元素
    if (avgBlockNumPerCore > 0) {
        for (int i = 0; i < coreNum; ++i) {
            tilingDataPointer->blockNumPerCore[i] = avgBlockNumPerCore;
        }
    }
    // 剩余块，从第0个核开始，依次向每个核分配1个块
    if (tailBlockNum > 0) {
        for (int i = 0; i < tailBlockNum; ++i) {
            tilingDataPointer->blockNumPerCore[i] += 1;
        }
    }
    // 计算每个核的offset
    int32_t accumulator = 0;
    for (int i = 0; i < actualCoreNum; ++i) {
        tilingDataPointer->beginBlockIndexPerCore[i] = accumulator;
        tilingDataPointer->offSetPerCore[i] = accumulator * gatingTilingSize;
        accumulator += tilingDataPointer->blockNumPerCore[i];
    }

    // 核间同步需要的Gm
    const int32_t syncSize = (coreNum * BYTES_32 *
                              tilingDataPointer->actualCoreNum + coreNum * BYTES_32 + BYTES_32) * 32;
    tilingDataPointer->syncSize = syncSize;
}

// 打印参数
void PrintTilingParams(const GatingTilingData *tilingDataPointer, uint64_t globalSortWorkspaceSize,
                       uint64_t cumSumWorkspaceSize)
{
    // 打印tiling信息
    MKI_LOG(INFO) << "globalSortWorkspaceSize: " << globalSortWorkspaceSize
                  << ", cumSumWorkspaceSize: " << cumSumWorkspaceSize
                  << ", topkExpertNum: " << tilingDataPointer->topkExpertNum
                  << ", topKNum: " << tilingDataPointer->topKNum
                  << ", topKNumPadded: " << tilingDataPointer->topKNumPadded
                  << ", cumSumNum: " << tilingDataPointer->cumSumNum
                  << ", cumSumNum32BytesPadded: " << tilingDataPointer->cumSumNum32BytesPadded
                  << ", actualCoreNum: " << tilingDataPointer->actualCoreNum
                  << ", blockNum: " << tilingDataPointer->blockNum
                  << ", tailBlockDataSize: " << tilingDataPointer->tailBlockDataSize
                  << ", syncSize: " << tilingDataPointer->syncSize
                  << ", deviceExpertNum: " << tilingDataPointer->deviceExpertNum
                  << ", cumSumInt64: " << tilingDataPointer->cumSumInt64;

    std::stringstream blockNumPerCoreStr;
    std::stringstream beginBlockIndexPerCoreStr;
    std::stringstream offsetPerCoreStr;
    for (int i = 0; i < tilingDataPointer->actualCoreNum; ++i) {
        blockNumPerCoreStr << tilingDataPointer->blockNumPerCore[i] << " ";
        beginBlockIndexPerCoreStr << tilingDataPointer->beginBlockIndexPerCore[i] << " ";
        offsetPerCoreStr << tilingDataPointer->offSetPerCore[i] << " ";
    }
    MKI_LOG(INFO) << "blockNumPerCore=" << blockNumPerCoreStr.str();
    MKI_LOG(INFO) << "beginBlockIndexPerCore=" << beginBlockIndexPerCoreStr.str();
    MKI_LOG(INFO) << "offsetPerCore=" << offsetPerCoreStr.str();
}

// 将launchParam中切片参数设置到kernelInfo中，在kernel中调用
Status GatingTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto coreNum = static_cast<int32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    MKI_CHECK(coreNum > 1, "Get insufficient num of vector cores", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(coreNum < MAX_CORE_NUM, "Get exceeds the limit num of vector cores, need less than 50",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    coreNum = coreNum - 1;  // 留一个核给cumSum聚合用

    uint8_t *tilingPtr = kernelInfo.GetTilingHostAddr();
    MKI_CHECK(tilingPtr != nullptr, "tiling data error", return Status::FailStatus(ERROR_INVALID_VALUE));
    auto tilingDataPointer = reinterpret_cast<GatingTilingData *>(tilingPtr);

    // 检查入参
    Status st = LaunchParamCheck(launchParam);
    MKI_CHECK(st.Ok(), "LaunchParam is invalid for gating tiling", return st);

    SetBaseTilingData(launchParam, tilingDataPointer);
    DoTiling(tilingDataPointer, coreNum);

    // 检查出参
    MKI_CHECK(tilingDataPointer->cumSumNum <= 1024, "CumSumNum cannot greater than 1024",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(tilingDataPointer->cumSumNum == 0 ||
                 tilingDataPointer->cumSumNum >= tilingDataPointer->topkExpertNum,
                 "CumSumNum must greater than topkExpertNum if CumSumNum not zero",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(tilingDataPointer->cumSumNum == 0 ||
                 tilingDataPointer->deviceExpert[tilingDataPointer->deviceExpertNum - 1] < tilingDataPointer->cumSumNum,
                 "Expert index cannot greater than CumSumNum - 1",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(tilingDataPointer->deviceExpert[0] >= 0,
                 "Expert index cannot less than zero",
                 return Status::FailStatus(ERROR_INVALID_VALUE));

    // globalSortWorkspaceSize存放排序所需结构体
    // 910B结构体占用空间是2倍，归并排序需要额外的2倍空间
    // 310P的排序结构体占用空间是8倍
    int32_t workspaceSizeFactor = PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B ?
        WORKSPACE_SIZE_FACTOR_910B : WORKSPACE_SIZE_FACTOR_310P;
    uint64_t globalSortWorkspaceSize = static_cast<uint64_t>(tilingDataPointer->topKNumPadded
        * workspaceSizeFactor * INT32_SIZE);

    // cumSum计算需要的workspace, memset在310P下必须大于2048，所以添加阈值
    uint64_t cumSumWorkspaceSize = static_cast<uint64_t>(tilingDataPointer->cumSumNum32BytesPadded
        * tilingDataPointer->actualCoreNum);
    cumSumWorkspaceSize *= (tilingDataPointer->cumSumInt64 != 0) ? INT64_SIZE : INT32_SIZE;
    uint64_t threshold = 2048 * 4;
    cumSumWorkspaceSize = cumSumWorkspaceSize > threshold ? cumSumWorkspaceSize : threshold;

    // 打印tiling信息
    PrintTilingParams(tilingDataPointer, globalSortWorkspaceSize, cumSumWorkspaceSize);

    kernelInfo.GetScratchSizes() = {globalSortWorkspaceSize, cumSumWorkspaceSize,
                                      static_cast<uint64_t>(tilingDataPointer->syncSize)};
    kernelInfo.SetMemsetInfo(GATING_GLOBAL_SORT_WORKSPACE_TENSOR_IDX, globalSortWorkspaceSize);
    kernelInfo.SetMemsetInfo(GATING_CUMSUM_WORKSPACE_TENSOR_IDX, cumSumWorkspaceSize);
    kernelInfo.SetMemsetInfo(GATING_SYNC_WORKSPACE_TENSOR_IDX, static_cast<uint64_t>(tilingDataPointer->syncSize));
    // 加1的原因是用这个核做cumSum聚合计算，因为cumSum聚合计算是标量计算，比较慢，需要单独一个核来做
    kernelInfo.SetBlockDim(tilingDataPointer->actualCoreNum + 1);
    
    // 设置tilingkey
    uint64_t tilingkey = GATING_TILING_ID_BASE;
    tilingkey += (tilingDataPointer->cumSumInt64 != 0) ? GATING_TILING_ID_TYPE : 0;
    MKI_LOG(INFO) << "tilingKey is " << tilingkey;
    kernelInfo.SetTilingId(tilingkey);
    return Status::OkStatus();
}
} // namespace AtbOps
