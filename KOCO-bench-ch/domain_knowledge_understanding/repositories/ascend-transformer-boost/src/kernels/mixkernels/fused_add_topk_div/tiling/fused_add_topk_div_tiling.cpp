/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "fused_add_topk_div_tiling.h"
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/math/math.h>
#include "atbops/params/params.h"
#include "tiling_data.h"

namespace AtbOps {
    constexpr uint32_t BYTE_BLOCK = 32;
    constexpr int32_t X_INPUT_INDEX = 0;
    constexpr int32_t ADD_NUM_INPUT_INDEX = 1;
    constexpr int32_t MAPPING_NUM_INPUT_INDEX = 2;
    constexpr int32_t MAPPING_TABLE_INPUT_INDEX = 3;
    constexpr int32_t FLOAT_BYTES = 4;

    constexpr uint32_t DIM_INDEX0 = 0;
    constexpr uint32_t DIM_INDEX1 = 1;

    constexpr uint32_t RESERVED_UB = 16 * 1024;
    constexpr int32_t BASE_COUNT = 256;
    constexpr int32_t NUM_FOUR = 4;
    constexpr int32_t SORT_UNIT = 32;
    constexpr int32_t GROUP_LIMIT = 8;
    constexpr int32_t MAPPING_TABLE_DIM_ONE_LIMIT = 128;
    constexpr int32_t NUM_SIXTEEN = 16;
    constexpr int32_t NUM_TEN = 10;
    constexpr uint32_t SYS_WORKSPACESIZE = 16 * 1024 * 1024;

    static std::map<TensorDType, int32_t> g_dtypeMap = {{TENSOR_DTYPE_FLOAT, 0},
                                                        {TENSOR_DTYPE_FLOAT16, 1},
                                                        {TENSOR_DTYPE_BF16, 2}};

    template <typename TilingData>
    class FusedAddTopkDivTiling {
    public:
        explicit FusedAddTopkDivTiling(FusedAddTopkDivTilingData &tilingDataPtr,
                                       const uint32_t inputCoreNum, const uint32_t inputUbSize)
        {
            this->firstDimSize = tilingDataPtr.firstDimSize;
            this->secondDimSize = tilingDataPtr.secondDimSize;
            this->addNumDimSize = tilingDataPtr.addNumDimSize;
            this->groupNum = tilingDataPtr.groupNum;
            this->groupTopk = tilingDataPtr.groupTopk;
            this->n = tilingDataPtr.n;
            this->k = tilingDataPtr.k;
            this->activateType = tilingDataPtr.activateType;
            this->isNorm = tilingDataPtr.isNorm;
            this->scale = tilingDataPtr.scale;
            this->groupEles = tilingDataPtr.groupEles;
            this->dtype = tilingDataPtr.dtype;
            this->ubSize = FloorAlign(inputUbSize, BYTE_BLOCK);
            this->coreNum = inputCoreNum;
            return;
        }

        void GetTiling(TilingData *tilingDataPtr);

    private:
        void GetTilingKey(TilingData &tilingDataPtr);
        void GetUsedCore();
        void SplitUb();
        void FillTilingData(TilingData &tilingDataPtr) const;
        template <typename T1, typename T2>
        inline T1 CeilAlign(T1 a, T2 b) const
        {
            return b == 0 ? a : (a + b - 1) / b * b;
        }
        template <typename T1, typename T2>
        inline T1 FloorAlign(T1 a, T2 b) const
        {
            return b == 0 ? a : (a) / b * b;
        }

    private:
        uint32_t firstDimSize = 0;
        uint32_t secondDimSize = 0;
        uint32_t addNumDimSize = 0;
        uint32_t groupNum = 0;
        uint32_t groupTopk;
        uint32_t n;
        uint32_t k = 0;
        uint32_t activateType;
        uint32_t isNorm;
        float scale = 1.0;
        uint32_t groupEles = 0;
        uint32_t ubSize = 0;
        uint32_t usedCoreNum = 0;
        uint32_t coreNum = 0;
        uint32_t batchPerCore = 1;
        uint32_t tailBatch = 0;
        uint32_t ubFactorElement = 0;
        uint32_t tilingKey = 0;
        uint32_t dtype = 1;
        uint64_t workspacePerCore = 0;
    };

    template <typename TilingData>
    void FusedAddTopkDivTiling<TilingData>::GetTilingKey(TilingData &tilingDataPtr)
    {
        tilingKey = tilingDataPtr.enableExpertMapping * NUM_TEN + dtype;
    }

    template <typename TilingData>
    void FusedAddTopkDivTiling<TilingData>::GetUsedCore()
    {
        if (firstDimSize <= coreNum) {
            batchPerCore = 1;
            usedCoreNum = firstDimSize;
            tailBatch = 0;
            return;
        }
        batchPerCore = coreNum == 0 ? firstDimSize : firstDimSize / coreNum;
        tailBatch = coreNum == 0 ? 0 : firstDimSize % coreNum;
        usedCoreNum = coreNum;
        return;
    }

    template <typename TilingData>
    void FusedAddTopkDivTiling<TilingData>::SplitUb()
    {
        uint32_t tilingDataSize = CeilAlign(sizeof(TilingData), BYTE_BLOCK);
        uint32_t canUseUbSize = FloorAlign(ubSize - tilingDataSize, BYTE_BLOCK);
        ubFactorElement = FloorAlign(canUseUbSize, BYTE_BLOCK);
    }

    template <typename TilingData>
    void FusedAddTopkDivTiling<TilingData>::FillTilingData(TilingData &tilingDataPtr) const
    {
        tilingDataPtr.firstDimSize = firstDimSize;
        tilingDataPtr.secondDimSize = secondDimSize;
        tilingDataPtr.addNumDimSize = addNumDimSize;
        tilingDataPtr.groupNum = groupNum;
        tilingDataPtr.groupTopk = groupTopk;
        tilingDataPtr.n = n;
        tilingDataPtr.k = k;
        tilingDataPtr.activateType = activateType;
        tilingDataPtr.isNorm = isNorm;
        tilingDataPtr.scale = scale;
        tilingDataPtr.groupEles = groupEles;
        tilingDataPtr.blockNum = usedCoreNum;
        tilingDataPtr.ubFactorElement = ubFactorElement;
        tilingDataPtr.batchPerCore = batchPerCore;
        tilingDataPtr.tailBatch = tailBatch;
        tilingDataPtr.tilingKey = tilingKey;
        uint64_t wsSize = BASE_COUNT * FLOAT_BYTES;
        tilingDataPtr.workspacePerCore = wsSize;
        tilingDataPtr.tempSize = firstDimSize * secondDimSize * FLOAT_BYTES;
    }

    template <typename TilingData>
    void FusedAddTopkDivTiling<TilingData>::GetTiling(TilingData *tilingDataPtr)
    {
        GetTilingKey(*tilingDataPtr);
        GetUsedCore();
        SplitUb();
        FillTilingData(*tilingDataPtr);
    }
 
    template <typename TilingData>
    void GetFusedAddTopkDivTiling(TilingData *tilingDataPtr, uint32_t coreNum, uint32_t ubSize)
    {
        class FusedAddTopkDivTiling<TilingData> tilingObj(*tilingDataPtr, coreNum, ubSize);
        tilingObj.GetTiling(tilingDataPtr);
    }

    static void PrintTilingData(const FusedAddTopkDivTilingData &tilingDataPtr)
    {
        MKI_LOG(INFO) << "firstDimSize is: " << tilingDataPtr.firstDimSize << "\n"
                      << "secondDimSize is: " << tilingDataPtr.secondDimSize << "\n"
                      << "addNumDimSize is: " << tilingDataPtr.addNumDimSize << "\n"
                      << "groupNum is: " << tilingDataPtr.groupNum << "\n"
                      << "grouptopk is: " << tilingDataPtr.groupTopk << "\n"
                      << "n is: " << tilingDataPtr.n << "\n"
                      << "k is: " << tilingDataPtr.k << "\n"
                      << "activateType is: " << tilingDataPtr.activateType << "\n"
                      << "isNorm is: " << tilingDataPtr.isNorm << "\n"
                      << "scale is: " << tilingDataPtr.scale << "\n"
                      << "groupEles is: " << tilingDataPtr.groupEles << "\n"
                      << "blockNum is: " << tilingDataPtr.blockNum << "\n"
                      << "dtype is: " << tilingDataPtr.dtype << "\n"
                      << "ubFactorElement is: " << tilingDataPtr.ubFactorElement << "\n"
                      << "batchPerCore is: " << tilingDataPtr.batchPerCore << "\n"
                      << "tailBatch is: " << tilingDataPtr.tailBatch << "\n"
                      << "tilingKey is: " << tilingDataPtr.tilingKey << "\n"
                      << "tempSize is: " << tilingDataPtr.tempSize << "\n"
                      << "enableExpertMapping is: " << tilingDataPtr.enableExpertMapping << "\n"
                      << "expertNum is: " << tilingDataPtr.expertNum << "\n"
                      << "tableDim is: " << tilingDataPtr.tableDim << "\n"
                      << "workspacePerCore is: " << tilingDataPtr.workspacePerCore;
    }
    template <typename T1, typename T2>
    Status CeilAlign(T1 a, T2 b)
    {
        return b == 0 ? a : (a + b - 1) / b * b;
    }
 
    Status GetInputInfo(const LaunchParam &launchParam, FusedAddTopkDivTilingData &tilingDataPtr)
    {
        auto inTensor0 = launchParam.GetInTensor(X_INPUT_INDEX).desc;
        auto inTensor1 = launchParam.GetInTensor(ADD_NUM_INPUT_INDEX).desc;
        tilingDataPtr.firstDimSize = inTensor0.dims[DIM_INDEX0];
        tilingDataPtr.secondDimSize = inTensor0.dims[DIM_INDEX1];
        tilingDataPtr.addNumDimSize = inTensor1.dims[DIM_INDEX0];
 
        auto param = AnyCast<OpParam::FusedAddTopkDiv>(launchParam.GetParam());
        tilingDataPtr.groupNum = static_cast<int32_t>(param.groupNum);
        tilingDataPtr.groupTopk = static_cast<int32_t>(param.groupTopk);
        tilingDataPtr.n = static_cast<int32_t>(param.n);
        tilingDataPtr.k = static_cast<int32_t>(param.k);
        tilingDataPtr.activateType = static_cast<int32_t>(param.activateType);
        tilingDataPtr.isNorm = static_cast<uint32_t>(param.isNorm);
        tilingDataPtr.enableExpertMapping = static_cast<uint32_t>(param.enableExpertMapping);
        tilingDataPtr.groupEles = tilingDataPtr.groupNum == 0 ? tilingDataPtr.secondDimSize :
                                   tilingDataPtr.secondDimSize / tilingDataPtr.groupNum;
        tilingDataPtr.scale = param.scale;
        if (tilingDataPtr.enableExpertMapping) {
            const Tensor &inTensor3 = launchParam.GetInTensor(MAPPING_TABLE_INPUT_INDEX);
            tilingDataPtr.expertNum = inTensor3.desc.dims[DIM_INDEX0];
            tilingDataPtr.tableDim = inTensor3.desc.dims[DIM_INDEX1];
        }
        return Status::OkStatus();
    }
 
    Status CheckMkiParam(const LaunchParam &launchParam)
    {
        auto param = AnyCast<OpParam::FusedAddTopkDiv>(launchParam.GetParam());
        auto groupNum = param.groupNum;
        auto groupTopk = param.groupTopk;
        auto k = param.k;
        auto n = param.n;
        auto activateType = param.activateType;
        auto b = launchParam.GetInTensor(X_INPUT_INDEX).desc.dims[DIM_1];
        MKI_CHECK(groupNum > 0, "groupNum should be greater than 0.",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(groupTopk > 0, "groupTopk should be greater than 0.",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(n > 0, "n should be greater than 0.",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(k > 0, "k should be greater than 0.",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(b % groupNum == 0, "b should be a multiple of groupNum.",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(groupTopk <= groupNum, "groupTopk should be less or equal to groupNum.",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(k <= b, "k should be less or equal to b",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(n <= b / groupNum, "n should be less or equal to b / groupNum.",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(b / groupNum <= SORT_UNIT, "b / groupNum should be less or equal to 32.",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(activateType == 0, "activateType currently only supports equal to 0.",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        if (b >= SORT_UNIT) {
            MKI_CHECK(groupNum == GROUP_LIMIT, "When b is greater or equal to 32, groupNum only supports 8.",
                      return Status::FailStatus(ERROR_INVALID_VALUE));
        }
        return Status::OkStatus();
    }
 
    Status FusedAddTopkDiv4Tiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
    {
        MKI_LOG(INFO) << " FusedAddTopkDivTiling START ";
        FusedAddTopkDivTilingData *tilingDataPtr =
        reinterpret_cast<AtbOps::FusedAddTopkDivTilingData *>(kernelInfo.GetTilingHostAddr());
        MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
 
        auto checkMkiParam = CheckMkiParam(launchParam);
        if (!checkMkiParam.Ok()) {
            return Status::FailStatus(ERROR_INVALID_VALUE);
        }
        uint32_t coreNum = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
        uint32_t ubSize = static_cast<uint32_t>(PlatformInfo::Instance().GetUbSize());
        uint32_t availableUb = ubSize - RESERVED_UB;
 
        auto inputDatatype = launchParam.GetInTensor(X_INPUT_INDEX).desc.dtype;
        tilingDataPtr->dtype = g_dtypeMap[inputDatatype];
 
        auto checkInputInfo = GetInputInfo(launchParam, *tilingDataPtr);
        if (!checkInputInfo.Ok()) {
            return Status::FailStatus(ERROR_INVALID_VALUE);
        }
        GetFusedAddTopkDivTiling<FusedAddTopkDivTilingData>(tilingDataPtr, coreNum, availableUb);
        uint32_t sysWorkspaceSize = SYS_WORKSPACESIZE;
        uint32_t tilingKey = tilingDataPtr->tilingKey;
        uint32_t blockNum = tilingDataPtr->blockNum;
        kernelInfo.SetBlockDim(blockNum);
        kernelInfo.SetTilingId(tilingKey);
        uint32_t syncWorkspaceSize = sysWorkspaceSize + blockNum * tilingDataPtr->workspacePerCore;
        kernelInfo.GetScratchSizes() = {syncWorkspaceSize};
        PrintTilingData(*tilingDataPtr);
        return Status::OkStatus();
    }
} // namespace AtbOps