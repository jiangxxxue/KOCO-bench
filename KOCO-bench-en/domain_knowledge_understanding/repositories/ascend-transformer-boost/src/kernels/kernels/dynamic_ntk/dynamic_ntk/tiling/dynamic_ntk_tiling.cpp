/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/utils/math/math.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/dynamic_ntk.h"
#include "tiling_data.h"
#include "dynamic_ntk_tiling.h"

namespace AsdOps {
const uint32_t UB_RESERVED_BUFF = 8 * 1024; // reserve 8k
const uint32_t GM_ALIGN_SIZE = 512;         // pack unit in cache 512B
const uint32_t ALIGN_SIZE = 32;             // align unit in cache 32B
const uint32_t DEFAULT_BUFFER_NUM = 2;
const uint32_t MAX_HEAD_DIM = 2048;
const uint32_t MAX_BATCH = 16;
const uint32_t HEAD_DIM_ALIGN = 32;
const uint32_t POS_ALIGN = ALIGN_SIZE / sizeof(int32_t);
const uint32_t OUT_TYPE_FLOAT = 2;
const uint32_t SEQLEN_INDEX = 2;
const uint32_t DYNAMIC_NTK_WKSP_TENSOR_IDX = 5;
const uint32_t SHIFT_AMOUNT = 2;
static constexpr uint32_t REMAIN_SPACE = 16 * 1024 * 1024;

constexpr uint64_t RecursiveSum() { return 0; }

template <typename T, typename... Args> constexpr uint64_t RecursiveSum(T templateId, Args... templateIds)
{
    return static_cast<uint64_t>(templateId) + 10U * RecursiveSum(templateIds...);
}

template <typename T> typename std::enable_if<std::is_integral<T>::value, T>::type FloorDiv(T x, T y)
{
    return y == 0 ? x : x / y;
}

template <typename T> typename std::enable_if<std::is_unsigned<T>::value, T>::type CeilDiv(T x, T y)
{
    if (y != 0 && x != 0) {
        const T quotient = x / y;
        return (x % y != 0) ? (quotient + 1) : quotient;
    }
    return x;
}

template <typename T> inline T MathMod(T num1, T num2)
{
    if (num2 == 0) {
        return num1;
    }
    return num1 % num2;
}

class DynamicNTKTiling {
public:
    DynamicNTKTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
        : launchParam_(launchParam), kernelInfo_(kernelInfo)
    {
    }

    bool CheckParams()
    {
        int32_t posDtype = launchParam_.GetInTensor(0).desc.dtype;
        MKI_CHECK(posDtype == TENSOR_DTYPE_INT32 || posDtype == TENSOR_DTYPE_UINT32,
                  "DynamicNTK positionIds dataType is not int32_t or uint32_t", return false);

        int32_t freqDtype = launchParam_.GetInTensor(1).desc.dtype;
        int32_t seqLenDtype = launchParam_.GetInTensor(SEQLEN_INDEX).desc.dtype;
        MKI_CHECK(freqDtype == TENSOR_DTYPE_FLOAT, "DynamicNTK InvFreqsIn dataType is not float32", return false);
        MKI_CHECK(seqLenDtype == TENSOR_DTYPE_INT32 || seqLenDtype == TENSOR_DTYPE_UINT32,
                  "DynamicNTK seqLens dataType is not int32_t or uint32_t", return false);

        MKI_CHECK((headDim_ % HEAD_DIM_ALIGN) == 0, "DynamicNTK the shape of InvFreqsIn is not align to 32",
                     return false);

        const auto &attrs = AnyCast<OpParam::DynamicNTK>(launchParam_.GetParam());
        outType_ = attrs.outType;
        MKI_CHECK((outType_ == 0 || outType_ == 1 || outType_ == OUT_TYPE_FLOAT),
                  "DynamicNTK the type value is invalid! only support 0 or 1 or 2", return false);
        MKI_CHECK(!(outType_ != 0 && outType_ != OUT_TYPE_FLOAT && is310p),
                  "DynamicNTK the type value is invalid! only support half/float in 310P", return false);
        return true;
    }

    Status DoTiling()
    {
        tilingData_ = reinterpret_cast<AsdOps::DynamicNTKTilingData *>(kernelInfo_.GetTilingHostAddr());
        MKI_CHECK(tilingData_ != nullptr, "tilingDataPtr should not be empty",
                     return Status::FailStatus(ERROR_INVALID_VALUE));
        const Mki::SVector<int64_t> &posDims = launchParam_.GetInTensor(0).desc.dims;
        const Mki::SVector<int64_t> &freqDims = launchParam_.GetInTensor(1).desc.dims;
        const Mki::SVector<int64_t> &seqLenDims = launchParam_.GetInTensor(SEQLEN_INDEX).desc.dims;
        MKI_CHECK_NO_LOG(posDims.size() == 1 && freqDims.size() == 2 && seqLenDims.size() == 1,
            return Status::FailStatus(ERROR_INVALID_VALUE, "positionIds or invFreqs or seqLen dim size is invalid"));
        MKI_CHECK_NO_LOG(posDims[0] <= std::numeric_limits<uint32_t>::max() && posDims[0] > 0,
            return Status::FailStatus(ERROR_INVALID_VALUE, "input posDims dim is invalid"));
        MKI_CHECK_NO_LOG(freqDims[1] <= MAX_HEAD_DIM / 2,
            return Status::FailStatus(ERROR_INVALID_VALUE, "input invFreqs dim is invalid"));
        MKI_CHECK_NO_LOG(seqLenDims[0] <= static_cast<int64_t>(MAX_BATCH) && seqLenDims[0] > 0,
            return Status::FailStatus(ERROR_INVALID_VALUE, "input seqLenDims dim is invalid"));
        numTokens_ = static_cast<uint32_t>(posDims[0]);
        headDim_ = static_cast<uint32_t>(freqDims[1] * SHIFT_AMOUNT);
        batchNum_ = static_cast<uint32_t>(seqLenDims[0]);
        uint32_t posCalcCnt = CeilDiv(numTokens_, POS_ALIGN);
        uint64_t ubSize = PlatformInfo::Instance().GetUbSize();
        uint32_t coreNum = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
        usedCores_ = posCalcCnt > coreNum ? coreNum : posCalcCnt;
        is310p = PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_310P;
        // UB上预留软同步的空间及reserve空间, 310p前32B无法使用
        const uint32_t seqLenSize = Mki::Utils::RoundUp(batchNum_ * static_cast<uint32_t>(sizeof(int32_t)), ALIGN_SIZE);
        MKI_CHECK(ubSize > static_cast<uint32_t>(UB_RESERVED_BUFF + (is310p ? ALIGN_SIZE : 0) + seqLenSize),
                  "ubSize is not enough", return Status::FailStatus(ERROR_INVALID_VALUE, "ubSize is not enough"));
        ubSize = ubSize - UB_RESERVED_BUFF - (is310p ? ALIGN_SIZE : 0);

        MKI_CHECK(CheckParams(), "DynamicNTK param is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE, "param is invalid"));

        const auto &attrs = AnyCast<OpParam::DynamicNTK>(launchParam_.GetParam());
        outType_ = attrs.outType;
        tilingData_->numTokens = numTokens_;
        tilingData_->headDim = headDim_;
        tilingData_->batchNum = batchNum_;

        CalcTiling(ubSize);
        PrintTiling();
        auto ret = CheckTiling();
        if (!ret.Ok()) {
            return ret;
        }

        uint64_t sysWorkspaceSize = REMAIN_SPACE;
        kernelInfo_.GetScratchSizes() = {sysWorkspaceSize};
        kernelInfo_.SetBlockDim(usedCores_);
        kernelInfo_.SetTilingId(GetTilingKey());
        return Status::OkStatus();
    }

    uint32_t GetMaxFreqTilePerUB(uint64_t ubSize, uint32_t freqsLen, uint32_t dbNum) const
    {
        // (freqTile * sizeof(float) + posTile * sizeof(int32) + 2 * freqTile * posTile * sizeof(float)) * DB_NUM
        //  + batchNum * sizeof(int32) <= ubSize
        const uint32_t minPosTile = FloorDiv(ALIGN_SIZE, static_cast<uint32_t>(sizeof(int32_t)));
        const uint32_t aliGnCnt = FloorDiv(ALIGN_SIZE, static_cast<uint32_t>(sizeof(float)));
        const uint32_t seqLenSize = Mki::Utils::RoundUp(batchNum_ * static_cast<uint32_t>(sizeof(int32_t)), ALIGN_SIZE);
        const uint32_t singleFreqSize =
            FloorDiv(FloorDiv(static_cast<uint32_t>(ubSize) - seqLenSize, dbNum) - ALIGN_SIZE,
                     static_cast<uint32_t>(sizeof(float)));
        const uint32_t maxFreqTile = singleFreqSize / (1 + 2 * minPosTile);
        return std::min(static_cast<uint32_t>(Mki::Utils::RoundDown(maxFreqTile, aliGnCnt)), freqsLen);
    }

    uint32_t GetMaxPosTilePerUB(uint64_t ubSize, uint32_t freqTileLen, uint32_t dbNum) const
    {
        // (freqTile * sizeof(float) + posTile * sizeof(int32) + 2 * freqTile * posTile * sizeof(float)) * DB_NUM
        //  + batchNum * sizeof(int32) <= ubSize
        const uint32_t alignCnt = FloorDiv(ALIGN_SIZE, static_cast<uint32_t>(sizeof(int32_t)));
        const uint32_t seqLenSize = Mki::Utils::RoundUp(batchNum_ * static_cast<uint32_t>(sizeof(int32_t)), ALIGN_SIZE);
        const uint32_t singlePosSize =
            FloorDiv(static_cast<uint32_t>(ubSize) - seqLenSize, dbNum) - freqTileLen * sizeof(float);
        const uint32_t maxPosTile =
            FloorDiv(singlePosSize, static_cast<uint32_t>(sizeof(int32_t) + 2 * freqTileLen * sizeof(float)));
        return Mki::Utils::RoundDown(maxPosTile, alignCnt);
    }
    /*
          |f|f|f|f|f|f|f|f|f|f|f|f|f|f|f|f|f|f|f|f|f|f|
          |---------tile-0------|--------tile-1-------|
        |p|---
        |p| |
        |p| |
        |p| core-0
        |p| |
        |p| |
        |p|---
        |p| |
        |p| |
        |p| core-1
        |p| |
        |p| |
        |p|---
        |p| |
        |p|...
     * */
    void CalcTilingWithInvFreqs(uint64_t ubSize)
    {
        // InvFreqs方向不分核，只分tile循环执行
        uint32_t freqsLen = headDim_ >> 1;
        auto freqTileLen = GetMaxFreqTilePerUB(ubSize, freqsLen, DEFAULT_BUFFER_NUM);
        auto freqTileNum = FloorDiv(freqsLen, freqTileLen);
        auto freqTailLen = MathMod(freqsLen, freqTileLen);
        auto posTileLen = GetMaxPosTilePerUB(ubSize, freqTileLen, DEFAULT_BUFFER_NUM);

        // positionIds方向
        /*
                    numTokens = 51   usedCores = 4
        positionIds |---core-0---|---core-1---|---core-2---|---core-3---|
                         16            16          8             11
                    -->
                    posLongCores=2   posLongLen=16
                    posShortCores=1  posShortLen=8     posShortOff=32
                                       posTailCoreLen=11 posTailCoreOff=40
         * */
        uint32_t alignCnt = FloorDiv(numTokens_, POS_ALIGN);
        // 计算每个核上分配的positionIds的长度
        uint32_t posLongCores = MathMod(alignCnt, usedCores_);
        uint32_t posShortCores = usedCores_ - posLongCores;
        uint32_t posShortLen = FloorDiv(alignCnt, usedCores_) * POS_ALIGN;
        uint32_t posLongLen = posLongCores > 0 ? (posShortLen + POS_ALIGN) : 0;
        uint32_t posTailCoreLen =
            (numTokens_ > alignCnt * POS_ALIGN) ? (numTokens_ - alignCnt * POS_ALIGN + posShortLen) : 0;
        posShortCores = posTailCoreLen > 0 ? (posShortCores - 1) : posShortCores;

        tilingData_->freqTileLen = freqTileLen;
        tilingData_->freqTileNum = freqTileNum;
        tilingData_->freqTailLen = freqTailLen;
        tilingData_->posTileLen = posTileLen;
        tilingData_->posLongCores = posLongCores;
        tilingData_->posShortCores = posShortCores;
        tilingData_->posLongLen = posLongLen;
        tilingData_->posShortLen = posShortLen;
        tilingData_->posTailCoreLen = posTailCoreLen;
    }

    void CalcTiling(uint64_t ubSize)
    {
        // 有InvFreqs输入的场景
        CalcTilingWithInvFreqs(ubSize);
    }

    uint64_t GetTilingKey() const { return RecursiveSum(0, outType_, 1); }

    Status CheckTiling() const
    {
        MKI_CHECK(tilingData_->freqTileLen > 0, "DynamicNTK-tiling freqTileLen invalid!",
                  return Status::FailStatus(ERROR_INVALID_VALUE, "freqTileLen invalid!"));
        MKI_CHECK(tilingData_->posTileLen > 0, "DynamicNTK-tiling posTileLen invalid!",
                  return Status::FailStatus(ERROR_INVALID_VALUE, "posTileLen invalid!"));
        uint32_t cores = tilingData_->posLongCores + tilingData_->posShortCores;
        MKI_CHECK(!(cores < usedCores_ && tilingData_->posTailCoreLen == 0),
                  "DynamicNTK-tiling posTailCoreLen invalid!",
                  return Status::FailStatus(ERROR_INVALID_VALUE, "posTailCoreLen invalid!"));
        MKI_CHECK(tilingData_->freqTileLen <= (headDim_ >> 1), "DynamicNTK-tiling freqTileLen invalid!",
                  return Status::FailStatus(ERROR_INVALID_VALUE, "freqTileLen invalid!"));
        uint32_t posLen = tilingData_->posLongCores * tilingData_->posLongLen +
                          tilingData_->posShortCores * tilingData_->posShortLen + tilingData_->posTailCoreLen;
        MKI_CHECK(posLen == numTokens_, "DynamicNTK-tiling calc posLen invalid!",
                  return Status::FailStatus(ERROR_INVALID_VALUE, "calc posLen invalid!"));
        return Status::OkStatus();
    }

private:
    void PrintTiling()
    {
        MKI_LOG(INFO) << "DynamicNTK outType " << outType_;
        MKI_LOG(INFO) << "DynamicNTK numTokens " << tilingData_->numTokens;
        MKI_LOG(INFO) << "DynamicNTK headDim " << tilingData_->headDim;
        MKI_LOG(INFO) << "DynamicNTK batchNum " << tilingData_->batchNum;
        MKI_LOG(INFO) << "DynamicNTK usedCores " << usedCores_;
        MKI_LOG(INFO) << "DynamicNTK freqTileLen " << tilingData_->freqTileLen;
        MKI_LOG(INFO) << "DynamicNTK freqTileNum " << tilingData_->freqTileNum;
        MKI_LOG(INFO) << "DynamicNTK posTileLen " << tilingData_->posTileLen;
        MKI_LOG(INFO) << "DynamicNTK posLongCores " << tilingData_->posLongCores;
        MKI_LOG(INFO) << "DynamicNTK posShortCores " << tilingData_->posShortCores;
        MKI_LOG(INFO) << "DynamicNTK posShortLen " << tilingData_->posShortLen;
        MKI_LOG(INFO) << "DynamicNTK posLongLen " << tilingData_->posLongLen;
        MKI_LOG(INFO) << "DynamicNTK posTailCoreLen " << tilingData_->posTailCoreLen;
        MKI_LOG(INFO) << "DynamicNTK tilingKey " << GetTilingKey();
    }

private:
    DynamicNTKTilingData *tilingData_ = nullptr;
    uint32_t usedCores_ = 0;
    uint32_t numTokens_ = 0;
    uint32_t headDim_ = 0;
    uint32_t batchNum_ = 0;
    int32_t outType_ = 0; // 0:fp16, 1:bf16
    uint32_t ubResvSize = 0;
    bool is310p = false;
    const LaunchParam &launchParam_;
    KernelInfo &kernelInfo_;
};

Status DynamicNTKTilingFunc(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    DynamicNTKTiling tilingObj(launchParam, kernelInfo);
    return tilingObj.DoTiling();
}
} // namespace AsdOps
