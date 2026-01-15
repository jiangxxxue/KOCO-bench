/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "gmm_add_tiling.h"

#include <climits>

#include "mki/utils/platform/platform_info.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "mki/utils/math/math.h"
#include "mki/types.h"
#include "gmm_add_tilingdata.h"
#include "atbops/params/params.h"
#include "tiling/platform/platform_ascendc.h"

namespace AtbOps {
constexpr uint64_t BEST_L1_PARTA = 256 * 1024;
constexpr uint64_t BEST_L1_PARTB = 128 * 1024;
constexpr int32_t BEST_BASEN = 256;
constexpr int32_t MAX_BASEM = 256;
constexpr uint32_t DATATYPE_SIZE = 2;
constexpr uint32_t FP32_DATATYPE_SIZE = 4;
constexpr uint64_t TILING_KEY_DEFAULT_FP16 = 0;
constexpr uint64_t TILING_KEY_DEFAULT_BF16 = 1;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint64_t DOUBLE_BUFFER_STEPKA_STEPKB = 2;
constexpr int32_t MAX_TENSOR_CONT = 128;
constexpr uint32_t INDEX_GROUP_LIST = 2;

static inline uint32_t SixteenAlign(uint32_t a, bool up = false)
{
    if (up) {
        a += 15; // 15: 16 bytes up-align
    }
    return a & ~15; // ~15: 16 bytes down-align
};

struct GMMCompileInfo {
    uint32_t aicNum;
    uint32_t aivNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l2Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;
    Mki::PlatformType socVersion;
    std::string socVersionStr = "";
};

const std::map<Mki::PlatformType, platform_ascendc::SocVersion> PLANTFORM_TYPE_MAP = {
    {Mki::PlatformType::ASCEND_310P, platform_ascendc::SocVersion::ASCEND310P},
    {Mki::PlatformType::ASCEND_910A, platform_ascendc::SocVersion::ASCEND910},
    {Mki::PlatformType::ASCEND_910B, platform_ascendc::SocVersion::ASCEND910B}};

class GmmAddTiling {
public:
    GmmTilingData tilingData;
    Mki::Status Init(const Mki::LaunchParam &launchParam);
    Mki::Status RunFusionKernelTiling(Mki::KernelInfo &kernelInfo);

protected:
    Mki::Status CalMmTiling();
    Mki::Status SetMmTiling();
    void SetTilingKey(Mki::KernelInfo &kernelInfo) const;
    Mki::Status GetAttrs(const Mki::LaunchParam &launchParam);
    int64_t GetBs(const Mki::SVector<int64_t> xShape) const;
    Mki::Status PrepareTilingData(const Mki::LaunchParam &launchParam);
    Mki::Status GetTensorShapeSplitK(const Mki::SVector<int64_t> &xShape, const Mki::SVector<int64_t> &wShape);
    Mki::Status SplitKSingleXSingleWeightSingleY(const Mki::SVector<int64_t> &xShape,
                                                 const Mki::SVector<int64_t> &wShape);

private:
    int32_t mList_[MAX_TENSOR_CONT];
    int32_t kList_[MAX_TENSOR_CONT];
    int32_t nList_[MAX_TENSOR_CONT];
    int64_t maxM_;
    int64_t maxN_;
    int64_t maxK_;
    int32_t minK_;
    int32_t baseM_;
    int32_t baseN_;
    int32_t baseK_;
    uint64_t ubSize_;
    uint32_t mmDataTypeSize_;
    uint32_t ubDivideBlkNum_;
    uint32_t ubIoBlkNum_;
    uint32_t ubBlockAlign_;
    uint64_t workspacesSize_; // for antiquant
    uint32_t groupNum_;
    bool transposeWeight_;
    bool transposeX_;
    bool isSingleWeight_;
    bool isSingleX_;
    bool isSingleY_;
    int32_t groupType_;
    int32_t xKDim_;
    int32_t weightNDim_;
    int32_t xDimNum_;
    bool antiquantPerformance_;
    int32_t actType_;
    Mki::TensorDType xDtype_;
    Mki::TensorDType weightDtype_;
    matmul_tiling::CubeFormat wFormat_;
    int32_t nzFactor_; // for weight nz format
};

Mki::Status GmmAddTiling::PrepareTilingData(const Mki::LaunchParam &launchParam)
{
    // 获取transpose和groupType
    MKI_CHECK(GetAttrs(launchParam).Ok(), "GetAttrs failed", return Mki::Status::FailStatus(Mki::ERROR_ATTR_NOT_EXIST));
    mmDataTypeSize_ = DATATYPE_SIZE;
    MKI_CHECK(mmDataTypeSize_ != 0, "GMM get dtype size is 0.",
              return Mki::Status::FailStatus(Mki::ERROR_ATTR_INVALID_TYPE));
    Mki::SVector<int64_t> xShape = launchParam.GetInTensor(0).desc.dims;
    groupNum_ = static_cast<uint32_t>(launchParam.GetInTensor(INDEX_GROUP_LIST).Numel());
    xDimNum_ = static_cast<int32_t>(xShape.size());
    xKDim_ = transposeX_ ? 0 : xDimNum_ - 1; // 0：x转置情况下0维为k；-1：x非转置最后一维为k

    Mki::SVector<int64_t> wShape = launchParam.GetInTensor(1).desc.dims;
    uint32_t wDimNum = wShape.size();
    MKI_CHECK(wDimNum >= 2, "wDimNum < 2, which is: " << wDimNum,
              return Mki::Status::FailStatus(Mki::ERROR_INVALID_VALUE));
    weightNDim_ = transposeWeight_ ? static_cast<int32_t>(wDimNum - 2) : // -2：w转置情况下导数第2维为n
                static_cast<int32_t>(wDimNum - 1);  // -1：w非转置最后一维为n
    nzFactor_ = 1;                                  // init
    isSingleWeight_ = true;
    isSingleX_ = true;
    isSingleY_ = true;
    tilingData.gmmBaseParams.singleWeight = static_cast<uint32_t>(isSingleWeight_);
    tilingData.gmmBaseParams.singleX = static_cast<uint32_t>(isSingleX_);
    tilingData.gmmBaseParams.singleY = static_cast<uint32_t>(isSingleY_);
    return GetTensorShapeSplitK(xShape, wShape);
}

Mki::Status GmmAddTiling::GetTensorShapeSplitK(const Mki::SVector<int64_t> &xShape, const Mki::SVector<int64_t> &wShape)
{
    if (isSingleX_ && isSingleWeight_ && isSingleY_) { // 切M，单单单
        return SplitKSingleXSingleWeightSingleY(xShape, wShape);
    }
    MKI_LOG_INFO << " gmmTiling: not support groupType_= " << groupType_ << " isSingleWeight_= " << isSingleWeight_
                 << "isSingleX_= " << isSingleX_ << " isSingleY_ " << isSingleY_;
    return Mki::Status::FailStatus(Mki::ERROR_INFERSHAPE_ERROR);
}

Mki::Status GmmAddTiling::SplitKSingleXSingleWeightSingleY(const Mki::SVector<int64_t> &xShape,
                                                           const Mki::SVector<int64_t> &wShape)
{
    MKI_LOG_INFO << ">>> xShape: ";
    for (auto x : xShape) {
        MKI_LOG_INFO << ">>> " << x;
    }
    MKI_LOG_INFO << ">>> wShape: ";
    for (auto w : wShape) {
        MKI_LOG_INFO << ">>> " << w;
    }
    int64_t m = GetBs(xShape);
    int64_t k = xShape.at(xKDim_);
    int64_t n = wShape.at(weightNDim_) * static_cast<int64_t>(nzFactor_);
    kList_[0] = -1;
    nList_[0] = static_cast<int32_t>(n);
    mList_[0] = static_cast<int32_t>(m);

    maxM_ = m;
    maxK_ = k;
    maxN_ = n;
    uint32_t numInOneBlk = AscendC::ONE_BLK_SIZE / mmDataTypeSize_;
    int64_t maxMKN = INT_MAX / numInOneBlk * numInOneBlk;
    MKI_CHECK(maxM_ <= maxMKN && maxN_ <= maxMKN && maxK_ <= maxMKN,
              "32B-aligned m, n or k axis is out of range int32!",
              return Mki::Status::FailStatus(Mki::ERROR_INFERSHAPE_ERROR));
    return Mki::Status::OkStatus();
}

Mki::Status GmmAddTiling::Init(const Mki::LaunchParam &launchParam)
{
    auto ret1 = memset_s(mList_, sizeof(mList_), 0, sizeof(int32_t) * MAX_TENSOR_CONT);
    auto ret2 = memset_s(kList_, sizeof(kList_), 0, sizeof(int32_t) * MAX_TENSOR_CONT);
    auto ret3 = memset_s(nList_, sizeof(nList_), 0, sizeof(int32_t) * MAX_TENSOR_CONT);
    MKI_CHECK(ret1 == 0 && ret2 == 0 && ret3 == 0, "memset Lists failed!",
              return Mki::Status::FailStatus(Mki::ERROR_INFERSHAPE_ERROR));
    workspacesSize_ = 0; // init
    groupNum_ = 0;       // init
    maxM_ = 0;           // init
    maxK_ = 0;           // init
    maxN_ = 0;           // init
    minK_ = INT32_MAX;   // init
    xDtype_ = Mki::TENSOR_DTYPE_UNDEFINED;
    weightDtype_ = Mki::TENSOR_DTYPE_UNDEFINED;

    MKI_CHECK(PrepareTilingData(launchParam).Ok(), "GMM PrepareTilingData failed.",
              return Mki::Status::FailStatus(Mki::ERROR_INFERSHAPE_ERROR));
    for (uint32_t i = 0; i < groupNum_; ++i) {
        tilingData.gmmArray.mList[i] = mList_[i];
        tilingData.gmmArray.kList[i] = kList_[i];
        tilingData.gmmArray.nList[i] = nList_[i];
    }
    tilingData.gmmBaseParams.groupNum = groupNum_;
    MKI_LOG_INFO << "gmmTiling: groupNum_ is" << groupNum_ << "maxM_ is" << maxM_ << ", maxK_ is " << maxK_
                 << ", maxN_ is " << maxN_ << ".";
    return Mki::Status::OkStatus();
}

void PrintTilingData(const Mki::KernelInfo &kernelInfo)
{
    auto *gmmTilingData = reinterpret_cast<GmmTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_LOG_INFO << ">> gmmBaseParams.groupNum: " << gmmTilingData->gmmBaseParams.groupNum;
    MKI_LOG_INFO << ">> gmmBaseParams.coreNum: " << gmmTilingData->gmmBaseParams.coreNum;
    MKI_LOG_INFO << ">> gmmBaseParams.activeType: " << gmmTilingData->gmmBaseParams.activeType;
    MKI_LOG_INFO << ">> gmmBaseParams.ubBaseK: " << gmmTilingData->gmmBaseParams.ubBaseK;
    MKI_LOG_INFO << ">> gmmBaseParams.ubBaseN: " << gmmTilingData->gmmBaseParams.ubBaseN;
    MKI_LOG_INFO << ">> gmmBaseParams.ubCalSize: " << gmmTilingData->gmmBaseParams.ubCalSize;
    MKI_LOG_INFO << ">> gmmBaseParams.ubRestBytes: " << gmmTilingData->gmmBaseParams.ubRestBytes;
    MKI_LOG_INFO << ">> gmmBaseParams.singleWeight: " << gmmTilingData->gmmBaseParams.singleWeight;
    MKI_LOG_INFO << ">> gmmBaseParams.singleX: " << gmmTilingData->gmmBaseParams.singleX;
    MKI_LOG_INFO << ">> gmmBaseParams.singleY: " << gmmTilingData->gmmBaseParams.singleY;
    MKI_LOG_INFO << ">> gmmBaseParams.groupType: " << gmmTilingData->gmmBaseParams.groupType;
    MKI_LOG_INFO << ">> gmmBaseParams.singleN: " << gmmTilingData->gmmBaseParams.singleN;
    MKI_LOG_INFO << ">> gmmBaseParams.groupListType: " << gmmTilingData->gmmBaseParams.groupListType;
    MKI_LOG_INFO << ">> gmmBaseParams.workspaceSize: " << gmmTilingData->gmmBaseParams.workspaceSize;
    for (uint32_t i = 0; i < 1; ++i) {
        MKI_LOG_INFO << ">>> gmmArray.mList[" << i << "]: " << gmmTilingData->gmmArray.mList[i];
        MKI_LOG_INFO << ">>> gmmArray.kList[" << i << "]: " << gmmTilingData->gmmArray.kList[i];
        MKI_LOG_INFO << ">>> gmmArray.nList[" << i << "]: " << gmmTilingData->gmmArray.nList[i];
    }
}

Mki::Status GmmAddTiling::RunFusionKernelTiling(Mki::KernelInfo &kernelInfo)
{
    MKI_LOG_INFO << "Begin Run GMM Tiling";
    ubSize_ = Mki::PlatformInfo::Instance().GetUbSize();
    const uint32_t &aicNum = Mki::PlatformInfo::Instance().GetCoreNum(Mki::CoreType::CORE_TYPE_CUBE);
    MKI_CHECK(CalMmTiling().Ok(), "GMM CalMmTiling failed",
              return Mki::Status::FailStatus(Mki::ERROR_LAUNCH_KERNEL_ERROR));

    MKI_CHECK(SetMmTiling().Ok(), "GMM SetMmTiling failed",
              return Mki::Status::FailStatus(Mki::ERROR_LAUNCH_KERNEL_ERROR));
    tilingData.mmTilingData.set_usedCoreNum(aicNum); // usedCoreNum is ai_core num
    tilingData.gmmBaseParams.coreNum = aicNum;       // ai cube number
    tilingData.gmmBaseParams.singleN = 0;            // 0 is the default value
    tilingData.gmmBaseParams.workspaceSize = 0;
    SetTilingKey(kernelInfo);
    MKI_LOG_INFO << "End Run GMM Tiling";
    MKI_LOG_INFO << "copy result to kernel info";
    auto *gmmTilingData = reinterpret_cast<GmmTilingData *>(kernelInfo.GetTilingHostAddr());
    gmmTilingData->gmmBaseParams.groupNum = tilingData.gmmBaseParams.groupNum;
    gmmTilingData->gmmBaseParams.coreNum = tilingData.gmmBaseParams.coreNum;
    gmmTilingData->gmmBaseParams.activeType = tilingData.gmmBaseParams.activeType;
    gmmTilingData->gmmBaseParams.ubBaseK = tilingData.gmmBaseParams.ubBaseK;
    gmmTilingData->gmmBaseParams.ubBaseN = tilingData.gmmBaseParams.ubBaseN;
    gmmTilingData->gmmBaseParams.ubCalSize = tilingData.gmmBaseParams.ubCalSize;
    gmmTilingData->gmmBaseParams.ubRestBytes = tilingData.gmmBaseParams.ubRestBytes;
    gmmTilingData->gmmBaseParams.singleWeight = tilingData.gmmBaseParams.singleWeight;
    gmmTilingData->gmmBaseParams.singleX = tilingData.gmmBaseParams.singleX;
    gmmTilingData->gmmBaseParams.singleY = tilingData.gmmBaseParams.singleY;
    gmmTilingData->gmmBaseParams.groupType = tilingData.gmmBaseParams.groupType;
    gmmTilingData->gmmBaseParams.singleN = tilingData.gmmBaseParams.singleN;
    gmmTilingData->gmmBaseParams.groupListType = tilingData.gmmBaseParams.groupListType;
    gmmTilingData->gmmBaseParams.workspaceSize = tilingData.gmmBaseParams.workspaceSize;
    for (uint32_t i = 0; i < groupNum_; ++i) {
        gmmTilingData->gmmArray.mList[i] = tilingData.gmmArray.mList[i];
        gmmTilingData->gmmArray.kList[i] = tilingData.gmmArray.kList[i];
        gmmTilingData->gmmArray.nList[i] = tilingData.gmmArray.nList[i];
    }
    tilingData.mmTilingData.SaveToBuffer(reinterpret_cast<uint8_t *>(&gmmTilingData->mmTilingData),
                                         tilingData.mmTilingData.GetDataSize());
    kernelInfo.SetBlockDim(tilingData.gmmBaseParams.coreNum);
    PrintTilingData(kernelInfo);
    return Mki::Status::OkStatus();
}

int64_t GmmAddTiling::GetBs(const Mki::SVector<int64_t> xShape) const
{
    // calc bs
    int64_t bs = 0; // init bs
    if (transposeX_) {
        bs = xShape.at(1); // x shape is [k, m] if x is transpose_
    } else {
        if (groupType_ == -1) { // -1：不分组场景，可能存在多维乘积为bs的情况
            bs = xShape.at(0);  // 0: x first dim
            size_t bsDimNum = xDimNum_ >= 1 ? xDimNum_ - 1 : 0; // 1: x last dim k, the other dimensions are bs
            for (size_t i = 1; i < bsDimNum; i++) {
                bs *= xShape.at(i);
            }
        } else {
            bs = xShape.at(0); // 分组场景下，x的shape为[m,k]，0为m轴
        }
    }
    return bs;
}

void GmmAddTiling::SetTilingKey(Mki::KernelInfo &kernelInfo) const
{
    if (weightDtype_ == Mki::TENSOR_DTYPE_FLOAT16) {
        kernelInfo.SetTilingId(TILING_KEY_DEFAULT_FP16);
    } else if (weightDtype_ == Mki::TENSOR_DTYPE_BF16) {
        kernelInfo.SetTilingId(TILING_KEY_DEFAULT_BF16);
    }
}

Mki::Status GmmAddTiling::GetAttrs(const Mki::LaunchParam &launchParam)
{
    auto &attr = Mki::AnyCast<AtbOps::OpParam::GmmAdd>(launchParam.GetParam());
    transposeWeight_ = attr.transposeB;
    transposeX_ = attr.transposeA;
    xDtype_ = launchParam.GetInTensor(1).desc.dtype;
    weightDtype_ = launchParam.GetInTensor(1).desc.dtype;
    wFormat_ = matmul_tiling::CubeFormat::ND;
    return Mki::Status::OkStatus();
}

static void InitPlatformInfo(matmul_tiling::PlatformInfo &platformInfo)
{
    platformInfo.socVersion = PLANTFORM_TYPE_MAP.at(Mki::PlatformInfo::Instance().GetPlatformType());
    platformInfo.l1Size = Mki::PlatformInfo::Instance().GetL1Size();
    platformInfo.l0CSize = Mki::PlatformInfo::Instance().GetL0CSize();
    platformInfo.ubSize = Mki::PlatformInfo::Instance().GetUbSize();
    platformInfo.l0ASize = Mki::PlatformInfo::Instance().GetL0ASize();
    platformInfo.l0BSize = Mki::PlatformInfo::Instance().GetL0BSize();
}

Mki::Status GmmAddTiling::SetMmTiling()
{
    matmul_tiling::DataType matmulDtype = static_cast<matmul_tiling::DataType>(xDtype_);
    matmul_tiling::PlatformInfo platformInfo;
    InitPlatformInfo(platformInfo);
    matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);

    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, false);
    mm.SetBType(matmul_tiling::TPosition::GM, wFormat_, matmulDtype, false);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm.SetBias(false);
    mm.SetOrgShape(maxM_, maxN_, maxK_);
    mm.SetShape(maxM_, baseN_, maxK_);
    mm.SetFixSplit(baseM_, baseN_, baseK_);
    mm.SetBufferSpace(Mki::PlatformInfo::Instance().GetL1Size(), Mki::PlatformInfo::Instance().GetL0CSize(), ubSize_);
    MKI_CHECK(mm.GetTiling(tilingData.mmTilingData) != -1, "matmul getTiling failed.",
              return Mki::Status::FailStatus(Mki::ERROR_LAUNCH_KERNEL_ERROR));
    // 计算开启dublebuffer之后搬运至L1是所需的参数
    uint64_t productMK = static_cast<uint64_t>(baseM_) * static_cast<uint64_t>(baseK_);
    uint64_t productNK = static_cast<uint64_t>(baseN_) * static_cast<uint64_t>(baseK_);
    MKI_CHECK(productMK <= std::numeric_limits<uint32_t>::max(),
              "product of M and K > uint32_t max value, is: productMK" << productMK,
              return Mki::Status::FailStatus(Mki::ERROR_INVALID_VALUE));
    MKI_CHECK(productNK <= std::numeric_limits<uint32_t>::max(),
              "product of N and K > uint32_t max value, is: productNK" << productNK,
              return Mki::Status::FailStatus(Mki::ERROR_INVALID_VALUE));
    uint32_t mmStepKa = (BEST_L1_PARTB >> 1) / (static_cast<uint32_t>(productMK) * mmDataTypeSize_);
    uint32_t mmStepKb = (BEST_L1_PARTA >> 1) / (static_cast<uint32_t>(productNK) * mmDataTypeSize_);

    if (mmStepKa > mmStepKb) {
        mmStepKa = mmStepKa / mmStepKb * mmStepKb;
    } else if (mmStepKa < mmStepKb) {
        mmStepKb = mmStepKb / mmStepKa * mmStepKa;
    }
    constexpr uint32_t stepM = 1; // 1: stepM set fixed value 1
    constexpr uint32_t stepN = 1; // 1: stepN set fixed value 1
    uint32_t mmDepthA1 = mmStepKa * DOUBLE_BUFFER_STEPKA_STEPKB * stepM;
    uint32_t mmDepthB1 = mmStepKb * DOUBLE_BUFFER_STEPKA_STEPKB * stepN;
    tilingData.mmTilingData.set_shareMode(0);
    tilingData.mmTilingData.set_dbL0C(1);           // disable double buffer for L0C
    tilingData.mmTilingData.set_baseM(baseM_);      // set precomputed baseM
    tilingData.mmTilingData.set_baseN(baseN_);      // set precomputed baseN
    tilingData.mmTilingData.set_baseK(baseK_);      // set precomputed baseK
    tilingData.mmTilingData.set_stepKa(mmStepKa);   // set precomputed mmStepKa
    tilingData.mmTilingData.set_depthA1(mmDepthA1); // set precomputed mmDepthA1
    tilingData.mmTilingData.set_stepKb(mmStepKb);   // set precomputed mmStepKb
    tilingData.mmTilingData.set_depthB1(mmDepthB1); // set precomputed mmDepthB1
    tilingData.mmTilingData.set_stepM(stepM);       // set precomputed stepM
    tilingData.mmTilingData.set_stepN(stepN);       // set precomputed stepN
    return Mki::Status::OkStatus();
}

Mki::Status GmmAddTiling::CalMmTiling()
{
    baseN_ = BEST_BASEN;
    // 基于使能double buffer的L0B内存计算baseK
    baseK_ = static_cast<int32_t>((Mki::PlatformInfo::Instance().GetL0BSize() / DOUBLE_BUFFER_L0A_L0B) /
                                  static_cast<uint64_t>(baseN_ * static_cast<int32_t>(mmDataTypeSize_)));
    baseK_ = static_cast<int32_t>(SixteenAlign(baseK_));
    MKI_CHECK(baseK_ > 0, "baseK_ cannot be 0.", return Mki::Status::FailStatus(Mki::ERROR_LAUNCH_KERNEL_ERROR));
    // 基于使能double buffer的L0A内存和L0C内存计算baseM(cube)
    uint32_t maxBaseM = static_cast<uint32_t>(Mki::PlatformInfo::Instance().GetL0CSize() /
                                              static_cast<uint64_t>(baseN_ * FP32_DATATYPE_SIZE));
    baseM_ = std::min<uint32_t>(
        (Mki::PlatformInfo::Instance().GetL0ASize() / DOUBLE_BUFFER_L0A_L0B) / (baseK_ * mmDataTypeSize_), maxBaseM);
    if (baseM_ > maxM_) {
        baseM_ = static_cast<int32_t>(SixteenAlign(maxM_, true));
    } else {
        baseM_ = static_cast<int32_t>(SixteenAlign(baseM_));
    }
    if (baseM_ > MAX_BASEM) {
        baseM_ = MAX_BASEM;
    }
    MKI_CHECK(baseM_ > 0, "baseM_ cannot be 0.", return Mki::Status::FailStatus(Mki::ERROR_LAUNCH_KERNEL_ERROR));

    return Mki::Status::OkStatus();
}

static Mki::Status TilingGMM(Mki::KernelInfo &kernelInfo, const Mki::LaunchParam &launchParam)
{
    static thread_local GmmAddTiling gmmAddTiling; // thread_local variable makes performance better
    MKI_CHECK(gmmAddTiling.Init(launchParam).Ok(), "Init GmmAddTiling failed",
              return Mki::Status::FailStatus(Mki::ERROR_ATTR_INVALID_TYPE));
    return gmmAddTiling.RunFusionKernelTiling(kernelInfo);
}

Mki::Status GetGmmAddTilingData(const Mki::LaunchParam &launchParam, Mki::KernelInfo &kernelInfo)
{
    MKI_CHECK(TilingGMM(kernelInfo, launchParam).Ok(), "tiling run failed",
              return Mki::Status::FailStatus(Mki::ERROR_LAUNCH_KERNEL_ERROR));
    return Mki::Status::OkStatus();
}
} // namespace AtbOps