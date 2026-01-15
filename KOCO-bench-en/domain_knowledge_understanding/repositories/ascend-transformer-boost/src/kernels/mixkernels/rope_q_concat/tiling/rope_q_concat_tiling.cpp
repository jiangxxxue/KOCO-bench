/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/const/op_const.h>
#include "atbops/params/params.h"
#include "tiling_data.h"
#include "rope_q_concat_tiling.h"
constexpr uint32_t MAX_PROCESS_NUM = 192 * 1024 / 16;
constexpr uint32_t TILING_BF16 = 22;
constexpr uint32_t TILING_FP32 = 21;
constexpr uint32_t TILING_FP16 = 20;
constexpr uint32_t REMAIN_TILING_SIZE = 14 * 8;
namespace AtbOps {


Status TilingKeyChose(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto platformType = PlatformInfo::Instance().GetPlatformType();
    if (platformType != PlatformType::ASCEND_910B) {
        MKI_LOG(ERROR) << "Only supports 800I A2/A3";
        return Status::FailStatus(ERROR_INVALID_VALUE);
    }
    if (launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16) {
        kernelInfo.SetTilingId(TILING_BF16);
    } else if (launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT) {
        kernelInfo.SetTilingId(TILING_FP32);
    } else {
        kernelInfo.SetTilingId(TILING_FP16);
    }
    return Status::OkStatus();
}

Status RopeNdProcess(const LaunchParam &launchParam, RopeQConcatTilingData &tilingDataPtr)
{
    auto &inTensor0 = launchParam.GetInTensor(0).desc;
    auto &inTensor1 = launchParam.GetInTensor(DIM_1).desc;
    auto &inTensor3 = launchParam.GetInTensor(DIM_3).desc;

    uint32_t ntokens = inTensor0.dims[0];
    uint32_t hiddenSizeQ = inTensor0.dims[DIM_1];

    auto cosSize = inTensor1.dims.size();
    uint32_t headDim = inTensor1.dims[cosSize - 1];
    uint32_t headNumQ = hiddenSizeQ / headDim;
    uint32_t concatSize = inTensor3.dims[DIM_2];

    // 当前场景只支持rotaryCoeff = 2的情况
    tilingDataPtr.rotaryCoeff = 2;
    uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    auto maxUbSize = static_cast<uint32_t>(PlatformInfo::Instance().GetUbSize()) - REMAIN_TILING_SIZE;

    uint32_t allHeadNum = ntokens * headNumQ;

    uint32_t tempCore = (allHeadNum + maxCore - 1) / maxCore;
    uint32_t realCore = (allHeadNum + tempCore - 1) / tempCore; // 实际运算核数
    uint32_t nlCoreRun = (allHeadNum + realCore - 1) / realCore; // 前核运算head数
    uint32_t lCoreRun =  allHeadNum - (realCore - 1) * nlCoreRun; // 尾核运算head数

    auto qDtype = launchParam.GetInTensor(0).desc.dtype;
    uint32_t dataTypeSize = qDtype == TENSOR_DTYPE_FLOAT ? 4 : 2;
    // q 4+dataTypeSize、reverseq 4、neg 4、sin 4+dataTypeSize、cos 4+dataTypeSize concat dataTypeSize
    uint32_t allSize = headDim * (3 * (4 + dataTypeSize) + 2 * 4) + concatSize * dataTypeSize; // rope内部升精度计算
    uint32_t maxNPerLoopForUb = maxUbSize / allSize;  // ub每次能载入最大行数（包括所有计算数据）
    MKI_CHECK(maxNPerLoopForUb > 0, "The input vector is too large. It is not supported currently.",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    uint32_t preCoreLoopTime = (nlCoreRun + maxNPerLoopForUb - 1) / maxNPerLoopForUb;  // 前核循环次数
    uint32_t preCoreLoopNLast = nlCoreRun - (preCoreLoopTime - 1) * maxNPerLoopForUb;  // 前核最后一批处理数据行数
    uint32_t lastCoreLoopTime = (lCoreRun + maxNPerLoopForUb - 1) / maxNPerLoopForUb;  // 尾核循环次数
    uint32_t lastCoreLoopNLast = lCoreRun - (lastCoreLoopTime - 1) * maxNPerLoopForUb;  // 尾核最后一批处理数据行数
    tilingDataPtr.hiddenSizeQ = hiddenSizeQ;
    tilingDataPtr.headNumQ = headNumQ;
    tilingDataPtr.headDim = headDim;
    tilingDataPtr.concatSize = concatSize;
    tilingDataPtr.ntokens = ntokens;
    tilingDataPtr.realCore = realCore;
    tilingDataPtr.nlCoreRun = nlCoreRun;
    tilingDataPtr.lCoreRun = lCoreRun;
    tilingDataPtr.maxNPerLoopForUb = maxNPerLoopForUb;
    tilingDataPtr.preCoreLoopTime = preCoreLoopTime;
    tilingDataPtr.preCoreLoopNLast = preCoreLoopNLast;
    tilingDataPtr.lastCoreLoopTime = lastCoreLoopTime;
    tilingDataPtr.lastCoreLoopNLast = lastCoreLoopNLast;
    return Status::OkStatus();
}

Status RopeQConcatTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    MKI_LOG(INFO) << " RopeQConcatTiling START ";
    uint8_t *tilingHost = kernelInfo.GetTilingHostAddr();
    RopeQConcatTilingData *tilingDataPtr = reinterpret_cast<RopeQConcatTilingData *>(tilingHost);
    if (tilingDataPtr == nullptr) {
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr should not be empty");
    }
    auto ret = TilingKeyChose(launchParam, kernelInfo);
    if (!ret.Ok()) {
        return Status::FailStatus(ERROR_INVALID_VALUE);
    }
    auto retProcess = RopeNdProcess(launchParam, *tilingDataPtr);
    if (!retProcess.Ok()) {
        return Status::FailStatus(ERROR_INVALID_VALUE);
    }

    MKI_LOG(INFO) << "RopeQConcat " << "\n"
        << "hiddenSizeQ " << tilingDataPtr->hiddenSizeQ <<"\n"
        << "headNumQ  " << tilingDataPtr->headNumQ  <<"\n"
        << "headDim " << tilingDataPtr->headDim <<"\n"
        << "concatSize " << tilingDataPtr->concatSize <<"\n"
        << "ntokens " << tilingDataPtr->ntokens <<"\n"
        << "realCore  " << tilingDataPtr->realCore  <<"\n"
        << "nlCoreRun " << tilingDataPtr->nlCoreRun <<"\n"
        << "lCoreRun  " << tilingDataPtr->lCoreRun  <<"\n"
        << "maxNPerLoopForUb " << tilingDataPtr->maxNPerLoopForUb <<"\n"
        << "preCoreLoopTime " << tilingDataPtr->preCoreLoopTime <<"\n"
        << "preCoreLoopNLast " << tilingDataPtr->preCoreLoopNLast <<"\n"
        << "lastCoreLoopTime " << tilingDataPtr->lastCoreLoopTime <<"\n"
        << "lastCoreLoopNLast " << tilingDataPtr->lastCoreLoopNLast <<"\n"
        << "rotaryCoeff " <<tilingDataPtr->rotaryCoeff;

    kernelInfo.SetBlockDim(tilingDataPtr->realCore);
    return Status::OkStatus();
}
} // namespace AtbOps
