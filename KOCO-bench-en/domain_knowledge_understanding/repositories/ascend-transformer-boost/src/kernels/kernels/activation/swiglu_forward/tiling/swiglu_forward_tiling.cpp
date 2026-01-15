/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "swiglu_forward_tiling.h"
#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/params.h"
#include "kernels/activation/swiglu_forward/tiling/swiglu_forward_backward_common_tiling.h"
#include "tiling_data.h"

namespace AsdOps {
using namespace Mki;
Status SwiGluForwardTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    MKI_LOG(INFO) << "----- [ Enter SwiGluForwardTiling ] -----";
    uint32_t totalCore = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    uint32_t ubSize = PlatformInfo::Instance().GetUbSize();
    MKI_LOG(INFO) << "----- ubSize -----" << ubSize;
    if (totalCore > MAX_CORE_NUMBER || ubSize <= UB_RESERVED_BUFF) {
        return Status::FailStatus(ERROR_INVALID_VALUE, "Compile Info is invalid, coreNum: " +
            std::to_string(totalCore) + ", ubSize: " + std::to_string(ubSize));
    }

    SwiGluForwardTilingData *tilingData = reinterpret_cast<SwiGluForwardTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingData != nullptr, "tilingData should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    int32_t dataType = launchParam.GetInTensor(0).desc.dtype;
    const SVector<int64_t> &inShape = launchParam.GetInTensor(0).desc.dims;
    MKI_CHECK(!inShape.empty(), "xshape should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "xshape should not be empty"));
    auto param = AnyCast<OpParam::Activation>(launchParam.GetParam());
    int32_t inDim = param.dim;
    if (inDim < 0) {
        inDim += static_cast<int32_t>(inShape.size());
    }
    MKI_CHECK(inDim >= 0 && inDim < static_cast<int32_t>(inShape.size()),
                 "The value of attr [dim] must be in the range [-" << inShape.size() <<", " <<
                 inShape.size() - 1 << "], but got [" << inDim << "].",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "input dim error"));

    if (!SetTotalShape<SwiGluForwardTilingData *>(inShape, inDim, tilingData)) {
        return Status::FailStatus(ERROR_INVALID_VALUE, "graph failed");
    };

    // 计算每核处理的数据情况
    uint64_t dataNumPerCore = CalDataPerCore<SwiGluForwardTilingData *>(totalCore, dataType, tilingData, kernelInfo);
    // 处理每核内的循环情况
    bool isGrad = false; // 正向
    bool isSuccess = CalDataPerLoop<SwiGluForwardTilingData *>(ubSize, dataNumPerCore, dataType, tilingData, isGrad);
    if (!isSuccess) {
        return Status::FailStatus(ERROR_INVALID_VALUE, "input error");
    }
    kernelInfo.SetTilingId(dataType);
    MKI_LOG(INFO) << "----- tilingKey -----" + std::to_string(dataType);
    return Status::OkStatus();
}
} // namespace AsdOps