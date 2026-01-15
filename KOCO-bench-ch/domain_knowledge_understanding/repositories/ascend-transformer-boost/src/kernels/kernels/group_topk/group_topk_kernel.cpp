/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/base/kernel_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/math/math.h>
#include "asdops/params/params.h"
#include "kernels/group_topk/tiling/group_topk_tiling.h"
#include "kernels/group_topk/tiling/tiling_data.h"

namespace AsdOps {

constexpr int32_t MAX_EXPERT = 1024;
constexpr int32_t MAX_GROUPNUM = 256;
constexpr int32_t MAX_KINNER = 32;

class GroupTopkKernel : public KernelBase {
public:
    explicit GroupTopkKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::GroupTopk), "GroupTopk valid: param type invalid",
                  return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "GroupTopk valid: input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "GroupTopk valid: output num invalid", return false);
        const SVector<int64_t> &varDims = launchParam.GetInTensor(0).desc.dims;
        const SVector<int64_t> &indicesDims = launchParam.GetInTensor(1).desc.dims;
        const int32_t varDimsSize = static_cast<int32_t>(varDims.size());
        const int32_t indicesDimsSize = static_cast<int32_t>(indicesDims.size());
        MKI_CHECK(varDimsSize == 2, "GroupTopk valid: var dims invalid", return false);
        MKI_CHECK(indicesDimsSize == 1, "GroupTopk valid: indices dims invalid", return false);
        const int64_t tokenNum = varDims.at(0);
        const int64_t expertNum = varDims.at(1);
        const int64_t indicesLength = indicesDims.at(0);
        MKI_CHECK(tokenNum > 0, "GroupTopk valid: var shape invalid", return false);
        MKI_CHECK(expertNum > 0 && expertNum <= MAX_EXPERT, "GroupTopk valid: var shape invalid", return false);
        MKI_CHECK(indicesLength == MAX_EXPERT, "GroupTopk valid: indices shape invalid, must be " << MAX_EXPERT,
                  return false);
        const auto param = AnyCast<OpParam::GroupTopk>(launchParam.GetParam());
        MKI_CHECK(param.groupNum > 0 && param.groupNum <= expertNum && expertNum % param.groupNum == 0,
                  "GroupTopk valid: param groupNum invalid", return false);
        MKI_CHECK(param.kInner > 0 && param.kInner <= expertNum / param.groupNum,
                  "GroupTopk valid: param kInner invalid", return false);
        MKI_CHECK(param.k > 0 && param.k <= param.groupNum, "GroupTopk valid: param k invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(GroupTopkTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override { return GroupTopkTiling(launchParam, kernelInfo_); }
};

REG_KERNEL_BASE(GroupTopkKernel);
} // namespace AsdOps