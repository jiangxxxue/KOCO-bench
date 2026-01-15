/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mki/base/kernel_base.h"
#include "mki_loader/op_register.h"
#include "mki/utils/log/log.h"
#include "mki/utils/const/op_const.h"
#include "atbops/params/params.h"
#include "tiling/tiling_data.h"
#include "tiling/moe_gmm_tiling.h"
#include "mixkernels/utils/common.h"

constexpr uint64_t DIM_SIZE_2 = 2;
constexpr uint64_t DIM_SIZE_3 = 3;
constexpr uint64_t DIM_SIZE_4 = 4;

namespace AtbOps {
using namespace Mki;
class MoeGmmKernel : public KernelBase {
public:
    explicit MoeGmmKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MoeGmm), "invalid op param.", return false);
        auto attr = AnyCast<OpParam::MoeGmm>(launchParam.GetParam());
        MKI_CHECK(attr.hiddenSize.at(0) % alignSize == 0 && attr.hiddenSize.at(1) % alignSize == 0,
                  "Missalign hidden size.", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == kNumInTensor, "inTensor count invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == kNumOutTensor, "outTensor count invalid", return false);

        const auto &inTensorDesc0 = launchParam.GetInTensor(0).desc;
        const auto &inTensorDesc1 = launchParam.GetInTensor(1).desc;
        const auto &outTensorDesc = launchParam.GetOutTensor(0).desc;

        MKI_CHECK(inTensorDesc0.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(inTensorDesc0.dtype == TENSOR_DTYPE_FLOAT16 || inTensorDesc0.dtype == TENSOR_DTYPE_BF16,
                  "tensor dtype invalid", return false);
        MKI_CHECK(inTensorDesc0.dims.size() == DIM_SIZE_2, "tensor dims invalid", return false);

        MKI_CHECK(inTensorDesc1.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(inTensorDesc1.dtype == TENSOR_DTYPE_FLOAT16 || inTensorDesc1.dtype == TENSOR_DTYPE_BF16,
                  "tensor dtype invalid", return false);
        MKI_CHECK(inTensorDesc1.dims.size() == DIM_SIZE_3, "tensor dims invalid", return false);

        MKI_CHECK(outTensorDesc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(outTensorDesc.dims.size() == DIM_SIZE_2, "tensor dims invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(MoeGmmTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        kernelInfo_.SetHwsyncIdx(0);
        return GetMoeGmmTiling(launchParam, kernelInfo_);
    }

private:
    const uint64_t kNumInTensor = 4;
    const uint64_t kNumOutTensor = 1;
    const uint32_t alignSize = 32;
};

class MoeGmmW8a8Kernel : public KernelBase {
public:
    explicit MoeGmmW8a8Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MoeGmm), "invalid op param.", return false);
        auto attr = AnyCast<OpParam::MoeGmm>(launchParam.GetParam());
        MKI_CHECK(attr.hiddenSize.at(0) % alignSize == 0 && attr.hiddenSize.at(1) % alignSize == 0,
                  "Missalign hidden size.", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == kNumInTensor, "inTensor count invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == kNumOutTensor, "outTensor count invalid", return false);

        const auto &inTensorDesc0 = launchParam.GetInTensor(0).desc;
        const auto &inTensorDesc1 = launchParam.GetInTensor(1).desc;
        const auto &outTensorDesc = launchParam.GetOutTensor(0).desc;

        MKI_CHECK(inTensorDesc0.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(inTensorDesc0.dtype == TENSOR_DTYPE_INT8, "tensor dtype invalid", return false);
        MKI_CHECK(inTensorDesc0.dims.size() == DIM_SIZE_2, "tensor dims invalid", return false);

        MKI_CHECK(inTensorDesc1.format == TENSOR_FORMAT_ND || inTensorDesc1.format == TENSOR_FORMAT_FRACTAL_NZ,
                  "tensor format invalid", return false);
        MKI_CHECK(inTensorDesc1.dtype == TENSOR_DTYPE_INT8, "tensor dtype invalid", return false);
        MKI_CHECK(inTensorDesc1.dims.size() == DIM_SIZE_3 || inTensorDesc1.dims.size() == DIM_SIZE_4,
                  "tensor dims invalid", return false);

        MKI_CHECK(outTensorDesc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(outTensorDesc.dtype == TENSOR_DTYPE_FLOAT16 || outTensorDesc.dtype == TENSOR_DTYPE_BF16,
                  "tensor dtype invalid", return false);
        MKI_CHECK(outTensorDesc.dims.size() == DIM_SIZE_2, "tensor dims invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(MoeGmmTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        kernelInfo_.SetHwsyncIdx(0);
        return GetMoeGmmTiling(launchParam, kernelInfo_);
    }

private:
    const uint64_t kNumInTensor = 6;
    const uint64_t kNumOutTensor = 1;
    const uint32_t alignSize = 32;
};

REG_KERNEL_BASE(MoeGmmKernel);
REG_KERNEL_BASE(MoeGmmW8a8Kernel);
} // namespace AtbOps