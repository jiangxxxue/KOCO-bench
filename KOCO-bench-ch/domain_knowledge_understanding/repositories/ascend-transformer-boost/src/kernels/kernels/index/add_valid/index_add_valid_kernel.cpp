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
#include <mki/utils/const/op_const.h>
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/params.h"
#include "tiling/index_add_valid_tiling.h"
#include "tiling/tiling_data.h"

namespace AsdOps {
class IndexAddValidKernel : public KernelBase {
public:
    explicit IndexAddValidKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B, "platform not support",
                      return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Index),
                     "index add valid: param type invalid", return false);
        auto opParam = AnyCast<OpParam::Index>(launchParam.GetParam());
        OpParam::Index::IndexType type = opParam.indexType;
        MKI_CHECK(type == OpParam::Index::INDEX_ADD_VALID,
                     "index add valid: param type doesn't match", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 4, "index add valid: input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "index add valid: output num invalid", return false);

        int64_t varDim = static_cast<int64_t>(launchParam.GetInTensor(DIM_0).desc.dims.size());
        int64_t indicesDim = static_cast<int64_t>(launchParam.GetInTensor(DIM_1).desc.dims.size());
        int64_t updatesDim = static_cast<int64_t>(launchParam.GetInTensor(DIM_2).desc.dims.size());
        int64_t validIndicesDim = static_cast<int64_t>(launchParam.GetInTensor(DIM_3).desc.dims.size());
        int64_t input0shape1 = static_cast<int64_t>(launchParam.GetInTensor(DIM_0).desc.dims.at(1));
        int64_t input1shape0 = static_cast<int64_t>(launchParam.GetInTensor(DIM_1).desc.dims.at(0));
        int64_t input2shape0 = static_cast<int64_t>(launchParam.GetInTensor(DIM_2).desc.dims.at(0));
        int64_t input2shape1 = static_cast<int64_t>(launchParam.GetInTensor(DIM_2).desc.dims.at(1));
        MKI_CHECK(varDim == DIM_2,
                     "index add valid: var dims invalid", return false);
        MKI_CHECK(indicesDim == DIM_1,
                     "index add valid: indices dims invalid", return false);
        MKI_CHECK(updatesDim == DIM_2,
                     "index add valid: updates dims invalid", return false);
        MKI_CHECK(validIndicesDim == DIM_1, "index add valid: validIndicesNum dims invalid", return false);
        MKI_CHECK(input0shape1 == input2shape1,
                     "index add valid: var shape doesn't match updates shape", return false);
        MKI_CHECK(input1shape0 == input2shape0,
                     "index add valid: indices shape doesn't match updates shape", return false);
        MKI_CHECK(launchParam.GetInTensor(DIM_0).desc.dtype == launchParam.GetInTensor(DIM_2).desc.dtype,
                     "index add valid: updates dtype doesn't match varTensor dtype", return false);
        MKI_CHECK(launchParam.GetInTensor(DIM_1).desc.dtype == launchParam.GetInTensor(DIM_3).desc.dtype,
                     "index add valid: indices dtype doesn't match validIndicesNum dtype", return false);

        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(IndexAddValidTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return IndexAddValidTiling(launchParam, kernelInfo_);
    }
};

// IndexAddValidF16Kernel
class IndexAddValidF16Kernel : public IndexAddValidKernel {
public:
    explicit IndexAddValidF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : IndexAddValidKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IndexAddValidKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16 ||
            launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
                     "index add valid:var dtype unsupported", return false);
        MKI_CHECK(launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_INT32,
                     "index add valid:indices dtype doesn't match", return false);
        MKI_CHECK(launchParam.GetInTensor(2).desc.dtype == launchParam.GetInTensor(0).desc.dtype,
                     "index add valid:updates dtype unsupported", return false);
        MKI_CHECK(launchParam.GetInTensor(3).desc.dtype == TENSOR_DTYPE_INT32,
                     "index add valid:validIndicesNum dtype doesn't match", return false);
        return true;
    }
};
REG_KERNEL_BASE(IndexAddValidF16Kernel);
} // namespace AsdOps