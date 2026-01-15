/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/base/kernel_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/params.h"
#include "kernels/matmul/tiling/tiling_data.h"
#include "kernels/matmul/common/common_tiling.h"
#include "kernels/matmul/common/common.h"
#include "kernels/matmul/tiling/pp_matmul_nd_tiling.h"

namespace AsdOps {
using namespace Mki;

inline bool CheckAsdOpsNZ(const LaunchParam &launchParam, size_t size)
{
    MKI_CHECK(launchParam.GetInTensorCount() == size, "inTensor count invalid", return false);
    MKI_CHECK(launchParam.GetOutTensorCount() == 1, "outTensor count invalid", return false);
    MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MatMul), "check param type failed!",
                 return false);

    const auto &inTensor0 = launchParam.GetInTensor(0);
    const auto &inTensor1 = launchParam.GetInTensor(1);
    const auto &outTensor = launchParam.GetOutTensor(0);
    MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
    MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16 || inTensor0.desc.dtype == TENSOR_DTYPE_INT8,
              "tensor dtype invalid", return false);
    MKI_CHECK(inTensor0.desc.dims.size() == 2 || inTensor0.desc.dims.size() == 3, "tensor dims invalid",
              return false);
    MKI_CHECK(inTensor1.desc.format == TENSOR_FORMAT_FRACTAL_NZ, "tensor format invalid", return false);
    MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16 || inTensor1.desc.dtype == TENSOR_DTYPE_INT8,
              "tensor dtype invalid", return false);
    MKI_CHECK(inTensor1.desc.dims.size() == 3 || inTensor1.desc.dims.size() == 4, "tensor dims invalid",
              return false);
    MKI_CHECK(outTensor.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
    MKI_CHECK(outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16 || outTensor.desc.dtype == TENSOR_DTYPE_BF16 ||
                  outTensor.desc.dtype == TENSOR_DTYPE_INT8,
              "tensor dtype invalid", return false);
    MKI_CHECK(outTensor.desc.dims.size() == 2 || outTensor.desc.dims.size() == 3, "tensor dims invalid",
              return false);
    return true;
}

class PpMatmulF16M300Kernel : public KernelBase {
public:
    explicit PpMatmulF16M300Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle) {}

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(PpTilingDataNd);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return PpTilingNd(launchParam, kernelInfo_);
    }
};

class PpMatmulF16NdM300Kernel : public PpMatmulF16M300Kernel {
public:
    explicit PpMatmulF16NdM300Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : PpMatmulF16M300Kernel(kernelName, handle) {}

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_310B, "platformtype is invalid",
                     return false);
        MKI_CHECK(CheckAsdOpsND(launchParam, 2), "CheckAsdOpsND failed", return false); // 输入参数数量为2
        return true;
    }
};

REG_KERNEL_BASE(PpMatmulF16NdM300Kernel);

class PpMatmulF16NzM300Kernel : public PpMatmulF16M300Kernel {
public:
    explicit PpMatmulF16NzM300Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : PpMatmulF16M300Kernel(kernelName, handle) {}

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_310B, "platformtype is invalid",
                     return false);
        MKI_CHECK(CheckAsdOpsNZ(launchParam, 2), "CheckAsdOpsNZ failed", return false); // 输入参数数量为2
        return true;
    }
};

REG_KERNEL_BASE(PpMatmulF16NzM300Kernel);

} // namespace AsdOps
