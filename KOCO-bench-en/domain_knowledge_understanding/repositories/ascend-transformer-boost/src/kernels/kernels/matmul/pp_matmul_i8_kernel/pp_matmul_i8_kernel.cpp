/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "asdops/params/params.h"
#include "mki/base/kernel_base.h"
#include "mki/utils/log/log.h"
#include "mki_loader/op_register.h"
#include "kernels/matmul/common/common.h"
#include "kernels/matmul/tiling/pp_matmul_tiling.h"
#include "kernels/matmul/tiling/tiling_data.h"

namespace AsdOps {
using namespace Mki;
class PpMatMulI8CommKernel : public KernelBase {
public:
    explicit PpMatMulI8CommKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(CheckAsdOpsND(launchParam, 5), "CheckAsdOpsND failed", return false);
        const auto &inTensor2 = launchParam.GetInTensor(2);
        const auto &attrs = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        if (attrs.withBias) {
            MKI_CHECK(inTensor2.desc.format == TENSOR_FORMAT_ND, "inTensor2 format invalid", return false);
            MKI_CHECK(inTensor2.desc.dtype == TENSOR_DTYPE_INT32, "inTensor2 dtype invalid", return false);
        }

        return true;
    }
    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(PpMatmulTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override { return PpMatmulTiling(launchParam, kernelInfo_); }
};

class PpMatMulI8Kernel : public PpMatMulI8CommKernel {
public:
    explicit PpMatMulI8Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : PpMatMulI8CommKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(PpMatMulI8CommKernel::CanSupport(launchParam), return false);
        const auto &inTensor3 = launchParam.GetInTensor(3);
        MKI_CHECK(inTensor3.desc.format == TENSOR_FORMAT_ND, "inTensor3 format invalid", return false);
        MKI_CHECK(inTensor3.desc.dtype == TENSOR_DTYPE_INT64 || inTensor3.desc.dtype == TENSOR_DTYPE_UINT64,
                  "inTensor3 dtype invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(PpMatMulI8Kernel);

class PpMatMulI8NdNzKernel : public PpMatMulI8Kernel {
public:
    explicit PpMatMulI8NdNzKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : PpMatMulI8Kernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        const auto &inTensor2 = launchParam.GetInTensor(2);
        const auto &attrs = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        if (attrs.withBias) {
            MKI_CHECK(inTensor2.desc.format == TENSOR_FORMAT_ND, "inTensor2 format invalid", return false);
            MKI_CHECK(inTensor2.desc.dtype == TENSOR_DTYPE_INT32, "inTensor2 dtype invalid", return false);
        }
        const auto &inTensor3 = launchParam.GetInTensor(3);
        MKI_CHECK(inTensor3.desc.format == TENSOR_FORMAT_ND, "inTensor3 format invalid", return false);
        MKI_CHECK(inTensor3.desc.dtype == TENSOR_DTYPE_INT64 || inTensor3.desc.dtype == TENSOR_DTYPE_UINT64,
                  "inTensor3 dtype invalid", return false);
        return true;
    }
};

REG_KERNEL_BASE(PpMatMulI8NdNzKernel);

class PpMatMulI8Bf16Kernel : public PpMatMulI8CommKernel {
public:
    explicit PpMatMulI8Bf16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : PpMatMulI8CommKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == size_t(5), "inTensor count invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "outTensor count invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MatMul), "check param type failed!",
                    return false);

        const auto &inTensor0 = launchParam.GetInTensor(0);
        const auto &inTensor1 = launchParam.GetInTensor(1);
        const auto &outTensor = launchParam.GetOutTensor(0);
        const auto &attrs = AnyCast<OpParam::MatMul>(launchParam.GetParam());

        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_INT8,
                "tensor dtype invalid", return false);
        MKI_CHECK(inTensor0.desc.dims.size() == 2 || inTensor0.desc.dims.size() == 3, "tensor dims invalid",
                return false);

        MKI_CHECK((inTensor1.desc.format == TENSOR_FORMAT_ND) || (inTensor1.desc.format == TENSOR_FORMAT_FRACTAL_NZ),
                    "tensor format invalid", return false);
        MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_INT8,
                "tensor dtype invalid", return false);
        MKI_CHECK(inTensor1.desc.dims.size() == 2 || inTensor1.desc.dims.size() == 3 ||
                    inTensor1.desc.dims.size() == 4, "tensor dims invalid", return false);

        MKI_CHECK(outTensor.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        if (attrs.quantMode == OpParam::MatMul::QuantMode::PER_TOKEN_SYMM) {
            MKI_CHECK(outTensor.desc.dtype == TENSOR_DTYPE_BF16 || outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16,
                      "tensor dtype invalid", return false);
        } else {
            MKI_CHECK(outTensor.desc.dtype == TENSOR_DTYPE_BF16, "tensor dtype invalid", return false);
        }
        MKI_CHECK(outTensor.desc.dims.size() == 2 || outTensor.desc.dims.size() == 3, "tensor dims invalid",
                return false);

        const auto &inTensor3 = launchParam.GetInTensor(3);
        MKI_CHECK(inTensor3.desc.format == TENSOR_FORMAT_ND, "inTensor3 format invalid", return false);
        MKI_CHECK(inTensor3.desc.dtype == TENSOR_DTYPE_FLOAT, "inTensor3 dtype invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        Status status = PpMatmulTiling(launchParam, kernelInfo_);
        MKI_CHECK(status.Ok(), "tiling return invalid value.", return status);
        kernelInfo_.SetHwsyncIdx(0);
        return status;
    }
};
REG_KERNEL_BASE(PpMatMulI8Bf16Kernel);

class PpMatMulI8WeightNzKernel : public PpMatMulI8CommKernel {
    static constexpr uint32_t NUM_INPUT = 5;

public:
    explicit PpMatMulI8WeightNzKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : PpMatMulI8CommKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == NUM_INPUT, "Invalid number of input tensor.", return false);

        const auto &inTensor0 = launchParam.GetInTensor(0);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "Invalid format of input tensor[0].", return false);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_INT8, "Invalid dtype of input tensor[0].", return false);

        const auto &inTensor1 = launchParam.GetInTensor(1);
        MKI_CHECK(inTensor1.desc.format == TENSOR_FORMAT_FRACTAL_NZ, "Invalid format of input tensor[1].",
                  return false);
        MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_INT8, "Invalid dtype of input tensor[1].", return false);

        const auto &inTensor2 = launchParam.GetInTensor(2);
        const auto &attrs = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        if (attrs.withBias) {
            MKI_CHECK(inTensor2.desc.format == TENSOR_FORMAT_ND, "inTensor2 format invalid", return false);
            MKI_CHECK(inTensor2.desc.dtype == TENSOR_DTYPE_INT32, "inTensor2 dtype invalid", return false);
        }

        const auto &inTensor3 = launchParam.GetInTensor(3);
        MKI_CHECK(inTensor3.desc.format == TENSOR_FORMAT_ND, "inTensor3 format invalid", return false);
        MKI_CHECK(inTensor3.desc.dtype == TENSOR_DTYPE_INT64, "inTensor3 dtype invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(PpMatMulI8WeightNzKernel);

} // namespace AsdOps
