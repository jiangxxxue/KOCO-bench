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
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki_loader/op_register.h>

#include "asdops/params/params.h"
#include "kernels/gather/tiling/gather_tiling.h"

namespace AsdOps {
using namespace Mki;
static constexpr size_t IN_X = 0;
static constexpr size_t IN_INDICES = 1;
static constexpr size_t OUT_Y = 0;
static constexpr uint64_t TENSOR_GATHER_AXIS_IDX = 2;

static constexpr size_t BYTES_64BITS = 8;
static constexpr size_t BYTES_32BITS = 4;
static constexpr size_t BYTES_16BITS = 2;
// static constexpr size_t BYTES_8BITS = 1;
template <size_t ELEM_SIZE, TensorDType INDICE_TYPE>
class GatherKernel : public KernelBase {
public:
    explicit GatherKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Gather), "check param type failed!", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "check inTensor count failed", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "check outTensor count failed", return false);
        TensorDType xDType = launchParam.GetInTensor(IN_X).desc.dtype;
        TensorDType indicesDType = launchParam.GetInTensor(IN_INDICES).desc.dtype;
        MKI_CHECK(GetTensorElementSize(xDType) == ELEM_SIZE, "data width not matched", return false);
        // support unsigned type, use GetTensorElementSize wrapped
        MKI_CHECK(GetTensorElementSize(indicesDType) == GetTensorElementSize(INDICE_TYPE), "indices dtype not matched",
                  return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Gather), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::Gather>(launchParam.GetParam());
        size_t constTensor0Size = Utils::GetConstTensorSize<int64_t>(param.axis);
        return launchBufferSize_ + constTensor0Size;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        Status status = GatherTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);
        kernelInfo_.SetConstTensorOffset(launchBufferSize_);
        auto param = AnyCast<OpParam::Gather>(launchParam.GetParam());
        kernelInfo_.AddConstTensorData<int64_t>(TENSOR_GATHER_AXIS_IDX, param.axis);
        return Status::OkStatus();
    }
};

using Gather16I64Kernel = GatherKernel<BYTES_16BITS, TENSOR_DTYPE_INT64>;
using Gather16I32Kernel = GatherKernel<BYTES_16BITS, TENSOR_DTYPE_INT32>;
using Gather32I64Kernel = GatherKernel<BYTES_32BITS, TENSOR_DTYPE_INT64>;
using Gather32I32Kernel = GatherKernel<BYTES_32BITS, TENSOR_DTYPE_INT32>;
using Gather64I64Kernel = GatherKernel<BYTES_64BITS, TENSOR_DTYPE_INT64>;
using Gather64I32Kernel = GatherKernel<BYTES_64BITS, TENSOR_DTYPE_INT32>;

REG_KERNEL_BASE(Gather16I64Kernel);
REG_KERNEL_BASE(Gather16I32Kernel);
REG_KERNEL_BASE(Gather32I64Kernel);
REG_KERNEL_BASE(Gather32I32Kernel);
REG_KERNEL_BASE(Gather64I64Kernel);
REG_KERNEL_BASE(Gather64I32Kernel);
} // namespace AsdOps
