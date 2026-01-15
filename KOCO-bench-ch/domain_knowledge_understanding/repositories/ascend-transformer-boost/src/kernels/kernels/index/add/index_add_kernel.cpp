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
#include "asdops/params/params.h"
#include "kernels/index/add/tiling/index_add_tiling.h"

namespace AsdOps {
using namespace Mki;
class IndexAddKernel : public KernelBase {
public:
    explicit IndexAddKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Index),
                  "index add: param type invalid", return false);
        auto opParam = AnyCast<OpParam::Index>(launchParam.GetParam());
        OpParam::Index::IndexType type = opParam.indexType;
        MKI_CHECK(type == OpParam::Index::INDEX_ADD, "index add: param type doesn't match", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 4, "index add: input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "index add: output num invalid", return false);

        int64_t numDims0 = static_cast<int64_t>(launchParam.GetInTensor(0).desc.dims.size());
        int64_t numDims1 = static_cast<int64_t>(launchParam.GetInTensor(1).desc.dims.size());
        int64_t numDims2 = static_cast<int64_t>(launchParam.GetInTensor(2).desc.dims.size());
        int64_t numDims3 = static_cast<int64_t>(launchParam.GetInTensor(3).desc.dims.size());

        MKI_CHECK(opParam.axis < numDims0 && opParam.axis >= -numDims0, "index add: axis invalid", return false);
        MKI_CHECK(numDims1 == 1, "index add: InTensor(1) dims invalid", return false);
        MKI_CHECK(numDims2 == numDims0, "index add: InTensor(2) dims invalid", return false);
        MKI_CHECK(numDims3 == 1, "index add: InTensor(3) dims invalid", return false);

        for (int64_t i = 0; i < numDims2; ++i) {
            int64_t dims2 = launchParam.GetInTensor(2).desc.dims[i];
            if (i == (opParam.axis + numDims0) || i == opParam.axis) {
                MKI_CHECK(dims2 == launchParam.GetInTensor(1).desc.dims[0], "index add: InTensor(2) shape invalid",
                             return false);
            } else {
                MKI_CHECK(dims2 == launchParam.GetInTensor(0).desc.dims[i], "index add: InTensor(2) shape invalid",
                             return false);
            }
        }

        MKI_CHECK(launchParam.GetInTensor(2).desc.dtype == launchParam.GetInTensor(0).desc.dtype,
                     "index add: InTensor(2) dtype doesn't match", return false);
        MKI_CHECK(launchParam.GetInTensor(3).desc.dtype == launchParam.GetInTensor(0).desc.dtype,
                     "index add: InTensor(3) dtype doesn't match", return false);

        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override { return launchBufferSize_; }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = IndexAddTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);

        return Status::OkStatus();
    }
};

// IndexAddF16Kernel
class IndexAddF16Kernel : public IndexAddKernel {
public:
    explicit IndexAddF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : IndexAddKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IndexAddKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype unsupported",
                     return false);
        MKI_CHECK(launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_INT32,
                     "index add: InTensor(1) dtype doesn't match", return false);
        return true;
    }
};
REG_KERNEL_BASE(IndexAddF16Kernel);
 
// IndexAddBF16Kernel
class IndexAddBF16Kernel : public IndexAddKernel {
public:
    explicit IndexAddBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : IndexAddKernel(kernelName, handle)
    {
    }
 
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IndexAddKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16, "tensor dtype unsupported",
                     return false);
        MKI_CHECK(launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_INT32,
                     "index add: InTensor(1) dtype doesn't match", return false);
        return true;
    }
};
REG_KERNEL_BASE(IndexAddBF16Kernel);
} // namespace AsdOps