/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendOpCommonLib is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "asdops/params/zeroslike.h"
#include "mki/base/kernel_base.h"
#include "mki/utils/log/log.h"
#include "mki/utils/math/math.h"
#include "mki/utils/math/tensor_utils.h"
#include "mki_loader/op_register.h"
#include "tiling/zeroslike_tiling.h"

namespace AsdOps {
using namespace Mki;
// ZeroLikeKernel
class ZerosLikeKernel : public KernelBase {
public:
    explicit ZerosLikeKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::ZerosLike), "zeroslike: param type invalid",
                  return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return ZerosLikeTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

// ZerosLikeF32Kernel
class ZerosLikeF32Kernel : public ZerosLikeKernel {
public:
    explicit ZerosLikeF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ZerosLikeKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ZerosLikeKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype unsupported",
                  return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype unsupported",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(ZerosLikeF32Kernel);
} // namespace AsdOps