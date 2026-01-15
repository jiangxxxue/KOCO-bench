/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#include "atbops/params/params.h"
#include "mixkernels/utils/common.h"
#include "tiling/paged_cache_load_tiling.h"
#include "tiling/paged_cache_load_tiling_dependency.h"

namespace AtbOps {
using namespace Mki;
constexpr static uint32_t TILING_SINGLE_ELENUM = 6;

class PagedCacheLoadKernel : public KernelBase {
public:
    explicit PagedCacheLoadKernel(const std::string &kernelName, const BinHandle *handle)
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::PagedCacheLoad),
            "PagedCacheLoad: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == DIM_7, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == DIM_2, "output num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        return sizeof(PagedCacheLoadTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = PagedCacheLoadTiling(launchParam, kernelInfo_);
        MKI_CHECK_NO_LOG(status.Ok(), return status);

        return Status::OkStatus();
    }
};

class PagedCacheLoadNzKernel : public PagedCacheLoadKernel {
public:
    explicit PagedCacheLoadNzKernel(const std::string &kernelName, const BinHandle *handle)
        : PagedCacheLoadKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(PagedCacheLoadKernel::CanSupport(launchParam), "failed to check support", return false);
        return true;
    }
};
REG_KERNEL_BASE(PagedCacheLoadNzKernel);

class PagedCacheLoadNdKernel : public PagedCacheLoadKernel {
public:
    explicit PagedCacheLoadNdKernel(const std::string &kernelName, const BinHandle *handle)
        : PagedCacheLoadKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(PagedCacheLoadKernel::CanSupport(launchParam), "failed to check support", return false);
        return true;
    }
};
REG_KERNEL_BASE(PagedCacheLoadNdKernel);
} // namespace AtbOps
