/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#include "tiling/mla_preprocess_tiling.h"
#include "tiling/mla_preprocess_tilingdata.h"
#include "mixkernels/utils/common.h"

namespace AtbOps {
using namespace Mki;
class MLAPreprocessKernel : public KernelBase {
public:
    explicit MLAPreprocessKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
        launchBufferSize_ = sizeof(MlaTilingData);
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MlaPreprocess), "invalid op param.", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(MlaTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        kernelInfo_.SetHwsyncIdx(0);
        return GetMLAProprecessTiling(launchParam, kernelInfo_);
    }

private:
};

class MLAPreprocessBF16Kernel : public MLAPreprocessKernel {
public:
    explicit MLAPreprocessBF16Kernel(const std::string &kernelName,
        const BinHandle *handle) noexcept
        : MLAPreprocessKernel(kernelName, handle) {}
};

REG_KERNEL_BASE(MLAPreprocessKernel);
REG_KERNEL_BASE(MLAPreprocessBF16Kernel);
} // namespace AtbOps