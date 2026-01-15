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
#include <mki/utils/log/log.h>
#include "atbops/params/params.h"
#include "tiling/pad_with_hidden_state_tiling.h"
#include "tiling/tiling_data.h"
#include "mixkernels/utils/common.h"

namespace AtbOps {
using namespace Mki;
class PadWithHiddenStateKernel : public KernelBase {
public:
    explicit PadWithHiddenStateKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "in tensor num is invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "out tensor num is invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(DIM_0).desc.dims.size() == DIM_3,
            "output0 dim num should be " << DIM_3, return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::PadWithHiddenState),
            "param type is invalid", return false);

        auto inTensor0 = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "in tensor 0 format invalid", return false);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16, "in tensor 0 dtype invalid", return false);
        MKI_CHECK(inTensor0.desc.dims.size() == DIM_2, "in tensor 0 dims num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(PadWithHiddenStateTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return PadWithHiddenStateTiling(launchParam, kernelInfo_);
    }
};

REG_KERNEL_BASE(PadWithHiddenStateKernel);

}  // namespace AtbOps
