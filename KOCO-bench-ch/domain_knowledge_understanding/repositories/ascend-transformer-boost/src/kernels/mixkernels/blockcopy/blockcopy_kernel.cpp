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
#include "atbops/params/params.h"
#include "mixkernels/utils/common.h"
#include "tiling/blockcopy_tiling.h"
#include "tiling/tiling_data.h"

namespace AtbOps {
using namespace Mki;
static constexpr uint32_t TENSOR_INPUT_NUM = 5;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 2;

class BlockCopyKernel : public KernelBase {
public:
    explicit BlockCopyKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
        MKI_LOG(INFO) << "BlockCopyKernel construct ";
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "in tensor num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "out tensor num invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::BlockCopy),
            "blockcopy: param type invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        return sizeof(BlockCopyTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = BlockCopyTiling(launchParam, kernelInfo_);
        MKI_CHECK_NO_LOG(status.Ok(), return status);

        return Status::OkStatus();
    }
};
REG_KERNEL_BASE(BlockCopyKernel);
} // namespace AtbOps