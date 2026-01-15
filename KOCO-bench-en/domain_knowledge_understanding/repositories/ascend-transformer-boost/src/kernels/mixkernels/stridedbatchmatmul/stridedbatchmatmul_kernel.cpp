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
#include "tiling/stridedbatchmatmul_tiling.h"
#include "tiling/tiling_data.h"
#include "tiling/tiling_api.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 2;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 1;
namespace AtbOps {
using namespace Mki;
class StridedBatchMatmulKernel : public KernelBase {
public:
    explicit StridedBatchMatmulKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "in tensor num is invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "out tensor num is invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::StridedBatchMatmul),
            "param type is invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        optiling::TCubeTiling tCubeTiling;
        uint32_t batch = static_cast<uint32_t>(AnyCast<OpParam::StridedBatchMatmul>(launchParam.GetParam()).batch);
        uint32_t tSize = tCubeTiling.GetDataSize();
        return sizeof(StridedBatchMatmulTilingData) + (sizeof(StridedBatchMatmulSampleTilingData) + tSize) * batch;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return StridedBatchMatmulTiling(launchParam, kernelInfo_);
    }
};
REG_KERNEL_BASE(StridedBatchMatmulKernel);
} // namespace AtbOps