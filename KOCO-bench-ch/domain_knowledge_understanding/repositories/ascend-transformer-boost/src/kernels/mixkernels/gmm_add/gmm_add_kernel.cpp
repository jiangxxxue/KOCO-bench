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
#include "tiling/gmm_add_tiling.h"
#include "tiling/gmm_add_tilingdata.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 4;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 1;
namespace AtbOps {
using namespace Mki;
class GmmAddKernel : public KernelBase {
public:
    explicit GmmAddKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
        launchBufferSize_ = sizeof(GmmTilingData);
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::GmmAdd), "invalid op param.", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "in tensor num is invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "out tensor num is invalid", return false);
        
        const auto &inTensorDesc0 = launchParam.GetInTensor(0).desc;
        const auto &inTensorDesc1 = launchParam.GetInTensor(1).desc;
        const auto &inTensorDesc2 = launchParam.GetInTensor(2).desc;
        const auto &inTensorDesc3 = launchParam.GetInTensor(3).desc;
        const auto &outTensorDesc = launchParam.GetOutTensor(0).desc;

        MKI_CHECK(inTensorDesc0.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(inTensorDesc0.dtype == TENSOR_DTYPE_FLOAT16 || inTensorDesc0.dtype == TENSOR_DTYPE_BF16,
                  "tensor dtype invalid", return false);
        MKI_CHECK(inTensorDesc0.dims.size() == DIM_2, "tensor dims invalid", return false);

        MKI_CHECK(inTensorDesc1.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(inTensorDesc1.dtype == TENSOR_DTYPE_FLOAT16 || inTensorDesc0.dtype == TENSOR_DTYPE_BF16,
                  "tensor dtype invalid", return false);
        MKI_CHECK(inTensorDesc1.dims.size() == DIM_2, "tensor dims invalid", return false);

        MKI_CHECK(inTensorDesc2.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(inTensorDesc2.dtype == TENSOR_DTYPE_INT64, "tensor dtype invalid", return false);
        MKI_CHECK(inTensorDesc2.Numel() > 0 && inTensorDesc2.Numel() <= 128, "tensor dims invalid", return false);

        MKI_CHECK(inTensorDesc3.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(inTensorDesc3.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype invalid", return false);
        MKI_CHECK(inTensorDesc3.dims.size() == DIM_2, "tensor dims invalid", return false);

        MKI_CHECK(outTensorDesc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(outTensorDesc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype invalid", return false);
        MKI_CHECK(outTensorDesc.dims.size() == DIM_2, "tensor dims invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(GmmTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        MKI_CHECK(GetGmmAddTilingData(launchParam, kernelInfo_).Ok(), "Get tiling data failed",
                  return Status::FailStatus(ERROR_LAUNCH_KERNEL_ERROR));
        return Status::OkStatus();
    }
};
REG_KERNEL_BASE(GmmAddKernel);
} // namespace AtbOps