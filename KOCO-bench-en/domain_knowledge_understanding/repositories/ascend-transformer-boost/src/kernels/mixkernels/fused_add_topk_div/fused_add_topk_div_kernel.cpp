/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "kernels/utils/common.h"
#include "tiling/fused_add_topk_div_tiling.h"
#include "tiling/tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 4;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 2;
namespace AtbOps {
using namespace Mki;
class FusedAddTopkDivKernel : public KernelBase {
public:
    explicit FusedAddTopkDivKernel(const std::string &kernelName,
                                   const BinHandle *handle) : KernelBase(kernelName, handle) {}
 
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "inTensor count invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "outTensor count invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::FusedAddTopkDiv), "opParam type invalid",
                  return false);

        auto inTensor0 = launchParam.GetInTensor(0);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16 || inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT ||
                  inTensor0.desc.dtype == TENSOR_DTYPE_BF16,
                  "inTensor x dtype invalid", return false);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "inTensor x format invalid", return false);
        auto inTensor1 = launchParam.GetInTensor(1);
        MKI_CHECK(inTensor1.desc.dtype == inTensor0.desc.dtype, "inTensor add_num dtype invalid", return false);
        MKI_CHECK(inTensor1.desc.format == inTensor0.desc.format, "inTensor add_num format invalid", return false);
        auto param = AnyCast<OpParam::FusedAddTopkDiv>(launchParam.GetParam());
        if (param.enableExpertMapping) {
            auto inTensor2 = launchParam.GetInTensor(2);
            MKI_CHECK(inTensor2.desc.dtype == TENSOR_DTYPE_INT32, "inTensor mapping_num dtype invalid", return false);
            MKI_CHECK(inTensor2.desc.format == TENSOR_FORMAT_ND, "inTensor mapping_num format invalid", return false);

            auto inTensor3 = launchParam.GetInTensor(3);
            MKI_CHECK(inTensor3.desc.dtype == TENSOR_DTYPE_INT32, "inTensor mapping_table dtype invalid", return false);
            MKI_CHECK(inTensor3.desc.format == TENSOR_FORMAT_ND, "inTensor mapping_table format invalid", return false);
        }
        auto outTensor0 = launchParam.GetOutTensor(0);
        MKI_CHECK(outTensor0.desc.dtype == TENSOR_DTYPE_FLOAT, "outTensor y dtype invalid", return false);
        auto outTensor1 = launchParam.GetOutTensor(1);
        MKI_CHECK(outTensor1.desc.dtype == TENSOR_DTYPE_INT32, "outTensor indices dtype invalid", return false);
        return true;
    }
 
    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(FusedAddTopkDivTilingData);
    }
 
    Status InitImpl(const LaunchParam &launchParam) override
    {
        return FusedAddTopkDiv4Tiling(launchParam, kernelInfo_);
    }
};
REG_KERNEL_BASE(FusedAddTopkDivKernel);
} // namespace AtbOps