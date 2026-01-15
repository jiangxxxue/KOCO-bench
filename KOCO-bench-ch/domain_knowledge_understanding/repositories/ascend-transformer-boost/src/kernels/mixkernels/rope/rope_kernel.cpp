/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <mki_loader/op_register.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/const/op_const.h>
#include "atbops/params/params.h"
#include "tiling/rope_tiling.h"
#include "tiling/tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 5;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 2;

namespace AtbOps {
using namespace Mki;
class AtbRopeKernel : public KernelBase {
public:
    explicit AtbRopeKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool RopeDtypeCheck(const LaunchParam &launchParam, TensorDType dtypeCheck) const
    {
        auto inTensor0 = launchParam.GetInTensor(0);
        auto inTensor1 = launchParam.GetInTensor(1);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "in tensor0 format is invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(DIM_0).desc.dtype == TENSOR_DTYPE_BF16
                || launchParam.GetInTensor(DIM_0).desc.dtype == TENSOR_DTYPE_FLOAT16,
                "in tensor0 dtype is invalid", return false);
        MKI_CHECK(inTensor0.desc.dtype == inTensor1.desc.dtype,
            "in tensor0 dtype != tensor1 dtype", return false);
        MKI_CHECK(inTensor1.desc.format == TENSOR_FORMAT_ND, "in tensor1 format is invalid", return false);
        for (size_t i = 0; i < TENSOR_OUTPUT_NUM; i++) {
            auto outTensor = launchParam.GetOutTensor(i);
            MKI_CHECK(outTensor.desc.format == inTensor0.desc.format, "out tensor format is invalid", return false);
            MKI_CHECK(outTensor.desc.dtype == inTensor0.desc.dtype, "out tensor dtype is invalid", return false);
        }
        if (dtypeCheck == TENSOR_DTYPE_BF16) {
            MKI_CHECK(launchParam.GetInTensor(DIM_2).desc.dtype == TENSOR_DTYPE_BF16,
                "in tensor2 dtype is invalid", return false);
            MKI_CHECK(launchParam.GetInTensor(DIM_3).desc.dtype == TENSOR_DTYPE_BF16,
                "in tensor3 dtype is invalid", return false);
        } else {
            if (launchParam.GetInTensor(DIM_2).desc.dtype == dtypeCheck) {
                MKI_CHECK(launchParam.GetInTensor(DIM_3).desc.dtype == dtypeCheck,
                    "in tensor3 dtype is invalid", return false);
            } else if (launchParam.GetInTensor(DIM_2).desc.dtype == TENSOR_DTYPE_FLOAT) {
                MKI_CHECK(launchParam.GetInTensor(DIM_3).desc.dtype == TENSOR_DTYPE_FLOAT,
                    "in tensor3 dtype is invalid", return false);
            } else {
                return false;
            }
        }
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 2,
            "in tensor0 dims is invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(1).desc.dims.size() == 2,
            "in tensor1 dims is invalid", return false);
        auto inTensor4 = launchParam.GetInTensor(4);
        MKI_CHECK(inTensor4.desc.format == TENSOR_FORMAT_ND, "in tensor4 format is invalid", return false);
        MKI_CHECK((inTensor4.desc.dtype == TENSOR_DTYPE_INT32) ||
                        (inTensor4.desc.dtype == TENSOR_DTYPE_UINT32), "in tensor4 dtype is invalid", return false);
        return true;
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "in tensor num is invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "out tensor num is invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Rope),
            "param type is invalid", return false);
        auto param = AnyCast<OpParam::Rope>(launchParam.GetParam());
        auto headdim = launchParam.GetInTensor(DIM_2).desc.dims[DIM_1];
        MKI_CHECK(param.rotaryCoeff == 2 || param.rotaryCoeff == 4 || param.rotaryCoeff == headdim,
            "param.rotaryCoeff is invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(DIM_0).desc.dims[DIM_1] % headdim == 0
            && launchParam.GetInTensor(DIM_1).desc.dims[DIM_1] % headdim == 0,
            "hiddenSizeQ, hiddenSizeK should be an integer multiple of headdim",
            return false);
        return RopeDtypeCheck(launchParam, TENSOR_DTYPE_FLOAT16) || RopeDtypeCheck(launchParam, TENSOR_DTYPE_BF16);
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(RopeTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return RopeTiling(launchParam, kernelInfo_);
    }
};

REG_KERNEL_BASE(AtbRopeKernel);
} // namespace AtbOps
