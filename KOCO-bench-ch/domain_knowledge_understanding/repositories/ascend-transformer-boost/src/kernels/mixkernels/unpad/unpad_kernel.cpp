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
#include <mki/utils/const/op_const.h>
#include "tiling/unpad_tiling.h"
#include "tiling/tiling_data.h"
#include "mixkernels/utils/common.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 4;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 3;
namespace AtbOps {
using namespace Mki;
class UnpadKernel : public KernelBase {
public:
    explicit UnpadKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool DimsCheck(const LaunchParam &launchParam) const
    {
        auto inTensor0 = launchParam.GetInTensor(0); // input_ids

        auto inTensor1 = launchParam.GetInTensor(DIM_1); // cum_offset_now
        auto inTensor2 = launchParam.GetInTensor(DIM_2); // token_num
        auto inTensor3 = launchParam.GetInTensor(DIM_3); // seq_len
        MKI_CHECK(inTensor0.desc.dims[0] > 0 && inTensor0.desc.dims[1] > 0,
                "in tensor0 dims[0] or dims[1] is invalid", return false);
        uint32_t batch = static_cast<uint32_t>(inTensor0.desc.dims[0]);
        MKI_CHECK(inTensor1.desc.dims[0] == batch && inTensor1.desc.dims[1] == 1,
                "in tensor1 dims[0] or dims[1] is invalid", return false);
        MKI_CHECK(inTensor2.desc.dims[0] == 1 && inTensor2.desc.dims[1] == 1,
                "in tensor2 dims[0] or dims[1] is invalid", return false);
        MKI_CHECK(inTensor3.desc.dims[0] == batch && inTensor3.desc.dims[1] == 1,
                "in tensor3 dims[0] or dims[1] is invalid", return false);

        uint32_t maxSeqlen = static_cast<uint32_t>(inTensor0.desc.dims[1]);
        MKI_CHECK(batch <= UINT32_MAX / maxSeqlen, "batch * maxSeqlen exceeds uint32_t maximum (overflow)", return false);
        uint32_t totalLen = batch * maxSeqlen;
        auto outTensor0 = launchParam.GetOutTensor(0); // x_remove_padding
        auto outTensor1 = launchParam.GetOutTensor(1); // cum_offsets_out
        auto outTensor2 = launchParam.GetOutTensor(2); // padding_offset
        MKI_CHECK(outTensor0.desc.dims[0] == 1 && outTensor0.desc.dims[1] == totalLen,
                "out tensor0 dims[0] or dims[1] is invalid", return false);
        MKI_CHECK(outTensor1.desc.dims[0] == batch && outTensor1.desc.dims[1] == 1,
                "out tensor1 dims[0] or dims[1] is invalid", return false);
        MKI_CHECK(outTensor2.desc.dims[0] == 1 && outTensor2.desc.dims[1] == totalLen,
                "out tensor2 dims[0] or dims[1] is invalid", return false);
        return true;
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM,
            "in tensor num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM,
            "out tensor num invalid", return false);
        auto inTensor0 = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_INT64,
            "in tensor 0 dtype invalid", return false);
        auto inTensor1 = launchParam.GetInTensor(DIM_1);
        MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_INT32,
            "in tensor 1 dtype invalid", return false);
        auto inTensor2 = launchParam.GetInTensor(DIM_2);
        MKI_CHECK(inTensor2.desc.dtype == TENSOR_DTYPE_INT64,
            "in tensor 2 dtype invalid", return false);
        auto inTensor3 = launchParam.GetInTensor(DIM_3);
        MKI_CHECK(inTensor3.desc.dtype == TENSOR_DTYPE_INT32,
            "in tensor 3 dtype invalid", return false);

        for (size_t i = 0; i < TENSOR_INPUT_NUM; i++) {
            auto inTensor = launchParam.GetInTensor(i);
            MKI_CHECK(inTensor.desc.format == TENSOR_FORMAT_ND,
                "in tensor " << i << " format invalid", return false);
            MKI_CHECK(inTensor.desc.dims.size() == DIM_2,
                "in tensor " << i << " dim num invalid", return false);
        }
        for (size_t i = 0; i < TENSOR_OUTPUT_NUM; i++) {
            auto outTensor = launchParam.GetOutTensor(i);
            MKI_CHECK(outTensor.desc.format == TENSOR_FORMAT_ND,
                "out tensor " << i << " format invalid", return false);
            MKI_CHECK(outTensor.desc.dims.size() == DIM_2,
                "out tensor " << i << " dim num invalid", return false);
        }
        auto outTensor0 = launchParam.GetOutTensor(0);
        MKI_CHECK(outTensor0.desc.dtype == TENSOR_DTYPE_INT64,
            "outTensor0 dtype invalid", return false);
        bool outputCheck = DimsCheck(launchParam);
        return outputCheck;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(UnpadTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return UnpadTiling(launchParam, kernelInfo_);
    }
};
REG_KERNEL_BASE(UnpadKernel);
} // namespace AtbOps
