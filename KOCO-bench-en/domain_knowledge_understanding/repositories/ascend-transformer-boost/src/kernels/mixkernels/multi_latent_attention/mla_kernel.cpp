/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <numeric>
#include <mki/kernel_info.h>
#include <mki/base/kernel_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "mixkernels/multi_latent_attention/tiling/mla_tiling.h"
#include "mixkernels/multi_latent_attention/tiling/mla_tiling_dependency.h"
#include "mixkernels/utils/common.h"

namespace AtbOps {
using namespace Mki;
constexpr uint32_t TILINGMIN = 512;
class MLAKernel : public KernelBase {
public:
    explicit MLAKernel(const std::string &kernelName, const BinHandle *handle)
        : KernelBase(kernelName, handle)
    {
        launchBufferSize_ = Utils::RoundUp((TILING_PARA_SIZE + TILING_HEAD_SIZE) * sizeof(uint32_t), TILINGMIN);
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = MLATiling(launchParam, kernelInfo_);
        MKI_CHECK_NO_LOG(status.Ok(), return status);

        kernelInfo_.SetHwsyncIdx(0);
        return Status::OkStatus();
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MLA),
            "multi latent attention: param type invalid", return false);
        auto param = AnyCast<OpParam::MLA>(launchParam.GetParam());
        auto batch = param.kvSeqLen.size();
        auto maxQSeqlen = param.qSeqLen.data() != nullptr ? *std::max_element(param.qSeqLen.begin(), param.qSeqLen.end()) : 1;
        auto mtpTp1Flag = ((param.headSize == M_LIMIT) ||
                           (launchParam.GetInTensor(DIM_0).desc.dtype == TENSOR_DTYPE_INT8 && maxQSeqlen > 1));
        if (mtpTp1Flag) {
            uint64_t taskNum = param.qSeqLen.data() == nullptr ? batch :
                               std::accumulate(param.qSeqLen.data(),
                                               param.qSeqLen.data() + batch, static_cast<int32_t>(0));
            uint32_t blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
            uint64_t bufferSize =
                Utils::RoundUp(launchBufferSize_ + TILING_PARA_SIZE_TP1 * (taskNum - 1) * sizeof(uint32_t) +
                TILING_PARA_SIZE_TP1 * blockDim * blockDim * sizeof(uint32_t) + blockDim * 2 * sizeof(uint32_t),
                TILINGMIN);
            return bufferSize;
        }
        uint64_t bufferSize =
            Utils::RoundUp(launchBufferSize_ + TILING_PARA_SIZE * (batch - 1) * sizeof(uint32_t), TILINGMIN);
        return bufferSize;
    }
};

class MLAPrefillKernel : public KernelBase {
public:
    explicit MLAPrefillKernel(const std::string &kernelName, const BinHandle *handle)
        : KernelBase(kernelName, handle)
    {
        launchBufferSize_ = Utils::RoundUp((TILING_PARA_SIZE_PREFILL + TILING_HEAD_SIZE_PREFILL) * 
                            sizeof(uint32_t), TILINGMIN);
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        auto param = AnyCast<OpParam::MLA>(launchParam.GetParam());
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 3 ||
                            launchParam.GetInTensor(0).desc.dims.size() == 2,
                        "input 0 dim num invalid", return false);
        MKI_LOG(INFO) << "launchParam.GetOutTensorCount(): " << launchParam.GetOutTensorCount();
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MLA),
            "paged attention: param type invalid", return false);
        auto param = AnyCast<OpParam::MLA>(launchParam.GetParam());
        auto batch = param.qSeqLen.size();
        MKI_CHECK(batch > 0 && batch <= ND_BATCH_LIMIT, "batch is invalid", return 0);
        MKI_CHECK(param.headSize > 0 && param.headSize <= M_LIMIT, "headsize is invalid", return 0);
        uint64_t bufferSize =
            Utils::RoundUp(launchBufferSize_ + TILING_PARA_SIZE_PREFILL * (batch - 1) * sizeof(uint32_t), TILINGMIN);
        return bufferSize;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        Status ret = MLAPrefillTiling(launchParam, kernelInfo_);
        MKI_CHECK_NO_LOG(ret.Ok(), return ret);
        kernelInfo_.SetHwsyncIdx(0);
        return Status::OkStatus();
    }
};

class MLAPrefillBF16Kernel : public MLAPrefillKernel {
public:
    explicit MLAPrefillBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : MLAPrefillKernel(kernelName, handle)
    {
    }
};

REG_KERNEL_BASE(MLAKernel);
REG_KERNEL_BASE(MLAPrefillKernel);
REG_KERNEL_BASE(MLAPrefillBF16Kernel);

} // namespace AtbOps

