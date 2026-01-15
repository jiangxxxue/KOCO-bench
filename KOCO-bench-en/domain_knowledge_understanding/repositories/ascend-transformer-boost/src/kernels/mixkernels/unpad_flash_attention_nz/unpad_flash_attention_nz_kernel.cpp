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
#include <mki/utils/math/math.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/checktensor/check_tensor.h>
#include "atbops/params/params.h"
#include "tiling/flash_attention_nz_tiling.h"
#include "tiling/flash_attention_nz_tiling_dependency.h"
#include "mixkernels/utils/common.h"
namespace AtbOps {
using namespace Mki;
static uint64_t GetUnpadCommonTilingSize(const LaunchParam &launchParam, uint64_t launchBufferSize, uint64_t paramSize)
{
    MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::UnpadFlashAttentionNz),
            "unpad_flash_attention: param type invalid", return 0);
    auto param = AnyCast<OpParam::UnpadFlashAttentionNz>(launchParam.GetParam());
    auto batch = param.qSeqLen.size();
    MKI_CHECK(batch > 0 && batch <= NZ_BATCH_LIMIT, "batch size " << batch << "invalid", return 0);
    return launchBufferSize * batch + paramSize;
}

class UnpadFlashAttentionNzKernel : public KernelBase {
public:
    explicit UnpadFlashAttentionNzKernel(const std::string &kernelName, const BinHandle *handle)
        : KernelBase(kernelName, handle)
    {
        launchBufferSize_ = NZ_REAL_CORE_TILING_SIZE * sizeof(uint32_t);
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        uint64_t paramSize = sizeof(uint32_t) * GetNzRealCoreTilingOffset();
        return GetUnpadCommonTilingSize(launchParam, launchBufferSize_, paramSize);
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::UnpadFlashAttentionNz),
            "unpad_flash_attention: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 7, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 4, "input 0 dim num invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(1).desc.dims.size() == 5 ||
                  launchParam.GetInTensor(1).desc.dims.size() == 4 ||
                  launchParam.GetInTensor(1).desc.dims.size() == 0, "input 1 dim num invalid", return false);
        auto param = AnyCast<OpParam::UnpadFlashAttentionNz>(launchParam.GetParam());
        auto scaleType = param.scaleType;
        MKI_CHECK(scaleType == OpParam::UnpadFlashAttentionNz::ScaleType::SCALE_LOGN ||
            scaleType == OpParam::UnpadFlashAttentionNz::ScaleType::SCALE_TOR,
                "scaletype invalid", return false);
        if (scaleType == OpParam::UnpadFlashAttentionNz::ScaleType::SCALE_LOGN) {
            auto &tensorLog = launchParam.GetInTensor(DIM_6);
            MKI_CHECK(!CheckEmptyTensor(tensorLog), "Input7 should not be null tensor", return false);
            auto inDtype = tensorLog.desc.dtype;
            if (param.precType == OpParam::UnpadFlashAttentionNz::PrecType::BMM1_FP32_EXP_FP32) {
                MKI_CHECK(inDtype == TENSOR_DTYPE_FLOAT, "LogNscale should be float", return false);
            } else {
                MKI_CHECK(inDtype == TENSOR_DTYPE_FLOAT16, "LogNscale should be fp16", return false);
            }
            MKI_CHECK(param.dataDimOrder == OpParam::UnpadFlashAttentionNz::DataDimOrder::TYPE_BSND &&
                      param.windowSize == 0,
                        "LogN can not support BNSD,SWA", return false);
        }
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return FlashAttentionNzTiling(launchParam, kernelInfo_);
    }
};
REG_KERNEL_BASE(UnpadFlashAttentionNzKernel);

class UnpadAlibiFlashAttentionNzKernel : public UnpadFlashAttentionNzKernel {
public:
    explicit UnpadAlibiFlashAttentionNzKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : UnpadFlashAttentionNzKernel(kernelName, handle) {}
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        auto param = AnyCast<OpParam::UnpadFlashAttentionNz>(launchParam.GetParam());
        auto scaleType = param.scaleType;
        if (scaleType == OpParam::UnpadFlashAttentionNz::ScaleType::SCALE_LOGN) {
            auto &tensorLog = launchParam.GetInTensor(DIM_6);
            auto maxQSeqlenIter = std::max_element(param.qSeqLen.begin(), param.qSeqLen.end());
            MKI_CHECK(tensorLog.desc.dims.at(0) > 0 && *maxQSeqlenIter > 0 &&
                        static_cast<unsigned long>(tensorLog.desc.dims.at(0)) >=
                        static_cast<unsigned long>(*maxQSeqlenIter),
                            "input 7 num invalid", return false);
        }
        MKI_CHECK(UnpadFlashAttentionNzKernel::CanSupport(launchParam), "failed to check support", return false);
        return true;
    }
};
class UnpadFlashAttentionNzEncoderKernel : public UnpadFlashAttentionNzKernel {
public:
    explicit UnpadFlashAttentionNzEncoderKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : UnpadFlashAttentionNzKernel(kernelName, handle) {}
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        auto param = AnyCast<OpParam::UnpadFlashAttentionNz>(launchParam.GetParam());
        auto scaleType = param.scaleType;
        if (scaleType == OpParam::UnpadFlashAttentionNz::ScaleType::SCALE_LOGN) {
            auto &tensorLog = launchParam.GetInTensor(DIM_6);
            auto maxQSeqlenIter = std::max_element(param.qSeqLen.begin(), param.qSeqLen.end());
            MKI_CHECK(tensorLog.desc.dims.at(0) > 0 && *maxQSeqlenIter > 0 &&
                        static_cast<unsigned long>(tensorLog.desc.dims.at(0)) >=
                        static_cast<unsigned long>(*maxQSeqlenIter),
                            "input 7 num invalid", return false);
        }
        MKI_CHECK(UnpadFlashAttentionNzKernel::CanSupport(launchParam), "failed to check support", return false);
        return true;
    }
};
class UnpadFlashAttentionNzDecoderKernel : public UnpadFlashAttentionNzKernel {
public:
    explicit UnpadFlashAttentionNzDecoderKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : UnpadFlashAttentionNzKernel(kernelName, handle) {}
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        auto param = AnyCast<OpParam::UnpadFlashAttentionNz>(launchParam.GetParam());
        auto scaleType = param.scaleType;
        auto batchSize = param.qSeqLen.size();
        if (scaleType == OpParam::UnpadFlashAttentionNz::ScaleType::SCALE_LOGN) {
            auto &tensorLog = launchParam.GetInTensor(DIM_6);
            MKI_CHECK(tensorLog.desc.dims.at(0) > 0 &&
                    static_cast<unsigned long>(tensorLog.desc.dims.at(0)) >= batchSize,
                    "input 7 dim num invalid", return false);
        }
        MKI_CHECK(UnpadFlashAttentionNzKernel::CanSupport(launchParam), "failed to check support", return false);
        return true;
    }
};
REG_KERNEL_BASE(UnpadAlibiFlashAttentionNzKernel);
REG_KERNEL_BASE(UnpadFlashAttentionNzEncoderKernel);
REG_KERNEL_BASE(UnpadFlashAttentionNzDecoderKernel);
} // namespace AtbOps