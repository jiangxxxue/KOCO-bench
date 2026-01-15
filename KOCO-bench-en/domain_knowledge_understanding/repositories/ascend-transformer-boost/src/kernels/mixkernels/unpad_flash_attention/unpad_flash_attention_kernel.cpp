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
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "tiling/flash_attention_tiling.h"
#include "mixkernels/utils/common.h"


namespace AtbOps {
using namespace Mki;
constexpr uint64_t TENSOR_Q_SEQLEN_IDX = 6;
constexpr uint64_t TENSOR_KV_SEQLEN_IDX = 7;
constexpr uint32_t TILINGMIN = 512;
class UnpadFlashAttentionKernel : public KernelBase {
public:
    explicit UnpadFlashAttentionKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
        launchBufferSize_ = Utils::RoundUp((TILING_PARA_SIZE + TILING_HEAD_SIZE) * sizeof(uint32_t), TILINGMIN);
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        if (!CheckParamType(launchParam)) { return false; }
        MKI_CHECK(launchParam.GetInTensorCount() == 12, "input num invalid", return false);
        auto &param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        auto dataShapeType = param.dataShapeType;
        if (dataShapeType == OpParam::UnpadFlashAttention::DataShapeType::TYPE_BNSD) {
            // encoder shape [B,N,S,D], // decoder shape [numTokens,hiddenSize]
            MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 2 ||
                          launchParam.GetInTensor(0).desc.dims.size() == 4,
                      "input 0 dim num invalid", return false);
            MKI_CHECK(param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_UNDEFINED &&
                      !param.compressHead,
                      "BNSD is can not support quant,compressHead", return false);
        } else {
            MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 3 ||
                          launchParam.GetInTensor(0).desc.dims.size() == 2,
                      "input 0 dim num invalid", return false);
        }

        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        if (!CheckParamType(launchParam)) { return false; }
        auto &param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        auto batch = param.qSeqLen.size();
        MKI_CHECK(batch > 0 && batch <= ND_BATCH_LIMIT, "batch is invalid", return 0);
        uint64_t bufferSize =
            Utils::RoundUp(launchBufferSize_ + TILING_PARA_SIZE * (batch - 1) * sizeof(uint32_t), TILINGMIN);
        return bufferSize;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        Status ret = FlashAttentionTiling(launchParam, kernelInfo_);
        MKI_CHECK_NO_LOG(ret.Ok(), return ret);
        kernelInfo_.SetHwsyncIdx(0);
        return Status::OkStatus();
    }
protected:
    bool CheckParamType(const LaunchParam& launchParam) const
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::UnpadFlashAttention),
                  "unpad_flash_attention: param type invalid", return false);
        return true;
    }
};

class UnpadFlashAttentionNdKernel : public UnpadFlashAttentionKernel {
public:
    explicit UnpadFlashAttentionNdKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : UnpadFlashAttentionKernel(kernelName, handle)
    {
    }
};

class UnpadFlashAttentionRazorFusionKernel : public UnpadFlashAttentionKernel {
public:
    explicit UnpadFlashAttentionRazorFusionKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : UnpadFlashAttentionKernel(kernelName, handle)
    {
    }
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        if (!CheckParamType(launchParam)) { return false; }
        MKI_CHECK(launchParam.GetInTensorCount() == 3, "input num invalid", return false);
        auto &param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        auto dataShapeType = param.dataShapeType;
        if (dataShapeType == OpParam::UnpadFlashAttention::DataShapeType::TYPE_BNSD) {
            // encoder shape [B,N,S,D], // decoder shape [numTokens,hiddenSize]
            MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 2 ||
                          launchParam.GetInTensor(0).desc.dims.size() == 4,
                      "input 0 dim num invalid", return false);
            MKI_CHECK(param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_UNDEFINED &&
                      !param.compressHead,
                      "BNSD is can not support quant,compressHead", return false);
        } else {
            MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 3 ||
                          launchParam.GetInTensor(0).desc.dims.size() == 2,
                      "input 0 dim num invalid", return false);
        }

        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        return true;
    }
};

class UnpadDynamicBatchFlashAttentionNdKernel : public UnpadFlashAttentionKernel {
public:
    explicit UnpadDynamicBatchFlashAttentionNdKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : UnpadFlashAttentionKernel(kernelName, handle)
    {
    }
};

class UnpadFlashAttentionBF16NdKernel : public UnpadFlashAttentionKernel {
public:
    explicit UnpadFlashAttentionBF16NdKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : UnpadFlashAttentionKernel(kernelName, handle)
    {
    }
};

class UnpadFlashAttentionBF16PrefixCacheNdKernel : public UnpadFlashAttentionKernel {
public:
    explicit UnpadFlashAttentionBF16PrefixCacheNdKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : UnpadFlashAttentionKernel(kernelName, handle)
    {
    }
    
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        if (!CheckParamType(launchParam)) { return false; }
        MKI_CHECK(launchParam.GetInTensorCount() == 6, "input num invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 3 ||
                        launchParam.GetInTensor(0).desc.dims.size() == 2,
                    "input 0 dim num invalid", return false);

        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        return true;
    }
};

class UnpadFlashAttentionDecoderNdKernel : public UnpadFlashAttentionKernel {
public:
    explicit UnpadFlashAttentionDecoderNdKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : UnpadFlashAttentionKernel(kernelName, handle)
    {
    }
};

class UnpadFlashAttentionMlaNdKernel : public UnpadFlashAttentionKernel {
public:
    explicit UnpadFlashAttentionMlaNdKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : UnpadFlashAttentionKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        MKI_CHECK(param.dataShapeType == OpParam::UnpadFlashAttention::DataShapeType::TYPE_BSND,
                  "MLA can not support BNSD", return false);
        MKI_CHECK(UnpadFlashAttentionKernel::CanSupport(launchParam), "failed to check support", return false);
        return true;
    }
};

class UnpadFlashAttentionMlaBF16NdKernel : public UnpadFlashAttentionKernel {
public:
    explicit UnpadFlashAttentionMlaBF16NdKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : UnpadFlashAttentionKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        MKI_CHECK(param.dataShapeType == OpParam::UnpadFlashAttention::DataShapeType::TYPE_BSND,
                  "MLA can not support BNSD", return false);
        MKI_CHECK(UnpadFlashAttentionKernel::CanSupport(launchParam), "failed to check support", return false);
        return true;
    }
};

class UnpadFlashAttentionMlaDecoderNdKernel : public UnpadFlashAttentionKernel {
public:
    explicit UnpadFlashAttentionMlaDecoderNdKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : UnpadFlashAttentionKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        MKI_CHECK(param.dataShapeType == OpParam::UnpadFlashAttention::DataShapeType::TYPE_BSND,
                  "MLA can not support BNSD", return false);
        MKI_CHECK(UnpadFlashAttentionKernel::CanSupport(launchParam), "failed to check support", return false);
        return true;
    }
};

class MultiLatentAttentionCombineCacheKernel : public UnpadFlashAttentionKernel {
public:
    explicit MultiLatentAttentionCombineCacheKernel(const std::string &kernelName,
        const BinHandle *handle) noexcept
        : UnpadFlashAttentionKernel(kernelName, handle) {}
    
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        if (!CheckParamType(launchParam)) { return false; }
        MKI_CHECK(launchParam.GetInTensorCount() == 9, "input num invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 3 ||
                    launchParam.GetInTensor(0).desc.dims.size() == 2, "input 0 dim num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        MKI_CHECK(param.dataShapeType == OpParam::UnpadFlashAttention::DataShapeType::TYPE_BSND,
                  "MLA can not support BNSD", return false);
        return true;
    }
};

class MultiLatentAttentionEncoderCombineCacheKernel : public MultiLatentAttentionCombineCacheKernel {
public:
    explicit MultiLatentAttentionEncoderCombineCacheKernel(const std::string &kernelName,
        const BinHandle *handle) noexcept
        : MultiLatentAttentionCombineCacheKernel(kernelName, handle) {}
};

class MultiLatentAttentionEncoderCombineCacheBFKernel : public MultiLatentAttentionCombineCacheKernel {
public:
    explicit MultiLatentAttentionEncoderCombineCacheBFKernel(const std::string &kernelName,
        const BinHandle *handle) noexcept
        : MultiLatentAttentionCombineCacheKernel(kernelName, handle) {}
};

class RelayAttentionKernel : public UnpadFlashAttentionKernel {
public:
    explicit RelayAttentionKernel(const std::string &kernelName,
        const BinHandle *handle) noexcept
        : UnpadFlashAttentionKernel(kernelName, handle)
        {
            launchBufferSize_ = Utils::RoundUp((TILING_PARA_SIZE +
                                               TILING_RELAY_HEAD_SIZE) * sizeof(uint32_t), TILINGMIN);
        }
    
    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        if (!CheckParamType(launchParam)) { return false; }
        auto &param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        auto batch = param.qSeqLen.size();
        auto kvHead = param.kvHead == 0 ? param.headSize : param.kvHead;
        MKI_CHECK(batch <= 60, "batch should not larger than 60", return false);
        MKI_CHECK(kvHead <= 8, "kvhead should not larger than 8", return false);
        auto blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
        MKI_CHECK(batch > 0 && batch <= ND_BATCH_LIMIT, "batch is invalid", return 0);
        auto shareBlockTiling = ((batch * kvHead + blockDim - 1) / blockDim + 1) * RELAY_BLOCK_TILING;
        uint64_t bufferSize =
            Utils::RoundUp(launchBufferSize_ + (shareBlockTiling * blockDim +
                           TILING_PARA_SIZE * (batch - 1)) * sizeof(uint32_t), TILINGMIN);
        return bufferSize;
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        if (!CheckParamType(launchParam)) { return false; }
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "input num invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 3 ||
                    launchParam.GetInTensor(0).desc.dims.size() == 2, "input 0 dim num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        MKI_CHECK(param.dataShapeType == OpParam::UnpadFlashAttention::DataShapeType::TYPE_BSND,
                  "Relay Attention can not support BNSD", return false);
        return true;
    }
};

REG_KERNEL_BASE(UnpadFlashAttentionNdKernel);
REG_KERNEL_BASE(UnpadFlashAttentionRazorFusionKernel);
REG_KERNEL_BASE(UnpadDynamicBatchFlashAttentionNdKernel);
REG_KERNEL_BASE(UnpadFlashAttentionDecoderNdKernel);
REG_KERNEL_BASE(UnpadFlashAttentionBF16NdKernel);
REG_KERNEL_BASE(UnpadFlashAttentionBF16PrefixCacheNdKernel);
REG_KERNEL_BASE(UnpadFlashAttentionMlaNdKernel);
REG_KERNEL_BASE(UnpadFlashAttentionMlaBF16NdKernel);
REG_KERNEL_BASE(UnpadFlashAttentionMlaDecoderNdKernel);
REG_KERNEL_BASE(MultiLatentAttentionEncoderCombineCacheKernel);
REG_KERNEL_BASE(MultiLatentAttentionEncoderCombineCacheBFKernel);
REG_KERNEL_BASE(RelayAttentionKernel);
} // namespace AtbOps