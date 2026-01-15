/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <algorithm>
#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/log/log.h>
#include "atbops/params/params.h"

static constexpr int32_t LONG_SEQ_LEN = 128;
static constexpr int32_t MLA_THRESHOLD = 256;
static constexpr int32_t SWA_COMPRESS_MASK_SIZE = 512;


namespace AtbOps {
using namespace Mki;
class UnpadFlashAttentionOperation : public OperationBase {
public:
    explicit UnpadFlashAttentionOperation(const std::string &opName) noexcept : OperationBase(opName) {}

Kernel *GetFlashAttentionEncoderNdKernel(TensorDType inDtype, const OpParam::UnpadFlashAttention& param,
    bool isMla) const
{
    if (inDtype == TENSOR_DTYPE_BF16) {
        return isMla ? GetKernelByName("UnpadFlashAttentionMlaBF16NdKernel")
                        : GetKernelByName("UnpadFlashAttentionBF16NdKernel");
    } else if (inDtype == TENSOR_DTYPE_INT8) {
        if (param.outDataType == TENSOR_DTYPE_BF16) {
            return GetKernelByName("UnpadFlashAttentionBF16NdKernel");
        } else if (param.outDataType == TENSOR_DTYPE_FLOAT16) {
            return GetKernelByName("UnpadFlashAttentionNdKernel");
        }
    }
    return isMla ? GetKernelByName("UnpadFlashAttentionMlaNdKernel")
                    : GetKernelByName("UnpadFlashAttentionNdKernel");
}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::UnpadFlashAttention), "OpParam is invalid",
                  return nullptr);
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        auto inDtype = launchParam.GetInTensor(0).desc.dtype;
        MKI_CHECK(param.headSize > 0, "headSize is invalid", return nullptr);
        auto embed = (launchParam.GetInTensor(0).desc.dims).at(1) / param.headSize; // headdim
        if (launchParam.GetInTensor(0).desc.dims.size() == DIM_3) {
            embed = (launchParam.GetInTensor(0).desc.dims).at(2); // 2 is head_size dim
        }
        auto isMla = (param.headDimV != 0 && (embed != param.headDimV || embed > MLA_THRESHOLD)) ? true : false;
        switch (param.type) {
            case OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ND:
            case OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION:
            case OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_ND:
                return GetFlashAttentionEncoderNdKernel(inDtype, param, isMla);
            case OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_FP32_ND:
                return isMla ? GetKernelByName("UnpadFlashAttentionMlaBF16NdKernel")
                             : GetKernelByName("UnpadFlashAttentionBF16NdKernel");
            case OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_DECODER_ND:
            case OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION_DECODER:
                return isMla ? GetKernelByName("UnpadFlashAttentionMlaDecoderNdKernel")
                             : GetKernelByName("UnpadFlashAttentionDecoderNdKernel");
            case OpParam::UnpadFlashAttention::UNPAD_ALIBI_FLASH_ATTENTION_ND:
                return isMla ? GetKernelByName("UnpadFlashAttentionMlaNdKernel")
                             : GetKernelByName("UnpadFlashAttentionNdKernel");
            case OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_COMBINE_CACHE:
            case OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_HIGH_PRECISION_COMBINE_CACHE:
                return GetKernelByName("MultiLatentAttentionEncoderCombineCacheKernel");
            case OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND:
                return GetKernelByName("RelayAttentionKernel");
            case OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND:
                return GetKernelByName("UnpadFlashAttentionBF16PrefixCacheNdKernel");
            case OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_RAZOR_FUSION:
                return GetKernelByName("UnpadFlashAttentionRazorFusionKernel");
            default:
                break;
        }
        return nullptr;
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::UnpadFlashAttention), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::UnpadFlashAttention>(specificParam);
        switch (param.type) {
            case OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_COMBINE_CACHE:
            case OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_HIGH_PRECISION_COMBINE_CACHE:
                return DIM_9; // mla COMBINE_CACHE has 9 input
            case OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND:
                return DIM_2; // relayAttention has 2 input
            case OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND:
                return DIM_6;
            case OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_RAZOR_FUSION:
                return DIM_3;
            default:
                break;
        }
        return DIM_12;
    }
    
    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::UnpadFlashAttention), "OpParam is invalid", return 0);
        return DIM_1; // 1 output
    }

    Status InferShapeCheck(const LaunchParam &launchParam) const
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::UnpadFlashAttention), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        MKI_LOG(INFO) << "infer shape param: " << param.headSize << ", type:" << param.type;
        auto scaleType = param.scaleType;
        MKI_CHECK(scaleType == OpParam::UnpadFlashAttention::ScaleType::SCALE_LOGN_FP32 ||
                    scaleType == OpParam::UnpadFlashAttention::ScaleType::SCALE_TOR,
                      "scaleType invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
        if (scaleType == OpParam::UnpadFlashAttention::ScaleType::SCALE_LOGN_FP32) {
            MKI_CHECK(param.type == OpParam::UnpadFlashAttention::Type::UNPAD_FLASH_ATTENTION_DECODER_ND ||
                param.type == OpParam::UnpadFlashAttention::Type::UNPAD_FLASH_ATTENTION_FP32_ND,
                  "logN is cannot support this kernel", return Status::FailStatus(ERROR_INVALID_VALUE));
            if (!CheckEmptyTensor(launchParam.GetInTensor(DIM_3))) {
                MKI_CHECK(launchParam.GetInTensor(DIM_3).desc.dims.at(0) != 0,
                      "logN is cannot support prefix Cache", return Status::FailStatus(ERROR_INVALID_VALUE));
            }
            auto &tensorLog = launchParam.GetInTensor(DIM_11);
            MKI_CHECK(!CheckEmptyTensor(tensorLog), "Input12 should not be null tensor",
                        return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Input12 should not be null tensor"));
            auto inDtype = tensorLog.desc.dtype;
            MKI_CHECK(inDtype == TENSOR_DTYPE_FLOAT, "LogNscale should be float",
                        return Status::FailStatus(ERROR_INVALID_VALUE, "LogNscale should be float"));
            MKI_CHECK(param.dataShapeType == OpParam::UnpadFlashAttention::DataShapeType::TYPE_BSND &&
                      !param.compressHead && param.windowSize == 0,
                        "LogN cannot support BNSD,SWA,compressHead", return Status::FailStatus(ERROR_INVALID_VALUE));
        }
        MKI_CHECK(InferKVHead(launchParam), "kvHead is missing in OpParam for GQA",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "kvHead is missing in OpParam for GQA"));
        return Status::OkStatus();
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        if (!InferShapeCheck(launchParam).Ok()) {
            return Status::FailStatus(ERROR_INVALID_VALUE);
        }
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        switch (param.type) {
            case OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_FP32_ND:
                return InferShapeUnpadFlashAttentionFp32(launchParam, outTensors);
            case OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ND:
            case OpParam::UnpadFlashAttention::UNPAD_ALIBI_FLASH_ATTENTION_ND:
            case OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_DECODER_ND:
                return InferShapeUnpadFlashAttention(launchParam, outTensors);
            case OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_COMBINE_CACHE:
            case OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_HIGH_PRECISION_COMBINE_CACHE:
                return InferShapeMultiLatentAttention(launchParam, outTensors);
            case OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION:
            case OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION_DECODER:
                return InferShapeDynamicBatchFlashAttention(launchParam, outTensors);
            case OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_ND:
                return InferShapeUnpadFlashAttentionEncoder(launchParam, outTensors);
            case OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND:
                return InferShapeUnpadFlashAttentionPrefixCacheEncoder(launchParam, outTensors);
            case OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_RAZOR_FUSION:
                return InferShapeUnpadFlashAttentionRazorFusion(launchParam, outTensors);
            case OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND:
                return InferShapeRelayAttentionDecoder(launchParam, outTensors);
            default: break;
        }
        return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Param is invalid");
    }

private:
    struct ShapeParam {
        int32_t maxQ;
        int32_t maxKv;
        uint32_t batch;
    };

    Status InferOutTensorsUnpadFlashAttention(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(param.headSize > 0, "headSize is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
        if (param.dataShapeType == OpParam::UnpadFlashAttention::DataShapeType::TYPE_BNSD) {
            outTensors[DIM_0].desc = tensorQ.desc;
            if (tensorQ.desc.dims.size() == DIM_4) {
                // q 的shape b,s,n,d encoder
                auto embed = (tensorQ.desc.dims).at(3);
                outTensors[DIM_0].desc = tensorQ.desc;
                outTensors[DIM_0].desc.dims[tensorQ.desc.dims.size() - 1] =
                    param.headDimV == 0 ? embed : param.headDimV;
            } else {
                // q 的shape bs,nd decoder
                auto embed = (launchParam.GetInTensor(0).desc.dims).at(1) / param.headSize;
                outTensors[DIM_0].desc = tensorQ.desc;
                outTensors[DIM_0].desc.dims[tensorQ.desc.dims.size() - 1] =
                    param.headDimV == 0 ? embed * param.headSize : param.headDimV * param.headSize;
            }
        } else {
            if (tensorQ.desc.dims.size() == DIM_3) {
                auto embed = (tensorQ.desc.dims).at(2);
                outTensors[DIM_0].desc = tensorQ.desc;
                outTensors[DIM_0].desc.dims[tensorQ.desc.dims.size() - 1] =
                    param.headDimV == 0 ? embed : param.headDimV;
            } else {
                auto embed = (launchParam.GetInTensor(0).desc.dims).at(1) / param.headSize;
                outTensors[DIM_0].desc = tensorQ.desc;
                outTensors[DIM_0].desc.dims[tensorQ.desc.dims.size() - 1] =
                    param.headDimV == 0 ? embed * param.headSize : param.headDimV * param.headSize;
            }
        }
        if (tensorQ.desc.dtype == TENSOR_DTYPE_INT8) {
            MKI_CHECK(param.outDataType == TENSOR_DTYPE_FLOAT16 || param.outDataType == TENSOR_DTYPE_BF16,
                      "param.outDataType is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
            outTensors[DIM_0].desc.dtype = param.outDataType;
        }

        return Status::OkStatus();
    }

    bool CheckUnpadDynamicBatchFlashAttentionInner(const LaunchParam &launchParam) const
    {
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto &tensorKcache = launchParam.GetInTensor(DIM_1);
        auto &tensorVcache = launchParam.GetInTensor(DIM_2);
        auto &tensorLayerId = launchParam.GetInTensor(DIM_3);
        auto &tensorMask = launchParam.GetInTensor(DIM_4); // mask.shape = [batch, max_seqlen, max_seqlen]
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        if (CheckEmptyTensor(tensorKcache) || CheckEmptyTensor(tensorVcache)) {
            MKI_CHECK(tensorLayerId.desc.dims.size() == 1, "Input3 dim num should be 1", return false);
        } else {
            if (param.dataShapeType == 1) {
                // [batch, headNum, maxSeq,headDim] todo 原bsnd有问题
                if (tensorKcache.desc.dims.size() == DIM_5) {
                    MKI_CHECK(tensorLayerId.desc.dims.size() == 1, "Input3 dim num should be 1", return false);
                } else {
                    MKI_CHECK(CheckEmptyTensor(tensorLayerId), "Input3 should be null tensor", return false);
                }
            } else {
                if (tensorKcache.desc.dims.size() == DIM_3) {
                    MKI_CHECK(tensorLayerId.desc.dims.size() == 1, "Input3 dim num should be 1", return false);
                } else {
                    MKI_CHECK((tensorKcache.desc.dims.size() == tensorQ.desc.dims.size()),
                              "The shape of input0 input1 and input2 should be same", return false);
                    MKI_CHECK(CheckEmptyTensor(tensorLayerId), "Input3 should be null tensor", return false);
                }
            }
        }
        MKI_CHECK_NO_LOG(CheckMask(launchParam, tensorQ, tensorMask), return false);
        return true;
    }

    Status InferShapeMultiLatentAttention(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        MKI_CHECK(CheckMultiLatentAttention(launchParam), "Failed to check launch param",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        OP_TILING_CHECK_STATUS_RETURN(InferOutTensorsUnpadFlashAttention(launchParam, outTensors));
        return Status::OkStatus();
    }

    bool InferKVHead(const LaunchParam &launchParam) const
    {
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        // kvHead不传时默认为0，若不为0表示已经传入
        MKI_CHECK_NO_LOG(param.kvHead == 0, return true);
        // , head * embed) DIM_2
        if (launchParam.GetInTensor(DIM_0).desc.dims.size() == DIM_2) {
            auto headMulEmbed = (launchParam.GetInTensor(DIM_0).desc.dims).at(DIM_1);
            auto embed = headMulEmbed / param.headSize;
            auto kvHeadMulEmbedIdx = launchParam.GetInTensor(DIM_1).desc.dims.size() - 1;
            auto isMla = (param.headDimV != 0 && (embed != param.headDimV || embed > MLA_THRESHOLD)) ? true : false;
            if (launchParam.GetInTensor(DIM_1).desc.dims.size() > 0 && !isMla) {
                auto kvHeadMulEmbed = (launchParam.GetInTensor(DIM_1).desc.dims).at(kvHeadMulEmbedIdx);
                if (kvHeadMulEmbed > embed) {
                    // kvHead不传时默认为非GQA场景，否则返回kvHead缺失信息
                    MKI_CHECK_NO_LOG(headMulEmbed == kvHeadMulEmbed, return false);
                }
            }
        }
        // , head, embed) DIM_3 / BNSD DIM_4
        if (launchParam.GetInTensor(DIM_0).desc.dims.size() == DIM_3 ||
            launchParam.GetInTensor(DIM_0).desc.dims.size() == DIM_4) {
            auto realKVHeadIdx = launchParam.GetInTensor(DIM_0).desc.dims.size() == DIM_3 ?
                                 launchParam.GetInTensor(DIM_1).desc.dims.size() - 2 :
                                 launchParam.GetInTensor(DIM_1).desc.dims.size() - 3;
            auto embedIdx = launchParam.GetInTensor(DIM_0).desc.dims.size() - 1;
            auto embed = (launchParam.GetInTensor(DIM_0).desc.dims).at(embedIdx);
            auto isMla = (param.headDimV != 0 && (embed != param.headDimV || embed > MLA_THRESHOLD)) ? true : false;
            if (launchParam.GetInTensor(DIM_1).desc.dims.size() > 0 && !isMla) {
                auto realKVHead = (launchParam.GetInTensor(DIM_1).desc.dims).at(realKVHeadIdx);
                // kvHead不传时默认为非GQA场景，否则返回kvHead缺失信息
                MKI_CHECK_NO_LOG(param.headSize == realKVHead, return false);
            }
        }
        return true;
    }

    Status InferShapeUnpadFlashAttentionFp32(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        MKI_CHECK(CheckUnpadFlashAttentionFp32(launchParam), "Failed to check launch param",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        OP_TILING_CHECK_STATUS_RETURN(InferOutTensorsUnpadFlashAttention(launchParam, outTensors));
        return Status::OkStatus();
    }

    Status InferShapeRelayAttentionDecoder(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        MKI_CHECK(CheckRelayAttention(launchParam), "Failed to check launch param",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        OP_TILING_CHECK_STATUS_RETURN(InferOutTensorsUnpadFlashAttention(launchParam, outTensors));
        return Status::OkStatus();
    }

    Status InferShapeUnpadFlashAttention(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        MKI_CHECK(CheckUnpadFlashAttention(launchParam), "Failed to check launch param",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        OP_TILING_CHECK_STATUS_RETURN(InferOutTensorsUnpadFlashAttention(launchParam, outTensors));
        return Status::OkStatus();
    }

    Status InferShapeDynamicBatchFlashAttention(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        MKI_CHECK(CheckUnpadDynamicBatchFlashAttention(launchParam), "Failed to check launch param",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        OP_TILING_CHECK_STATUS_RETURN(InferOutTensorsUnpadFlashAttention(launchParam, outTensors));
        return Status::OkStatus();
    }

    Status InferShapeUnpadFlashAttentionRazorFusion(const LaunchParam &launchParam,
        SVector<Tensor> &outTensors) const
    {
        MKI_CHECK(CheckUnpadFlashAttentionRazorFusion(launchParam), "Failed to check launch param",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        OP_TILING_CHECK_STATUS_RETURN(InferOutTensorsUnpadFlashAttention(launchParam, outTensors));
        return Status::OkStatus();
    }

    Status InferShapeUnpadFlashAttentionPrefixCacheEncoder(const LaunchParam &launchParam,
        SVector<Tensor> &outTensors) const
    {
        MKI_CHECK(CheckUnpadFlashAttentionPrefixCacheEncoder(launchParam), "Failed to check launch param",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        OP_TILING_CHECK_STATUS_RETURN(InferOutTensorsUnpadFlashAttention(launchParam, outTensors));
        return Status::OkStatus();
    }

    Status InferShapeUnpadFlashAttentionEncoder(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        MKI_CHECK(CheckUnpadFlashAttentionEncoder(launchParam), "Failed to check launch param",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        OP_TILING_CHECK_STATUS_RETURN(InferOutTensorsUnpadFlashAttention(launchParam, outTensors));
        return Status::OkStatus();
    }

    bool MlaCheckQuant(const LaunchParam &launchParam) const
    {
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto &tensorKcache = launchParam.GetInTensor(DIM_1);
        auto &tensorQKscale = launchParam.GetInTensor(DIM_4);
        auto &tensorQKoffset = launchParam.GetInTensor(DIM_5);
        auto &tensorPVscale = launchParam.GetInTensor(DIM_6);
        auto &tensorPVoffset = launchParam.GetInTensor(DIM_7);
        auto &tensorOfflineQuant = launchParam.GetInTensor(DIM_8);
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        MKI_CHECK(param.dataShapeType == 0, "only support BSND", return false);
        MKI_CHECK(param.scaleType == 0, "not support LOGN", return false);
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_INT8 && tensorKcache.desc.dtype == TENSOR_DTYPE_INT8,
                  "Q, K should be int8", return false);
        MKI_CHECK(!CheckEmptyTensor(tensorQKscale) && !CheckEmptyTensor(tensorPVscale), "need SymmetricQuant scales",
                  return false);
        MKI_CHECK(CheckEmptyTensor(tensorQKoffset) && CheckEmptyTensor(tensorPVoffset), "only support SymmetricQuant",
                  return false);

        MKI_CHECK(tensorQKscale.desc.dtype == TENSOR_DTYPE_FLOAT && tensorPVscale.desc.dtype == TENSOR_DTYPE_FLOAT,
                  "QKscale, KVscale should be float", return false);

        MKI_CHECK(tensorQKscale.desc.dims[DIM_0] == param.headSize, "QKscale dim invalid, should be " << param.headSize,
                  return false);
        MKI_CHECK(tensorPVscale.desc.dims[DIM_0] == param.headSize,
                  "tensorPVscale dim invalid, should be " << param.headSize, return false);
        if (param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_QKV_OFFLINE) {
            MKI_CHECK(!CheckEmptyTensor(tensorOfflineQuant), "need offline quant scales", return false);
            MKI_CHECK(tensorOfflineQuant.desc.dtype == TENSOR_DTYPE_FLOAT, "offline scale should be float",
                      return false);
            MKI_CHECK(tensorOfflineQuant.desc.dims[DIM_0] == param.headSize,
                      "OfflineQuant scale dim invalid, should be " << param.headSize, return false);
        }
        return true;
    }

    bool CheckQuant(const LaunchParam &launchParam) const
    {
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto &tensorKcache = launchParam.GetInTensor(DIM_1);
        auto &tensorVcache = launchParam.GetInTensor(DIM_2);
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        MKI_CHECK(param.dataShapeType == 0, "only support BSND", return false);
        MKI_CHECK(param.scaleType == 0, "not support LOGN", return false);
        const int32_t index1 = 1;
        const int32_t index2 = 2;
        auto embed = (launchParam.GetInTensor(0).desc.dims).at(index1) / param.headSize;
        if (launchParam.GetInTensor(0).desc.dims.size() == DIM_3) {
            embed = (launchParam.GetInTensor(0).desc.dims).at(index2);
        }
        auto isMla = (param.headDimV != 0 && (embed != param.headDimV || embed > MLA_THRESHOLD)) ? true : false;
        MKI_CHECK(!isMla, "mla not support in quant", return false);
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_INT8 && tensorKcache.desc.dtype == TENSOR_DTYPE_INT8 &&
                      tensorVcache.desc.dtype == TENSOR_DTYPE_INT8,
                  "Q, K, V should be int8", return false);

        auto &tensorQKscale = launchParam.GetInTensor(DIM_6);
        auto &tensorQKoffset = launchParam.GetInTensor(DIM_7);
        auto &tensorPVscale = launchParam.GetInTensor(DIM_8);
        auto &tensorPVoffset = launchParam.GetInTensor(DIM_9);

        MKI_CHECK(!CheckEmptyTensor(tensorQKscale) && !CheckEmptyTensor(tensorPVscale), "need SymmetricQuant scales",
                  return false);
        MKI_CHECK(CheckEmptyTensor(tensorQKoffset) && CheckEmptyTensor(tensorPVoffset), "only support SymmetricQuant",
                  return false);

        MKI_CHECK(tensorQKscale.desc.dtype == TENSOR_DTYPE_FLOAT && tensorPVscale.desc.dtype == TENSOR_DTYPE_FLOAT,
                  "QKscale, KVscale should be float", return false);

        MKI_CHECK(tensorQKscale.desc.dims[DIM_0] == param.headSize, "QKscale dim invalid, should be " << param.headSize,
                  return false);
        MKI_CHECK(tensorPVscale.desc.dims[DIM_0] == param.headSize,
                  "tensorPVscale dim invalid, should be " << param.headSize, return false);
        if (param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_QKV_OFFLINE) {
            auto &tensorOfflineQuant = launchParam.GetInTensor(DIM_10);
            MKI_CHECK(!CheckEmptyTensor(tensorOfflineQuant), "need offline quant scales", return false);
            MKI_CHECK(tensorOfflineQuant.desc.dtype == TENSOR_DTYPE_FLOAT, "offline scale should be float",
                      return false);
            MKI_CHECK(tensorOfflineQuant.desc.dims[DIM_0] == param.headSize,
                      "OfflineQuant scale dim invalid, should be " << param.headSize, return false);
        }
        return true;
    }

    bool CheckUnpadFlashAttentionFp32(const LaunchParam &launchParam) const
    {
        auto &tensorQ = launchParam.GetInTensor(DIM_0);      // Q.shape = [batch*seqlen, hiddensize]
        auto &tensorVcache = launchParam.GetInTensor(DIM_2); // V.shape = [layerNum, batch, max_seqlen, hiddensize]
        auto &tensorKcache = launchParam.GetInTensor(DIM_1); // K.shape = [layerNum, batch, max_seqlen, hiddensize]
        auto &tensorLayerId = launchParam.GetInTensor(DIM_3);
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorQ.desc.dtype == TENSOR_DTYPE_BF16 ||
                      tensorQ.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input0 dtype " << GetStrWithDType(tensorQ.desc.dtype)
                                  << " invalid, should be float16 or bfloat16 or int8",
                  return false);
        MKI_CHECK(tensorQ.desc.dtype == tensorKcache.desc.dtype && tensorKcache.desc.dtype == tensorVcache.desc.dtype,
                  "tensorQ K V must be the same dtype,here Q dtype is "
                      << GetStrWithDType(tensorQ.desc.dtype) << ", K dtype is "
                      << GetStrWithDType(tensorVcache.desc.dtype) << ", and V dtype is "
                      << GetStrWithDType(tensorVcache.desc.dtype),
                  return false);
        MKI_CHECK(CheckEmptyTensor(tensorLayerId) || tensorLayerId.desc.dims.size() == 1,
                  "Input3 dim num should be 1 or null tensor", return false);
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        MKI_CHECK(CheckMask(launchParam, tensorQ, launchParam.GetInTensor(DIM_4)), "check mask shape fail",
                  return false);
        if (param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_QKV_ONLINE ||
            param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_QKV_OFFLINE) {
            MKI_CHECK(CheckQuant(launchParam), "check quant type fail", return false);
        }
        auto scaleType = param.scaleType;
        if (scaleType == OpParam::UnpadFlashAttention::ScaleType::SCALE_LOGN_FP32) {
            auto &tensorLog = launchParam.GetInTensor(DIM_11);
            int64_t dimLog = tensorLog.desc.dims.at(0);
            int64_t dimQ = tensorQ.desc.dims.at(0);
            auto batchSize = param.qSeqLen.size();
            MKI_CHECK(dimQ > 0 && dimLog > 0, "input 12 num invalid", return false);
            MKI_CHECK(batchSize * static_cast<unsigned long>(dimLog) >=
                static_cast<unsigned long>(dimQ),
                    "input 12 num invalid", return false);
        }
        return true;
    }

    bool CheckUnpadFAKvBatchwise(const OpParam::UnpadFlashAttention &param, const Tensor &tensorQ) const
    {
        uint32_t batch = param.kvSeqLen.size();
        MKI_CHECK(param.kTensorList.size() == batch && param.vTensorList.size() == batch, "kv batch num invalid",
                  return false);
        auto kShape = param.kTensorList[0].desc.dims;
        if (param.dataShapeType == 1) {
            // todo N,S,D
            MKI_CHECK(kShape.size() == 3,
                      "kv batch shape size wrong,bnsd tensorList shape is [headNum,maxSeq,headDim] ", return false);
        } else {
            // max_kv_seqlen,hiddenSize
            MKI_CHECK(kShape.size() == 2, "kv batch shape size wrong", return false);
        }

        for (size_t i = 0; i < batch; ++i) {
            MKI_CHECK(param.kTensorList[i].data != nullptr && param.vTensorList[i].data != nullptr,
                      "kv batchwise ptr cannot be nullptr", return false);
            MKI_CHECK(param.kTensorList[i].desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                          param.kTensorList[i].desc.dtype == TENSOR_DTYPE_BF16 ||
                          param.kTensorList[i].desc.dtype == TENSOR_DTYPE_INT8,
                      "k dtype invalid, should be either fp16 or bf16 or int8", return false);
            MKI_CHECK(param.vTensorList[i].desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                          param.vTensorList[i].desc.dtype == TENSOR_DTYPE_BF16 ||
                          param.vTensorList[i].desc.dtype == TENSOR_DTYPE_INT8,
                      "v dtype invalid, should be either fp16 or bf16 or int8", return false);
            MKI_CHECK(param.kTensorList[i].desc.dtype == tensorQ.desc.dtype &&
                          param.vTensorList[i].desc.dtype == tensorQ.desc.dtype,
                      "Q K V dtype should be the same", return false);
        }
        return true;
    }

    bool CheckMultiLatentAttention(const LaunchParam &launchParam) const
    {
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorQ.desc.dtype == TENSOR_DTYPE_BF16 ||
                      tensorQ.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input0 dtype " << GetStrWithDType(tensorQ.desc.dtype) << " invalid, should be float16 or BF16",
                  return false);
        MKI_CHECK(tensorQ.desc.dims.size() == DIM_3 || tensorQ.desc.dims.size() == DIM_2,
                  "Input0 dim num " << tensorQ.desc.dims.size() << " invalid, dim should be 2 or 3", return false);
        auto &tensorK = launchParam.GetInTensor(DIM_1);
        MKI_CHECK(tensorK.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorK.desc.dtype == TENSOR_DTYPE_BF16 ||
                      tensorK.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input1 dtype " << tensorK.desc.dtype << " invalid, should be float16 or BF16", return false);
        MKI_CHECK(tensorK.desc.dims.size() == DIM_4 || tensorK.desc.dims.size() == DIM_3 ||
                  tensorK.desc.dims.size() == DIM_2,
                  "Input1 dim num " << tensorK.desc.dims.size() << " invalid, dim should be 2 or 3 or 4", return false);
        MKI_CHECK(tensorQ.desc.dtype == tensorK.desc.dtype,
                  "tensorQ K must be the same dtype, here Q dtype is "
                      << GetStrWithDType(tensorQ.desc.dtype) << ", K dtype is "
                      << GetStrWithDType(tensorK.desc.dtype),
                  return false);
        MKI_CHECK(param.maskType != OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI &&
                  param.maskType != OpParam::UnpadFlashAttention::MASK_TYPE_LOOK_AHEAD,
                  "MLA kernel not support this mask type", return false);
        auto &tensorLayerId = launchParam.GetInTensor(DIM_2);
        MKI_CHECK(CheckEmptyTensor(tensorLayerId) || tensorLayerId.desc.dims.size() == 1,
                  "Input2 dim num should be 1 or null tensor", return false);
        auto ret = CheckMask(launchParam, tensorQ, launchParam.GetInTensor(DIM_3));
        MKI_CHECK(ret, "mask invalid.", return false);
        if (param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_QKV_OFFLINE ||
            param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_QKV_ONLINE) {
            MKI_CHECK(MlaCheckQuant(launchParam), "check quant type fail", return false);
        }
        return true;
    }

    bool CheckAttentionCommon(const OpParam::UnpadFlashAttention &param, const Tensor &tensorQ) const
    {
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorQ.desc.dtype == TENSOR_DTYPE_BF16,
                  "Input0 dtype " << GetStrWithDType(tensorQ.desc.dtype)
                                  << " invalid, should be float16 or bfloat16",
                  return false);
        MKI_CHECK(param.headSize > 0, "headSize is invalid", return false);
        MKI_CHECK((param.kvHead > 0 && param.kvHead <= param.headSize && param.headSize % param.kvHead == 0) ||
                        (param.kvHead == 0),
                    "kvHead is invalid", return false);
        MKI_CHECK(param.qSeqLen.size() == param.kvSeqLen.size(), "qSeqLen size Not equal to kvSeqLen",
                    return false);
        MKI_CHECK(param.qSeqLen.data() != nullptr, "qSeq cannot be nullptr",
                    return false);
        MKI_CHECK(param.kvSeqLen.data() != nullptr, "kvSeq cannot be nullptr",
                    return false);
        MKI_CHECK(param.qSeqLen.size() > 0 && param.qSeqLen.size() <= INT32_MAX, "batch is invalid",
              return false);
        MKI_CHECK(param.isTriuMask == 0 || param.isTriuMask == 1, "param isTriuMask is invalid",
                    return false);
        MKI_CHECK(param.cacheType == OpParam::UnpadFlashAttention::CACHE_TYPE_NORM,
                  "param cacheType should be CACHE_TYPE_NORM",
                  return false);
        return true;
    }

    bool CheckUnpadRelayKvBatchwise(const OpParam::UnpadFlashAttention &param, const Tensor &tensorQ) const
    {
        auto kShape = param.kTensorList[0].desc.dims;
        MKI_CHECK(kShape.size() == DIM_3 || kShape.size() == 2,
                  "kv shape wrong, dims should be 2 or 3, support [S, N, D] or [S, N * D]", return false);
        if (kShape.size() == DIM_3) {
            MKI_CHECK(kShape[2] <= 256, "D should not bigger than 256", return false);
        } else {
            long dimention = kShape[1] / param.headSize;
            MKI_CHECK(dimention <= 256, "D should not bigger than 256", return false);
        }
        for (size_t i = 0; i < param.kvSeqLen.size(); ++i) {
            MKI_CHECK(param.kTensorList[i].data != nullptr && param.vTensorList[i].data != nullptr,
                      "kv batchwise ptr cannot be nullptr", return false);
            MKI_CHECK(param.kTensorList[i].desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                          param.kTensorList[i].desc.dtype == TENSOR_DTYPE_BF16,
                      "k dtype invalid, should be either fp16 or bf16", return false);
            MKI_CHECK(param.vTensorList[i].desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                          param.vTensorList[i].desc.dtype == TENSOR_DTYPE_BF16,
                      "v dtype invalid, should be either fp16 or bf16", return false);
            MKI_CHECK(param.kTensorList[i].desc.dtype == tensorQ.desc.dtype &&
                          param.vTensorList[i].desc.dtype == tensorQ.desc.dtype,
                      "Q and K V dtype should be the same", return false);
        }
        auto kshareShape = param.kShareTensorList[0].desc.dims;
        MKI_CHECK(kshareShape.size() == DIM_3 || kshareShape.size() == 2,
                    "kv share shape wrong, dims should be 2 or 3, support [S, N, D] or [S, N * D]", return false);
        if (kshareShape.size() == DIM_3) {
            MKI_CHECK(kshareShape[2] <= 256, "D should not bigger than 256", return false);
        } else {
            int headDim = kshareShape[1] / param.headSize;
            MKI_CHECK(headDim <= 256, "D should not bigger than 256", return false);
        }
        for (size_t i = 0; i < param.kvShareLen.size(); ++i) {
            MKI_CHECK(param.kShareTensorList[i].data != nullptr && param.vShareTensorList[i].data != nullptr,
                    "kv share batchwise ptr cannot be nullptr", return false);
            MKI_CHECK(param.kShareTensorList[i].desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                        param.kShareTensorList[i].desc.dtype == TENSOR_DTYPE_BF16,
                    "k share dtype invalid, should be either fp16 or bf16", return false);
            MKI_CHECK(param.vShareTensorList[i].desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                        param.vShareTensorList[i].desc.dtype == TENSOR_DTYPE_BF16,
                    "v share dtype invalid, should be either fp16 or bf16", return false);
            MKI_CHECK(param.kShareTensorList[i].desc.dtype == tensorQ.desc.dtype &&
                        param.vShareTensorList[i].desc.dtype == tensorQ.desc.dtype,
                    "Q and K V share dtype should be the same", return false);
        }
        return true;
    }

    bool CheckRelayAttention(const LaunchParam &launchParam) const
    {
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        uint32_t batch = param.kvSeqLen.size();
        MKI_CHECK(CheckAttentionCommon(param, tensorQ), "Flash Attention common check failed", return false);
        MKI_CHECK(param.kTensorList.size() == param.vTensorList.size() &&
                  batch == param.vTensorList.size(), "k and v cache have different batch number",
                  return false);
        MKI_CHECK(param.kShareTensorList.size() == param.vShareTensorList.size() &&
                  param.kvShareLen.size() == param.kShareTensorList.size(),
                  "k share and v share cache have different batch number",
                  return false);
        MKI_CHECK(param.kTensorList.size() > 0 && param.kvShareLen.size() > 0, "kv and kv share have least one batch",
                  return false);
        MKI_CHECK(param.batchRunStatus.size() == 0, "relay attention not support dynamic batch", return false);
        MKI_CHECK(param.isClamp == 0, "relay attention not support clamp", return false);
        MKI_CHECK(CheckUnpadRelayKvBatchwise(param, tensorQ), "Relay Attention kv check failed", return false);
        MKI_CHECK(param.dataShapeType == 0, "only support BSND", return false);
        MKI_CHECK(param.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_NONE,
            "only support no mask", return false);
        MKI_CHECK(param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_UNDEFINED,
                  "relay attention not support quant", return false);
        return true;
    }

    bool CheckUnpadFlashAttention(const LaunchParam &launchParam) const
    {
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        auto &tensorQ = launchParam.GetInTensor(DIM_0); // Q.shape = [batch*seqlen, hiddensize]
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorQ.desc.dtype == TENSOR_DTYPE_BF16 ||
                      tensorQ.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input0 dtype " << GetStrWithDType(tensorQ.desc.dtype)
                                  << " invalid, should be float16 or bfloat16 or int8",
                  return false);
        auto &tensorKcache = launchParam.GetInTensor(DIM_1); // K.shape = [layerNum, batch, max_seqlen, hiddensize]
        auto &tensorVcache = launchParam.GetInTensor(DIM_2); // V.shape = [layerNum, batch, max_seqlen, hiddensize]
        if (CheckEmptyTensor(tensorKcache) || CheckEmptyTensor(tensorVcache)) {
            MKI_CHECK(CheckEmptyTensor(tensorKcache) && CheckEmptyTensor(tensorVcache),
                      "normal k and v should both be empty tensor if batches are split", return false);
            MKI_CHECK(CheckUnpadFAKvBatchwise(param, tensorQ), "kv batchwise settings invalid", return false);
        } else {
            if (param.dataShapeType == 1) {
                MKI_CHECK(CheckBNSD(tensorKcache, tensorVcache, tensorQ), "check bnsd invalid", return false);
            } else {
                MKI_CHECK(CheckBSND(tensorKcache, tensorVcache, tensorQ), "check bsnd invalid", return false);
            }
        }
        auto &tensorLayerId = launchParam.GetInTensor(DIM_3);
        MKI_CHECK(CheckEmptyTensor(tensorLayerId) || tensorLayerId.desc.dims.size() == 1,
                  "Input3 dim num should be 1 or null tensor", return false);
        auto ret = CheckMask(launchParam, tensorQ, launchParam.GetInTensor(DIM_4));
        MKI_CHECK(ret, "check mask shape fail", return false);
        if (param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_QKV_ONLINE ||
            param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_QKV_OFFLINE) {
            MKI_CHECK(CheckQuant(launchParam), "check quant type fail", return false);
        }
        auto scaleType = param.scaleType;
        if (scaleType == OpParam::UnpadFlashAttention::ScaleType::SCALE_LOGN_FP32) {
            MKI_CHECK(param.type == OpParam::UnpadFlashAttention::Type::UNPAD_FLASH_ATTENTION_DECODER_ND,
                  "logN is cannot support this kernel", return false);
            auto batchSize = param.qSeqLen.size();
            auto &tensorLog = launchParam.GetInTensor(DIM_11);
            int64_t dimLog = tensorLog.desc.dims.at(0);
            MKI_CHECK(dimLog > 0, "input 12 num invalid", return false);
            MKI_CHECK(static_cast<unsigned long>(dimLog) >= batchSize,
                      "input 12 num invalid", return false);
        }
        return true;
    }

    bool CheckBSND(const Mki::Tensor &tensorKcache, const Mki::Tensor &tensorVcache, const Mki::Tensor &tensorQ) const
    {
        static const size_t KV_CACHE_DIM_NUM = 4;
        MKI_CHECK(tensorKcache.desc.dims.size() == KV_CACHE_DIM_NUM,
                  "Input1 dim num " << tensorKcache.desc.dims.size() << " invalid, should be " << KV_CACHE_DIM_NUM,
                  return false);
        MKI_CHECK(tensorVcache.desc.dims.size() == KV_CACHE_DIM_NUM,
                  "Input2 dim num " << tensorVcache.desc.dims.size() << " invalid, should be " << KV_CACHE_DIM_NUM,
                  return false);
        auto batch = tensorKcache.desc.dims[DIM_1];
        auto maxSeqlen = tensorKcache.desc.dims[DIM_2];
        MKI_CHECK(tensorVcache.desc.dims[DIM_1] == batch && tensorVcache.desc.dims[DIM_2] == maxSeqlen,
                  "Shape of input1 should be batch  input2 should be maxSeqlen", return false);
        MKI_CHECK(tensorKcache.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorKcache.desc.dtype == TENSOR_DTYPE_BF16 ||
                      tensorKcache.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input1 dtype " << GetStrWithDType(tensorKcache.desc.dtype)
                                  << " invalid, should be float16 or bfloat16 or int8",
                  return false);
        MKI_CHECK(tensorVcache.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorVcache.desc.dtype == TENSOR_DTYPE_BF16 ||
                      tensorVcache.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input2 dtype " << GetStrWithDType(tensorVcache.desc.dtype)
                                  << " invalid, should be float16 or bfloat16 or int8",
                  return false);
        MKI_CHECK(tensorQ.desc.dtype == tensorKcache.desc.dtype && tensorKcache.desc.dtype == tensorVcache.desc.dtype,
                  "tensorQ K V must be the same dtype,here Q dtype is "
                      << GetStrWithDType(tensorQ.desc.dtype) << ", K dtype is "
                      << GetStrWithDType(tensorVcache.desc.dtype) << ", and V dtype is "
                      << GetStrWithDType(tensorVcache.desc.dtype),
                  return false);
        return true;
    }

    bool CheckBNSD(const Mki::Tensor &tensorKcache, const Mki::Tensor &tensorVcache, const Mki::Tensor &tensorQ) const
    {
        static const size_t BNSD_KV_CACHE_DIM_NUM = 5;
        MKI_CHECK(tensorKcache.desc.dims.size() == BNSD_KV_CACHE_DIM_NUM,
                  "BNSD:Input1 dim num " << tensorKcache.desc.dims.size() << " invalid, should be "
                                         << BNSD_KV_CACHE_DIM_NUM,
                  return false);
        MKI_CHECK(tensorVcache.desc.dims.size() == BNSD_KV_CACHE_DIM_NUM,
                  "BNSD:Input2 dim num " << tensorVcache.desc.dims.size() << " invalid, should be "
                                         << BNSD_KV_CACHE_DIM_NUM,
                  return false);
        auto maxSeq = tensorKcache.desc.dims[DIM_3];
        auto batch = tensorKcache.desc.dims[DIM_1];
        auto headNum = tensorKcache.desc.dims[DIM_2];
        MKI_CHECK(tensorKcache.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorKcache.desc.dtype == TENSOR_DTYPE_BF16 ||
                      tensorKcache.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input1 dtype " << GetStrWithDType(tensorKcache.desc.dtype)
                                  << " invalid, should be float16 or bfloat16 or int8",
                  return false);
        MKI_CHECK(tensorVcache.desc.dims[DIM_1] == batch && tensorVcache.desc.dims[DIM_3] == maxSeq &&
                      tensorVcache.desc.dims[DIM_2] == headNum,
                  "Shape of input0/input1 should be same", return false);
        MKI_CHECK(tensorVcache.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorVcache.desc.dtype == TENSOR_DTYPE_BF16 ||
                      tensorVcache.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input2 dtype " << GetStrWithDType(tensorVcache.desc.dtype)
                                  << "data type invalid, should be float16 or bfloat16 or int8",
                  return false);
        MKI_CHECK(tensorQ.desc.dtype == tensorKcache.desc.dtype && tensorKcache.desc.dtype == tensorVcache.desc.dtype,
                  "tensorQ K V must be the same dtype,here Q Tensor dtype is "
                      << GetStrWithDType(tensorQ.desc.dtype) << ", K dtype is "
                      << GetStrWithDType(tensorVcache.desc.dtype) << ", and V dtype is "
                      << GetStrWithDType(tensorVcache.desc.dtype),
                  return false);
        return true;
    }

    bool CheckUnpadDynamicBatchFlashAttention(const LaunchParam &launchParam) const
    {
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        auto &tensorQ = launchParam.GetInTensor(DIM_0); // Q.shape = [batch*seqlen, hiddensize]
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorQ.desc.dtype == TENSOR_DTYPE_BF16 ||
                      tensorQ.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input0 dtype " << GetStrWithDType(tensorQ.desc.dtype)
                                  << " invalid, should be float16 or bf16 or int8",
                  return false);
        auto &tensorKcache = launchParam.GetInTensor(DIM_1); // K.shape = [batch, max_seqlen, hiddensize]
        auto &tensorVcache = launchParam.GetInTensor(DIM_2); // V.shape = [batch, max_seqlen, hiddensize]
        if (CheckEmptyTensor(tensorKcache) || CheckEmptyTensor(tensorVcache)) {
            MKI_CHECK(CheckEmptyTensor(tensorKcache) && CheckEmptyTensor(tensorVcache),
                      " Normal k and v should both be empty tensor if batches are split", return false);
            MKI_CHECK(CheckUnpadFAKvBatchwise(param, tensorQ), " Kv batchwise settings invalid", return false);
        } else {
            MKI_CHECK(tensorKcache.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorKcache.desc.dtype == TENSOR_DTYPE_BF16 ||
                          tensorKcache.desc.dtype == TENSOR_DTYPE_INT8,
                      "Input1 dtype " << GetStrWithDType(tensorKcache.desc.dtype)
                                      << " invalid, should be float16 or bf16 or int8",
                      return false);
            MKI_CHECK(tensorVcache.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorVcache.desc.dtype == TENSOR_DTYPE_BF16 ||
                          tensorVcache.desc.dtype == TENSOR_DTYPE_INT8,
                      "Input2 dtype " << GetStrWithDType(tensorVcache.desc.dtype)
                                      << " invalid, should be float16 or bf16 or int8",
                      return false);
            MKI_CHECK(tensorVcache.desc.dims.size() == tensorKcache.desc.dims.size(),
                      "Input1 dim num " << tensorKcache.desc.dims.size() << " does not match input2 dim num"
                                        << tensorVcache.desc.dims.size(),
                      return false);
            if (param.dataShapeType == 1) {
                MKI_CHECK(tensorKcache.desc.dims.size() == DIM_4 || tensorKcache.desc.dims.size() == DIM_5,
                          "Input1 dim num " << tensorKcache.desc.dims.size() << " invalid, should be 2 or 3",
                          return false);
                MKI_CHECK(tensorVcache.desc.dims.size() == DIM_4 || tensorVcache.desc.dims.size() == DIM_5,
                          "Input2 dim num " << tensorVcache.desc.dims.size() << " invalid, should be 2 or 3",
                          return false);
            } else {
                MKI_CHECK(tensorKcache.desc.dims.size() == DIM_3 || tensorKcache.desc.dims.size() == DIM_2,
                          "Input1 dim num " << tensorKcache.desc.dims.size() << " invalid, should be 2 or 3",
                          return false);
                MKI_CHECK(tensorVcache.desc.dims.size() == DIM_3 || tensorVcache.desc.dims.size() == DIM_2,
                          "Input2 dim num " << tensorVcache.desc.dims.size() << " invalid, should be 2 or 3",
                          return false);
            }
        }
        MKI_CHECK(CheckUnpadDynamicBatchFlashAttentionInner(launchParam), "param check invalid ", return false);
        if (param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_QKV_ONLINE ||
            param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_QKV_OFFLINE) {
            MKI_CHECK(CheckQuant(launchParam), "check quant type fail", return false);
        }
        return true;
    }

    bool CheckUnpadFlashAttentionRazorFusion(const LaunchParam &launchParam) const
    {
        // Q.shape = [num_tokens, num_heads, head_size] or [num_tokens, num_heads,* head_size]
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorQ.desc.dtype == TENSOR_DTYPE_BF16,
                  "Input0 dtype " << GetStrWithDType(tensorQ.desc.dtype)
                                  << " invalid, should be float16 or BF16",
                  return false);
        MKI_CHECK(tensorQ.desc.dims.size() == DIM_3 || tensorQ.desc.dims.size() == DIM_2,
                  "Input0 dim num " << tensorQ.desc.dims.size() << " invalid, dim should be 2 or 3", return false);

        auto &tensorK = launchParam.GetInTensor(DIM_1);
        MKI_CHECK(tensorK.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorK.desc.dtype == TENSOR_DTYPE_BF16,
                  "Input1 dtype " << tensorK.desc.dtype << " invalid, should be float16 or BF16", return false);
        auto &tensorV = launchParam.GetInTensor(DIM_2);
        MKI_CHECK(tensorV.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorV.desc.dtype == TENSOR_DTYPE_BF16,
                  "Input2 dtype " << tensorV.desc.dtype << " invalid, should be float16 or BF16", return false);
        MKI_CHECK(tensorK.desc.dims.size() == DIM_2 || tensorK.desc.dims.size() == DIM_3,
                  "Input1 dim num " << tensorK.desc.dims.size() << " invalid, dim should be 2 or 3", return false);
        MKI_CHECK(tensorV.desc.dims.size() == DIM_2 || tensorV.desc.dims.size() == DIM_3,
                  "Input2 dim num " << tensorV.desc.dims.size() << " invalid, dim should be 2 or 3", return false);
        MKI_CHECK(param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_UNDEFINED,
            "prefix based FlashAttention don't support quant", return false);
        MKI_CHECK(param.scaleType == OpParam::UnpadFlashAttention::ScaleType::SCALE_TOR,
            "prefix based FlashAttention only support scale tor", return false);
        return true;
    }

    bool CheckUnpadFlashAttentionPrefixCacheEncoder(const LaunchParam &launchParam) const
    {
        // Q.shape = [num_tokens, num_heads, head_size] or [num_tokens, num_heads,* head_size]
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorQ.desc.dtype == TENSOR_DTYPE_BF16,
                  "Input0 dtype " << GetStrWithDType(tensorQ.desc.dtype)
                                  << " invalid, should be float16 or BF16",
                  return false);
        MKI_CHECK(tensorQ.desc.dims.size() == DIM_3 || tensorQ.desc.dims.size() == DIM_2,
                  "Input0 dim num " << tensorQ.desc.dims.size() << " invalid, dim should be 2 or 3", return false);

        // K.shape = [num_blocks, blockSize, head_num * head_size] or [num_blocks, blockSize, head_num, head_size]
        // V.shape = [num_blocks, blockSize, head_num * head_size] or [num_blocks, blockSize, head_num, head_size]
        auto &tensorK = launchParam.GetInTensor(DIM_1);
        MKI_CHECK(tensorK.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorK.desc.dtype == TENSOR_DTYPE_BF16,
                  "Input1 dtype " << tensorK.desc.dtype << " invalid, should be float16 or BF16", return false);
        auto &tensorV = launchParam.GetInTensor(DIM_2);
        MKI_CHECK(tensorV.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorV.desc.dtype == TENSOR_DTYPE_BF16,
                  "Input2 dtype " << tensorV.desc.dtype << " invalid, should be float16 or BF16", return false);
        //   cl todo remove == 2 just for testcase now
        MKI_CHECK(tensorK.desc.dims.size() == DIM_3 || tensorK.desc.dims.size() == DIM_4,
                  "Input1 dim num " << tensorK.desc.dims.size() << " invalid, dim should be 3 or 4", return false);
        MKI_CHECK(tensorV.desc.dims.size() == DIM_3 || tensorV.desc.dims.size() == DIM_4,
                  "Input2 dim num " << tensorV.desc.dims.size() << " invalid, dim should be 3 or 4", return false);
        MKI_CHECK(param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_UNDEFINED,
            "prefix based FlashAttention don't support quant", return false);
        MKI_CHECK(param.scaleType == OpParam::UnpadFlashAttention::ScaleType::SCALE_TOR,
            "prefix based FlashAttention only support scale tor", return false);
        auto &blockTable = launchParam.GetInTensor(DIM_3);
        MKI_CHECK(blockTable.desc.dtype == TENSOR_DTYPE_INT32, "blockTable dtype " << blockTable.desc.dtype
            << " invalid, should be int32", return false);
        MKI_CHECK(blockTable.desc.dims.size() == DIM_2,
            "blockTable dim num " << blockTable.desc.dims.size() << " invalid, dim should be 2", return false);
        auto &tensorMask = launchParam.GetInTensor(DIM_4);
        return CheckMask(launchParam, tensorQ, tensorMask);
    }

    bool CheckUnpadFlashAttentionEncoder(const LaunchParam &launchParam) const
    {
        // Q.shape = [num_tokens, num_heads, head_size] or [num_tokens. num_heads,* head_size]
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorQ.desc.dtype == TENSOR_DTYPE_BF16 ||
                      tensorQ.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input0 dtype " << GetStrWithDType(tensorQ.desc.dtype)
                                  << " invalid, should be float16 or BF16 or int8",
                  return false);
        MKI_CHECK(tensorQ.desc.dims.size() == DIM_3 || tensorQ.desc.dims.size() == DIM_2 ||
                tensorQ.desc.dims.size() == DIM_4,
                  "Input0 dim num " << tensorQ.desc.dims.size() << " invalid, dim should be 2 or 3 or 4", return false);

        // K.shape = [num_tokens, num_heads, head_size] or [num_tokens. num_heads,* head_size]
        // V.shape = [num_tokens, num_heads, head_size] or [num_tokens. num_heads,* head_size]
        auto &tensorK = launchParam.GetInTensor(DIM_1);
        MKI_CHECK(tensorK.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorK.desc.dtype == TENSOR_DTYPE_BF16 ||
                      tensorK.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input1 dtype " << tensorK.desc.dtype << " invalid, should be float16 or BF16 or int8", return false);
        auto &tensorV = launchParam.GetInTensor(DIM_2);
        MKI_CHECK(tensorV.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorV.desc.dtype == TENSOR_DTYPE_BF16 ||
                      tensorV.desc.dtype == TENSOR_DTYPE_INT8,
                  "Input2 dtype " << tensorV.desc.dtype << " invalid, should be float16 or BF16 or int8", return false);
        MKI_CHECK(tensorK.desc.dims.size() == DIM_3 || tensorK.desc.dims.size() == DIM_2 ||
                    tensorK.desc.dims.size() == DIM_4,
                  "Input1 dim num " << tensorK.desc.dims.size() << " invalid, dim should be 2 or 3 or 4", return false);
        MKI_CHECK(tensorV.desc.dims.size() == DIM_3 || tensorV.desc.dims.size() == DIM_2 ||
                    tensorV.desc.dims.size() == DIM_4,
                  "Input2 dim num " << tensorV.desc.dims.size() << " invalid, dim should be 2 or 3 or 4", return false);
        MKI_CHECK((tensorV.desc.dims.size() == tensorK.desc.dims.size()) &&
                      (tensorQ.desc.dims.size() == tensorK.desc.dims.size()),
                  "The shape of input0 input1 and input2 should be same", return false);
        if (param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_QKV_ONLINE ||
            param.quantType == OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_QKV_OFFLINE) {
            MKI_CHECK(CheckQuant(launchParam), "check quant type fail", return false);
        }
        auto &tensorMask = launchParam.GetInTensor(DIM_4);
        return CheckMask(launchParam, tensorQ, tensorMask);
    }

    bool FindMask(std::vector<std::pair<SVector<int64_t>, bool>> &pairs, SVector<int64_t> &curShape, bool nz) const
    {
        auto target = std::find_if(pairs.begin(), pairs.end(), [curShape, nz](std::pair<SVector<int64_t>, bool> iter) {
            if (!iter.second || curShape.size() != iter.first.size()) {
                return false;
            }
            uint32_t count = 0;
            for (int32_t i = curShape.size() - 1; i >= 0; i--, count++) {
                // batch, head应该完全一致，maxQ和maxKv保证能够覆盖 DIM_3/DIM_2为对应mask的后3维/后2维
                if (count < (nz ? DIM_3 : DIM_2)) {
                    if (iter.first[i] > curShape[i]) {
                        return false;
                    }
                } else {
                    if (iter.first[i] != curShape[i]) {
                        return false;
                    }
                }
            }
            return true;
        });
        return target != pairs.end() && (!nz || (*target).first[DIM_3] == FP16_ALIGN_NUM);
    }

    bool CheckNdMask(const Tensor &tensorMask, Tensor &q, const ShapeParam &shapePara,
                     const OpParam::UnpadFlashAttention &param) const
    {
        auto maxQ = shapePara.maxQ;
        auto maxKv = shapePara.maxKv;
        auto batch = shapePara.batch;
        auto headSize = param.headSize;
        constexpr int32_t longSeqAlibiLen = 256;
        auto currentShape = tensorMask.desc.dims;
        auto sz = currentShape.size();
        MKI_CHECK(sz >= DIM_2, "mask invalid, please check.", return false);
        auto maskLen = currentShape[sz - 1];
        bool alibi = param.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI;
        if (param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND &&
            (param.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI_COMPRESS ||
             param.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
             param.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN ||
             param.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI_COMPRESS_128)) {
            alibi = true;
        }
        auto norm = param.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_NORM;
        auto lookAhead = param.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_LOOK_AHEAD;
        auto isLongSeq = (param.isTriuMask == 1) && (maskLen == LONG_SEQ_LEN);
        auto kvHead = param.kvHead == 0 ? headSize : param.kvHead;
        auto isAlibiCompress = maskLen == LONG_SEQ_LEN && currentShape[sz - DIM_2] != maskLen && alibi;
        auto isSwaCompress = param.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_SWA_COMPRESS;
        std::vector<std::pair<SVector<int64_t>, bool>> supports = {
            {{maxQ, maxKv}, true},
            {{LONG_SEQ_LEN, LONG_SEQ_LEN}, isLongSeq},
            {{batch, LONG_SEQ_LEN, LONG_SEQ_LEN}, isLongSeq},
            {{longSeqAlibiLen, longSeqAlibiLen}, alibi && sz == DIM_2},
            {{q.desc.dims[DIM_0], maxKv}, lookAhead},
            {{batch, maxQ, maxKv}, norm},
            {{headSize, maxQ, maxKv}, alibi},
            {{static_cast<int32_t>(batch) / kvHead, maxQ, maxKv}, norm && param.compressHead},
            {{headSize, maxQ, maxKv}, alibi && param.compressHead},
            {{headSize, maxQ, LONG_SEQ_LEN}, isAlibiCompress},
            {{batch, headSize, maxQ, maxKv}, true},
            {{static_cast<int32_t>(batch) / kvHead, headSize, maxQ, maxKv}, alibi && param.compressHead},
            {{SWA_COMPRESS_MASK_SIZE, SWA_COMPRESS_MASK_SIZE}, isSwaCompress}
        };
        // 保证mask一定能覆盖S，核内不会出现异常，用户保证1.避免多传;2.数值正常
        MKI_CHECK(FindMask(supports, currentShape, false), "current mask shape is unsupported!", return false);
        return true;
    }

    bool CheckMaskPre(const Tensor &tensorMask, const Tensor &q) const
    {
        MKI_CHECK(q.desc.dtype == tensorMask.desc.dtype || q.desc.dtype == TENSOR_DTYPE_INT8,
                  "mask data type not consitent with q", return false);
        MKI_CHECK(tensorMask.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorMask.desc.dtype == TENSOR_DTYPE_BF16,
                  "mask dtype should be float16 or bfloat16", return false);
        return true;
    }

    bool CheckMask(const LaunchParam &launchParam, Tensor q, Tensor mask) const
    {
        auto param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
        auto head = param.headSize;
        auto qSeqLen = param.qSeqLen;
        auto kvSeqLen = param.kvSeqLen;
        uint32_t batch = kvSeqLen.size();
        auto maxQSeqlenIter = std::max_element(qSeqLen.begin(), qSeqLen.end());
        auto maxQ = maxQSeqlenIter != qSeqLen.end() ? *maxQSeqlenIter : 1;
        if (param.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_SWA_NORM && maxQ == 1) {
            MKI_CHECK(CheckEmptyTensor(mask), "mask type inconsistent", return false);
            // swa decode mask should be empty
        } else {
            if (param.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_NONE ||
                param.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_CAUSAL_MASK) {
                MKI_CHECK(CheckEmptyTensor(mask), "mask type inconsistent", return false);
            } else {
                MKI_CHECK(!CheckEmptyTensor(mask), "mask type inconsistent", return false);
            }
        }
        if (CheckEmptyTensor(mask)) {
            return true;
        }
        MKI_CHECK_NO_LOG(CheckMaskPre(mask, q), return false);
        MKI_CHECK(batch > 0, "batch invalid, please check", return false);
        // head
        MKI_CHECK(head > 0, "head invalid, please check", return false);
        // maxKvSeqlen
        auto maxKvSeqlenIter = std::max_element(kvSeqLen.begin(), kvSeqLen.end());
        MKI_CHECK(maxKvSeqlenIter != kvSeqLen.end() && *maxKvSeqlenIter > 0, "kvSeqlen invalid, please check",
                  return false);
        auto minKvSeqlenIter = std::min_element(kvSeqLen.begin(), kvSeqLen.end());
        MKI_CHECK((minKvSeqlenIter != kvSeqLen.end() && *minKvSeqlenIter >= 0),
                  "kvSeqlen min value invalid, please check", return false);
        // maxQSeqlen
        MKI_CHECK(maxQ > 0, "qSeqlen max value invalid, please check", return false);
        auto minQSeqlenIter = std::min_element(qSeqLen.begin(), qSeqLen.end());
        MKI_CHECK((minQSeqlenIter == qSeqLen.end()) || ((minQSeqlenIter != qSeqLen.end() && *minQSeqlenIter >= 0)),
                  "qSeqlen min value invalid, please check", return false);
        MKI_LOG(INFO) << "[batch, head, maxQ, maxKv]: [" << batch << ", " << head << ", " << maxQ << ", "
                      << *maxKvSeqlenIter << "]";
        MKI_CHECK(*maxKvSeqlenIter >= maxQ, "maxQ & maxKv inconsistent.", return false);
        ShapeParam shapePara = {maxQ, *maxKvSeqlenIter, batch};
        return CheckNdMask(mask, q, shapePara, param);
    }
};

REG_OPERATION(UnpadFlashAttentionOperation);
} //    namespace AtbOps