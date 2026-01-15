/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <mki/base/operation_base.h>
#include <mki/utils/log/log.h>
#include <mki/utils/const/op_const.h>
#include <mki_loader/op_register.h>
#include <mki/utils/checktensor/check_tensor.h>
#include "atbops/params/params.h"
#include "acl/acl_rt.h"
#include "acl/acl.h"

static constexpr int32_t DEQUANT_EYE_SIZE = 1024;
static constexpr int32_t MAX_DEQUANT_SIZE = 128;
static constexpr int32_t LONG_SEQ_LEN = 128;
static constexpr int32_t MLA_THRESHOLD = 256;

namespace AtbOps {
using namespace Mki;
class PagedAttentionOperation : public OperationBase {
public:
    explicit PagedAttentionOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::PagedAttention),
            "OpParam is invalid", return nullptr);
        auto param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        auto embeddingSizeQK = launchParam.GetInTensor(1).desc.dims.at(DIM_3);
        auto embeddingSizeV = param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND ?
                              launchParam.GetInTensor(2).desc.dims.at(DIM_3) : param.headDimV;
        bool isMLA = (embeddingSizeQK > MLA_THRESHOLD || embeddingSizeV > MLA_THRESHOLD ||
                          embeddingSizeQK != embeddingSizeV);
        switch (param.type) {
            case OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND:
                if (isMLA) {
                    return GetKernelByName("PagedMultiLatentAttentionSplitCacheMaskNdKernel");
                }
                return GetKernelByName("PagedAttentionMaskNdKernel");
            case OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_COMBINE_CACHE_MASK_ND:
                return GetKernelByName("PagedMultiLatentAttentionCombineCacheMaskNdKernel");
            case OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_MULTI_TOKEN_PREDICTION_MASK_ND:
                return GetKernelByName("PagedMultiLatentAttentionMultiTokenPredictionMaskNdKernel");
            case OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK:
                return GetKernelByName("PagedAttentionDecoderNzMaskKernel");
            case OpParam::PagedAttention::PAGED_ATTENTION_NZ:
                return GetKernelByName("PagedAttentionDecoderNzKernel");
            default:
                break;
        }
        MKI_LOG(ERROR) << "Unsupport PagedAttention type " << param.type;
        return nullptr;
    }
    
    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::PagedAttention), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::PagedAttention>(specificParam);
        switch (param.type) {
            case OpParam::PagedAttention::PAGED_ATTENTION_NZ:
                return DIM_5;
            case OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK:
                return DIM_7;
            case OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND:
                return DIM_12;
            case OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_COMBINE_CACHE_MASK_ND:
                return DIM_9;
            case OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_MULTI_TOKEN_PREDICTION_MASK_ND:
                return DIM_4;
            default:
                break;
        }
        return DIM_1;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::PagedAttention), "OpParam is invalid", return 0);
        return DIM_1;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::PagedAttention),
            "OpParam is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        auto param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        switch (param.type) {
            case OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND:
            case OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_COMBINE_CACHE_MASK_ND:
            case OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_MULTI_TOKEN_PREDICTION_MASK_ND:
                return InferShapePagedAttention(launchParam, outTensors);
            case OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK:
            case OpParam::PagedAttention::PAGED_ATTENTION_NZ:
                return InferShapePagedAttentionDecoderNz(launchParam, outTensors);
            default:
                break;
        }
        return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Param is invalid");
    }

private:
    struct ShapeParam {
        int32_t maxQ;
        int32_t maxKv;
        uint32_t batch;
    };

    Status InferShapePagedAttention(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        auto param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        MKI_CHECK(CheckPagedAttention(launchParam), "Failed to check launch param",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto &tensorK = launchParam.GetInTensor(DIM_1);
        auto embedDimV = param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND ?
                         launchParam.GetInTensor(2).desc.dims.at(DIM_3) : param.headDimV;
        auto embedDimK = tensorK.desc.dims[tensorK.desc.dims.size() - 1];
        outTensors[DIM_0].desc = tensorQ.desc;
        if (tensorQ.desc.dtype == TENSOR_DTYPE_INT8) {
            outTensors[DIM_0].desc.dtype = param.outDataType;
        }
        if (tensorQ.desc.dims.size() == DIM_2) {
            outTensors[DIM_0].desc.dims[DIM_1] = tensorQ.desc.dims[DIM_1] / embedDimK * embedDimV;
        } else {
            outTensors[DIM_0].desc.dims[DIM_2] = embedDimV;
        }
        return Status::OkStatus();
    }

    bool CheckRazorRope(const LaunchParam &launchParam, const OpParam::PagedAttention &param) const
    {
        auto &razorOffset = launchParam.GetInTensor(DIM_9);
        if (!CheckEmptyTensor(razorOffset)) {
            MKI_CHECK(param.compressHead, "Razor rope parameter compressHead invalid", return false);
            MKI_CHECK(CheckEmptyTensor(launchParam.GetInTensor(DIM_4)),
                "Razor rope mask should be empty tensor", return false);
            MKI_CHECK(razorOffset.desc.dtype == TENSOR_DTYPE_FLOAT,
                         "Input9 dtype " << GetStrWithDType(razorOffset.desc.dtype) << " invalid, should be float32",
                         return false);
            MKI_CHECK(razorOffset.desc.dims.size() == 2, "Input9 dim num invalid, should be " << 2, return false);
            auto &tensorKcache = launchParam.GetInTensor(DIM_1);
            MKI_CHECK(razorOffset.desc.dims[DIM_0] == tensorKcache.desc.dims[DIM_0],
                         "Input9 dim shape0 invalid, should be " << tensorKcache.desc.dims[DIM_0], return false);
            MKI_CHECK(razorOffset.desc.dims[DIM_1] == tensorKcache.desc.dims[DIM_1],
                         "Input9 dim shape1 invalid, should be " << tensorKcache.desc.dims[DIM_1], return false);
        }
        return true;
    }
    
    bool CheckPagedAttention(const LaunchParam &launchParam) const
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::PagedAttention),
            "OpParam is invalid", return false);
        auto param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        MKI_CHECK(PageAttentionParamCheck(param), "check pageattention param fail", return false);
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorQ.desc.dtype == TENSOR_DTYPE_BF16 ||
            tensorQ.desc.dtype == TENSOR_DTYPE_INT8,
            "Input0 dtype " << GetStrWithDType(tensorQ.desc.dtype) << " invalid, should be float16 or BF16",
            return false);
        // Q.shape = [numTokens, numHeads, headSize] [numTokens, numHeads * headSize]
        int64_t tensorQDims = static_cast<int64_t>(tensorQ.desc.dims.size());
        MKI_CHECK(tensorQDims == DIM_3 || tensorQDims == DIM_2,
            "Input0 dim num " << tensorQDims << " invalid, should be 2 or 3 ", return false);
        MKI_CHECK(tensorQ.desc.dims[DIM_0] > 0 && tensorQ.desc.dims[DIM_1] > 0,
            "Shape of Input0 invalid ", return false);
        int64_t embeddingDimQ = 0;
        if (tensorQDims == DIM_3) {
            embeddingDimQ = tensorQ.desc.dims[DIM_2];
            MKI_CHECK(tensorQ.desc.dims[DIM_1] == static_cast<int64_t>(param.headSize),
                "Shape of Input0 invalid", return false);
        } else {
            embeddingDimQ = tensorQ.desc.dims[DIM_1] / static_cast<int64_t>(param.headSize);
        }
        MKI_CHECK(embeddingDimQ > 0, "Shape of Input0 invalid, headSize must > 0", return false);
        if (param.type != OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND) {
            MKI_CHECK(CheckPagedMLAttentionCache(launchParam, param, embeddingDimQ),
                        "check cache shape fail", return false);
        } else {
            MKI_CHECK(CheckPagedAttentionCache(launchParam, param, embeddingDimQ),
                        "check cache shape fail", return false);
            MKI_CHECK(CheckRazorRope(launchParam, param), "check razor rope parameter fail", return false);
        }
        MKI_CHECK(CheckMaskParam(launchParam), "check mask shape fail", return false);
        MKI_CHECK(param.scaleType == OpParam::PagedAttention::ScaleType::SCALE_LOGN_FP32 ||
                    param.scaleType == OpParam::PagedAttention::ScaleType::SCALE_TOR,
                    "scaletype invalid", return false);
        if (param.scaleType == OpParam::PagedAttention::ScaleType::SCALE_LOGN_FP32) {
            MKI_CHECK(param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND,
                        "check kernel type", return false);
            auto &tensorLog = launchParam.GetInTensor(DIM_11);
            MKI_CHECK(!CheckEmptyTensor(tensorLog), "Input12 should not be null tensor", return false);
            MKI_CHECK(tensorLog.desc.dims.at(0) >= tensorQ.desc.dims.at(0),
                      "input 12 num invalid", return false);
            MKI_CHECK(tensorLog.desc.dtype == TENSOR_DTYPE_FLOAT, "LogNscale should be float", return false);
            MKI_CHECK(param.quantType == OpParam::PagedAttention::QuantType::TYPE_QUANT_UNDEFINED &&
                      !param.compressHead && param.dataShapeType == OpParam::PagedAttention::DataShapeType::BSND,
                      "LogN does not support quant,compressHead and bnsd", return false);
        }
        return true;
    }

    inline bool PageAttentionParamCheck(const OpParam::PagedAttention &param) const
    {
        MKI_CHECK(param.headSize > 0, "headSize is invalid", return false);
        MKI_CHECK((param.kvHead > 0 && param.kvHead <= param.headSize && param.headSize % param.kvHead == 0) ||
                    (param.kvHead == 0), "kvHead is invalid", return false);
        return true;
    }
    inline bool PageAttentionNZLognCheck(const LaunchParam &launchParam, const OpParam::PagedAttention &param) const
    {
        auto scaleType = param.scaleType;
        if (scaleType == OpParam::PagedAttention::ScaleType::SCALE_LOGN) {
            MKI_CHECK(param.type == OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK,
                        "check kernel type", return false);
            auto &tensorLog = launchParam.GetInTensor(DIM_6);
            auto &tensorQ = launchParam.GetInTensor(DIM_0);
            MKI_CHECK(!CheckEmptyTensor(tensorLog), "Input7 should not be null tensor", return false);
            MKI_CHECK(tensorLog.desc.dims.at(0) >= tensorQ.desc.dims.at(0),
                      "input 7 num invalid", return false);
            auto inDtype = tensorLog.desc.dtype;
            MKI_CHECK(inDtype == TENSOR_DTYPE_FLOAT16, "LogNscale should be float16", return false);
            MKI_CHECK(param.quantType == OpParam::PagedAttention::QuantType::TYPE_QUANT_UNDEFINED &&
                      !param.compressHead && param.dataShapeType == OpParam::PagedAttention::DataShapeType::BSND,
                      "LogN does not support quant,compressHead,BNSD", return false);
        }
        return true;
    }

    bool CheckPagedMLAttentionCache(const LaunchParam &launchParam, const OpParam::PagedAttention &param,
                                  int64_t embeddingDimQ) const
    {
        auto &tensorKcache = launchParam.GetInTensor(DIM_1); // K.shape = [num_blocks, block_size, num_heads, head_size]
        MKI_CHECK(tensorKcache.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorKcache.desc.dtype == TENSOR_DTYPE_BF16 ||
                         tensorKcache.desc.dtype == TENSOR_DTYPE_INT8,
                     "Input1 dtype " << GetStrWithDType(tensorKcache.desc.dtype)
                                     << " invalid, should be float16 or bfloat16 or int8",
                     return false);
        static const size_t KV_CACHE_DIM_NUM = 4;
        MKI_CHECK(tensorKcache.desc.dims.size() == KV_CACHE_DIM_NUM,
                     "Input1 dim num invalid, should be " << KV_CACHE_DIM_NUM, return false);
        auto numBlocks = tensorKcache.desc.dims[DIM_0];
        auto blockSize = tensorKcache.desc.dims[DIM_1];
        auto kvHead = tensorKcache.desc.dims[DIM_2];
        auto embeddingDimK = tensorKcache.desc.dims[DIM_3];
        auto embeddingDimV = param.headDimV;
        MKI_CHECK(embeddingDimV <= embeddingDimK,
                    "embeddingDimV should not exceed embeddingDimK for combined cache paged-MLA", return false);
        MKI_CHECK(numBlocks > 0 && blockSize > 0 && kvHead > 0 && embeddingDimK == embeddingDimQ,
                     "Shape of key_cache is valid, must be [numBlocks, blockSize, numHeads, headSize]", return false);
        if (tensorKcache.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorKcache.desc.dtype == TENSOR_DTYPE_BF16) {
            MKI_CHECK(kvHead == 1, "MLA operation ONLY supports MQA with 16bit input ", return false);
        }
        if (!param.compressHead) {
            MKI_CHECK(kvHead == static_cast<int64_t>(param.kvHead > 0 ? param.kvHead : param.headSize),
                         "param kvHead must equal key_cache dim2", return false);
        } else {
            MKI_CHECK(tensorKcache.desc.dims[DIM_2] == DIM_1, "DIM 2 should be 1", return false);
            MKI_CHECK(launchParam.GetInTensor(DIM_0).desc.dims[DIM_0] * param.kvHead ==
                             static_cast<int64_t>(param.kvSeqLen.size()),
                         "qk inconsistent for RA compress head", return false);
        }
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        if (tensorKcache.desc.dtype == TENSOR_DTYPE_INT8) {
            if (tensorQ.desc.dtype != TENSOR_DTYPE_INT8) {
                MKI_CHECK(param.identityM.size() == DEQUANT_EYE_SIZE, "identityM must 32 * 32", return false);
                MKI_CHECK(CheckPagedAttentionDeQuant(launchParam), "dequant check invalid", return false);
            } else {
                MKI_CHECK(CheckPagedAttentionQuant(launchParam, param, 1), "quant check invalid", return false);
            }
        }
        return true;
    }

    bool CheckPagedAttentionCache(const LaunchParam &launchParam, const OpParam::PagedAttention &param,
                                  int64_t embeddingDimQ) const
    {
        auto &tensorKcache = launchParam.GetInTensor(DIM_1); // K.shape = [num_blocks, block_size, num_heads, head_size]
        MKI_CHECK(tensorKcache.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorKcache.desc.dtype == TENSOR_DTYPE_BF16 ||
                         tensorKcache.desc.dtype == TENSOR_DTYPE_INT8,
                     "Input1 dtype " << GetStrWithDType(tensorKcache.desc.dtype)
                                     << " invalid, should be float16 or bfloat16 or int8",
                     return false);
        auto &tensorVcache = launchParam.GetInTensor(DIM_2); // V.shape = [num_blocks, block_size, num_heads, head_size]
        MKI_CHECK(tensorVcache.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorVcache.desc.dtype == TENSOR_DTYPE_BF16 ||
                         tensorVcache.desc.dtype == TENSOR_DTYPE_INT8,
                     "Input2 dtype " << GetStrWithDType(tensorVcache.desc.dtype)
                                     << " invalid, should be float16 or bfloat16 or int8",
                     return false);
        static const size_t KV_CACHE_DIM_NUM = 4;
        MKI_CHECK(tensorKcache.desc.dims.size() == KV_CACHE_DIM_NUM,
                     "Input1 dim num invalid, should be " << KV_CACHE_DIM_NUM, return false);
        MKI_CHECK(tensorVcache.desc.dims.size() == KV_CACHE_DIM_NUM,
                     "Input2 dim num invalid, should be " << KV_CACHE_DIM_NUM, return false);
        auto numBlocks = tensorKcache.desc.dims[DIM_0];
        auto blockSize = tensorKcache.desc.dims[DIM_1];
        auto kvHead = tensorKcache.desc.dims[DIM_2];
        if (param.dataShapeType == 1) {
            blockSize = tensorKcache.desc.dims[DIM_2];
            kvHead = tensorKcache.desc.dims[DIM_1];
        }

        auto embeddingDimK = tensorKcache.desc.dims[DIM_3];
        MKI_CHECK(numBlocks > 0 && blockSize > 0 && kvHead > 0 && embeddingDimK == embeddingDimQ,
                     "Shape of key_cache is valid, must be [numBlocks, blockSize, numHeads, headSize]", return false);

        if (!param.compressHead) {
            MKI_CHECK(kvHead == static_cast<int64_t>(param.kvHead > 0 ? param.kvHead : param.headSize),
                      "param kvHead must equal key_cache dim2", return false);
        } else {
            MKI_CHECK(kvHead == DIM_1, "DIM 2 should be 1", return false);
            MKI_CHECK(launchParam.GetInTensor(DIM_0).desc.dims[DIM_0] * param.kvHead ==
                          static_cast<int64_t>(param.kvSeqLen.size()),
                      "qk inconsistent for RA compress head", return false);
        }
        for (uint32_t dim = 0; dim < DIM_3; ++dim) {
            MKI_CHECK(tensorKcache.desc.dims[dim] == tensorVcache.desc.dims[dim],
                      "kv shape should be the same, except for headSize", return false);
        }
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        if (tensorKcache.desc.dtype == TENSOR_DTYPE_INT8) {
            if (tensorQ.desc.dtype != TENSOR_DTYPE_INT8) {
                MKI_CHECK(param.identityM.size() == DEQUANT_EYE_SIZE, "identityM must 32 * 32", return false);
                MKI_CHECK(CheckPagedAttentionDeQuant(launchParam), "dequant check invalid", return false);
            } else {
                MKI_CHECK(CheckPagedAttentionQuant(launchParam, param, 0), "quant check invalid", return false);
            }
        }
        
        return true;
    }

    bool CheckRazorAndSetKVHead(const LaunchParam &launchParam, int32_t &kvHead) const
    {
        auto param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto &tensorK = launchParam.GetInTensor(DIM_1);
        if (param.compressHead) {
            if (tensorK.desc.dtype == TENSOR_DTYPE_INT8) {
                MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16, "quant tensorQ must be fp16 in razor attention",
                            return false);
            }
            kvHead = static_cast<int32_t>(param.kvHead);
        }
        return true;
    }

    bool CheckPagedAttentionQuant(const LaunchParam &launchParam,
                                  const OpParam::PagedAttention &param, bool isMla) const
    {
        uint32_t deqScale1Idx = isMla ? DIM_4 : DIM_5;
        uint32_t deqScale2Idx = isMla ? DIM_6 : DIM_7;
        uint32_t scaleIdx = isMla ? DIM_8 : DIM_10;
        uint32_t offset1Idx = isMla ? DIM_5 : DIM_6;
        uint32_t offset2Idx = isMla ? DIM_7 : DIM_8;
        uint32_t tensorMaskIdx = isMla ? DIM_3 : DIM_4;
        auto &deqScale1 = launchParam.GetInTensor(deqScale1Idx);
        auto &deqScale2 = launchParam.GetInTensor(deqScale2Idx);
        auto &scale = launchParam.GetInTensor(scaleIdx);
        auto &offset1 = launchParam.GetInTensor(offset1Idx);
        auto &offset2 = launchParam.GetInTensor(offset2Idx);
        auto outTensortType = param.outDataType;
        auto &tensorMask = launchParam.GetInTensor(tensorMaskIdx);
        MKI_CHECK(param.quantType == OpParam::PagedAttention::TYPE_QUANT_QKV_ONLINE ||
                        param.quantType == OpParam::PagedAttention::TYPE_QUANT_QKV_OFFLINE,
                        "QuantType is invalid", return false);
        MKI_CHECK(outTensortType == Mki::TENSOR_DTYPE_FLOAT16 || outTensortType == Mki::TENSOR_DTYPE_BF16,
                        "outTensor dtype invalid, should be FP16 OR BF16 form torch", return false);
        if (param.maskType != OpParam::PagedAttention::MASK_TYPE_NONE) {
            MKI_CHECK(tensorMask.desc.dtype == outTensortType,
                        "maskType should be same as outDataType", return false);
        }
        MKI_CHECK(deqScale1.desc.dtype == TENSOR_DTYPE_FLOAT,
                    "Input6 dtype invalid, should be float32 form torch", return false);
        MKI_CHECK(deqScale1.desc.dtype == deqScale2.desc.dtype && deqScale1.desc.dims == deqScale2.desc.dims,
                    "Input6 dtype or dims invalid, should be same with input8 ", return false);
        MKI_CHECK(CheckEmptyTensor(offset1), " Input7 should be null tensor ", return false);
        MKI_CHECK(CheckEmptyTensor(offset2), " Input9 should be null tensor ", return false);
        auto head = param.headSize;
        MKI_CHECK(!param.compressHead, "Razor rope parameter compressHead invalid, quant not support", return false);
        MKI_CHECK((deqScale1.desc.dims.size() == 1) && (deqScale1.desc.dims[DIM_0] == head),
            "descale1 shape shoud be [headnum]", return false);
        MKI_CHECK((deqScale2.desc.dims.size() == 1) && (deqScale2.desc.dims[DIM_0] == head),
            "descale1 shape shoud be [headnum]", return false);
        bool isQuantOffline = (param.quantType == OpParam::PagedAttention::TYPE_QUANT_QKV_OFFLINE);
        if (isQuantOffline) {
            MKI_CHECK(scale.desc.dtype == TENSOR_DTYPE_FLOAT,
                    "Scale dtype invalid, should be float32 form torch", return false);
            MKI_CHECK((scale.desc.dims.size() == 1) && (scale.desc.dims[DIM_0] == head),
                    "Scale shape shoud be [headnum]", return false);
        }
        return true;
    }

    bool CheckPagedMLAttentionDeQuant(const LaunchParam &launchParam) const
    {
        auto &tensorKcache = launchParam.GetInTensor(DIM_1);
        auto &deqScale1 = launchParam.GetInTensor(DIM_4);
        auto &offset1 = launchParam.GetInTensor(DIM_5);
        auto &deqScale2 = launchParam.GetInTensor(DIM_6);
        auto &offset2 = launchParam.GetInTensor(DIM_7);
        MKI_CHECK(deqScale1.desc.dtype == deqScale2.desc.dtype && deqScale1.desc.dims == deqScale2.desc.dims,
                     "Input5 dtype or dims invalid, should be same with input8 ", return false);
        if (tensorKcache.desc.dtype == TENSOR_DTYPE_INT8) {
            MKI_CHECK(tensorKcache.desc.dims[DIM_3] <= MAX_DEQUANT_SIZE, "int8 kv headdim must <= 128",
                         return false);
            MKI_CHECK(tensorKcache.desc.dims[DIM_1] <= tensorKcache.desc.dims[DIM_3],
                         "int8 kv blocksize must <= headdim", return false);
            MKI_CHECK(!CheckEmptyTensor(deqScale1), " Input5 should not be null tensor ", return false);
            MKI_CHECK(deqScale1.desc.dtype == TENSOR_DTYPE_UINT64 || deqScale1.desc.dtype == TENSOR_DTYPE_FLOAT ||
                             deqScale1.desc.dtype == TENSOR_DTYPE_INT64,
                         "Input6 dtype invalid, should be uint64_t or int64_t form torch", return false);
            // 校验deqScale的维度
            auto &tensorQ = launchParam.GetInTensor(DIM_0);
            auto headDim = tensorQ.desc.dims[DIM_2];
            int32_t kvHead = tensorKcache.desc.dims[DIM_2];
            MKI_CHECK(deqScale1.desc.dims[DIM_0] == kvHead * headDim, "Input5 shape not valid", return false);
            if (!CheckEmptyTensor(offset1)) {
                MKI_CHECK(offset1.desc.dims[DIM_0] == kvHead * headDim, "Input6 shape not valid", return false);
                MKI_CHECK(offset1.desc.dtype == TENSOR_DTYPE_INT32, "Input6 dtype invalid, should be int32_t ",
                             return false);
            }
            if (!CheckEmptyTensor(offset2)) {
                MKI_CHECK(offset2.desc.dims[DIM_0] == kvHead * headDim, "Input8 shape not valid", return false);
                MKI_CHECK(offset2.desc.dtype == TENSOR_DTYPE_INT32, "Input8 dtype invalid, should be int32_t ",
                             return false);
            }
        } else {
            MKI_CHECK(CheckEmptyTensor(deqScale1), " Input5 should be null tensor ", return false);
            MKI_CHECK(CheckEmptyTensor(deqScale2), " Input7 should be null tensor ", return false);
            MKI_CHECK(CheckEmptyTensor(offset1), " Input6 should be null tensor ", return false);
            MKI_CHECK(CheckEmptyTensor(offset2), " Input8 should be null tensor ", return false);
        }
        return true;
    }

    bool CheckPagedAttentionDeQuant(const LaunchParam &launchParam) const
    {
        auto &tensorKcache = launchParam.GetInTensor(DIM_1);
        auto &tensorVcache = launchParam.GetInTensor(DIM_2);
        auto &deqScale1 = launchParam.GetInTensor(DIM_5);
        auto &offset1 = launchParam.GetInTensor(DIM_6);
        auto &deqScale2 = launchParam.GetInTensor(DIM_7);
        auto &offset2 = launchParam.GetInTensor(DIM_8);
        MKI_CHECK(tensorVcache.desc.dtype == tensorKcache.desc.dtype,
            "Input1 dtype" << GetStrWithDType(tensorKcache.desc.dtype)
            << " invalid, should be same with input2 " << GetStrWithDType(tensorVcache.desc.dtype),
            return false);
        MKI_CHECK(deqScale1.desc.dtype == deqScale2.desc.dtype && deqScale1.desc.dims == deqScale2.desc.dims,
                     "Input6 dtype or dims invalid, should be same with input8 ", return false);
        if (tensorKcache.desc.dtype == TENSOR_DTYPE_INT8) {
            if (deqScale1.desc.dtype != TENSOR_DTYPE_FLOAT) {
                MKI_CHECK(tensorVcache.desc.dims[DIM_3] <= MAX_DEQUANT_SIZE, "int8 kv headdim must <= 128",
                          return false);
                MKI_CHECK(tensorVcache.desc.dims[DIM_1] <= tensorVcache.desc.dims[DIM_3],
                          "int8 kv blocksize must <= headdim", return false);
            }
            MKI_CHECK(!CheckEmptyTensor(deqScale1), " Input6 should not be null tensor ", return false);
            MKI_CHECK(deqScale1.desc.dtype == TENSOR_DTYPE_UINT64 || deqScale1.desc.dtype == TENSOR_DTYPE_FLOAT ||
                             deqScale1.desc.dtype == TENSOR_DTYPE_INT64,
                         "Input6 dtype invalid, should be uint64_t or int64_t or float form torch", return false);
            // 校验deqScale的维度
            auto &tensorQ = launchParam.GetInTensor(DIM_0);
            auto headDim = tensorQ.desc.dims[DIM_2];
            int32_t kvHead = tensorKcache.desc.dims[DIM_2];
            MKI_CHECK(CheckRazorAndSetKVHead(launchParam, kvHead), "check razor rope parameter fail",
                         return false);
            MKI_CHECK(deqScale1.desc.dims[DIM_0] == kvHead * headDim, "Input6 shape not valid", return false);
            if (!CheckEmptyTensor(offset1)) {
                MKI_CHECK(offset1.desc.dims[DIM_0] == kvHead * headDim, "Input7 shape not valid", return false);
                MKI_CHECK(offset1.desc.dtype == TENSOR_DTYPE_INT32, "Input7 dtype invalid, should be int32_t ",
                             return false);
            }
            if (!CheckEmptyTensor(offset2)) {
                MKI_CHECK(offset2.desc.dims[DIM_0] == kvHead * headDim, "Input9 shape not valid", return false);
                MKI_CHECK(offset2.desc.dtype == TENSOR_DTYPE_INT32, "Input9 dtype invalid, should be int32_t ",
                             return false);
            }
        } else {
            MKI_CHECK(CheckEmptyTensor(deqScale1), " Input6 should be null tensor ", return false);
            MKI_CHECK(CheckEmptyTensor(deqScale2), " Input8 should be null tensor ", return false);
            MKI_CHECK(CheckEmptyTensor(offset1), " Input7 should be null tensor ", return false);
            MKI_CHECK(CheckEmptyTensor(offset2), " Input9 should be null tensor ", return false);
        }
        return true;
    }

    bool CheckMaskParam(const LaunchParam &launchParam) const
    {
        auto param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        auto &tensorMask = (param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND) ?
                            launchParam.GetInTensor(DIM_4) : launchParam.GetInTensor(DIM_3);
        if (param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND) {
            return CheckMask(launchParam, tensorQ, tensorMask);
        }
        return true;
    }

    bool CheckMask(const LaunchParam &launchParam, Tensor q, Tensor mask, bool nzFlag = false) const
    {
        auto param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        MKI_CHECK((param.maskType == OpParam::PagedAttention::MASK_TYPE_NONE) ^ (!CheckEmptyTensor(mask)),
                     "mask type inconsistent", return false);
        if (CheckEmptyTensor(mask)) {
            return true;
        }
        MKI_CHECK_NO_LOG(CheckMaskPre(mask, q, nzFlag), return false);
        auto head = param.headSize;
        auto qSeqLen = param.qSeqLen;
        auto kvSeqLen = param.kvSeqLen;
        if ((param.type == OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK) &&
            kvSeqLen.size() == 0) {
            auto batch = qSeqLen.size() > 0 ? qSeqLen.size() : launchParam.GetInTensor(DIM_4).desc.dims[0];
            auto currentShape = mask.desc.dims;
            MKI_CHECK(currentShape.size() == DIM_4, "mask invalid, please check.", return false);
            MKI_CHECK(currentShape[DIM_3] == FP16_ALIGN_NUM, "not nz mask", return false);
            MKI_CHECK(currentShape[DIM_2] >= FP16_ALIGN_NUM, "not nz decoder mask", return false);
            auto cs0 = currentShape[DIM_0];
            MKI_CHECK((cs0 == 1) || (cs0 == head) || (cs0 == static_cast<int64_t>(batch)) ||
                             (cs0 == static_cast<int64_t>(static_cast<int32_t>(batch) * head)),
                         "not nz decoder mask", return false);
        } else {
            // batch
            uint32_t batch = kvSeqLen.size();
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
            auto maxQSeqlenIter = std::max_element(qSeqLen.begin(), qSeqLen.end());
            auto maxQ = maxQSeqlenIter != qSeqLen.end() ? *maxQSeqlenIter : 1;
            MKI_CHECK(maxQ > 0, "qSeqlen max value invalid, please check", return false);
            auto minQSeqlenIter = std::min_element(qSeqLen.begin(), qSeqLen.end());
            MKI_CHECK((minQSeqlenIter == qSeqLen.end()) ||
                             ((minQSeqlenIter != qSeqLen.end() && *minQSeqlenIter >= 0)),
                         "qSeqlen min value invalid, please check", return false);
            MKI_LOG(INFO) << "[batch, head, maxQ, maxKv]: [" << batch << ", " << head << ", " << maxQ << ", "
                          << *maxKvSeqlenIter << "]";
            MKI_CHECK(*maxKvSeqlenIter >= maxQ, "maxQ & maxKv inconsistent.", return false);
            ShapeParam shapePara = {maxQ, *maxKvSeqlenIter, batch};
            return CheckMaskImpl(mask, q, shapePara, param, nzFlag);
        }
        return true;
    }

    bool CheckMaskImpl(const Tensor &tensorMask, Tensor &q, ShapeParam &shapePara, OpParam::PagedAttention &param,
                       bool nzFlag) const
    {
        return nzFlag ? CheckNzMask(tensorMask, shapePara, param) : CheckNdMask(tensorMask, q, shapePara, param);
    }

    bool CheckNzMask(const Tensor &tensorMask, const ShapeParam &shapePara, const OpParam::PagedAttention &param) const
    {
        auto headSize = param.headSize;
        constexpr int32_t longSeqAlibiLen = 256;
        auto currentShape = tensorMask.desc.dims;
        auto sz = currentShape.size();
        MKI_CHECK(sz == DIM_4, "mask invalid, please check.", return false);
        auto maskLen = currentShape[DIM_2];
        auto alibi = param.maskType == OpParam::PagedAttention::MASK_TYPE_ALIBI;
        auto isLongSeq = (param.isTriuMask == 1) && (maskLen == LONG_SEQ_LEN);
        auto maxNzQ = (shapePara.maxQ + FP16_ALIGN_NUM - 1) / FP16_ALIGN_NUM * FP16_ALIGN_NUM;
        auto maxKv = (shapePara.maxKv + FP16_ALIGN_NUM - 1) / FP16_ALIGN_NUM * FP16_ALIGN_NUM;
        auto batch = shapePara.batch;
        auto isAlibiCompress =
            currentShape[DIM_1] * currentShape[DIM_3] == LONG_SEQ_LEN && currentShape[DIM_2] > LONG_SEQ_LEN && alibi;
        auto alibiDim2 = currentShape[DIM_1] * currentShape[DIM_3] == longSeqAlibiLen &&
                         currentShape[DIM_2] == longSeqAlibiLen && alibi;
        std::vector<std::pair<SVector<int64_t>, bool>> supports = {
            {{1, maxKv / FP16_ALIGN_NUM, maxNzQ, FP16_ALIGN_NUM}, true},
            {{1, longSeqAlibiLen / FP16_ALIGN_NUM, longSeqAlibiLen, FP16_ALIGN_NUM}, alibiDim2},
            {{1, LONG_SEQ_LEN / FP16_ALIGN_NUM, LONG_SEQ_LEN, FP16_ALIGN_NUM}, isLongSeq},
            {{batch, maxKv / FP16_ALIGN_NUM, maxNzQ, FP16_ALIGN_NUM}, true},
            {{headSize, LONG_SEQ_LEN / FP16_ALIGN_NUM, maxNzQ, FP16_ALIGN_NUM}, isAlibiCompress},
            {{headSize, maxKv / FP16_ALIGN_NUM, maxNzQ, FP16_ALIGN_NUM}, alibi},
            {{static_cast<int32_t>(batch) * headSize, maxKv / FP16_ALIGN_NUM, maxNzQ, FP16_ALIGN_NUM}, alibi},
        };
        // 保证mask一定能覆盖S，核内不会出现异常，用户保证1.避免多传;2.数值正常
        MKI_CHECK(FindMask(supports, currentShape, true), "current mask shape is unsupported!", return false);
        return true;
    }

    bool FindMask(std::vector<std::pair<SVector<int64_t>, bool>> &pairs, SVector<int64_t> &curShape,
                  bool nz) const
    {
        auto target =
            std::find_if(pairs.begin(), pairs.end(), [curShape, nz](std::pair<SVector<int64_t>, bool> iter) {
                if (!iter.second || curShape.size() != iter.first.size()) {
                    return false;
                }
                uint32_t count = 0;
                for (int32_t i = curShape.size() - 1; i >= 0; i--, count++) {
                    // batch, head应该完全一致，maxQ和maxKv保证能够覆盖
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

    Status InferShapePagedAttentionDecoderNz(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        MKI_CHECK(CheckPagedAttentionDecoderNz(launchParam), "Failed to check run info",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check run info"));
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        outTensors[DIM_0].desc = tensorQ.desc;

        return Status::OkStatus();
    }

    bool CheckPagedAttentionDecoderNz(const LaunchParam &launchParam) const
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::PagedAttention),
            "OpParam is invalid", return false);
        auto param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
        MKI_CHECK(PageAttentionParamCheck(param), "check pageattention param fail", return false);
        auto &tensorQ = launchParam.GetInTensor(DIM_0); // Q.shape = [1, hidden_size/16, n_tokens, 16]
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16,
            "Input0 dtype " << GetStrWithDType(tensorQ.desc.dtype) << " invalid, should be float16", return false);
        static const size_t Q_CACHE_DIM_NUM = 4;
        MKI_CHECK(tensorQ.desc.dims.size() == Q_CACHE_DIM_NUM,
            "Input0 dim num " << tensorQ.desc.dims.size() << " invalid, should be " << Q_CACHE_DIM_NUM, return false);
        auto &tensorKcache = launchParam.GetInTensor(DIM_1); // K.shape = [num_blocks, hidden_size/16, block_size, 16]
        MKI_CHECK(tensorKcache.desc.dtype == TENSOR_DTYPE_FLOAT16,
            "Input1 dtype " << GetStrWithDType(tensorKcache.desc.dtype) << " invalid, should be float16", return false);
        auto &tensorVcache = launchParam.GetInTensor(DIM_2); // V.shape = [num_blocks, hidden_size/16, block_size, 16]
        MKI_CHECK(tensorVcache.desc.dtype == TENSOR_DTYPE_FLOAT16,
            "Input2 dtype " << GetStrWithDType(tensorVcache.desc.dtype) << " invalid, should be float16", return false);
        static const size_t KV_CACHE_DIM_NUM = 4;
        MKI_CHECK(tensorKcache.desc.dims.size() == KV_CACHE_DIM_NUM, "Input1 dim num " <<
            tensorKcache.desc.dims.size() << " invalid, should be " << KV_CACHE_DIM_NUM, return false);
        MKI_CHECK(tensorVcache.desc.dims.size() == KV_CACHE_DIM_NUM, "Input2 dim num " <<
            tensorVcache.desc.dims.size() << " invalid, should be " << KV_CACHE_DIM_NUM, return false);
        MKI_CHECK(tensorKcache.desc.dims[DIM_3] == 16, "K_cache Shape should be in nz format", return false);
        MKI_CHECK(tensorVcache.desc.dims[DIM_3] == 16, "V_cache Shape should be in nz format", return false);
        auto &blockTables = launchParam.GetInTensor(DIM_3); // K.shape = [num_blocks, hidden_size/16, block_size, 16]
        MKI_CHECK(blockTables.desc.dtype == TENSOR_DTYPE_INT32,
            "Input3 dtype " << GetStrWithDType(blockTables.desc.dtype) << " invalid, should be int32", return false);
        auto &blockContexLen = launchParam.GetInTensor(DIM_4); // V.shape = [num_blocks, hidden_size/16, block_size, 16]
        MKI_CHECK(blockContexLen.desc.dtype == TENSOR_DTYPE_INT32, "Input4 dtype " <<
            GetStrWithDType(blockContexLen.desc.dtype) << " invalid, should be int32", return false);
        static const size_t BLOCK_TABLE_DIM_NUM = 2;
        MKI_CHECK(blockTables.desc.dims.size() == BLOCK_TABLE_DIM_NUM, "Input3 dim num " <<
            blockTables.desc.dims.size() << " invalid, should be " << BLOCK_TABLE_DIM_NUM, return false);
        static const size_t BLOCK_LEN_DIM_NUM = 1;
        MKI_CHECK(blockContexLen.desc.dims.size() == BLOCK_LEN_DIM_NUM, "Input4 dim num " <<
            blockContexLen.desc.dims.size() << " invalid, should be " << BLOCK_LEN_DIM_NUM, return false);
        if (param.type == OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK) {
            MKI_CHECK(CheckMask(launchParam, tensorQ, launchParam.GetInTensor(DIM_5), true),
                "mask invalid", return false);
        }
        MKI_CHECK(param.scaleType == OpParam::PagedAttention::ScaleType::SCALE_LOGN ||
                    param.scaleType == OpParam::PagedAttention::ScaleType::SCALE_TOR,
                      "scaletype invalid", return false);
        MKI_CHECK(!(param.scaleType == OpParam::PagedAttention::SCALE_LOGN && param.qSeqLen.size() > 0),
                "logn scale attention is not supported in lookahead", return false);
        MKI_CHECK(PageAttentionNZLognCheck(launchParam, param), "check pageattention logN fail", return false);
        return true;
    }

    bool CheckMaskPre(const Tensor &tensorMask, const Tensor &q, bool nzFlag) const
    {
        if (nzFlag) {
            MKI_CHECK(q.desc.dtype == tensorMask.desc.dtype, "mask data type not consitent with q", return false);
        }
        MKI_CHECK(tensorMask.desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                         (!nzFlag && tensorMask.desc.dtype == TENSOR_DTYPE_BF16),
                     "Input4 dtype should be float16 or bfloat16", return false);
        return true;
    }

    bool CheckNdMask(const Tensor &tensorMask, Tensor &q, const ShapeParam &shapePara,
        const OpParam::PagedAttention &param) const
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
        auto alibi = param.maskType == OpParam::PagedAttention::MASK_TYPE_ALIBI;
        auto norm = param.maskType == OpParam::PagedAttention::MASK_TYPE_NORM;
        auto lookAhead = param.maskType == OpParam::PagedAttention::MASK_TYPE_LOOK_AHEAD;
        auto isLongSeq = (param.isTriuMask == 1) && (maskLen == LONG_SEQ_LEN);
        auto kvHead = param.kvHead == 0 ? headSize : param.kvHead;
        auto isAlibiCompress = maskLen == LONG_SEQ_LEN && currentShape[sz - DIM_2] != maskLen && alibi;
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
        };
        // 保证mask一定能覆盖S，核内不会出现异常，用户保证1.避免多传;2.数值正常
        MKI_CHECK(FindMask(supports, currentShape, false), "current mask shape is unsupported!", return false);
        return true;
    }
};

REG_OPERATION(PagedAttentionOperation);
} // namespace AtbOps
