/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "self_attention_encoder_fusion_ops_runner.h"
#include <cmath>
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
#include "param.h"

static constexpr uint32_t VALUE_TENSOR_POS = 2;

namespace atb {

SelfAttentionEncoderFusionOpsRunner::SelfAttentionEncoderFusionOpsRunner(const infer::SelfAttentionParam &param)
    : OpsRunner("SelfAttentionEncoderFusionOpsRunner", RUNNER_TYPE_SELF_ATTENTION_PA_ENCODER), param_(param)
{
    needKernelGraphModify_ = true;
    skipSetUpKernelGraphWhenCacheHit_ = false;
    ATB_LOG(INFO) << "SelfAttentionEncoderFusionOpsRunner::SelfAttentionEncoderFusionOpsRunner called";

    bool needMask = (param_.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED);
    bool isMaskCompress = (param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
                           param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
                           param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN);
    bool needQKVOnlineQuant = (param_.quantType == atb::infer::SelfAttentionParam::TYPE_QUANT_QKV_ONLINE);
    bool needQKVOfflineQuant = (param_.quantType == atb::infer::SelfAttentionParam::TYPE_QUANT_QKV_OFFLINE);
    bool needQKVQuant = (needQKVOnlineQuant || needQKVOfflineQuant);
    bool needLogN = (param_.scaleType == atb::infer::SelfAttentionParam::SCALE_TYPE_LOGN);
    bool isMla = param_.mlaVHeadSize > 0;
    std::size_t intensorSize = needMask ? (isMaskCompress ? 6 : 5) : 4; // 6, 5, 4: flash encoder input num
    if (needQKVOnlineQuant) {
        intensorSize += 4; // 4: qkDescale、qkOffset、vpvDescale、vpvOffset
    }
    if (needQKVOfflineQuant) {
        intensorSize += 5; // 5: qkDescale、qkOffset、vpvDescale、vpvOffset、pScale
    }
    if (needLogN) {
        intensorSize++;
    }
    if (isMla) {
        intensorSize--;
    }
    kernelGraph_.inTensors.resize(intensorSize);
    kernelGraph_.outTensors.resize(1);

    int inTensorStart = 0;
    Mki::Tensor &query = kernelGraph_.inTensors.at(inTensorStart++); // shape (batch, head_num, head_size)
    Mki::Tensor &key = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *value = isMla ? &nullTensor_ : &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *mask = needMask ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    Mki::Tensor &seqLen = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *layerId = &nullTensor_;
    Mki::Tensor *slopes = isMaskCompress ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    Mki::Tensor *qkDescale = &nullTensor_;
    Mki::Tensor *qkOffset = &nullTensor_;
    Mki::Tensor *vpvDescale = &nullTensor_;
    Mki::Tensor *vpvOffset = &nullTensor_;
    if (needQKVQuant) {
        qkDescale = &kernelGraph_.inTensors.at(inTensorStart++);
        qkOffset = &kernelGraph_.inTensors.at(inTensorStart++);
        vpvDescale = &kernelGraph_.inTensors.at(inTensorStart++);
        vpvOffset = &kernelGraph_.inTensors.at(inTensorStart++);
        qkOffset = &nullTensor_;
        vpvOffset = &nullTensor_;
    }

    Mki::Tensor *pScale = needQKVOfflineQuant ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    Mki::Tensor *logN = needLogN ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;

    Mki::Tensor &output = kernelGraph_.outTensors.at(0);

    ATB_LOG(INFO) << GetLogPrefix() << "seqLen dataSize:" << seqLen.dataSize;
    kernelGraph_.nodes.resize(1);
    auto &flashAttentionEncoderNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::UnpadFlashAttention unpadFlashAttentionParam;
    SetFAParam(unpadFlashAttentionParam);

    flashAttentionEncoderNode.opDesc = {0, "UnpadFlashAttentionOperation", unpadFlashAttentionParam};
    if (isMla) {
        flashAttentionEncoderNode.inTensors = {&query,   &key,       layerId,   mask,  qkDescale,
                                               qkOffset, vpvDescale, vpvOffset, pScale};
    } else {
        flashAttentionEncoderNode.inTensors = {&query,    &key,     value,      layerId,   mask,   slopes,
                                               qkDescale, qkOffset, vpvDescale, vpvOffset, pScale, logN};
    }
    flashAttentionEncoderNode.outTensors = {&output};
    flashAttentionEncoderNode.inTensorViewFuncs.resize(flashAttentionEncoderNode.inTensors.size()); // view
    flashAttentionEncoderNode.inferShapePreFunc = [](Mki::LaunchParam &launchParam) { // format, dtype 设置
        for (size_t i = 0; i < launchParam.GetInTensorCount(); i++) {
            launchParam.GetInTensor(i).desc.format = Mki::TENSOR_FORMAT_ND;
        }
    };
}

bool SelfAttentionEncoderFusionOpsRunner::NeedModifySlopes(const OpsTensorPack &opsTensorPack)
{
    bool isMaskCompress = param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
                          param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
                          param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN;
    if (param_.mlaVHeadSize > 0 || !isMaskCompress) {
        return false;
    } else {
        return opsTensorPack.inTensors.at(3).desc.dims.size() == 3 && // 3: maskId, 3: [head_num, seqlen, 128]
               opsTensorPack.inTensors.at(3).desc.dims[2] == 128;     // 3: maskId, 2: 2nd dim
    }
}

Status SelfAttentionEncoderFusionOpsRunner::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    const size_t seqLenId = param_.mlaVHeadSize > 0 ?
                                (param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED ? 2 : 3) :
                                (param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED ? 3 : 4);
    SelfAttentionFusionVariantPackParam newParam;
    bool ret = newParam.BuildFromTensorEncoder(opsTensorPack.inTensors, seqLenId);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " build param from host tensor fail";
        return ERROR_INVALID_PARAM;
    }

    auto &flashAttentionNode = kernelGraph_.nodes.at(0); // 0: flashAttentionEncoder节点位置

    AtbOps::OpParam::UnpadFlashAttention unpadFlashAttentionParam;

    unpadFlashAttentionParam.qSeqLen = newParam.seqLen;
    unpadFlashAttentionParam.kvSeqLen = newParam.tokenOffset;

    uint32_t hiddenSizePos = kernelGraph_.inTensors.at(VALUE_TENSOR_POS).desc.dims.size() - 1;
    int32_t hiddenSizeV = static_cast<int32_t>(kernelGraph_.inTensors.at(VALUE_TENSOR_POS).desc.dims[hiddenSizePos]);
    if (param_.mlaVHeadSize > 0) {
        unpadFlashAttentionParam.headDimV = static_cast<int32_t>(param_.mlaVHeadSize);
    } else if (hiddenSizePos == 1) {
        int32_t headNumV = (param_.kvHeadNum > 0) ? param_.kvHeadNum : param_.headNum;
        // param_.headNum 一定大于0
        unpadFlashAttentionParam.headDimV = hiddenSizeV / headNumV;
    } else {
        unpadFlashAttentionParam.headDimV = hiddenSizeV;
    }
    ATB_LOG(INFO) << "headDimV is:" << unpadFlashAttentionParam.headDimV;
    SetFAParam(unpadFlashAttentionParam);

    flashAttentionNode.opDesc = {0, "UnpadFlashAttentionOperation", unpadFlashAttentionParam};
    if (NeedModifySlopes(opsTensorPack)) {
        flashAttentionNode.inTensors.at(5) = &nullTensor_; // 5: slopes
    }

    ATB_LOG(INFO) << GetLogPrefix() << " update AsdOps::OpParam::UnpadFlashAttention.headNum:" << param_.headNum
                  << ", seqLen.size:" << newParam.seqLen.size() << ", tokenOffset.size:" << newParam.tokenOffset.size()
                  << ", qkScale:" << param_.qkScale << ", kvHead:" << param_.kvHeadNum;
    return NO_ERROR;
}

void SelfAttentionEncoderFusionOpsRunner::SetFAParam(AtbOps::OpParam::UnpadFlashAttention &flashAttentionParam)
{
    if (param_.mlaVHeadSize > 0) {
        flashAttentionParam.type =
            param_.kernelType == atb::infer::SelfAttentionParam::KERNELTYPE_HIGH_PRECISION ?
                AtbOps::OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_HIGH_PRECISION_COMBINE_CACHE :
                AtbOps::OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_COMBINE_CACHE;
    } else {
        flashAttentionParam.type = param_.kernelType == atb::infer::SelfAttentionParam::KERNELTYPE_HIGH_PRECISION ?
                                       AtbOps::OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_FP32_ND :
                                       AtbOps::OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_ND;
    }
    flashAttentionParam.headSize = param_.headNum;
    flashAttentionParam.tor = param_.qkScale;
    flashAttentionParam.kvHead = param_.kvHeadNum;
    flashAttentionParam.windowSize = param_.windowSize;
    if (param_.inputLayout == atb::infer::InputLayout::TYPE_BNSD) {
        flashAttentionParam.dataShapeType = AtbOps::OpParam::UnpadFlashAttention::DataShapeType::TYPE_BNSD;
        flashAttentionParam.headDimV = 0; // BNSD不支持mla
    }
    flashAttentionParam.cacheType = static_cast<AtbOps::OpParam::UnpadFlashAttention::CacheType>(param_.cacheType);
    flashAttentionParam.maskType =
        param_.maskType == infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS ?
            static_cast<AtbOps::OpParam::UnpadFlashAttention::MaskType>(1) : // 1: norm mask
            ((param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
              param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
              param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN) ?
                 static_cast<AtbOps::OpParam::UnpadFlashAttention::MaskType>(2) : // 2: alibi mask
                 static_cast<AtbOps::OpParam::UnpadFlashAttention::MaskType>(param_.maskType));
    if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_NORM) {
        flashAttentionParam.maskType = AtbOps::OpParam::UnpadFlashAttention::MaskType::MASK_TYPE_SWA_NORM;
    } else if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_COMPRESS) {
        flashAttentionParam.maskType = AtbOps::OpParam::UnpadFlashAttention::MaskType::MASK_TYPE_SWA_COMPRESS;
    }
    flashAttentionParam.isAlibiMaskSqrt = (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT);
    flashAttentionParam.isTriuMask = param_.isTriuMask;
    flashAttentionParam.alibiLeftAlign =
        (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN);
    flashAttentionParam.quantType = static_cast<AtbOps::OpParam::UnpadFlashAttention::QuantType>(param_.quantType);
    flashAttentionParam.outDataType = static_cast<Mki::TensorDType>(param_.outDataType);
    flashAttentionParam.scaleType = param_.scaleType == atb::infer::SelfAttentionParam::SCALE_TYPE_LOGN ?
                                        AtbOps::OpParam::UnpadFlashAttention::SCALE_LOGN_FP32 :
                                        AtbOps::OpParam::UnpadFlashAttention::SCALE_TOR;
}

SelfAttentionEncoderFusionOpsRunner::~SelfAttentionEncoderFusionOpsRunner() {}
} // namespace atb