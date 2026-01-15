/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "self_attention_prefix_encoder_ops_runner.h"
#include <cmath>
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
#include "param.h"

namespace {
static constexpr uint32_t VALUE_TENSOR_POS = 2;
static constexpr uint32_t MASK_TENSOR_POS = 4;
static constexpr uint32_t SEQLEN_TENSOR_POS = 5;
static constexpr uint32_t KVSEQLEN_TENSOR_POS = 6;
static constexpr uint32_t SLOPES_TENSOR_POS = 7;
} // namespace


namespace atb {

SelfAttentionPrefixEncoderOpsRunner::SelfAttentionPrefixEncoderOpsRunner(const infer::SelfAttentionParam &param)
    : OpsRunner("SelfAttentionPrefixEncoderOpsRunner", RUNNER_TYPE_SELF_ATTENTION_PREFIX_ENCODER), param_(param)
{
    needKernelGraphModify_ = true;
    skipSetUpKernelGraphWhenCacheHit_ = false;
    // 8, 7, 6: flash encoder input num + blockTables + seqLen + kvSeqLen
    std::size_t intensorSize = 8;
    bool hasSlopes = true;
    if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS) {
        // 7: without slopes
        intensorSize = 7;
        hasSlopes = false;
    }
    needMask_ = param_.maskType != infer::SelfAttentionParam::MASK_TYPE_CAUSAL_MASK;
    kernelGraph_.inTensors.resize(intensorSize);
    kernelGraph_.outTensors.resize(1);

    int inTensorStart = 0;
    Mki::Tensor &query = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &key = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *value = &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *blockTables = &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *mask = needMask_ ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    Mki::Tensor &seqLen = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &kvSeqLen = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *slopes = hasSlopes && needMask_ ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;

    Mki::Tensor &attnOut = kernelGraph_.outTensors.at(0);

    ATB_LOG(INFO) << GetLogPrefix() << "seqLen dataSize:" << seqLen.dataSize;
    ATB_LOG(INFO) << GetLogPrefix() << "kvSeqLen dataSize:" << kvSeqLen.dataSize;
    kernelGraph_.nodes.resize(1);
    auto &flashAttentionEncoderNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::UnpadFlashAttention unpadFlashAttentionParam;
    SetFAParam(unpadFlashAttentionParam);

    flashAttentionEncoderNode.opDesc = {0, "UnpadFlashAttentionOperation", unpadFlashAttentionParam};
    flashAttentionEncoderNode.inTensors = {&query, &key, value, blockTables, mask, slopes};
    flashAttentionEncoderNode.outTensors = {&attnOut};
    flashAttentionEncoderNode.inTensorViewFuncs.resize(flashAttentionEncoderNode.inTensors.size()); // view
    flashAttentionEncoderNode.inferShapePreFunc = [](Mki::LaunchParam &launchParam) { // format, dtype 设置
        for (size_t i = 0; i < launchParam.GetInTensorCount(); i++) {
            launchParam.GetInTensor(i).desc.format = Mki::TENSOR_FORMAT_ND;
        }
    };
}

void SelfAttentionPrefixEncoderOpsRunner::SetIsMask128(const OpsTensorPack &opsTensorPack)
{
    isMask128_ = false;
    if (opsTensorPack.inTensors.at(MASK_TENSOR_POS).desc.dims.size() == 3 && // 3: maskId, 3: [head_num, seqlen, 128]
        opsTensorPack.inTensors.at(MASK_TENSOR_POS).desc.dims[2] == 128) {   // 3: maskId, 2: 2nd dim)
        isMask128_ = true;
    }
}

Status SelfAttentionPrefixEncoderOpsRunner::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    // query, key, value, blockTables, mask, seqlen, kvSeqLen, slopes
    SelfAttentionFusionVariantPackParam newParam;
    uint32_t seqlen_pos = SEQLEN_TENSOR_POS;
    uint32_t kvSeqlen_pos = KVSEQLEN_TENSOR_POS;
    if (!needMask_) {
        seqlen_pos = SEQLEN_TENSOR_POS - 1;
        kvSeqlen_pos = KVSEQLEN_TENSOR_POS - 1;
    }
    bool ret = newParam.BuildFromTensorPrefixEncoder(opsTensorPack.inTensors, seqlen_pos, kvSeqlen_pos);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " build param from host tensor fail";
        return ERROR_INVALID_PARAM;
    }
    auto &flashAttentionNode = kernelGraph_.nodes.at(0); // 0: flashAttentionEncoder节点位置

    AtbOps::OpParam::UnpadFlashAttention unpadFlashAttentionParam;
    SetIsMask128(opsTensorPack);
    SetFAParam(unpadFlashAttentionParam);
    unpadFlashAttentionParam.qSeqLen = newParam.seqLen;
    unpadFlashAttentionParam.kvSeqLen = newParam.kvSeqLen;
    flashAttentionNode.opDesc = {0, "UnpadFlashAttentionOperation", unpadFlashAttentionParam};
    ATB_LOG(INFO) << GetLogPrefix() << " update AsdOps::OpParam::UnpadFlashAttention.headNum: " << param_.headNum
                  << ", seqLen.size: " << newParam.seqLen.size() << ", kvSeqLen.size: " << newParam.kvSeqLen.size()
                  << ", qkScale: " << param_.qkScale << ", kvHead: " << param_.kvHeadNum
                  << ", is mask 128: " << isMask128_;
    return NO_ERROR;
}

void SelfAttentionPrefixEncoderOpsRunner::SetFAParam(AtbOps::OpParam::UnpadFlashAttention &flashAttentionParam)
{
    flashAttentionParam.type = AtbOps::OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND;
    flashAttentionParam.headSize = param_.headNum;
    flashAttentionParam.tor = param_.qkScale;
    flashAttentionParam.kvHead = param_.kvHeadNum;
    if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS) {
        flashAttentionParam.maskType = static_cast<AtbOps::OpParam::UnpadFlashAttention::MaskType>(
            AtbOps::OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI_COMPRESS);
    } else if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT) {
        flashAttentionParam.maskType = static_cast<AtbOps::OpParam::UnpadFlashAttention::MaskType>(
            AtbOps::OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI_COMPRESS_SQRT);
        flashAttentionParam.isAlibiMaskSqrt = true;
    } else if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS) {
        flashAttentionParam.maskType = static_cast<AtbOps::OpParam::UnpadFlashAttention::MaskType>(
            AtbOps::OpParam::UnpadFlashAttention::MASK_TYPE_NORM);
    } else if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_CAUSAL_MASK) {
        flashAttentionParam.maskType = static_cast<AtbOps::OpParam::UnpadFlashAttention::MaskType>(
            AtbOps::OpParam::UnpadFlashAttention::MASK_TYPE_CAUSAL_MASK);
    }
    // [head_num, seqlen, 128]
    if (isMask128_ && needMask_) {
        flashAttentionParam.maskType = static_cast<AtbOps::OpParam::UnpadFlashAttention::MaskType>(
            AtbOps::OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI_COMPRESS_128);
    }
    flashAttentionParam.isTriuMask = param_.isTriuMask;
}

SelfAttentionPrefixEncoderOpsRunner::~SelfAttentionPrefixEncoderOpsRunner() {}
} // namespace atb
