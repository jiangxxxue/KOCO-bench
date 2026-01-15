/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "multi_latent_attention_ops_runner_prefill.h"
#include <atb/utils/log.h>
#include <atbops/params/params.h>

namespace {
static const uint32_t IN_TENSOR_NUM_BASE = 13;
static const uint32_t OUT_TENSOR_NUM_BASE = 1;
static const uint32_t CONTEXTLENS_INDEX = 6;
static const uint32_t QSEQLEN_INDEX = 5;
} // namespace

namespace atb {
MultiLatentAttentionOpsRunnerPrefill::MultiLatentAttentionOpsRunnerPrefill(
    const infer::MultiLatentAttentionParam &param)
    : OpsRunner("MultiLatentAttentionOpsRunner", RUNNER_TYPE_MULTILATENTATTENTION), param_(param)
{
    ATB_LOG(INFO) << "MultiLatentAttentionOpsRunnerPrefill::MultiLatentAttentionOpsRunnerPrefill called";
    needKernelGraphModify_ = true;
    skipSetUpKernelGraphWhenCacheHit_ = false;
}

MultiLatentAttentionOpsRunnerPrefill::~MultiLatentAttentionOpsRunnerPrefill() {}

Status MultiLatentAttentionOpsRunnerPrefill::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;
    kernelGraph_.inTensors.resize(IN_TENSOR_NUM_BASE);
    int inTensorStart = 0;
    Mki::Tensor &query = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &queryRope = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &key = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &keyRope = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &value = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &qSeqLen = kernelGraph_.inTensors.at(inTensorStart++);
    (void)qSeqLen;
    Mki::Tensor &kvSeqLen = kernelGraph_.inTensors.at(inTensorStart++);
    (void)kvSeqLen;
    Mki::Tensor &mask = (param_.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_MASK_FREE) ||
                            (param_.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_SWA_NORM) ?
                            kernelGraph_.inTensors.at(inTensorStart++) : nullTensor_;
    Mki::Tensor &alibiCoeff = nullTensor_;
    Mki::Tensor &dequantQK = nullTensor_;
    Mki::Tensor &offsetQK = nullTensor_;
    Mki::Tensor &dequantPV = nullTensor_;
    Mki::Tensor &offsetPV = nullTensor_;
    Mki::Tensor &quantP = nullTensor_;
    Mki::Tensor &logN = nullTensor_;

    kernelGraph_.outTensors.resize(OUT_TENSOR_NUM_BASE);
    Mki::Tensor &output = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &mlaNode = kernelGraph_.nodes.at(0);
    AtbOps::OpParam::MLA asdParam;
    asdParam.type = AtbOps::OpParam::MLA::PREFILL_SPLIT_CACHE;
    asdParam.headSize = param_.headNum;
    asdParam.tor = param_.qkScale;
    asdParam.kvHead = param_.kvHeadNum;
    asdParam.isRing = 0;
    asdParam.windowSize = param_.windowSize;
    if (param_.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_MASK_FREE) {
        asdParam.maskType = AtbOps::OpParam::MLA::MaskType::MASK_TYPE_MASK_FREE;
    } else if (param_.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_CAUSAL_MASK) {
        asdParam.maskType = AtbOps::OpParam::MLA::MaskType::MASK_TYPE_CAUSAL_MASK;
    } else if (param_.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_SWA_NORM) {
        asdParam.maskType = AtbOps::OpParam::MLA::MaskType::MASK_TYPE_SWA_NORM;
    } else {
        asdParam.maskType = AtbOps::OpParam::MLA::MaskType::MASK_TYPE_NONE;
    }

    mlaNode.opDesc = {0, "MLAOperation", asdParam};
    mlaNode.inTensors = {&query,     &queryRope, &key,       &keyRope,  &value,  &mask, &alibiCoeff,
                         &dequantQK, &offsetQK,  &dequantPV, &offsetPV, &quantP, &logN};
    mlaNode.outTensors = {&output};
    return NO_ERROR;
}

Status MultiLatentAttentionOpsRunnerPrefill::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    auto &mlaNode = kernelGraph_.nodes.at(0);
    bool ret = newParam_.BuildFromTensor(opsTensorPack.inTensors, CONTEXTLENS_INDEX, QSEQLEN_INDEX, true);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " build param from host tensor fail";
        return ERROR_INVALID_PARAM;
    }
    if (param_.maskType != infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_CAUSAL_MASK &&
        newParam_.contextLens != newParam_.qSeqlen) {
        ATB_LOG(ERROR) << GetLogPrefix() << " qSeqLen and kvSeqLen should be same";
        return ERROR_INVALID_PARAM;
    }
    AtbOps::OpParam::MLA asdParam;
    asdParam.type = AtbOps::OpParam::MLA::PREFILL_SPLIT_CACHE;
    asdParam.headSize = param_.headNum;
    asdParam.tor = param_.qkScale;
    asdParam.kvHead = param_.kvHeadNum;
    asdParam.kvSeqLen = newParam_.contextLens;
    asdParam.qSeqLen = newParam_.qSeqlen;
    asdParam.isRing = 0;
    asdParam.windowSize = param_.windowSize;
    if (param_.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_MASK_FREE) {
        asdParam.maskType = AtbOps::OpParam::MLA::MaskType::MASK_TYPE_MASK_FREE;
    } else if (param_.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_CAUSAL_MASK) {
        asdParam.maskType = AtbOps::OpParam::MLA::MaskType::MASK_TYPE_CAUSAL_MASK;
    } else if (param_.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_SWA_NORM) {
        asdParam.maskType = AtbOps::OpParam::MLA::MaskType::MASK_TYPE_SWA_NORM;
    } else {
        asdParam.maskType = AtbOps::OpParam::MLA::MaskType::MASK_TYPE_NONE;
    }
    mlaNode.opDesc = {0, "MLAOperation", asdParam};
    return NO_ERROR;
}
} // namespace atb