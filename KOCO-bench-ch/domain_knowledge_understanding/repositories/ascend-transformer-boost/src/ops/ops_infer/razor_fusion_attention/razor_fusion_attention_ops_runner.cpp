/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "razor_fusion_attention_ops_runner.h"
#include <atbops/params/params.h>
#include "atb/utils/log.h"

namespace atb {

RazorFusionAttentionOpsRunner::RazorFusionAttentionOpsRunner(const infer::RazorFusionAttentionParam &param)
    : OpsRunner("RazorFusionAttentionOpsRunner", RUNNER_TYPE_RAZOR_FUSION_ATTENTION), param_(param)
{
    needKernelGraphModify_ = true;
    skipSetUpKernelGraphWhenCacheHit_ = false;

    const std::size_t intensorSize = 3;
    const std::size_t outtensorSize = 1;
    kernelGraph_.inTensors.resize(intensorSize);
    kernelGraph_.outTensors.resize(outtensorSize);

    int inTensorStart = 0;
    Mki::Tensor &query = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &key = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *value = &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &attnOut = kernelGraph_.outTensors.at(0);
    kernelGraph_.nodes.resize(1);
    auto &razorFusionAttentionNode = kernelGraph_.nodes.at(0);
    AtbOps::OpParam::UnpadFlashAttention attentionParam;
    SetRFAParam(attentionParam);

    razorFusionAttentionNode.opDesc = {0, "UnpadFlashAttentionOperation", attentionParam};
    razorFusionAttentionNode.inTensors = {&query, &key, value};
    razorFusionAttentionNode.outTensors = {&attnOut};
    razorFusionAttentionNode.inTensorViewFuncs.resize(razorFusionAttentionNode.inTensors.size());
}

Status RazorFusionAttentionOpsRunner::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    auto &razorFusionAttentionNode = kernelGraph_.nodes.at(0);
    AtbOps::OpParam::UnpadFlashAttention attentionParam;
    SetRFAParam(attentionParam);
    int64_t batch = opsTensorPack.inTensors.at(0).desc.dims[0] / (param_.razorLen * param_.tileQ + param_.textQLen);
    std::vector<int32_t> qSeqLen(batch, param_.razorLen * param_.tileQ + param_.textQLen);
    attentionParam.qSeqLen = qSeqLen;
    std::vector<int32_t> kvSeqLen(batch, param_.razorLen * param_.tileKv + param_.textKvLen);
    attentionParam.kvSeqLen = kvSeqLen;
    razorFusionAttentionNode.opDesc = {0, "UnpadFlashAttentionOperation", attentionParam};
    return NO_ERROR;
}

void RazorFusionAttentionOpsRunner::SetRFAParam(AtbOps::OpParam::UnpadFlashAttention &attentionParam)
{
    attentionParam.type = AtbOps::OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_RAZOR_FUSION;
    attentionParam.headSize = param_.headNum;
    attentionParam.kvHead = param_.kvHeadNum;
    attentionParam.tor = param_.qkScale;
    attentionParam.maskType = AtbOps::OpParam::UnpadFlashAttention::MaskType::MASK_TYPE_NONE;
    attentionParam.razorLen = param_.razorLen;
    attentionParam.preTokens = param_.preTokens;
    attentionParam.nextTokens = param_.nextTokens;
    attentionParam.tileQ = param_.tileQ;
    attentionParam.tileKv = param_.tileKv;
    attentionParam.textQLen = param_.textQLen;
    attentionParam.textKvLen = param_.textKvLen;
}

RazorFusionAttentionOpsRunner::~RazorFusionAttentionOpsRunner() {}
} // namespace atb
