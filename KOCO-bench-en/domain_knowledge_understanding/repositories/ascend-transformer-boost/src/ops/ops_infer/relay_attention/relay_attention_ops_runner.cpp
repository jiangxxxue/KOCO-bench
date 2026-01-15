/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <asdops/params/params.h>
#include "atb/utils/tensor_util.h"
#include "param.h"
#include "relay_attention_ops_runner.h"

static const uint32_t UNDEFINED_IN_TENSOR_NUM = 10;
static const uint32_t UNDEFINED_OUT_TENSOR_NUM = 1;


namespace atb {

RelayAttentionOpsRunner::RelayAttentionOpsRunner(const infer::RelayAttentionParam &param)
    : OpsRunner("RelayAttentionOpsRunner", RUNNER_TYPE_RELAY_ATTENTION), param_(param)
{
    needKernelGraphModify_ = true;
    skipSetUpKernelGraphWhenCacheHit_ = false;
    ATB_LOG(INFO) << "RelayAttentionOpsRunner::RelayAttentionOpsRunner called";

    kernelGraph_.inTensors.resize(UNDEFINED_IN_TENSOR_NUM);
    int inTensorStart = 0;
    Mki::Tensor &query = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &key = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &value = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &keyShare = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &valueShare = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &attentionMask = kernelGraph_.inTensors.at(inTensorStart++);
    (void)attentionMask;

    kernelGraph_.outTensors.resize(UNDEFINED_OUT_TENSOR_NUM);
    Mki::Tensor &output = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &relayAttentionNode = kernelGraph_.nodes.at(0);
    AtbOps::OpParam::UnpadFlashAttention flashAttentionParam;

    relayAttentionNode.opDesc = {0, "UnpadFlashAttentionOperation", flashAttentionParam};
    relayAttentionNode.inTensors = {&query, &key, &value, &keyShare, &valueShare, &nullTensor_};
    relayAttentionNode.outTensors = {&output};
}

Status RelayAttentionOpsRunner::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    bool ret = newParam_.BuildFromTensor(opsTensorPack.inTensors);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " build param from host tensor fail";
        return ERROR_INVALID_PARAM;
    }

    auto &relayAttentionNode = kernelGraph_.nodes.at(0);
    AtbOps::OpParam::UnpadFlashAttention flashAttentionParam;

    SetRAParam(flashAttentionParam);
    flashAttentionParam.qSeqLen = newParam_.seqLen;
    flashAttentionParam.kvSeqLen = newParam_.kvSeqLen;
    flashAttentionParam.kvShareMap = newParam_.kvShareMap;
    flashAttentionParam.kvShareLen = newParam_.kvShareLen;

    flashAttentionParam.kTensorList.resize(newParam_.key.size());
    for (std::size_t i = 0; i < newParam_.key.size(); i++) {
        TensorUtil::ConvertAtbTensor2OpsTensor(newParam_.key.at(i), flashAttentionParam.kTensorList.at(i));
    }
    relayAttentionNode.inTensors.at(1) = &nullTensor_; // 1:key Tensor

    flashAttentionParam.vTensorList.resize(newParam_.value.size());
    for (std::size_t i = 0; i < newParam_.value.size(); i++) {
        TensorUtil::ConvertAtbTensor2OpsTensor(newParam_.value.at(i), flashAttentionParam.vTensorList.at(i));
    }
    relayAttentionNode.inTensors.at(2) = &nullTensor_; // 2:value Tensor

    flashAttentionParam.kShareTensorList.resize(newParam_.keyShare.size());
    for (std::size_t i = 0; i < newParam_.keyShare.size(); i++) {
        TensorUtil::ConvertAtbTensor2OpsTensor(newParam_.keyShare.at(i), flashAttentionParam.kShareTensorList.at(i));
    }
    relayAttentionNode.inTensors.at(3) = &nullTensor_; // 3:keyShare Tensor

    flashAttentionParam.vShareTensorList.resize(newParam_.valueShare.size());
    for (std::size_t i = 0; i < newParam_.valueShare.size(); i++) {
        TensorUtil::ConvertAtbTensor2OpsTensor(newParam_.valueShare.at(i), flashAttentionParam.vShareTensorList.at(i));
    }
    relayAttentionNode.inTensors.at(4) = &nullTensor_; // 4:valueShare Tensor

    relayAttentionNode.inTensors = {relayAttentionNode.inTensors.at(0),  // 0:query Tensor
                                    relayAttentionNode.inTensors.at(5)}; // 5:mask Tensor
    relayAttentionNode.opDesc = {0, "UnpadFlashAttentionOperation", flashAttentionParam};
    ATB_LOG(INFO) << GetLogPrefix() << " update AtbOps::OpParam::UnpadFlashAttention.headNum:" << param_.headNum
                  << ", qkScale:" << param_.qkScale << ", kvHeadNum:" << param_.kvHeadNum
                  << ", maskType:" << param_.maskType;
    return NO_ERROR;
}

void RelayAttentionOpsRunner::SetRAParam(AtbOps::OpParam::UnpadFlashAttention &flashAttentionParam)
{
    flashAttentionParam.type = AtbOps::OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND;
    flashAttentionParam.headSize = param_.headNum;
    flashAttentionParam.tor = param_.qkScale;
    flashAttentionParam.kvHead = param_.kvHeadNum;
    flashAttentionParam.maskType = static_cast<AtbOps::OpParam::UnpadFlashAttention::MaskType>(param_.maskType);
}

RelayAttentionOpsRunner::~RelayAttentionOpsRunner() {}
} // namespace atb