/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ring_mla_ops_runner.h"
#include <cmath>
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
#include "param.h"

namespace {
// query_split1, query_split2, key_split1, key_split2, value, mask, seqLen, prevOut (optional), prevLse(optional)
static constexpr uint32_t VALUE_TENSOR_POS = 2;
static constexpr uint32_t SEQLEN_TENSOR_POS = 6;
static constexpr uint32_t IN_TENSOR_NUM = 15;
static constexpr uint32_t OUT_TENSOR_NUM = 2;
} // namespace


namespace atb {

RingMLAOpsRunner::RingMLAOpsRunner(const infer::RingMLAParam &param)
    : OpsRunner("RingMLAOpsRunner", RUNNER_TYPE_RING_MLA), param_(param)
{
    needKernelGraphModify_ = true;
    skipSetUpKernelGraphWhenCacheHit_ = false;
    isInputSoftmaxLse_ = param_.calcType == infer::RingMLAParam::CalcType::CALC_TYPE_DEFAULT;
    isNoMask_ = param_.maskType == infer::RingMLAParam::MaskType::NO_MASK;
    kernelGraph_.inTensors.resize(IN_TENSOR_NUM);
    kernelGraph_.outTensors.resize(OUT_TENSOR_NUM);

    int inTensorStart = 0;
    Mki::Tensor *query_split1 = &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *query_split2 = &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *key_split1 = &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *key_split2 = &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *value = &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *mask = &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *seqLen = &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *prevOut = isInputSoftmaxLse_ ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    Mki::Tensor *prevLse = isInputSoftmaxLse_ ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;

    if (isNoMask_) {
        mask = &nullTensor_;
    }

    int outTensorStart = 0;
    Mki::Tensor *attnOut = &kernelGraph_.outTensors.at(outTensorStart++);
    Mki::Tensor *softmaxLse = &kernelGraph_.outTensors.at(outTensorStart++);

    ATB_LOG(INFO) << GetLogPrefix() << "Ctor seqLen dataSize: " << seqLen->dataSize;
    kernelGraph_.nodes.resize(1);
    auto &ringMLANode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::RINGMLA ringMLAParam;
    SetRingMLAParam(ringMLAParam);

    ringMLANode.opDesc = {0, "RINGMLAOperation", ringMLAParam};

    // flashAttentionEncoderNode.inTensors = {&query_split1, query_split2, &key_split1, key_split2, value,
    // mask, slopes, qkDescale, qkOffset, vpvDescale, vpvOffset, pScale, logN, prevOut, prevLse};
    ringMLANode.inTensors = {query_split1, query_split2, key_split1,   key_split2,   value,
                             mask,         &nullTensor_, &nullTensor_, &nullTensor_, &nullTensor_,
                             &nullTensor_, &nullTensor_, &nullTensor_, prevOut,      prevLse};

    if (isInputSoftmaxLse_) {
        *attnOut = *prevOut;
    }
    ringMLANode.outTensors = {attnOut, softmaxLse};
    ringMLANode.inTensorViewFuncs.resize(ringMLANode.inTensors.size()); // view
    ringMLANode.inferShapePreFunc = [](Mki::LaunchParam &launchParam) { // format, dtype 设置
        for (size_t i = 0; i < launchParam.GetInTensorCount(); i++) {
            launchParam.GetInTensor(i).desc.format = Mki::TENSOR_FORMAT_ND;
        }
    };
}

Status RingMLAOpsRunner::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    // query, key, value, mask, seqlen,
    RingMLAVariantPackParam newParam;
    bool ret = newParam.BuildFromTensor(opsTensorPack.inTensors, SEQLEN_TENSOR_POS);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " build param from host tensor fail";
        return ERROR_INVALID_PARAM;
    }
    auto &ringMLANode = kernelGraph_.nodes.at(0); // 0: RingMLA节点位置

    AtbOps::OpParam::RINGMLA ringMLAParam;
    SetRingMLAParam(ringMLAParam);
    ringMLAParam.qSeqLen = newParam.qSeqLen;
    ringMLAParam.kvSeqLen = newParam.kvSeqLen;
    if (ringMLAParam.kvSeqLen.size() == 0) {
        ringMLAParam.kvSeqLen = newParam.qSeqLen;
    }

    ATB_LOG(INFO) << GetLogPrefix() << "seqLen dataSize: " << newParam.qSeqLen.size();
    ATB_LOG(INFO) << GetLogPrefix() << "kvSeqLen dataSize: " << newParam.kvSeqLen.size();
    ringMLANode.opDesc = {0, "RINGMLAOperation", ringMLAParam};
    ATB_LOG(INFO) << GetLogPrefix() << " update AsdOps::OpParam::RINGMLAParam.type: " << ringMLAParam.type
                  << ", headNum: " << param_.headNum << ", qSeqLen.size: " << ringMLAParam.qSeqLen.size()
                  << ", kvSeqLen.size: " << ringMLAParam.kvSeqLen.size() << ", qkScale: " << ringMLAParam.tor
                  << ", kvHead: " << ringMLAParam.kvHead << ", maskType: " << ringMLAParam.maskType;
    return NO_ERROR;
}

void RingMLAOpsRunner::SetRingMLAParam(AtbOps::OpParam::RINGMLA &ringMLAParam)
{
    ringMLAParam.isRing = isInputSoftmaxLse_;
    ringMLAParam.headSize = param_.headNum;
    ringMLAParam.kvHead = param_.kvHeadNum;
    ringMLAParam.tor = param_.qkScale;
    ringMLAParam.type = AtbOps::OpParam::RINGMLA::Type::PREFILL_SPLIT_CACHE;
    if (param_.maskType == infer::RingMLAParam::MaskType::NO_MASK) {
        ringMLAParam.maskType =
            static_cast<AtbOps::OpParam::RINGMLA::MaskType>(AtbOps::OpParam::RINGMLA::MaskType::MASK_TYPE_NONE);
    } else if (param_.maskType == infer::RingMLAParam::MaskType::MASK_TYPE_TRIU) {
        ringMLAParam.maskType =
            static_cast<AtbOps::OpParam::RINGMLA::MaskType>(AtbOps::OpParam::RINGMLA::MaskType::MASK_TYPE_NORM);
        ringMLAParam.isTriuMask = 1;
    }
}

RingMLAOpsRunner::~RingMLAOpsRunner() {}
} // namespace atb
