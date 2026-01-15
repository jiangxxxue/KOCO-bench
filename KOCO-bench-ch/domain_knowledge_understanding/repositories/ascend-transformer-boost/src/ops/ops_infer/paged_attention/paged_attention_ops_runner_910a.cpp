/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "paged_attention_ops_runner_910a.h"
#include <cmath>
#include <asdops/params/params.h>
#include <atbops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/config.h"
#include "paged_attention_runner_utils.h"

namespace atb {
void PATransQViewFunc910a(const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims)
{
    if (oldDims.size() < 3) { // 3: 最小维度
        return;
    }
    newDims = {1, oldDims.at(0), oldDims.at(1) * oldDims.at(2)}; // 1, 2: 设置新张量形状
}

PagedAttentionOpsRunner910A::PagedAttentionOpsRunner910A(const infer::PagedAttentionParam &param)
    : OpsRunner("PagedAttentionOpsRunner", RUNNER_TYPE_PAGED_ATTENTION), param_(param)
{
    needKernelGraphModify_ = true;
    ATB_LOG(INFO) << "PagedAttentionOpsRunner910A::PagedAttentionOpsRunner910A called";
}

Status PagedAttentionOpsRunner910A::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    bool needMask = (param_.maskType != atb::infer::PagedAttentionParam::UNDEFINED);
    bool needQLens = (param_.calcType == atb::infer::PagedAttentionParam::CALC_TYPE_SPEC);
    bool needLogN = (param_.scaleType == atb::infer::PagedAttentionParam::SCALE_TYPE_LOGN);

    std::size_t intensorSize = 5; // 5: q, k, v, block_tables, contextLens
    if (needMask) {
        intensorSize += 1;
    }
    if (needQLens) {
        intensorSize += 1;
    }
    if (needLogN) {
        intensorSize += 1;
    }
    kernelGraph_.inTensors.resize(intensorSize);
    kernelGraph_.outTensors.resize(1);

    size_t tensorId = 0;
    Mki::Tensor &query = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &keyCache = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &valueCache = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &blockTables = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &contextLens = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor *mask = needMask ? &kernelGraph_.inTensors.at(tensorId++) : &nullTensor_;
    Mki::Tensor *qLens = needQLens ? &kernelGraph_.inTensors.at(tensorId++) : &nullTensor_;
    Mki::Tensor *logN = needLogN ? &kernelGraph_.inTensors.at(tensorId++) : &nullTensor_;

    (void)qLens;

    Mki::Tensor &contextOut = kernelGraph_.outTensors.at(0);

    auto attnMaskFormat = needMask ? opsTensorPack.inTensors.at(5).desc.format :
                                     static_cast<Mki::TensorFormat>(ACL_FORMAT_UNDEFINED); // 5: attnMask tensor
    kernelGraph_.internalTensors.resize(
        ((attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND)) && needMask) ? 4 : 3); // 4, 3: 中间tensor数

    size_t internalTensorId = 0;
    Mki::Tensor &transdataQResultTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &mulQTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor *transdataAttnMaskTensor =
        ((attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND)) && needMask) ?
            &kernelGraph_.internalTensors.at(internalTensorId++) :
            mask;
    Mki::Tensor &context = kernelGraph_.internalTensors.at(internalTensorId++);

    kernelGraph_.nodes.resize((attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND) && needMask) ?
                                  5 : // 5: attn mask 为nd时
                                  4); // 4: attn mask 为nz时

    size_t nodeId = 0;
    auto &transdataQNode = kernelGraph_.nodes.at(nodeId++);
    transdataQNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataQNode.inTensors = {&query};
    transdataQNode.outTensors = {&transdataQResultTensor};
    transdataQNode.inTensorViewFuncs.resize(transdataQNode.inTensors.size());
    transdataQNode.inTensorViewFuncs[0] = &PATransQViewFunc910a;
    transdataQNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        ntokens_ = launchParam.GetInTensor(0).desc.dims.at(1);
        hiddenSize_ = launchParam.GetInTensor(0).desc.dims.at(2); // 2: 第三维
    };

    // muls
    auto &mulsQNode = kernelGraph_.nodes.at(nodeId++);
    mulsQNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, 1.0})};
    mulsQNode.inTensors = {&transdataQResultTensor};
    mulsQNode.inTensorViewFuncs.resize(mulsQNode.inTensors.size());
    mulsQNode.outTensors = {&mulQTensor};

    // Convert attentionMask from ND format to NZ format
    if ((attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND)) && needMask) {
        auto &transdataAttnMaskNode = kernelGraph_.nodes[nodeId++];
        transdataAttnMaskNode.opDesc = {
            0, "TransdataOperation",
            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
        transdataAttnMaskNode.inTensors = {mask};
        transdataAttnMaskNode.outTensors = {transdataAttnMaskTensor};
    }

    auto &pagedAttentionNode = kernelGraph_.nodes.at(nodeId++);
    AtbOps::OpParam::PagedAttention inPagedAttention;
    inPagedAttention.headSize = param_.headNum;
    inPagedAttention.tor = param_.qkScale;
    inPagedAttention.kvHead = param_.kvHeadNum;
    inPagedAttention.maskType = static_cast<AtbOps::OpParam::PagedAttention::MaskType>(param_.maskType);
    inPagedAttention.type = AtbOps::OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK;
    inPagedAttention.scaleType = (param_.scaleType == atb::infer::PagedAttentionParam::SCALE_TYPE_LOGN) ?
                                     AtbOps::OpParam::PagedAttention::SCALE_LOGN :
                                     AtbOps::OpParam::PagedAttention::SCALE_TOR;
    pagedAttentionNode.opDesc = {0, "PagedAttentionOperation", inPagedAttention};

    if (needMask && (attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND))) {
        pagedAttentionNode.inTensors = {
            &mulQTensor, &keyCache, &valueCache, &blockTables, &contextLens, transdataAttnMaskTensor, logN};
    } else {
        pagedAttentionNode.inTensors = {&mulQTensor, &keyCache, &valueCache, &blockTables, &contextLens, mask, logN};
    }
    pagedAttentionNode.outTensors = {&context};
    pagedAttentionNode.inTensorViewFuncs.resize(pagedAttentionNode.inTensors.size()); // view
    pagedAttentionNode.tilingCacheEnable = false;

    auto &transdataAttnNode = kernelGraph_.nodes.at(nodeId++);
    transdataAttnNode.opDesc = {
        0, "TransdataOperation",
        AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {ntokens_, hiddenSize_}})};
    transdataAttnNode.inTensors = {&context};
    transdataAttnNode.outTensors = {&contextOut};
    transdataAttnNode.inferShapePreFunc = [=](Mki::LaunchParam &launchParam) {
        launchParam.SetParam(
            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {ntokens_, hiddenSize_}}));
    };
    newParam_.batchRunStatus.reserve(128); // 128: 预留大小
    newParam_.contextLens.reserve(128);    // 128: 预留大小
    newParam_.qLens.reserve(128);          // 128: 预留大小
    return NO_ERROR;
}

PagedAttentionOpsRunner910A::~PagedAttentionOpsRunner910A() {}

Status PagedAttentionOpsRunner910A::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    bool needMask = (param_.maskType != atb::infer::PagedAttentionParam::UNDEFINED);
    bool needQLens = (param_.calcType == atb::infer::PagedAttentionParam::CALC_TYPE_SPEC);
    int qLensIndex = needMask ? 6 : 5; // 6,5: qLensIndex 位置

    bool ret = newParam_.BuildFromTensor310P(opsTensorPack.inTensors, needQLens, qLensIndex);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " build param from host tensor fail";
        return ERROR_INVALID_PARAM;
    }
    auto attnMaskFormat =
        needMask ? opsTensorPack.inTensors.at(5).desc.format : static_cast<Mki::TensorFormat>(ACL_FORMAT_UNDEFINED);
    int paNodeId =
        (needMask && (attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND))) ? 3 : 2; // 2, 3 : pa node位置
    auto &pagedAttentionNode = kernelGraph_.nodes.at(paNodeId);
    AtbOps::OpParam::PagedAttention inPagedAttention;
    inPagedAttention.headSize = param_.headNum;
    inPagedAttention.tor = param_.qkScale;
    inPagedAttention.kvHead = param_.kvHeadNum;
    inPagedAttention.maskType = static_cast<AtbOps::OpParam::PagedAttention::MaskType>(param_.maskType);
    inPagedAttention.type = AtbOps::OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK;
    inPagedAttention.scaleType = (param_.scaleType == atb::infer::PagedAttentionParam::SCALE_TYPE_LOGN) ?
                                     AtbOps::OpParam::PagedAttention::SCALE_LOGN :
                                     AtbOps::OpParam::PagedAttention::SCALE_TOR;
    inPagedAttention.qSeqLen = newParam_.qLens;
    pagedAttentionNode.opDesc = {0, "PagedAttentionOperation", inPagedAttention};
    ATB_LOG(INFO) << GetLogPrefix() << " update AtbOps::OpParam::PagedAttention.headNum:" << param_.headNum
                  << ", qkScale:" << param_.qkScale << ", kvHead:" << param_.kvHeadNum
                  << ", qLens: " << newParam_.qLens.size();
    return NO_ERROR;
}

void PagedAttentionOpsRunner910A::SetParam(const Mki::Any &param)
{
    infer::PagedAttentionParam newParam = Mki::AnyCast<infer::PagedAttentionParam>(param);
    if (!IsParamEqual(newParam, param_)) {
        param_ = newParam;
        isParamUpdated_ = true;
    }
}
} // namespace atb