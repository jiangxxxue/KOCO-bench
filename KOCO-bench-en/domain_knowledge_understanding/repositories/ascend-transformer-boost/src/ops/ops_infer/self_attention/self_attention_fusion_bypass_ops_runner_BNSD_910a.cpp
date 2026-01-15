/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "self_attention_fusion_bypass_ops_runner_BNSD_910a.h"
#include <cmath>
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
#include "param.h"
#include "atb/utils/runner_util.h"
#include "self_attention_runner_utils.h"

namespace atb {
void TransQViewFuncBypassBNSD910a(const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims)
{
    if (oldDims.size() < 2) { // 2: 最小维度
        ATB_LOG(ERROR) << "qunery intensor expects dimNum to be at least 2";
        newDims.clear();
        return;
    }
    if (oldDims.size() != 4) {                       // 4: 维度长度
        newDims = {1, oldDims.at(0), oldDims.at(1)}; // 0, 1: 设置新张量形状
        return;
    }
    newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)}; // 2, 3: 设置新张量形状
}

void TransAttnMaskViewFuncBypassBNSD910a(const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims)
{
    if (oldDims.size() != 4) { // 4: 维度长度
        ATB_LOG(ERROR) << "attnMask intensor support alibi, shape should exceed four dim";
        return;
    }
    newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)}; // 2, 3: remains the same
}

void FlashAttentionInferShapePreFuncBypassBNSD910a(Mki::LaunchParam &launchParam)
{
    if (launchParam.GetInTensors().size() < 4) { // 4: inTensor数量不少于4
        ATB_LOG(ERROR) << "inTensor num should be at least 5";
        return;
    }
    launchParam.GetInTensor(3).desc.dtype = Mki::TENSOR_DTYPE_UINT32; // 3: 设置第四个输入张量的dtype
}

SelfAttentionFusionBypassOpsRunnerBNSD910A::SelfAttentionFusionBypassOpsRunnerBNSD910A(
    const infer::SelfAttentionParam &param)
    : OpsRunner("SelfAttentionFusionBypassOpsRunnerBNSD910A", RUNNER_TYPE_SELF_ATTENTION_KV_BYPASS_BNSD), param_(param)
{
    needKernelGraphModify_ = true;
    ATB_LOG(INFO) << "SelfAttentionFusionBypassOpsRunnerBNSD910A::SelfAttentionFusionBypassOpsRunnerBNSD910A called";
}

Status SelfAttentionFusionBypassOpsRunnerBNSD910A::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    bool needMask = (param_.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED);
    bool isMaskCompress = (param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
                           param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT);

    std::size_t intensorSize = 6; // 6: 设置inTensor数
    if (needMask) {
        intensorSize++;
    }
    if (param_.batchRunStatusEnable) {
        intensorSize++;
    }
    if (isMaskCompress) {
        intensorSize++;
    }
    kernelGraph_.inTensors.resize(intensorSize); // 9: 设置inTensor数
    size_t tensorId = 0;
    Mki::Tensor &mixedQuery = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &cacheK = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &cacheV = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor *attentionMask = needMask ? &kernelGraph_.inTensors.at(tensorId++) : &nullTensor_;
    Mki::Tensor &tokenOffset = kernelGraph_.inTensors.at(tensorId++);
    (void)tokenOffset;
    Mki::Tensor &seqLen = kernelGraph_.inTensors.at(tensorId++);
    (void)seqLen;
    Mki::Tensor &layerId = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor *slopes = isMaskCompress ? &kernelGraph_.inTensors.at(tensorId++) : &nullTensor_;

    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &contextOut = kernelGraph_.outTensors.at(0);
    auto attnMaskFormat = needMask ? opsTensorPack.inTensors.at(3).desc.format : // bypass true 3
                                     static_cast<Mki::TensorFormat>(ACL_FORMAT_UNDEFINED);
    kernelGraph_.internalTensors.resize(
        (attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_FRACTAL_NZ) || (!needMask)) ?
            3 :
            (param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI ? 4 : 3)); // 4, 3: 设置中间tensor数

    size_t internalTensorId = 0;
    Mki::Tensor &transdataQResultTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &divOut = kernelGraph_.internalTensors.at(internalTensorId++);

    Mki::Tensor *transdataAttnMaskTensor =
        ((attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND)) && needMask) ?
            &kernelGraph_.internalTensors.at(internalTensorId++) :
            attentionMask;

    Mki::Tensor &context = kernelGraph_.internalTensors.at(internalTensorId++);

    kernelGraph_.nodes.resize(
        (attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND) && needMask) ? 5 : 4); // 5, 4: 设置总节点数
    size_t nodeId = 0;

    auto &transdataQNode = kernelGraph_.nodes.at(nodeId++);
    transdataQNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataQNode.inTensors = {&mixedQuery};
    transdataQNode.outTensors = {&transdataQResultTensor};
    transdataQNode.inTensorViewFuncs.resize(transdataQNode.inTensors.size());
    transdataQNode.inTensorViewFuncs[0] = &TransQViewFuncBypassBNSD910a;
    transdataQNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        if (launchParam.GetInTensor(0).desc.dims.size() > 2) { // 2: 至少三维
            batchMulsHeadNum_ = launchParam.GetInTensor(0).desc.dims.at(0);
            maxSeqLen_ = launchParam.GetInTensor(0).desc.dims.at(1);
            headDim_ = launchParam.GetInTensor(0).desc.dims.at(2); // 2: 第三维
        } else {
            ATB_LOG(ERROR) << "inTensor dim should be at least 3";
        }
    };

    // muls
    auto &mulsQNode = kernelGraph_.nodes.at(nodeId++);
    mulsQNode.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, param_.qScale})};
    mulsQNode.inTensors = {&transdataQResultTensor};
    mulsQNode.outTensors = {&divOut};

    // Convert attentionMask from ND format to NZ format
    if ((attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND)) && needMask) {
        auto &transdataAttnMaskNode = kernelGraph_.nodes.at(nodeId++);
        transdataAttnMaskNode.opDesc = {
            0, "TransdataOperation",
            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
        transdataAttnMaskNode.inTensors = {attentionMask};
        transdataAttnMaskNode.outTensors = {transdataAttnMaskTensor};
        transdataAttnMaskNode.inTensorViewFuncs.resize(transdataAttnMaskNode.inTensors.size());
        transdataAttnMaskNode.inTensorViewFuncs[0] = &TransAttnMaskViewFuncBypassBNSD910a;
    }

    // mix.h type=UNPAD_FLASH_ATTENTION
    auto &flashAttentionNode = kernelGraph_.nodes.at(nodeId++);
    AtbOps::OpParam::UnpadFlashAttentionNz flashAttentionQParam;
    SetFAParam(flashAttentionQParam);
    flashAttentionNode.opDesc = {0, "UnpadFlashAttentionNzOperation", flashAttentionQParam};
    if (needMask && attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND)) {
        newParam_.attnMaskFormat = static_cast<aclFormat>(attnMaskFormat);
        flashAttentionNode.inTensors = {&divOut, &cacheK,     &cacheV, &layerId, transdataAttnMaskTensor,
                                        slopes,  &nullTensor_};
    } else {
        flashAttentionNode.inTensors = {&divOut, &cacheK, &cacheV, &layerId, attentionMask, slopes, &nullTensor_};
    }

    flashAttentionNode.outTensors = {&context};

    flashAttentionNode.inferShapePreFunc = &FlashAttentionInferShapePreFuncBypassBNSD910a;

    auto &transdataAttnNode = kernelGraph_.nodes.at(nodeId++);
    transdataAttnNode.opDesc = {
        0, "TransdataOperation",
        AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {maxSeqLen_, headDim_}})};
    transdataAttnNode.inTensors = {&context};
    transdataAttnNode.outTensors = {&contextOut};
    transdataAttnNode.inferShapePreFunc = [=](Mki::LaunchParam &launchParam) {
        launchParam.SetParam(
            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {maxSeqLen_, headDim_}}));
    };

    newParam_.tokenOffset.reserve(128);    // 128::预留大小
    newParam_.seqLen.reserve(128);         // 128::预留大小
    newParam_.batchRunStatus.reserve(128); // 128::预留大小
    return NO_ERROR;
}

Status SelfAttentionFusionBypassOpsRunnerBNSD910A::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    bool needMask = (param_.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED);
    bool hasKV = (param_.kvcacheCfg != atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS);
    bool ret = newParam_.BuildFromTensor(opsTensorPack.inTensors, param_.batchRunStatusEnable, needMask, hasKV);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " build param from host tensor fail";
        return ERROR_INVALID_PARAM;
    }

    auto &flashAttentionNode = kernelGraph_.nodes.at(
        (newParam_.attnMaskFormat == ACL_FORMAT_ND && needMask) ? 3 : 2); // 3, 2: flashAttention节点位置

    ATB_LOG(INFO) << "newParam_.attnMaskFormat" << newParam_.attnMaskFormat;

    AtbOps::OpParam::UnpadFlashAttentionNz inMix;

    SetFAParam(inMix);
    inMix.qSeqLen = newParam_.seqLen;
    inMix.kvSeqLen = newParam_.tokenOffset;
    inMix.batchRunStatus = newParam_.batchRunStatus;
    if (newParam_.isUseTensorList) { // kvcache使用tensorList
        Status st = SetKVCacheTensorList(inMix, flashAttentionNode);
        if (st != NO_ERROR) {
            return st;
        }
    }

    flashAttentionNode.opDesc = {0, "UnpadFlashAttentionNzOperation", inMix};

    ATB_LOG(INFO) << GetLogPrefix() << " update AtbOps::OpParam::UnpadFlashAttentionNz.headNum:" << param_.headNum
                  << ", qkScale:" << param_.qkScale << ", kvHeadNum:" << param_.kvHeadNum;
    return NO_ERROR;
}

Status SelfAttentionFusionBypassOpsRunnerBNSD910A::SetKVCacheTensorList(AtbOps::OpParam::UnpadFlashAttentionNz &param,
                                                                        KernelGraphNode &flashAttentionNode)
{
    if (param_.calcType != atb::infer::SelfAttentionParam::DECODER) {
        ATB_LOG(ERROR) << "param_.calcType should be DECODER!";
        return ERROR_INVALID_PARAM;
    }
    // 1:kcache Tensor 0:layer维度
    uint64_t kLayer = static_cast<uint64_t>(flashAttentionNode.inTensors.at(1)->desc.dims.at(0));
    // 2:vcache Tensor 0:layer维度
    uint64_t vLayer = static_cast<uint64_t>(flashAttentionNode.inTensors.at(2)->desc.dims.at(0));
    if (kLayer != 1 || vLayer != 1) {
        ATB_LOG(ERROR) << "Layer dim should be 1 when use tensorList in kCache and vCache.";
        return ERROR_INVALID_PARAM;
    }
    uint64_t batchSize = newParam_.seqLen.size();
    if (newParam_.kCache.size() != batchSize || newParam_.vCache.size() != batchSize) {
        ATB_LOG(ERROR) << "TensorListSize of kCache or vCache is not correct! kCacheSize:" << newParam_.kCache.size()
                       << " vCacheSize:" << newParam_.vCache.size() << " batchSize:" << batchSize;
        return ERROR_INVALID_PARAM;
    }
    param.kTensorList.resize(newParam_.kCache.size());
    param.vTensorList.resize(newParam_.vCache.size());
    for (std::size_t i = 0; i < newParam_.kCache.size(); i++) {
        TensorUtil::ConvertAtbTensor2OpsTensor(newParam_.kCache.at(i), param.kTensorList.at(i));
        TensorUtil::ConvertAtbTensor2OpsTensor(newParam_.vCache.at(i), param.vTensorList.at(i));
    }
    flashAttentionNode.inTensors.at(1) = &nullTensor_; // 1:kcache Tensor
    flashAttentionNode.inTensors.at(2) = &nullTensor_; // 2:vcache Tensor
    return NO_ERROR;
}

AtbOps::OpParam::UnpadFlashAttentionNz::Type SelfAttentionFusionBypassOpsRunnerBNSD910A::GetFaType()
{
    AtbOps::OpParam::UnpadFlashAttentionNz::Type faType =
        (param_.calcType == atb::infer::SelfAttentionParam::DECODER) ?
            AtbOps::OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_DECODER :
            ((param_.calcType == atb::infer::SelfAttentionParam::ENCODER) ?
                 AtbOps::OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_ENCODER :
                 AtbOps::OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ);
    faType = (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI ||
              param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
              param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT) ?
                 AtbOps::OpParam::UnpadFlashAttentionNz::UNPAD_ALIBI_FLASH_ATTENTION_NZ :
                 faType;
    faType = param_.batchRunStatusEnable ? AtbOps::OpParam::UnpadFlashAttentionNz::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION :
                                           faType;
    ATB_LOG(INFO) << GetLogPrefix() << "Type:" << faType;
    return faType;
}

void SelfAttentionFusionBypassOpsRunnerBNSD910A::SetFAParam(AtbOps::OpParam::UnpadFlashAttentionNz &flashAttentionParam)
{
    flashAttentionParam.type = GetFaType();

    flashAttentionParam.headSize = param_.headNum;
    flashAttentionParam.tor = param_.qkScale;
    flashAttentionParam.kvHead = param_.kvHeadNum;
    flashAttentionParam.isTriuMask = param_.isTriuMask;

    flashAttentionParam.maskType =
        param_.maskType == infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS ?
            static_cast<AtbOps::OpParam::UnpadFlashAttentionNz::MaskType>(1) : // 1: norm mask
            ((param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
              param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT) ?
                 static_cast<AtbOps::OpParam::UnpadFlashAttentionNz::MaskType>(2) : // 2: alibi mask
                 static_cast<AtbOps::OpParam::UnpadFlashAttentionNz::MaskType>(param_.maskType));
    flashAttentionParam.isAlibiMaskSqrt = (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT);
    flashAttentionParam.dataDimOrder =
        static_cast<AtbOps::OpParam::UnpadFlashAttentionNz::DataDimOrder>(param_.inputLayout);
}

void SelfAttentionFusionBypassOpsRunnerBNSD910A::SetParam(const Mki::Any &param)
{
    infer::SelfAttentionParam newParam = Mki::AnyCast<infer::SelfAttentionParam>(param);
    if (!IsParamEqual(newParam, param_)) {
        param_ = newParam;
        isParamUpdated_ = true;
    }
}

SelfAttentionFusionBypassOpsRunnerBNSD910A::~SelfAttentionFusionBypassOpsRunnerBNSD910A() {}
} // namespace atb
