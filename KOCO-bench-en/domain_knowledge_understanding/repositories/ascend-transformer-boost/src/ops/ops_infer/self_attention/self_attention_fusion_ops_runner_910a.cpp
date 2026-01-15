/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "self_attention_fusion_ops_runner_910a.h"
#include <cmath>
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
#include "param.h"
#include "atb/utils/runner_util.h"
#include "self_attention_runner_utils.h"

namespace atb {
void TransKVViewFunc910a(const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims)
{
    if (oldDims.size() < 2) { // 2: 最小维度
        ATB_LOG(ERROR) << "intensor key/value's dimNum shoule be at least 2";
        newDims.clear();
        return;
    }
    if (oldDims.size() != 4) {                       // 4: 维度长度
        newDims = {1, oldDims.at(0), oldDims.at(1)}; // 0, 1: 设置新张量形状
        return;
    }
    newDims = {1, oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)}; // 2, 3: 设置新张量形状
}

void TransQViewFunc910a(const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims)
{
    if (oldDims.size() < 2) { // 2: 最小维度
        ATB_LOG(ERROR) << "intensor query's dimNum shoule be at least 2";
        newDims.clear();
        return;
    }
    if (oldDims.size() != 4) {                       // 4: 维度长度
        newDims = {1, oldDims.at(0), oldDims.at(1)}; // 0, 1: 设置新张量形状
        return;
    }
    newDims = {oldDims.at(0), oldDims.at(1), oldDims.at(2) * oldDims.at(3)}; // 2, 3: 设置新张量形状
}

void TransAttnMaskViewFunc910a(const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims)
{
    if (oldDims.size() != 4) { // 4: 维度长度
        ATB_LOG(ERROR) << "attnMask intensor support alibi, shape should exceed four dim";
        newDims.clear();
        return;
    }
    newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2), oldDims.at(3)}; // 2, 3: remains the same
}

void FlashAttentionInferShapePreFunc910a(Mki::LaunchParam &launchParam)
{
    if (launchParam.GetInTensors().size() < 4) { // 4: inTensor数量不少于4
        ATB_LOG(ERROR) << "inTensor num should be at least 5";
        return;
    }
    launchParam.GetInTensor(3).desc.dtype = Mki::TENSOR_DTYPE_UINT32; // 3: 设置第四个输入张量的dtype
}

SelfAttentionFusionOpsRunner910A::SelfAttentionFusionOpsRunner910A(const infer::SelfAttentionParam &param)
    : OpsRunner("SelfAttentionFusionOpsRunner", RUNNER_TYPE_SELF_ATTENTION), param_(param)
{
    needKernelGraphModify_ = true;
    needMask_ = param.maskType != infer::SelfAttentionParam::MASK_TYPE_UNDEFINED &&
                !(param.calcType == infer::SelfAttentionParam::DECODER &&
                  param.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_NORM);
    ATB_LOG(INFO) << "SelfAttentionFusionOpsRunner910A::SelfAttentionFusionOpsRunner910A called";
}

Status SelfAttentionFusionOpsRunner910A::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    bool isMaskCompress = (param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
                           param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT);
    bool needLogN = (param_.scaleType == atb::infer::SelfAttentionParam::SCALE_TYPE_LOGN);

    std::size_t intensorSize = 8; // 8: 设置inTensor数
    if (needMask_) {
        intensorSize++;
    }
    if (param_.batchRunStatusEnable) {
        intensorSize++;
    }
    if (isMaskCompress) {
        intensorSize++;
    }
    if (needLogN) {
        intensorSize++;
    }
    kernelGraph_.inTensors.resize(intensorSize); // 9: 设置inTensor数
    size_t tensorId = 0;
    Mki::Tensor &mixedQuery = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &mixedKey = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &mixedValue = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &cacheK = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &cacheV = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor *attentionMask = needMask_ ? &kernelGraph_.inTensors.at(tensorId++) : &nullTensor_;
    Mki::Tensor &tokenOffset = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &seqLen = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &layerId = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor *slopes = isMaskCompress ? &kernelGraph_.inTensors.at(tensorId++) : &nullTensor_;
    Mki::Tensor *logN = needLogN ? &kernelGraph_.inTensors.at(tensorId++) : &nullTensor_;

    ATB_LOG(INFO) << GetLogPrefix() << "tokenOffset dataSize:" << tokenOffset.dataSize;
    ATB_LOG(INFO) << GetLogPrefix() << "seqLen dataSize:" << seqLen.dataSize;

    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &contextOut = kernelGraph_.outTensors.at(0);
    auto attnMaskFormat =
        needMask_ ? opsTensorPack.inTensors.at(5).desc.format : static_cast<Mki::TensorFormat>(ACL_FORMAT_UNDEFINED);
    kernelGraph_.internalTensors.resize(
        (attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_FRACTAL_NZ) || (!needMask_)) ?
            7 :
            (param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI ? 8 : 7)); // 7, 8: 设置总节点数

    size_t internalTensorId = 0;
    Mki::Tensor &transdataKResultTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &transdataVResultTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &transdataQResultTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &transposeQResultTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &divOut = kernelGraph_.internalTensors.at(internalTensorId++);

    Mki::Tensor *transdataAttnMaskTensor =
        ((attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND)) && needMask_) ?
            &kernelGraph_.internalTensors.at(internalTensorId++) :
            attentionMask;

    Mki::Tensor &context = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &contextTranspose = kernelGraph_.internalTensors.at(internalTensorId++);

    kernelGraph_.nodes.resize((attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND) && needMask_) ?
                                  11 :
                                  10); // 10, 11: 设置总节点数
    size_t nodeId = 0;

    auto &transdataKNode = kernelGraph_.nodes.at(nodeId++);
    transdataKNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataKNode.inTensors = {&mixedKey};
    transdataKNode.outTensors = {&transdataKResultTensor};
    transdataKNode.inTensorViewFuncs.resize(transdataKNode.inTensors.size());
    transdataKNode.inTensorViewFuncs[0] = &TransKVViewFunc910a;

    auto &kCacheNode = kernelGraph_.nodes.at(nodeId++);
    AtbOps::OpParam::KVCache kCacheParam;
    kCacheParam.type = AtbOps::OpParam::KVCache::KVCACHE_NZ_PARAMS;
    kCacheParam.qSeqLen = {};
    kCacheParam.kvSeqLen = {};
    kCacheParam.batchRunStatus = {};
    kCacheNode.opDesc = {0, "KVCacheOperation", kCacheParam};
    kCacheNode.inTensors = {&transdataKResultTensor, &layerId, &cacheK};
    kCacheNode.outTensors = {&cacheK};

    auto &transdataVNode = kernelGraph_.nodes.at(nodeId++);
    transdataVNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataVNode.inTensors = {&mixedValue};
    transdataVNode.outTensors = {&transdataVResultTensor};
    transdataVNode.inTensorViewFuncs.resize(transdataVNode.inTensors.size());
    transdataVNode.inTensorViewFuncs[0] = &TransKVViewFunc910a;

    auto &vCacheNode = kernelGraph_.nodes.at(nodeId++);
    AtbOps::OpParam::KVCache vCacheParam;
    vCacheParam.type = AtbOps::OpParam::KVCache::KVCACHE_NZ_PARAMS;
    vCacheParam.qSeqLen = {};
    vCacheParam.kvSeqLen = {};
    vCacheParam.batchRunStatus = {};
    vCacheNode.opDesc = {0, "KVCacheOperation", vCacheParam};
    vCacheNode.inTensors = {&transdataVResultTensor, &layerId, &cacheV};
    vCacheNode.outTensors = {&cacheV};

    auto &transdataQNode = kernelGraph_.nodes.at(nodeId++);
    transdataQNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    transdataQNode.inTensors = {&mixedQuery};
    transdataQNode.outTensors = {&transdataQResultTensor};
    transdataQNode.inTensorViewFuncs.resize(transdataQNode.inTensors.size());
    transdataQNode.inTensorViewFuncs[0] = &TransQViewFunc910a;
    transdataQNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        if (launchParam.GetInTensor(0).desc.dims.size() > 2) { // 2: 至少三维
            batch_ = launchParam.GetInTensor(0).desc.dims.at(0);
            ntokens_ = launchParam.GetInTensor(0).desc.dims.at(1);
            hiddenSize_ = launchParam.GetInTensor(0).desc.dims.at(2); // 2: 第三维
        } else {
            ATB_LOG(ERROR) << "inTensor dim should be at least 3";
        }
    };

    auto &permuteQAfterNode = kernelGraph_.nodes.at(nodeId++);
    AsdOps::OpParam::Transpose permuteQNodeAfterParam = {{1, 0, 2, 3}};
    permuteQAfterNode.opDesc = {0, "TransposeOperation", permuteQNodeAfterParam};
    permuteQAfterNode.inTensors = {&transdataQResultTensor};
    permuteQAfterNode.outTensors = {&transposeQResultTensor};

    // muls
    auto &mulsQNode = kernelGraph_.nodes.at(nodeId++);
    mulsQNode.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, param_.qScale})};
    mulsQNode.inTensors = {&transposeQResultTensor};
    mulsQNode.inTensorViewFuncs.resize(mulsQNode.inTensors.size());
    mulsQNode.inTensorViewFuncs[0] = [](const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims) {
        if (oldDims.size() > 3) { // 3: 维度必须大于3
            newDims = {1, oldDims.at(0), oldDims.at(1) * oldDims.at(2), oldDims.at(3)};
        } else {
            ATB_LOG(ERROR) << "oldDim should be at least 4";
            newDims.clear();
        }
    };
    mulsQNode.outTensors = {&divOut};

    // Convert attentionMask from ND format to NZ format
    if ((attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND)) && needMask_) {
        auto &transdataAttnMaskNode = kernelGraph_.nodes.at(nodeId++);
        transdataAttnMaskNode.opDesc = {
            0, "TransdataOperation",
            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
        transdataAttnMaskNode.inTensors = {attentionMask};
        transdataAttnMaskNode.outTensors = {transdataAttnMaskTensor};
        transdataAttnMaskNode.inTensorViewFuncs.resize(transdataAttnMaskNode.inTensors.size());
        transdataAttnMaskNode.inTensorViewFuncs[0] = &TransAttnMaskViewFunc910a;
    }

    // mix.h mixType=UNPAD_FLASH_ATTENTION
    auto &flashAttentionNode = kernelGraph_.nodes.at(nodeId++);
    AtbOps::OpParam::UnpadFlashAttentionNz flashAttentionQParam;
    SetFAParam(flashAttentionQParam);
    flashAttentionNode.opDesc = {0, "UnpadFlashAttentionNzOperation", flashAttentionQParam};
    if (needMask_ && attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND)) {
        newParam_.attnMaskFormat = static_cast<aclFormat>(attnMaskFormat);
        flashAttentionNode.inTensors = {&divOut, &cacheK, &cacheV, &layerId, transdataAttnMaskTensor, slopes, logN};
    } else {
        flashAttentionNode.inTensors = {&divOut, &cacheK, &cacheV, &layerId, attentionMask, slopes, logN};
    }
    flashAttentionQParam.type = GetFaType();

    flashAttentionNode.outTensors = {&context};

    flashAttentionNode.inferShapePreFunc = &FlashAttentionInferShapePreFunc910a;

    auto &permuteContextNode = kernelGraph_.nodes.at(nodeId++);
    AsdOps::OpParam::Transpose permuteContextParam = {{1, 0, 2, 3}};
    permuteContextNode.opDesc = {0, "TransposeOperation", permuteContextParam};
    permuteContextNode.inTensors = {&context};
    permuteContextNode.inTensorViewFuncs.resize(permuteContextNode.inTensors.size());
    permuteContextNode.inTensorViewFuncs[0] = [&](const Mki::SVector<int64_t> &oldDims,
                                                  Mki::SVector<int64_t> &newDims) {
        if (oldDims.size() > 3) { // 3： 维度必须大于3
            newDims = {oldDims.at(1), batch_, oldDims.at(2) / batch_, oldDims.at(3)};
        } else {
            ATB_LOG(ERROR) << "oldDim should be at least 4";
            newDims.clear();
        }
    };
    permuteContextNode.outTensors = {&contextTranspose};

    auto &transdataAttnNode = kernelGraph_.nodes.at(nodeId++);
    transdataAttnNode.opDesc = {
        0, "TransdataOperation",
        AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {ntokens_, hiddenSize_}})};
    transdataAttnNode.inTensors = {&contextTranspose};
    transdataAttnNode.outTensors = {&contextOut};
    transdataAttnNode.inferShapePreFunc = [=](Mki::LaunchParam &launchParam) {
        launchParam.SetParam(
            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {ntokens_, hiddenSize_}}));
    };

    newParam_.tokenOffset.reserve(128);    // 128::预留大小
    newParam_.seqLen.reserve(128);         // 128::预留大小
    newParam_.batchRunStatus.reserve(128); // 128::预留大小
    return NO_ERROR;
}

Status SelfAttentionFusionOpsRunner910A::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    bool hasKV = (param_.kvcacheCfg != atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS);
    bool ret = newParam_.BuildFromTensor(opsTensorPack.inTensors, param_.batchRunStatusEnable, needMask_, hasKV);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " build param from host tensor fail";
        return ERROR_INVALID_PARAM;
    }
    size_t tokenOffsetTensorIdBase = 5; // 6, 5: tokenOffset id 4, 3: Kv0 tokenOffset id
    const size_t tokenOffsetTensorId = needMask_ ? tokenOffsetTensorIdBase + 1 : tokenOffsetTensorIdBase;
    ret = ModifyKVCacheNode(tokenOffsetTensorId);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " set kvcache param fail";
        return ERROR_INVALID_TENSOR_DIM;
    }
    auto &flashAttentionNode = kernelGraph_.nodes.at(
        (newParam_.attnMaskFormat == ACL_FORMAT_ND && needMask_) ? 8 : 7); // 7, 8: flashAttention节点位置

    AtbOps::OpParam::UnpadFlashAttentionNz::Type faType = GetFaType();
    ATB_LOG(INFO) << "newParam_.attnMaskFormat" << newParam_.attnMaskFormat;

    AtbOps::OpParam::UnpadFlashAttentionNz unpadFlashAttentionNzParam;

    SetFAParam(unpadFlashAttentionNzParam);
    unpadFlashAttentionNzParam.qSeqLen = newParam_.seqLen;
    unpadFlashAttentionNzParam.kvSeqLen = newParam_.tokenOffset;
    unpadFlashAttentionNzParam.batchRunStatus = newParam_.batchRunStatus;
    flashAttentionNode.opDesc = {0, "UnpadFlashAttentionNzOperation", unpadFlashAttentionNzParam};

    ATB_LOG(INFO) << GetLogPrefix() << " update AtbOps::OpParam::UnpadFlashAttentionNz.headNum:" << param_.headNum
                  << ", qkScale:" << param_.qkScale << ", kvHeadNum:" << param_.kvHeadNum << ", type: " << faType;
    return NO_ERROR;
}

bool SelfAttentionFusionOpsRunner910A::ModifyKVCacheNode(const size_t tokenOffsetTensorId)
{
    size_t tensorId = 3; // 3: cacheK
    Mki::Tensor &cacheK = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &cacheV = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &tokenOffset = kernelGraph_.inTensors.at(tokenOffsetTensorId);
    Mki::Tensor &seqLen = kernelGraph_.inTensors.at(tokenOffsetTensorId + 1);
    Mki::Tensor &layerId = kernelGraph_.inTensors.at(tokenOffsetTensorId + 2); // tokenOffsetTensorId+2: layerId
    Mki::Tensor &transdataKResultTensor = kernelGraph_.internalTensors.at(0);
    Mki::Tensor &transdataVResultTensor = kernelGraph_.internalTensors.at(1);

    bool kvcacheWithParam = tokenOffset.data == nullptr || seqLen.data == nullptr;

    auto &kCacheNode = kernelGraph_.nodes.at(1); // 1: KCache节点位置
    AtbOps::OpParam::KVCache kCacheParam;
    kCacheParam.batchRunStatus = newParam_.batchRunStatus;
    kCacheNode.inTensors = {&transdataKResultTensor, &layerId, &cacheK};
    if (kvcacheWithParam) {
        kCacheParam.type = AtbOps::OpParam::KVCache::KVCACHE_NZ_PARAMS;
        kCacheParam.seqLen = newParam_.seqLen;
        kCacheParam.tokenOffset = newParam_.tokenOffset;
    } else {
        kCacheParam.type = AtbOps::OpParam::KVCache::KVCACHE_NZ;
        kCacheNode.inTensors.push_back(&tokenOffset);
        kCacheNode.inTensors.push_back(&seqLen);
    }
    kCacheNode.opDesc = {0, "KVCacheOperation", kCacheParam};

    auto &vCacheNode = kernelGraph_.nodes.at(3); // 3: VCache节点位置
    AtbOps::OpParam::KVCache vCacheParam;
    vCacheParam.batchRunStatus = newParam_.batchRunStatus;
    vCacheNode.inTensors = {&transdataVResultTensor, &layerId, &cacheV};
    if (kvcacheWithParam) {
        vCacheParam.type = AtbOps::OpParam::KVCache::KVCACHE_NZ_PARAMS;
        vCacheParam.seqLen = newParam_.seqLen;
        vCacheParam.tokenOffset = newParam_.tokenOffset;
    } else {
        vCacheParam.type = AtbOps::OpParam::KVCache::KVCACHE_NZ;
        vCacheNode.inTensors.push_back(&tokenOffset);
        vCacheNode.inTensors.push_back(&seqLen);
    }
    vCacheNode.opDesc = {0, "KVCacheOperation", vCacheParam};
    return true;
}

AtbOps::OpParam::UnpadFlashAttentionNz::Type SelfAttentionFusionOpsRunner910A::GetFaType()
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
    return faType;
}

void SelfAttentionFusionOpsRunner910A::SetFAParam(AtbOps::OpParam::UnpadFlashAttentionNz &flashAttentionParam)
{
    flashAttentionParam.type = GetFaType();

    flashAttentionParam.headSize = param_.headNum;
    flashAttentionParam.tor = param_.qkScale;
    flashAttentionParam.kvHead = param_.kvHeadNum;
    flashAttentionParam.isTriuMask = param_.isTriuMask;
    flashAttentionParam.windowSize = static_cast<int32_t>(param_.windowSize);
    flashAttentionParam.cacheType = static_cast<AtbOps::OpParam::UnpadFlashAttentionNz::CacheType>(param_.cacheType);

    flashAttentionParam.maskType =
        param_.maskType == infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS ?
            static_cast<AtbOps::OpParam::UnpadFlashAttentionNz::MaskType>(1) : // 1: norm mask
            ((param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
              param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT) ?
                 static_cast<AtbOps::OpParam::UnpadFlashAttentionNz::MaskType>(2) : // 2: alibi mask
                 static_cast<AtbOps::OpParam::UnpadFlashAttentionNz::MaskType>(param_.maskType));
    if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_NORM) {
        flashAttentionParam.maskType = AtbOps::OpParam::UnpadFlashAttentionNz::MaskType::MASK_TYPE_SWA_NORM;
    } else if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_COMPRESS) {
        flashAttentionParam.maskType = AtbOps::OpParam::UnpadFlashAttentionNz::MaskType::MASK_TYPE_SWA_COMPRESS;
    }
    flashAttentionParam.isAlibiMaskSqrt = (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT);
    flashAttentionParam.scaleType = param_.scaleType == atb::infer::SelfAttentionParam::SCALE_TYPE_LOGN ?
                                        AtbOps::OpParam::UnpadFlashAttentionNz::SCALE_LOGN :
                                        AtbOps::OpParam::UnpadFlashAttentionNz::SCALE_TOR;
}

void SelfAttentionFusionOpsRunner910A::SetParam(const Mki::Any &param)
{
    infer::SelfAttentionParam newParam = Mki::AnyCast<infer::SelfAttentionParam>(param);
    if (!IsParamEqual(newParam, param_)) {
        param_ = newParam;
        isParamUpdated_ = true;
        needMask_ = param_.maskType != infer::SelfAttentionParam::MASK_TYPE_UNDEFINED &&
                    !(param_.calcType == infer::SelfAttentionParam::DECODER &&
                      param_.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_NORM);
    }
}

SelfAttentionFusionOpsRunner910A::~SelfAttentionFusionOpsRunner910A() {}
} // namespace atb