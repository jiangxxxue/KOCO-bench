/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "self_attention_fusion_ops_runner.h"
#include <cmath>
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
#include "param.h"

static constexpr uint32_t VALUE_TENSOR_POS = 2;

namespace atb {
void SelfAttentionFusionViewFunc(const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims)
{
    if (oldDims.size() == 2) { // 2: 判断原张量的维数
        newDims = oldDims;
    } else if (oldDims.size() == 4) {                                             // 4: 判断原张量的维数
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)}; // 2, 3: 设置新张量形状
    } else {
        ATB_LOG(ERROR) << "SelfAttention intensor qLayer | kLayer | vLayer dimNum need 2 or 4";
        newDims.clear();
    }
}

void FlashAttentionInferShapePreFunc(Mki::LaunchParam &launchParam)
{
    launchParam.GetInTensor(3).desc.dtype = Mki::TENSOR_DTYPE_UINT32; // 3: 设置第四个输入张量的dtype
}

SelfAttentionFusionOpsRunner::SelfAttentionFusionOpsRunner(const infer::SelfAttentionParam &param)
    : OpsRunner("SelfAttentionFusionOpsRunner", RUNNER_TYPE_SELF_ATTENTION), param_(param)
{
    needKernelGraphModify_ = true;
    skipSetUpKernelGraphWhenCacheHit_ = false;
    ATB_LOG(INFO) << "SelfAttentionFusionOpsRunner::SelfAttentionFusionOpsRunner called";

    bool needMask = param_.maskType != infer::SelfAttentionParam::MASK_TYPE_UNDEFINED &&
                    !(param_.calcType == infer::SelfAttentionParam::DECODER &&
                      param_.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_NORM);
    bool isMaskCompress = (param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
                           param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT);
    bool needLogN = (param_.scaleType == atb::infer::SelfAttentionParam::SCALE_TYPE_LOGN);
    std::size_t intensorSize = 8; // 8: 设置inTensor数
    if (needMask) {
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

    kernelGraph_.inTensors.resize(intensorSize);
    size_t tensorId = 0;
    Mki::Tensor &mixedQuery = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &mixedKey = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &mixedValue = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &cacheK = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &cacheV = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor *attentionMask = needMask ? &kernelGraph_.inTensors.at(tensorId++) : &nullTensor_;
    Mki::Tensor &tokenOffset = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &seqLen = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &layerId = kernelGraph_.inTensors.at(tensorId++);
    if (param_.batchRunStatusEnable) {
        tensorId++;
    }
    Mki::Tensor *slopes = isMaskCompress ? &kernelGraph_.inTensors.at(tensorId++) : &nullTensor_;
    Mki::Tensor *logN = needLogN ? &kernelGraph_.inTensors.at(tensorId++) : &nullTensor_;

    ATB_LOG(INFO) << GetLogPrefix() << "tokenOffset dataSize:" << tokenOffset.dataSize;
    ATB_LOG(INFO) << GetLogPrefix() << "seqLen dataSize:" << seqLen.dataSize;

    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &context = kernelGraph_.outTensors.at(0);
    kernelGraph_.internalTensors.resize(1);
    Mki::Tensor &divOut = kernelGraph_.internalTensors.at(0);
    kernelGraph_.nodes.resize(4); // 4: 设置总节点数
    size_t nodeId = 0;

    auto &kCacheNode = kernelGraph_.nodes.at(nodeId++);
    AtbOps::OpParam::KVCache kCacheParam;
    kCacheParam.type = param_.batchRunStatusEnable ? AtbOps::OpParam::KVCache::KVCACHE_DYNAMIC_BATCH :
                                                     AtbOps::OpParam::KVCache::KVCACHE_ND;
    kCacheNode.opDesc = {0, "KVCacheOperation", kCacheParam};
    kCacheNode.inTensors = {&mixedKey, &layerId, &cacheK};
    kCacheNode.outTensors = {&cacheK};
    kCacheNode.inTensorViewFuncs.resize(kCacheNode.inTensors.size());
    kCacheNode.inTensorViewFuncs[0] = &SelfAttentionFusionViewFunc;

    auto &vCacheNode = kernelGraph_.nodes.at(nodeId++);
    AtbOps::OpParam::KVCache vCacheParam;
    vCacheParam.type = param_.batchRunStatusEnable ? AtbOps::OpParam::KVCache::KVCACHE_DYNAMIC_BATCH :
                                                     AtbOps::OpParam::KVCache::KVCACHE_ND;
    vCacheNode.opDesc = {0, "KVCacheOperation", vCacheParam};
    vCacheNode.inTensors = {&mixedValue, &layerId, &cacheV};
    vCacheNode.outTensors = {&cacheV};
    vCacheNode.inTensorViewFuncs.resize(vCacheNode.inTensors.size());
    vCacheNode.inTensorViewFuncs[0] = &SelfAttentionFusionViewFunc;
    // muls
    auto &mulsQNode = kernelGraph_.nodes.at(nodeId++);
    mulsQNode.opDesc = {0, "ElewiseOperation",
                        AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, param_.qScale})};
    mulsQNode.inTensors = {&mixedQuery};
    mulsQNode.outTensors = {&divOut};

    // mix.h mixType=UNPAD_FLASH_ATTENTION
    auto &flashAttentionNode = kernelGraph_.nodes.at(nodeId++);
    AtbOps::OpParam::UnpadFlashAttention flashAttentionQParam;
    SetFAParam(flashAttentionQParam);

    flashAttentionNode.opDesc = {0, "UnpadFlashAttentionOperation", flashAttentionQParam};
    ATB_LOG(INFO) << "flashAttentionQParam.type: " << flashAttentionQParam.type;
    flashAttentionNode.inTensors = {&divOut,      &cacheK,      &cacheV,      &layerId,     attentionMask, slopes,
                                    &nullTensor_, &nullTensor_, &nullTensor_, &nullTensor_, &nullTensor_,  logN};
    flashAttentionNode.outTensors = {&context};
    flashAttentionNode.inTensorViewFuncs.resize(flashAttentionNode.inTensors.size());
    flashAttentionNode.inTensorViewFuncs[0] = &SelfAttentionFusionViewFunc;
    flashAttentionNode.inferShapePreFunc = &FlashAttentionInferShapePreFunc;

    newParam_.tokenOffset.reserve(128);    // 128::预留大小
    newParam_.seqLen.reserve(128);         // 128: 预留大小
    newParam_.batchRunStatus.reserve(128); // 128:预留大小
}

Status SelfAttentionFusionOpsRunner::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    bool needMask = param_.maskType != infer::SelfAttentionParam::MASK_TYPE_UNDEFINED &&
                    !(param_.calcType == infer::SelfAttentionParam::DECODER &&
                      param_.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_NORM);
    bool hasKV = (param_.kvcacheCfg != atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS);
    bool ret = newParam_.BuildFromTensor(opsTensorPack.inTensors, param_.batchRunStatusEnable, needMask, hasKV);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " build param from host tensor fail";
        return ERROR_INVALID_PARAM;
    }
    size_t tokenOffsetTensorIdBase = 5; // 6, 5: tokenOffset id 4, 3: Kv0 tokenOffset id
    const size_t tokenOffsetTensorId = needMask ? tokenOffsetTensorIdBase + 1 : tokenOffsetTensorIdBase;
    ret = ModifyKVCacheNode(tokenOffsetTensorId);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " set kvcache param fail";
        return ERROR_INVALID_TENSOR_DIM;
    }
    auto &flashAttentionNode = kernelGraph_.nodes.at(3); // 3: flashAttention节点位置
    AtbOps::OpParam::UnpadFlashAttention flashAttentionQParam;
    SetFAParam(flashAttentionQParam);

    flashAttentionQParam.qSeqLen = newParam_.seqLen;
    flashAttentionQParam.kvSeqLen = newParam_.tokenOffset;
    flashAttentionQParam.batchRunStatus = newParam_.batchRunStatus;

    uint32_t hiddenSizePos = kernelGraph_.inTensors.at(VALUE_TENSOR_POS).desc.dims.size() - 1;
    int32_t hiddenSizeV = static_cast<int32_t>(kernelGraph_.inTensors.at(VALUE_TENSOR_POS).desc.dims[hiddenSizePos]);
    if (hiddenSizePos == 1) { // [nTokens, HiddenSize]
        int32_t headNumV =
            (flashAttentionQParam.kvHead > 0) ? flashAttentionQParam.kvHead : flashAttentionQParam.headSize;
        // flashAttentionQParam.headSize = param_.headNum, 一定大于0
        flashAttentionQParam.headDimV = hiddenSizeV / headNumV;
    } else { // [batch, seqlen, head_num, head_size] or [ntokens, head_num, head_size]
        flashAttentionQParam.headDimV = hiddenSizeV;
    }

    flashAttentionNode.opDesc = {0, "UnpadFlashAttentionOperation", flashAttentionQParam};

    ATB_LOG(INFO) << GetLogPrefix() << " update AtbOps::OpParam::UnpadFlashAttention.headNum:" << param_.headNum
                  << ", qkScale:" << param_.qkScale << ", kvHead:" << param_.kvHeadNum
                  << ", calcType: " << param_.calcType << ", maskType: " << param_.maskType
                  << ", kernelTypeType: " << param_.kernelType << ", windowSize: " << param_.windowSize;
    return NO_ERROR;
}

bool SelfAttentionFusionOpsRunner::ModifyKVCacheNode(const size_t tokenOffsetTensorId)
{
    size_t tensorId = 1;
    Mki::Tensor &mixedKey = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &mixedValue = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &cacheK = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &cacheV = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &tokenOffset = kernelGraph_.inTensors.at(tokenOffsetTensorId);
    Mki::Tensor &seqLen = kernelGraph_.inTensors.at(tokenOffsetTensorId + 1);
    Mki::Tensor &layerId = kernelGraph_.inTensors.at(tokenOffsetTensorId + 2); // tokenOffsetTensorId+2: layerId

    bool kvcacheWithParam = tokenOffset.data == nullptr || seqLen.data == nullptr;

    auto &kCacheNode = kernelGraph_.nodes.at(0); // 0: KCache节点位置
    AtbOps::OpParam::KVCache kCacheParam;
    SetKVCacheParam(kCacheParam, kvcacheWithParam);
    kCacheNode.inTensors = {&mixedKey, &layerId, &cacheK};
    if (!kvcacheWithParam) {
        kCacheNode.inTensors.push_back(&tokenOffset);
        kCacheNode.inTensors.push_back(&seqLen);
    }
    kCacheNode.opDesc = {0, "KVCacheOperation", kCacheParam};

    auto &vCacheNode = kernelGraph_.nodes.at(1); // 1: VCache节点位置
    AtbOps::OpParam::KVCache vCacheParam;
    SetKVCacheParam(vCacheParam, kvcacheWithParam);
    vCacheNode.inTensors = {&mixedValue, &layerId, &cacheV};
    if (!kvcacheWithParam) {
        vCacheNode.inTensors.push_back(&tokenOffset);
        vCacheNode.inTensors.push_back(&seqLen);
    }
    vCacheNode.opDesc = {0, "KVCacheOperation", vCacheParam};
    ATB_LOG(INFO) << GetLogPrefix() << " update AtbOps::OpParam::KVCache.type:" << vCacheParam.type;
    return true;
}

void SelfAttentionFusionOpsRunner::SetKVCacheParam(AtbOps::OpParam::KVCache &kCacheParam, bool kvcacheWithParam)
{
    kCacheParam.batchRunStatus = newParam_.batchRunStatus;
    if (kvcacheWithParam) {
        kCacheParam.type = param_.batchRunStatusEnable ? AtbOps::OpParam::KVCache::KVCACHE_DYNAMIC_BATCH_PARAMS :
                                                         AtbOps::OpParam::KVCache::KVCACHE_ND_PARAMS;
        kCacheParam.seqLen = newParam_.seqLen;
        kCacheParam.tokenOffset = newParam_.tokenOffset;
    } else {
        kCacheParam.type = param_.batchRunStatusEnable ? AtbOps::OpParam::KVCache::KVCACHE_DYNAMIC_BATCH :
                                                         AtbOps::OpParam::KVCache::KVCACHE_ND;
    }
}

AtbOps::OpParam::UnpadFlashAttention::Type SelfAttentionFusionOpsRunner::GetFaType()
{
    AtbOps::OpParam::UnpadFlashAttention::Type faType =
        (param_.calcType == atb::infer::SelfAttentionParam::DECODER) ?
            AtbOps::OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_DECODER_ND :
            (param_.kernelType == atb::infer::SelfAttentionParam::KERNELTYPE_HIGH_PRECISION ?
                 AtbOps::OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_FP32_ND :
                 AtbOps::OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ND);
    faType = param_.batchRunStatusEnable ?
                 (param_.calcType == infer::SelfAttentionParam::DECODER ?
                      AtbOps::OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION_DECODER :
                      AtbOps::OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION) :
                 faType;
    return faType;
}

void SelfAttentionFusionOpsRunner::SetFAParam(AtbOps::OpParam::UnpadFlashAttention &flashAttentionParam)
{
    flashAttentionParam.type = GetFaType();

    flashAttentionParam.isClamp = (param_.clampType == infer::SelfAttentionParam::CLAMP_TYPE_MIN_MAX);
    flashAttentionParam.clampMin = param_.clampMin;
    flashAttentionParam.clampMax = param_.clampMax;

    flashAttentionParam.headSize = param_.headNum;
    flashAttentionParam.kvHead = param_.kvHeadNum;
    flashAttentionParam.tor = param_.qkScale;
    flashAttentionParam.isTriuMask = param_.isTriuMask;

    flashAttentionParam.windowSize = param_.windowSize;
    flashAttentionParam.cacheType = static_cast<AtbOps::OpParam::UnpadFlashAttention::CacheType>(param_.cacheType);

    flashAttentionParam.maskType =
        param_.maskType == infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS ?
            static_cast<AtbOps::OpParam::UnpadFlashAttention::MaskType>(1) : // 1: norm mask
            ((param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
              param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT) ?
                 static_cast<AtbOps::OpParam::UnpadFlashAttention::MaskType>(2) : // 2: alibi mask
                 static_cast<AtbOps::OpParam::UnpadFlashAttention::MaskType>(param_.maskType));
    if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_NORM) {
        flashAttentionParam.maskType = AtbOps::OpParam::UnpadFlashAttention::MaskType::MASK_TYPE_SWA_NORM;
    } else if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_COMPRESS) {
        flashAttentionParam.maskType = AtbOps::OpParam::UnpadFlashAttention::MaskType::MASK_TYPE_SWA_COMPRESS;
    }
    flashAttentionParam.isAlibiMaskSqrt = (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT);
    flashAttentionParam.scaleType = param_.scaleType == atb::infer::SelfAttentionParam::SCALE_TYPE_LOGN ?
                                        AtbOps::OpParam::UnpadFlashAttention::SCALE_LOGN_FP32 :
                                        AtbOps::OpParam::UnpadFlashAttention::SCALE_TOR;
}

SelfAttentionFusionOpsRunner::~SelfAttentionFusionOpsRunner() {}
} // namespace atb