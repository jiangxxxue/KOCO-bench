/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <asdops/params/params.h>
#include "self_attention_encoder_fusion_ops_runner_910a.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/log.h"
#include "param.h"
#include "self_attention_runner_utils.h"


namespace atb {
void TransQKVEncoderViewFunc910a(const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims)
{
    if (oldDims.size() != 3) { // 3: q, k, v 必须三维
        ATB_LOG(ERROR) << "The dimNum of q, k, v should all be 3";
        newDims.clear();
        return;
    }
    newDims = {1, oldDims.at(0), oldDims.at(1) * oldDims.at(2)};
}

SelfAttentionEncoderFusionOpsRunner910A::SelfAttentionEncoderFusionOpsRunner910A(const infer::SelfAttentionParam &param)
    : OpsRunner("SelfAttentionEncoderFusionOpsRunner910A", RUNNER_TYPE_SELF_ATTENTION_PA_ENCODER), param_(param)
{
    needKernelGraphModify_ = true;
    ATB_LOG(INFO) << "SelfAttentionEncoderFusionOpsRunner910A::SelfAttentionEncoderFusionOpsRunner910A called";
}

Status SelfAttentionEncoderFusionOpsRunner910A::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    bool needMask = (param_.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED);
    bool isMaskCompress = (param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
                           param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT);
    bool needLogN = (param_.scaleType == atb::infer::SelfAttentionParam::SCALE_TYPE_LOGN);
    std::size_t intensorSize = needMask ? (isMaskCompress ? 6 : 5) : 4; // 6, 5, 4: flash encoder input num
    if (needLogN) {
        intensorSize++;
    }
    kernelGraph_.inTensors.resize(intensorSize);
    kernelGraph_.outTensors.resize(1);
    int inTensorStart = 0;
    Mki::Tensor &query = kernelGraph_.inTensors.at(inTensorStart++); // shape (1, n_tokens, hiddensize)
    Mki::Tensor &key = kernelGraph_.inTensors.at(inTensorStart++);   // (1, n_tokens, hiddensize)
    Mki::Tensor &value = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *mask = needMask ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    Mki::Tensor &seqLen = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *layerId = &nullTensor_;
    Mki::Tensor *slopes = isMaskCompress ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    Mki::Tensor *logN = needLogN ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;

    Mki::Tensor &output = kernelGraph_.outTensors.at(0);

    ATB_LOG(INFO) << GetLogPrefix() << "seqLen dataSize:" << seqLen.dataSize;

    auto attnMaskFormat = needMask ? opsTensorPack.inTensors.at(3).desc.format :
                                     static_cast<Mki::TensorFormat>(ACL_FORMAT_UNDEFINED); // 3: attnMask tensor

    newParam_.attnMaskFormat = static_cast<aclFormat>(attnMaskFormat);
    kernelGraph_.internalTensors.resize(
        (attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_FRACTAL_NZ) || (!needMask)) ?
            4 : // 4: attn mask 为nz或不需要模型传mask时
            5); // 5: attn mask 为nd时
    size_t internalTensorId = 0;
    Mki::Tensor &queryNz = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &keyNz = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &valueNz = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor *attnMaskNz = (attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND) && needMask) ?
                                  &kernelGraph_.internalTensors.at(internalTensorId++) :
                                  mask;
    Mki::Tensor &outNz = kernelGraph_.internalTensors.at(internalTensorId++);
    kernelGraph_.nodes.resize((attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_FRACTAL_NZ) || (!needMask)) ?
                                  5 : // 5: attn mask 为nz或不需要mask时
                                  6); // 6: attn mask 为nd时

    size_t nodeId = 0;
    auto &qTransdataNode = kernelGraph_.nodes.at(nodeId++);
    qTransdataNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    qTransdataNode.inTensors = {&query};
    qTransdataNode.outTensors = {&queryNz};
    qTransdataNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        if (launchParam.GetInTensor(0).desc.dims.size() < 3) { // 3: q必须至少三维
            ATB_LOG(ERROR) << "expect intensor dimNum to be at least 3, but got: "
                           << launchParam.GetInTensor(0).desc.dims.size();
            return;
        }
        ntokens_ = launchParam.GetInTensor(0).desc.dims.at(1);
        hiddenSize_ = launchParam.GetInTensor(0).desc.dims.at(2); // 2: 第三维
    };
    qTransdataNode.inTensorViewFuncs.resize(qTransdataNode.inTensors.size());
    qTransdataNode.inTensorViewFuncs[0] = &TransQKVEncoderViewFunc910a;

    auto &kTransdataNode = kernelGraph_.nodes.at(nodeId++);
    kTransdataNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    kTransdataNode.inTensors = {&key};
    kTransdataNode.outTensors = {&keyNz};
    kTransdataNode.inTensorViewFuncs.resize(kTransdataNode.inTensors.size());
    kTransdataNode.inTensorViewFuncs[0] = &TransQKVEncoderViewFunc910a;

    auto &vTransdataNode = kernelGraph_.nodes.at(nodeId++);

    vTransdataNode.opDesc = {0, "TransdataOperation",
                             AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
    vTransdataNode.inTensors = {&value};
    vTransdataNode.outTensors = {&valueNz};
    vTransdataNode.inTensorViewFuncs.resize(vTransdataNode.inTensors.size());
    vTransdataNode.inTensorViewFuncs[0] = &TransQKVEncoderViewFunc910a;

    // if attn mask is nd and is needed, do transdata
    if (attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND) && needMask) {
        ATB_LOG(INFO) << "attnMaskTransdataNode IN " << nodeId;
        auto &attnMaskTransdataNode = kernelGraph_.nodes.at(nodeId++);
        attnMaskTransdataNode.opDesc = {
            0, "TransdataOperation",
            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, {0, 0}})};
        attnMaskTransdataNode.inTensors = {mask};
        attnMaskTransdataNode.outTensors = {attnMaskNz};
    }
    auto &flashAttentionEncoderNode = kernelGraph_.nodes.at(nodeId++);

    AtbOps::OpParam::UnpadFlashAttentionNz unpadFlashAttentionNzParam;

    SetFAParam(unpadFlashAttentionNzParam);
    flashAttentionEncoderNode.opDesc = {0, "UnpadFlashAttentionNzOperation", unpadFlashAttentionNzParam};

    if (attnMaskFormat == static_cast<Mki::TensorFormat>(ACL_FORMAT_ND) && needMask) {
        flashAttentionEncoderNode.inTensors = {&queryNz, &keyNz, &valueNz, layerId, attnMaskNz, slopes, logN};
    } else {
        flashAttentionEncoderNode.inTensors = {&queryNz, &keyNz, &valueNz, layerId, mask, slopes, logN};
    }
    flashAttentionEncoderNode.outTensors = {&outNz};

    auto &transdataOutNode = kernelGraph_.nodes.at(nodeId++);
    transdataOutNode.opDesc = {
        0, "TransdataOperation",
        AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {ntokens_, hiddenSize_}})};
    transdataOutNode.inTensors = {&outNz};
    transdataOutNode.outTensors = {&output};
    transdataOutNode.inferShapePreFunc = [=](Mki::LaunchParam &launchParam) {
        launchParam.SetParam(
            AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, {ntokens_, hiddenSize_}}));
    };
    return NO_ERROR;
}

bool SelfAttentionEncoderFusionOpsRunner910A::NeedModifySlopes(const OpsTensorPack &opsTensorPack)
{
    bool isMaskCompress = param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
                          param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT;
    if (!isMaskCompress) {
        return false;
    } else {
        return opsTensorPack.inTensors.at(3).desc.dims.size() == 4 && // 3: maskId, 4: 非 [1,256//16,256,16] 的情况
               (opsTensorPack.inTensors.at(3).desc.dims[0] != 1 ||    // 3: maskId
                opsTensorPack.inTensors.at(3).desc.dims[1] != 16 ||   // 3: maskId, 16: mask shape
                opsTensorPack.inTensors.at(3).desc.dims[2] != 256);   // 3: maskId, 256: mask shape
    }
}

Status SelfAttentionEncoderFusionOpsRunner910A::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    const size_t seqLenId = param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED ? 3 : 4;
    bool ret = newParam_.BuildFromTensorEncoder(opsTensorPack.inTensors, seqLenId);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " build param from host tensor fail";
        return ERROR_INVALID_PARAM;
    }
    auto &flashAttentionNode =
        kernelGraph_.nodes.at((newParam_.attnMaskFormat == ACL_FORMAT_ND &&
                               param_.maskType != infer::SelfAttentionParam::MASK_TYPE_UNDEFINED) ?
                                  4 :
                                  3); // 4, 3: flashAttentionEncoder节点位置
    AtbOps::OpParam::UnpadFlashAttentionNz faEncoderNzCacheParam;
    SetFAParam(faEncoderNzCacheParam);
    faEncoderNzCacheParam.qSeqLen = newParam_.seqLen;
    faEncoderNzCacheParam.kvSeqLen = newParam_.tokenOffset;
    faEncoderNzCacheParam.batchRunStatus = {};

    ATB_LOG(INFO) << GetLogPrefix() << "type: " << faEncoderNzCacheParam.type;
    flashAttentionNode.opDesc = {0, "UnpadFlashAttentionNzOperation", faEncoderNzCacheParam};
    if (NeedModifySlopes(opsTensorPack)) {
        flashAttentionNode.inTensors.at(5) = &nullTensor_; // 5: slopes
    }
    ATB_LOG(INFO) << GetLogPrefix() << " update AsdOps::OpParam::UnpadFlashAttentionNz.headNum:" << param_.headNum
                  << ", seqLen.size:" << newParam_.seqLen.size()
                  << ", tokenOffset.size:" << newParam_.tokenOffset.size() << ", qkScale:" << param_.qkScale
                  << ", kvHead:" << param_.kvHeadNum;
    return NO_ERROR;
}

void SelfAttentionEncoderFusionOpsRunner910A::SetFAParam(AtbOps::OpParam::UnpadFlashAttentionNz &flashAttentionParam)
{
    flashAttentionParam.type = AtbOps::OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_ENCODER_NOCACHE;
    flashAttentionParam.headSize = param_.headNum;
    flashAttentionParam.tor = param_.qkScale;
    flashAttentionParam.kvHead = param_.kvHeadNum;
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
    flashAttentionParam.alibiLeftAlign =
        (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN);
    flashAttentionParam.isTriuMask = param_.isTriuMask;
    flashAttentionParam.scaleType = param_.scaleType == atb::infer::SelfAttentionParam::SCALE_TYPE_LOGN ?
                                        AtbOps::OpParam::UnpadFlashAttentionNz::SCALE_LOGN :
                                        AtbOps::OpParam::UnpadFlashAttentionNz::SCALE_TOR;
}

void SelfAttentionEncoderFusionOpsRunner910A::SetParam(const Mki::Any &param)
{
    infer::SelfAttentionParam newParam = Mki::AnyCast<infer::SelfAttentionParam>(param);
    if (!IsParamEqual(newParam, param_)) {
        param_ = newParam;
        isParamUpdated_ = true;
    }
}

SelfAttentionEncoderFusionOpsRunner910A::~SelfAttentionEncoderFusionOpsRunner910A() {}
} // namespace atb
