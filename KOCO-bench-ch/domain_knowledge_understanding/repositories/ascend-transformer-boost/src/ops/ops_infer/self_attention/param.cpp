/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "param.h"
#include "atb/utils.h"
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"

namespace atb {
bool SelfAttentionFusionVariantPackParam::BuildFromTensor(const SVector<Mki::Tensor> &inTensors, bool hasBatch,
                                                          bool hasMask, bool hasKV)
{
    size_t tokenOffsetTensorIdBase = 3;
    if (hasKV) {
        tokenOffsetTensorIdBase += 2; // 2: hasKV
    }
    // 6, 5: tokenOffset id 4, 3: Kv0 tokenOffset id
    const size_t tokenOffsetTensorId = hasMask ? tokenOffsetTensorIdBase + 1 : tokenOffsetTensorIdBase;
    const Mki::Tensor &tokenOffsetTensor = inTensors.at(tokenOffsetTensorId);
    size_t seqLenTensorIdBase = 4;
    if (hasKV) {
        seqLenTensorIdBase += 2; // 2: hasKV
    } else {
        SetKVCacheParam(inTensors);
    }
    // 7, 6: seqLen id  5, 4: Kv0 seqLen id
    const size_t seqLenTensorId = hasMask ? seqLenTensorIdBase + 1 : seqLenTensorIdBase;
    const Mki::Tensor &seqLenTensor = inTensors.at(seqLenTensorId);
    ATB_LOG(INFO) << "tensor.hostData is null, tokenOffsetTensor.hostData:" << tokenOffsetTensor.hostData
                  << ", seqLenTensor.hostData:" << seqLenTensor.hostData
                  << " ,tokenOffsetTensorId: " << tokenOffsetTensorId << ", seqLenTensorId: " << seqLenTensorId;
    if (!tokenOffsetTensor.hostData || !seqLenTensor.hostData) {
#ifdef _DEBUG
        ATB_LOG(ERROR) << "tensor.hostData is null, tokenOffsetTensor.hostData:" << tokenOffsetTensor.hostData
                       << ", seqLenTensor.hostData:" << seqLenTensor.hostData
                       << " ,tokenOffsetTensorId: " << tokenOffsetTensorId << ", seqLenTensorId: " << seqLenTensorId;
#else
        ATB_LOG(ERROR) << "tensor.hostData is null"
                       << " ,tokenOffsetTensorId: " << tokenOffsetTensorId << " , seqLenTensorId: " << seqLenTensorId;
#endif
        return false;
    }

    SetTokenOffsetParam(tokenOffsetTensor);

    seqLen.resize(seqLenTensor.Numel());
    int32_t *seqLenTensorHostData = (int32_t *)seqLenTensor.hostData;
    for (size_t i = 0; i < seqLen.size(); ++i) {
        seqLen[i] = seqLenTensorHostData[i];
    }

    if (hasBatch) {
        const Mki::Tensor &batchRunStatusTensor = inTensors.at(hasMask ? 9 : 8); // 9, 8: batchStatus id
        batchRunStatus.resize(batchRunStatusTensor.Numel());
        if (!batchRunStatusTensor.hostData) {
            ATB_LOG(ERROR) << "batchRunStatusTensor inTensors(9) hostData is null";
            return false;
        }
        int32_t *batchRunStatusHostData = reinterpret_cast<int32_t *>(batchRunStatusTensor.hostData);
        for (size_t i = 0; i < batchRunStatus.size(); ++i) {
            batchRunStatus[i] = batchRunStatusHostData[i];
        }
    } else {
        batchRunStatus.clear();
    }

    return true;
}

bool SelfAttentionFusionVariantPackParam::BuildFromTensorEncoder(const SVector<Mki::Tensor> &inTensors,
                                                                 const size_t seqLenTensorId)
{
    const Mki::Tensor &seqLenTensor = inTensors.at(seqLenTensorId);
    if (!seqLenTensor.hostData) {
#ifdef _DEBUG
        ATB_LOG(ERROR) << "tensor.hostData is null, seqLenTensor.hostData:" << seqLenTensor.hostData;
#else
        ATB_LOG(ERROR) << "tensor.hostData is null";
#endif
        return false;
    }

    ATB_LOG(INFO) << "seqLenTensor len:" << seqLenTensor.Numel();
    // dims = [2, batch] or dims = [batch]
    uint32_t batch = seqLenTensor.desc.dims[0];
    uint32_t tokenOffsetIndexOffset = 0;
    if (seqLenTensor.desc.dims.size() > 1) {
        batch = seqLenTensor.desc.dims[1];
        tokenOffsetIndexOffset = batch;
    }
    tokenOffset.resize(batch);
    int32_t *tokenOffsetTensorHostData = (int32_t *)seqLenTensor.hostData;
    for (size_t i = 0; i < tokenOffset.size(); ++i) {
        tokenOffset[i] = tokenOffsetTensorHostData[i + tokenOffsetIndexOffset];
    }

    seqLen.resize(batch);
    int32_t *seqLenTensorHostData = (int32_t *)seqLenTensor.hostData;
    for (size_t i = 0; i < seqLen.size(); ++i) {
        seqLen[i] = seqLenTensorHostData[i];
    }
    return true;
}

bool SelfAttentionFusionVariantPackParam::BuildFromTensorPrefixEncoder(const SVector<Mki::Tensor> &inTensors,
                                                                       const size_t seqLenTensorId,
                                                                       const size_t kvSeqLenTensorId)
{
    const Mki::Tensor &seqLenTensor = inTensors.at(seqLenTensorId);
    if (!seqLenTensor.hostData) {
#ifdef _DEBUG
        ATB_LOG(DEBUG) << "tensor.hostData is null at " << seqLenTensorId
                       << ", seqLenTensor.hostData: " << seqLenTensor.hostData;
#else
        ATB_LOG(DEBUG) << "tensor.hostData is null";
#endif
        return false;
    }
    ATB_LOG(INFO) << "seqLenTensor len:" << seqLenTensor.Numel();
    seqLen.resize(seqLenTensor.Numel());
    int32_t *seqLenTensorHostData = (int32_t *)seqLenTensor.hostData;
    for (size_t i = 0; i < seqLen.size(); ++i) {
        seqLen[i] = seqLenTensorHostData[i];
    }

    const Mki::Tensor &kvSeqLenTensor = inTensors.at(kvSeqLenTensorId);
    if (!kvSeqLenTensor.hostData) {
#ifdef _DEBUG
        ATB_LOG(DEBUG) << "tensor.hostData is null at " << kvSeqLenTensorId
                       << ", kvSeqLenTensor.hostData: " << kvSeqLenTensor.hostData;
#else
        ATB_LOG(DEBUG) << "tensor.hostData is null";
#endif
        return false;
    }
    kvSeqLen.resize(kvSeqLenTensor.Numel());
    int32_t *kvSeqLenTensorHostData = (int32_t *)kvSeqLenTensor.hostData;
    for (size_t i = 0; i < kvSeqLen.size(); ++i) {
        kvSeqLen[i] = kvSeqLenTensorHostData[i];
    }
    return true;
}

void SelfAttentionFusionVariantPackParam::SetKVCacheParam(const SVector<Mki::Tensor> &inTensors)
{
    const Mki::Tensor &kCacheTensor = inTensors.at(1);
    const Mki::Tensor &vCacheTensor = inTensors.at(2);
    if (kCacheTensor.hostData != nullptr && kCacheTensor.data == nullptr && vCacheTensor.hostData != nullptr &&
        vCacheTensor.data == nullptr) {
        ATB_LOG(INFO) << "kCache and vCache use tensorList.";
        kCache = *reinterpret_cast<std::vector<atb::Tensor> *>(kCacheTensor.hostData);
        vCache = *reinterpret_cast<std::vector<atb::Tensor> *>(vCacheTensor.hostData);
        isUseTensorList = true;
    }
}

void SelfAttentionFusionVariantPackParam::SetTokenOffsetParam(const Mki::Tensor &tokenOffsetTensor)
{
    tokenOffset.resize(tokenOffsetTensor.Numel());
    int32_t *tokenOffsetTensorHostData = (int32_t *)tokenOffsetTensor.hostData;
    for (size_t i = 0; i < tokenOffset.size(); ++i) {
        tokenOffset[i] = tokenOffsetTensorHostData[i];
    }
}
} // namespace atb
