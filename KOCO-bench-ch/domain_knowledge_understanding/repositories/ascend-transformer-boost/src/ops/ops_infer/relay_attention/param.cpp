/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
bool RelayAttentionVariantPackParam::BuildFromTensor(const SVector<Mki::Tensor> &inTensors)
{
    if (!HostDataCheck(inTensors)) {
        return false;
    }
    const Mki::Tensor &keyTensor = inTensors.at(1);        // 1:keyTensor
    const Mki::Tensor &valueTensor = inTensors.at(2);      // 2:valueTensor
    const Mki::Tensor &keyShareTensor = inTensors.at(3);   // 3:keyShareTensor
    const Mki::Tensor &valueShareTensor = inTensors.at(4); // 4:valueShareTensor
    bool keyData = keyTensor.hostData != nullptr && keyTensor.data == nullptr;
    bool valueData = valueTensor.hostData != nullptr && valueTensor.data == nullptr;
    bool keyShareData = keyShareTensor.hostData != nullptr && keyShareTensor.data == nullptr;
    bool valueShareData = valueShareTensor.hostData != nullptr && valueShareTensor.data == nullptr;
    if (keyData && valueData && keyShareData && valueShareData) {
        key = *reinterpret_cast<std::vector<atb::Tensor> *>(keyTensor.hostData);
        value = *reinterpret_cast<std::vector<atb::Tensor> *>(valueTensor.hostData);
        keyShare = *reinterpret_cast<std::vector<atb::Tensor> *>(keyShareTensor.hostData);
        valueShare = *reinterpret_cast<std::vector<atb::Tensor> *>(valueShareTensor.hostData);
        ReintCastShapeFix(keyTensor, key);
        ReintCastShapeFix(valueTensor, value);
        ReintCastShapeFix(keyShareTensor, keyShare);
        ReintCastShapeFix(valueShareTensor, valueShare);
    }
    return true;
}

void RelayAttentionVariantPackParam::ReintCastShapeFix(const Mki::Tensor tensor, std::vector<atb::Tensor> &tensorList) const
{
    if (tensor.desc.dims.size() - 1 != tensorList[0].desc.shape.dimNum) {
        size_t diffDimNum = static_cast<size_t>(tensorList[0].desc.shape.dimNum);
        for (size_t i = 0; i < tensorList.size(); i++) {
            for (size_t j = diffDimNum; j < tensor.desc.dims.size() - 1; ++j) {
                tensorList[i].desc.shape.dims[j] = 1;
            }
            tensorList[i].desc.shape.dimNum = tensor.desc.dims.size() - 1;
        }
    }
}

bool RelayAttentionVariantPackParam::HostDataCheck(const SVector<Mki::Tensor> &inTensors)
{
    const Mki::Tensor &seqLenTensor = inTensors.at(6); // 6:seqLen
    if (!seqLenTensor.hostData) {
        ATB_LOG(ERROR) << "seqLenTensor hostData is null";
        return false;
    }
    seqLen.resize(seqLenTensor.Numel());
    int32_t *seqLenHostData = (int32_t *)seqLenTensor.hostData;
    for (size_t i = 0; i < seqLen.size(); ++i) {
        seqLen[i] = seqLenHostData[i];
    }
    const Mki::Tensor &kvSeqLenTensor = inTensors.at(7); // 7:kvSeqLen
    if (!kvSeqLenTensor.hostData) {
        ATB_LOG(ERROR) << "kvSeqLenTensor hostData is null";
        return false;
    }
    kvSeqLen.resize(kvSeqLenTensor.Numel());
    int32_t *kvSeqLenHostData = (int32_t *)kvSeqLenTensor.hostData;
    for (size_t i = 0; i < kvSeqLen.size(); ++i) {
        kvSeqLen[i] = kvSeqLenHostData[i];
    }
    const Mki::Tensor &kvShareMapTensor = inTensors.at(8); // 8:kvShareMap
    if (!kvShareMapTensor.hostData) {
        ATB_LOG(ERROR) << "kvShareMapTensor hostData is null";
        return false;
    }
    kvShareMap.resize(kvShareMapTensor.Numel());
    int32_t *kvShareMapHostData = (int32_t *)kvShareMapTensor.hostData;
    for (size_t i = 0; i < kvShareMap.size(); ++i) {
        kvShareMap[i] = kvShareMapHostData[i];
    }
    const Mki::Tensor &kvShareLenTensor = inTensors.at(9); // 9:kvShareLen
    if (!kvShareLenTensor.hostData) {
        ATB_LOG(ERROR) << "kvShareLenTensor hostData is null";
        return false;
    }
    kvShareLen.resize(kvShareLenTensor.Numel());
    int32_t *kvShareLenHostData = (int32_t *)kvShareLenTensor.hostData;
    for (size_t i = 0; i < kvShareLen.size(); ++i) {
        kvShareLen[i] = kvShareLenHostData[i];
    }
    return true;
}

} // namespace atb