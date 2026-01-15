/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
bool RingMLAVariantPackParam::BuildFromTensor(const SVector<Mki::Tensor> &inTensors, const size_t seqLenTensorId)
{
    const Mki::Tensor &seqLenTensor = inTensors.at(seqLenTensorId);
    if (!seqLenTensor.hostData) {
#ifdef _DEBUG
        ATB_LOG(ERROR) << "seqLenTensor.hostData is null, seqLenTensor.hostData: " << seqLenTensor.hostData;
#else
        ATB_LOG(ERROR) << "seqLenTensor.hostData is null";
#endif
        return false;
    }

    ATB_LOG(INFO) << "Ring MLA seqLenTensor total size: " << seqLenTensor.Numel();
    // dims = [2, batch] or dims = [batch]
    uint32_t batch = seqLenTensor.desc.dims[0];
    int32_t *seqLenTensorHostData = (int32_t *)seqLenTensor.hostData;

    if (seqLenTensor.desc.dims.size() > 1) {
        batch = seqLenTensor.desc.dims[1];
        kvSeqLen.resize(batch);
        for (size_t i = 0; i < kvSeqLen.size(); ++i) {
            kvSeqLen[i] = seqLenTensorHostData[i + batch];
        }
    }

    qSeqLen.resize(batch);
    for (size_t i = 0; i < qSeqLen.size(); ++i) {
        qSeqLen[i] = seqLenTensorHostData[i];
    }
    if (batch == seqLenTensor.Numel()) {
        kvSeqLen = qSeqLen;
    }

    std::stringstream ss;
    if (qSeqLen.size() > 0) {
        ss << "qSeqLen: ";
        for (size_t i = 0; i < qSeqLen.size(); ++i) {
            ss << qSeqLen[i] << "\t";
        }
    }

    if (kvSeqLen.size() > 0) {
        ss << "kvSeqLen: ";
        for (size_t i = 0; i < qSeqLen.size(); ++i) {
            ss << kvSeqLen[i] << "\t";
        }
    }
    ATB_LOG(INFO) << "RIngMLA BuildFromTensor " << ss.str();
    return true;
}
} // namespace atb
