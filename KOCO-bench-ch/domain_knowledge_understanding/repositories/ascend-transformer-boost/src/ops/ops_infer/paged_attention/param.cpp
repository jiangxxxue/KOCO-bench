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
bool PagedAttentionFusionVariantPackParam::BuildFromTensor(const SVector<Mki::Tensor> &inTensors,
                                                           bool batchRunStatusEnable, size_t batchRunStatusIndex,
                                                           bool needQLens, size_t qLensIndex,
                                                           size_t contextLensTensorId)
{
    batchRunStatus.clear();
    contextLens.clear();
    qLens.clear();
    const Mki::Tensor &contextLensTensor = inTensors.at(contextLensTensorId);
    if (!contextLensTensor.hostData) {
#ifdef _DEBUG
        ATB_LOG(ERROR) << "tensor.hostData is null, contextLensTensor.hostData:" << contextLensTensor.hostData;
#else
        ATB_LOG(ERROR) << "tensor.hostData is null";
#endif
        return false;
    }
    contextLens.resize(contextLensTensor.Numel());
    int32_t *contextLensTensorHostData = (int32_t *)contextLensTensor.hostData;
    for (size_t i = 0; i < contextLens.size(); ++i) {
        contextLens[i] = contextLensTensorHostData[i];
    }

    if (batchRunStatusEnable) {
        const Mki::Tensor &batchRunStatusTensor = inTensors.at(batchRunStatusIndex);
        batchRunStatus.resize(batchRunStatusTensor.Numel());
        if (!batchRunStatusTensor.hostData) {
            ATB_LOG(ERROR) << "batchRunStatusTensor inTensors(" << batchRunStatusIndex << ") hostData is null";
            return false;
        }
        int32_t *batchRunStatusHostData = reinterpret_cast<int32_t *>(batchRunStatusTensor.hostData);
        for (size_t i = 0; i < batchRunStatus.size(); ++i) {
            batchRunStatus[i] = batchRunStatusHostData[i];
        }
    } else {
        batchRunStatus.clear();
    }
    if (needQLens) {
        if (!ParseQLens(inTensors, qLensIndex)) {
            return false;
        }
    } else {
        qLens.clear();
    }

    return true;
}

bool PagedAttentionFusionVariantPackParam::BuildFromTensor310P(const SVector<Mki::Tensor> &inTensors, bool needQLens,
                                                               int qLensIndex)
{
    batchRunStatus.clear();
    contextLens.clear();
    qLens.clear();
    if (needQLens) {
        if (!ParseQLens(inTensors, qLensIndex)) {
            return false;
        }
    } else {
        qLens.clear();
    }

    return true;
}

bool PagedAttentionFusionVariantPackParam::ParseQLens(const SVector<Mki::Tensor> &inTensors, int qLensIndex)
{
    const Mki::Tensor &qLensTensor = inTensors.at(qLensIndex);
    qLens.resize(qLensTensor.Numel());
    if (!qLensTensor.hostData) {
        ATB_LOG(ERROR) << "qLensTensor inTensors(" << qLensIndex << ") hostData is null";
        return false;
    }
    int32_t *qLensHostData = reinterpret_cast<int32_t *>(qLensTensor.hostData);
    for (size_t i = 0; i < qLens.size(); ++i) {
        qLens[i] = qLensHostData[i];
    }
    return true;
}
} // namespace atb