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
bool MultiLatentAttentionVariantPackParam::BuildFromTensor(const SVector<Mki::Tensor> &inTensors,
                                                           size_t contextLensTensorId, size_t qSeqlenTensorId,
                                                           bool needQLens)
{
    contextLens.clear();
    const Mki::Tensor &contextLensTensor = inTensors.at(contextLensTensorId);
    if (!contextLensTensor.hostData) {
        ATB_LOG(ERROR) << "contextLensTensor.hostData is null";
        return false;
    }
    contextLens.resize(contextLensTensor.Numel());
    int32_t *contextLensTensorHostData = (int32_t *)contextLensTensor.hostData;
    for (size_t i = 0; i < contextLens.size(); ++i) {
        contextLens[i] = contextLensTensorHostData[i];
    }
    qSeqlen.clear();
    if (needQLens) {
        const Mki::Tensor &qSeqlenTensor = inTensors.at(qSeqlenTensorId);
        if (!qSeqlenTensor.hostData) {
            ATB_LOG(ERROR) << "qSeqlenTensor.hostData is null";
            return false;
        }
        qSeqlen.resize(qSeqlenTensor.Numel());
        int32_t *qSeqlenTensorHostData = (int32_t *)qSeqlenTensor.hostData;
        for (size_t i = 0; i < qSeqlen.size(); ++i) {
            qSeqlen[i] = qSeqlenTensorHostData[i];
        }
    }
    return true;
}
} // namespace atb