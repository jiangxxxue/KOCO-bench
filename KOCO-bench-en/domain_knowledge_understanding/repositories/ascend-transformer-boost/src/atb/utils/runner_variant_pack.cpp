/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/runner_variant_pack.h"
#include <sstream>
#include "atb/utils/tensor_util.h"
#include "atb/utils/log.h"

namespace atb {
std::string RunnerVariantPack::ToString() const
{
    if (isInTensorCanFree.size() != inTensors.size() || isOutTensorNeedMalloc.size() != outTensors.size()) {
        ATB_LOG(ERROR) << "RunnerVariantPack error! inTensor size: " << inTensors.size()
                       << ", isInTensorCanFree size: " << isInTensorCanFree.size()
                       << ", outTensor size: " << outTensors.size()
                       << ", isOutTensorNeedMalloc size: " << isOutTensorNeedMalloc.size();
        return "";
    }
    std::stringstream ss;
#ifdef _DEBUG
    ss << "tilingBuffer:" << static_cast<void *>(tilingBuffer)
       << ", workspaceBuffer:" << static_cast<void *>(workspaceBuffer)
       << ", intermediateBuffer:" << static_cast<void *>(intermediateBuffer);
#endif
    ss << " tilingBufferSize:" << tilingBufferSize << ", workspaceBufferSize:" << workspaceBufferSize
       << ", intermediateBufferSize:" << intermediateBufferSize << std::endl;

    for (size_t i = 0; i < inTensors.size(); ++i) {
        ss << " inTensors[" << i << "]: " << TensorUtil::TensorToString(inTensors.at(i))
           << " can free: " << isInTensorCanFree.at(i) << std::endl;
    }

    for (size_t i = 0; i < outTensors.size(); ++i) {
        ss << " outTensors[" << i << "]: " << TensorUtil::TensorToString(outTensors.at(i))
           << " need malloc: " << isOutTensorNeedMalloc.at(i) << std::endl;
    }

    return ss.str();
}
} // namespace atb