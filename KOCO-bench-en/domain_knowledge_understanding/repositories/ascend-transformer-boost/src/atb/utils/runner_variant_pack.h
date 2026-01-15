/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_RUNNER_VARIANT_PACK_H
#define ATB_RUNNER_VARIANT_PACK_H
#include <string>
#include <cstdint>
#include <mki/utils/any/any.h>
#include "atb/types.h"
#include "atb/svector.h"
#include "atb/context/context_base.h"
#include "atb/utils/mstx_mem_register.h"

namespace atb {
struct RunnerVariantPack {
    SVector<Tensor> inTensors;
    SVector<Tensor> outTensors;
    SVector<bool> isInTensorCanFree;
    SVector<bool> isOutTensorNeedMalloc;
    uint8_t *hostTilingBuffer = nullptr;
    uint8_t *tilingBuffer = nullptr;
    uint64_t tilingBufferSize = 0;
    uint8_t *workspaceBuffer = nullptr;
    uint64_t workspaceBufferSize = 0;
    uint8_t *intermediateBuffer = nullptr;
    uint64_t intermediateBufferSize = 0;
    uint8_t *argsDeviceBuffer = nullptr;
    uint8_t *argsHostBuffer = nullptr;
    ContextBase *context = nullptr;
    MstxMemRegister *mstxMemRegister = nullptr;
    std::string ToString() const;
};
} // namespace atb
#endif