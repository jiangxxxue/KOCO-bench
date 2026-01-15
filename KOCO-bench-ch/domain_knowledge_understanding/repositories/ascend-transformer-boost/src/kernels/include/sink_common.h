/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SINK_COMMON_H_
#define SINK_COMMON_H_

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <mki/utils/any/any.h>
#include <mki/bin_handle.h>
#include <mki/kernel_info.h>
#include <mki/launch_param.h>
#include <mki/utils/status/status.h>

namespace AsdOps {
    enum AttrType : int {
    UNDEFINED_TYPE = -1,
    BASIC_TYPE = 0,
    SVECTOR_TYPE,
    VECTOR_TYPE,
};

constexpr int32_t SHIFT_BITS = 32;

template<typename T>
size_t GetOffset(size_t i, uint64_t &type);

template<typename T>
const uint8_t *GetMkiSpecificAttr(void *attrs, size_t index, uint64_t &type)
{
    const Mki::Any *params = reinterpret_cast<const Mki::Any *>(attrs);
    const uint8_t *base = reinterpret_cast<const uint8_t *>(&Mki::AnyCast<T>(*params));
    size_t offset = GetOffset<T>(index, type);
    return base + offset;
}
}

namespace ops {
using GetAttrAdditional = const uint8_t *(*)(void *, size_t, uint64_t &);
}   // namespace ops

// InferShape utils
namespace opInferShape {
Mki::Status CallGeInferShape(const char *opType, const Mki::LaunchParam &launchParam,
                             Mki::SVector<Mki::Tensor> &outTensors, ops::GetAttrAdditional func);
}   // namespace opInferShape

// Tiling utils
namespace optiling {
Mki::Status CallGeTiling(const char *opType, const Mki::BinHandle &binHandle, const Mki::LaunchParam &launchParam,
                         ops::GetAttrAdditional getAttrFunc, Mki::KernelInfo &kernelInfo);
}   // namespace optiling

#endif  // SINK_COMMON_H_