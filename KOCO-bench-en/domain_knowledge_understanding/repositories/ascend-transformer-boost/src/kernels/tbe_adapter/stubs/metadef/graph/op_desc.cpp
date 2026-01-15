/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "graph/op_desc.h"

#include "graph/ge_tensor.h"

#define UNUSED_VALUE(x) (void)(x)
namespace ge {
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t OpDesc::GetInputsSize() const { return 0; }

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t OpDesc::GetOutputsSize() const { return 0; }

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDescPtr OpDesc::MutableInputDesc(const uint32_t index) const
{
    UNUSED_VALUE(index);
    return nullptr;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDescPtr OpDesc::MutableOutputDesc(const uint32_t index) const
{
    UNUSED_VALUE(index);
    return nullptr;
}

int32_t OpDesc::GetInputIndexByName(const std::string &name) const
{
    UNUSED_VALUE(name);
    return 0;
}
} // namespace ge
