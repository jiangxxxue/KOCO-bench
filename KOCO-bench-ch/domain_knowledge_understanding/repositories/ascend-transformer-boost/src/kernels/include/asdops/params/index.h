/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_PARAMS_INDEX_H
#define ASDOPS_PARAMS_INDEX_H

#include <string>
#include <sstream>

namespace AsdOps {
namespace OpParam {
struct Index {
    enum IndexType {
        INDEX_UNDEFINED = 0,
        INDEX_ADD,
        INDEX_ADD_VALID,
    };
    IndexType indexType;
    int64_t axis = 0;

    bool operator==(const Index &other) const
    {
        return (this->indexType == other.indexType) && (this->axis == other.axis);
    };
};
} // namespace OpParam
} // namespace AsdOps

#endif