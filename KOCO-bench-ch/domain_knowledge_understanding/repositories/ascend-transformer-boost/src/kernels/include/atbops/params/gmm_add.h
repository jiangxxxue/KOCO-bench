/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATBOPS_PARAMS_TBE_GROUPED_MATMUL_H
#define ATBOPS_PARAMS_TBE_GROUPED_MATMUL_H

namespace AtbOps {
namespace OpParam {
struct GmmAdd {
public:
    bool transposeB = false;
    bool transposeA = false;
    bool operator==(const GmmAdd &other) const
    {
        bool ret = transposeA == other.transposeA && transposeB == other.transposeB;
        return ret;
    }
};
} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_TBE_GROUPED_MATMUL_H