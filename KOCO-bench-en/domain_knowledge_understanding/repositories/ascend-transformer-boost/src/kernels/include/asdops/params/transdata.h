/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_PARAMS_TRANSDATA_H
#define ASDOPS_PARAMS_TRANSDATA_H

#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>

namespace AsdOps {
namespace OpParam {
struct Transdata {
    enum TransdataType { UNDEFINED = 0, FRACTAL_NZ_TO_ND, ND_TO_FRACTAL_NZ };
    TransdataType transdataType = UNDEFINED;
    Mki::SVector<int64_t> outCrops = {0, 0};

    bool operator==(const Transdata &other) const
    {
        return this->transdataType == other.transdataType && this->outCrops == other.outCrops;
    }
};
} // namespace OpParam
} // namespace AsdOps

#endif // ASDOPS_PARAMS_TRANSDATA_H