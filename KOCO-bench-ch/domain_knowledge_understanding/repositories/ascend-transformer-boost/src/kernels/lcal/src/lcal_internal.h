/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCAL_INTERNAL_H
#define LCAL_INTERNAL_H

#include <string>
#include <acl/acl_base.h>
#include "lcal_types.h"
#include "coc_kernel_args.h"
#include "ccl_kernel_args.h"

namespace Lcal {

int RegistKernel(const int32_t opGroup = 0);

int64_t Count2Size(int64_t count, const HcclDataType &dataType);

int LoadMTE(LcalType cclType, AscendCCLKernelArgs &args, uint32_t blockDim, HcclDataType dataType, aclrtStream stream);

int LoadMTE(LcalType cclType, CCLGatherArgs &args, uint32_t blockDim, HcclDataType dataType, aclrtStream stream);

int ComputeOverComm(LcalType cocType, CoCKernelArgs kernelArgs, HcclDataType dataType, aclrtStream stream);
}

#endif