/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_ACL_KERNEL_H
#define ATB_ACL_KERNEL_H
#include <acl/acl.h>
#include "atb/types.h"
#include "atb/svector.h"

namespace atb {
class AclKernel {
public:
    AclKernel();
    ~AclKernel();
    aclError Run(const std::string &kernelName, const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors,
                 aclrtStream stream);

private:
    aclError AclKernelRun();
    Status Reset();

private:
    std::string opType_ = "";
    SVector<aclTensorDesc *> inTensorsDesc_;
    SVector<aclTensorDesc *> outTensorsDesc_;
    SVector<aclDataBuffer *> inTensorsBuffer_;
    SVector<aclDataBuffer *> outTensorsBuffer_;
    aclopAttr *opAttr_ = nullptr;
    aclrtStream stream_ = nullptr;
};
} // namespace atb
#endif