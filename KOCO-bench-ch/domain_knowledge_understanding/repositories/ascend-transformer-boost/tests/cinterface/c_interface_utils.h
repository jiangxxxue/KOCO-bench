/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef C_INTERFACE_UTILS_H
#define C_INTERFACE_UTILS_H
#include <gtest/gtest.h>
#include "atb/atb_acl.h"
namespace atb {
namespace cinterfaceTest {
int64_t GetTensorSize(const aclTensor *input);
aclnnStatus Init(atb::Context **context, aclrtStream *stream, int64_t *deviceId);
aclnnStatus Destroy(atb::Context **context, aclrtStream *stream);
aclnnStatus CreateInOutData(size_t num, uint8_t **inoutHost, uint8_t **inoutDevice, size_t *inoutSize);
void CreateACLTensorInOut(const std::vector<int64_t> dims, aclDataType type, aclFormat format, aclTensor **list,
                          size_t &i, void *inout);
} // namespace cinterfaceTest
} // namespace atb
#endif
