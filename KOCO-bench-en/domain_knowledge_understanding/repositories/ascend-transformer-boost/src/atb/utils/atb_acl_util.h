/*

Copyright (c) 2025 Huawei Technologies Co., Ltd.
This file is a part of the CANN Open Software.
Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ATB_ACL_UTIL_H
#define ATB_ACL_UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

int64_t GetTensorSize(const aclTensor *input);

atb::Status aclTensorToAtbTensor(const aclTensor *aclTensorSrc, atb::Tensor *atbTensorDst);

atb::Status aclTensorToAtbTensorHost(const aclTensor *aclTensorSrc, atb::Tensor *atbTensorDst);
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // ATB_ACL_UTIL_H
