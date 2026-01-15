/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "c_interface_utils.h"
#include "acl/acl.h"
namespace atb {
namespace cinterfaceTest {
int64_t GetTensorSize(const aclTensor *input)
{
    const op::Shape shape = input->GetViewShape();
    const size_t dims = shape.GetDimNum();
    int64_t size = 1;
    for (size_t i = 0; i < dims; ++i) {
        size *= shape.GetDim(i);
    }
    return size;
}

aclnnStatus Init(atb::Context **context, aclrtStream *stream, int64_t *deviceId)
{
    aclInit(nullptr);
    aclrtSetDevice(*deviceId);
    aclrtCreateStream(stream);
    atb::CreateContext(context);
    (*context)->SetExecuteStream(*stream);
    return ACL_ERROR_NONE;
}

aclnnStatus Destroy(atb::Context **context, aclrtStream *stream)
{
    aclrtDestroyStream(*stream);
    DestroyContext(*context);
    aclFinalize();
    return ACL_ERROR_NONE;
}

aclnnStatus CreateInOutData(size_t num, uint8_t **inoutHost, uint8_t **inoutDevice, size_t *inoutSize)
{
    for (size_t i = 0; i < num; i++) {
        aclrtMallocHost((void **)(&(inoutHost[i])), inoutSize[i]);
        aclrtMalloc((void **)(&(inoutDevice[i])), inoutSize[i], ACL_MEM_MALLOC_HUGE_FIRST);
    }
    return ACL_ERROR_NONE;
}

void CreateACLTensorInOut(const std::vector<int64_t> dims, aclDataType type, aclFormat format, aclTensor **list,
                          size_t &i, void *inout)
{
    list[i++] = aclCreateTensor(dims.data(), dims.size(), type, nullptr, 0, format, nullptr, 0, inout);
}
} // namespace cinterfaceTest
} // namespace atb
