/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/tensor_check.h"
#include <acl/acl.h>
#include "atb/types.h"
#include "atb/utils.h"
#include "atb/utils/log.h"

namespace atb {
// 256GB
const uint64_t MAX_TENSOR_SIZE = 256uLL * 1024uLL * 1024uLL * 1024uLL;

Status TensorCheck::CheckTensorShape(const Tensor &tensor)
{
    return CheckTensorShape(tensor.desc);
}

Status TensorCheck::CheckTensorShape(const TensorDesc &tensorDesc)
{
    if (!TensorCheck::IsDimNumValid(tensorDesc.shape.dimNum)) {
        ATB_LOG(ERROR) << "tensor dimNum " << tensorDesc.shape.dimNum << " is invalid, should >0 && <= MAX_DIM(8)";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (!TensorCheck::IsTensorSizeValid(tensorDesc)) {
        return ERROR_INVALID_TENSOR_SIZE;
    }
    return NO_ERROR;
}

bool TensorCheck::IsTensorDType(const Tensor &tensor, const aclDataType &dType)
{
    return IsTensorDType(tensor.desc, dType);
}

bool TensorCheck::IsTensorFormat(const Tensor &tensor, const aclFormat &format)
{
    return IsTensorFormat(tensor.desc, format);
}

bool TensorCheck::IsTensorDescDimNumValid(const TensorDesc &tensorDesc, uint64_t dimNum)
{
    return tensorDesc.shape.dimNum == dimNum;
}

bool TensorCheck::IsTensorSizeValid(const Tensor &tensor)
{
    return IsTensorSizeValid(tensor.desc);
}

bool TensorCheck::IsTensorSizeValid(const TensorDesc &tensorDesc)
{
    uint64_t tensorSize = Utils::GetTensorSize(tensorDesc);
    if (tensorSize != 0 && tensorSize <= MAX_TENSOR_SIZE) {
        return true;
    }
    ATB_LOG(ERROR) << "tensor size: " << tensorSize << " is invalid, should > 0 & < 256g";
    return false;
}

bool TensorCheck::IsSupportDtype(const SVector<aclDataType> &dtypes, aclDataType dtype)
{
    for (auto d : dtypes) {
        if (d == dtype) {
            return true;
        }
    }
    return false;
}

Status TensorCheck::TensorDescsEqual(const TensorDesc &tensorDesc0, const TensorDesc &tensorDesc1)
{
    if (!IsTensorDescDimNumValid(tensorDesc1, tensorDesc0.shape.dimNum)) {
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    for (size_t i = 0; i < tensorDesc1.shape.dimNum; i++) {
        if (tensorDesc1.shape.dims[i] != tensorDesc0.shape.dims[i]) {
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}
} // namespace atb