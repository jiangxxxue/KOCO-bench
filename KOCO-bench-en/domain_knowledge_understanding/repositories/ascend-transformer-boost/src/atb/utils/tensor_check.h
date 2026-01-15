/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_TENSOR_CHECK_H
#define ATB_TENSOR_CHECK_H

#include <string>
#include <mki/tensor.h>
#include "atb/utils/singleton.h"
#include "mki/utils/operationir/operation_ir.h"
#include "atb/types.h"
#include "atb/utils/config.h"
#include "atb/utils/runner_variant_pack.h"

namespace atb {
class TensorCheck {
public:
    static bool IsTensorDType(const Tensor &tensor, const aclDataType &dType);
    inline static bool IsTensorDType(const TensorDesc &tensorDesc, const aclDataType &dType)
    {
        return tensorDesc.dtype == dType;
    }
    static bool IsTensorFormat(const Tensor &tensor, const aclFormat &format);
    inline static bool IsTensorFormat(const TensorDesc &tensorDesc, const aclFormat &format)
    {
        return tensorDesc.format == format;
    }
    static Status CheckTensorShape(const Tensor &tensor);
    static Status CheckTensorShape(const TensorDesc &tensorDesc);
    static bool IsTensorDescDimNumValid(const TensorDesc &tensorDesc, uint64_t dimNum);
    static bool IsTensorSizeValid(const Tensor &tensor);
    static bool IsTensorSizeValid(const TensorDesc &tensorDesc);
    static inline bool IsDimNumValid(const uint64_t &dimNum)
    {
        return dimNum != 0 && dimNum <= MAX_DIM;
    }
    inline static bool IsEmptyTensor(const Tensor &tensor)
    {
        bool emptyShape = (tensor.desc.shape.dimNum == 0);
        for (size_t i = 0; !emptyShape && i < tensor.desc.shape.dimNum; i++) {
            if (tensor.desc.shape.dims[i] == 0) {
                emptyShape = true;
            }
        }
        return emptyShape && tensor.dataSize == 0 && !tensor.deviceData && !tensor.hostData;
    }
    // The tensor has no dimensions or if any dimension is zero, both indicate an empty tensor.
    inline static bool IsEmptyTensor(const TensorDesc &tensorDesc)
    {
        if (tensorDesc.shape.dimNum == 0) {
            return true;
        }
        for (size_t i = 0; i < tensorDesc.shape.dimNum; i++) {
            if (tensorDesc.shape.dims[i] == 0) {
                return true;
            }
        }
        return false;
    }
    static Status TensorDescsEqual(const TensorDesc &tensorDesc0, const TensorDesc &tensorDesc1);

    static bool CheckBF16(const SVector<TensorDesc> &tensorDescs)
    {
        if (GetSingleton<Config>().Is310P() || GetSingleton<Config>().Is910A() || GetSingleton<Config>().Is310B()) {
            for (size_t tensorId = 0; tensorId < tensorDescs.size(); tensorId++) {
                if (tensorDescs.at(tensorId).dtype == ACL_BF16) {
                    return false;
                }
            }
        }
        return true;
    }

private:
    static bool IsSupportDtype(const SVector<aclDataType> &dtypes, aclDataType dtype);
};
} // namespace atb
#endif