/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "atb/utils/common_utils.h"
#include <sstream>
#include <atb/utils/log.h>

namespace atb {
std::string GenerateOperationName(const std::string &opType, const std::vector<int64_t> &ids)
{
    std::string opName = opType;
    for (size_t i = 0; i < ids.size(); i++) {
        opName += "_";
        opName += std::to_string(ids.at(i));
    }
    return opName;
}

HcclDataType GetHcclDtype(const aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT:
            return HCCL_DATA_TYPE_FP32;
        case ACL_FLOAT16:
            return HCCL_DATA_TYPE_FP16;
        case ACL_INT8:
            return HCCL_DATA_TYPE_INT8;
        case ACL_INT32:
            return HCCL_DATA_TYPE_INT32;
        case ACL_UINT8:
            return HCCL_DATA_TYPE_UINT8;
        case ACL_INT16:
            return HCCL_DATA_TYPE_INT16;
        case ACL_UINT16:
            return HCCL_DATA_TYPE_UINT16;
        case ACL_UINT32:
            return HCCL_DATA_TYPE_UINT32;
        case ACL_INT64:
            return HCCL_DATA_TYPE_INT64;
        case ACL_BF16:
            return HCCL_DATA_TYPE_BFP16;
        default:
            ATB_LOG(ERROR) << "not support dtype:" << dtype;
            return static_cast<HcclDataType>(255); // RESERVED TYPE
    }
}

HcclReduceOp GetAllReduceType(const std::string &allReduceType)
{
    if (allReduceType == "sum") {
        return HCCL_REDUCE_SUM;
    } else if (allReduceType == "prod") {
        return HCCL_REDUCE_PROD;
    } else if (allReduceType == "max") {
        return HCCL_REDUCE_MAX;
    } else if (allReduceType == "min") {
        return HCCL_REDUCE_MIN;
    } else {
        return HCCL_REDUCE_RESERVED;
    }
}
} // namespace atb