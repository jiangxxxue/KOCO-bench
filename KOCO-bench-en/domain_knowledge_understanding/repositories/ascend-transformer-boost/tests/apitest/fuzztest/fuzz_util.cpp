/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "fuzz_util.h"
#include "secodeFuzz.h"
#include <limits.h>
#include <float.h>

namespace atb {
const int ACL_DATA_RANDOM_NUM = 28;
const int ACL_FORMAT_RANDOM_NUM = 36;
const int DIMNUM_RANDOM_NUM = 9;
const int BOOL_RANDOM_NUM = 2;
const int SINGLE_DIM_RANDOM_NUM = 2000;

const std::vector<std::string> FuzzUtil::errorType_ = {"NO_ERROR", "ERROR_INVALID_PARAM", "ERROR_INVALID_GRAPH", "ERROR_INTERNAL_ERROR",
                    "ERROR_RT_FAIL", "ERROR_INVALID_IN_TENSOR_NUM", "ERROR_INVALID_TENSOR_DTYPE",
                    "ERROR_INVALID_TENSOR_FORMAT", "ERROR_INVALID_TENSOR_DIM", "ERROR_INVALID_TENSOR_SIZE",
                    "ERROR_OPERATION_NULL_RUNNER", "ERROR_GRAPH_INFERSHAPE_FUNC_FAIL", "ERROR_CANN_ERROR", "ERROR_INVALID_TENSOR_INI_MATCH"};

aclDataType FuzzUtil::GetRandomAclDataType(int input)
{
    return aclDataType(input % ACL_DATA_RANDOM_NUM);
}

aclFormat FuzzUtil::GetRandomAclFormat(int input)
{
    return aclFormat(input % ACL_FORMAT_RANDOM_NUM);
}

uint64_t FuzzUtil::GetRandomDimNum(uint32_t input)
{
    return input % DIMNUM_RANDOM_NUM;
}

bool FuzzUtil::GetRandomBool(uint32_t &fuzzIndex)
{   
    u16 randomNum = *(u16 *) DT_SetGetU16(&g_Element[fuzzIndex++], 0);
    return randomNum % BOOL_RANDOM_NUM;
}

TensorDesc FuzzUtil::GetRandomTensorDesc(uint32_t &fuzzIndex)
{
    TensorDesc desc;
    u32 dtypeRdNum = *(u32*) DT_SetGetS32(&g_Element[fuzzIndex++], 0);
    u32 formatRdNum = *(u32*) DT_SetGetS32(&g_Element[fuzzIndex++], 0);
    u32 dimNumRdNum = *(u32*) DT_SetGetS32(&g_Element[fuzzIndex++], 0);
    desc.dtype = GetRandomAclDataType(dtypeRdNum);
    desc.format = GetRandomAclFormat(formatRdNum);
    for (size_t i = 0; i < MAX_DIM; i++) {
        uint32_t dim = *(u32*) DT_SetGetS32(&g_Element[fuzzIndex++], 0);
        dim = dim % SINGLE_DIM_RANDOM_NUM;
        desc.shape.dims[i] = dim;
    }
    desc.shape.dimNum = GetRandomDimNum(dimNumRdNum);
    return desc;
}

Status FuzzUtil::SetupAndExecute(Operation *operation, SVector<TensorDesc> inTensorDescs, SVector<TensorDesc> outTensorDescs)
{
    OperationTest opTest;
    opTest.FloatRand(FLT_MIN, FLT_MAX);
    opTest.IntRand(INT_MIN, INT_MAX);
    Status st = opTest.Run(operation, inTensorDescs);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "Execute error!";
    }
    return st;
}
}
