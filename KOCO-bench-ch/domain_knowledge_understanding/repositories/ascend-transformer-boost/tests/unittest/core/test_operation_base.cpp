/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <vector>
#include <string>
#include <string.h>
#include <cstdlib>
#include <gtest/gtest.h>
#include <atb/utils/log.h>
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "atb/operation/operation_base.h"
#include "atb/utils/tensor_util.h"

using namespace atb;

TEST(TestOperationBase, TestGetName)
{
    setenv("ATB_PROFILING_ENABLE", "1", 1);
    setenv("ATB_SAVE_TENSOR", "1", 1);
    setenv("ATB_LOG_LEVEL", "INFO", 1);
    atb::infer::SplitParam param;
    param.splitDim = 0;
    param.splitNum = 2;
    atb::Operation *op = nullptr;
    Status st = atb::CreateOperation<atb::infer::SplitParam>(param, &op);
    EXPECT_EQ(st, NO_ERROR);
    // GetName
    EXPECT_EQ(op->GetName(), "SplitOperation");
    Mki::SVector<Mki::TensorDesc> opsInTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {2, 2}}
    };
    atb::SVector<atb::TensorDesc> inTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsInTensorDescs, inTensorDescs);
    atb::Tensor inTensor = {inTensorDescs.at(0)};
    Mki::SVector<Mki::TensorDesc> opsOutTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {1, 2}},
        {Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, {1, 2}}
    };
    atb::SVector<atb::TensorDesc> outTensorDescs;
    TensorUtil::OpsTensorDescs2AtbTensorDescs(opsOutTensorDescs, outTensorDescs);
    atb::Tensor outTensor0 = {outTensorDescs.at(0)};
    atb::Tensor outTensor1 = {outTensorDescs.at(1)};
    const VariantPack &variantPack = {{inTensor}, {outTensor0, outTensor1}};
    uint64_t workspaceSize = 0;
    uint64_t &refWorkspaceSize = workspaceSize;
    EXPECT_EQ(op->Setup(variantPack, refWorkspaceSize, nullptr), ERROR_INVALID_PARAM);
    DestroyOperation(op);
    setenv("ATB_PROFILING_ENABLE", "0", 1);
    setenv("ATB_SAVE_TENSOR", "0", 1);
    setenv("ATB_LOG_LEVEL", "FATAL", 1);
}