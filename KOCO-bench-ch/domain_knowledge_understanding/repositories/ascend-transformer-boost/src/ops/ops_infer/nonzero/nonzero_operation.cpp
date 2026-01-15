/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nonzero_operation.h"
#include "atb/utils/config.h"
#include "nonzero_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

static const int32_t IN_TENSOR_NUM = 1;
static const int32_t OUT_TENSOR_NUM = 2;

namespace atb {
template <> Status CreateOperation(const infer::NonzeroParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        ATB_LOG(ERROR) << "operation nullptr";
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "only support Atlas 800I A2 inference product";
        return ERROR_INVALID_PARAM;
    }

    *operation = new (std::nothrow) NonzeroOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

NonzeroOperation::NonzeroOperation(const infer::NonzeroParam &param) : OperationBase("NonzeroOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("NonzeroOperation");
}

NonzeroOperation::~NonzeroOperation() {}

uint32_t NonzeroOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t NonzeroOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status NonzeroOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                        SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(1) = inTensorDescs.at(0);
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dimNum;

    outTensorDescs.at(0).shape.dims[1] = static_cast<int64_t>(Utils::GetTensorNumel(inTensorDescs.at(0)));
    outTensorDescs.at(1).shape.dimNum = 1;
    outTensorDescs.at(1).shape.dims[0] = 1;
    return NO_ERROR;
}

Status NonzeroOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    Status st = SetupDimNumCheck(inTensors, outTensors);
    return st;
}

Status NonzeroOperation::SetupDimNumCheck(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    // 第一个outtensor的dim[0]为intensor的dimNum, 且第一个outtensor的dim[1]为intensor Numel, 第二个outtensor应为[1]
    int64_t inTensorDimNum = inTensors.at(0).desc.shape.dimNum;
    if (outTensors.at(0).desc.shape.dims[0] != inTensorDimNum ||
        outTensors.at(0).desc.shape.dims[1] != static_cast<int64_t>(Utils::GetTensorNumel(inTensors.at(0).desc))) {
        ATB_LOG(ERROR) << "invalid tensor dim number, outTensors.at(0).dim[0] should be inTensorDimNum, "
                          "outTensors.at(0).dim[1] == intensor Numel!";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (outTensors.at(1).desc.shape.dimNum != 1 || outTensors.at(1).desc.shape.dims[0] != 1) {
        ATB_LOG(ERROR) << "invalid outtensor dim number, outTensors.at(1) dimNum should be 1 and dim[0] == 1";
        return ERROR_INVALID_TENSOR_DIM;
    }

    return NO_ERROR;
}

std::shared_ptr<Runner> NonzeroOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<NonzeroRunner>(param_);
}

} // namespace atb