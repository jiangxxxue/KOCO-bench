/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_operation.h"
#include "atb/utils/config.h"
#include "all_gather_hccl_runner.h"
#include "all_gather_lccl_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {

static const int32_t IN_TENSOR_NUM = 1;
static const int32_t OUT_TENSOR_NUM = 1;

template <> Status CreateOperation(const infer::AllGatherParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (opParam.backend != "hccl" && opParam.backend != "lccl") {
        ATB_LOG(ERROR) << "backend is " << opParam.backend << "backend must either be hccl or lccl";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.backend == "lccl" && GetSingleton<Config>().Is310P()) {
        ATB_LOG(ERROR) << "AllGather lccl is not support in Atlas inference products";
        return ERROR_INVALID_PARAM;
    }
    if (OperationUtil::DistributedInitCheck<infer::AllGatherParam>(opParam) != NO_ERROR) {
        ATB_LOG(ERROR) << "AllGatherOperation DistributedInitCheck failed";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) AllGatherOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

AllGatherOperation::AllGatherOperation(const infer::AllGatherParam &param)
    : OperationBase("AllGatherOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("AllGatherOperation");
}

AllGatherOperation::~AllGatherOperation() {}

uint32_t AllGatherOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t AllGatherOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status AllGatherOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                          SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum + 1;
    outTensorDescs.at(0).shape.dims[0] = param_.rankSize;
    for (uint64_t i = 0; i < inTensorDescs.at(0).shape.dimNum; i++) {
        outTensorDescs.at(0).shape.dims[i + 1] = inTensorDescs.at(0).shape.dims[i];
    }
    return NO_ERROR;
}

Status AllGatherOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(0).shape.dimNum >= MAX_DIM) {
        ATB_LOG(ERROR) << "inTensor(0) dimNum should <  MAX_DIM(8)";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status AllGatherOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    if (inTensors.at(0).desc.shape.dimNum >= MAX_DIM) {
        ATB_LOG(ERROR) << "inTensor(0) dimNum should <  MAX_DIM(8)";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (outTensors.at(0).desc.shape.dimNum != (inTensors.at(0).desc.shape.dimNum + 1)) {
        ATB_LOG(ERROR) << "outTensor dim should be one larger than inTensor dim";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (outTensors.at(0).desc.shape.dims[0] != param_.rankSize) {
        ATB_LOG(ERROR) << "outTensor first dimension does not match rankSize";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> AllGatherOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (param_.backend == "hccl") {
        if (param_.hcclComm == nullptr) {
            return std::make_shared<AllGatherHcclRunner>(param_, !param_.rankTableFile.empty());
        } else {
            return std::make_shared<AllGatherHcclRunner>(param_, param_.hcclComm);
        }
    } else if (param_.backend == "lccl") {
        return std::make_shared<AllGatherLcclRunner>(param_, context);
    }
    ATB_LOG(FATAL) << "AllGatherOperation::AllGatherOperation backend " << param_.backend << "is not exist.";
    return std::shared_ptr<Runner>();
}

nlohmann::json AllGatherOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb