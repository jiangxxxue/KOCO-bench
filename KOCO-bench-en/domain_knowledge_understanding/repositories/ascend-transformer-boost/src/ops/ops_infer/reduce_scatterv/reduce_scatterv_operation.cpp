/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatterv_operation.h"
#include "atb/utils/config.h"
#include "reduce_scatterv_hccl_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

static constexpr uint32_t IN_TENSOR_NUM = 5;
static constexpr uint32_t OUT_TENSOR_NUM = 1;
static constexpr int32_t NUM_2 = 2;
static constexpr int32_t NUM_3 = 3;
static constexpr uint32_t NUM_4 = 4;
namespace {
bool ParamCheck(const atb::infer::ReduceScatterVParam &opParam)
{
    if (opParam.backend != "hccl") {
        ATB_LOG(ERROR) << "backend is " << opParam.backend << "backend must be hccl";
        return false;
    }
    if (opParam.reduceType != "sum" && opParam.reduceType != "max" && opParam.reduceType != "min") {
        ATB_LOG(ERROR) << "reduceType is " << opParam.reduceType
                       << " ReduceScatterParam reduceType must be one of the following sum, max, min";
        return false;
    }
    if (opParam.rankSize < NUM_2) {
        ATB_LOG(ERROR) << "ReduceScatterV ranksize must be larger than 1, current ranksize: " << opParam.rankSize;
        return atb::ERROR_INVALID_PARAM;
    }
    if (atb::OperationUtil::DistributedInitCheck<atb::infer::ReduceScatterVParam>(opParam) != atb::NO_ERROR) {
        ATB_LOG(ERROR) << "ReduceScatterVOperation DistributedInitCheck failed";
        return false;
    }
    return true;
}
} // namespace

namespace atb {
OPERATION_PARAM_FUNCS(ReduceScatterVOperation, infer::ReduceScatterVParam)

ReduceScatterVOperation::ReduceScatterVOperation(const infer::ReduceScatterVParam &param)
    : OperationBase("ReduceScatterVOperation"), param_(param)
{
    if (GetSingleton<Config>().Is310P()) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("ReduceScatterVOperation310p");
    } else {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("ReduceScatterVOperation");
    }
}

ReduceScatterVOperation::~ReduceScatterVOperation() {}

uint32_t ReduceScatterVOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t ReduceScatterVOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status ReduceScatterVOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
    SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(NUM_4).shape.dims[0]; // y: shape[ranksize]
    ATB_LOG(INFO) << GetLogPrefix() << " inTensorDescs Size:" << inTensorDescs.size()
                  << " outTensorDescs Size:" << outTensorDescs.size();
    return NO_ERROR;
}

Status ReduceScatterVOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    (void)inTensorDescs;
    if (inTensorDescs.at(0).shape.dimNum != NUM_2) { // 2: 2维
        ATB_LOG(ERROR) << "inTensor(0) dimNum should be 2";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status ReduceScatterVOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
    const SVector<Tensor> &outTensors) const
{
    if (inTensors.at(0).desc.shape.dimNum != outTensors.at(0).desc.shape.dimNum) {
        ATB_LOG(ERROR) << "inTensor dimNum is " << inTensors.at(0).desc.shape.dimNum << "outTensor dimNum is "
                       << outTensors.at(0).desc.shape.dimNum << " inTensor dim should be equal outTensor dim";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> ReduceScatterVOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (param_.backend == "hccl") {
        if (param_.hcclComm == nullptr) {
            return std::make_shared<ReduceScatterVHcclRunner>(param_, !param_.rankTableFile.empty());
        } else {
            return std::make_shared<ReduceScatterVHcclRunner>(param_, param_.hcclComm);
        }
    }

    ATB_LOG(ERROR) << "ReduceScatterVOperation::ReduceScatterVOperation backend " << param_.backend << "is not exist.";
    return std::shared_ptr<Runner>();
}

infer::ReduceScatterVParam ReduceScatterVOperation::GetParam() const
{
    return param_;
}

void ReduceScatterVOperation::SetParam(const infer::ReduceScatterVParam &param)
{
    param_ = param;
    runner_ = nullptr;
}

nlohmann::json ReduceScatterVOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb