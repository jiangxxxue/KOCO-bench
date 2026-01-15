/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_operation.h"
#include "atb/utils/config.h"
#include "reduce_scatter_lccl_runner.h"
#include "reduce_scatter_hccl_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {

static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 1;

template <> Status CreateOperation(const infer::ReduceScatterParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (opParam.backend != "hccl" && opParam.backend != "lccl") {
        ATB_LOG(ERROR) << "backend is " << opParam.backend << "backend must either be hccl or lccl";
        return ERROR_INVALID_PARAM;
    }
    if (GetSingleton<Config>().Is310P()) {
        ATB_LOG(ERROR) << "ReduceScatter is not support in Atlas 300I Duo inference products";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.reduceType != "sum" && opParam.reduceType != "prod" && opParam.reduceType != "max" &&
        opParam.reduceType != "min") {
        ATB_LOG(ERROR) << "reduceType is " << opParam.reduceType
                       << "ReduceScatterParam reduceType must be one of the following sum, prod, max, min";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.backend == "lccl" && opParam.reduceType == "prod") {
        ATB_LOG(ERROR) << "reduceType is " << opParam.reduceType << "lccl not support prod ";
        return ERROR_INVALID_PARAM;
    }
    if (OperationUtil::DistributedInitCheck<infer::ReduceScatterParam>(opParam) != NO_ERROR) {
        ATB_LOG(ERROR) << "ReduceScatterOperation DistributedInitCheck failed";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) ReduceScatterOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

ReduceScatterOperation::ReduceScatterOperation(const infer::ReduceScatterParam &param)
    : OperationBase("ReduceScatterOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("ReduceScatterOperation");
}

ReduceScatterOperation::~ReduceScatterOperation() {}

uint32_t ReduceScatterOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t ReduceScatterOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status ReduceScatterOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                              SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0] / param_.rankSize;
    ATB_LOG(INFO) << GetLogPrefix() << " inTensorDescs Size:" << inTensorDescs.size()
                  << " outTensorDescs Size:" << outTensorDescs.size();
    return NO_ERROR;
}

Status ReduceScatterOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(0).shape.dims[0] % param_.rankSize != 0) {
        ATB_LOG(ERROR) << "inTensor(0) dimNum is " << inTensorDescs.at(0).shape.dims[0] << " rankSize is "
                       << param_.rankSize << " inTensor(0) dimNum should be an integer multiple of ranksize";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (param_.backend == "hccl") {
        return HcclDtypeCheck(inTensorDescs.at(0));
    } else if (param_.backend == "lccl") {
        return LcclDtypeCheck(inTensorDescs.at(0));
    }
    return NO_ERROR;
}

Status ReduceScatterOperation::HcclDtypeCheck(const TensorDesc &inTensorDesc) const
{
    if (GetSingleton<Config>().Is910B()) {
        if (param_.reduceType == "prod") {
            if (inTensorDesc.dtype == ACL_INT16 || inTensorDesc.dtype == ACL_BF16) {
                ATB_LOG(ERROR) << "Atlas A2 inference products hccl reducescatter prod is not support int16 , bf16";
                return ERROR_INVALID_TENSOR_DTYPE;
            }
        }
    }
    if (GetSingleton<Config>().Is910A()) {
        if (inTensorDesc.dtype == ACL_INT16 || inTensorDesc.dtype == ACL_BF16) {
            ATB_LOG(ERROR) << "Atlas inference products hccl reducescatter is not support int16 , bf16";
            return ERROR_INVALID_TENSOR_DTYPE;
        }
    }
    return NO_ERROR;
}

Status ReduceScatterOperation::LcclDtypeCheck(const TensorDesc &inTensorDesc) const
{
    if (inTensorDesc.dtype == ACL_INT64) {
        ATB_LOG(ERROR) << "lccl reduceScatter is not support int64";
        return ERROR_INVALID_TENSOR_DTYPE;
    }
    return NO_ERROR;
}

Status ReduceScatterOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    if (inTensors.at(0).desc.shape.dimNum != outTensors.at(0).desc.shape.dimNum) {
        ATB_LOG(ERROR) << "inTensor dimNum is " << inTensors.at(0).desc.shape.dimNum << "outTensor dimNum is "
                       << outTensors.at(0).desc.shape.dimNum << " inTensor dim should be equal outTensor dim";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensors.at(0).desc.shape.dims[0] != param_.rankSize * outTensors.at(0).desc.shape.dims[0]) {
        ATB_LOG(ERROR) << " inTensor first dimension is" << inTensors.at(0).desc.shape.dims[0]
                       << " outTensor first dimension is" << outTensors.at(0).desc.shape.dims[0]
                       << " inTensor first dimension must be rankSize* (outTensor first dimension) ";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> ReduceScatterOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (param_.backend == "lccl") {
        return std::make_shared<ReduceScatterLcclRunner>(param_, context);
    } else if (param_.backend == "hccl") {
        if (param_.hcclComm == nullptr) {
            return std::make_shared<ReduceScatterHcclRunner>(param_, !param_.rankTableFile.empty());
        } else {
            return std::make_shared<ReduceScatterHcclRunner>(param_, param_.hcclComm);
        }
    }

    ATB_LOG(ERROR) << "ReduceScatterOperation::ReduceScatterOperation backend " << param_.backend << "is not exist.";
    return std::shared_ptr<Runner>();
}

nlohmann::json ReduceScatterOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb