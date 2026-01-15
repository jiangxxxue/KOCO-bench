/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gating_operation.h"
#include <algorithm>
#include <unordered_set>
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/config.h"
#include "gating_ops_runner.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const int32_t MAX_CUM_SUM_NUM = 1024;
static const uint32_t IN_TENSOR_NUM = 2;
static const uint32_t GATING_TP_OUT_TENSOR_NUM = 3;
static const uint32_t GATING_EP_OUT_TENSOR_NUM = 4;
static const size_t DIM_2 = 2;

template <> Status CreateOperation(const infer::GatingParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    ATB_LOG(INFO) << "CreateOperation GatingParam topkExpertNum: " << opParam.topkExpertNum
                  << ", cumSumNum: " << opParam.cumSumNum
                  << ", deviceExpert: " << OperationUtil::VectorToString(opParam.deviceExpert);
    if (opParam.cumSumNum < 0 || opParam.cumSumNum > MAX_CUM_SUM_NUM) {
        ATB_LOG(ERROR) << "param cumSumNum should be in range [0, " << MAX_CUM_SUM_NUM << "]";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.cumSumNum == 0) {
        if (opParam.topkExpertNum != 1) {
            ATB_LOG(ERROR) << "param topkExpertNum should be 1";
            return ERROR_INVALID_PARAM;
        }
    } else {
        if (opParam.topkExpertNum <= 0 || opParam.topkExpertNum > opParam.cumSumNum) {
            ATB_LOG(ERROR) << "param topkExpertNum should be in range (0, cumSumNum]";
            return ERROR_INVALID_PARAM;
        }
    }
    if (!opParam.deviceExpert.empty()) {
        if (!GetSingleton<Config>().Is910B()) {
            ATB_LOG(ERROR) << "EP only support Atlas 800I A2 inference product";
            return ERROR_INVALID_PARAM;
        }
        if (opParam.cumSumNum == 0) {
            ATB_LOG(ERROR) << "in EP, param cumSumNum should not be 0";
            return ERROR_INVALID_PARAM;
        }
        std::unordered_set<int32_t> experts;
        for (size_t i = 0; i < opParam.deviceExpert.size(); i++) {
            if (opParam.deviceExpert.at(i) < 0 || opParam.deviceExpert.at(i) >= opParam.cumSumNum) {
                ATB_LOG(ERROR) << "param deviceExpert elements should be in range [0, cumSumNum)";
                return ERROR_INVALID_PARAM;
            }
            if (experts.find(opParam.deviceExpert.at(i)) != experts.end()) {
                ATB_LOG(ERROR) << "param deviceExpert elements should be unique";
                return ERROR_INVALID_PARAM;
            }
            experts.insert(opParam.deviceExpert.at(i));
        }
    }

    *operation = new (std::nothrow) GatingOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

GatingOperation::GatingOperation(const infer::GatingParam &param) : OperationBase("GatingOperation"), param_(param)
{
    std::string operationIrName =
        param_.deviceExpert.empty() ?
            (param_.cumSumInt64 ? "GatingOperationInt64" : "GatingOperation") :
            (param_.cumSumInt64 ? "GatingOperationExpertParallelismInt64" : "GatingOperationExpertParallelism");
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(operationIrName);
}

GatingOperation::~GatingOperation() {}

uint32_t GatingOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t GatingOperation::GetOutputNum() const
{
    return param_.deviceExpert.empty() ? GATING_TP_OUT_TENSOR_NUM : GATING_EP_OUT_TENSOR_NUM;
}

Status GatingOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                       SVector<TensorDesc> &outTensorDescs) const
{
    size_t outTensorId = 0;
    TensorDesc &tokenIndexTensorDesc = outTensorDescs.at(outTensorId++);
    TensorDesc &cumSumTensorDesc = outTensorDescs.at(outTensorId++);
    TensorDesc &originalIndexTensorDesc = outTensorDescs.at(outTensorId++);

    tokenIndexTensorDesc = inTensorDescs.at(0);
    cumSumTensorDesc = inTensorDescs.at(0);
    if (param_.cumSumInt64) {
        cumSumTensorDesc.dtype = ACL_INT64;
    }
    originalIndexTensorDesc = inTensorDescs.at(0);

    if (param_.deviceExpert.empty()) {
        cumSumTensorDesc.shape.dims[0] = param_.cumSumNum == 0 ? 1 : param_.cumSumNum;
    } else {
        cumSumTensorDesc.shape.dims[0] = static_cast<int64_t>(param_.deviceExpert.size());
        TensorDesc &validIndexTensorDesc = outTensorDescs.at(outTensorId++);
        validIndexTensorDesc = inTensorDescs.at(0);
        validIndexTensorDesc.shape.dims[0] = 1;
    }

    return NO_ERROR;
}

Status GatingOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return InTensorDescsCheck(inTensorDescs);
}

Status GatingOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    Status status = InTensorDescsCheck(inTensorDescs);
    if (status != NO_ERROR) {
        return status;
    }
    int64_t topkDim0 = inTensors.at(0).desc.shape.dims[0];
    return OutTensorsCheck(outTensors, topkDim0);
}

std::shared_ptr<Runner> GatingOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<GatingOpsRunner>(param_);
}

Status GatingOperation::InTensorDescsCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    size_t inTensorId = 0;
    TensorDesc topkTensorDesc = inTensorDescs.at(inTensorId++);
    TensorDesc idxArrTensorDesc = inTensorDescs.at(inTensorId++);

    if (!TensorCheck::IsTensorDescDimNumValid(topkTensorDesc, 1)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 dimNum should be 1 but get " << topkTensorDesc.shape.dimNum;
        return ERROR_INVALID_TENSOR_SIZE;
    }
    if (topkTensorDesc.shape.dims[0] % param_.topkExpertNum != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 dim0 [" << topkTensorDesc.shape.dims[0]
                       << "] should be an integer multiple of param topkExpertNum [" << param_.topkExpertNum << "]";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (!TensorCheck::IsTensorDescDimNumValid(idxArrTensorDesc, 1)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor1 dimNum should be 1 but get " << idxArrTensorDesc.shape.dimNum;
        return ERROR_INVALID_TENSOR_SIZE;
    }
    if (idxArrTensorDesc.shape.dims[0] != topkTensorDesc.shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor1 dim0 [" << idxArrTensorDesc.shape.dims[0]
                       << "] and inTensor0 dim0 [" << topkTensorDesc.shape.dims[0] << "] should be equal";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status GatingOperation::OutTensorsCheck(const SVector<Tensor> &outTensors, int64_t topkDim0) const
{
    size_t outTensorId = 0;
    TensorDesc tokenIndexTensorDesc = outTensors.at(outTensorId++).desc;
    TensorDesc cumSumTensorDesc = outTensors.at(outTensorId++).desc;
    TensorDesc originalIndexTensorDesc = outTensors.at(outTensorId++).desc;
    if (!TensorCheck::IsTensorDescDimNumValid(tokenIndexTensorDesc, 1)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor0 dimNum should be 1 but get "
                       << tokenIndexTensorDesc.shape.dimNum;
        return ERROR_INVALID_TENSOR_SIZE;
    }
    if (tokenIndexTensorDesc.shape.dims[0] != topkDim0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor0 dim0 [" << tokenIndexTensorDesc.shape.dims[0]
                       << "] and inTensor0 dim0 [" << topkDim0 << "] should be equal";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (!TensorCheck::IsTensorDescDimNumValid(cumSumTensorDesc, 1)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor1 dimNum should be 1 but get " << cumSumTensorDesc.shape.dimNum;
        return ERROR_INVALID_TENSOR_SIZE;
    }
    int32_t expertNum = param_.cumSumNum;
    if (expertNum == 0) {
        expertNum = 1;
    } else if (!param_.deviceExpert.empty()) {
        expertNum = static_cast<int32_t>(param_.deviceExpert.size());
    }
    if (cumSumTensorDesc.shape.dims[0] != expertNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor1 dim0 [" << cumSumTensorDesc.shape.dims[0] << "] should be "
                       << expertNum;
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (!TensorCheck::IsTensorDescDimNumValid(originalIndexTensorDesc, 1)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor2 dimNum should be 1 but get "
                       << originalIndexTensorDesc.shape.dimNum;
        return ERROR_INVALID_TENSOR_SIZE;
    }
    if (originalIndexTensorDesc.shape.dims[0] != topkDim0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor2 dim0 [" << originalIndexTensorDesc.shape.dims[0]
                       << "] and inTensor0 dim0 [" << topkDim0 << "] should be equal";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (!param_.deviceExpert.empty()) {
        TensorDesc validIndexTensorDesc = outTensors.at(outTensorId++).desc;
        if (!TensorCheck::IsTensorDescDimNumValid(validIndexTensorDesc, 1)) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor3 dimNum should be 1 but get "
                           << validIndexTensorDesc.shape.dimNum;
            return ERROR_INVALID_TENSOR_SIZE;
        }
        if (validIndexTensorDesc.shape.dims[0] != 1) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor3 dim0 [" << validIndexTensorDesc.shape.dims[0]
                           << "] should be 1";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }

    return NO_ERROR;
}
} // namespace atb