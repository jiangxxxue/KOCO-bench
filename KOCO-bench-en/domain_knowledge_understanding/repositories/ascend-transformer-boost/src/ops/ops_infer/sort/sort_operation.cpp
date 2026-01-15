/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sort_operation.h"
#include "atb/utils/log.h"
#include "sort_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/singleton.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"
namespace {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 2;
static const int64_t LIMITED_NUM = 580000000;
static const int64_t ZERO = 0;
static const int64_t ONE = 1;
static const int64_t SIXTEEN = 16;
bool ParamCheck(const atb::infer::SortParam &opParam)
{
    (void)opParam;
    return true;
}
} // namespace

namespace atb {
OPERATION_PARAM_FUNCS(SortOperation, infer::SortParam)

SortOperation::SortOperation(const infer::SortParam &param) : OperationBase("SortOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("SortOperation");
}

SortOperation::~SortOperation() {}

uint32_t SortOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t SortOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status SortOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                     SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(1) = inTensorDescs.at(0);
    outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = param_.num[0];
    outTensorDescs.at(1).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = param_.num[0];
    outTensorDescs.at(1).dtype = ACL_INT32;
    return NO_ERROR;
}

Status SortOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    const TensorDesc &xTensorDesc = inTensorDescs.at(0);
    if (!GetSingleton<Config>().Is910B() && xTensorDesc.dtype == ACL_FLOAT) {
        ATB_LOG(ERROR) << "SortOperation of the float type supports only the Atlas 800 A2 inference products.";
        return ERROR_INVALID_TENSOR_DTYPE;
    }
    Status status = ParamCheckImpl(xTensorDesc);
    if (status != NO_ERROR) {
        return status;
    }
    return NO_ERROR;
}

Status SortOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    const TensorDesc &xTensorDesc = inTensors.at(0).desc;
    Status status = ParamCheckImpl(xTensorDesc);
    if (status != NO_ERROR) {
        return status;
    }
    ATB_LOG(DEBUG) << "outTensors Size:" << outTensors.size();
    return NO_ERROR;
}

Status SortOperation::ParamCheckImpl(const TensorDesc &xTensorDesc) const
{
    if (param_.num.size() != 1) {
        ATB_LOG(ERROR) << "ParamCheckImpl:param.num size should be 1.";
        return ERROR_INVALID_PARAM;
    }
    if (param_.num.at(0) <= 0 || xTensorDesc.shape.dims[xTensorDesc.shape.dimNum - 1] < param_.num.at(0)) {
        ATB_LOG(ERROR) << "ParamCheckImpl:param.num should >0 && <= lastdim";
        return ERROR_INVALID_PARAM;
    }
    int64_t numSize = param_.num.size();
    if (GetSingleton<Config>().Is310P() && numSize == ONE && param_.num[ZERO] < SIXTEEN) {
        uint64_t xDimNum = xTensorDesc.shape.dimNum;
        for (size_t i = 0; i < xDimNum - 1; i++) {
            int64_t dim = xTensorDesc.shape.dims[i];
            if (dim > LIMITED_NUM) {
                ATB_LOG(ERROR) << "ParamCheckImpl: dim should be less than or equal to " << LIMITED_NUM
                            << " in Atlas 300I Duo inference products.";
                return ERROR_INVALID_TENSOR_DIM;
            }
        }
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> SortOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<SortOpsRunner>(param_);
}

nlohmann::json SortOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

infer::SortParam SortOperation::GetParam() const
{
    return param_;
}

void SortOperation::SetParam(const infer::SortParam &param)
{
    param_ = param;
    runner_ = nullptr;
}

} // namespace atb
