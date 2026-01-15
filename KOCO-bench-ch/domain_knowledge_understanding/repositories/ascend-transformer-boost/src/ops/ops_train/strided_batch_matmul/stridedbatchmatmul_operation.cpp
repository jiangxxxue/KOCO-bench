/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "stridedbatchmatmul_operation.h"
#include "atb/utils/config.h"
#include "atb/utils/log.h"
#include "stridedbatchmatmul_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"
#include "atb/utils/singleton.h"

namespace {
bool ParamCheck(const atb::train::StridedBatchMatmulParam &opParam)
{
    (void)opParam;
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        ATB_LOG(ERROR) << "StridedBatchMatmulOperation is only supported in Atlas 800I A2 inference product.";
        return false;
    }
    return true;
}
} // namespace

namespace atb {
static const uint32_t IN_TENSOR_NUM = 2;
static const uint32_t OUT_TENSOR_NUM = 1;

OPERATION_PARAM_FUNCS(StridedBatchMatmulOperation, train::StridedBatchMatmulParam)

StridedBatchMatmulOperation::StridedBatchMatmulOperation(const train::StridedBatchMatmulParam &param)
    : OperationBase("StridedBatchMatmulOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("StridedBatchMatmulOperation");
}

StridedBatchMatmulOperation::~StridedBatchMatmulOperation() {}

uint32_t StridedBatchMatmulOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t StridedBatchMatmulOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status StridedBatchMatmulOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                   SVector<TensorDesc> &outTensorDescs) const
{
    int32_t sizeOfM = static_cast<int32_t>(param_.m.size());
    int32_t sizeOfN = static_cast<int32_t>(param_.n.size());
    if (sizeOfM != param_.batch || sizeOfN != param_.batch) {
        ATB_LOG(ERROR) << GetLogPrefix() << "param is not support, batch: " << param_.batch << "size of m:" << sizeOfM
                       << "size of n:" << sizeOfN;
        return ERROR_INVALID_PARAM;
    }
    if (param_.headNum <= 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "param is not support, headNum:" << param_.headNum;
        return ERROR_INVALID_PARAM;
    }
    __int128 outdims = 0;
    for (int i = 0; i < param_.batch; i++) {
        if (param_.m[i] <= 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "param is not support, m[" << i << "]:" << param_.m[i];
            return ERROR_INVALID_PARAM;
        }
        if (param_.n[i] <= 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "param is not support, n[" << i << "]:" << param_.n[i];
            return ERROR_INVALID_PARAM;
        }
        outdims += static_cast<__int128>(param_.m[i]) * param_.n[i];
    }
    outdims = outdims * param_.headNum;
    if (outdims > INT64_MAX) {
        ATB_LOG(ERROR) << GetLogPrefix() << "param is too large, infershape overflows.";
        return ERROR_INVALID_PARAM;
    }

    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = 1;
    outTensorDescs.at(0).shape.dims[0] = outdims;
    return NO_ERROR;
}

std::shared_ptr<Runner> StridedBatchMatmulOperation::CreateRunner(Context &context) const
{
    ContextBase *contextBase = dynamic_cast<ContextBase *>(&context);
    if (!contextBase) {
        ATB_LOG(DEBUG) << "context cast to contextBase failed!";
        return nullptr;
    }
    RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_STRIDEDBATCHMATMUL);
    Runner *runner = pool.MallocRunner<StridedBatchMatmulOpsRunner, train::StridedBatchMatmulParam>(param_);
    if (!runner) {
        ATB_LOG(DEBUG) << "MallocRunner from pool failed!";
        return std::make_shared<StridedBatchMatmulOpsRunner>(param_);
    }
    return std::shared_ptr<Runner>(runner, [&pool](Runner *runner) { pool.FreeRunner(runner); });
}

nlohmann::json StridedBatchMatmulOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

train::StridedBatchMatmulParam StridedBatchMatmulOperation::GetParam() const
{
    return param_;
}

void StridedBatchMatmulOperation::SetParam(const train::StridedBatchMatmulParam &param)
{
    param_ = param;
    runner_ = nullptr;
}

} // namespace atb