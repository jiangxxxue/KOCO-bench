/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_to_allv_operation.h"
#include "atb/utils/config.h"
#include "all_to_allv_hccl_runner.h"
#include "atb/utils.h"
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

template <> Status CreateOperation(const infer::AllToAllVParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (opParam.backend != "hccl") {
        ATB_LOG(ERROR) << "backend is " << opParam.backend << "backend must be hccl";
        return ERROR_INVALID_PARAM;
    }
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "AllToAllV only supports Atlas 800I A2 inference product";
        return ERROR_INVALID_PARAM;
    }
    if (OperationUtil::DistributedInitCheck<infer::AllToAllVParam>(opParam) != NO_ERROR) {
        ATB_LOG(ERROR) << "AllToAllVOperation DistributedInitCheck failed";
        return ERROR_INVALID_PARAM;
    }
    size_t size = opParam.sendCounts.size();
    if (size != static_cast<size_t>(opParam.rankSize) || size != opParam.recvCounts.size() ||
        size != opParam.sdispls.size() || size != opParam.rdispls.size()) {
        ATB_LOG(ERROR)
            << "AllToAllVOperation param sendCounts,recvCounts,sdispls,rdispls size should be equal to ranksize";
        return ERROR_INVALID_PARAM;
    }
    for (size_t i = 0; i < size; i++) {
        if (opParam.sendCounts[i] < 0 || opParam.sdispls[i] < 0 || opParam.recvCounts[i] < 0 ||
            opParam.rdispls[i] < 0) {
            ATB_LOG(ERROR) << "AllToAllVOperation param sendCounts,recvCounts,sdispls,rdispls should more than zero";
            return ERROR_INVALID_PARAM;
        }
    }
    *operation = new (std::nothrow) AllToAllVOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new AllToAllOperation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

AllToAllVOperation::AllToAllVOperation(const infer::AllToAllVParam &param)
    : OperationBase("AllToAllVOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("AllToAllVOperation");
}

AllToAllVOperation::~AllToAllVOperation() {}

uint32_t AllToAllVOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t AllToAllVOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status AllToAllVOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                          SVector<TensorDesc> &outTensorDescs) const
{
    Status status = ParamCheck(inTensorDescs.at(0));
    if (status != NO_ERROR) {
        return status;
    }
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).shape.dimNum = 2; // 2：维数
    outTensorDescs.at(0).shape.dims[0] = 1;
    int64_t count = 0;
    for (const auto &recvCount : param_.recvCounts) {
        if (count > std::numeric_limits<int64_t>::max() - recvCount) {
            ATB_LOG(ERROR) << " ,AllToAllVOperation sum(recvCounts) will overflow ";
            return ERROR_INVALID_PARAM;
        }
        count += recvCount;
    }
    if (count <= 0) {
        ATB_LOG(ERROR) << "AllToAllVOperation sum(recvCounts) should more than 0";
        return ERROR_INVALID_PARAM;
    }
    for (size_t i = 0; i < param_.recvCounts.size(); i++) {
        if (param_.recvCounts[i] > std::numeric_limits<int64_t>::max() - param_.rdispls[i] ||
            param_.recvCounts[i] + param_.rdispls[i] > count) {
            ATB_LOG(ERROR) << "AllToAllVOperation recvCounts + rdispls is out of bounds";
            return ERROR_INVALID_PARAM;
        }
    }
    outTensorDescs.at(0).shape.dims[1] = count;
    return NO_ERROR;
}

Status AllToAllVOperation::ParamCheck(const TensorDesc &inTensorDesc) const
{
    uint64_t inputCount = Utils::GetTensorNumel(inTensorDesc);
    for (size_t i = 0; i < param_.sendCounts.size(); i++) {
        if (param_.sendCounts[i] > std::numeric_limits<int64_t>::max() - param_.sdispls[i] ||
            static_cast<uint64_t>(param_.sendCounts[i] + param_.sdispls[i]) > inputCount) {
            ATB_LOG(ERROR) << "AllToAllVOperation sendCounts + sdispls is out of bounds";
            return ERROR_INVALID_PARAM;
        }
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> AllToAllVOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (param_.backend == "hccl") {
        if (param_.hcclComm == nullptr) {
            return std::make_shared<AllToAllVHcclRunner>(param_, !param_.rankTableFile.empty());
        } else {
            return std::make_shared<AllToAllVHcclRunner>(param_, param_.hcclComm);
        }
    }
    return std::shared_ptr<Runner>();
}

nlohmann::json AllToAllVOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb