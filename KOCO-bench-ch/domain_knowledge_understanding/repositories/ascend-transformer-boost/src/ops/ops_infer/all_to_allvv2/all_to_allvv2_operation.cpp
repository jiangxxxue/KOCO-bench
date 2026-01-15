/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_to_allvv2_operation.h"
#include "atb/utils/config.h"
#include "all_to_allvv2_hccl_runner.h"
#include "atb/utils.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace {
bool ParamCheck(const atb::infer::AllToAllVV2Param &opParam)
{
    if (opParam.backend != "hccl") {
        ATB_LOG(ERROR) << "backend is " << opParam.backend << ", backend must be hccl";
        return false;
    }
    // -1: 未指定rank号多卡情况
    if (opParam.rank == -1 && opParam.rankSize > 1) {
        ATB_LOG(ERROR) << "AllToAllVV2Operation multi-card, rank must be specified";
        return false;
    }
    if (atb::OperationUtil::DistributedInitCheck<atb::infer::AllToAllVV2Param>(opParam) != atb::NO_ERROR) {
        ATB_LOG(ERROR) << "AllToAllVV2Operation DistributedInitCheck failed";
        return false;
    }
    return true;
}
} // namespace

namespace atb {
static const int32_t IN_TENSOR_NUM = 6;
static const int32_t OUT_TENSOR_NUM = 1;

OPERATION_PARAM_FUNCS(AllToAllVV2Operation, infer::AllToAllVV2Param)

AllToAllVV2Operation::AllToAllVV2Operation(const infer::AllToAllVV2Param &param)
    : OperationBase("AllToAllVV2Operation"), param_(param)
{
    if (GetSingleton<Config>().Is310P()) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("AllToAllVV2Operation310p");
    } else {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("AllToAllVV2Operation");
    }
}

AllToAllVV2Operation::~AllToAllVV2Operation() {}

uint32_t AllToAllVV2Operation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t AllToAllVV2Operation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status AllToAllVV2Operation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                            SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).shape.dimNum = 2; // 2：维数
    outTensorDescs.at(0).shape.dims[0] = 1;
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(IN_TENSOR_NUM - 1).shape.dims[0]; // recvCounts的所有元素之和
    return NO_ERROR;
}

int64_t AllToAllVV2CalculateTensorSize(const SVector<Tensor> &tensors)
{
    std::size_t size = tensors.at(0).desc.shape.dimNum;
    int64_t result = 1;
    for (std::size_t i = 0; i < size; ++i) {
        result *= tensors.at(0).desc.shape.dims[i];
    }
    return result;
}

Status AllToAllVV2Operation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    int64_t count = 0;
    int64_t inputCount = AllToAllVV2CalculateTensorSize(inTensors);
    int64_t *sendCounts = static_cast<int64_t *>(inTensors[1].hostData);
    int64_t *sdispls = static_cast<int64_t *>(inTensors[2].hostData);
    int64_t *recvCounts = static_cast<int64_t *>(inTensors[3].hostData);
    int64_t *rdispls = static_cast<int64_t *>(inTensors[4].hostData);
    for (int i = 1; i < IN_TENSOR_NUM - 1; i++) {
        if (param_.rankSize != inTensors.at(i).desc.shape.dims[0]) {
            ATB_LOG(ERROR)
                << "AllToAllVV2Operation param sendCounts,recvCounts,sdispls,rdispls size should be equal to ranksize";
            return ERROR_INVALID_PARAM;
        }
    }
    for (int i = 0; i < param_.rankSize; i++) {
        if (sendCounts[i] < 0 || sdispls[i] < 0 || recvCounts[i] < 0 || rdispls[i] < 0) {
            ATB_LOG(ERROR)
                << "AllToAllVV2Operation param sendCounts,recvCounts,sdispls,rdispls should be more than zero";
            return ERROR_INVALID_PARAM;
        }
    }
    for (int i = 0; i < param_.rankSize; i++) {
        if (count > std::numeric_limits<int64_t>::max() - recvCounts[i]) {
            ATB_LOG(ERROR) << " ,AllToAllVV2Operation sum(recvCounts) will overflow ";
            return ERROR_INVALID_PARAM;
        }
        count += recvCounts[i];
    }
    if (count <= 0) {
        ATB_LOG(ERROR) << "AllToAllVV2Operation sum(recvCounts) should be more than 0";
        return ERROR_INVALID_PARAM;
    }
    for (int i = 0; i < param_.rankSize; i++) {
        if (recvCounts[i] > std::numeric_limits<int64_t>::max() - rdispls[i] || recvCounts[i] + rdispls[i] > count) {
            ATB_LOG(ERROR) << "AllToAllVV2Operation the sum of recvCounts and rdispls is out of bounds";
            return ERROR_INVALID_PARAM;
        }
        if (sendCounts[i] > std::numeric_limits<int64_t>::max() - sdispls[i] ||
            sendCounts[i] + sdispls[i] > inputCount) {
            ATB_LOG(ERROR) << "AllToAllVV2Operation the sum of sendCounts and sdispls is out of bounds";
            return ERROR_INVALID_PARAM;
        }
    }
    if (outTensors.at(0).desc.shape.dims[1] != count) {
        ATB_LOG(ERROR) << "AllToAllVV2Operation outTensor second dimension should be equal to sum(recvCounts)";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> AllToAllVV2Operation::CreateRunner(Context &context) const
{
    (void)context;
    if (param_.backend == "hccl") {
        if (param_.hcclComm == nullptr) {
            return std::make_shared<AllToAllVV2HcclRunner>(param_, !param_.rankTableFile.empty());
        } else {
            return std::make_shared<AllToAllVV2HcclRunner>(param_, param_.hcclComm);
        }
    }
    return std::shared_ptr<Runner>();
}

infer::AllToAllVV2Param AllToAllVV2Operation::GetParam() const
{
    return param_;
}

void AllToAllVV2Operation::SetParam(const infer::AllToAllVV2Param &param)
{
    atb::infer::AllToAllVV2Param newParam = param;
    // -1: 未指定rank号单卡情况，默认指定rank0
    if (param.rank == -1 && param.rankSize == 1) {
        newParam.rank = 0;
    }
    param_ = newParam;
    runner_ = nullptr;
}

nlohmann::json AllToAllVV2Operation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb