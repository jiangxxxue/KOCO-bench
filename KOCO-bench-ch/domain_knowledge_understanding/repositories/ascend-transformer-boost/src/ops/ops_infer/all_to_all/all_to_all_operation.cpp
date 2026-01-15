/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_to_all_operation.h"
#include "atb/utils/config.h"
#include "all_to_all_hccl_runner.h"
#include "all_to_all_lccl_runner.h"
#include "atb/utils.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const int32_t IN_TENSOR_NUM = 1;
static const int32_t OUT_TENSOR_NUM = 1;
static const int32_t TRANSPOSE_IN_TENSOR_DIM_NUM = 2;
static const int64_t MAX_W_SIZE = 90LL * 1024LL;                 // 90KB
static const uint64_t MAX_TENSOR_SIZE = 190LL * 1024LL * 1024LL; // 190MB

template <> Status CreateOperation(const infer::AllToAllParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (opParam.backend != "hccl" && opParam.backend != "lccl") {
        ATB_LOG(ERROR) << "backend is " << opParam.backend << "backend must be hccl or lccl";
        return ERROR_INVALID_PARAM;
    }
    const char *socName = aclrtGetSocName();
    if (!socName) {
        ATB_LOG(ERROR) << "aclrtGetSocName failed!";
        return false;
    }
    if (opParam.backend == "lccl" && !opParam.transpose &&
        std::string(socName).find("Ascend910_93") == std::string::npos) {
        ATB_LOG(ERROR) << "AllToAll lccl without transpose only supports Atlas 800T A3 or Atlas 900 A3 Superpod";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.backend == "hccl") {
        if (!GetSingleton<Config>().Is910B() && std::string(socName).find("Ascend910_93") == std::string::npos) {
            ATB_LOG(ERROR) << "AllToAll hccl only supports Atlas 800I A2/A3 or Atlas 900 A3 Superpod";
            return ERROR_INVALID_PARAM;
        }
        if (opParam.transpose) {
            ATB_LOG(ERROR) << "AllToAll hccl doesn't support transpose";
            return ERROR_INVALID_PARAM;
        }
    }
    if (OperationUtil::DistributedInitCheck<infer::AllToAllParam>(opParam) != NO_ERROR) {
        ATB_LOG(ERROR) << "AllToAllOperation DistributedInitCheck failed";
        return ERROR_INVALID_PARAM;
    }
        if (opParam.backend == "lccl" && opParam.rankSize % 2 != 0) { // 2 : Even ranksize
        ATB_LOG(ERROR) << "AllToAll lccl only supports even ranksize";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) AllToAllOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new AllToAllOperation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

AllToAllOperation::AllToAllOperation(const infer::AllToAllParam &param)
    : OperationBase("AllToAllOperation"), param_(param)
{
    std::string opKey = "AllToAllOperation";
    if (param.transpose) {
        opKey += "Transpose";
    }
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(opKey);
}

AllToAllOperation::~AllToAllOperation() {}

uint32_t AllToAllOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t AllToAllOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status AllToAllOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (!param_.transpose) {
        return NO_ERROR;
    }
    if (inTensorDescs.at(0).shape.dimNum != TRANSPOSE_IN_TENSOR_DIM_NUM) { // 2: transpose only support dimNum
        ATB_LOG(ERROR) << "inTensor[0] dimNum should be " << TRANSPOSE_IN_TENSOR_DIM_NUM
                       << ", but got: " << inTensorDescs.at(0).shape.dimNum;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(0).shape.dims[1] % param_.rankSize != 0) {
        ATB_LOG(ERROR) << "intensors[0].dims[0] must be an integer multiple of ranksize but got dims[0]: "
                       << inTensorDescs.at(0).shape.dims[1] << ", rankSize: " << param_.rankSize;
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t wSize = inTensorDescs.at(0).shape.dims[TRANSPOSE_IN_TENSOR_DIM_NUM - 1] *
                    static_cast<int64_t>(sizeof(inTensorDescs.at(0).dtype));
    if (wSize / param_.rankSize >= MAX_W_SIZE) {
        ATB_LOG(ERROR) << "intensors[0].dims[1] / rankSize must be no greater than 90K, but got bytes: " << wSize;
        return ERROR_INVALID_TENSOR_DIM;
    }
    uint64_t tensorSize = Utils::GetTensorSize(inTensorDescs.at(0));
    if (tensorSize > MAX_TENSOR_SIZE) {
        ATB_LOG(ERROR) << "intensors[0] total tensor size must be no greater than 190MB, but got bytes: " << tensorSize;
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status AllToAllOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                         SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    if (param_.backend == "lccl" && param_.transpose) {
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0] * param_.rankSize;
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1] / param_.rankSize;
    }
    return 0;
}

Status AllToAllOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    Status st = InferShapeCheckImpl(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    if (!param_.transpose && !TensorUtil::TensorDescEqual(inTensors.at(0).desc, outTensors.at(0).desc)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "intensor desc and outtensor desc should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (param_.transpose) {
        if (inTensors.at(0).desc.shape.dimNum != TRANSPOSE_IN_TENSOR_DIM_NUM) {
            ATB_LOG(ERROR) << "invalid inTensor dimNum, should be 2, but got inTensors[0] dimNum: "
                           << inTensors.at(0).desc.shape.dimNum;
            return ERROR_INVALID_TENSOR_DIM_NUM;
        }
        if (outTensors.at(0).desc.shape.dimNum != TRANSPOSE_IN_TENSOR_DIM_NUM) {
            ATB_LOG(ERROR) << "invalid outTensor dimNum, should be 2, but got outTensors[0] dimNum: "
                           << outTensors.at(0).desc.shape.dimNum;
            return ERROR_INVALID_TENSOR_DIM_NUM;
        }
        if (outTensors.at(0).desc.shape.dims[0] != inTensors.at(0).desc.shape.dims[0] * param_.rankSize) {
            ATB_LOG(ERROR) << "invalid outTensor dims[0] should be intensors[0].dims[0], * rankSize, i.e. "
                           << inTensors.at(0).desc.shape.dims[0] << " * " << param_.rankSize << ", but got "
                           << outTensors.at(0).desc.shape.dims[0];
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (outTensors.at(0).desc.shape.dims[1] * param_.rankSize != inTensors.at(0).desc.shape.dims[1]) {
            ATB_LOG(ERROR) << "invalid outTensor dims[1], should be intensors[0].dims[1]/rankSize, i.e. "
                           << inTensors.at(0).desc.shape.dims[1] << " / " << param_.rankSize << ", but got "
                           << outTensors.at(0).desc.shape.dims[1];
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> AllToAllOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (param_.backend == "hccl") {
        if (param_.hcclComm == nullptr) {
            return std::make_shared<AllToAllHcclRunner>(param_, !param_.rankTableFile.empty());
        } else {
            return std::make_shared<AllToAllHcclRunner>(param_, param_.hcclComm);
        }
    } else if (param_.backend == "lccl") {
        return std::make_shared<AllToAllLcclRunner>(param_, context);
    }
    return std::shared_ptr<Runner>();
}

nlohmann::json AllToAllOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb
