/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce_operation.h"
#include "atb/utils/config.h"
#include "all_reduce_hccl_runner.h"
#include "all_reduce_lccl_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const int32_t IN_TENSOR_NUM = 1;
static const int32_t OUT_TENSOR_NUM = 1;
static const int32_t IN_TENSOR_NUM_WITH_QUANT = 3;

Status CheckAllReduceQuantParam(const infer::AllReduceParam &opParam)
{
    if (opParam.backend == "lccl" && opParam.quantType != atb::infer::AllReduceParam::QUANT_TYPE_UNQUANT &&
        opParam.allReduceType != "sum") {
        ATB_LOG(ERROR) << "allReduceType is " << opParam.allReduceType << "AllReduce Quant allReduceType must be sum";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.backend == "hccl" && opParam.quantType != atb::infer::AllReduceParam::QUANT_TYPE_UNQUANT) {
        ATB_LOG(ERROR) << "quantType: " << opParam.quantType << "  Hccl is not support quant";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.outDataType != ACL_FLOAT16 && opParam.quantType != atb::infer::AllReduceParam::QUANT_TYPE_UNQUANT) {
        ATB_LOG(ERROR) << "outDataType: " << opParam.outDataType << " is not support quant";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.backend == "lccl" && (opParam.quantType < atb::infer::AllReduceParam::QUANT_TYPE_UNQUANT ||
                                      opParam.quantType >= atb::infer::AllReduceParam::QUANT_TYPE_MAX)) {
        ATB_LOG(ERROR) << "quantType: " << opParam.quantType << " is invalid QuantType";
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

Status CheckAllReduceParamValidity(const infer::AllReduceParam &opParam)
{
    if (CheckAllReduceQuantParam(opParam) != NO_ERROR) {
        return ERROR_INVALID_PARAM;
    }
    if (opParam.backend != "hccl" && opParam.backend != "lccl") {
        ATB_LOG(ERROR) << "backend is " << opParam.backend << "backend must either be hccl or lccl";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.backend == "lccl" && GetSingleton<Config>().Is310P()) {
        ATB_LOG(ERROR) << "AllReduce lccl is not support in Atlas inference products";
        return ERROR_INVALID_PARAM;
    }
    if (OperationUtil::DistributedInitCheck<infer::AllReduceParam>(opParam) != NO_ERROR) {
        ATB_LOG(ERROR) << "AllReduceOperation DistributedInitCheck failed";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.allReduceType != "sum" && opParam.allReduceType != "prod" && opParam.allReduceType != "max" &&
        opParam.allReduceType != "min") {
        ATB_LOG(ERROR) << "allReduceType is " << opParam.allReduceType
                       << "allReduceParam allReduceType must be one of the following sum, prod, max, min";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.backend == "lccl" && opParam.allReduceType == "prod") {
        ATB_LOG(ERROR) << "allReduceType is " << opParam.allReduceType << "lccl not support prod ";
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::AllReduceParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (CheckAllReduceParamValidity(opParam) != NO_ERROR) {
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) AllReduceOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

AllReduceOperation::AllReduceOperation(const infer::AllReduceParam &param)
    : OperationBase("AllReduceOperation"), param_(param)
{
    if (param_.quantType == atb::infer::AllReduceParam::QUANT_TYPE_PER_TENSOR ||
        param_.quantType == atb::infer::AllReduceParam::QUANT_TYPE_PER_CHANNEL) {
        operationIr_ = atb::GetSingleton<AtbOperationIrCfg>().GetOperationIr("AllReduceOperationQuant");
    } else {
        operationIr_ = atb::GetSingleton<AtbOperationIrCfg>().GetOperationIr("AllReduceOperation");
    }
}

AllReduceOperation::~AllReduceOperation() {}

uint32_t AllReduceOperation::GetInputNum() const
{
    if (param_.quantType == atb::infer::AllReduceParam::QUANT_TYPE_PER_TENSOR ||
        param_.quantType == atb::infer::AllReduceParam::QUANT_TYPE_PER_CHANNEL) {
        return IN_TENSOR_NUM_WITH_QUANT;
    }
    return IN_TENSOR_NUM;
}

uint32_t AllReduceOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status AllReduceOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                          SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    if (param_.quantType == atb::infer::AllReduceParam::QUANT_TYPE_PER_TENSOR ||
        param_.quantType == atb::infer::AllReduceParam::QUANT_TYPE_PER_CHANNEL) {
        outTensorDescs.at(0).dtype = param_.outDataType;
    }
    return NO_ERROR;
}

Status AllReduceOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (DtypeCheck(inTensorDescs.at(0)) != NO_ERROR) {
        return ERROR_INVALID_TENSOR_DTYPE;
    }
    int n = inTensorDescs.at(0).shape.dims[inTensorDescs.at(0).shape.dimNum - 1];
    if (inTensorDescs.size() > 1) {
        const int32_t offsetPosition = 1;
        const int32_t scalePosition = 2;
        return QuantShapeCheck(inTensorDescs.at(offsetPosition), inTensorDescs.at(scalePosition), n);
    }
    return NO_ERROR;
}

Status AllReduceOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    if (DtypeCheck(inTensors.at(0).desc) != NO_ERROR) {
        return ERROR_INVALID_TENSOR_DTYPE;
    }
    ATB_LOG(DEBUG) << "outTensors size:" << outTensors.size();
    if (!TensorUtil::TensorShapeEqual(inTensors.at(0).desc.shape, outTensors.at(0).desc.shape)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "intensor's shape and outtensor's shape should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int n = inTensors.at(0).desc.shape.dims[inTensors.at(0).desc.shape.dimNum - 1];
    if (inTensors.size() > 1) {
        const int32_t offsetPosition = 1; // 1: offsetPos
        const int32_t scalePosition = 2;  // 2: scalePos
        return QuantShapeCheck(inTensors.at(offsetPosition).desc, inTensors.at(scalePosition).desc, n);
    }
    return NO_ERROR;
}

Status AllReduceOperation::DtypeCheck(const TensorDesc &inTensorDesc) const
{
    if (param_.backend == "lccl" && inTensorDesc.dtype == ACL_INT64) {
        ATB_LOG(ERROR) << "lccl allreduce is not support int64";
        return ERROR_INVALID_TENSOR_DTYPE;
    }
    if (param_.backend == "hccl" && GetSingleton<Config>().Is910A()) {
        if (inTensorDesc.dtype == ACL_INT16 || inTensorDesc.dtype == ACL_BF16) {
            ATB_LOG(ERROR) << "Atlas inference products hccl allreduce is not support int16 , bf16";
            return ERROR_INVALID_TENSOR_DTYPE;
        }
    }
    if (param_.backend == "hccl" && param_.allReduceType == "prod" && GetSingleton<Config>().Is910B()) {
        if (inTensorDesc.dtype == ACL_INT16 || inTensorDesc.dtype == ACL_BF16) {
            ATB_LOG(ERROR) << "Atlas A2 inference products hccl allreduce prod is not support int16 , bf16";
            return ERROR_INVALID_TENSOR_DTYPE;
        }
    }
    if (param_.backend == "hccl" && GetSingleton<Config>().Is310P()) {
        if (inTensorDesc.dtype == ACL_INT64 || inTensorDesc.dtype == ACL_BF16) {
            ATB_LOG(ERROR) << " Atlas 300I Duo inference products hccl allreduce is not support int64 , bf16";
            return ERROR_INVALID_TENSOR_DTYPE;
        }
        if (param_.allReduceType == "max" || param_.allReduceType == "min" || param_.allReduceType == "prod") {
            if (inTensorDesc.dtype == ACL_INT16) {
                ATB_LOG(ERROR)
                    << " Atlas 300I Duo inference products hccl allreducetype max,min,prod is not support int16";
                return ERROR_INVALID_TENSOR_DTYPE;
            }
        }
    }
    return NO_ERROR;
}

Status AllReduceOperation::QuantShapeCheck(const TensorDesc &scale, const TensorDesc &offset, int n) const
{
    const uint64_t lastDimNum = 16;
    // input张量的最后一维(n)大小必须为16的整数倍
    if (n % lastDimNum != 0) {
        ATB_LOG(ERROR) << "quantType: " << param_.quantType << "input's last dimension or is not a multiple of 16";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (offset.shape.dimNum != 1 || offset.shape.dims[0] != 1) {
        ATB_LOG(ERROR) << "quantType: " << param_.quantType << "offset shape must be 1";
        return ERROR_INVALID_TENSOR_DIM;
    }
    const int32_t scaleMaxDimNum = 2;
    if (scale.shape.dimNum > scaleMaxDimNum) {
        ATB_LOG(ERROR) << "quantType: " << param_.quantType << "scale dimNum is not support";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (param_.quantType == atb::infer::AllReduceParam::QUANT_TYPE_PER_TENSOR) {
        if (scale.shape.dimNum != 1 || scale.shape.dims[0] != 1) {
            ATB_LOG(ERROR) << "quantType: " << param_.quantType << "scale shape must be 1";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (param_.quantType == atb::infer::AllReduceParam::QUANT_TYPE_PER_CHANNEL) {
        if (n > 4194304) { // 4194304: 限制通道数最大值
            ATB_LOG(ERROR) << GetLogPrefix() << "quant channel num: " << n << " should be no more than 4194304(2^22)";
            return ERROR_INVALID_TENSOR_DIM;
        }
        // scale形状为[1,n]
        const int32_t scaleShapeNum = 2;
        if (scale.shape.dimNum == scaleShapeNum &&
            (scale.shape.dims[scale.shape.dimNum - 1] != n || scale.shape.dims[0] != 1)) {
            ATB_LOG(ERROR) << "quantType: " << param_.quantType
                           << " scale Num is 2  input dimension must be equal and scale shape must be [1,n]";
            return ERROR_INVALID_TENSOR_DIM;
        }
        // scale形状为[n]
        if (scale.shape.dimNum == 1 && scale.shape.dims[scale.shape.dimNum - 1] != n) {
            ATB_LOG(ERROR) << "quantType: " << param_.quantType << "scale Num is 1 scale shape must be n";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> AllReduceOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (param_.backend == "hccl") {
        if (param_.hcclComm == nullptr) {
            return std::make_shared<AllReduceHcclRunner>(param_, !param_.rankTableFile.empty());
        } else {
            return std::make_shared<AllReduceHcclRunner>(param_, param_.hcclComm);
        }
    } else if (param_.backend == "lccl") {
        return std::make_shared<AllReduceLcclRunner>(param_, context);
    }
    return std::shared_ptr<Runner>();
}

nlohmann::json AllReduceOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb