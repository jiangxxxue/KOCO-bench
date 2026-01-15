/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "faupdate_operation.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/config.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "faupdate_ops_runner.h"
#include "atb/operation/op_param_funcs.h"


static const uint32_t SP_MIN = 1;
static const uint32_t SP_MAX = 16;
static const int32_t IN_TENSOR_NUM = 2;
static const int32_t OUT_TENSOR_NUM = 1;
static const uint32_t IN_TENSOR_0 = 0;
static const uint32_t IN_TENSOR_1 = 1;
static const uint32_t OUT_TENSOR_0 = 0;
static const uint64_t DIM_NUM_2 = 2;
static const uint64_t DIM_NUM_3 = 3;
static const int64_t DIM_INDEX_0 = 0;
static const int64_t DIM_INDEX_1 = 1;
static const int64_t DIM_INDEX_2 = 2;
static const int64_t HEAD_DIM_MIN = 8;
static const int64_t HEAD_DIM_MAX = 512;
static const int64_t ALIGNED_TO_8 = 8;

namespace atb {
template <> Status CreateOperation(const infer::FaUpdateParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    ATB_LOG(INFO) << "CreateOperation FaUpdateParam faUpdateType: " << opParam.faUpdateType << ", sp: " << opParam.sp;
    if (opParam.faUpdateType < infer::FaUpdateParam::DECODE_UPDATE ||
        opParam.faUpdateType > infer::FaUpdateParam::DECODE_UPDATE) {
        ATB_LOG(ERROR) << "param faUpdateType should be DECODE_UPDATE";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.faUpdateType == infer::FaUpdateParam::DECODE_UPDATE) {
        if (!GetSingleton<Config>().Is910B()) {
            ATB_LOG(ERROR) << "DECODE_UPDATE only support Atlas 800I A2/A3 inference product";
            return ERROR_INVALID_PARAM;
        }
        if (opParam.sp < SP_MIN || opParam.sp > SP_MAX) {
            ATB_LOG(ERROR) << "param sp should be in range: [" << SP_MIN << " - " << SP_MAX << "].";
            return ERROR_INVALID_PARAM;
        }
    }

    *operation = new (std::nothrow) FaUpdateOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_INTERNAL_ERROR;
    }
    return NO_ERROR;
}

FaUpdateOperation::FaUpdateOperation(const infer::FaUpdateParam &param)
    : OperationBase("FaUpdateOperation"), param_(param)
{
    if (param_.faUpdateType == infer::FaUpdateParam::DECODE_UPDATE) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("FaUpdateOperation");
    }
}

FaUpdateOperation::~FaUpdateOperation() {}

uint32_t FaUpdateOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t FaUpdateOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status FaUpdateOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                         SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(IN_TENSOR_0);
    outTensorDescs.at(0).shape.dims[DIM_INDEX_0] = inTensorDescs.at(IN_TENSOR_1).shape.dims[DIM_INDEX_1];
    outTensorDescs.at(0).shape.dims[DIM_INDEX_1] = inTensorDescs.at(IN_TENSOR_1).shape.dims[DIM_INDEX_2];
    return NO_ERROR;
}

Status FaUpdateOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return InTensorDescsCheck(inTensorDescs);
}

Status FaUpdateOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    Status status = InTensorDescsCheck(inTensorDescs);
    if (status != NO_ERROR) {
        return status;
    }

    TensorDesc goTensorDesc = inTensors.at(IN_TENSOR_1).desc;
    TensorDesc outTensorDesc = outTensors.at(OUT_TENSOR_0).desc;
    if (outTensorDesc.shape.dims[DIM_INDEX_0] != goTensorDesc.shape.dims[DIM_INDEX_1] ||
        outTensorDesc.shape.dims[DIM_INDEX_1] != goTensorDesc.shape.dims[DIM_INDEX_2]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "output Tensor shape error";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

nlohmann::json FaUpdateOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

std::shared_ptr<Runner> FaUpdateOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<FaUpdateOpsRunner>(param_);
}

Status FaUpdateOperation::InTensorDescsCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.faUpdateType == infer::FaUpdateParam::DECODE_UPDATE) {
        return DecodeUpdateInTensorDescsCheck(inTensorDescs);
    }
    return NO_ERROR;
}

Status FaUpdateOperation::DecodeUpdateInTensorDescsCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    TensorDesc lseTensorDesc = inTensorDescs.at(IN_TENSOR_0);
    TensorDesc goTensorDesc = inTensorDescs.at(IN_TENSOR_1);

    //  inTensor0 lse
    if (!TensorCheck::IsTensorDescDimNumValid(lseTensorDesc, DIM_NUM_2)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 dimNum should be 2 but get " << lseTensorDesc.shape.dimNum;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    int64_t lseTensorDim0 = lseTensorDesc.shape.dims[DIM_INDEX_0];
    int64_t lseTensorDim1 = lseTensorDesc.shape.dims[DIM_INDEX_1];
    //  inTensor1 go
    if (!TensorCheck::IsTensorDescDimNumValid(goTensorDesc, DIM_NUM_3)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor1 dimNum should be 3 but get " << goTensorDesc.shape.dimNum;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    int64_t goTensorDim0 = goTensorDesc.shape.dims[DIM_INDEX_0];
    int64_t goTensorDim1 = goTensorDesc.shape.dims[DIM_INDEX_1];
    int64_t goTensorDim2 = goTensorDesc.shape.dims[DIM_INDEX_2];
    //  check dim0
    if (lseTensorDim0 != param_.sp || goTensorDim0 != param_.sp) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 dim0 and inTensor1 dim0 must be equal to sp";
        return ERROR_INVALID_TENSOR_DIM;
    }
    //  check dim1
    if (lseTensorDim1 != goTensorDim1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 dim1 and inTensor1 dim1 must be equal";
        return ERROR_INVALID_TENSOR_DIM;
    }
    //  check inTensor1 dim2 (headDim)
    if (goTensorDim2 < HEAD_DIM_MIN || goTensorDim2 > HEAD_DIM_MAX || goTensorDim2 % ALIGNED_TO_8 != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor1 dim2 should be in range [8, 512], and should be aligned to 8";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}
} // namespace atb