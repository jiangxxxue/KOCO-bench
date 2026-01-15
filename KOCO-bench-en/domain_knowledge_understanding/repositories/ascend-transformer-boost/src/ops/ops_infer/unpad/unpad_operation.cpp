/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "unpad_operation.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "unpad_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

static const uint32_t IN_TENSOR_NUM = 4;
static const uint32_t OUT_TENSOR_NUM = 3;
static const uint32_t DIM_2 = 2;
static const uint32_t DIM_3 = 3;

namespace atb {
template <> Status CreateOperation(const infer::UnpadParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new UnpadOperation(opParam);
    return NO_ERROR;
}

UnpadOperation::UnpadOperation(const infer::UnpadParam &param) : OperationBase("UnpadOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("UnpadOperation");
}

UnpadOperation::~UnpadOperation() {}

uint32_t UnpadOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t UnpadOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status UnpadOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                      SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(1);
    outTensorDescs.at(1) = inTensorDescs.at(1);
    outTensorDescs.at(DIM_2) = inTensorDescs.at(1);
    outTensorDescs.at(0).dtype = ACL_INT64;
    outTensorDescs.at(0).shape.dims[0] = 1;
    int64_t dim = 1;
    for (size_t i = 0; i < inTensorDescs.at(0).shape.dimNum; i++) {
        dim *= inTensorDescs.at(0).shape.dims[i];
    }
    outTensorDescs.at(0).shape.dims[1] = dim;
    outTensorDescs.at(DIM_2).shape.dims[0] = 1;
    outTensorDescs.at(DIM_2).shape.dims[1] = dim;
    return NO_ERROR;
}

Status UnpadOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(1).shape.dims[1] != 1 || inTensorDescs.at(DIM_3).shape.dims[1] != 1) {
        ATB_LOG(ERROR) << "intensor2 dim[1] and intensor4 dim[1] should be 1";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(DIM_2).shape.dims[0] != 1 || inTensorDescs.at(DIM_2).shape.dims[1] != 1) {
        ATB_LOG(ERROR) << "intensor3 dim[0] and dim[1] should be 1";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (!TensorCheck::IsTensorDescDimNumValid(inTensorDescs[0], DIM_2) ||
        !TensorCheck::IsTensorDescDimNumValid(inTensorDescs[1], DIM_2) ||
        !TensorCheck::IsTensorDescDimNumValid(inTensorDescs[DIM_2], DIM_2) ||
        !TensorCheck::IsTensorDescDimNumValid(inTensorDescs[DIM_3], DIM_2)) {
        ATB_LOG(ERROR) << "inTensor dim num is not support, inTensor only support 2";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(0).shape.dims[0] != inTensorDescs.at(DIM_3).shape.dims[0] ||
        inTensorDescs.at(1).shape.dims[0] != inTensorDescs.at(DIM_3).shape.dims[0]) {
        ATB_LOG(ERROR) << "input_ids, cum_offsets_now, seq_len should have same batch shape ";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status UnpadOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    if (inTensors.at(1).desc.shape.dims[1] != 1 || inTensors.at(DIM_3).desc.shape.dims[1] != 1) {
        ATB_LOG(ERROR) << "intensor2 dim[1] and intensor4 dim[1] should be 1";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensors.at(DIM_2).desc.shape.dims[0] != 1 || inTensors.at(DIM_2).desc.shape.dims[1] != 1) {
        ATB_LOG(ERROR) << "intensor3 dim[0] and dim[1] should be 1";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (!TensorCheck::IsTensorDescDimNumValid(inTensors[0].desc, DIM_2) ||
        !TensorCheck::IsTensorDescDimNumValid(inTensors[1].desc, DIM_2) ||
        !TensorCheck::IsTensorDescDimNumValid(inTensors[DIM_2].desc, DIM_2) ||
        !TensorCheck::IsTensorDescDimNumValid(inTensors[DIM_3].desc, DIM_2)) {
        ATB_LOG(ERROR) << "inTensor dim num is not support, inTensor only support 2";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensors.at(0).desc.shape.dims[0] != inTensors.at(DIM_3).desc.shape.dims[0] ||
        inTensors.at(1).desc.shape.dims[0] != inTensors.at(DIM_3).desc.shape.dims[0]) {
        ATB_LOG(ERROR) << "input_ids, cum_offsets_now, seq_len should have same batch shape ";
        return ERROR_INVALID_TENSOR_DIM;
    }
    ATB_LOG(DEBUG) << "outTensors size:" << outTensors.size();
    return NO_ERROR;
}

std::shared_ptr<Runner> UnpadOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<UnpadOpsRunner>(param_);
}
} // namespace atb