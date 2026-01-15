/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kv_cache_operation.h"
#include "atb/utils/config.h"
#include "kv_cache_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
template <> Status CreateOperation(const infer::KvCacheParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new KvCacheOperation(opParam);
    return NO_ERROR;
}

KvCacheOperation::KvCacheOperation(const infer::KvCacheParam &param) : OperationBase("KvCacheOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("KvCacheOperation");
}

KvCacheOperation::~KvCacheOperation() {}

uint32_t KvCacheOperation::GetInputNum() const
{
    const uint32_t inTensorNum = 5;
    return inTensorNum;
}

uint32_t KvCacheOperation::GetOutputNum() const
{
    return 0;
}

Status KvCacheOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                        SVector<TensorDesc> &outTensorDescs) const
{
    ATB_LOG(INFO) << GetLogPrefix() << "inTensorDescs Size:" << inTensorDescs.size()
                  << "outTensorDescs Size:" << outTensorDescs.size();
    return NO_ERROR;
}

Status KvCacheOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    Status st =
        GetSingleton<Config>().Is910B() ? InferShapeDimCheck(inTensorDescs) : InferShapeDimCheck310P(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

Status KvCacheOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    Status st = GetSingleton<Config>().Is910B() ? SetupDimCheck(inTensors) : SetupDimCheck310P(inTensors);
    if (st != NO_ERROR) {
        return st;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "outTensor Size:" << outTensors.size();
    return NO_ERROR;
}

Status KvCacheOperation::InferShapeDimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    if ((inTensorDescs.at(0).shape.dimNum != 2 &&  // 2: 2 dims
         inTensorDescs.at(0).shape.dimNum != 4) || // 4: 4 dims
        inTensorDescs.at(1).shape.dimNum != 1 ||   // 1: layerId 1: 1 dim
        inTensorDescs.at(2).shape.dimNum != 4 ||   // 2: past 4: 4 dims
        inTensorDescs.at(3).shape.dimNum != 1 ||   // 3: tokenOffest 1: 1 dim
        inTensorDescs.at(4).shape.dimNum != 1) {   // 4: seqLen 1: 1 dim
        ATB_LOG(ERROR) << "invalid intensor dimNums";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    int64_t hiddenSize = inTensorDescs.at(2).shape.dims[3]; // 2: past 3: 3rd dim
    if (inTensorDescs.at(2).dtype == ACL_FLOAT16) {         // 2: past
        if (hiddenSize % 16 != 0) {                         // 16: memory alignment
            ATB_LOG(ERROR) << "hiddenSizes a should be a multiple of 16";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (inTensorDescs.at(2).dtype == ACL_INT8) { // 2: past
        if (hiddenSize % 32 != 0) {              // 32: memory alignment
            ATB_LOG(ERROR) << "hiddenSizes should be a multiple of 32";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    int64_t batch = inTensorDescs.at(2).shape.dims[1];         // 2: past 1: 1st dim
    if (inTensorDescs.at(0).shape.dimNum == 2) {               // 2: new_kv 2维
        if (hiddenSize != inTensorDescs.at(0).shape.dims[1]) { // 1: 1st dim
            ATB_LOG(ERROR) << "hiddenSizes should be same";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (inTensorDescs.at(0).shape.dimNum == 4) {                   // 2: new_kv 4维
        if (hiddenSize != inTensorDescs.at(0).shape.dims[2] *      // 2: 2nd dim
                              inTensorDescs.at(0).shape.dims[3]) { // 3: 3rd dim
            ATB_LOG(ERROR) << "hiddenSizes should be same";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (batch != inTensorDescs.at(3).shape.dims[0] || // 3: tokenOffset
        batch != inTensorDescs.at(4).shape.dims[0]) { // 4: seqLen
        ATB_LOG(ERROR) << "batches should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status KvCacheOperation::SetupDimCheck(const SVector<Tensor> &inTensors) const
{
    if ((inTensors.at(0).desc.shape.dimNum != 2 &&  // 2: 2 dims
         inTensors.at(0).desc.shape.dimNum != 4) || // 4: 4 dims
        inTensors.at(1).desc.shape.dimNum != 1 ||   // 1: layerId 1: 1 dim
        inTensors.at(2).desc.shape.dimNum != 4 ||   // 2: past 4: 4 dims
        inTensors.at(3).desc.shape.dimNum != 1 ||   // 3: tokenOffest 1: 1 dim
        inTensors.at(4).desc.shape.dimNum != 1) {   // 4: seqLen 1: 1 dim
        ATB_LOG(ERROR) << "invalid intensor dimNums";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t hiddenSize = inTensors.at(2).desc.shape.dims[3]; // 2: past 3: 3rd dim
    if (inTensors.at(2).desc.dtype == ACL_FLOAT16) {         // 2: past
        if (hiddenSize % 16 != 0) {                          // 16: memory alignment
            ATB_LOG(ERROR) << "hiddenSizes should be a multiple of 16";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (inTensors.at(2).desc.dtype == ACL_INT8) { // 2: past
        if (hiddenSize % 32 != 0) {               // 32: memory alignment
            ATB_LOG(ERROR) << "hiddenSizes should be a multiple of 32";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    int64_t batch = inTensors.at(2).desc.shape.dims[1];         // 2: past 1: 1st dim
    if (inTensors.at(0).desc.shape.dimNum == 2) {               // 2: new_kv 2维
        if (hiddenSize != inTensors.at(0).desc.shape.dims[1]) { // 1: 1st dim
            ATB_LOG(ERROR) << "hiddenSizes should be same";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (inTensors.at(0).desc.shape.dimNum == 4) {                   // 2: new_kv 4维
        if (hiddenSize != inTensors.at(0).desc.shape.dims[2] *      // 2: 2nd dim
                              inTensors.at(0).desc.shape.dims[3]) { // 3: 3rd dim
            ATB_LOG(ERROR) << "hiddenSizes should be same";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (batch != inTensors.at(3).desc.shape.dims[0] || // 3: tokenOffset
        batch != inTensors.at(4).desc.shape.dims[0]) { // 4: seqLen
        ATB_LOG(ERROR) << "batches should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status KvCacheOperation::InferShapeDimCheck310P(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(0).shape.dimNum != 4 || // 0: newKv 4: 4 dims
        inTensorDescs.at(1).shape.dimNum != 1 || // 1: layerId 1: 1 dim
        inTensorDescs.at(2).shape.dimNum != 5 || // 2: past 5: 5 dims
        inTensorDescs.at(3).shape.dimNum != 1 || // 3: tokenOffest 1: 1 dim
        inTensorDescs.at(4).shape.dimNum != 1) { // 4: seqLen 1: 1 dim
        ATB_LOG(ERROR) << "invalid intensor dimNums";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t hiddenSize = inTensorDescs.at(0).shape.dims[1]; // 0: newKv 1: 1st dim hiddenSize//16
    if (inTensorDescs.at(0).dtype == ACL_FLOAT16) {         // 0: newKv
        if (hiddenSize % 16 != 0) {                         // 16: memory alignment
            ATB_LOG(ERROR) << "hiddenSizes should be a multiple of 16";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (inTensorDescs.at(0).dtype == ACL_INT8) { // 0: newKv
        if (hiddenSize % 32 != 0) {              // 32: memory alignment
            ATB_LOG(ERROR) << "hiddenSizes should be a multiple of 32";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (hiddenSize != inTensorDescs.at(2).shape.dims[2]) { // 2: past 2: 3nd dim
        ATB_LOG(ERROR) << "hiddenSizes should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t batch = inTensorDescs.at(2).shape.dims[1]; // 2: past 1: 1st dim
    if (batch != inTensorDescs.at(3).shape.dims[0] ||  // 3: tokenOffset
        batch != inTensorDescs.at(4).shape.dims[0]) {  // 4: seqLen
        ATB_LOG(ERROR) << "batches should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status KvCacheOperation::SetupDimCheck310P(const SVector<Tensor> &inTensors) const
{
    if (inTensors.at(0).desc.shape.dimNum != 4 || // 0: newKv 4: 4 dims
        inTensors.at(1).desc.shape.dimNum != 1 || // 1: layerId 1: 1 dim
        inTensors.at(2).desc.shape.dimNum != 5 || // 2: past 5: 5 dims
        inTensors.at(3).desc.shape.dimNum != 1 || // 3: tokenOffest 1: 1 dim
        inTensors.at(4).desc.shape.dimNum != 1) { // 4: seqLen 1: 1 dim
        ATB_LOG(ERROR) << "invalid intensor dimNums";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t hiddenSize = inTensors.at(0).desc.shape.dims[1]; // 0: newKv 1: 1st dim hiddenSize//16
    if (inTensors.at(0).desc.dtype == ACL_FLOAT16) {         // 0: newKv
        if (hiddenSize % 16 != 0) {                          // 16: memory alignment
            ATB_LOG(ERROR) << "hiddenSizes should be a multiple of 16";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (inTensors.at(0).desc.dtype == ACL_INT8) { // 0: newKv
        if (hiddenSize % 32 != 0) {               // 32: memory alignment
            ATB_LOG(ERROR) << "hiddenSizes should be a multiple of 32";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (hiddenSize != inTensors.at(2).desc.shape.dims[2]) { // 2: past 2: 3nd dim
        ATB_LOG(ERROR) << "hiddenSizes should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t batch = inTensors.at(2).desc.shape.dims[1]; // 2: past 1: 1st dim
    if (batch != inTensors.at(3).desc.shape.dims[0] ||  // 3: tokenOffset
        batch != inTensors.at(4).desc.shape.dims[0]) {  // 4: seqLen
        ATB_LOG(ERROR) << "batches should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> KvCacheOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<KvCacheOpsRunner>(param_);
}
} // namespace atb