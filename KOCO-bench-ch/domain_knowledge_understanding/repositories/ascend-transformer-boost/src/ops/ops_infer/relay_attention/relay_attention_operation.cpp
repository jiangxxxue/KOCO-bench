/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "relay_attention_operation.h"
#include "relay_attention_ops_runner.h"

#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/config.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

static const uint32_t IN_TENSOR_NUM = 10;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint32_t MAX_BATCH = 60;
static const int32_t MAX_KVHEADNUM = 8;

namespace atb {

template <> Status CreateOperation(const infer::RelayAttentionParam &opParam, Operation **operation)
{
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "RelayAttention only support Atlas 800I A2 inference product";
        return ERROR_INVALID_PARAM;
    }
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (opParam.headNum <= 0) {
        ATB_LOG(ERROR) << "headNum should be greater than zero!";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.kvHeadNum < 0 || opParam.kvHeadNum > MAX_KVHEADNUM) {
        ATB_LOG(ERROR) << "The value of kvHeadNum ranges from 0 to 8!";
        return ERROR_INVALID_PARAM;
    }
    if (!(opParam.kvHeadNum == 0 ||
          ((opParam.kvHeadNum <= opParam.headNum) && (opParam.headNum % opParam.kvHeadNum == 0)))) {
        ATB_LOG(ERROR) << "If kvHeadNum is not 0, the value must be less than or equal to headNum "
                       << "and the value of headNum must be an integer multiple of kvHeadNum.";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.maskType != infer::RelayAttentionParam::MASK_TYPE_UNDEFINED) {
        ATB_LOG(ERROR) << "maskType should be set as MASK_TYPE_UNDEFINED!";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) RelayAttentionOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

RelayAttentionOperation::RelayAttentionOperation(const infer::RelayAttentionParam &param)
    : OperationBase("RelayAttentionOperation"), param_(param)
{
    InitOpIni();
    ATB_LOG(INFO) << GetLogPrefix() << "RelayAttentionParam headNum: " << param.headNum
                  << ", qkScale: " << param.qkScale << ", kvHeadNum: " << param.kvHeadNum
                  << ", maskType: " << param.maskType;
}

RelayAttentionOperation::~RelayAttentionOperation() {}

uint32_t RelayAttentionOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t RelayAttentionOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status RelayAttentionOperation::InferShapeDimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t batchQuery = inTensorDescs.at(0).shape.dims[0];      // 0:Query
    int64_t batchKey = inTensorDescs.at(1).shape.dims[0];        // 1:Key
    int64_t batchValue = inTensorDescs.at(2).shape.dims[0];      // 2:Value
    int64_t batchSeqLen = inTensorDescs.at(6).shape.dims[0];     // 6:SeqLen
    int64_t batchKvSeqLen = inTensorDescs.at(7).shape.dims[0];   // 7:KvSeqLen
    int64_t batchKvShareMap = inTensorDescs.at(8).shape.dims[0]; // 8:KvShareMap
    if (batchKey != batchValue || batchKey != batchSeqLen || batchKey != batchKvSeqLen || batchKey != batchKvShareMap ||
        batchKey != batchQuery) {
        ATB_LOG(ERROR) << GetLogPrefix() << "first dim of query, key, value, seqLen, kvSeqLen and kvShareMap "
                       << "should be equal";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (batchQuery > MAX_BATCH) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Batch must be less than or equal to 60.";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t sharedGroupsCountKeyShare = inTensorDescs.at(3).shape.dims[0];   // 3:KeyShare
    int64_t sharedGroupsCountValueShare = inTensorDescs.at(4).shape.dims[0]; // 4:ValueShare
    int64_t sharedGroupsCountkvShareLen = inTensorDescs.at(9).shape.dims[0]; // 9:kvShareLen
    if (sharedGroupsCountKeyShare != sharedGroupsCountValueShare ||
        sharedGroupsCountKeyShare != sharedGroupsCountkvShareLen) {
        ATB_LOG(ERROR) << GetLogPrefix() << "first dim of KeyShare, ValueShare, kvShareLen should be equal";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (!InferShapeDimCheckKVShare(inTensorDescs) || !InferShapeDimCheckKV(inTensorDescs)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

bool RelayAttentionOperation::InferShapeDimCheckKV(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t sharedGroupsLenKeyShare = inTensorDescs.at(3).shape.dims[1]; // 3:KeyShare
    if (sharedGroupsLenKeyShare != inTensorDescs.at(4).shape.dims[1]) {  // 4:ValueShare
        ATB_LOG(ERROR) << GetLogPrefix() << "second dim of KeyShare, ValueShare should be equal";
        return false;
    }
    int64_t unSharedGroupsLenKey = inTensorDescs.at(1).shape.dims[1]; // 1:Key
    if (unSharedGroupsLenKey != inTensorDescs.at(2).shape.dims[1]) {  // 2:Value
        ATB_LOG(ERROR) << GetLogPrefix() << "second dim of Key, Value should be equal";
        return false;
    }
    int64_t headNumKey = inTensorDescs.at(1).shape.dims[2]; // 1:Key
    if (headNumKey != inTensorDescs.at(2).shape.dims[2]) {  // 2:Value
        ATB_LOG(ERROR) << GetLogPrefix() << "3rd dim of Key, Value should be equal";
        return false;
    }
    if (inTensorDescs.at(1).shape.dimNum == 4 &&                // 1:Key; 4:[B,S,N,D];
        inTensorDescs.at(2).shape.dimNum == 4) {                // 2:Value; 4:[B,S,N,D]
        int64_t headDimKey = inTensorDescs.at(1).shape.dims[3]; // 1:Key
        if (headDimKey != inTensorDescs.at(2).shape.dims[3]) {  // 2:Value; 3:4th dim
            ATB_LOG(ERROR) << GetLogPrefix() << "4th dim of Key, Value should be equal";
            return false;
        }
    }
    int64_t headNumKeyShare = inTensorDescs.at(3).shape.dims[2]; // 3:KeyShare
    if (headNumKeyShare != inTensorDescs.at(4).shape.dims[2]) {  // 4:ValueShare; 2:3rd dim
        ATB_LOG(ERROR) << GetLogPrefix() << "3rd dim of KeyShare, ValueShare should be equal";
        return false;
    }
    if (inTensorDescs.at(3).shape.dimNum == 4 &&                     // 3:KeyShare; 4:[B,S,N,D];
        inTensorDescs.at(4).shape.dimNum == 4) {                     // 4:ValueShare; 4:[B,S,N,D]
        int64_t headDimKeyShare = inTensorDescs.at(3).shape.dims[3]; // 3:KeyShare
        if (headDimKeyShare != inTensorDescs.at(4).shape.dims[3]) {  // 4:ValueShare
            ATB_LOG(ERROR) << GetLogPrefix() << "4th dim of KeyShare, ValueShare should be equal";
            return false;
        }
    }
    return true;
}

bool RelayAttentionOperation::InferShapeDimCheckKVShare(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(2).shape.dimNum == 4 &&                // 2:Value; 4:[B,S,N,D];
        inTensorDescs.at(3).shape.dimNum == 4) {                // 3:KeyShare; 4:[B,S,N,D]
        int64_t headNumKey = inTensorDescs.at(2).shape.dims[2]; // 2:Value
        if (headNumKey != inTensorDescs.at(3).shape.dims[2]) {  // 3:KeyShare
            ATB_LOG(ERROR) << GetLogPrefix() << "3rd dim of Key, Value, KeyShare, ValueShare should be equal";
            return false;
        }
        int64_t headDimKey = inTensorDescs.at(2).shape.dims[3]; // 2:Value
        if (headDimKey != inTensorDescs.at(3).shape.dims[3]) {  // 3:KeyShare
            ATB_LOG(ERROR) << GetLogPrefix() << "4th dim of Key, Value, KeyShare, ValueShare should be equal";
            return false;
        }
    }
    if (inTensorDescs.at(2).shape.dimNum == 3 &&                // 2:Value; 3:[B,S,N*D];
        inTensorDescs.at(3).shape.dimNum == 3) {                // 3:KeyShare; 3:[B,S,N*D]
        int64_t headNumKey = inTensorDescs.at(2).shape.dims[2]; // 2:Value
        if (headNumKey != inTensorDescs.at(3).shape.dims[2]) {  // 3:KeyShare
            ATB_LOG(ERROR) << GetLogPrefix() << "3rd dim of Key, Value, KeyShare, ValueShare should be equal";
            return false;
        }
    }
    if (inTensorDescs.at(2).shape.dimNum == 3 &&                // 2:Value; 3:[B,S,N*D];
        inTensorDescs.at(3).shape.dimNum == 4) {                // 3:KeyShare; 4:[B,S,N,D]
        int64_t headNumKey = inTensorDescs.at(2).shape.dims[2]; // 2:Value
        if (headNumKey != inTensorDescs.at(3).shape.dims[2] * inTensorDescs.at(3).shape.dims[3]) { // 3:KeyShare
            ATB_LOG(ERROR) << GetLogPrefix() << "The third dimension of key and value must be equal to the product "
                           << "of the third and fourth dimensions of KeyShare and ValueShare.";
            return false;
        }
    }
    if (inTensorDescs.at(2).shape.dimNum == 4 &&                     // 2:Value; 4:[B,S,N,D];
        inTensorDescs.at(3).shape.dimNum == 3) {                     // 3:KeyShare; 3:[B,S,N*D]
        int64_t headNumKeyShare = inTensorDescs.at(3).shape.dims[2]; // 3:KeyShare
        if (headNumKeyShare != inTensorDescs.at(2).shape.dims[2] * inTensorDescs.at(2).shape.dims[3]) { // 2:Value
            ATB_LOG(ERROR) << GetLogPrefix() << "The third dimension of KeyShare and ValueShare must be equal to "
                           << "the product of the third and fourth dimensions of Key and Value.";
            return false;
        }
    }
    return true;
}

Status RelayAttentionOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(0).shape.dimNum != 2) { // 2:[nTokens, qHiddenSize]
        ATB_LOG(ERROR) << GetLogPrefix() << "dim number of query should be 2";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(1).shape.dimNum != 3 && inTensorDescs.at(1).shape.dimNum != 4) { // 3:[B,S,N*D]; 4:[B,S,N,D]
        ATB_LOG(ERROR) << GetLogPrefix() << "dim number of key should be 3 or 4";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(2).shape.dimNum != 3 && // 2:Value; 3:[B,S,N*D];
        inTensorDescs.at(2).shape.dimNum != 4) { // 4:[B,S,N,D]
        ATB_LOG(ERROR) << GetLogPrefix() << "dim number of value should be 3 or 4";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(3).shape.dimNum != 3 && inTensorDescs.at(3).shape.dimNum != 4) { // 3:[B,S,N*D]; 4:[B,S,N,D]
        ATB_LOG(ERROR) << GetLogPrefix() << "dim number of keyShare should be 3 or 4";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(4).shape.dimNum != 3 && inTensorDescs.at(4).shape.dimNum != 4) { // 3:[B,S,N*D]; 4:[B,S,N,D]
        ATB_LOG(ERROR) << GetLogPrefix() << "dim number of valueShare should be 3 or 4";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(6).shape.dimNum != 1) { // 6:seqLen; 1:[batch]
        ATB_LOG(ERROR) << GetLogPrefix() << "dim number of seqLen should be 1";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(7).shape.dimNum != 1) { // 7:kvSeqLen; 1:[batch]
        ATB_LOG(ERROR) << GetLogPrefix() << "dim number of kvSeqLen should be 1";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(8).shape.dimNum != 1) { // 8:kvShareMap; 1:[batch]
        ATB_LOG(ERROR) << GetLogPrefix() << "dim number of kvShareMap should be 1";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(9).shape.dimNum != 1) { // 9:kvShareLen; 1:[sharedGroupsCount]
        ATB_LOG(ERROR) << GetLogPrefix() << "dim number of kvShareLen should be 1";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    Status st = NO_ERROR;
    st = InferShapeDimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

Status RelayAttentionOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                               SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    return NO_ERROR;
}

Status RelayAttentionOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                               const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    Status st = NO_ERROR;
    st = InferShapeCheckImpl(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    TensorDesc outTensorDesc = outTensors.at(0).desc;
    if (!TensorUtil::TensorDescEqual(outTensorDesc, inTensorDescs.at(0))) {
        ATB_LOG(ERROR) << "outTensor shape should be the same as inTensor";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

void RelayAttentionOperation::InitOpIni()
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("RelayAttentionOperation");
    if (operationIr_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "No matched param for op ini";
    }
    return;
}

std::shared_ptr<Runner> RelayAttentionOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<RelayAttentionOpsRunner>(param_);
}

nlohmann::json RelayAttentionOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

} // namespace atb