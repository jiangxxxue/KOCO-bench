/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "reshape_and_cache_omni_operation.h"
#include "reshape_and_cache_omni_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"

namespace {
bool ParamCheck(const atb::infer::ReshapeAndCacheOmniParam opParam)
{
    (void) opParam;
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        ATB_LOG(ERROR) << "ReshapeAndCacheOmni only supports Atlas 800I A2 inference product";
        return false;
    }
    return true;
}
}

namespace atb {

static const int64_t NZBLOCKSIZE = 16;
static const uint32_t IN_TENSOR_0_KEY = 0;
static const uint32_t IN_TENSOR_1_VALUE = 1;
static const uint32_t IN_TENSOR_2_KEYCACHE = 2;
static const uint32_t IN_TENSOR_3_VALUECACHE = 3;
static const uint32_t IN_TENSOR_4_SLOTMAPPING = 4;
static const uint32_t IN_TENSOR_5_WINS = 5;
static const uint32_t IN_TENSOR_6_SEQLEN = 6;
static const uint32_t IN_TENSOR_7_OFFSET_INDEX = 7;
 
OPERATION_PARAM_FUNCS(ReshapeAndCacheOmniOperation, infer::ReshapeAndCacheOmniParam)

ReshapeAndCacheOmniOperation::ReshapeAndCacheOmniOperation(const infer::ReshapeAndCacheOmniParam &param)
    : OperationBase("ReshapeAndCacheOmniOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("ReshapeAndCacheOmniOperation");
}
 
ReshapeAndCacheOmniOperation::~ReshapeAndCacheOmniOperation() {}
 
uint32_t ReshapeAndCacheOmniOperation::GetInputNum() const
{
    uint32_t inTensorNumBase = 8;
    return inTensorNumBase;
}
 
uint32_t ReshapeAndCacheOmniOperation::GetOutputNum() const
{
    const uint32_t outTensorNum = 2;
    return outTensorNum;
}

infer::ReshapeAndCacheOmniParam ReshapeAndCacheOmniOperation::GetParam() const
{
    return param_;
}
 
void ReshapeAndCacheOmniOperation::SetParam(const infer::ReshapeAndCacheOmniParam &param)
{
    param_ = param;
    runner_ = nullptr;
}

Status ReshapeAndCacheOmniOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
    SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(IN_TENSOR_2_KEYCACHE);
    outTensorDescs.at(1) = inTensorDescs.at(IN_TENSOR_3_VALUECACHE);
    return NO_ERROR;
}
 
Status ReshapeAndCacheOmniOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(IN_TENSOR_0_KEY).shape.dimNum != 3 ||          // 0: key 3: 3 dims
        inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dimNum != 3 ||        // 1: value 3: 3 dims
        inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dimNum != 4 ||     // 2: keyCache 4: 4 dims
        inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dimNum != 4 ||   // 3: valueCache 4: 4 dims
        inTensorDescs.at(IN_TENSOR_4_SLOTMAPPING).shape.dimNum != 1 ||  // 4: slotMapping 1: 1 dim
        inTensorDescs.at(IN_TENSOR_5_WINS).shape.dimNum != 1 ||         // 5: wins 1: 1 dim
        inTensorDescs.at(IN_TENSOR_6_SEQLEN).shape.dimNum != 1 ||       // 6: seqLen 1: 1 dim
        inTensorDescs.at(IN_TENSOR_7_OFFSET_INDEX).shape.dimNum != 1) { // 7: offsetIndex 1: 1 dim
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid intensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    return DimCheck(inTensorDescs);
}

Status ReshapeAndCacheOmniOperation::DimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t numTokens = inTensorDescs.at(IN_TENSOR_0_KEY).shape.dims[0];      // 0: key
    int64_t numHead = inTensorDescs.at(IN_TENSOR_0_KEY).shape.dims[1];        // 1: key num_head
    int64_t headDim = inTensorDescs.at(IN_TENSOR_0_KEY).shape.dims[2];        // 2: key head_size
    int64_t numBlocks = inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[0]; // 0: keyCache num_blocks
    int64_t blockSize = inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[1]; // 1: keyCache block_size
    int64_t batch = inTensorDescs.at(IN_TENSOR_6_SEQLEN).shape.dims[0];       // 0: seqLen batch
    if (headDim % 32 != 0) { // 32:
        ATB_LOG(ERROR) << GetLogPrefix() << "headDim should be multiples of 32";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (numTokens != inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "numToken should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (numHead != inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dims[1] ||
        numHead * batch != inTensorDescs.at(IN_TENSOR_4_SLOTMAPPING).shape.dims[0] ||
        numHead * batch != inTensorDescs.at(IN_TENSOR_5_WINS).shape.dims[0] ||
        numHead * batch != inTensorDescs.at(IN_TENSOR_7_OFFSET_INDEX).shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor shape is invalid, please check batch or numHead";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (headDim != inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dims[2] ||      // 2: 第三维
        headDim != inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[3] ||   // 3： 第四维
        headDim != inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dims[3]) { // 3： 第四维
        ATB_LOG(ERROR) << GetLogPrefix() << "headDim should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (numBlocks != inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "numBlocks should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (blockSize != inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dims[1]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "blockSize should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[2] != 1 ||    // 2：第三维
        inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dims[2] != 1) {  // 2： 第三维
        ATB_LOG(ERROR) << GetLogPrefix() << "the 3rd dim of key_cache and value_cache should be 1";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (batch * numHead >= 500) { // 500: 算子约束
        ATB_LOG(ERROR) << GetLogPrefix() << "batch * numHead should be less than 500";
        return ERROR_INVALID_TENSOR_DIM;
    }

    return NO_ERROR;
}
 
Status ReshapeAndCacheOmniOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                    const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensorDescs.push_back(inTensors.at(i).desc);
    }
    Status status = DimCheck(inTensorDescs);
    if (status != NO_ERROR) {
        return status;
    }
    if (!TensorUtil::TensorDescEqual(inTensorDescs.at(IN_TENSOR_2_KEYCACHE), outTensors.at(0).desc) ||
        !TensorUtil::TensorDescEqual(inTensorDescs.at(IN_TENSOR_3_VALUECACHE), outTensors.at(1).desc)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "shapes of inTensors and outTensors should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}
 
std::shared_ptr<Runner> ReshapeAndCacheOmniOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<ReshapeAndCacheOmniOpsRunner>(param_);
}
} // namespace atb
