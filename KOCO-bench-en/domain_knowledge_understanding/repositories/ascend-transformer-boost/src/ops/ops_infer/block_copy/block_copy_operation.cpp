/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "block_copy_operation.h"
#include "block_copy_ops_runner.h"
#include "atb/utils/log.h"
#include "atb/utils/config.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/utils/tensor_util.h"
#include "atb/operation/op_param_funcs.h"

namespace {
constexpr static uint32_t CACHE_DIM = 4;
constexpr static uint32_t INDICES_DIM = 1;
constexpr static size_t INPUT_SRC_BLOCK = 2;
constexpr static size_t INPUT_DST_BLOCK = 3;
constexpr static size_t INPUT_CUMSUM = 4;
constexpr static size_t NZBLOCKSIZE = 16;
} // namespace

namespace atb {
template <> Status CreateOperation(const infer::BlockCopyParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (!GetSingleton<Config>().Is910B() && !GetSingleton<Config>().Is310P()) {
        ATB_LOG(ERROR) << "only support Atlas 800I A2 inference product and Atlas 300I Duo inference product";
        return ERROR_INVALID_PARAM;
    }
    *operation = new BlockCopyOperation(opParam);
    return NO_ERROR;
}

BlockCopyOperation::BlockCopyOperation(const infer::BlockCopyParam &param)
    : OperationBase("BlockCopyOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("BlockCopyOperation");
}

BlockCopyOperation::~BlockCopyOperation() {}

uint32_t BlockCopyOperation::GetInputNum() const
{
    const uint32_t inTensorNum = 5;
    return inTensorNum;
}

uint32_t BlockCopyOperation::GetOutputNum() const
{
    return 0;
}

Status BlockCopyOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t blockCount = inTensorDescs.at(0).shape.dims[0];
    if (!TensorUtil::TensorShapeEqual(inTensorDescs.at(0).shape, inTensorDescs.at(1).shape)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "kCache shape is not equal vCache shape";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(0).shape.dimNum != CACHE_DIM || inTensorDescs.at(1).shape.dimNum != CACHE_DIM) {
        ATB_LOG(ERROR) << GetLogPrefix() << "cache shape is not " << CACHE_DIM;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(INPUT_SRC_BLOCK).shape.dimNum != INDICES_DIM ||
        inTensorDescs.at(INPUT_DST_BLOCK).shape.dimNum != INDICES_DIM) {
        ATB_LOG(ERROR) << GetLogPrefix() << "indices shape is not " << INDICES_DIM;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (!TensorUtil::TensorShapeEqual(inTensorDescs.at(INPUT_CUMSUM).shape, inTensorDescs.at(INPUT_SRC_BLOCK).shape)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "cumSum shape is not equal srcBlockIndices shape";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(INPUT_SRC_BLOCK).shape.dims[0] > blockCount ||
        inTensorDescs.at(INPUT_DST_BLOCK).shape.dims[0] > blockCount) {
        ATB_LOG(ERROR) << GetLogPrefix() << "indices shape[0] is greater than blockCount";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status BlockCopyOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                          SVector<TensorDesc> &outTensorDescs) const
{
    ATB_LOG(INFO) << GetLogPrefix() << "inTensorDescs Size:" << inTensorDescs.size()
                  << "outTensorDescs Size:" << outTensorDescs.size();
    return NO_ERROR;
}

Status BlockCopyOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    (void)outTensors;
    int64_t blockCount = inTensors.at(0).desc.shape.dims[0];
    if (!TensorUtil::TensorDescEqual(inTensors.at(0).desc, inTensors.at(1).desc)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "kCache desc is not equal vCache desc";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensors.at(0).desc.shape.dimNum != CACHE_DIM || inTensors.at(1).desc.shape.dimNum != CACHE_DIM) {
        ATB_LOG(ERROR) << GetLogPrefix() << "cache shape is not 4";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensors.at(INPUT_SRC_BLOCK).desc.shape.dimNum != INDICES_DIM ||
        inTensors.at(INPUT_DST_BLOCK).desc.shape.dimNum != INDICES_DIM) {
        ATB_LOG(ERROR) << GetLogPrefix() << "indices shape is not 1";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (!TensorUtil::TensorShapeEqual(inTensors.at(INPUT_CUMSUM).desc.shape,
                                      inTensors.at(INPUT_SRC_BLOCK).desc.shape)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "cumSum shape is not equal srcBlockIndices shape";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensors.at(INPUT_SRC_BLOCK).desc.shape.dims[0] > blockCount ||
        inTensors.at(INPUT_DST_BLOCK).desc.shape.dims[0] > blockCount) {
        ATB_LOG(ERROR) << GetLogPrefix() << "indices shape[0] is greater than blockCount";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (GetSingleton<Config>().Is310P()) {
        if ((inTensors.at(0).desc.dtype != ACL_FLOAT16) ||
            (inTensors.at(1).desc.dtype != ACL_FLOAT16)) {
            ATB_LOG(ERROR) << "Atlas 300I Duo inference products only support fp16";
            return ERROR_INVALID_TENSOR_DTYPE;
        }
        auto status = SetupDimCheck310P(inTensors);
        if (status != NO_ERROR) {
            return status;
        }
    } else if (inTensors.at(0).desc.format == aclFormat::ACL_FORMAT_FRACTAL_NZ ||
            inTensors.at(1).desc.format == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
                return ERROR_INVALID_TENSOR_FORMAT;
    }
    return NO_ERROR;
}

Status BlockCopyOperation::SetupDimCheck310P(const SVector<atb::Tensor> &inTensors) const
{
    if ((inTensors.at(0).desc.shape.dimNum != 4) ||  // kCache: 4 dim
        (inTensors.at(1).desc.shape.dimNum != 4) ||  // vCache: 4 dim
        (inTensors.at(2).desc.shape.dimNum != 1) ||  // 2: src
        (inTensors.at(3).desc.shape.dimNum != 1) ||  // 3: dst
        (inTensors.at(4).desc.shape.dimNum != 1)  // 4: cumsum
    ) {
        ATB_LOG(ERROR) << "invalid intensor dimNums";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensors.at(0).desc.format == aclFormat::ACL_FORMAT_FRACTAL_NZ ||
        inTensors.at(1).desc.format == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
            if ((inTensors.at(0).desc.shape.dims[3] != NZBLOCKSIZE) ||
                (inTensors.at(1).desc.shape.dims[3] != NZBLOCKSIZE) ||
                (inTensors.at(0).desc.shape.dims[2] % NZBLOCKSIZE != 0) ||
                (inTensors.at(1).desc.shape.dims[2] % NZBLOCKSIZE != 0)) { // 2: dim
                    ATB_LOG(ERROR) << GetLogPrefix() << "NZ format tensor dim should be aligned to 16";
                    return ERROR_INVALID_TENSOR_DIM;
                }
    } else {
        if ((inTensors.at(0).desc.shape.dims[3] * inTensors.at(0).desc.shape.dims[2] * inTensors.at(0).desc.shape.dims[1]) % NZBLOCKSIZE != 0 ||
            (inTensors.at(1).desc.shape.dims[3] * inTensors.at(1).desc.shape.dims[2] * inTensors.at(0).desc.shape.dims[1]) % NZBLOCKSIZE != 0) {
                ATB_LOG(ERROR) << GetLogPrefix() << "ND format product of the first three tensor dim should be aligned to 16";
                return ERROR_INVALID_TENSOR_DIM;
            }
    }
    if (inTensors.at(2).desc.shape.dims[0] != inTensors.at(4).desc.shape.dims[0]) {
        ATB_LOG(ERROR) << "src dim should be same as cumsum";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return  NO_ERROR;
}

std::shared_ptr<Runner> BlockCopyOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<BlockCopyOpsRunner>(param_);
}
} // namespace atb
