/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "customize_block_copy_operation.h"
#include "customize_block_copy_ops_runner.h"
#include "atb/utils/log.h"
#include "atb/utils/config.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/singleton.h"
#include "atb/utils/tensor_util.h"
#include "atb/operation/op_param_funcs.h"
#include "atb/operation/customize_operation_ir_cfg.h"

namespace {
constexpr static uint32_t CACHE_DIM = 4;
constexpr static uint32_t INDICES_DIM = 1;
constexpr static size_t INPUT_SRC_BLOCK = 2;
constexpr static size_t INPUT_DST_BLOCK = 3;
constexpr static size_t INPUT_CUMSUM = 4;
} // namespace

namespace atb {
/**
 * @brief CreateOperation 模板特化，用于工厂注册阶段创建实例。
 * @param[in] opParam    用户参数
 * @param[out] operation 输出的 Operation 指针
 * @return Status        创建结果，NO_ERROR 表示成功
 */
template <> Status CreateOperation(const customize::BlockCopyParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "only support Atlas 800I A2 inference product";
        return ERROR_INVALID_PARAM;
    }
    *operation = new CustomizeBlockCopyOperation(opParam);
    return NO_ERROR;
}

CustomizeBlockCopyOperation::CustomizeBlockCopyOperation(const customize::BlockCopyParam &param)
    : OperationBase("CustomizeBlockCopyOperation"), param_(param)
{
    operationIr_ = GetSingleton<CustomizeOperationIrCfg>().GetOperationIr("CustomizeBlockCopyOperation");
}

CustomizeBlockCopyOperation::~CustomizeBlockCopyOperation() {}

uint32_t CustomizeBlockCopyOperation::GetInputNum() const
{
    const uint32_t inTensorNum = 5;
    return inTensorNum;
}

uint32_t CustomizeBlockCopyOperation::GetOutputNum() const
{
    return 0;
}

Status CustomizeBlockCopyOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
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

Status CustomizeBlockCopyOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                   SVector<TensorDesc> &outTensorDescs) const
{
    ATB_LOG(INFO) << GetLogPrefix() << "inTensorDescs Size:" << inTensorDescs.size()
                  << "outTensorDescs Size:" << outTensorDescs.size();
    return NO_ERROR;
}

Status CustomizeBlockCopyOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                   const SVector<Tensor> &outTensors) const
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
    return NO_ERROR;
}

std::shared_ptr<Runner> CustomizeBlockCopyOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<CustomizeBlockCopyOpsRunner>(param_);
}
} // namespace atb
