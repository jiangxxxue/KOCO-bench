/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "grouped_matmul_inplace_add_operation.h"
#include "atb/operation/op_param_funcs.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/config.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "grouped_matmul_inplace_add_ops_runner.h"

static constexpr uint32_t IN_TENSOR_NUM = 4;
static constexpr uint32_t OUT_TENSOR_NUM = 1;
static constexpr uint64_t DIM_NUM_1 = 1;
static constexpr uint64_t DIM_NUM_2 = 2;
static constexpr uint64_t DIM_NUM_3 = 3;
static constexpr uint64_t MAX_GROUP_NUM = 128;
static constexpr uint64_t MAX_SHAPE = 65535;

namespace {
bool ParamCheck(const atb::infer::GroupedMatmulInplaceAddParam &opParam)
{
    atb::ExternalError error;
    error.errorType = atb::ERROR_INVALID_PARAM;
    error.solutionDesc = "Please check the value of params";
    bool is910B = atb::GetSingleton<atb::Config>().Is910B();
    if (!is910B) {
        error.errorDesc = "Platform is not Atlas 800I A2 inference product, operation is not supported,";
        error.errorData = atb::OperationUtil::ConcatInfo("is Atlas 800I A2 = ", is910B);
        error.solutionDesc = "Please check the platform.";
        ATB_LOG(ERROR) << error;
        return false;
    }
    if (opParam.transposeB) {
        error.errorDesc = "param transposeB only can be false,";
        error.errorData = atb::OperationUtil::ConcatInfo(error.errorData, "which is ", opParam.transposeB);
        ATB_LOG(ERROR) << error;
        return false;
    }
    return true;
}
} // namespace
namespace atb {
OPERATION_PARAM_FUNCS(GroupedMatmulInplaceAddOperation, infer::GroupedMatmulInplaceAddParam)

GroupedMatmulInplaceAddOperation::GroupedMatmulInplaceAddOperation(const infer::GroupedMatmulInplaceAddParam &param)
    : OperationBase("GroupedMatmulInplaceAddOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("GroupedMatmulInplaceAddOperation");
}

GroupedMatmulInplaceAddOperation::~GroupedMatmulInplaceAddOperation() {}

uint32_t GroupedMatmulInplaceAddOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t GroupedMatmulInplaceAddOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

infer::GroupedMatmulInplaceAddParam GroupedMatmulInplaceAddOperation::GetParam() const
{
    return param_;
}

void GroupedMatmulInplaceAddOperation::SetParam(const infer::GroupedMatmulInplaceAddParam &param)
{
    param_ = param;
    runner_ = nullptr;
}

Status GroupedMatmulInplaceAddOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                        SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(3); // 3为GroupedMatmulInplaceAdd的最后一个Tensor，即为输入也为输出
    return NO_ERROR;
}

Status GroupedMatmulInplaceAddOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return InTensorDescsCheck(inTensorDescs);
}

Status GroupedMatmulInplaceAddOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                        const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    Status status = InTensorDescsCheck(inTensorDescs);
    if (status != NO_ERROR) {
        return status;
    }
    return OutTensorCheck(outTensors, inTensorDescs);
}

std::shared_ptr<Runner> GroupedMatmulInplaceAddOperation::CreateRunner(Context &context) const
{
    ContextBase *contextBase = dynamic_cast<ContextBase *>(&context);
    if (!contextBase) {
        ATB_LOG(ERROR) << "context cast to contextBase failed!";
        return nullptr;
    }
    RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_GROUPED_MATMUL_INPLACE_ADD);
    Runner *runner = pool.MallocRunner<GroupedMatmulInplaceAddOpsRunner, infer::GroupedMatmulInplaceAddParam>(param_);
    if (!runner) {
        ATB_LOG(DEBUG) << "MallocRunner from pool failed!";
        return std::make_shared<GroupedMatmulInplaceAddOpsRunner>(param_);
    }
    return std::shared_ptr<Runner>(runner, [&pool](Runner *runner) { pool.FreeRunner(runner); });
}

nlohmann::json GroupedMatmulInplaceAddOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

Status GroupedMatmulInplaceAddOperation::InTensorShapeCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t m = OperationUtil::GetXTensorM(inTensorDescs.at(0), param_.transposeA);
    int64_t xK = OperationUtil::GetXTensorK(inTensorDescs.at(0), param_.transposeA);
    int64_t weightK = OperationUtil::GetYTensorK(inTensorDescs.at(1), param_.transposeB);
    int64_t n = OperationUtil::GetYTensorN(inTensorDescs.at(1), param_.transposeB);
    int64_t numGroup = inTensorDescs.at(2).shape.dims[0];  // 2 表示groupList
    int64_t outshape0 = inTensorDescs.at(3).shape.dims[0]; // 3 表示out
    int64_t outshape1 = inTensorDescs.at(3).shape.dims[1]; // 3 表示out

    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    if (static_cast<uint64_t>(m) > MAX_SHAPE || m <= 0 || static_cast<uint64_t>(n) > MAX_SHAPE || n <= 0) {
        error.errorDesc = "value of m and value of n should be in (0, 65535],";
        error.errorData = OperationUtil::ConcatInfo("m = ", m, ", n = ", n);
        error.solutionDesc = "Please check the shape of inTensors.";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }

    if (xK != weightK) {
        error.errorDesc = "value of inTensor0 k and value of inTensor1 k should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor0 k = ", xK, ", inTensor1 k = ", weightK);
        error.solutionDesc = "Please check the shape of inTensors.";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }

    if (static_cast<uint64_t>(numGroup) > MAX_GROUP_NUM || numGroup <= 0) {
        error.errorDesc = "numGroup should be in [1, 128],";
        error.errorData = OperationUtil::ConcatInfo("numGroup = ", numGroup);
        error.solutionDesc = "Please check the shape of inTensors.";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }

    if (!((outshape0 % numGroup == 0 && outshape0 / numGroup == m && outshape1 == n) ||
          (outshape1 % numGroup == 0 && outshape1 / numGroup == n && outshape0 == m))) {
        error.errorDesc = "shape of out should be [numGroup * m, n] or [m, numGroup * n],";
        error.errorData = OperationUtil::ConcatInfo("shape0 of out = ", outshape0, ", shape1 of out = ", outshape1);
        error.solutionDesc = "Please check the shape of inTensors.";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}
/**
 * @brief 检查输入张量的描述是否有效
 * @param inTensorDescs 输入张量的描述
 * @return 返回状态码，如果检查通过则返回NO_ERROR
 */
Status GroupedMatmulInplaceAddOperation::InTensorDescsCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    const uint64_t tensorDims[IN_TENSOR_NUM] = {DIM_NUM_2, DIM_NUM_2, DIM_NUM_1, DIM_NUM_2};
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    error.solutionDesc = "Please check shape of inTensors";
    // 遍历所有输入张量
    for (size_t i = 0; i < IN_TENSOR_NUM; i++) {
        const TensorDesc &intensor = inTensorDescs.at(i);
        if (!TensorCheck::IsTensorDescDimNumValid(intensor, tensorDims[i])) {
            error.errorDesc = OperationUtil::ConcatInfo("inTensor", i, " dimNum should be ", tensorDims[i], ",");
            error.errorData = OperationUtil::ConcatInfo("which equals to ", intensor.shape.dimNum);
            ATB_LOG(ERROR) << GetLogPrefix() << error;
            return error.errorType;
        }
    }
    // Tensor间shape校验
    if (GroupedMatmulInplaceAddOperation::InTensorShapeCheck(inTensorDescs) != NO_ERROR) {
        return error.errorType;
    }
    return NO_ERROR;
}

Status GroupedMatmulInplaceAddOperation::OutTensorCheck(const SVector<Tensor> &outTensors,
                                                        const SVector<TensorDesc> &inTensorDescs) const
{
    Status status = TensorCheck::TensorDescsEqual(inTensorDescs[3], outTensors[0].desc); // 3 指最后一个输入Tensor
    if (status != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor[3] and outTensor should be the same";
        return status;
    }
    return NO_ERROR;
}
} // namespace atb