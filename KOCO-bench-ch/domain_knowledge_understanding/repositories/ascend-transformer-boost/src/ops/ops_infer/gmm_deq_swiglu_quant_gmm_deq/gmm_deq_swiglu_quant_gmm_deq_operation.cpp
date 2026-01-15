/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/operation/op_param_funcs.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/config.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"

#include "gmm_deq_swiglu_quant_gmm_deq_ops_runner.h"
#include "gmm_deq_swiglu_quant_gmm_deq_operation.h"

namespace {
constexpr int64_t MAX_M = 131072;
constexpr int64_t MAX_GROUP_COUNT = 32;
constexpr int64_t SUPPORTED_K1 = 7168;
constexpr int64_t SUPPORTED_N1 = 4096;
constexpr int64_t SUPPORTED_K2 = 2048;
constexpr int64_t SUPPORTED_N2 = 7168;

constexpr size_t INPUT_NUM = 7;
constexpr size_t OUTPUT_NUM = 1;

struct InTensorIndex {
    static constexpr size_t X1 = 0;
    static constexpr size_t WEIGHT1 = 1;
    static constexpr size_t SCALE1 = 2;
    static constexpr size_t PER_TOKEN_SCALE1 = 3;
    static constexpr size_t GROUP_LIST = 4;
    static constexpr size_t WEIGHT2 = 5;
    static constexpr size_t SCALE2 = 6;
};

constexpr size_t X_DIMS = 2;
struct XDimIndex {
    static constexpr size_t M = 0;
    static constexpr size_t K = 1;
};

constexpr size_t WEIGHT_DIMS = 3;
template <bool IsTrans = false>
struct WeightDimIndex {
    static constexpr size_t GROUP_COUNT = 0;
    static constexpr size_t K = IsTrans ? 2 : 1;
    static constexpr size_t N = IsTrans ? 1 : 2;
};

constexpr size_t SCALE_DIMS = 2;
struct ScaleDimIndex {
    static constexpr size_t GROUP_COUNT = 0;
    static constexpr size_t N = 1;
};

constexpr size_t PER_TOKEN_SCALE_DIMS = 1;
struct PerTokenScaleDimIndex {
    static constexpr size_t M = 0;
};

constexpr size_t GROUP_LIST_DIMS = 1;
struct GroupListDimIndex {
    static constexpr size_t GROUP_COUNT = 0;
};

constexpr size_t OUTPUT_DIMS = 2;
struct OutputDimIndex {
    static constexpr size_t M = 0;
    static constexpr size_t N = 1;
};

bool ParamCheck(const atb::infer::GmmDeqSwigluQuantGmmDeqParam &opParam)
{
    (void)opParam;
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        ATB_LOG(ERROR) << "GmmDeqSwigluQuantGmmDeqOperation is only supported in Atlas 800I A2 inference product.";
        return false;
    }

    if (opParam.outputType != atb::infer::GmmDeqSwigluQuantGmmDeqParam::OUTPUT_FLOAT16) {
        ATB_LOG(ERROR) << "Param outputType only support OUTPUT_FLOAT16 (" <<
            static_cast<int>(atb::infer::GmmDeqSwigluQuantGmmDeqParam::OUTPUT_FLOAT16) << ").";
        return false;
    }

    if (opParam.groupListType != atb::infer::GmmDeqSwigluQuantGmmDeqParam::GROUP_LIST_CUMSUM) {
        ATB_LOG(ERROR) << "Param groupListType only support GROUP_LIST_CUM_SUM (" <<
            static_cast<int>(atb::infer::GmmDeqSwigluQuantGmmDeqParam::GROUP_LIST_CUMSUM) << ").";
        return false;
    }

    if (opParam.weightUpPermuteType == atb::infer::GmmDeqSwigluQuantGmmDeqParam::PERMUTE_INVALID) {
        ATB_LOG(ERROR) << "Param weightUpPermuteType has invalid value.";
        return false;
    }

    if (opParam.transposeWeightUp) {
        ATB_LOG(ERROR) << "Param transposeWeightUp only support false.";
        return false;
    }

    if (!opParam.transposeWeightDown) {
        ATB_LOG(ERROR) << "Param transposeWeightDown only support true.";
        return false;
    }

    return true;
}

atb::Status CheckInTensorsDescDimNum(const atb::SVector<atb::TensorDesc> &inTensorDescs)
{
    if (!atb::TensorCheck::IsTensorDescDimNumValid(inTensorDescs.at(InTensorIndex::X1), X_DIMS)) {
        ATB_LOG(ERROR) << "x1 dim num is not support for GmmDeqSwigluQuantGmmDeqOperation.";
        return atb::ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (!atb::TensorCheck::IsTensorDescDimNumValid(inTensorDescs.at(InTensorIndex::WEIGHT1), WEIGHT_DIMS)) {
        ATB_LOG(ERROR) << "weight1 dim num is not support for GmmDeqSwigluQuantGmmDeqOperation.";
        return atb::ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (!atb::TensorCheck::IsTensorDescDimNumValid(inTensorDescs.at(InTensorIndex::SCALE1), SCALE_DIMS)) {
        ATB_LOG(ERROR) << "scale1 dim num is not support for GmmDeqSwigluQuantGmmDeqOperation.";
        return atb::ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (!atb::TensorCheck::IsTensorDescDimNumValid(
        inTensorDescs.at(InTensorIndex::PER_TOKEN_SCALE1), PER_TOKEN_SCALE_DIMS)) {
        ATB_LOG(ERROR) << "pertokenScale1 dim num is not support for GmmDeqSwigluQuantGmmDeqOperation.";
        return atb::ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (!atb::TensorCheck::IsTensorDescDimNumValid(inTensorDescs.at(InTensorIndex::GROUP_LIST), GROUP_LIST_DIMS)) {
        ATB_LOG(ERROR) << "groupList dim num is not support for GmmDeqSwigluQuantGmmDeqOperation.";
        return atb::ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (!atb::TensorCheck::IsTensorDescDimNumValid(inTensorDescs.at(InTensorIndex::WEIGHT2), WEIGHT_DIMS)) {
        ATB_LOG(ERROR) << "weight2 dim num is not support for GmmDeqSwigluQuantGmmDeqOperation.";
        return atb::ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (!atb::TensorCheck::IsTensorDescDimNumValid(inTensorDescs.at(InTensorIndex::SCALE2), SCALE_DIMS)) {
        ATB_LOG(ERROR) << "scale2 dim num is not support for GmmDeqSwigluQuantGmmDeqOperation.";
        return atb::ERROR_INVALID_TENSOR_DIM_NUM;
    }
    return atb::NO_ERROR;
}

bool CheckX1Shape(const atb::TensorDesc &x1Desc, int64_t m)
{
    return x1Desc.shape.dims[XDimIndex::M] == m &&
        x1Desc.shape.dims[XDimIndex::K] == SUPPORTED_K1;
}

bool CheckWeight1Shape(const atb::TensorDesc &weight1Desc, int64_t groupCount)
{
    return weight1Desc.shape.dims[WeightDimIndex<false>::GROUP_COUNT] == groupCount &&
        weight1Desc.shape.dims[WeightDimIndex<false>::K] == SUPPORTED_K1 &&
        weight1Desc.shape.dims[WeightDimIndex<false>::N] == SUPPORTED_N1;
}

bool CheckScale1Shape(const atb::TensorDesc &scale1Desc, int64_t groupCount)
{
    return scale1Desc.shape.dims[ScaleDimIndex::GROUP_COUNT] == groupCount &&
        scale1Desc.shape.dims[ScaleDimIndex::N] == SUPPORTED_N1;
}

bool CheckPerTokenScale1Shape(const atb::TensorDesc &perTokenScale1Desc, int64_t m)
{
    return perTokenScale1Desc.shape.dims[PerTokenScaleDimIndex::M] == m;
}

bool CheckGroupListShape(const atb::TensorDesc &groupListDesc, int64_t groupCount)
{
    return groupListDesc.shape.dims[GroupListDimIndex::GROUP_COUNT] == groupCount;
}

bool CheckWeight2Shape(const atb::TensorDesc &weight2Desc, int64_t groupCount)
{
    return weight2Desc.shape.dims[WeightDimIndex<true>::GROUP_COUNT] == groupCount &&
        weight2Desc.shape.dims[WeightDimIndex<true>::N] == SUPPORTED_N2 &&
        weight2Desc.shape.dims[WeightDimIndex<true>::K] == SUPPORTED_K2;
}

bool CheckScale2Shape(const atb::TensorDesc &scale2Desc, int64_t groupCount)
{
    return scale2Desc.shape.dims[ScaleDimIndex::GROUP_COUNT] == groupCount &&
        scale2Desc.shape.dims[ScaleDimIndex::N] == SUPPORTED_N2;
}

atb::Status CheckInTensorsShape(const atb::SVector<atb::TensorDesc> &inTensorDescs)
{
    int64_t m = atb::OperationUtil::GetXTensorM(inTensorDescs.at(InTensorIndex::X1), false);
    int64_t groupCount = inTensorDescs.at(InTensorIndex::GROUP_LIST).shape.dims[GroupListDimIndex::GROUP_COUNT];
    if (m > MAX_M) {
        ATB_LOG(ERROR) << "Unsupported shape for GmmDeqSwigluQuantGmmDeqOperation, m must no greater than " << MAX_M;
        return atb::ERROR_INVALID_TENSOR_DIM;
    }
    if (groupCount > MAX_GROUP_COUNT) {
        ATB_LOG(ERROR) << "Unsupported shape for GmmDeqSwigluQuantGmmDeqOperation, group count must no greater than "
            << MAX_GROUP_COUNT;
        return atb::ERROR_INVALID_TENSOR_DIM;
    }

    if (!CheckX1Shape(inTensorDescs.at(InTensorIndex::X1), m)) {
        ATB_LOG(ERROR) << "Unsupported x1 shape for GmmDeqSwigluQuantGmmDeq";
        return atb::ERROR_INVALID_TENSOR_DIM;
    }
    if (!CheckWeight1Shape(inTensorDescs.at(InTensorIndex::WEIGHT1), groupCount)) {
        ATB_LOG(ERROR) << "Unsupported weight1 shape for GmmDeqSwigluQuantGmmDeq";
        return atb::ERROR_INVALID_TENSOR_DIM;
    }
    if (!CheckScale1Shape(inTensorDescs.at(InTensorIndex::SCALE1), groupCount)) {
        ATB_LOG(ERROR) << "Unsupported scale1 shape for GmmDeqSwigluQuantGmmDeq";
        return atb::ERROR_INVALID_TENSOR_DIM;
    }
    if (!CheckPerTokenScale1Shape(inTensorDescs.at(InTensorIndex::PER_TOKEN_SCALE1), m)) {
        ATB_LOG(ERROR) << "Unsupported perTokenScale1 shape for GmmDeqSwigluQuantGmmDeq";
        return atb::ERROR_INVALID_TENSOR_DIM;
    }
    if (!CheckGroupListShape(inTensorDescs.at(InTensorIndex::GROUP_LIST), groupCount)) {
        ATB_LOG(ERROR) << "Unsupported groupList shape for GmmDeqSwigluQuantGmmDeq";
        return atb::ERROR_INVALID_TENSOR_DIM;
    }
    if (!CheckWeight2Shape(inTensorDescs.at(InTensorIndex::WEIGHT2), groupCount)) {
        ATB_LOG(ERROR) << "Unsupported weight2 shape for GmmDeqSwigluQuantGmmDeq";
        return atb::ERROR_INVALID_TENSOR_DIM;
    }
    if (!CheckScale2Shape(inTensorDescs.at(InTensorIndex::SCALE2), groupCount)) {
        ATB_LOG(ERROR) << "Unsupported scale2 shape for GmmDeqSwigluQuantGmmDeq";
        return atb::ERROR_INVALID_TENSOR_DIM;
    }
    return atb::NO_ERROR;
}

atb::Status CheckInTensorDescs(const atb::SVector<atb::TensorDesc> &inTensorDescs)
{
    atb::Status status = CheckInTensorsDescDimNum(inTensorDescs);
    if (status != atb::NO_ERROR) {
        return status;
    }

    return CheckInTensorsShape(inTensorDescs);
}

atb::Status CheckInTensors(const atb::SVector<atb::Tensor> &inTensors)
{
    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    return CheckInTensorDescs(inTensorDescs);
}

atb::Status CheckOutTensors(const atb::SVector<atb::Tensor> &inTensors, const atb::SVector<atb::Tensor> &outTensors)
{
    if (!atb::TensorCheck::IsTensorDescDimNumValid(outTensors.at(0).desc, OUTPUT_DIMS)) {
        ATB_LOG(ERROR) << "outTensor dim num is not support for GmmDeqSwigluQuantGmmDeqOperation.";
        return atb::ERROR_INVALID_TENSOR_DIM_NUM;
    }

    int64_t m = atb::OperationUtil::GetXTensorM(inTensors.at(InTensorIndex::X1).desc, false);
    if (outTensors.at(0).desc.shape.dims[OutputDimIndex::M] != m ||
        outTensors.at(0).desc.shape.dims[OutputDimIndex::N] != SUPPORTED_N2) {
        ATB_LOG(ERROR) << "outTensor shape is not consistent with inTensor for GmmDeqSwigluQuantGmmDeqOperation.";
        return atb::ERROR_INVALID_TENSOR_SIZE;
    }
    return atb::NO_ERROR;
}
} // namespace

namespace atb {
OPERATION_PARAM_FUNCS(GmmDeqSwigluQuantGmmDeqOperation, infer::GmmDeqSwigluQuantGmmDeqParam)

GmmDeqSwigluQuantGmmDeqOperation::GmmDeqSwigluQuantGmmDeqOperation(const infer::GmmDeqSwigluQuantGmmDeqParam &param)
    : OperationBase("GmmDeqSwigluQuantGmmDeqOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("GmmDeqSwigluQuantGmmDeqOperation");
}

GmmDeqSwigluQuantGmmDeqOperation::~GmmDeqSwigluQuantGmmDeqOperation() {}

uint32_t GmmDeqSwigluQuantGmmDeqOperation::GetInputNum() const
{
    return INPUT_NUM;
}

uint32_t GmmDeqSwigluQuantGmmDeqOperation::GetOutputNum() const
{
    return OUTPUT_NUM;
}

infer::GmmDeqSwigluQuantGmmDeqParam GmmDeqSwigluQuantGmmDeqOperation::GetParam() const
{
    return param_;
}

void GmmDeqSwigluQuantGmmDeqOperation::SetParam(const infer::GmmDeqSwigluQuantGmmDeqParam &param)
{
    param_ = param;
    runner_ = nullptr;
}

Status GmmDeqSwigluQuantGmmDeqOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                        SVector<TensorDesc> &outTensorDescs) const
{
    int64_t m = OperationUtil::GetXTensorM(inTensorDescs.at(InTensorIndex::X1), false);
    auto &outDesc = outTensorDescs.at(0);
    outDesc.dtype = ACL_FLOAT16;
    outDesc.format = ACL_FORMAT_ND;
    outDesc.shape.dimNum = OUTPUT_DIMS;
    outDesc.shape.dims[OutputDimIndex::M] = m;
    outDesc.shape.dims[OutputDimIndex::N] = SUPPORTED_N2;
    return NO_ERROR;
}

Status GmmDeqSwigluQuantGmmDeqOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return CheckInTensorDescs(inTensorDescs);
}

Status GmmDeqSwigluQuantGmmDeqOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                        const SVector<Tensor> &outTensors) const
{
    Status status = CheckInTensors(inTensors);
    if (status != NO_ERROR) {
        return status;
    }
    return CheckOutTensors(inTensors, outTensors);
}

std::shared_ptr<Runner> GmmDeqSwigluQuantGmmDeqOperation::CreateRunner(Context &context) const
{
    ContextBase *contextBase = dynamic_cast<ContextBase *>(&context);
    if (!contextBase) {
        ATB_LOG(ERROR) << "context cast to contextBase failed!";
        return nullptr;
    }
    RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_GMM_DEQ_SWIGLU_QUANT_GMM_DEQ);
    Runner *runner = pool.MallocRunner<GmmDeqSwigluQuantGmmDeqOpsRunner, infer::GmmDeqSwigluQuantGmmDeqParam>(param_);
    if (!runner) {
        ATB_LOG(WARN) << "MallocRunner from pool failed!";
        return std::make_shared<GmmDeqSwigluQuantGmmDeqOpsRunner>(param_);
    }

    return std::shared_ptr<Runner>(runner, [poolPtr = &pool](Runner *runner) { poolPtr->FreeRunner(runner); });
}

nlohmann::json GmmDeqSwigluQuantGmmDeqOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb