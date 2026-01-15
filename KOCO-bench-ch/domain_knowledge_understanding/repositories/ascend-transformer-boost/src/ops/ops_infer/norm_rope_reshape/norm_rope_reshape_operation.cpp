/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "norm_rope_reshape_operation.h"
#include <cmath>
#include "norm_rope_reshape_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"
#include "atb/utils/operation_util.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"
 
namespace atb {
static const uint32_t IN_TENSOR_SIX = 6;
static const uint32_t OUT_TENSOR_ZERO = 0;
static const uint32_t DIM_ZERO = 0;
static const uint32_t DIM_ONE = 1;
static const uint32_t DIM_TWO = 2;
static const uint32_t DIM_THREE = 3;
static const uint32_t IN_TENSOR_COUNT_SEVEN = 7;
static const uint32_t OUT_TENSOR_COUNT_ONE = 1;
static const float THRESHOLD = std::numeric_limits<float>::min(); // The non-negative minimum value of float type is 1.17549e-038.
static const uint32_t ELEVEN = 11;
static const uint32_t FLOAT16SIZE = 2;
static const uint32_t MAXUBSIZE = 196352;
 
 
template <> Status CreateOperation(const infer::NormRopeReshapeParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (std::fabs(opParam.epsilon) < THRESHOLD) {
        ATB_LOG(ERROR) << "Invalid epsilon, it's recommended to init a nonzero value for eps.";
        return ERROR_INVALID_PARAM;
    }
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "NormRopeReshapeOperation only supports Atlas 800I A2 inference products";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) NormRopeReshapeOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}
 
NormRopeReshapeOperation::NormRopeReshapeOperation(const infer::NormRopeReshapeParam &param)
    : OperationBase("NormRopeReshapeOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("NormRopeReshapeOperation");
}
 
NormRopeReshapeOperation::~NormRopeReshapeOperation() {}
 
uint32_t NormRopeReshapeOperation::GetInputNum() const
{
    return IN_TENSOR_COUNT_SEVEN;
}
 
uint32_t NormRopeReshapeOperation::GetOutputNum() const
{
    return OUT_TENSOR_COUNT_ONE;
}
 
Status NormRopeReshapeOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(OUT_TENSOR_ZERO) = inTensorDescs.at(IN_TENSOR_SIX);
    return NO_ERROR;
}
 
Status NormRopeReshapeOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (ELEVEN * inTensorDescs.at(0).shape.dims[DIM_TWO] * FLOAT16SIZE +
        inTensorDescs.at(6).shape.dims[DIM_THREE] * FLOAT16SIZE >= MAXUBSIZE) { // 6 keycachein
        ATB_LOG(ERROR) << GetLogPrefix() << "11 * inTensorDescs[0].shape.dim[2] * 2 +"
                                         << "inTensorDescs[6].shape.dim[3] * 2 shold less 196352.";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (!CheckDim910B(inTensorDescs)) {
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (!NormRopeReshapeCheckImpl910B(inTensorDescs)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}
 
Status NormRopeReshapeOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensorDescs.push_back(inTensors.at(i).desc);
    }
    aclDataType targetType = inTensorDescs[6].dtype;
    Status st = CheckOutTensorSame(inTensorDescs[6], outTensors[0].desc, targetType);
    if (st != NO_ERROR) {
        return st;
    }
    return InferShapeCheckImpl(inTensorDescs);
}
 
std::shared_ptr<Runner> NormRopeReshapeOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<NormRopeReshapeOpsRunner>(param_);
}
 

bool NormRopeReshapeOperation::CheckDim910B(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(0).shape.dimNum != 3 || // 0: x 3: 3 dims
        inTensorDescs.at(1).shape.dimNum != 1 || // 1: gamma 1: 1 dim
        inTensorDescs.at(2).shape.dimNum != 2 || // 2: keyRope 2: 2 dims
        inTensorDescs.at(3).shape.dimNum != 2 || // 3: cos 2: 2 dims
        inTensorDescs.at(4).shape.dimNum != 2 || // 4: sin 2: 2 dims
        inTensorDescs.at(5).shape.dimNum != 1 || // 5: slotMapping 1: 1 dim
        inTensorDescs.at(6).shape.dimNum != 4) { // 6: keycachein 4: 4 dims
        ATB_LOG(ERROR) << "invalid intensor dimNum";
        return false;
    }
    return true;
}
 
bool NormRopeReshapeOperation::NormRopeReshapeCheckImpl910B(const SVector<TensorDesc> &inTensorDescs) const
{
    uint32_t idx = 0;
    TensorDesc xTensorDesc = inTensorDescs.at(idx++);
    TensorDesc gammaTensorDesc = inTensorDescs.at(idx++);
    TensorDesc keyRopeTensorDesc = inTensorDescs.at(idx++);
    TensorDesc cosTensorDesc = inTensorDescs.at(idx++);
    TensorDesc sinTensorDesc = inTensorDescs.at(idx++);
    idx++;
    TensorDesc keycacheInTensorDesc = inTensorDescs.at(idx++);
    if (!GammaBetaTensorCheck(xTensorDesc, gammaTensorDesc)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "gammaTensorDesc error";
        return false;
    }
    if (xTensorDesc.shape.dims[DIM_ZERO] != keyRopeTensorDesc.shape.dims[DIM_ZERO]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "The first dimension of the first input tensor "
                                         << "and the third input tensor should be equal";
        return false;
    }
    if (xTensorDesc.shape.dims[DIM_ZERO] != cosTensorDesc.shape.dims[DIM_ZERO]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "The first dimension of the first input tensor "
                                         << "and the fourth input tensor should be equal";
        return false;
    }
    if (keyRopeTensorDesc.shape.dims[DIM_ONE] != cosTensorDesc.shape.dims[DIM_ONE] ||
        keyRopeTensorDesc.shape.dims[DIM_ONE] != sinTensorDesc.shape.dims[DIM_ONE]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "The second dimension of the third input tensor "
                                         << "should be equal to the second dimension of the fourth input tensor "
                                         << "and the fifth input tensor";
        return false;
    }
    if (keycacheInTensorDesc.shape.dims[DIM_THREE] != xTensorDesc.shape.dims[DIM_TWO] +
                                                      keyRopeTensorDesc.shape.dims[DIM_ONE]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "The fourth dimension of the seventh input tensor"
                                         << " should equal the third dimension of the first input tensor"
                                         << " plus the second dimension of the third input tensor";
        return false;
    }
    if (xTensorDesc.shape.dims[2] % 16 != 0 || // 1: 2nd dim 16: Align units
        gammaTensorDesc.shape.dims[0] % 16 != 0) { // 1: 2nd dim 16: Align units
        ATB_LOG(ERROR) << GetLogPrefix() << "The last dimension num of the first input tensor, "
                                         << "the second input tensor should be a multiple of 16";
        return false;
    }
    return true;
}
 
Status NormRopeReshapeOperation::CheckOutTensorSame
(const TensorDesc &tensorDesc1, const TensorDesc &tensorDesc2, const aclDataType &targetType) const
{
    if (targetType != tensorDesc2.dtype) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor dtype of the sixth intensor and the outtensor should be same";
        return ERROR_INVALID_TENSOR_DTYPE;
    }
    if (tensorDesc1.format != tensorDesc2.format) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor formatof the sixth intensor and the outtensor should be same";
        return ERROR_INVALID_TENSOR_FORMAT;
    }
    if (tensorDesc1.shape.dimNum != tensorDesc2.shape.dimNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor dimNum of the sixth intensor and the outtensor should be same";
        return ERROR_INVALID_TENSOR_SIZE;
    }
    for (uint64_t i = 0; i < tensorDesc1.shape.dimNum; i++) {
        if (tensorDesc1.shape.dims[i] != tensorDesc2.shape.dims[i]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor dims of the sixth intensor and the outtensor should be same";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}
 
bool NormRopeReshapeOperation::GammaBetaTensorCheck(
    const TensorDesc &xTensorDesc, const TensorDesc &tensorDesc2) const
{
    int embedDim = xTensorDesc.shape.dims[xTensorDesc.shape.dimNum - 1];
    if (xTensorDesc.dtype != tensorDesc2.dtype || xTensorDesc.format != tensorDesc2.format) {
        ATB_LOG(ERROR) << GetLogPrefix() << "gamma and bias should have the same dtype and format as x";
        return false;
    }
    if (tensorDesc2.shape.dims[tensorDesc2.shape.dimNum - 1] != embedDim) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, inTensor last dim is not equal to embed_dim";
        return false;
    }
    return true;
}
 
nlohmann::json NormRopeReshapeOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb