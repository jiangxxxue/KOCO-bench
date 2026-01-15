/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "norm_rope_reshape_ops_runner.h"
#include "atb/utils/log.h"
 
namespace atb {
static const uint64_t IN_TENSOR_COUNT_SEVEN = 7;
 
void NormRopeReshapeOpsRunner::SetNormRopeReshapeParam(
    const infer::NormRopeReshapeParam &inferParam,
    AtbOps::OpParam::RmsNormAndRopeAndReshapeAndCache &asdopsParam) const
{
    asdopsParam.precisionMode = inferParam.precisionMode;
    asdopsParam.epsilon = inferParam.epsilon;
    asdopsParam.rotaryCoeff = inferParam.rotaryCoeff;
}
 
void NormRopeReshapeOpsRunner::BuildNormRopeReshapeGraph(
    const AtbOps::OpParam::RmsNormAndRopeAndReshapeAndCache &normRopeReshapeParam)
{
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_SEVEN);
    size_t inId = 0;
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &keyRopeTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &cosTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &sinTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &slotMappingTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &keycacheInTensor = kernelGraph_.inTensors.at(inId++);
 
    kernelGraph_.outTensors.resize(1);
    size_t outId = 0;
    Mki::Tensor &keycacheOutTensor = kernelGraph_.outTensors.at(outId++);
 
    kernelGraph_.nodes.resize(1);
    auto &normRopeReshapeNode = kernelGraph_.nodes.at(0);
    normRopeReshapeNode.opDesc = {0, "RmsNormAndRopeAndReshapeAndCacheOperation", normRopeReshapeParam};
    normRopeReshapeNode.inTensors = {&xTensor, &gammaTensor, &keyRopeTensor, &cosTensor, &sinTensor,
                                     &slotMappingTensor, &keycacheInTensor};
    normRopeReshapeNode.outTensors = {&keycacheOutTensor};
}
 
NormRopeReshapeOpsRunner::NormRopeReshapeOpsRunner(const infer::NormRopeReshapeParam &param)
    : OpsRunner("NormRopeReshapeOpsRunner", RUNNER_TYPE_NORM_ROPE_RESHAPE), param_(param)
{
    AtbOps::OpParam::RmsNormAndRopeAndReshapeAndCache rmsNormAndRopeAndReshapeAndCacheParam;
    SetNormRopeReshapeParam(param_, rmsNormAndRopeAndReshapeAndCacheParam);
    BuildNormRopeReshapeGraph(rmsNormAndRopeAndReshapeAndCacheParam);
    ATB_LOG(INFO) << GetName() << " RmsNormAndRopeAndReshapeAndCacheOperation opDesc "
                  << "RmsNormAndRopeAndReshapeAndCacheParam epsilon:" << rmsNormAndRopeAndReshapeAndCacheParam.epsilon
                  << "precisionMode:" << rmsNormAndRopeAndReshapeAndCacheParam.precisionMode
                  << "rotaryCoeff:" << rmsNormAndRopeAndReshapeAndCacheParam.rotaryCoeff;
}
 
NormRopeReshapeOpsRunner::~NormRopeReshapeOpsRunner() {}
} // namespace atb