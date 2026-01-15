/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atb/utils/log.h>
#include <asdops/params/params.h>
#include <atbops/params/params.h>
#include "atb/utils/utils_internal.h"
#include "grouped_matmul_with_routing_runner.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM = 4;
static const uint32_t OUT_TENSOR_NUM = 1;
static constexpr size_t SIZE_2 = 2;
static constexpr size_t SIZE_3 = 3;
static constexpr size_t SIZE_4 = 4;
static constexpr size_t SIZE_5 = 5;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;
static constexpr int64_t DEFAULT_ALIGN = 16;
static constexpr int64_t INT8_ALIGN = 32;
static const uint32_t TRANSPOSE_HIDDEN_SIZE = 1;
static const uint32_t NO_TRANSPOSE_HIDDEN_SIZE = 2;
GroupedMatmulWithRoutingRunner::GroupedMatmulWithRoutingRunner(const infer::GroupedMatmulWithRoutingParam &param)
    : OpsRunner("GroupedMatmulWithRoutingRunner", RUNNER_TYPE_GROUPED_MATMUL_WITH_ROUTING), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "GroupedMatmulWithRoutingRunner::GroupedMatmulWithRoutingRunner called:";
}

/**
 * 组图
 * @param opsTensorPack
 * @return
 */
Status GroupedMatmulWithRoutingRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;
    int64_t inTensorNum = 0;
    Mki::Tensor &inputTensor = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &ecountTensor = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &indiceIntensor = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);
    kernelGraph_.nodes.resize(1);
    auto &moeGmmNode = kernelGraph_.nodes.at(0);
    AtbOps::OpParam::MoeGmm opParam;
    opParam.moeGmmDequantType = AtbOps::OpParam::MoeGmm::NO_DEQUANT;
    if (param_.outDataType == ACL_BF16) {
        opParam.moeGmmDequantType = AtbOps::OpParam::MoeGmm::DEQ_BF16;
    } else if (param_.outDataType == ACL_FLOAT16) {
        opParam.moeGmmDequantType = AtbOps::OpParam::MoeGmm::DEQ_FP16;
    }
    opParam.moeGmmMode = static_cast<AtbOps::OpParam::MoeGmm::MoeGmmMode>(param_.groupedMatmulType);
    opParam.transposeB = param_.transposeB ? 1 : 0;
    opParam.topK = static_cast<uint32_t>(param_.topK);
    opParam.hiddenSize.at(0) = inputTensor.desc.dims.at(1);
    opParam.hiddenSize.at(1) = param_.transposeB ? weightTensor.desc.dims.at(TRANSPOSE_HIDDEN_SIZE) :
                                                   weightTensor.desc.dims.at(NO_TRANSPOSE_HIDDEN_SIZE);
    moeGmmNode.opDesc = {0, "MoeGmmOperation", opParam};
    if (param_.outDataType == ACL_DT_UNDEFINED) {
        moeGmmNode.inTensors = {&inputTensor, &weightTensor, &ecountTensor, &indiceIntensor};
    } else {
        bool isWeightNz = weightTensor.desc.format == Mki::TENSOR_FORMAT_FRACTAL_NZ;
        Mki::Tensor &weightscale = kernelGraph_.inTensors.at(inTensorNum++);
        Mki::Tensor &activatescale = kernelGraph_.inTensors.at(inTensorNum++);
        moeGmmNode.inTensors = {&inputTensor,    &weightTensor, &ecountTensor,
                                &indiceIntensor, &weightscale,  &activatescale};
        moeGmmNode.inTensorViewFuncs.resize(moeGmmNode.inTensors.size());
        if (isWeightNz) {
            moeGmmNode.inTensorViewFuncs.at(1) = [&](const Mki::SVector<int64_t> &oldDims,
                                                     Mki::SVector<int64_t> &newDims) {
                int64_t align = INT8_ALIGN;
                if (oldDims.size() == SIZE_3) {
                    newDims = {oldDims.at(0), UtilsInternal::AlignUp(oldDims.at(DIM_2), align) / align,
                               UtilsInternal::AlignUp(oldDims.at(1), DEFAULT_ALIGN), align};
                    ATB_LOG(INFO) << " nz reshape - before: " << oldDims << "; after: " << newDims;
                } else {
                    newDims = oldDims;
                }
            };
        }
    }
    moeGmmNode.outTensors = {&outTensor};
    return NO_ERROR;
}

GroupedMatmulWithRoutingRunner::~GroupedMatmulWithRoutingRunner() {}
} // namespace atb
