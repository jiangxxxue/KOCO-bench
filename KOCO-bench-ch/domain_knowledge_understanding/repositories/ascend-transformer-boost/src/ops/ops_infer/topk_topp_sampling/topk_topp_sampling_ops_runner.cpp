/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "topk_topp_sampling_ops_runner.h"
#include <limits>
#include <asdops/params/params.h>
#include "atb/utils/tensor_util.h"
#include "atb/utils/log.h"

namespace atb {
static const uint64_t INTERNAL_TENSOR_COUNT = 5;
static const uint64_t NODE_COUNT = 5;
static const uint64_t EXPONENTIAL_INTERNAL_TENSOR_COUNT = 11;
static const uint64_t EXPONENTIAL_NODE_COUNT = 11;
static const uint64_t LOGPROBS_EXPONENTIAL_INTERNAL_TENSOR_COUNT = 16;
static const uint64_t LOGPROBS_EXPONENTIAL_OUT_TENSOR_COUNT = 3;
static const uint64_t LOGPROBS_EXPONENTIAL_NODE_COUNT = 17;
static const uint64_t MULTINOMIAL_INTERNAL_TENSOR_COUNT = 8;
static const uint64_t MULTINOMIAL_NODE_COUNT = 8;
static const uint64_t LOGPROBS_MULTINOMIAL_OUT_TENSOR_COUNT = 3;
static const uint64_t LOGPROBS_MULTINOMIAL_NODE_COUNT = 9;

TopkToppSamplingOpsRunner::TopkToppSamplingOpsRunner(const infer::TopkToppSamplingParam &param)
    : OpsRunner("TopkToppSamplingOpsRunner", RUNNER_TYPE_TOPP), param_(param)
{
}

Status TopkToppSamplingOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    ATB_LOG(INFO) << "TopkToppSamplingOpsRunner::SetupKernelGraph called.";
    kernelGraph_.inTensors = opsTensorPack.inTensors;
    kernelGraph_.outTensors = opsTensorPack.outTensors;
    switch (param_.topkToppSamplingType) {
        case atb::infer::TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_SAMPLING:
            return SetupBatchTopKExponentialSampling();
        case atb::infer::TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_SAMPLING:
            return SetupBatchTopKMultinomialSampling();
        case atb::infer::TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_LOGPROBS_SAMPLING:
            return SetupLogProbsBatchTopKMultinomialSampling();
        case atb::infer::TopkToppSamplingParam::TopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_LOGPROBS_SAMPLING:
            return SetupLogProbsBatchTopKExponentialSampling();
        case atb::infer::TopkToppSamplingParam::TopkToppSamplingType::SINGLE_TOPK_SAMPLING:
            return SetupSingleTopKSampling();
        default:
            ATB_LOG(ERROR) << GetLogPrefix() << "UnSupported TopkToppSamplingType: " << param_.topkToppSamplingType;
            return ERROR_INVALID_PARAM;
    }
}

Status TopkToppSamplingOpsRunner::SetupBatchTopKExponentialSampling()
{
    int64_t inTensorNum = 0;
    auto &probsTensor = kernelGraph_.inTensors.at(inTensorNum++);
    auto &topKTensor = kernelGraph_.inTensors.at(inTensorNum++);
    auto &topPTensor = kernelGraph_.inTensors.at(inTensorNum++);
    auto &expTensor = kernelGraph_.inTensors.at(inTensorNum++);

    int64_t outTensorNum = 0;
    auto &indicesSampledI32Tensor = kernelGraph_.outTensors.at(outTensorNum++); // [batch, 1]
    auto &probsSampledTensor = kernelGraph_.outTensors.at(outTensorNum++);      // [batch, 1]

    kernelGraph_.internalTensors.resize(EXPONENTIAL_INTERNAL_TENSOR_COUNT);
    int64_t internalTensorNum = 0;
    auto &probsSortedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &indicesSortedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &topKGatherLogitsTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &topKGreaterMaskTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &topKProbsFilledSortedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &cumsumedProbsTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &topPGreaterMaskTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &topPProbsFilledTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &probsDividedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &indexArgmaxTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &probsArgmaxTensor = kernelGraph_.internalTensors.at(internalTensorNum++);

    kernelGraph_.nodes.resize(EXPONENTIAL_NODE_COUNT); // exponential sampling has 11 nodes
    int64_t nodeNum = 0;
    auto &sortNode = kernelGraph_.nodes.at(nodeNum++);
    auto &topKGatherNode = kernelGraph_.nodes.at(nodeNum++);
    auto &topKGreaterNode = kernelGraph_.nodes.at(nodeNum++);
    auto &topKFillNode = kernelGraph_.nodes.at(nodeNum++);
    auto &cumsumNode = kernelGraph_.nodes.at(nodeNum++);
    auto &topPGreaterNode = kernelGraph_.nodes.at(nodeNum++);
    auto &topPFillNode = kernelGraph_.nodes.at(nodeNum++);
    auto &expDivisionNode = kernelGraph_.nodes.at(nodeNum++);
    auto &argmaxNode = kernelGraph_.nodes.at(nodeNum++);
    auto &indexArgmaxGatherNode = kernelGraph_.nodes.at(nodeNum++);
    auto &probsArgmaxGatherNode = kernelGraph_.nodes.at(nodeNum++);

    sortNode.opDesc = {0, "SortOperation", {}};
    sortNode.inTensors = {&probsTensor};
    sortNode.outTensors = {&probsSortedTensor, &indicesSortedTensor};
    sortNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        const auto &dims = launchParam.GetInTensor(0).desc.dims;
        dimNum_ = dims.size();
        lastDim_ = dims[dimNum_ - 1];
        launchParam.SetParam(AsdOps::OpParam::Sort({{static_cast<int32_t>(lastDim_)}}));
    };

    topKGatherNode.opDesc = {0, "GatherOperation", AsdOps::OpParam::Gather()};
    topKGatherNode.inTensors = {&probsSortedTensor, &topKTensor};
    topKGatherNode.outTensors = {&topKGatherLogitsTensor};
    topKGatherNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Gather({axis, {axis}}));
    };

    topKGreaterNode.opDesc = {0, "ElewiseOperation",
                              AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_LESS})};
    topKGreaterNode.inTensors = {&probsSortedTensor, &topKGatherLogitsTensor};
    topKGreaterNode.outTensors = {&topKGreaterMaskTensor};

    topKFillNode.opDesc = {0, "FillOperation", AsdOps::OpParam::Fill({true, {0}, {}})};
    topKFillNode.inTensors = {&probsSortedTensor, &topKGreaterMaskTensor};
    topKFillNode.outTensors = {&topKProbsFilledSortedTensor};

    cumsumNode.opDesc = {0, "CumsumOperation", {}};
    cumsumNode.inTensors = {&topKProbsFilledSortedTensor};
    cumsumNode.outTensors = {&cumsumedProbsTensor};
    cumsumNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Cumsum({{axis}, false, false}));
    };

    topPGreaterNode.opDesc = {0, "ElewiseOperation",
                              AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_GREATER})};
    topPGreaterNode.inTensors = {&cumsumedProbsTensor, &topPTensor};
    topPGreaterNode.outTensors = {&topPGreaterMaskTensor};

    topPFillNode.opDesc = {0, "FillOperation", AsdOps::OpParam::Fill({true, {0}, {}})};
    topPFillNode.inTensors = {&topKProbsFilledSortedTensor, &topPGreaterMaskTensor};
    topPFillNode.outTensors = {&topPProbsFilledTensor};

    expDivisionNode.opDesc = {0, "ElewiseOperation",
                              AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_REALDIV})};
    expDivisionNode.inTensors = {&topPProbsFilledTensor, &expTensor};
    expDivisionNode.outTensors = {&probsDividedTensor};

    argmaxNode.opDesc = {0, "SortOperation", AsdOps::OpParam::Sort({{1}})};
    argmaxNode.inTensors = {&probsDividedTensor};
    argmaxNode.outTensors = {&probsArgmaxTensor, &indexArgmaxTensor};

    indexArgmaxGatherNode.opDesc = {0, "GatherOperation", AsdOps::OpParam::Gather()};
    indexArgmaxGatherNode.inTensors = {&indicesSortedTensor, &indexArgmaxTensor};
    indexArgmaxGatherNode.outTensors = {&indicesSampledI32Tensor};
    indexArgmaxGatherNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Gather({axis, {axis}}));
    };

    probsArgmaxGatherNode.opDesc = {0, "GatherOperation", AsdOps::OpParam::Gather()};
    probsArgmaxGatherNode.inTensors = {&probsSortedTensor, &indexArgmaxTensor};
    probsArgmaxGatherNode.outTensors = {&probsSampledTensor};
    probsArgmaxGatherNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Gather({axis, {axis}}));
    };

    return NO_ERROR;
}

Status TopkToppSamplingOpsRunner::SetupBatchTopKMultinomialSampling()
{
    int64_t inTensorNum = 0;
    auto &probsTensor = kernelGraph_.inTensors.at(inTensorNum++);
    auto &topKTensor = kernelGraph_.inTensors.at(inTensorNum++);
    auto &topPTensor = kernelGraph_.inTensors.at(inTensorNum++);

    int64_t outTensorNum = 0;
    auto &indicesSampledI32Tensor = kernelGraph_.outTensors.at(outTensorNum++); // [batch, 1]
    auto &probsSampledTensor = kernelGraph_.outTensors.at(outTensorNum++);      // [batch, 1]

    kernelGraph_.internalTensors.resize(MULTINOMIAL_INTERNAL_TENSOR_COUNT);
    int64_t internalTensorNum = 0;
    auto &probsSortedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &indicesSortedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &topKGatherLogitsTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &topKGreaterMaskTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &topKProbsFilledSortedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &cumsumedProbsTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &indicesTopPSampledTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &selectRangeTensor = kernelGraph_.internalTensors.at(internalTensorNum++);

    kernelGraph_.nodes.resize(MULTINOMIAL_NODE_COUNT); // exponential sampling has 8 nodes
    int64_t nodeNum = 0;
    auto &sortNode = kernelGraph_.nodes.at(nodeNum++);
    auto &topKGatherNode = kernelGraph_.nodes.at(nodeNum++);
    auto &topKGreaterNode = kernelGraph_.nodes.at(nodeNum++);
    auto &topKFillNode = kernelGraph_.nodes.at(nodeNum++);
    auto &cumsumNode = kernelGraph_.nodes.at(nodeNum++);
    auto &mixTopPNode = kernelGraph_.nodes.at(nodeNum++);
    auto &indexTopPGatherNode = kernelGraph_.nodes.at(nodeNum++);
    auto &probsTopPGatherNode = kernelGraph_.nodes.at(nodeNum++);

    sortNode.opDesc = {0, "SortOperation", {}};
    sortNode.inTensors = {&probsTensor};
    sortNode.outTensors = {&probsSortedTensor, &indicesSortedTensor};
    sortNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        const auto &dims = launchParam.GetInTensor(0).desc.dims;
        dimNum_ = dims.size();
        lastDim_ = dims[dimNum_ - 1];
        launchParam.SetParam(AsdOps::OpParam::Sort({{static_cast<int32_t>(lastDim_)}}));
    };

    topKGatherNode.opDesc = {0, "GatherOperation", AsdOps::OpParam::Gather()};
    topKGatherNode.inTensors = {&probsSortedTensor, &topKTensor};
    topKGatherNode.outTensors = {&topKGatherLogitsTensor};
    topKGatherNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Gather({axis, {axis}}));
    };

    topKGreaterNode.opDesc = {0, "ElewiseOperation",
                              AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_LESS})};
    topKGreaterNode.inTensors = {&probsSortedTensor, &topKGatherLogitsTensor};
    topKGreaterNode.outTensors = {&topKGreaterMaskTensor};

    topKFillNode.opDesc = {0, "FillOperation", AsdOps::OpParam::Fill({true, {0}, {}})};
    topKFillNode.inTensors = {&probsSortedTensor, &topKGreaterMaskTensor};
    topKFillNode.outTensors = {&topKProbsFilledSortedTensor};

    cumsumNode.opDesc = {0, "CumsumOperation", {}};
    cumsumNode.inTensors = {&topKProbsFilledSortedTensor};
    cumsumNode.outTensors = {&cumsumedProbsTensor};
    cumsumNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Cumsum({{axis}, false, false}));
    };

    AtbOps::OpParam::Toppsample toppParam;
    toppParam.randSeed = param_.randSeeds;
    mixTopPNode.opDesc = {0, "ToppsampleOperation", toppParam};
    mixTopPNode.inTensors = {&cumsumedProbsTensor, &topPTensor};
    mixTopPNode.outTensors = {&indicesTopPSampledTensor, &selectRangeTensor};

    indexTopPGatherNode.opDesc = {0, "GatherOperation", AsdOps::OpParam::Gather()};
    indexTopPGatherNode.inTensors = {&indicesSortedTensor, &indicesTopPSampledTensor};
    indexTopPGatherNode.outTensors = {&indicesSampledI32Tensor};
    indexTopPGatherNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Gather({axis, {axis}}));
    };

    probsTopPGatherNode.opDesc = {0, "GatherOperation", AsdOps::OpParam::Gather()};
    probsTopPGatherNode.inTensors = {&probsSortedTensor, &indicesTopPSampledTensor};
    probsTopPGatherNode.outTensors = {&probsSampledTensor};
    probsTopPGatherNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Gather({axis, {axis}}));
    };
    return NO_ERROR;
}

Status TopkToppSamplingOpsRunner::SetupLogProbsBatchTopKExponentialSampling()
{
    int64_t inTensorNum = 0;
    auto &probsTensor = kernelGraph_.inTensors.at(inTensorNum++);
    auto &topKTensor = kernelGraph_.inTensors.at(inTensorNum++);
    auto &topPTensor = kernelGraph_.inTensors.at(inTensorNum++);
    auto &expTensor = kernelGraph_.inTensors.at(inTensorNum++);

    kernelGraph_.outTensors.resize(LOGPROBS_EXPONENTIAL_OUT_TENSOR_COUNT);
    int64_t outTensorNum = 0;
    auto &indicesSampledI32Tensor = kernelGraph_.outTensors.at(outTensorNum++); // [batch, 1]
    auto &probsSampledTensor = kernelGraph_.outTensors.at(outTensorNum++);      // [batch, 1]
    auto &logProbsTensor = kernelGraph_.outTensors.at(outTensorNum++);

    kernelGraph_.internalTensors.resize(LOGPROBS_EXPONENTIAL_INTERNAL_TENSOR_COUNT);
    int64_t internalTensorNum = 0;
    auto &probsSortedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &indicesSortedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &topKGatherLogitsTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &topKGreaterMaskTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &topKProbsFilledSortedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &cumsumedProbsTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &topPGreaterMaskTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &topPProbsFilledTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &probsDividedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &indexArgmaxTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &probsArgmaxTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &logProbsFilledTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &logProbsMaskOrTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &logProbsRangeTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &logProbsReduceTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &logProbsCastTensor = kernelGraph_.internalTensors.at(internalTensorNum++);

    kernelGraph_.nodes.resize(LOGPROBS_EXPONENTIAL_NODE_COUNT);
    int64_t nodeNum = 0;
    auto &sortNode = kernelGraph_.nodes.at(nodeNum++);
    auto &topKGatherNode = kernelGraph_.nodes.at(nodeNum++);
    auto &topKGreaterNode = kernelGraph_.nodes.at(nodeNum++);
    auto &topKFillNode = kernelGraph_.nodes.at(nodeNum++);
    auto &cumsumNode = kernelGraph_.nodes.at(nodeNum++);
    auto &topPGreaterNode = kernelGraph_.nodes.at(nodeNum++);
    auto &topPFillNode = kernelGraph_.nodes.at(nodeNum++);
    auto &expDivisionNode = kernelGraph_.nodes.at(nodeNum++);
    auto &argmaxNode = kernelGraph_.nodes.at(nodeNum++);
    auto &indexArgmaxGatherNode = kernelGraph_.nodes.at(nodeNum++);
    auto &probsArgmaxGatherNode = kernelGraph_.nodes.at(nodeNum++);
    auto &logProbsFilledNode = kernelGraph_.nodes.at(nodeNum++);
    auto &logProbsMaskOrNode = kernelGraph_.nodes.at(nodeNum++);
    auto &logProbsMaskFilledNode = kernelGraph_.nodes.at(nodeNum++);
    auto &logProbsReduceNode = kernelGraph_.nodes.at(nodeNum++);
    auto &logProbsElewiseCastNode = kernelGraph_.nodes.at(nodeNum++);
    auto &logProbsNode = kernelGraph_.nodes.at(nodeNum++);

    sortNode.opDesc = {0, "SortOperation", {}};
    sortNode.inTensors = {&probsTensor};
    sortNode.outTensors = {&probsSortedTensor, &indicesSortedTensor};
    sortNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        const auto &dims = launchParam.GetInTensor(0).desc.dims;
        dimNum_ = dims.size();
        lastDim_ = dims[dimNum_ - 1];
        launchParam.SetParam(AsdOps::OpParam::Sort({{static_cast<int32_t>(lastDim_)}}));
    };

    topKGatherNode.opDesc = {0, "GatherOperation", AsdOps::OpParam::Gather()};
    topKGatherNode.inTensors = {&probsSortedTensor, &topKTensor};
    topKGatherNode.outTensors = {&topKGatherLogitsTensor};
    topKGatherNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Gather({axis, {axis}}));
    };

    topKGreaterNode.opDesc = {0, "ElewiseOperation",
                              AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_LESS})};
    topKGreaterNode.inTensors = {&probsSortedTensor, &topKGatherLogitsTensor};
    topKGreaterNode.outTensors = {&topKGreaterMaskTensor};

    topKFillNode.opDesc = {0, "FillOperation", AsdOps::OpParam::Fill({true, {0}, {}})};
    topKFillNode.inTensors = {&probsSortedTensor, &topKGreaterMaskTensor};
    topKFillNode.outTensors = {&topKProbsFilledSortedTensor};

    cumsumNode.opDesc = {0, "CumsumOperation", {}};
    cumsumNode.inTensors = {&topKProbsFilledSortedTensor};
    cumsumNode.outTensors = {&cumsumedProbsTensor};
    cumsumNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Cumsum({{axis}, false, false}));
    };

    topPGreaterNode.opDesc = {0, "ElewiseOperation",
                              AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_GREATER})};
    topPGreaterNode.inTensors = {&cumsumedProbsTensor, &topPTensor};
    topPGreaterNode.outTensors = {&topPGreaterMaskTensor};

    topPFillNode.opDesc = {0, "FillOperation", AsdOps::OpParam::Fill({true, {0}, {}})};
    topPFillNode.inTensors = {&topKProbsFilledSortedTensor, &topPGreaterMaskTensor};
    topPFillNode.outTensors = {&topPProbsFilledTensor};

    expDivisionNode.opDesc = {0, "ElewiseOperation",
                              AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_REALDIV})};
    expDivisionNode.inTensors = {&topPProbsFilledTensor, &expTensor};
    expDivisionNode.outTensors = {&probsDividedTensor};

    argmaxNode.opDesc = {0, "SortOperation", AsdOps::OpParam::Sort({{1}})};
    argmaxNode.inTensors = {&probsDividedTensor};
    argmaxNode.outTensors = {&probsArgmaxTensor, &indexArgmaxTensor};

    indexArgmaxGatherNode.opDesc = {0, "GatherOperation", AsdOps::OpParam::Gather()};
    indexArgmaxGatherNode.inTensors = {&indicesSortedTensor, &indexArgmaxTensor};
    indexArgmaxGatherNode.outTensors = {&indicesSampledI32Tensor};
    indexArgmaxGatherNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Gather({axis, {axis}}));
    };

    probsArgmaxGatherNode.opDesc = {0, "GatherOperation", AsdOps::OpParam::Gather()};
    probsArgmaxGatherNode.inTensors = {&probsSortedTensor, &indexArgmaxTensor};
    probsArgmaxGatherNode.outTensors = {&probsSampledTensor};
    probsArgmaxGatherNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Gather({axis, {axis}}));
    };

    auto &dims = probsTensor.desc.dims;
    logProbsFilledNode.opDesc = {0, "FillOperation", AsdOps::OpParam::Fill({false, {1}, {dims[0], dims[1]}})};
    logProbsFilledNode.outTensors = {&logProbsFilledTensor};

    AsdOps::OpParam::Elewise orParam;
    orParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_LOGICAL_OR;

    logProbsMaskOrNode.opDesc = {0, "ElewiseOperation", orParam};
    logProbsMaskOrNode.inTensors = {&topPGreaterMaskTensor, &topKGreaterMaskTensor};
    logProbsMaskOrNode.outTensors = {&logProbsMaskOrTensor};

    logProbsMaskFilledNode.opDesc = {0, "FillOperation", AsdOps::OpParam::Fill({true, {0}, {}})};
    logProbsMaskFilledNode.inTensors = {&logProbsFilledTensor, &logProbsMaskOrTensor};
    logProbsMaskFilledNode.outTensors = {&logProbsRangeTensor};

    AsdOps::OpParam::Reduce reduceParam;
    reduceParam.reduceType = AsdOps::OpParam::Reduce::ReduceType::REDUCE_SUM;

    logProbsReduceNode.opDesc = {0, "ReduceOperation", reduceParam};
    logProbsReduceNode.inTensors = {&logProbsRangeTensor};
    logProbsReduceNode.outTensors = {&logProbsReduceTensor};
    logProbsReduceNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        launchParam.SetParam(AsdOps::OpParam::Reduce({AsdOps::OpParam::Reduce::ReduceType::REDUCE_SUM, {1}}));
    };

    AsdOps::OpParam::Elewise elsewiseParam;
    elsewiseParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_CAST;
    elsewiseParam.outTensorType = Mki::TensorDType::TENSOR_DTYPE_INT32;

    logProbsElewiseCastNode.opDesc = {0, "ElewiseOperation", elsewiseParam};
    logProbsElewiseCastNode.inTensors = {&logProbsReduceTensor};
    logProbsElewiseCastNode.outTensors = {&logProbsCastTensor};

    logProbsNode.opDesc = {0, "LogprobsSampleOperation", {}};
    logProbsNode.inTensors = {&probsSortedTensor, &cumsumedProbsTensor, &logProbsCastTensor};
    logProbsNode.outTensors = {&logProbsTensor};
    logProbsNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        launchParam.SetParam(AsdOps::OpParam::LogprobsSample({param_.logProbsSize}));
    };

    return NO_ERROR;
}

Status TopkToppSamplingOpsRunner::SetupLogProbsBatchTopKMultinomialSampling()
{
    int64_t inTensorNum = 0;
    auto &probsTensor = kernelGraph_.inTensors.at(inTensorNum++);
    auto &topKTensor = kernelGraph_.inTensors.at(inTensorNum++);
    auto &topPTensor = kernelGraph_.inTensors.at(inTensorNum++);
    auto &reedTensor = kernelGraph_.inTensors.at(inTensorNum++);

    kernelGraph_.outTensors.resize(LOGPROBS_MULTINOMIAL_OUT_TENSOR_COUNT);
    int64_t outTensorNum = 0;
    auto &indicesSampledI32Tensor = kernelGraph_.outTensors.at(outTensorNum++); // [batch, 1]
    auto &probsSampledTensor = kernelGraph_.outTensors.at(outTensorNum++);      // [batch, 1]
    auto &logProbsTensor = kernelGraph_.outTensors.at(outTensorNum++); // [batch, logprobsSize]

    kernelGraph_.internalTensors.resize(MULTINOMIAL_INTERNAL_TENSOR_COUNT);
    int64_t internalTensorNum = 0;
    auto &probsSortedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &indicesSortedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &topKGatherLogitsTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &topKGreaterMaskTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &topKProbsFilledSortedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &cumsumedProbsTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &indicesTopPSampledTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &selectRangeTensor = kernelGraph_.internalTensors.at(internalTensorNum++);

    kernelGraph_.nodes.resize(LOGPROBS_MULTINOMIAL_NODE_COUNT);
    int64_t nodeNum = 0;
    auto &sortNode = kernelGraph_.nodes.at(nodeNum++);
    auto &topKGatherNode = kernelGraph_.nodes.at(nodeNum++);
    auto &topKGreaterNode = kernelGraph_.nodes.at(nodeNum++);
    auto &topKFillNode = kernelGraph_.nodes.at(nodeNum++);
    auto &cumsumNode = kernelGraph_.nodes.at(nodeNum++);
    auto &mixTopPNode = kernelGraph_.nodes.at(nodeNum++);
    auto &indexTopPGatherNode = kernelGraph_.nodes.at(nodeNum++);
    auto &probsTopPGatherNode = kernelGraph_.nodes.at(nodeNum++);
    auto &logProbsNode = kernelGraph_.nodes.at(nodeNum++);

    sortNode.opDesc = {0, "SortOperation", {}};
    sortNode.inTensors = {&probsTensor};
    sortNode.outTensors = {&probsSortedTensor, &indicesSortedTensor};
    sortNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        const auto &dims = launchParam.GetInTensor(0).desc.dims;
        dimNum_ = dims.size();
        lastDim_ = dims[dimNum_ - 1];
        launchParam.SetParam(AsdOps::OpParam::Sort({{static_cast<int32_t>(lastDim_)}}));
    };

    topKGatherNode.opDesc = {0, "GatherOperation", AsdOps::OpParam::Gather()};
    topKGatherNode.inTensors = {&probsSortedTensor, &topKTensor};
    topKGatherNode.outTensors = {&topKGatherLogitsTensor};
    topKGatherNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Gather({axis, {axis}}));
    };

    topKGreaterNode.opDesc = {0, "ElewiseOperation",
                              AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_LESS})};
    topKGreaterNode.inTensors = {&probsSortedTensor, &topKGatherLogitsTensor};
    topKGreaterNode.outTensors = {&topKGreaterMaskTensor};

    topKFillNode.opDesc = {0, "FillOperation", AsdOps::OpParam::Fill({true, {0}, {}})};
    topKFillNode.inTensors = {&probsSortedTensor, &topKGreaterMaskTensor};
    topKFillNode.outTensors = {&topKProbsFilledSortedTensor};

    cumsumNode.opDesc = {0, "CumsumOperation", {}};
    cumsumNode.inTensors = {&topKProbsFilledSortedTensor};
    cumsumNode.outTensors = {&cumsumedProbsTensor};
    cumsumNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Cumsum({{axis}, false, false}));
    };

    AtbOps::OpParam::ToppsampleRand toppParam;
    mixTopPNode.opDesc = {0, "ToppsampleRandOperation", toppParam};
    mixTopPNode.inTensors = {&cumsumedProbsTensor, &topPTensor, &reedTensor};
    mixTopPNode.outTensors = {&indicesTopPSampledTensor, &selectRangeTensor};

    indexTopPGatherNode.opDesc = {0, "GatherOperation", AsdOps::OpParam::Gather()};
    indexTopPGatherNode.inTensors = {&indicesSortedTensor, &indicesTopPSampledTensor};
    indexTopPGatherNode.outTensors = {&indicesSampledI32Tensor};
    indexTopPGatherNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Gather({axis, {axis}}));
    };

    probsTopPGatherNode.opDesc = {0, "GatherOperation", AsdOps::OpParam::Gather()};
    probsTopPGatherNode.inTensors = {&probsSortedTensor, &indicesTopPSampledTensor};
    probsTopPGatherNode.outTensors = {&probsSampledTensor};
    probsTopPGatherNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Gather({axis, {axis}}));
    };
    logProbsNode.opDesc = {0, "LogprobsSampleOperation", {}};
    logProbsNode.inTensors = {&probsSortedTensor, &cumsumedProbsTensor, &selectRangeTensor};
    logProbsNode.outTensors = {&logProbsTensor};
    logProbsNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        launchParam.SetParam(AsdOps::OpParam::LogprobsSample({param_.logProbsSize}));
    };
    return NO_ERROR;
}

Status TopkToppSamplingOpsRunner::SetupSingleTopKSampling()
{
    int64_t inTensorNum = 0;
    auto &probsTensor = kernelGraph_.inTensors.at(inTensorNum++); // [batch, voc_len]
    auto &pTensor = kernelGraph_.inTensors.at(inTensorNum++);     // [1] or [batch, 1]

    int64_t outTensorNum = 0;
    auto &indicesSampledI32Tensor = kernelGraph_.outTensors.at(outTensorNum++); // [batch, 1]
    auto &probsSampledTensor = kernelGraph_.outTensors.at(outTensorNum++);      // [batch, 1]

    kernelGraph_.internalTensors.resize(INTERNAL_TENSOR_COUNT); // topp has 4 internel tensors
    int64_t internalTensorNum = 0;
    auto &probsSortedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &indicesSortedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &probsSumedTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &indicesSortedSampledTensor = kernelGraph_.internalTensors.at(internalTensorNum++);
    auto &selectRangeTensor = kernelGraph_.internalTensors.at(internalTensorNum++);

    kernelGraph_.nodes.resize(NODE_COUNT); // topp has 5 nodes
    int64_t nodeNum = 0;
    auto &topkNode = kernelGraph_.nodes.at(nodeNum++);
    auto &cumsumNode = kernelGraph_.nodes.at(nodeNum++);
    auto &toppSamplingNode = kernelGraph_.nodes.at(nodeNum++);
    auto &gatherIndicesNode = kernelGraph_.nodes.at(nodeNum++);
    auto &gatherProbsNode = kernelGraph_.nodes.at(nodeNum++);

    topkNode.opDesc = {0, "SortOperation", {}};
    topkNode.inTensors = {&probsTensor};
    topkNode.outTensors = {&probsSortedTensor, &indicesSortedTensor};
    topkNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        const auto &dims = launchParam.GetInTensor(0).desc.dims;
        dimNum_ = dims.size();
        lastDim_ = dims[dimNum_ - 1];
        int64_t topk = std::min(std::min(lastDim_, static_cast<int64_t>(param_.topk)),
                                static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
        topk = topk == 0 ? lastDim_ : topk;
        launchParam.SetParam(AsdOps::OpParam::Sort({{static_cast<int32_t>(topk)}}));
    };

    cumsumNode.opDesc = {0, "CumsumOperation", {}};
    cumsumNode.inTensors = {&probsSortedTensor};
    cumsumNode.outTensors = {&probsSumedTensor};
    cumsumNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Cumsum({{axis}, false, false}));
    };

    AtbOps::OpParam::Toppsample toppParam;
    std::vector<uint32_t> randSeeds = {param_.randSeed};
    toppParam.randSeed = randSeeds;
    toppSamplingNode.opDesc = {0, "ToppsampleOperation", toppParam};
    toppSamplingNode.inTensors = {&probsSumedTensor, &pTensor};
    toppSamplingNode.outTensors = {&indicesSortedSampledTensor, &selectRangeTensor};

    gatherIndicesNode.opDesc = {0, "GatherOperation", AsdOps::OpParam::Gather()};
    gatherIndicesNode.inTensors = {&indicesSortedTensor, &indicesSortedSampledTensor};
    gatherIndicesNode.outTensors = {&indicesSampledI32Tensor};
    gatherIndicesNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Gather({axis, {axis}}));
    };

    gatherProbsNode.opDesc = {0, "GatherOperation", AsdOps::OpParam::Gather()};
    gatherProbsNode.inTensors = {&probsSortedTensor, &indicesSortedSampledTensor};
    gatherProbsNode.outTensors = {&probsSampledTensor};
    gatherProbsNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        int64_t axis = static_cast<int64_t>(dimNum_ - 1);
        launchParam.SetParam(AsdOps::OpParam::Gather({axis, {axis}}));
    };

    return NO_ERROR;
}

void TopkToppSamplingOpsRunner::SetParam(const Mki::Any &param)
{
    auto newParam = Mki::AnyCast<infer::TopkToppSamplingParam>(param);
    if (newParam.logProbsSize != param_.logProbsSize) {
        param_ = newParam;
        isParamUpdated_ = true;
    }
}


TopkToppSamplingOpsRunner::~TopkToppSamplingOpsRunner() {}
} // namespace atb
