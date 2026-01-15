/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mla_preprocess_ops_runner_split.h"
#include <atb/utils/log.h>
#include <atbops/params/params.h>

namespace {
static const uint32_t IN_TENSOR_NUM_SPLIT = 24;
static const uint32_t OUT_TENSOR_NUM_SPLIT = 4;
static const uint32_t WUK_INDEX = 18;
} // namespace
namespace atb {
MlaPreprocessOpsRunnerSplit::MlaPreprocessOpsRunnerSplit(const infer::MlaPreprocessParam &param)
    : OpsRunner("MlaPreprocessOpsRunnerSplit", RUNNER_TYPE_MLAPREPROCESS_SPLIT), param_(param)
{
    ATB_LOG(INFO) << "MlaPreprocessOpsRunnerSplit::MlaPreprocessOpsRunnerSplit called";
    needKernelGraphModify_ = true;
}
MlaPreprocessOpsRunnerSplit::~MlaPreprocessOpsRunnerSplit() {}

Status MlaPreprocessOpsRunnerSplit::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;
    kernelGraph_.inTensors.resize(IN_TENSOR_NUM_SPLIT);
    size_t inTensorStart = 0;
    Mki::Tensor &input = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &gamma0 = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &beta0 = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &quantScale0 = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &quantOffset0 = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &wdqkv = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &deScale0 = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &bias0 = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &gamma1 = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &beta1 = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &quantScale1 = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &quantOffset1 = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &wuq = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &deScale1 = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &bias1 = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &gamma2 = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &cos = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &sin = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &wuk = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &kvCache = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &kRope = kernelGraph_.inTensors.at(inTensorStart++);
    (void)kRope;
    Mki::Tensor &slotmapping = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &ctkvScale = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &qNopeScale = kernelGraph_.inTensors.at(inTensorStart++);
    kernelGraph_.outTensors.resize(OUT_TENSOR_NUM_SPLIT);
    size_t outTensorStart = 0;
    Mki::Tensor &qOut0 = kernelGraph_.outTensors.at(outTensorStart++);
    Mki::Tensor &kvCacheOut0 = kernelGraph_.outTensors.at(outTensorStart++);
    Mki::Tensor &qOut1 = kernelGraph_.outTensors.at(outTensorStart++);
    Mki::Tensor &kvCacheOut1 = kernelGraph_.outTensors.at(outTensorStart++);
    kernelGraph_.nodes.resize(1);
    auto &mlaPreprocessNode = kernelGraph_.nodes.at(0);
    AtbOps::OpParam::MlaPreprocess asdParam;
    asdParam.cacheMode = 0;
    mlaPreprocessNode.opDesc = {0, "MlaPreprocessOperation", asdParam};
    mlaPreprocessNode.inTensors = {&input,    &gamma0,   &beta0,       &quantScale0,  &quantOffset0, &wdqkv, &bias0,
                                   &gamma1,   &beta1,    &quantScale1, &quantOffset1, &gamma2,       &sin,   &cos,
                                   &sin,      &cos,      &kvCache,     &slotmapping,  &wuq,          &bias1, &wuk,
                                   &deScale0, &deScale1, &ctkvScale,   &qNopeScale};
    mlaPreprocessNode.outTensors = {&qOut0, &kvCacheOut0, &qOut1, &kvCacheOut1};
    return NO_ERROR;
}

Status MlaPreprocessOpsRunnerSplit::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    int64_t tokenNum = opsTensorPack.inTensors.at(0).desc.dims[0];
    int64_t headNum = opsTensorPack.inTensors.at(WUK_INDEX).desc.dims[0];
    ATB_LOG(INFO) << "MlaPreprocessOpsRunnerSplit::ModifyKernelGraph tokenNum: " << tokenNum << " headNum: " << headNum;

    AtbOps::OpParam::MlaPreprocess asdParam;
    asdParam.N = static_cast<uint32_t>(tokenNum);
    asdParam.headNum = static_cast<uint32_t>(headNum);
    asdParam.cacheMode = param_.cacheMode;
    asdParam.quantMode = static_cast<AtbOps::OpParam::MlaPreprocess::QuantMode>(param_.quantMode);
    auto &mlaPreprocessNode = kernelGraph_.nodes.at(0);
    mlaPreprocessNode.opDesc = {0, "MlaPreprocessOperation", asdParam};
    return NO_ERROR;
}
} // namespace atb