/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "stridedbatchmatmul_ops_runner.h"
#include <atb/utils/log.h>
#include <atbops/params/params.h>
#include "atb/utils/param_compare.h"


namespace atb {
static const uint64_t IN_TENSOR_COUNT = 2;
static const uint64_t OUT_TENSOR_COUNT = 1;

StridedBatchMatmulOpsRunner::StridedBatchMatmulOpsRunner(const train::StridedBatchMatmulParam &param)
    : OpsRunner("StridedBatchMatmulOpsRunner", RUNNER_TYPE_STRIDEDBATCHMATMUL), param_(param)
{
}
StridedBatchMatmulOpsRunner::~StridedBatchMatmulOpsRunner() {}

Status StridedBatchMatmulOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    int inId = 0;
    Mki::Tensor &x1Tensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &x2Tensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(1);
    int outId = 0;
    Mki::Tensor &resultTensor = kernelGraph_.outTensors.at(outId++);

    kernelGraph_.nodes.resize(1);
    auto &stridedBatchMatmulNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::StridedBatchMatmul asdParam;
    asdParam.transA = param_.transposeA;
    asdParam.transB = param_.transposeB;
    asdParam.batch = param_.batch;
    asdParam.headNum = param_.headNum;
    asdParam.m = param_.m;
    asdParam.n = param_.n;
    asdParam.k = param_.k;
    asdParam.lda = param_.lda;
    asdParam.ldb = param_.ldb;
    asdParam.ldc = param_.ldc;
    asdParam.strideA = param_.strideA;
    asdParam.strideB = param_.strideB;
    asdParam.strideC = param_.strideC;
    stridedBatchMatmulNode.opDesc = {0, "StridedBatchMatmulOperation", asdParam};
    stridedBatchMatmulNode.inTensors = {&x1Tensor, &x2Tensor};
    stridedBatchMatmulNode.outTensors = {&resultTensor};

    return NO_ERROR;
}

void StridedBatchMatmulOpsRunner::SetParam(const Mki::Any &param)
{
    train::StridedBatchMatmulParam newParam = Mki::AnyCast<train::StridedBatchMatmulParam>(param);
    if (!(newParam == param_)) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "StridedBatchMatmulOpsRunner Param Changed!";
        param_ = newParam;
        isParamUpdated_ = true;
    }
}
} // namespace atb