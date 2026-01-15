/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "linear_sparse_ops_runner.h"
#include "atb/utils/config.h"
#include "atb/utils/log.h"
#include "atb/utils/runner_util.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/singleton.h"

static constexpr size_t DIM_3 = 3;
static constexpr size_t SIZE_2 = 2;
static constexpr size_t SIZE_3 = 3;
static constexpr int64_t INT8_ALIGN = 32;

namespace atb {
LinearSparseOpsRunner::LinearSparseOpsRunner(const infer::LinearSparseParam &param)
    : OpsRunner("LinearSparseOpsRunner", RUNNER_TYPE_LINEAR_SPARSE), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearSparseOpsRunner::LinearSparseOpsRunner called";
}

LinearSparseOpsRunner::~LinearSparseOpsRunner() {}

Status LinearSparseOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;
    InitMemberVar();
    return SetupKernelGraph310p();
}

/**
 * 初始化runner的成员变量
 */
void LinearSparseOpsRunner::InitMemberVar()
{
    // merge axis
    Mki::SVector<int64_t> xTensorDims = kernelGraph_.inTensors.at(0).desc.dims;
    needMergeAxis_ = xTensorDims.size() == DIM_3;
    if (needMergeAxis_) {
        xTensorDims = {xTensorDims.at(0) * xTensorDims.at(1), xTensorDims.at(2)};
    }
    // init node params
    matmulParam_ = {param_.transposeA, param_.transposeB};
    int64_t m = param_.transposeA ? xTensorDims.at(1) : xTensorDims.at(0);
    int64_t k = param_.transposeA ? xTensorDims.at(0) : xTensorDims.at(1);
    int64_t n = kernelGraph_.inTensors.at(2).desc.dims.at(kernelGraph_.inTensors.at(2).desc.dims.size() - 1);
    matmulParam_.oriShape = {m, k, n};
    matmulParam_.withBias = true;
    matmulParam_.enDequant = true;
    matmulParam_.tilingN = param_.tilingN;
    matmulParam_.tilingK = param_.tilingK;
    matmulParam_.enShuffleK = GetSingleton<Config>().IsMatmulShuffleKEnable();
    transdataNd2NzParam_ = {AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ};
    transdataNz2NdParam_ = {AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND};
    transdataNz2NdParam_.outCrops = {m, n};
}

/**
 * 310P构图
 * @return 构图完成状态
 */
Status LinearSparseOpsRunner::SetupKernelGraph310p()
{
    // init inTensor/outTensor
    size_t inTensorId = 0;
    Mki::Tensor &inputTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &biasTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &scaleTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &idxTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);
    // init nodes
    kernelGraph_.nodes.resize(SIZE_3);
    size_t nodeId = 0;
    KernelGraphNode &transdataANode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &matmulNode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &transdataOutNode = kernelGraph_.nodes.at(nodeId++);
    // init internal tensors
    kernelGraph_.internalTensors.resize(SIZE_2);
    size_t internalTensorId = 0;
    Mki::Tensor &transdataAOutTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &matmulOutTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    // transdata input tensor to nz
    transdataANode.opDesc = {0, "TransdataOperation", transdataNd2NzParam_};
    transdataANode.inTensors = {&inputTensor};
    transdataANode.outTensors = {&transdataAOutTensor};
    RunnerUtil::TransdataSqueeze(transdataANode, needMergeAxis_);
    // matmul
    matmulNode.opDesc = {0, "MatMulOperation", matmulParam_};
    matmulNode.inTensors = {&transdataAOutTensor, &weightTensor, &biasTensor, &scaleTensor, &idxTensor};
    matmulNode.outTensors = {&matmulOutTensor};
    RunnerUtil::ConfigViewFuncs(matmulNode, needMergeAxis_);
    matmulNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        size_t inTensorId = 1;
        Mki::Tensor &runInfoWeight = launchParam.GetInTensor(inTensorId++);
        Mki::Tensor &runInfoBias = launchParam.GetInTensor(inTensorId++);
        Mki::Tensor &runInfoDeqScale = launchParam.GetInTensor(inTensorId++);
        Mki::Tensor &runInfoCompressIdx = launchParam.GetInTensor(inTensorId++);
        runInfoWeight.desc.format = Mki::TENSOR_FORMAT_FRACTAL_NZ;
        int64_t k1 = RunnerUtil::AlignUp(matmulParam_.oriShape.at(1), INT8_ALIGN) / INT8_ALIGN;
        int64_t n1 = RunnerUtil::AlignUp(matmulParam_.oriShape.at(2), DEFAULT_ALIGN);
        runInfoWeight.desc.dims = {1, k1, n1, INT8_ALIGN};
        if (runInfoBias.desc.dims.size() == 1) {
            runInfoBias.desc.dims = {1, runInfoBias.desc.dims.at(0)};
        }
        runInfoDeqScale.desc.dtype = Mki::TENSOR_DTYPE_UINT64;
        if (runInfoDeqScale.desc.dims.size() == 1) {
            runInfoDeqScale.desc.dims = {1, runInfoDeqScale.desc.dims.at(0)};
        }
        runInfoCompressIdx.desc.dims = {1, runInfoCompressIdx.desc.dims.at(0)};
    };
    // transdata out tensor to nd
    transdataOutNode.opDesc = {0, "TransdataOperation", transdataNz2NdParam_};
    transdataOutNode.inTensors = {&matmulOutTensor};
    transdataOutNode.outTensors = {&outTensor};

    return NO_ERROR;
}
} // namespace atb