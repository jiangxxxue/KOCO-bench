/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "linear_ops_runner.h"
#include "atb/utils/config.h"
#include "atb/utils/log.h"
#include "atb/utils/runner_util.h"
#include "atb/utils/singleton.h"
#include "atb/utils/utils_internal.h"

static constexpr size_t SIZE_2 = 2;
static constexpr size_t SIZE_3 = 3;
static constexpr size_t SIZE_4 = 4;
static constexpr size_t SIZE_5 = 5;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;
static constexpr int64_t DEFAULT_ALIGN = 16;
static constexpr int64_t INT8_ALIGN = 32;
static constexpr int64_t MATMUL_TRANSPOSE_THRESHOLD = 65535;

namespace atb {
LinearOpsRunner::LinearOpsRunner(const infer::LinearParam &param)
    : OpsRunner("LinearOpsRunner", RUNNER_TYPE_LINEAR), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearOpsRunner::LinearOpsRunner";

    elewiseAddParam_.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_ADD;
    transdataNdToNzParam_.transdataType = AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ;
    transdataNzToNdParam_.transdataType = AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND;

    matmulMergeAxis_ = [&](const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims) {
        if (oldDims.size() == SIZE_3) {
            newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(DIM_2)};
            ATB_LOG(INFO) << " merge axis - before: " << oldDims << "; after: " << newDims;
        } else {
            newDims = oldDims;
        }
    };
    matmulNzReshape_ = [&](const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims) {
        int64_t align = matmulParam_.enDequant ? INT8_ALIGN : DEFAULT_ALIGN;
        if (oldDims.size() == SIZE_2) {
            newDims = {1, UtilsInternal::AlignUp(oldDims.at(1), align) / align,
                       UtilsInternal::AlignUp(oldDims.at(0), DEFAULT_ALIGN), align};
            ATB_LOG(INFO) << " nz reshape - before: " << oldDims << "; after: " << newDims;
        } else if (oldDims.size() == SIZE_3) {
            newDims = {oldDims.at(0), UtilsInternal::AlignUp(oldDims.at(DIM_2), align) / align,
                       UtilsInternal::AlignUp(oldDims.at(1), DEFAULT_ALIGN), align};
            ATB_LOG(INFO) << " nz reshape - before: " << oldDims << "; after: " << newDims;
        } else {
            newDims = oldDims;
        }
    };
    transdataUnsqueeze_ = [&](const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims) {
        if (oldDims.size() == SIZE_2) {
            newDims = {1, oldDims.at(0), oldDims.at(1)};
            ATB_LOG(INFO) << GetLogPrefix() << "transdata unsqueeze - before: " << oldDims << "; after: " << newDims;
        } else if (oldDims.size() == SIZE_3) {
            newDims = {1, oldDims.at(0) * oldDims.at(1), oldDims.at(DIM_2)};
            ATB_LOG(INFO) << GetLogPrefix() << "transdata unsqueeze - before: " << oldDims << "; after: " << newDims;
        } else {
            newDims = oldDims;
        }
    };
    elewiseAddUnsqueeze_ = [&](const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims) {
        if (oldDims.size() == SIZE_2 && oldDims.at(0) != 1) {
            newDims = {oldDims.at(0), 1, oldDims.at(1)};
            ATB_LOG(INFO) << GetLogPrefix() << "elewiseAdd unsqueeze - before: " << oldDims << "; after: " << newDims;
        } else {
            newDims = oldDims;
        }
    };
}

LinearOpsRunner::~LinearOpsRunner() {}

Status LinearOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;

    matmulParam_.transposeA = param_.transposeA;
    matmulParam_.transposeB = param_.transposeB;
    matmulParam_.enShuffleK = GetSingleton<Config>().IsMatmulShuffleKEnable();
    matmulParam_.enDequant = param_.outDataType != ACL_DT_UNDEFINED;
    matmulParam_.outDtype = static_cast<Mki::TensorDType>(param_.outDataType);

    size_t inTensorId = 0;
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(inTensorId++);
    isWeightNz_ = weightTensor.desc.format == Mki::TENSOR_FORMAT_FRACTAL_NZ;
    SetupNeedMergeAxis(xTensor, weightTensor);
    SetupMatmulOriShape(xTensor, weightTensor);
    transdataNzToNdParam_.outCrops = {matmulParam_.oriShape.at(0), matmulParam_.oriShape.at(2)};
    if (matmulParam_.enDequant) {
        if (GetSingleton<Config>().Is910B() && param_.quantMode == infer::LinearParam::PER_TOKEN) {
                return SetupKernelGraphMatmulDequantPerToken910B();
        } else if (GetSingleton<Config>().Is910B() || GetSingleton<Config>().Is310B()) {
            return SetupKernelGraphMatmulDequant910B();
        }
        return isWeightNz_ ? SetupKernelGraphMatmulDequantWeightNzNot910B() :
                             SetupKernelGraphMatmulDequantWeightNdNot910B();
    }
    if (param_.enAccum) {
        return SetupKernelGraphMatmulAccum();
    }
    if (param_.matmulType == infer::LinearParam::MATMUL_EIN_SUM) {
        return param_.hasBias ? SetupKernelGraphMatmulEinElewiseAdd() : SetupKernelGraphMatmulEin();
    }
    if (param_.hasBias) {
        Mki::Tensor &biasTensor = kernelGraph_.inTensors.at(inTensorId++);
        if (biasTensor.desc.dtype == Mki::TENSOR_DTYPE_FLOAT) {
            return SetupKernelGraphMatmulWithBias();
        }
        if (GetSingleton<Config>().Is910B() || GetSingleton<Config>().Is310B()) {
            return SetupKernelGraphMatmulElewiseAdd910B();
        }
        return isWeightNz_ ? SetupKernelGraphMatmulElewiseAddWeightNzNot910B() :
                             SetupKernelGraphMatmulElewiseAddWeightNdNot910B();
    } else {
        if (xTensor.desc.dtype == Mki::TENSOR_DTYPE_FLOAT && weightTensor.desc.dtype == Mki::TENSOR_DTYPE_FLOAT) {
            moeGateCorrParam_.transposeA = param_.transposeA;
            moeGateCorrParam_.transposeB = param_.transposeB;
            return SetupKernelGraphMoeGateCorr();
        }
        if (GetSingleton<Config>().Is910B() || GetSingleton<Config>().Is310B()) {
            return SetupKernelGraphMatmul910B();
        }
        return isWeightNz_ ? SetupKernelGraphMatmulWeightNzNot910B() : SetupKernelGraphMatmulWeightNdNot910B();
    }
}

Status LinearOpsRunner::SetupKernelGraphMatmul910B()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearOpsRunner::SetupKernelGraphMatmulA2";

    InitKernelGraph(SIZE_2, 1, 0, 1);

    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(0);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(1);

    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    KernelGraphNode &matmulNode = kernelGraph_.nodes.at(0);

    matmulNode.opDesc = {0, "MatMulOperation", matmulParam_};
    matmulNode.inTensors = {&xTensor, &weightTensor};
    matmulNode.outTensors = {&outTensor};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    if (xNeedMergeAxis_) {
        matmulNode.inTensorViewFuncs.at(0) = matmulMergeAxis_;
    }
    if (isWeightNz_) {
        matmulNode.inTensorViewFuncs.at(1) = matmulNzReshape_;
    } else if (weightNeedMergeAxis_) {
        matmulNode.inTensorViewFuncs.at(1) = matmulMergeAxis_;
    }

    return NO_ERROR;
}

Status LinearOpsRunner::SetupKernelGraphMatmulWeightNdNot910B()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearOpsRunner::SetupKernelGraphMatmulWeightNdNotA2";

    InitKernelGraph(SIZE_2, 1, SIZE_3, SIZE_4);

    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(0);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(1);

    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    size_t internalTensorId = 0;
    Mki::Tensor &transdataAOutTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &transdataBOutTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &matmulOutTensor = kernelGraph_.internalTensors.at(internalTensorId++);

    size_t nodeId = 0;
    KernelGraphNode &transdataANode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &transdataBNode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &matmulNode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &transdataOutNode = kernelGraph_.nodes.at(nodeId++);

    transdataANode.opDesc = {0, "TransdataOperation", transdataNdToNzParam_};
    transdataANode.inTensors = {&xTensor};
    transdataANode.outTensors = {&transdataAOutTensor};
    transdataANode.inTensorViewFuncs.resize(transdataANode.inTensors.size());
    if (xTensor.desc.dims.size() == SIZE_2 || xNeedMergeAxis_) {
        transdataANode.inTensorViewFuncs.at(0) = transdataUnsqueeze_;
    }

    transdataBNode.opDesc = {0, "TransdataOperation", transdataNdToNzParam_};
    transdataBNode.inTensors = {&weightTensor};
    transdataBNode.outTensors = {&transdataBOutTensor};
    transdataBNode.inTensorViewFuncs.resize(transdataBNode.inTensors.size());
    if (weightTensor.desc.dims.size() == SIZE_2) {
        transdataBNode.inTensorViewFuncs.at(0) = transdataUnsqueeze_;
    }

    matmulNode.opDesc = {0, "MatMulOperation", matmulParam_};
    matmulNode.inTensors = {&transdataAOutTensor, &transdataBOutTensor};
    matmulNode.outTensors = {&matmulOutTensor};

    transdataOutNode.opDesc = {0, "TransdataOperation", transdataNzToNdParam_};
    transdataOutNode.inTensors = {&matmulOutTensor};
    transdataOutNode.outTensors = {&outTensor};

    return NO_ERROR;
}

Status LinearOpsRunner::SetupKernelGraphMatmulWeightNzNot910B()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearOpsRunner::SetupKernelGraphMatmulWeightNzNotA2";

    InitKernelGraph(SIZE_2, 1, SIZE_2, SIZE_3);

    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(0);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(1);

    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    Mki::Tensor &transdataAOutTensor = kernelGraph_.internalTensors.at(0);
    Mki::Tensor &matmulOutTensor = kernelGraph_.internalTensors.at(1);

    size_t nodeId = 0;
    KernelGraphNode &transdataANode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &matmulNode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &transdataOutNode = kernelGraph_.nodes.at(nodeId++);

    transdataANode.opDesc = {0, "TransdataOperation", transdataNdToNzParam_};
    transdataANode.inTensors = {&xTensor};
    transdataANode.outTensors = {&transdataAOutTensor};
    transdataANode.inTensorViewFuncs.resize(transdataANode.inTensors.size());
    if (xTensor.desc.dims.size() == SIZE_2 || xNeedMergeAxis_) {
        transdataANode.inTensorViewFuncs.at(0) = transdataUnsqueeze_;
    }

    matmulNode.opDesc = {0, "MatMulOperation", matmulParam_};
    matmulNode.inTensors = {&transdataAOutTensor, &weightTensor};
    matmulNode.outTensors = {&matmulOutTensor};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    if (isWeightNz_) {
        matmulNode.inTensorViewFuncs.at(1) = matmulNzReshape_;
    }

    transdataOutNode.opDesc = {0, "TransdataOperation", transdataNzToNdParam_};
    transdataOutNode.inTensors = {&matmulOutTensor};
    transdataOutNode.outTensors = {&outTensor};

    return NO_ERROR;
}

Status LinearOpsRunner::SetupKernelGraphMatmulElewiseAdd910B()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearOpsRunner::SetupKernelGraphMatmulElewiseAddA2";

    InitKernelGraph(SIZE_3, 1, 1, SIZE_2);

    size_t inTensorId = 0;
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &biasTensor = kernelGraph_.inTensors.at(inTensorId++);

    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    Mki::Tensor &matmulOutTensor = kernelGraph_.internalTensors.at(0);

    KernelGraphNode &matmulNode = kernelGraph_.nodes.at(0);
    KernelGraphNode &addNode = kernelGraph_.nodes.at(1);

    matmulNode.opDesc = {0, "MatMulOperation", matmulParam_};
    matmulNode.inTensors = {&xTensor, &weightTensor};
    matmulNode.outTensors = {&matmulOutTensor};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    if (xNeedMergeAxis_) {
        matmulNode.inTensorViewFuncs.at(0) = matmulMergeAxis_;
    }
    if (isWeightNz_) {
        matmulNode.inTensorViewFuncs.at(1) = matmulNzReshape_;
    } else if (weightNeedMergeAxis_) {
        matmulNode.inTensorViewFuncs.at(1) = matmulMergeAxis_;
    }

    addNode.opDesc = {0, "ElewiseOperation", elewiseAddParam_};
    addNode.inTensors = {&matmulOutTensor, &biasTensor};
    addNode.outTensors = {&outTensor};
    addNode.inTensorViewFuncs.resize(addNode.inTensors.size());
    addNode.inTensorViewFuncs.at(1) = elewiseAddUnsqueeze_;

    return NO_ERROR;
}

Status LinearOpsRunner::SetupKernelGraphMatmulElewiseAddWeightNdNot910B()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearOpsRunner::SetupKernelGraphMatmulElewiseAddWeightNdNotA2";

    InitKernelGraph(SIZE_3, 1, SIZE_4, SIZE_5);

    size_t inTensorId = 0;
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &biasTensor = kernelGraph_.inTensors.at(inTensorId++);

    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    size_t internalTensorId = 0;
    Mki::Tensor &transdataAOutTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &transdataBOutTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &matmulOutTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &transdataOutTensor = kernelGraph_.internalTensors.at(internalTensorId++);

    size_t nodeId = 0;
    KernelGraphNode &transdataANode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &transdataBNode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &matmulNode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &transdataOutNode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &addNode = kernelGraph_.nodes.at(nodeId++);

    transdataANode.opDesc = {0, "TransdataOperation", transdataNdToNzParam_};
    transdataANode.inTensors = {&xTensor};
    transdataANode.outTensors = {&transdataAOutTensor};
    transdataANode.inTensorViewFuncs.resize(transdataANode.inTensors.size());
    if (xTensor.desc.dims.size() == SIZE_2 || xNeedMergeAxis_) {
        transdataANode.inTensorViewFuncs.at(0) = transdataUnsqueeze_;
    }

    transdataBNode.opDesc = {0, "TransdataOperation", transdataNdToNzParam_};
    transdataBNode.inTensors = {&weightTensor};
    transdataBNode.outTensors = {&transdataBOutTensor};
    transdataBNode.inTensorViewFuncs.resize(transdataBNode.inTensors.size());
    if (weightTensor.desc.dims.size() == SIZE_2) {
        transdataBNode.inTensorViewFuncs.at(0) = transdataUnsqueeze_;
    }

    matmulNode.opDesc = {0, "MatMulOperation", matmulParam_};
    matmulNode.inTensors = {&transdataAOutTensor, &transdataBOutTensor};
    matmulNode.outTensors = {&matmulOutTensor};

    transdataOutNode.opDesc = {0, "TransdataOperation", transdataNzToNdParam_};
    transdataOutNode.inTensors = {&matmulOutTensor};
    transdataOutNode.outTensors = {&transdataOutTensor};

    addNode.opDesc = {0, "ElewiseOperation", elewiseAddParam_};
    addNode.inTensors = {&transdataOutTensor, &biasTensor};
    addNode.outTensors = {&outTensor};
    addNode.inTensorViewFuncs.resize(addNode.inTensors.size());
    addNode.inTensorViewFuncs.at(1) = elewiseAddUnsqueeze_;

    return NO_ERROR;
}

Status LinearOpsRunner::SetupKernelGraphMatmulElewiseAddWeightNzNot910B()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearOpsRunner::SetupKernelGraphMatmulElewiseAddWeightNzNotA2";

    InitKernelGraph(SIZE_3, 1, SIZE_3, SIZE_4);

    size_t inTensorId = 0;
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &biasTensor = kernelGraph_.inTensors.at(inTensorId++);

    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    size_t internalTensorId = 0;
    Mki::Tensor &transdataAOutTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &matmulOutTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &transdataOutTensor = kernelGraph_.internalTensors.at(internalTensorId++);

    size_t nodeId = 0;
    KernelGraphNode &transdataANode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &matmulNode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &transdataOutNode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &addNode = kernelGraph_.nodes.at(nodeId++);

    transdataANode.opDesc = {0, "TransdataOperation", transdataNdToNzParam_};
    transdataANode.inTensors = {&xTensor};
    transdataANode.outTensors = {&transdataAOutTensor};
    transdataANode.inTensorViewFuncs.resize(transdataANode.inTensors.size());
    if (xTensor.desc.dims.size() == SIZE_2 || xNeedMergeAxis_) {
        transdataANode.inTensorViewFuncs.at(0) = transdataUnsqueeze_;
    }

    matmulNode.opDesc = {0, "MatMulOperation", matmulParam_};
    matmulNode.inTensors = {&transdataAOutTensor, &weightTensor};
    matmulNode.outTensors = {&matmulOutTensor};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    if (isWeightNz_) {
        matmulNode.inTensorViewFuncs.at(1) = matmulNzReshape_;
    }

    transdataOutNode.opDesc = {0, "TransdataOperation", transdataNzToNdParam_};
    transdataOutNode.inTensors = {&matmulOutTensor};
    transdataOutNode.outTensors = {&transdataOutTensor};

    addNode.opDesc = {0, "ElewiseOperation", elewiseAddParam_};
    addNode.inTensors = {&transdataOutTensor, &biasTensor};
    addNode.outTensors = {&outTensor};
    addNode.inTensorViewFuncs.resize(addNode.inTensors.size());
    addNode.inTensorViewFuncs.at(1) = elewiseAddUnsqueeze_;

    return NO_ERROR;
}

Status LinearOpsRunner::SetupKernelGraphMatmulWithBias()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearOpsRunner::SetupKernelGraphMatmulWithBias";

    InitKernelGraph(SIZE_3, 1, 0, 1);

    size_t inTensorId = 0;
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &biasTensor = kernelGraph_.inTensors.at(inTensorId++);

    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    KernelGraphNode &matmulNode = kernelGraph_.nodes.at(0);

    matmulParam_.withBias = true;
    matmulParam_.matmulType = AsdOps::OpParam::MatMul::MatMulType::MATMUL_WITH_BIAS;
    matmulNode.opDesc = {0, "MatMulOperation", matmulParam_};
    matmulNode.inTensors = {&xTensor, &weightTensor, &biasTensor};
    matmulNode.outTensors = {&outTensor};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    if (xNeedMergeAxis_) {
        matmulNode.inTensorViewFuncs.at(0) = matmulMergeAxis_;
    }
    if (weightNeedMergeAxis_) {
        matmulNode.inTensorViewFuncs.at(1) = matmulMergeAxis_;
    }

    return NO_ERROR;
}

Status LinearOpsRunner::SetupKernelGraphMatmulAccum()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearOpsRunner::SetupKernelGraphMatmulAccum";
    isParamUpdated_ = false;

    kernelGraph_.inTensors.resize(SIZE_3);

    size_t inTensorId = 0;
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &accumTensor = kernelGraph_.inTensors.at(inTensorId++);

    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    if (xTensor.desc.dims.size() == SIZE_2 && xTensor.desc.dims.at(1) > MATMUL_TRANSPOSE_THRESHOLD) {
        if (kernelGraph_.nodes.size() != 2) {
            isParamUpdated_ = true;
        }
        InitKernelGraph(SIZE_3, 1, 1, 2);
        Mki::Tensor &transposedXtensor = kernelGraph_.internalTensors.at(0);

        size_t nodeId = 0;
        KernelGraphNode &transposeNode = kernelGraph_.nodes.at(nodeId++);
        KernelGraphNode &matmulNode = kernelGraph_.nodes.at(nodeId++);

        AsdOps::OpParam::Transpose tranposeparam = {{1, 0}};
        transposeNode.opDesc = {0, "TransposeOperation", tranposeparam};
        transposeNode.inTensors = {&xTensor};
        transposeNode.outTensors = {&transposedXtensor};

        bool matmulTransposeA  = !param_.transposeA;
        matmulParam_.transposeA = matmulTransposeA;
        SetupMatmulOriShape(transposedXtensor, weightTensor);

        matmulParam_.matmulType = AsdOps::OpParam::MatMul::MatMulType::MATMUL_ACCUM_ATOMIC;
        matmulNode.opDesc = {1, "MatMulOperation", matmulParam_};
        matmulNode.inTensors = {&transposedXtensor, &weightTensor, &accumTensor};
        matmulNode.outTensors = {&outTensor};
        matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
        if (xNeedMergeAxis_) {
            matmulNode.inTensorViewFuncs.at(0) = matmulMergeAxis_;
        }
    } else {
        if (kernelGraph_.nodes.size() != 1) {
            isParamUpdated_ = true;
        }
        InitKernelGraph(SIZE_3, 1, 0, 1);
        KernelGraphNode &matmulNode = kernelGraph_.nodes.at(0);

        matmulParam_.matmulType = AsdOps::OpParam::MatMul::MatMulType::MATMUL_ACCUM_ATOMIC;
        matmulNode.opDesc = {0, "MatMulOperation", matmulParam_};
        matmulNode.inTensors = {&xTensor, &weightTensor, &accumTensor};
        matmulNode.outTensors = {&outTensor};
        matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
        if (xNeedMergeAxis_) {
            matmulNode.inTensorViewFuncs.at(0) = matmulMergeAxis_;
        }
    }
    return NO_ERROR;
}

Status LinearOpsRunner::SetupKernelGraphMatmulEin()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearOpsRunner::SetupKernelGraphMatmulEin";
 
    InitKernelGraph(SIZE_2, 1, 0, 1);
 
    size_t inTensorId = 0;
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(inTensorId++);
 
    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);
 
    KernelGraphNode &matmulNode = kernelGraph_.nodes.at(0);
 
    matmulParam_.matmulType = AsdOps::OpParam::MatMul::MatMulType::MATMUL_EIN_SUM;
    matmulNode.opDesc = { 0, "MatMulOperation", matmulParam_ };
    matmulNode.inTensors = { &xTensor, &weightTensor};
    matmulNode.outTensors = { &outTensor };
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    if (isWeightNz_) {
        matmulNode.inTensorViewFuncs.at(1) = matmulNzReshape_;
    }
 
    return NO_ERROR;
}

Status LinearOpsRunner::SetupKernelGraphMatmulEinElewiseAdd()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearOpsRunner::SetupKernelGraphMatmulEinElewiseAdd";
 
    InitKernelGraph(SIZE_3, 1, 1, SIZE_2);
 
    size_t inTensorId = 0;
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &biasTensor = kernelGraph_.inTensors.at(inTensorId++);
 
    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    Mki::Tensor &matmuloutTensor = kernelGraph_.internalTensors.at(0);
 
    KernelGraphNode &matmulNode = kernelGraph_.nodes.at(0);
    KernelGraphNode &addNode = kernelGraph_.nodes.at(1);
 
    matmulParam_.matmulType = AsdOps::OpParam::MatMul::MatMulType::MATMUL_EIN_SUM;
    matmulNode.opDesc = { 0, "MatMulOperation", matmulParam_ };
    matmulNode.inTensors = { &xTensor, &weightTensor};
    matmulNode.outTensors = { &matmuloutTensor };
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    if (isWeightNz_) {
        matmulNode.inTensorViewFuncs.at(1) = matmulNzReshape_;
    }

    addNode.opDesc = {0, "ElewiseOperation", elewiseAddParam_};
    addNode.inTensors = {&matmuloutTensor, &biasTensor};
    addNode.outTensors = {&outTensor};
    addNode.inTensorViewFuncs.resize(addNode.inTensors.size());
    addNode.inTensorViewFuncs.at(1) = elewiseAddUnsqueeze_;
 
    return NO_ERROR;
}

Status LinearOpsRunner::SetupKernelGraphMatmulDequant910B()
{
    ATB_LOG(INFO) << GetLogPrefix() << "SetupKernelGraphMatmulDequantA2";

    size_t inTensorNum = param_.hasBias ? SIZE_4 : SIZE_3;
    if (param_.hasBias) {
        matmulParam_.quantMode = AsdOps::OpParam::MatMul::QuantMode::PER_CHANNEL_ASYMM;
    } else {
        matmulParam_.quantMode = AsdOps::OpParam::MatMul::QuantMode::PER_CHANNEL_SYMM;
    }

    InitKernelGraph(inTensorNum, 1, 0, 1);

    size_t inTensorId = 0;
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &biasTensor = param_.hasBias ? kernelGraph_.inTensors.at(inTensorId++) : nullTensor_;
    Mki::Tensor &descaleTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &perTokenScaleTensor = nullTensor_;

    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    KernelGraphNode &matmulNode = kernelGraph_.nodes.at(0);

    matmulParam_.withBias = param_.hasBias;
    matmulNode.opDesc = {0, "MatMulOperation", matmulParam_};
    matmulNode.inTensors = {&xTensor, &weightTensor, &biasTensor, &descaleTensor, &perTokenScaleTensor};
    matmulNode.outTensors = {&outTensor};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    if (xNeedMergeAxis_) {
        matmulNode.inTensorViewFuncs.at(0) = matmulMergeAxis_;
    }
    if (isWeightNz_) {
        matmulNode.inTensorViewFuncs.at(1) = matmulNzReshape_;
    } else if (weightNeedMergeAxis_) {
        matmulNode.inTensorViewFuncs.at(1) = matmulMergeAxis_;
    }

    return NO_ERROR;
}

Status LinearOpsRunner::SetupKernelGraphMatmulDequantPerToken910B()
{
    ATB_LOG(INFO) << GetLogPrefix() << "SetupKernelGraphMatmulDequantPerTokenA2";

    size_t inTensorNum = param_.hasBias ? SIZE_5 : SIZE_4;
    InitKernelGraph(inTensorNum, 1, 0, 1);
    
    matmulParam_.quantMode = AsdOps::OpParam::MatMul::QuantMode::PER_TOKEN_SYMM;

    size_t inTensorId = 0;
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &biasTensor = param_.hasBias ? kernelGraph_.inTensors.at(inTensorId++) : nullTensor_;
    Mki::Tensor &descaleTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &perTokenScaleTensor = kernelGraph_.inTensors.at(inTensorId++);

    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    KernelGraphNode &matmulNode = kernelGraph_.nodes.at(0);

    matmulParam_.withBias = param_.hasBias;
    matmulNode.opDesc = {0, "MatMulOperation", matmulParam_};
    matmulNode.inTensors = {&xTensor, &weightTensor, &biasTensor, &descaleTensor, &perTokenScaleTensor};
    matmulNode.outTensors = {&outTensor};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    if (xNeedMergeAxis_) {
        matmulNode.inTensorViewFuncs.at(0) = matmulMergeAxis_;
    }
    if (isWeightNz_) {
        matmulNode.inTensorViewFuncs.at(1) = matmulNzReshape_;
    } else if (weightNeedMergeAxis_) {
        matmulNode.inTensorViewFuncs.at(1) = matmulMergeAxis_;
    }

    return NO_ERROR;
}

Status LinearOpsRunner::SetupKernelGraphMatmulDequantWeightNdNot910B()
{
    ATB_LOG(INFO) << GetLogPrefix() << "SetupKernelGraphMatmulDequantWeightNdNotA2";

    size_t inTensorNum = param_.hasBias ? SIZE_4 : SIZE_3;
    if (param_.hasBias) {
        matmulParam_.quantMode = AsdOps::OpParam::MatMul::QuantMode::PER_CHANNEL_ASYMM;
    } else {
        matmulParam_.quantMode = AsdOps::OpParam::MatMul::QuantMode::PER_CHANNEL_SYMM;
    }
    InitKernelGraph(inTensorNum, 1, SIZE_3, SIZE_4);

    size_t inTensorId = 0;
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &biasTensor = param_.hasBias ? kernelGraph_.inTensors.at(inTensorId++) : nullTensor_;
    Mki::Tensor &deqScaleTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &perTokenScaleTensor = nullTensor_;

    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    size_t internalTensorId = 0;
    Mki::Tensor &transdataAOutTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &transdataBOutTensor = kernelGraph_.internalTensors.at(internalTensorId++);
    Mki::Tensor &matmulOutTensor = kernelGraph_.internalTensors.at(internalTensorId++);

    size_t nodeId = 0;
    KernelGraphNode &transdataANode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &transdataBNode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &matmulNode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &transdataOutNode = kernelGraph_.nodes.at(nodeId++);

    transdataANode.opDesc = {0, "TransdataOperation", transdataNdToNzParam_};
    transdataANode.inTensors = {&xTensor};
    transdataANode.outTensors = {&transdataAOutTensor};
    transdataANode.inTensorViewFuncs.resize(transdataANode.inTensors.size());
    if (xTensor.desc.dims.size() == SIZE_2 || xNeedMergeAxis_) {
        transdataANode.inTensorViewFuncs.at(0) = transdataUnsqueeze_;
    }

    transdataBNode.opDesc = {0, "TransdataOperation", transdataNdToNzParam_};
    transdataBNode.inTensors = {&weightTensor};
    transdataBNode.outTensors = {&transdataBOutTensor};
    transdataBNode.inTensorViewFuncs.resize(transdataBNode.inTensors.size());
    if (weightTensor.desc.dims.size() == SIZE_2) {
        transdataBNode.inTensorViewFuncs.at(0) = transdataUnsqueeze_;
    }

    matmulParam_.withBias = param_.hasBias;
    matmulNode.opDesc = {0, "MatMulOperation", matmulParam_};
    matmulNode.inTensors = {&transdataAOutTensor, &transdataBOutTensor, &biasTensor, &deqScaleTensor, &perTokenScaleTensor};
    matmulNode.outTensors = {&matmulOutTensor};

    transdataOutNode.opDesc = {0, "TransdataOperation", transdataNzToNdParam_};
    transdataOutNode.inTensors = {&matmulOutTensor};
    transdataOutNode.outTensors = {&outTensor};

    return NO_ERROR;
}

Status LinearOpsRunner::SetupKernelGraphMatmulDequantWeightNzNot910B()
{
    ATB_LOG(INFO) << GetLogPrefix() << "SetupKernelGraphMatmulDequantWeightNzNotA2";

    size_t inTensorNum = param_.hasBias ? SIZE_4 : SIZE_3;
    if (param_.hasBias) {
        matmulParam_.quantMode = AsdOps::OpParam::MatMul::QuantMode::PER_CHANNEL_ASYMM;
    } else {
        matmulParam_.quantMode = AsdOps::OpParam::MatMul::QuantMode::PER_CHANNEL_SYMM;
    }
    InitKernelGraph(inTensorNum, 1, SIZE_2, SIZE_3);

    size_t inTensorId = 0;
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &biasTensor = param_.hasBias ? kernelGraph_.inTensors.at(inTensorId++) : nullTensor_;
    Mki::Tensor &deqScaleTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &perTokenScaleTensor = nullTensor_;

    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    Mki::Tensor &transdataAOutTensor = kernelGraph_.internalTensors.at(0);
    Mki::Tensor &matmulOutTensor = kernelGraph_.internalTensors.at(1);

    size_t nodeId = 0;
    KernelGraphNode &transdataANode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &matmulNode = kernelGraph_.nodes.at(nodeId++);
    KernelGraphNode &transdataOutNode = kernelGraph_.nodes.at(nodeId++);

    transdataANode.opDesc = {0, "TransdataOperation", transdataNdToNzParam_};
    transdataANode.inTensors = {&xTensor};
    transdataANode.outTensors = {&transdataAOutTensor};
    transdataANode.inTensorViewFuncs.resize(transdataANode.inTensors.size());
    if (xTensor.desc.dims.size() == SIZE_2 || xNeedMergeAxis_) {
        transdataANode.inTensorViewFuncs.at(0) = transdataUnsqueeze_;
    }

    matmulParam_.withBias = param_.hasBias;
    matmulNode.opDesc = {0, "MatMulOperation", matmulParam_};
    matmulNode.inTensors = {&transdataAOutTensor, &weightTensor, &biasTensor, &deqScaleTensor, &perTokenScaleTensor};
    matmulNode.outTensors = {&matmulOutTensor};
    matmulNode.inTensorViewFuncs.resize(matmulNode.inTensors.size());
    if (isWeightNz_) {
        matmulNode.inTensorViewFuncs.at(1) = matmulNzReshape_;
    }

    transdataOutNode.opDesc = {0, "TransdataOperation", transdataNzToNdParam_};
    transdataOutNode.inTensors = {&matmulOutTensor};
    transdataOutNode.outTensors = {&outTensor};

    return NO_ERROR;
}

Status LinearOpsRunner::SetupKernelGraphMoeGateCorr()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearOpsRunner::SetupKernelGraphMoeGateCorr";

    InitKernelGraph(SIZE_2, 1, 0, 1);

    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(0);
    Mki::Tensor &weightTensor = kernelGraph_.inTensors.at(1);
    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    KernelGraphNode &matmulNode = kernelGraph_.nodes.at(0);

    matmulNode.opDesc = {0, "MoeGateCorrOperation", moeGateCorrParam_};
    matmulNode.inTensors = {&xTensor, &weightTensor};
    matmulNode.outTensors = {&outTensor};

    return NO_ERROR;
}

void LinearOpsRunner::SetupMatmulOriShape(const Mki::Tensor &xTensor, const Mki::Tensor &weightTensor)
{
    if (param_.matmulType == infer::LinearParam::MATMUL_EIN_SUM) {
        SetupMatmulOriShapeEin(xTensor, weightTensor);
    } else {
        int64_t &m = matmulParam_.oriShape.at(0);
        int64_t &k = matmulParam_.oriShape.at(1);
        int64_t &n = matmulParam_.oriShape.at(DIM_2);
        if (xTensor.desc.dims.size() == SIZE_2) {
            m = param_.transposeA ? xTensor.desc.dims.at(1) : xTensor.desc.dims.at(0);
            k = param_.transposeA ? xTensor.desc.dims.at(0) : xTensor.desc.dims.at(1);
        } else if (xTensor.desc.dims.size() == SIZE_3) {
            m = param_.transposeA ? xTensor.desc.dims.at(DIM_2) : xTensor.desc.dims.at(1);
            if (xNeedMergeAxis_) {
                m *= xTensor.desc.dims.at(0);
            }
            k = param_.transposeA ? xTensor.desc.dims.at(1) : xTensor.desc.dims.at(DIM_2);
        }
        if (weightTensor.desc.dims.size() == SIZE_2) {
            n = param_.transposeB ? weightTensor.desc.dims.at(0) : weightTensor.desc.dims.at(1);
        } else if (weightTensor.desc.dims.size() == SIZE_3) {
            n = param_.transposeB ? weightTensor.desc.dims.at(1) : weightTensor.desc.dims.at(DIM_2);
        } else if (weightTensor.desc.dims.size() == SIZE_4) {
            n = param_.transposeB ? weightTensor.desc.dims.at(DIM_2) :
                                    weightTensor.desc.dims.at(1) * weightTensor.desc.dims.at(DIM_3);
        }
    }
}

void LinearOpsRunner::SetupMatmulOriShapeEin(const Mki::Tensor &xTensor, const Mki::Tensor &weightTensor)
{
    int64_t &m = matmulParam_.oriShape.at(0);
    int64_t &k = matmulParam_.oriShape.at(1);
    int64_t &n = matmulParam_.oriShape.at(DIM_2);
    m = param_.transposeA ? xTensor.desc.dims.at(DIM_2) : xTensor.desc.dims.at(0);
    k = param_.transposeA ? xTensor.desc.dims.at(0) : xTensor.desc.dims.at(DIM_2);
    if (weightTensor.desc.format == Mki::TensorFormat::TENSOR_FORMAT_ND) {
        n = param_.transposeB ? weightTensor.desc.dims.at(1) : weightTensor.desc.dims.at(DIM_2);
    } else if (weightTensor.desc.format == Mki::TensorFormat::TENSOR_FORMAT_FRACTAL_NZ) {
        n = param_.transposeB ? weightTensor.desc.dims.at(DIM_2) :
                                weightTensor.desc.dims.at(1) * weightTensor.desc.dims.at(DIM_3);
    }
}

void LinearOpsRunner::InitKernelGraph(size_t inTensorNum, size_t outTensorNum, size_t internalTensorNum, size_t nodeNum)
{
    kernelGraph_.inTensors.resize(inTensorNum);
    kernelGraph_.outTensors.resize(outTensorNum);
    kernelGraph_.internalTensors.resize(internalTensorNum);
    kernelGraph_.nodes.resize(nodeNum);
}

void LinearOpsRunner::SetupNeedMergeAxis(const Mki::Tensor &xTensor, const Mki::Tensor &weightTensor)
{
    xNeedMergeAxis_ = false;
    weightNeedMergeAxis_ = false;
    if (xTensor.desc.dims.size() == SIZE_2) {
        weightNeedMergeAxis_ = weightTensor.desc.dims.size() == SIZE_3;
    } else if (xTensor.desc.dims.size() == SIZE_3) {
        if (weightTensor.desc.dims.size() == SIZE_2) {
            xNeedMergeAxis_ = true;
        } else {
            if (xTensor.desc.dims.at(0) != 1 && weightTensor.desc.dims.at(0) == 1) {
                xNeedMergeAxis_ = true;
                weightNeedMergeAxis_ = weightTensor.desc.dims.size() == SIZE_3;
            }
        }
    }
}

} // namespace atb