/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "atb/utils/param_to_json.h"
#include "atb/infer_op_params.h"
#include "atb/train_op_params.h"
#include "atb/common_op_params.h"
#include "atb/utils/common_utils.h"

namespace atb {
template <> nlohmann::json OpParamToJson(const infer::ActivationParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["activationType"] = opParam.activationType;
    paramsJson["scale"] = opParam.scale;
    paramsJson["dim"] = opParam.dim;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::AllGatherParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["rank"] = opParam.rank;
    paramsJson["rankSize"] = opParam.rankSize;
    paramsJson["rankRoot"] = opParam.rankRoot;
    paramsJson["backend"] = opParam.backend;
    paramsJson["hcclComm"] = reinterpret_cast<uint64_t>(opParam.hcclComm);
    paramsJson["commMode"] = opParam.commMode;
    paramsJson["rankTableFile"] = opParam.rankTableFile;
    paramsJson["commDomain"] = opParam.commDomain;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::AllGatherVParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["rank"] = opParam.rank;
    paramsJson["rankSize"] = opParam.rankSize;
    paramsJson["rankRoot"] = opParam.rankRoot;
    paramsJson["backend"] = opParam.backend;
    paramsJson["hcclComm"] = reinterpret_cast<uint64_t>(opParam.hcclComm);
    paramsJson["commMode"] = opParam.commMode;
    paramsJson["rankTableFile"] = opParam.rankTableFile;
    paramsJson["commDomain"] = opParam.commDomain;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::AllReduceParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["rank"] = opParam.rank;
    paramsJson["rankSize"] = opParam.rankSize;
    paramsJson["rankRoot"] = opParam.rankRoot;
    paramsJson["backend"] = opParam.backend;
    paramsJson["allReduceType"] = opParam.allReduceType;
    paramsJson["hcclComm"] = reinterpret_cast<uint64_t>(opParam.hcclComm);
    paramsJson["commMode"] = opParam.commMode;
    paramsJson["rankTableFile"] = opParam.rankTableFile;
    paramsJson["commDomain"] = opParam.commDomain;
    paramsJson["quantType"] = opParam.quantType;
    paramsJson["outDataType"] = opParam.outDataType;
    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::AsStridedParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["size"] = SVectorToVector(opParam.size);
    paramsJson["stride"] = SVectorToVector(opParam.stride);
    paramsJson["offset"] = SVectorToVector(opParam.offset);

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::BroadcastParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["rank"] = opParam.rank;
    paramsJson["rankSize"] = opParam.rankSize;
    paramsJson["rankRoot"] = opParam.rankRoot;
    paramsJson["hcclComm"] = reinterpret_cast<uint64_t>(opParam.hcclComm);
    paramsJson["commMode"] = opParam.commMode;
    paramsJson["backend"] = opParam.backend;
    paramsJson["rankTableFile"] = opParam.rankTableFile;
    paramsJson["commDomain"] = opParam.commDomain;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::ReduceScatterParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["rank"] = opParam.rank;
    paramsJson["rankSize"] = opParam.rankSize;
    paramsJson["rankRoot"] = opParam.rankRoot;
    paramsJson["hcclComm"] = reinterpret_cast<uint64_t>(opParam.hcclComm);
    paramsJson["reduceType"] = opParam.reduceType;
    paramsJson["commMode"] = opParam.commMode;
    paramsJson["backend"] = opParam.backend;
    paramsJson["rankTableFile"] = opParam.rankTableFile;
    paramsJson["commDomain"] = opParam.commDomain;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::ReduceScatterVParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["rank"] = opParam.rank;
    paramsJson["rankSize"] = opParam.rankSize;
    paramsJson["rankRoot"] = opParam.rankRoot;
    paramsJson["sendCounts"] = opParam.sendCounts;
    paramsJson["sdispls"] = opParam.sdispls;
    paramsJson["recvCount"] = opParam.recvCount;
    paramsJson["hcclComm"] = reinterpret_cast<uint64_t>(opParam.hcclComm);
    paramsJson["reduceType"] = opParam.reduceType;
    paramsJson["commMode"] = opParam.commMode;
    paramsJson["backend"] = opParam.backend;
    paramsJson["rankTableFile"] = opParam.rankTableFile;
    paramsJson["commDomain"] = opParam.commDomain;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::ConcatParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["concatDim"] = opParam.concatDim;
    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::CumsumParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["axes"] = SVectorToVector(opParam.axes);
    paramsJson["exclusive"] = opParam.exclusive;
    paramsJson["reverse"] = opParam.reverse;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::DynamicNTKParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["outputType"] = opParam.outDataType;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::ElewiseParam &opParam)
{
    nlohmann::json quantParamsJson;
    quantParamsJson["inputOffset"] = opParam.quantParam.inputOffset;
    quantParamsJson["inputScale"] = opParam.quantParam.inputScale;
    quantParamsJson["asymmetric"] = opParam.quantParam.asymmetric;

    nlohmann::json mulsParamJson;
    mulsParamJson["varAttr"] = opParam.mulsParam.varAttr;

    nlohmann::json paramsJson;
    paramsJson["elewiseType"] = opParam.elewiseType;
    paramsJson["quantParam"] = quantParamsJson;
    paramsJson["mulsParam"] = mulsParamJson;
    paramsJson["outTensorType"] = opParam.outTensorType;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const train::FastSoftMaxParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["headNum"] = opParam.headNum;
    paramsJson["qSeqLen"] = opParam.qSeqLen;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const train::FastSoftMaxGradParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["headNum"] = opParam.headNum;
    paramsJson["qSeqLen"] = opParam.qSeqLen;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::FillParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["outDim"] = SVectorToVector(opParam.outDim);
    paramsJson["value"] = SVectorToVector(opParam.value);
    paramsJson["withMask"] = opParam.withMask;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::GatherParam &opParam)
{
    nlohmann::json gatherParamsJson;
    gatherParamsJson["axis"] = opParam.axis;
    gatherParamsJson["batchDims"] = opParam.batchDims;

    return gatherParamsJson;
}

template <> nlohmann::json OpParamToJson(const train::GenAttentionMaskParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["seqLen"] = SVectorToVector(opParam.seqLen);
    paramsJson["headNum"] = opParam.headNum;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::GroupedMatmulInplaceAddParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["transposeA"] = opParam.transposeA;
    paramsJson["transposeB"] = opParam.transposeB;
    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::LayerNormParam &opParam)
{
    nlohmann::json normParam;
    normParam["quantType"] = opParam.normParam.quantType;
    normParam["epsilon"] = opParam.normParam.epsilon;
    normParam["beginNormAxis"] = opParam.normParam.beginNormAxis;
    normParam["beginParamsAxis"] = opParam.normParam.beginParamsAxis;

    nlohmann::json preNormParam;
    preNormParam["quantType"] = opParam.preNormParam.quantType;
    preNormParam["epsilon"] = opParam.preNormParam.epsilon;
    preNormParam["opMode"] = opParam.preNormParam.opMode;
    preNormParam["zoomScaleValue"] = opParam.preNormParam.zoomScaleValue;

    nlohmann::json postNormParam;
    postNormParam["quantType"] = opParam.postNormParam.quantType;
    postNormParam["epsilon"] = opParam.postNormParam.epsilon;
    postNormParam["opMode"] = opParam.postNormParam.opMode;
    postNormParam["zoomScaleValue"] = opParam.postNormParam.zoomScaleValue;

    nlohmann::json paramJson;
    paramJson["layerType"] = opParam.layerType;
    paramJson["normParam"] = normParam;
    paramJson["preNormParam"] = preNormParam;
    paramJson["postNormParam"] = postNormParam;

    return paramJson;
}

template <> nlohmann::json OpParamToJson(const infer::LayerNormWithStrideParam &opParam)
{
    nlohmann::json normParam;
    normParam["quantType"] = opParam.normParam.quantType;
    normParam["epsilon"] = opParam.normParam.epsilon;
    normParam["beginNormAxis"] = opParam.normParam.beginNormAxis;
    normParam["beginParamsAxis"] = opParam.normParam.beginParamsAxis;

    nlohmann::json preNormParam;
    preNormParam["quantType"] = opParam.preNormParam.quantType;
    preNormParam["epsilon"] = opParam.preNormParam.epsilon;
    preNormParam["opMode"] = opParam.preNormParam.opMode;
    preNormParam["zoomScaleValue"] = opParam.preNormParam.zoomScaleValue;

    nlohmann::json postNormParam;
    postNormParam["quantType"] = opParam.postNormParam.quantType;
    postNormParam["epsilon"] = opParam.postNormParam.epsilon;
    postNormParam["opMode"] = opParam.postNormParam.opMode;
    postNormParam["zoomScaleValue"] = opParam.postNormParam.zoomScaleValue;

    nlohmann::json layerNormWithStrideParamJson;
    layerNormWithStrideParamJson["layerType"] = opParam.layerType;
    layerNormWithStrideParamJson["normParam"] = normParam;
    layerNormWithStrideParamJson["preNormParam"] = preNormParam;
    layerNormWithStrideParamJson["postNormParam"] = postNormParam;

    return layerNormWithStrideParamJson;
}

template <> nlohmann::json OpParamToJson(const infer::LinearParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["transposeA"] = opParam.transposeA;
    paramsJson["transposeB"] = opParam.transposeB;
    paramsJson["hasBias"] = opParam.hasBias;
    paramsJson["outDataType"] = opParam.outDataType;
    paramsJson["enAccum"] = opParam.enAccum;
    paramsJson["matmulType"] = opParam.matmulType;
    paramsJson["quantMode"] = opParam.quantMode;
    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::LinearParallelParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["transWeight"] = opParam.transWeight;
    paramsJson["rank"] = opParam.rank;
    paramsJson["rankSize"] = opParam.rankSize;
    paramsJson["rankRoot"] = opParam.rankRoot;
    paramsJson["hasResidual"] = opParam.hasResidual;
    paramsJson["backend"] = opParam.backend;
    paramsJson["commMode"] = opParam.commMode;
    paramsJson["rankTableFile"] = opParam.rankTableFile;
    paramsJson["type"] = opParam.type;
    paramsJson["keepIntermediate"] = opParam.keepIntermediate;
    paramsJson["quantType"] = opParam.quantType;
    paramsJson["quantGroupSize"] = opParam.quantGroupSize;
    paramsJson["outDataType"] = opParam.outDataType;
    paramsJson["commDomain"] = opParam.commDomain;
    paramsJson["local_expert_nums"] = opParam.moeInfo.localExpertNums;
    paramsJson["epSize"] = opParam.moeInfo.epSize;
    paramsJson["tpSize"] = opParam.moeInfo.tpSize;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::LinearSparseParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["transposeA"] = opParam.transposeA;
    paramsJson["transposeB"] = opParam.transposeB;
    paramsJson["tilingK"] = opParam.tilingK;
    paramsJson["tilingN"] = opParam.tilingN;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::SwigluQuantParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["quantType"] = opParam.quantType;
    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::MultinomialParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["numSamples"] = opParam.numSamples;
    paramsJson["randSeed"] = opParam.randSeed;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::OnehotParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["axis"] = opParam.axis;
    paramsJson["depth"] = opParam.depth;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::IndexAddParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["indexType"] = opParam.indexType;
    paramsJson["axis"] = opParam.axis;
    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::PagedAttentionParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["headNum"] = opParam.headNum;
    paramsJson["qkScale"] = opParam.qkScale;
    paramsJson["kvHeadNum"] = opParam.kvHeadNum;
    paramsJson["maskType"] = opParam.maskType;
    paramsJson["batchRunStatusEnable"] = opParam.batchRunStatusEnable;
    paramsJson["quantType"] = opParam.quantType;
    paramsJson["hasQuantOffset"] = opParam.hasQuantOffset;
    paramsJson["compressType"] = opParam.compressType;
    paramsJson["calcType"] = opParam.calcType;

    paramsJson["mlaVHeadSize"] = opParam.mlaVHeadSize;
    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::ReduceParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["axis"] = SVectorToVector(opParam.axis);
    paramsJson["reduceType"] = opParam.reduceType;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::RelayAttentionParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["headNum"] = opParam.headNum;
    paramsJson["qkScale"] = opParam.qkScale;
    paramsJson["kvHeadNum"] = opParam.kvHeadNum;
    paramsJson["maskType"] = opParam.maskType;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::RepeatParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["multiples"] = SVectorToVector(opParam.multiples);

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::RmsNormParam &opParam)
{
    nlohmann::json normParamJson;
    normParamJson["quantType"] = opParam.normParam.quantType;
    normParamJson["epsilon"] = opParam.normParam.epsilon;
    normParamJson["layerNormEps"] = opParam.normParam.layerNormEps;
    normParamJson["rstd"] = opParam.normParam.rstd;
    normParamJson["precisionMode"] = opParam.normParam.precisionMode;
    normParamJson["modelType"] = opParam.normParam.modelType;

    nlohmann::json preNormParamJson;
    preNormParamJson["quantType"] = opParam.preNormParam.quantType;
    preNormParamJson["epsilon"] = opParam.preNormParam.epsilon;
    preNormParamJson["hasBias"] = opParam.preNormParam.hasBias;

    nlohmann::json postNormParamJson;
    postNormParamJson["quantType"] = opParam.postNormParam.quantType;
    postNormParamJson["epsilon"] = opParam.postNormParam.epsilon;
    postNormParamJson["hasBias"] = opParam.postNormParam.hasBias;

    nlohmann::json rmsNormParamsJson;
    rmsNormParamsJson["layerType"] = opParam.layerType;
    rmsNormParamsJson["normParam"] = normParamJson;
    rmsNormParamsJson["preNormParam"] = preNormParamJson;
    rmsNormParamsJson["postNormParam"] = postNormParamJson;
    return rmsNormParamsJson;
}

template <> nlohmann::json OpParamToJson(const infer::RmsNormWithStrideParam &opParam)
{
    nlohmann::json normParamJson;
    normParamJson["quantType"] = opParam.normParam.quantType;
    normParamJson["epsilon"] = opParam.normParam.epsilon;
    normParamJson["layerNormEps"] = opParam.normParam.layerNormEps;
    normParamJson["rstd"] = opParam.normParam.rstd;
    normParamJson["precisionMode"] = opParam.normParam.precisionMode;
    normParamJson["modelType"] = opParam.normParam.modelType;

    nlohmann::json preNormParamJson;
    preNormParamJson["quantType"] = opParam.preNormParam.quantType;
    preNormParamJson["epsilon"] = opParam.preNormParam.epsilon;
    preNormParamJson["hasBias"] = opParam.preNormParam.hasBias;

    nlohmann::json postNormParamJson;
    postNormParamJson["quantType"] = opParam.postNormParam.quantType;
    postNormParamJson["epsilon"] = opParam.postNormParam.epsilon;
    postNormParamJson["hasBias"] = opParam.postNormParam.hasBias;

    nlohmann::json rmsNormWithStrideParamsJson;
    rmsNormWithStrideParamsJson["layerType"] = opParam.layerType;
    rmsNormWithStrideParamsJson["normParam"] = normParamJson;
    rmsNormWithStrideParamsJson["preNormParam"] = preNormParamJson;
    rmsNormWithStrideParamsJson["postNormParam"] = postNormParamJson;
    return rmsNormWithStrideParamsJson;
}

template <> nlohmann::json OpParamToJson(const infer::RopeParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["rotaryCoeff"] = opParam.rotaryCoeff;
    paramsJson["cosFormat"] = opParam.cosFormat;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const train::RopeGradParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["c_qSeqLen"] = opParam.qSeqLen;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::SelfAttentionParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["headNum"] = opParam.headNum;
    paramsJson["isTriuMask"] = opParam.isTriuMask;
    paramsJson["qScale"] = opParam.qScale;
    paramsJson["qkScale"] = opParam.qkScale;
    paramsJson["batchRunStatusEnable"] = opParam.batchRunStatusEnable;
    paramsJson["kvHeadNum"] = opParam.kvHeadNum;
    paramsJson["calcType"] = opParam.calcType;
    paramsJson["maskType"] = opParam.maskType;
    paramsJson["kernelType"] = opParam.kernelType;
    paramsJson["clampType"] = opParam.clampType;
    paramsJson["clampMin"] = opParam.clampMin;
    paramsJson["clampMax"] = opParam.clampMax;
    paramsJson["kvcacheCfg"] = opParam.kvcacheCfg;
    paramsJson["inputLayout"] = opParam.inputLayout;
    paramsJson["scaleType"] = opParam.scaleType;
    paramsJson["quantType"] = opParam.quantType;
    paramsJson["outDataType"] = opParam.outDataType;
    paramsJson["mlaVHeadSize"] = opParam.mlaVHeadSize;
    paramsJson["windowSize"] = opParam.windowSize;
    paramsJson["cacheType"] = opParam.cacheType;
    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::SetValueParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["starts"] = SVectorToVector(opParam.starts);
    paramsJson["ends"] = SVectorToVector(opParam.ends);
    paramsJson["strides"] = SVectorToVector(opParam.strides);

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::SliceParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["offsets"] = SVectorToVector(opParam.offsets);
    paramsJson["size"] = SVectorToVector(opParam.size);

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::SoftmaxParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["axes"] = SVectorToVector(opParam.axes);

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::SortParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["num"] = SVectorToVector(opParam.num);

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::SplitParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["splitNum"] = opParam.splitNum;
    paramsJson["splitDim"] = opParam.splitDim;
    paramsJson["splitSizes"] = SVectorToVector(opParam.splitSizes);

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const train::StridedBatchMatmulParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["transposeA"] = opParam.transposeA;
    paramsJson["transposeB"] = opParam.transposeB;
    paramsJson["batch"] = opParam.batch;
    paramsJson["headNum"] = opParam.headNum;
    paramsJson["m"] = opParam.m;
    paramsJson["n"] = opParam.n;
    paramsJson["k"] = opParam.k;
    paramsJson["lda"] = opParam.lda;
    paramsJson["ldb"] = opParam.ldb;
    paramsJson["ldc"] = opParam.ldc;
    paramsJson["strideA"] = opParam.strideA;
    paramsJson["strideB"] = opParam.strideB;
    paramsJson["strideC"] = opParam.strideC;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::TopkToppSamplingParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["topkToppSamplingType"] = opParam.topkToppSamplingType;
    paramsJson["randSeeds"] = opParam.randSeeds;
    paramsJson["randSeed"] = opParam.randSeed;
    paramsJson["topk"] = opParam.topk;
    paramsJson["logProbsSize"] = opParam.logProbsSize;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::TransdataParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["transdataType"] = opParam.transdataType;
    paramsJson["outCrops"] = SVectorToVector(opParam.outCrops);

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::TransposeParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["perm"] = SVectorToVector(opParam.perm);

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::ReshapeAndCacheParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["compressType"] = opParam.compressType;
    paramsJson["kvCacheCfg"] = opParam.kvCacheCfg;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::GatingParam &opParam)
{
    nlohmann::json gatingParamsJson;
    gatingParamsJson["topkExpertNum"] = opParam.topkExpertNum;
    gatingParamsJson["cumSumNum"] = opParam.cumSumNum;
    gatingParamsJson["deviceExpert"] = opParam.deviceExpert;

    return gatingParamsJson;
}

template <> nlohmann::json OpParamToJson(const infer::SendParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["rank"] = opParam.rank;
    paramsJson["rankSize"] = opParam.rankSize;
    paramsJson["rankRoot"] = opParam.rankRoot;
    paramsJson["destRank"] = opParam.destRank;
    paramsJson["backend"] = opParam.backend;
    paramsJson["hcclComm"] = reinterpret_cast<uint64_t>(opParam.hcclComm);
    paramsJson["commMode"] = opParam.commMode;
    paramsJson["rankTableFile"] = opParam.rankTableFile;
    paramsJson["commDomain"] = opParam.commDomain;
    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::RecvParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["rank"] = opParam.rank;
    paramsJson["rankSize"] = opParam.rankSize;
    paramsJson["rankRoot"] = opParam.rankRoot;
    paramsJson["srcRank"] = opParam.srcRank;
    paramsJson["backend"] = opParam.backend;
    paramsJson["hcclComm"] = reinterpret_cast<uint64_t>(opParam.hcclComm);
    paramsJson["commMode"] = opParam.commMode;
    paramsJson["rankTableFile"] = opParam.rankTableFile;
    paramsJson["commDomain"] = opParam.commDomain;
    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::AllToAllParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["rank"] = opParam.rank;
    paramsJson["rankSize"] = opParam.rankSize;
    paramsJson["rankRoot"] = opParam.rankRoot;
    paramsJson["backend"] = opParam.backend;
    paramsJson["hcclComm"] = reinterpret_cast<uint64_t>(opParam.hcclComm);
    paramsJson["commMode"] = opParam.commMode;
    paramsJson["rankTableFile"] = opParam.rankTableFile;
    paramsJson["commDomain"] = opParam.commDomain;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::AllToAllVParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["rank"] = opParam.rank;
    paramsJson["rankSize"] = opParam.rankSize;
    paramsJson["rankRoot"] = opParam.rankRoot;
    paramsJson["sendCounts"] = opParam.sendCounts;
    paramsJson["sdispls"] = opParam.sdispls;
    paramsJson["recvCounts"] = opParam.recvCounts;
    paramsJson["rdispls"] = opParam.rdispls;
    paramsJson["backend"] = opParam.backend;
    paramsJson["hcclComm"] = reinterpret_cast<uint64_t>(opParam.hcclComm);
    paramsJson["commMode"] = opParam.commMode;
    paramsJson["rankTableFile"] = opParam.rankTableFile;
    paramsJson["commDomain"] = opParam.commDomain;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::AllToAllVV2Param &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["rank"] = opParam.rank;
    paramsJson["rankSize"] = opParam.rankSize;
    paramsJson["rankRoot"] = opParam.rankRoot;
    paramsJson["backend"] = opParam.backend;
    paramsJson["hcclComm"] = reinterpret_cast<uint64_t>(opParam.hcclComm);
    paramsJson["commMode"] = opParam.commMode;
    paramsJson["rankTableFile"] = opParam.rankTableFile;
    paramsJson["commDomain"] = opParam.commDomain;

    return paramsJson;
}


template <> nlohmann::json OpParamToJson(const train::LaserAttentionParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["headNum"] = opParam.headNum;
    paramsJson["inputLayout"] = opParam.inputLayout;
    paramsJson["scaleValue"] = opParam.scaleValue;
    paramsJson["keepProb"] = opParam.keepProb;
    paramsJson["preTokens"] = opParam.preTokens;
    paramsJson["nextTokens"] = opParam.nextTokens;
    paramsJson["sparseMode"] = opParam.sparseMode;
    paramsJson["innerPrecise"] = opParam.innerPrecise;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const train::LaserAttentionGradParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["headNum"] = opParam.headNum;
    paramsJson["inputLayout"] = opParam.inputLayout;
    paramsJson["scaleValue"] = opParam.scaleValue;
    paramsJson["keepProb"] = opParam.keepProb;
    paramsJson["preTokens"] = opParam.preTokens;
    paramsJson["nextTokens"] = opParam.nextTokens;
    paramsJson["sparseMode"] = opParam.sparseMode;
    paramsJson["innerPrecise"] = opParam.innerPrecise;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::GroupTopkParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["groupNum"] = opParam.groupNum;
    paramsJson["k"] = opParam.k;
    paramsJson["groupMultiFlag"] = opParam.groupMultiFlag;
    paramsJson["n"] = opParam.n;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::GroupedMatmulWithRoutingParam &opParam)
{
    nlohmann::json moeGroupMatmulParamsJson;
    moeGroupMatmulParamsJson["transposeB"] = opParam.transposeB;
    moeGroupMatmulParamsJson["topK"] = opParam.topK;
    moeGroupMatmulParamsJson["moeType"] = opParam.groupedMatmulType;
    moeGroupMatmulParamsJson["outDataType"] = opParam.outDataType;
    return moeGroupMatmulParamsJson;
}

template <> nlohmann::json OpParamToJson(const infer::CohereLayerNormParam &opParam)
{
    nlohmann::json cohereLayerNormParamsJson;
    cohereLayerNormParamsJson["epsilon"] = opParam.epsilon;
    return cohereLayerNormParamsJson;
}

template <> nlohmann::json OpParamToJson(const infer::GatherPreRmsNormParam &opParam)
{
    nlohmann::json gatherPreRmsNormParamJson;
    gatherPreRmsNormParamJson["epsilon"] = opParam.epsilon;

    return gatherPreRmsNormParamJson;
}

template <> nlohmann::json OpParamToJson(const infer::MlaPreprocessParam &opParam)
{
    nlohmann::json mlaPreprocessParamJson;
    mlaPreprocessParamJson["wdqDim"] = opParam.wdqDim;
    mlaPreprocessParamJson["qRopeDim"] = opParam.qRopeDim;
    mlaPreprocessParamJson["kRopeDim"] = opParam.kRopeDim;
    mlaPreprocessParamJson["epsilon"] = opParam.epsilon;
    mlaPreprocessParamJson["qRotaryCoeff"] = opParam.qRotaryCoeff;
    mlaPreprocessParamJson["kRotaryCoeff"] = opParam.kRotaryCoeff;
    mlaPreprocessParamJson["transposeWdq"] = opParam.transposeWdq;
    mlaPreprocessParamJson["transposeWuq"] = opParam.transposeWuq;
    mlaPreprocessParamJson["transposeWuk"] = opParam.transposeWuk;
    mlaPreprocessParamJson["cacheMode"] = opParam.cacheMode;
    mlaPreprocessParamJson["quantMode"] = opParam.quantMode;
    return mlaPreprocessParamJson;
}

template <> nlohmann::json OpParamToJson(const infer::MultiLatentAttentionParam &opParam)
{
    nlohmann::json multiLatentAttentionParamJson;
    multiLatentAttentionParamJson["headNum"] = opParam.headNum;
    multiLatentAttentionParamJson["qkScale"] = opParam.qkScale;
    multiLatentAttentionParamJson["kvHeadNum"] = opParam.kvHeadNum;
    multiLatentAttentionParamJson["maskType"] = opParam.maskType;
    multiLatentAttentionParamJson["calcType"] = opParam.calcType;
    multiLatentAttentionParamJson["cacheMode"] = opParam.cacheMode;
    multiLatentAttentionParamJson["windowSize"] = opParam.windowSize;
    return multiLatentAttentionParamJson;
}

template <> nlohmann::json OpParamToJson(const common::EventParam &opParam)
{
    nlohmann::json eventParamsJson;
    eventParamsJson["operatorType"] = opParam.operatorType;
    eventParamsJson["event"] = reinterpret_cast<uint64_t>(opParam.event);
    return eventParamsJson;
}

template <> nlohmann::json OpParamToJson(const infer::NormRopeReshapeParam &opParam)
{
    nlohmann::json normRopeReshapeParamParamsJson;
    normRopeReshapeParamParamsJson["precisionMode"] = opParam.precisionMode;
    normRopeReshapeParamParamsJson["epsilon"] = opParam.epsilon;
    normRopeReshapeParamParamsJson["rotaryCoeff"] = opParam.rotaryCoeff;
    return normRopeReshapeParamParamsJson;
}

template <> nlohmann::json OpParamToJson(const infer::RopeQConcatParam &opParam)
{
    (void)opParam;
    nlohmann::json ropeQConcataramsJson;
    return ropeQConcataramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::FusedAddTopkDivParam &opParam)
{
    nlohmann::json fusedAddTopkDivParamsJson;
    fusedAddTopkDivParamsJson["groupNum"] = opParam.groupNum;
    fusedAddTopkDivParamsJson["groupTopk"] = opParam.groupTopk;
    fusedAddTopkDivParamsJson["n"] = opParam.n;
    fusedAddTopkDivParamsJson["k"] = opParam.k;
    fusedAddTopkDivParamsJson["activationType"] = opParam.activationType;
    fusedAddTopkDivParamsJson["isNorm"] = opParam.isNorm;
    fusedAddTopkDivParamsJson["scale"] = opParam.scale;
    fusedAddTopkDivParamsJson["enableExpertMapping"] = opParam.enableExpertMapping;
    return fusedAddTopkDivParamsJson;
}

template <> nlohmann::json OpParamToJson(const infer::ReshapeAndCacheWithStrideParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["compressType"] = opParam.compressType;
    paramsJson["kvCacheCfg"] = opParam.kvCacheCfg;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::RazorFusionAttentionParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["headNum"] = opParam.headNum;
    paramsJson["kvHeadNum"] = opParam.kvHeadNum;
    paramsJson["qkScale"] = opParam.qkScale;
    paramsJson["razorLen"] = opParam.razorLen;
    paramsJson["preTokens"] = opParam.preTokens;
    paramsJson["nextTokens"] = opParam.nextTokens;
    paramsJson["tileQ"] = opParam.tileQ;
    paramsJson["tileKv"] = opParam.tileKv;
    paramsJson["textQLen"] = opParam.textQLen;
    paramsJson["textKvLen"] = opParam.textKvLen;

    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::FaUpdateParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["faUpdateType"] = opParam.faUpdateType;
    paramsJson["sp"] = opParam.sp;
    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::ScatterElementsV2Param &opParam)
{
    nlohmann::json scatterElementsV2ParamsJson;
    scatterElementsV2ParamsJson["axis"] = opParam.axis;
    scatterElementsV2ParamsJson["reduction"] = opParam.reduction;

    return scatterElementsV2ParamsJson;
}

template <> nlohmann::json OpParamToJson(const infer::GmmDeqSwigluQuantGmmDeqParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["outputType"] = opParam.outputType;
    paramsJson["groupListType"] = opParam.groupListType;
    paramsJson["weightUpPermuteType"] = opParam.weightUpPermuteType;
    paramsJson["transposeWeightUp"] = opParam.transposeWeightUp;
    paramsJson["transposeWeightDown"] = opParam.transposeWeightDown;
    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::MmDeqSwigluQuantMmDeqParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["outputType"] = opParam.outputType;
    paramsJson["weightUpPermuteType"] = opParam.weightUpPermuteType;
    paramsJson["transposeWeightUp"] = opParam.transposeWeightUp;
    paramsJson["transposeWeightDown"] = opParam.transposeWeightDown;
    return paramsJson;
}

template <> nlohmann::json OpParamToJson(const infer::RingMLAParam &opParam)
{
    nlohmann::json paramsJson;
    paramsJson["calcType"] = opParam.calcType;
    paramsJson["headNum"] = opParam.headNum;
    paramsJson["kvHeadNum"] = opParam.kvHeadNum;
    paramsJson["qkScale"] = opParam.qkScale;
    paramsJson["kernelType"] = opParam.kernelType;
    paramsJson["maskType"] = opParam.maskType;
    paramsJson["inputLayout"] = opParam.inputLayout;
    return paramsJson;
}

} // namespace atb
