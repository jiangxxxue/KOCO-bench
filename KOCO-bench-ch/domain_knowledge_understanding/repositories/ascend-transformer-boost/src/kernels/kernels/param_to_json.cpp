/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>
#include <vector>
#include <sstream>
#include <nlohmann/json.hpp>
#include <mki/utils/SVector/SVector.h>
#include <mki/utils/stringify/stringify.h>
#include <mki/utils/any/any.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"


namespace AsdOps {
using namespace Mki;

template <typename T> std::vector<T> SVectorToVector(const SVector<T> &svector)
{
    std::vector<T> tmpVec;
    tmpVec.resize(svector.size());
    for (size_t i = 0; i < svector.size(); i++) {
        tmpVec.at(i) = svector.at(i);
    }
    return tmpVec;
};

std::string ActivationToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Activation specificParam = AnyCast<OpParam::Activation>(asdOpsParam);

    paramsJson["activationType"] = specificParam.activationType;
    paramsJson["scale"] = specificParam.scale;
    paramsJson["dim"] = specificParam.dim;
    paramsJson["approx"] = specificParam.approx;

    return paramsJson.dump();
}

std::string AsStridedToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::AsStrided specificParam = AnyCast<OpParam::AsStrided>(asdOpsParam);

    paramsJson["size"] = SVectorToVector(specificParam.size);
    paramsJson["stride"] = SVectorToVector(specificParam.stride);
    paramsJson["offset"] = SVectorToVector(specificParam.offset);

    return paramsJson.dump();
}

std::string ConcatToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Concat specificParam = AnyCast<OpParam::Concat>(asdOpsParam);

    paramsJson["concatDim"] = specificParam.concatDim;

    return paramsJson.dump();
}

std::string CopyToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Copy specificParam = AnyCast<OpParam::Copy>(asdOpsParam);

    paramsJson["dstSize"] = SVectorToVector(specificParam.dstSize);
    paramsJson["dstStride"] = SVectorToVector(specificParam.dstStride);
    paramsJson["dstOffset"] = SVectorToVector(specificParam.dstOffset);

    return paramsJson.dump();
}

std::string CumsumToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Cumsum specificParam = AnyCast<OpParam::Cumsum>(asdOpsParam);

    paramsJson["axis"] = SVectorToVector(specificParam.axis);
    paramsJson["exclusive"] = specificParam.exclusive;
    paramsJson["reverse"] = specificParam.reverse;

    return paramsJson.dump();
}

std::string ElewiseToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Elewise specificParam = AnyCast<OpParam::Elewise>(asdOpsParam);

    paramsJson["elewiseType"] = specificParam.elewiseType;
    paramsJson["varAttr"] = specificParam.varAttr;
    paramsJson["inputScale"] = specificParam.inputScale;
    paramsJson["inputOffset"] = specificParam.inputOffset;
    paramsJson["outTensorType"] = specificParam.outTensorType;

    return paramsJson.dump();
}

std::string ExpandToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Expand specificParam = AnyCast<OpParam::Expand>(asdOpsParam);

    paramsJson["shape"] = SVectorToVector(specificParam.shape);

    return paramsJson.dump();
}

std::string FillToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Fill specificParam = AnyCast<OpParam::Fill>(asdOpsParam);

    paramsJson["withMask"] = specificParam.withMask;
    paramsJson["value"] = SVectorToVector(specificParam.value);
    paramsJson["outDim"] = SVectorToVector(specificParam.outDim);

    return paramsJson.dump();
}

std::string GatherToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Gather specificParam = AnyCast<OpParam::Gather>(asdOpsParam);

    paramsJson["batchDims"] = specificParam.batchDims;
    paramsJson["axis"] = SVectorToVector(specificParam.axis);

    return paramsJson.dump();
}

std::string GroupTopkToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::GroupTopk specificParam = AnyCast<OpParam::GroupTopk>(asdOpsParam);
 
    paramsJson["groupNum"] = specificParam.groupNum;
    paramsJson["k"] = specificParam.k;
    paramsJson["kInner"] = specificParam.kInner;
 
    return paramsJson.dump();
}

std::string IndexToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Index specificParam = AnyCast<OpParam::Index>(asdOpsParam);

    paramsJson["indexType"] = specificParam.indexType;
    paramsJson["axis"] = specificParam.axis;

    return paramsJson.dump();
}

std::string MatMulToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::MatMul specificParam = AnyCast<OpParam::MatMul>(asdOpsParam);

    paramsJson["transposeA"] = specificParam.transposeA;
    paramsJson["transposeB"] = specificParam.transposeB;
    paramsJson["oriShape"] = SVectorToVector(specificParam.oriShape);
    paramsJson["withBias"] = specificParam.withBias;
    paramsJson["enDequant"] = specificParam.enDequant;
    paramsJson["tilingN"] = specificParam.tilingN;
    paramsJson["tilingK"] = specificParam.tilingK;

    return paramsJson.dump();
}

std::string MultinomialToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Multinomial specificParam = AnyCast<OpParam::Multinomial>(asdOpsParam);

    paramsJson["numSamples"] = specificParam.numSamples;
    paramsJson["randSeed"] = specificParam.randSeed;

    return paramsJson.dump();
}

std::string DynamicNTKToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::DynamicNTK specificParam = AnyCast<OpParam::DynamicNTK>(asdOpsParam);

    paramsJson["outType"] = specificParam.outType;

    return paramsJson.dump();
}

std::string NormToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Norm specificParam = AnyCast<OpParam::Norm>(asdOpsParam);

    paramsJson["normType"] = specificParam.normType;
    paramsJson["beginNormAxis"] = specificParam.beginNormAxis;
    paramsJson["beginParamsAxis"] = specificParam.beginParamsAxis;
    paramsJson["opsMode"] = specificParam.opsMode;
    paramsJson["epsilon"] = specificParam.epsilon;
    paramsJson["zoomScaleValue"] = specificParam.zoomScaleValue;
    paramsJson["inGamma"] = specificParam.inGamma;
    paramsJson["inBeta"] = specificParam.inBeta;
    paramsJson["inRes"] = specificParam.inRes;
    paramsJson["inNormBias"] = specificParam.inNormBias;
    paramsJson["outMean"] = specificParam.outMean;
    paramsJson["outVarience"] = specificParam.outVarience;
    paramsJson["outResQuant"] = specificParam.outResQuant;
    paramsJson["outRes"] = specificParam.outRes;
    paramsJson["precisionMode"] = specificParam.precisionMode;
    paramsJson["gemmaMode"] = specificParam.gemmaMode;
    paramsJson["isDynamicQuant"] = specificParam.isDynamicQuant;
    paramsJson["isSymmetric"] = specificParam.isSymmetric;

    return paramsJson.dump();
}

std::string OnehotToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Onehot specificParam = AnyCast<OpParam::Onehot>(asdOpsParam);

    paramsJson["axis"] = specificParam.axis;
    paramsJson["depth"] = SVectorToVector(specificParam.depth);

    return paramsJson.dump();
}

std::string ReduceToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Reduce specificParam = AnyCast<OpParam::Reduce>(asdOpsParam);

    paramsJson["reduceType"] = specificParam.reduceType;
    paramsJson["axis"] = SVectorToVector(specificParam.axis);

    return paramsJson.dump();
}

std::string ReverseToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Reverse specificParam = AnyCast<OpParam::Reverse>(asdOpsParam);

    paramsJson["axis"] = SVectorToVector(specificParam.axis);

    return paramsJson.dump();
}

std::string SliceToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Slice specificParam = AnyCast<OpParam::Slice>(asdOpsParam);

    paramsJson["offsets"] = SVectorToVector(specificParam.offsets);
    paramsJson["size"] = SVectorToVector(specificParam.size);

    return paramsJson.dump();
}

std::string SoftmaxToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Softmax specificParam = AnyCast<OpParam::Softmax>(asdOpsParam);

    paramsJson["axes"] = SVectorToVector(specificParam.axes);

    return paramsJson.dump();
}

std::string SortToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Sort specificParam = AnyCast<OpParam::Sort>(asdOpsParam);

    paramsJson["num"] = SVectorToVector(specificParam.num);

    return paramsJson.dump();
}

std::string SplitToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Split specificParam = AnyCast<OpParam::Split>(asdOpsParam);

    paramsJson["splitDim"] = specificParam.splitDim;
    paramsJson["splitNum"] = specificParam.splitNum;

    return paramsJson.dump();
}

std::string TransdataToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Transdata specificParam = AnyCast<OpParam::Transdata>(asdOpsParam);

    paramsJson["transdataType"] = specificParam.transdataType;
    paramsJson["outCrops"] = SVectorToVector(specificParam.outCrops);

    return paramsJson.dump();
}

std::string TransposeToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::Transpose specificParam = AnyCast<OpParam::Transpose>(asdOpsParam);

    paramsJson["perm"] = SVectorToVector(specificParam.perm);

    return paramsJson.dump();
}

std::string ZerosLikeToJson(const Any &asdOpsParam)
{
    (void)asdOpsParam;
    return "{}";
}

std::string LogprobsSampleParamToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::LogprobsSample specificParam = AnyCast<OpParam::LogprobsSample>(asdOpsParam);

    paramsJson["logprobsSize"] = specificParam.logprobsSize;
    return paramsJson.dump();
}

std::string MoeGateCorrToJson(const Any &asdOpsParam)
{
    nlohmann::json paramsJson;
    OpParam::MoeGateCorr specificParam = AnyCast<OpParam::MoeGateCorr>(asdOpsParam);

    paramsJson["transposeA"] = specificParam.transposeA;
    paramsJson["transposeB"] = specificParam.transposeB;

    return paramsJson.dump();
}

REG_STRINGIFY(OpParam::Activation, ActivationToJson);
REG_STRINGIFY(OpParam::AsStrided, AsStridedToJson);
REG_STRINGIFY(OpParam::Concat, ConcatToJson);
REG_STRINGIFY(OpParam::Copy, CopyToJson);
REG_STRINGIFY(OpParam::Cumsum, CumsumToJson);
REG_STRINGIFY(OpParam::Elewise, ElewiseToJson);
REG_STRINGIFY(OpParam::Expand, ExpandToJson);
REG_STRINGIFY(OpParam::Fill, FillToJson);
REG_STRINGIFY(OpParam::Gather, GatherToJson);
REG_STRINGIFY(OpParam::GroupTopk, GroupTopkToJson);
REG_STRINGIFY(OpParam::Index, IndexToJson);
REG_STRINGIFY(OpParam::MatMul, MatMulToJson);
REG_STRINGIFY(OpParam::Multinomial, MultinomialToJson);
REG_STRINGIFY(OpParam::DynamicNTK, DynamicNTKToJson);
REG_STRINGIFY(OpParam::Norm, NormToJson);
REG_STRINGIFY(OpParam::Onehot, OnehotToJson);
REG_STRINGIFY(OpParam::Reduce, ReduceToJson);
REG_STRINGIFY(OpParam::Reverse, ReverseToJson);
REG_STRINGIFY(OpParam::Slice, SliceToJson);
REG_STRINGIFY(OpParam::Softmax, SoftmaxToJson);
REG_STRINGIFY(OpParam::Sort, SortToJson);
REG_STRINGIFY(OpParam::Split, SplitToJson);
REG_STRINGIFY(OpParam::Transdata, TransdataToJson);
REG_STRINGIFY(OpParam::Transpose, TransposeToJson);
REG_STRINGIFY(OpParam::ZerosLike, ZerosLikeToJson);
REG_STRINGIFY(OpParam::LogprobsSample, LogprobsSampleParamToJson);
REG_STRINGIFY(OpParam::MoeGateCorr, MoeGateCorrToJson);
} // namespace AsdOps
