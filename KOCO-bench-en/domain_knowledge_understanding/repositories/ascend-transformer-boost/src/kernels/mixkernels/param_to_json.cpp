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
#include "atbops/params/params.h"
#include "mki/utils/SVector/SVector.h"
#include "mki/utils/stringify/stringify.h"
#include "mki/utils/any/any.h"
#include "mki/utils/log/log.h"

using namespace Mki;

namespace AtbOps {
template <typename T> std::vector<T> SVectorToVector(const SVector<T> &svector)
{
    std::vector<T> tmpVec;
    tmpVec.resize(svector.size());
    for (size_t i = 0; i < svector.size(); i++) {
        tmpVec.at(i) = svector.at(i);
    }
    return tmpVec;
};

std::string FastSoftMaxGradToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::FastSoftMaxGrad specificParam = AnyCast<OpParam::FastSoftMaxGrad>(param);

    paramsJson["qSeqLen"] = specificParam.qSeqLen;
    paramsJson["headNum"] = specificParam.headNum;

    return paramsJson.dump();
}

std::string FastSoftMaxToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::FastSoftMax specificParam = AnyCast<OpParam::FastSoftMax>(param);

    paramsJson["qSeqLen"] = specificParam.qSeqLen;
    paramsJson["headNum"] = specificParam.headNum;

    return paramsJson.dump();
}

std::string FusionToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::Fusion specificParam = AnyCast<OpParam::Fusion>(param);

    paramsJson["fusion"] = specificParam.fusionType;

    return paramsJson.dump();
}

std::string GatingToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::Gating specificParam = AnyCast<OpParam::Gating>(param);

    paramsJson["headSize"] = specificParam.headSize;
    paramsJson["headNum"] = specificParam.headNum;

    return paramsJson.dump();
}

std::string GenAttentionMaskToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::GenAttentionMask specificParam = AnyCast<OpParam::GenAttentionMask>(param);

    paramsJson["qSeqLen"] = specificParam.qSeqLen;
    paramsJson["headNum"] = specificParam.headNum;

    return paramsJson.dump();
}

std::string KVCacheToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::KVCache specificParam = AnyCast<OpParam::KVCache>(param);

    paramsJson["type"] = specificParam.type;
    paramsJson["qSeqLen"] = specificParam.qSeqLen;
    paramsJson["kvSeqLen"] = specificParam.kvSeqLen;
    paramsJson["batchRunStatus"] = specificParam.batchRunStatus;
    paramsJson["seqLen"] = specificParam.seqLen;
    paramsJson["tokenOffset"] = specificParam.tokenOffset;

    return paramsJson.dump();
}

std::string PadWithHiddenStateToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::PadWithHiddenState specificParam = AnyCast<OpParam::PadWithHiddenState>(param);

    paramsJson["qSeqLen"] = specificParam.qSeqLen;
    paramsJson["maxSeqLen"] = specificParam.maxSeqLen;

    return paramsJson.dump();
}

std::string PadToJson(const Any &param)
{
    (void)param;
    return "{}";
}

std::string RopeQConcatToJson(const Any &param)
{
    (void)param;
    return "{}";
}

std::string SwigluQuantToJson(const Any &param)
{
    (void)param;
    return "{}";
}
std::string PagedAttentionToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::PagedAttention specificParam = AnyCast<OpParam::PagedAttention>(param);

    paramsJson["type"] = specificParam.type;
    paramsJson["isTriuMask"] = specificParam.isTriuMask;
    paramsJson["identityM"] = specificParam.identityM;
    paramsJson["headSize"] = specificParam.headSize;
    paramsJson["tor"] = specificParam.tor;
    paramsJson["kvHead"] = specificParam.kvHead;
    paramsJson["headSize"] = specificParam.headSize;
    paramsJson["maskType"] = specificParam.maskType;
    paramsJson["qSeqLen"] = specificParam.qSeqLen;
    paramsJson["kvSeqLen"] = specificParam.kvSeqLen;
    paramsJson["batchRunStatus"] = specificParam.batchRunStatus;
    paramsJson["compressHead"] = specificParam.compressHead;
    paramsJson["quantType"] = specificParam.quantType;
    paramsJson["outDataType"] = specificParam.outDataType;
    paramsJson["scaleType"] = specificParam.scaleType;
    paramsJson["headDimV"] = specificParam.headDimV;
    return paramsJson.dump();
}

std::string MLAToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::MLA specificParam = AnyCast<OpParam::MLA>(param);

    paramsJson["type"] = specificParam.type;
    paramsJson["tor"] = specificParam.tor;
    paramsJson["kvHead"] = specificParam.kvHead;
    paramsJson["headSize"] = specificParam.headSize;
    paramsJson["qSeqLen"] = specificParam.qSeqLen;
    paramsJson["kvSeqLen"] = specificParam.kvSeqLen;
    paramsJson["isRing"] = specificParam.isRing;
    paramsJson["windowSize"] = specificParam.windowSize;
    return paramsJson.dump();
}

std::string ReshapeAndCacheToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::ReshapeAndCache specificParam = AnyCast<OpParam::ReshapeAndCache>(param);

    paramsJson["type"] = specificParam.type;

    return paramsJson.dump();
}

std::string PagedCacheLoadToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::PagedCacheLoad specificParam = AnyCast<OpParam::PagedCacheLoad>(param);

    paramsJson["type"] = specificParam.type;
    paramsJson["cuSeqLens"] = specificParam.cuSeqLens;
    paramsJson["hasSeqStarts"] = specificParam.hasSeqStarts;

    return paramsJson.dump();
}

std::string RopeGradToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::RopeGrad specificParam = AnyCast<OpParam::RopeGrad>(param);

    paramsJson["qSeqLen"] = specificParam.qSeqLen;

    return paramsJson.dump();
}

std::string BlockCopyToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::BlockCopy specificParam = AnyCast<OpParam::BlockCopy>(param);
    paramsJson["type"] = specificParam.type;
    return paramsJson.dump();
}

std::string RopeToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::Rope specificParam = AnyCast<OpParam::Rope>(param);

    paramsJson["rotaryCoeff"] = specificParam.rotaryCoeff;
    paramsJson["cosFormat"] = specificParam.cosFormat;

    return paramsJson.dump();
}

std::string StridedBatchMatmulToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::StridedBatchMatmul specificParam = AnyCast<OpParam::StridedBatchMatmul>(param);

    paramsJson["headNum"] = specificParam.headNum;
    paramsJson["batch"] = specificParam.batch;
    paramsJson["transA"] = specificParam.transA;
    paramsJson["transB"] = specificParam.transB;
    paramsJson["m"] = specificParam.m;
    paramsJson["k"] = specificParam.k;
    paramsJson["v"] = specificParam.n;
    paramsJson["lda"] = specificParam.lda;
    paramsJson["ldb"] = specificParam.ldb;
    paramsJson["ldc"] = specificParam.ldc;
    paramsJson["strideA"] = specificParam.strideA;
    paramsJson["strideB"] = specificParam.strideB;
    paramsJson["strideC"] = specificParam.strideC;

    return paramsJson.dump();
}

std::string ToppsampleToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::Toppsample specificParam = AnyCast<OpParam::Toppsample>(param);

    paramsJson["randSeed"] = specificParam.randSeed;

    return paramsJson.dump();
}

std::string UnpadFlashAttentionNzToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::UnpadFlashAttentionNz specificParam = AnyCast<OpParam::UnpadFlashAttentionNz>(param);

    paramsJson["type"] = specificParam.type;
    paramsJson["headSize"] = specificParam.headSize;
    paramsJson["qSeqLen"] = specificParam.qSeqLen;
    paramsJson["kvSeqLen"] = specificParam.kvSeqLen;
    paramsJson["tor"] = specificParam.tor;
    paramsJson["kvHead"] = specificParam.kvHead;
    paramsJson["batchRunStatus"] = specificParam.batchRunStatus;
    paramsJson["isTriuMask"] = specificParam.isTriuMask;
    paramsJson["maskType"] = specificParam.maskType;
    paramsJson["alibiLeftAlign"] = specificParam.alibiLeftAlign;
    paramsJson["isAlibiMaskSqrt"] = specificParam.isAlibiMaskSqrt;
    paramsJson["compressHead"] = specificParam.compressHead;
    paramsJson["dataDimOrder"] = specificParam.dataDimOrder;
    paramsJson["scaleType"] = specificParam.scaleType;
    paramsJson["windowSize"] = specificParam.windowSize;
    paramsJson["cacheType"] = specificParam.cacheType;
    paramsJson["precType"] = specificParam.precType;

    std::stringstream ss;
    for (size_t i = 0; i < specificParam.kTensorList.size(); ++i) {
        ss << "\nkTensorList[" << i << "]: " << specificParam.kTensorList.at(i).ToString();
    }
    for (size_t i = 0; i < specificParam.vTensorList.size(); ++i) {
        ss << "\nvTensorList[" << i << "]: " << specificParam.vTensorList.at(i).ToString();
    }
    return paramsJson.dump() + ss.str();
}

std::string UnpadFlashAttentionToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::UnpadFlashAttention specificParam = AnyCast<OpParam::UnpadFlashAttention>(param);

    paramsJson["type"] = specificParam.type;
    paramsJson["headSize"] = specificParam.headSize;
    paramsJson["qSeqLen"] = specificParam.qSeqLen;
    paramsJson["kvSeqLen"] = specificParam.kvSeqLen;
    paramsJson["tor"] = specificParam.tor;
    paramsJson["kvHead"] = specificParam.kvHead;
    paramsJson["batchRunStatus"] = specificParam.batchRunStatus;
    paramsJson["isClamp"] = specificParam.isClamp;
    paramsJson["clampMin"] = specificParam.clampMin;
    paramsJson["clampMax"] = specificParam.clampMax;
    paramsJson["maskType"] = specificParam.maskType;
    paramsJson["alibiLeftAlign"] = specificParam.alibiLeftAlign;
    paramsJson["isAlibiMaskSqrt"] = specificParam.isAlibiMaskSqrt;
    paramsJson["compressHead"] = specificParam.compressHead;
    paramsJson["isTriuMask"] = specificParam.isTriuMask;
    paramsJson["quantType"] = specificParam.quantType;
    paramsJson["outDataType"] = specificParam.outDataType;
    paramsJson["scaleType"] = specificParam.scaleType;
    paramsJson["windowSize"] = specificParam.windowSize;
    paramsJson["cacheType"] = specificParam.cacheType;
    paramsJson["headDimV"] = specificParam.headDimV;
    paramsJson["kvShareMap"] = specificParam.kvShareMap;
    paramsJson["kvShareLen"] = specificParam.kvShareLen;

    std::stringstream ss;
    for (size_t i = 0; i < specificParam.kTensorList.size(); ++i) {
        ss << "\nkTensorList[" << i << "]: " << specificParam.kTensorList.at(i).ToString();
    }
    for (size_t i = 0; i < specificParam.vTensorList.size(); ++i) {
        ss << "\nvTensorList[" << i << "]: " << specificParam.vTensorList.at(i).ToString();
    }

    for (size_t i = 0; i < specificParam.kShareTensorList.size(); ++i) {
        ss << "\nkShareTensorList[" << i << "]: " << specificParam.kShareTensorList.at(i).ToString();
    }
    for (size_t i = 0; i < specificParam.vShareTensorList.size(); ++i) {
        ss << "\nvShareTensorList[" << i << "]: " << specificParam.vShareTensorList.at(i).ToString();
    }
    return paramsJson.dump() + ss.str();
}

std::string UnpadWithHiddenStateToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::UnpadWithHiddenState specificParam = AnyCast<OpParam::UnpadWithHiddenState>(param);

    paramsJson["qSeqLen"] = specificParam.qSeqLen;
    paramsJson["maxSeqLen"] = specificParam.maxSeqLen;

    return paramsJson.dump();
}

std::string UnpadToJson(const Any &param)
{
    (void)param;
    return "{}";
}

std::string RINGMLAToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::RINGMLA specificParam = AnyCast<OpParam::RINGMLA>(param);

    paramsJson["type"] = specificParam.type;
    paramsJson["tor"] = specificParam.tor;
    paramsJson["kvHead"] = specificParam.kvHead;
    paramsJson["headSize"] = specificParam.headSize;
    paramsJson["qSeqLen"] = specificParam.qSeqLen;
    paramsJson["kvSeqLen"] = specificParam.kvSeqLen;
    paramsJson["isRing"] = specificParam.isRing;
    return paramsJson.dump();
}

std::string GmmDeqSwigluQuantGmmDeqToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::GmmDeqSwigluQuantGmmDeq specificParam = AnyCast<OpParam::GmmDeqSwigluQuantGmmDeq>(param);

    paramsJson["outputType"] = specificParam.outputType;
    paramsJson["groupListType"] = specificParam.groupListType;
    paramsJson["weightUpPermuteType"] = specificParam.weightUpPermuteType;
    paramsJson["transposeWeightUp"] = specificParam.transposeWeightUp;
    paramsJson["transposeWeightDown"] = specificParam.transposeWeightDown;
    return paramsJson.dump();
}

std::string MmDeqSwigluQuantMmDeqToJson(const Any &param)
{
    nlohmann::json paramsJson;
    OpParam::MmDeqSwigluQuantMmDeq specificParam = AnyCast<OpParam::MmDeqSwigluQuantMmDeq>(param);

    paramsJson["outputType"] = specificParam.outputType;
    paramsJson["weightUpPermuteType"] = specificParam.weightUpPermuteType;
    paramsJson["transposeWeightUp"] = specificParam.transposeWeightUp;
    paramsJson["transposeWeightDown"] = specificParam.transposeWeightDown;
    return paramsJson.dump();
}

REG_STRINGIFY(OpParam::BlockCopy, BlockCopyToJson);
REG_STRINGIFY(OpParam::FastSoftMaxGrad, FastSoftMaxGradToJson);
REG_STRINGIFY(OpParam::FastSoftMax, FastSoftMaxToJson);
REG_STRINGIFY(OpParam::Gating, GatingToJson);
REG_STRINGIFY(OpParam::GenAttentionMask, GenAttentionMaskToJson);
REG_STRINGIFY(OpParam::KVCache, KVCacheToJson);
REG_STRINGIFY(OpParam::PadWithHiddenState, PadWithHiddenStateToJson);
REG_STRINGIFY(OpParam::Pad, PadToJson);
REG_STRINGIFY(OpParam::PagedAttention, PagedAttentionToJson);
REG_STRINGIFY(OpParam::MLA, MLAToJson);
REG_STRINGIFY(OpParam::RINGMLA, RINGMLAToJson);
REG_STRINGIFY(OpParam::ReshapeAndCache, ReshapeAndCacheToJson);
REG_STRINGIFY(OpParam::PagedCacheLoad, PagedCacheLoadToJson);
REG_STRINGIFY(OpParam::RopeGrad, RopeGradToJson);
REG_STRINGIFY(OpParam::Rope, RopeToJson);
REG_STRINGIFY(OpParam::StridedBatchMatmul, StridedBatchMatmulToJson);
REG_STRINGIFY(OpParam::Toppsample, ToppsampleToJson);
REG_STRINGIFY(OpParam::UnpadFlashAttentionNz, UnpadFlashAttentionNzToJson);
REG_STRINGIFY(OpParam::UnpadFlashAttention, UnpadFlashAttentionToJson);
REG_STRINGIFY(OpParam::UnpadWithHiddenState, UnpadWithHiddenStateToJson);
REG_STRINGIFY(OpParam::Unpad, UnpadToJson);
REG_STRINGIFY(OpParam::RopeQConcat, RopeQConcatToJson);
REG_STRINGIFY(OpParam::SwigluQuant, SwigluQuantToJson);
REG_STRINGIFY(OpParam::GmmDeqSwigluQuantGmmDeq, GmmDeqSwigluQuantGmmDeqToJson);
REG_STRINGIFY(OpParam::MmDeqSwigluQuantMmDeq, MmDeqSwigluQuantMmDeqToJson);
} // namespace AtbOps
