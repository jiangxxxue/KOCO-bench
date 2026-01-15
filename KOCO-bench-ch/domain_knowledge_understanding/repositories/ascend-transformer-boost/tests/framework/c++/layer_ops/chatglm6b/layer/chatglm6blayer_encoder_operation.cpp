/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "chatglm6blayer_encoder_operation.h"

namespace atb_speed {

enum Chatglm6BLayerEncoderTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_NORMBIAS,
    IN_QKVMIXDWEIGHT,
    IN_QKVMIXDBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEARBIAS,
    IN_SELFOUTNORMWEIGHT,
    IN_SELFOUTNORMBIAS,
    IN_FFNLINEARWEIGHT,
    IN_FFNLINEARBIAS,
    IN_FFNOUTLINEARWEIGHT,
    IN_FFNOUTLINEARBIAS,
    IN_POSITIONIDS,
    IN_COSTABLE,
    IN_SINTABLE,
    IN_ATTENTIONMASK,
    IN_SEQLEN,
    OUT_GLMLAYEROUT,
    OUT_PRESENTKEY,
    OUT_PRESENTVALUE,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDLINEAROUTQKV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_FFNOUT,
    INTERMIDATE_FFNLINEAROUT,
};

static const uint64_t IN_TENSOR_COUNT = 18;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 9;
static const uint64_t NODE_COUNT = 10;

atb::Status CreateChatGlm6BLayerEncoderOperation(const ChatGlm6BLayerEncoderParam &param, atb::Operation **operation)
{
#if 0
        atb::GraphParam opGraph;
        opGraph.name = "LayerEncoderOperation";
        opGraph.inTensorNum = IN_TENSOR_COUNT;
        opGraph.outTensorNum = OUT_TENSOR_COUNT;
        opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
        opGraph.nodes.resize(NODE_COUNT);

        size_t nodeId = 0;
        atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
        atb::Node &mixdQkvLinearNode = opGraph.nodes.at(nodeId++);
        atb::Node &positionEmbeddingNode = opGraph.nodes.at(nodeId++);
        atb::Node &selfAttentionNode = opGraph.nodes.at(nodeId++);
        atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
        atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
        atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
        atb::Node &ffnNode = opGraph.nodes.at(nodeId++);
        atb::Node &ffnLinearNode = opGraph.nodes.at(nodeId++);
        atb::Node &ffnResidualAddNode = opGraph.nodes.at(nodeId++);

        atb::infer_old::NormParam inputNormParam;
        inputNormParam.layerNormEps = param.layerNormEps;
        CreateOperation(inputNormParam, &inputNormNode.operation);
        inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_NORMBIAS};
        inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

        atb::infer::LinearParam mixdQkvLinearParam;
        CreateOperation(mixdQkvLinearParam, &mixdQkvLinearNode.operation);
        mixdQkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXDWEIGHT, IN_QKVMIXDBIAS};
        mixdQkvLinearNode.outTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV};

        atb::infer_old::PositionEmbeddingFusionParam positionEmbeddingParam;
        positionEmbeddingParam.headNum = param.headNum;
        CreateOperation(positionEmbeddingParam, &positionEmbeddingNode.operation);
        positionEmbeddingNode.inTensorIds = {INTERMIDATE_MIXEDLINEAROUTQKV, IN_POSITIONIDS, IN_COSTABLE, IN_SINTABLE,
                                             IN_SEQLEN};
        positionEmbeddingNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, OUT_PRESENTKEY, OUT_PRESENTVALUE};

        atb::infer_old::SelfAttentionParam selfAttentionParam;
        selfAttentionParam.headDim = param.headDim;
        selfAttentionParam.headNum = param.headNum;
        selfAttentionParam.qScale = param.qScale;
        selfAttentionParam.qkScale = param.qkScale;
        CreateOperation(selfAttentionParam, &selfAttentionNode.operation);
        selfAttentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, OUT_PRESENTKEY, OUT_PRESENTVALUE, IN_ATTENTIONMASK};
        selfAttentionNode.outTensorIds = {INTERMIDATE_SELFOUT};

        atb::infer::LinearParam selfOutLinearParam;
        CreateOperation(selfOutLinearParam, &selfOutLinearNode.operation);
        selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARBIAS};
        selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

        atb::infer_old::AddParam selfResidualAddParam;
        selfResidualAddParam.scale = param.residualAddScale;
        CreateOperation(selfResidualAddParam, &selfResidualAddNode.operation);
        selfResidualAddNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, INTERMIDATE_SELFLINEAROUT};
        selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

        atb::infer_old::NormParam selfNormParam;
        selfNormParam.layerNormEps = param.layerNormEps;
        CreateOperation(selfNormParam, &selfNormNode.operation);
        selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT, IN_SELFOUTNORMBIAS};
        selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

        atb::infer::LinearActivationParam ffnParam;
        CreateOperation(ffnParam, &ffnNode.operation);
        ffnNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_FFNLINEARWEIGHT, IN_FFNLINEARBIAS};
        ffnNode.outTensorIds = {INTERMIDATE_FFNOUT};

        atb::infer::LinearParam ffnLinearParam;
        CreateOperation(ffnLinearParam, &ffnLinearNode.operation);
        ffnLinearNode.inTensorIds = {INTERMIDATE_FFNOUT, IN_FFNOUTLINEARWEIGHT, IN_FFNOUTLINEARBIAS};
        ffnLinearNode.outTensorIds = {INTERMIDATE_FFNLINEAROUT};

        atb::infer_old::AddParam ffnResidualAddParam;
        ffnResidualAddParam.scale = param.residualAddScale;
        CreateOperation(ffnResidualAddParam, &ffnResidualAddNode.operation);
        ffnResidualAddNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, INTERMIDATE_FFNLINEAROUT};
        ffnResidualAddNode.outTensorIds = {OUT_GLMLAYEROUT};

        opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                     atb::SVector<atb::TensorDesc> &outTensorDescs) {
            const size_t dim2 = 2;
            const size_t dim3 = 3;
            outTensorDescs.at(0) = inTensorDescs.at(0);
            outTensorDescs.at(1) = inTensorDescs.at(0);
            outTensorDescs.at(1).shape.dims[dim2] = param.headNum;
            outTensorDescs.at(1).shape.dimNum++;
            outTensorDescs.at(1).shape.dims[dim3] = param.dk;
            const size_t tensorId2 = 2;
            outTensorDescs.at(tensorId2) = outTensorDescs.at(1);
            return atb::NO_ERROR;
        };

        atb::CreateOperation(opGraph, operation);
#endif
    return atb::NO_ERROR;
}
} // namespace atb_speed