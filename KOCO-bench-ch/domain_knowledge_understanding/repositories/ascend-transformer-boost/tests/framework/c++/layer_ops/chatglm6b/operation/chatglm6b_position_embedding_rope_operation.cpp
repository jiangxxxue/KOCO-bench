/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "chatglm6b_position_embedding_rope_operation.h"

namespace atb_speed {
enum ChatGlm6BPositionEmbeddingTensorId {
    IN_MIXEDQKV_ID = 0,                      // [seqLen, batch, 3 * hiddenSize], half
    IN_POSITION_IDS_ID,                      // [batch, 2, seqLen], int64
    IN_COS_TABLE_ID,                         // [maxSeqLen, 1, headDim / 2], half
    IN_SIN_TABLE_ID,                         // [maxSeqLen, 1, headDim / 2], half
    IN_SEQLEN_ID,                            // [batch], int32
    OUT_QEMBEDDED_ID,                        // [seqLen, batch, headNum, headDim], half
    OUT_KEMBEDDED_ID,                        // [seqLen, batch, headNum, headDim], half
    OUT_VALUE,                               // [seqLen, batch, headNum, headDim], half
    INTERMEDIATE_QLAYER_ID,                  // [seqLen, batch, headNum, headDim], half
    INTERMEDIATE_KLAYER_ID,                  // [seqLen, batch, headNum, headDim], half
    INTERMEDIATE_POSITION_IDS0_ID,           // [batch, 1, seqLen], int64
    INTERMEDIATE_POSITION_IDS1_ID,           // [batch, 1, seqLen], int64
    INTERMEDIATE_POSITION_IDS_TRANSPOSE0_ID, // [seqLen, batch], int64
    INTERMEDIATE_POSITION_IDS_TRANSPOSE1_ID, // [seqLen, batch], int64
    INTERMEDIATE_COS0_ID,                    // [seqLen, batch, headDim / 2], half
    INTERMEDIATE_COS1_ID,                    // [seqLen, batch, headDim / 2], half
    INTERMEDIATE_SIN0_ID,                    // [seqLen, batch, headDim / 2], half
    INTERMEDIATE_SIN1_ID,                    // [seqLen, batch, headDim / 2], half
    INTERMEDIATE_COSSUM_ID,                  // [seqLen, batch, headDim], half
    INTERMEDIATE_SINSUM_ID                   // [seqLen, batch, headDim], half
};

static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 12;
static const uint64_t NODE_COUNT = 11;

void Squeeze1(const atb::Dims &oldShape, atb::Dims &newShape)
{
    if (oldShape.dims[1] == 1) {
        newShape.dimNum = oldShape.dimNum - 1;
        newShape.dims[0] = oldShape.dims[0];
        for (size_t i = 1; i < newShape.dimNum; i++) {
            newShape.dims[i] = oldShape.dims[i + 1];
        }
    } else {
        newShape = oldShape;
    }
}

void RopeCosSinReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - 1;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    newShape.dims[1] = oldShape.dims[2]; // 2: 设置新张量第二维的长度
}

void RopeQKReshapeFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - 2;
    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    newShape.dims[1] = oldShape.dims[2] * oldShape.dims[3]; // 2, 3: 设置新张量第二维的长度
}

atb::Status CreateChatGlm6BPositionEmbeddingOperation(const ChatGlm6BPositionEmbeddingParam &param,
                                                      atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "PositionEmbeddingOperation";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    auto &splitQKVNode = opGraph.nodes.at(nodeId++);
    auto &splitPositionIdsNode = opGraph.nodes.at(nodeId++);
    auto &transposePositionIds0Node = opGraph.nodes.at(nodeId++);
    auto &transposePositionIds1Node = opGraph.nodes.at(nodeId++);
    auto &embeddingCos0Node = opGraph.nodes.at(nodeId++);
    auto &embeddingCos1Node = opGraph.nodes.at(nodeId++);
    auto &embeddingSin0Node = opGraph.nodes.at(nodeId++);
    auto &embeddingSin1Node = opGraph.nodes.at(nodeId++);
    auto &concateCosNode = opGraph.nodes.at(nodeId++);
    auto &concateSinNode = opGraph.nodes.at(nodeId++);
    auto &ropeNode = opGraph.nodes.at(nodeId++);

    atb::infer::SplitParam splitQKVParam;
    splitQKVParam.splitDim = 3; // 3: 在第四维上进行切分
    splitQKVParam.splitNum = 3; // 3: 进行三等分
    CreateOperation(splitQKVParam, &splitQKVNode.operation);
    splitQKVNode.inTensorIds = {IN_MIXEDQKV_ID};
    splitQKVNode.outTensorIds = {INTERMEDIATE_QLAYER_ID, INTERMEDIATE_KLAYER_ID, OUT_VALUE};
    splitQKVNode.inTensorReshapeFuncs.resize(splitQKVNode.inTensorIds.size());
    splitQKVNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = oldShape.dimNum + 1;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;                    // 2: 设置张量第三维的大小
        newShape.dims[3] = oldShape.dims[2] / param.headNum; // 2, 3: 设置张量第四维的大小
    };

    atb::infer::SplitParam splitPositionIdsParam;
    splitPositionIdsParam.splitDim = 1; // 1: 在第二维上进行切分
    splitPositionIdsParam.splitNum = 2; // 2: 进行二等分
    CreateOperation(splitPositionIdsParam, &splitPositionIdsNode.operation);
    splitPositionIdsNode.inTensorIds = {IN_POSITION_IDS_ID};
    splitPositionIdsNode.outTensorIds = {INTERMEDIATE_POSITION_IDS0_ID, INTERMEDIATE_POSITION_IDS1_ID};

    atb::infer::TransposeParam transposePositionIds0Param;
    transposePositionIds0Param.perm = {1, 0}; // 1, 0: 对前两维进行转置
    CreateOperation(transposePositionIds0Param, &transposePositionIds0Node.operation);
    transposePositionIds0Node.inTensorIds = {INTERMEDIATE_POSITION_IDS0_ID};
    transposePositionIds0Node.outTensorIds = {INTERMEDIATE_POSITION_IDS_TRANSPOSE0_ID};
    transposePositionIds0Node.inTensorReshapeFuncs.resize(transposePositionIds0Node.inTensorIds.size());
    transposePositionIds0Node.inTensorReshapeFuncs.at(0) = &Squeeze1;

    atb::infer::TransposeParam transposePositionIds1Param;
    transposePositionIds1Param.perm = {1, 0}; // 1, 0: 对前两维进行转置
    CreateOperation(transposePositionIds1Param, &transposePositionIds1Node.operation);
    transposePositionIds1Node.inTensorIds = {INTERMEDIATE_POSITION_IDS1_ID};
    transposePositionIds1Node.outTensorIds = {INTERMEDIATE_POSITION_IDS_TRANSPOSE1_ID};
    transposePositionIds1Node.inTensorReshapeFuncs.resize(transposePositionIds1Node.inTensorIds.size());
    transposePositionIds1Node.inTensorReshapeFuncs.at(0) = &Squeeze1;

    atb::infer::GatherParam embeddingCos0Param;
    CreateOperation(embeddingCos0Param, &embeddingCos0Node.operation);
    embeddingCos0Node.inTensorIds = {IN_COS_TABLE_ID, INTERMEDIATE_POSITION_IDS_TRANSPOSE0_ID};
    embeddingCos0Node.outTensorIds = {INTERMEDIATE_COS0_ID};
    embeddingCos0Node.inTensorReshapeFuncs.resize(embeddingCos0Node.inTensorIds.size());
    embeddingCos0Node.inTensorReshapeFuncs.at(0) = &Squeeze1;

    atb::infer::GatherParam embeddingCos1Param;
    CreateOperation(embeddingCos1Param, &embeddingCos1Node.operation);
    embeddingCos1Node.inTensorIds = {IN_COS_TABLE_ID, INTERMEDIATE_POSITION_IDS_TRANSPOSE1_ID};
    embeddingCos1Node.outTensorIds = {INTERMEDIATE_COS1_ID};
    embeddingCos1Node.inTensorReshapeFuncs.resize(embeddingCos1Node.inTensorIds.size());
    embeddingCos1Node.inTensorReshapeFuncs.at(0) = &Squeeze1;

    atb::infer::GatherParam embeddingSin0Param;
    CreateOperation(embeddingSin0Param, &embeddingSin0Node.operation);
    embeddingSin0Node.inTensorIds = {IN_SIN_TABLE_ID, INTERMEDIATE_POSITION_IDS_TRANSPOSE0_ID};
    embeddingSin0Node.outTensorIds = {INTERMEDIATE_SIN0_ID};
    embeddingSin0Node.inTensorReshapeFuncs.resize(embeddingSin0Node.inTensorIds.size());
    embeddingSin0Node.inTensorReshapeFuncs.at(0) = &Squeeze1;

    atb::infer::GatherParam embeddingSin1Param;
    CreateOperation(embeddingSin1Param, &embeddingSin1Node.operation);
    embeddingSin1Node.inTensorIds = {IN_SIN_TABLE_ID, INTERMEDIATE_POSITION_IDS_TRANSPOSE1_ID};
    embeddingSin1Node.outTensorIds = {INTERMEDIATE_SIN1_ID};
    embeddingSin1Node.inTensorReshapeFuncs.resize(embeddingSin1Node.inTensorIds.size());
    embeddingSin1Node.inTensorReshapeFuncs.at(0) = &Squeeze1;

    atb::infer::ConcatParam concateCosParam;
    concateCosParam.concatDim = 2; // 2: 在第三维上进行concat
    CreateOperation(concateCosParam, &concateCosNode.operation);
    concateCosNode.inTensorIds = {INTERMEDIATE_COS0_ID, INTERMEDIATE_COS1_ID};
    concateCosNode.outTensorIds = {INTERMEDIATE_COSSUM_ID};

    atb::infer::ConcatParam concateSinParam;
    concateSinParam.concatDim = 2; // 2: 在第三维上进行concat
    CreateOperation(concateSinParam, &concateSinNode.operation);
    concateSinNode.inTensorIds = {INTERMEDIATE_SIN0_ID, INTERMEDIATE_SIN1_ID};
    concateSinNode.outTensorIds = {INTERMEDIATE_SINSUM_ID};

    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 4; // 4: 设置旋转系数
    CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMEDIATE_QLAYER_ID, INTERMEDIATE_KLAYER_ID, INTERMEDIATE_COSSUM_ID,
                            INTERMEDIATE_SINSUM_ID, IN_SEQLEN_ID};
    ropeNode.outTensorIds = {OUT_QEMBEDDED_ID, OUT_KEMBEDDED_ID};
    ropeNode.inTensorReshapeFuncs = {&RopeQKReshapeFunc, &RopeQKReshapeFunc, &RopeCosSinReshapeFunc,
                                     &RopeCosSinReshapeFunc};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 4; // 3表示输出维度
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1]; // 1: 设置张量第二维长度
        outTensorDescs.at(0).shape.dims[2] = param.headNum;                     // 2: 设置张量第三维长度
        outTensorDescs.at(0).shape.dims[3] =                                    // 3: 设置张量第四维长度
            inTensorDescs.at(0).shape.dims[2] / param.headNum / 3;              // 2, 3: 设置张量第四维长度
        outTensorDescs.at(1) = outTensorDescs.at(0);                            // 1: 设置第二个张量的描述
        outTensorDescs.at(2) = outTensorDescs.at(0);                            // 2: 设置第三个张量的描述
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace atb_speed