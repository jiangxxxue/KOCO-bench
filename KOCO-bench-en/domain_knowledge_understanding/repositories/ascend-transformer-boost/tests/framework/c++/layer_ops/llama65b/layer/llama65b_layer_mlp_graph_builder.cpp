/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "llama65b_layer_mlp_graph_builder.h"
#include <atb/atb_infer.h>

namespace atb_speed {

static uint64_t DIM3 = 3;

atb::Operation* Linear(const LlamaMlpParamGb &param)
{
    atb::Operation* op = nullptr;
    atb::infer::LinearParam linearParam;
    linearParam.hasBias = false;
    linearParam.transposeB = param.transpose;
    CreateOperation(linearParam, &op);
    return op;
}

atb::Operation* Split(const LlamaMlpParamGb &param)
{
    atb::Operation* op = nullptr;
    atb::infer::SplitParam splitParam = {2, 2};
    CreateOperation(splitParam, &op);
    return op;
}

atb::Operation* Swish(const LlamaMlpParamGb &param)
{
    atb::Operation* op = nullptr;
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CreateOperation(activationParam, &op);
    return op;
}

atb::Operation* Mul(const LlamaMlpParamGb &param)
{
    atb::Operation* op = nullptr;
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(elewiseParam, &op);
    return op;
}

atb::Status CreateLlamaMlpOperationByGraphOpBuilder(const LlamaMlpParamGb &param, atb::Operation **operation)
{
    atb::InferShapeFunc inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (param.transpose == true) {
            outTensorDescs.at(0).shape.dimNum = DIM3;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
            outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(1).shape.dims[0] / 2;
        } else {
            outTensorDescs.at(0).shape.dimNum = DIM3;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
            outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(1).shape.dims[1] / 2;
        }
        return atb::NO_ERROR;
    };

    atb::ReshapeFunc reshape_01_2 = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };
    atb::ReshapeFunc unsqueueze_0 = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum: 3
        newShape.dims[0] = 1;
        newShape.dims[1] = oldShape.dims[0];
        newShape.dims[2] = oldShape.dims[1];
    };
    atb::GraphOpBuilder* graphOpBuilder;
    CreateGraphOpBuilder(&graphOpBuilder);

    graphOpBuilder->Init(
        "LlamaMlpGraphOp",
        inferShapeFunc,
        {"hidden_states", "weight"},
        {"mlp_out"}
    );

    graphOpBuilder->Reshape("hidden_states", reshape_01_2, "hidden_states_");
    graphOpBuilder->AddOperation(Linear(param), {"hidden_states_", "weight"}, {"linear_out"});
    graphOpBuilder->Reshape("linear_out", unsqueueze_0, "linear_out_");
    graphOpBuilder->AddOperation(Split(param), {"linear_out_"}, {"gate_out", "up_out"});
    graphOpBuilder->AddOperation(Swish(param), {"gate_out"}, {"swish_out"});
    graphOpBuilder->AddOperation(Mul(param), {"swish_out", "up_out"}, {"mlp_out"});

    *operation = graphOpBuilder->Build();
    DestroyGraphOpBuilder(graphOpBuilder);
    return atb::NO_ERROR;
}
} // namespace atb_speed