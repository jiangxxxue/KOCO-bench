#include "atb/atb_graph_op.h"
#include "utils/utils.h"
atb::Status CreateGraphOperation(atb::Operation **operation)
{
 // 构图流程
 // 图算子的输入a,b,c,d
 // 计算公式：(a+b) + (c+d)
 // 输入是4个参数，输出是1个参数，有3个add算子，中间产生的临时输出是2个
 atb::GraphParam opGraph;
 opGraph.inTensorNum = 4;
 opGraph.outTensorNum = 1;
 opGraph.internalTensorNum = 2;
 opGraph.nodes.resize(3);
 enum InTensorId
 { // 定义各TensorID
 IN_TENSOR_A = 0,
 IN_TENSOR_B,
 IN_TENSOR_C,
 IN_TENSOR_D,
 ADD3_OUT,
 ADD1_OUT,
 ADD2_OUT
 };
 size_t nodeId = 0;
 atb::Node &addNode = opGraph.nodes.at(nodeId++);
 atb::Node &addNode2 = opGraph.nodes.at(nodeId++);
 atb::Node &addNode3 = opGraph.nodes.at(nodeId++);
 atb::Operation *op = nullptr;
 atb::infer::ElewiseParam addParam;
 addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
 auto status = atb::CreateOperation(addParam, &addNode.operation);
 CHECK_RET(status, "addParam CreateOperation failed. status: " + std::to_string(status));
 addNode.inTensorIds = {IN_TENSOR_A, IN_TENSOR_B};
 addNode.outTensorIds = {ADD1_OUT};
 atb::infer::ElewiseParam addParam2;
 addParam2.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
 status = atb::CreateOperation(addParam2, &addNode2.operation);
 CHECK_RET(status, "addParam2 CreateOperation failed. status: " + std::to_string(status));
 addNode2.inTensorIds = {IN_TENSOR_C, IN_TENSOR_D};
 addNode2.outTensorIds = {ADD2_OUT};
 atb::infer::ElewiseParam addParam3;
 addParam3.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
 status = CreateOperation(addParam3, &addNode3.operation);
 CHECK_RET(status, "addParam3 CreateOperation failed. status: " + std::to_string(status));
 addNode3.inTensorIds = {ADD1_OUT, ADD2_OUT};
 addNode3.outTensorIds = {ADD3_OUT};
 status = atb::CreateOperation(opGraph, operation);
 CHECK_RET(status, "GraphParam CreateOperation failed. status: " + std::to_string(status));
 return atb::NO_ERROR;
}
