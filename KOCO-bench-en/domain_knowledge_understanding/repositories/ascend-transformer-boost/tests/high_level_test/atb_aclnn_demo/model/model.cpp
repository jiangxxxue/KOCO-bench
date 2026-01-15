#define USE_MEMPOOL
#include "model/model.h"
#include "aclnn/aclnn_gelu_operation.h"
#include "utils/utils.h"
#include "atb/atb_graph_op.h"
#include "memory/memory_utils.h"
void Model::InitResource(uint32_t deviceId)
{
 // 配置deviceId
 deviceId_ = deviceId;
 auto ret = aclrtSetDevice(deviceId_);
 CHECK_RET(ret, "aclrtSetDevice failed. ret: " + std::to_string(ret));
 // 创建context，配置stream
 ret = atb::CreateContext(&mode_context_);
 CHECK_RET(ret, "ATB CreateContext failed. ret: " + std::to_string(ret));
 ret = aclrtCreateStream(&model_stream_);
 CHECK_RET(ret, "aclrtCreateStream failed. ret: " + std::to_string(ret));
 mode_context_->SetExecuteStream(model_stream_);
}
void Model::CreateModelGraph()
{
 LOG_INFO("CreateModelGraph start");
  // 这里以模型中有2个节点参与演示
 nodes_.resize(2);
 for (size_t i = 0; i < nodes_.size(); i++) {
 auto node = Node();
 nodes_[i] = node;
 }
 model_inTensors_.resize(Mode_INPUT_SIZE);
 model_outTensors_.resize(Mode_OUTPUT_SIZE);
 internalTensors_.resize(1);
 size_t nodeId = 0;
 CreateGraphOpLayer(nodeId++);
 // step2：创建aclnn算子的Node
 CreateAclnnOpLayer(nodeId);
 LOG_INFO("CreateModelGraph end");
}
void Model::CreateGraphOpLayer(size_t nodeId)
{
 // 创建图算子的opreation
 Node &graph_node = nodes_[nodeId];
 auto ret = CreateGraphOperation(&graph_node.operation_);
 CHECK_RET(ret, "CreateGraphOperation failed");
 graph_node.inTensors_.resize(graph_node.operation_->GetInputNum());
 // 设置图算子node节点的输入
 // 因为图算子的输入就是整个model的输入，因此这里直接从model的inTensors_赋值
 size_t layerInTensorId = 0;
 graph_node.inTensors_.at(layerInTensorId++) = &model_inTensors_.at(IN_TENSOR_A);
 graph_node.inTensors_.at(layerInTensorId++) = &model_inTensors_.at(IN_TENSOR_B);
 graph_node.inTensors_.at(layerInTensorId++) = &model_inTensors_.at(IN_TENSOR_C);
 graph_node.inTensors_.at(layerInTensorId++) = &model_inTensors_.at(IN_TENSOR_D);
 // 设置图算子node节点的输出，因为只有一个中间节点
 graph_node.outTensors_ = {&internalTensors_.at(0)};
 graph_node.outTensorTypes_ = {TensorType::INTERNAL_TENSOR};
};
void Model::CreateAclnnOpLayer(size_t nodeId)
{
 // 创建aclnn算子的opreation
 Node &aclnn_node = nodes_[nodeId];
 AclnnGeluParam AclnnGeluParam;
 AclnnGeluParam.geluApproximate = -1;
 aclnn_node.operation_ = new GeluOperation("Gelu", AclnnGeluParam);
 aclnn_node.inTensors_.resize(aclnn_node.operation_->GetInputNum());
 // 设置aclnn算子node节点的输入
 // 因为图算子的输出就是aclnn算子的的输入，
 size_t layerInTensorId = 0;
 aclnn_node.inTensors_.at(layerInTensorId++) = &internalTensors_.at(0);
 // 设置aclnn算子node节点的输出，model的输出
 aclnn_node.outTensors_ = {&model_outTensors_.at(GLUE_OUT)};
 aclnn_node.outTensorTypes_ = {TensorType::NOT_INTERNAL_TENSOR};
}
void Model::CreateModelInput()
{
 LOG_INFO("CreateModelInput start");
 atb::SVector<atb::TensorDesc> intensorDescs;
 intensorDescs.resize(Mode_INPUT_SIZE);
 CreateInTensorDescs(intensorDescs);
 CreateInTensors(model_inTensors_, intensorDescs);
 LOG_INFO("CreateModelInput end");
}
void Model::CreateModelOutput()
{
 LOG_INFO("CreateModelOutput start");
 atb::SVector<atb::TensorDesc> outtensorDescs;
 outtensorDescs.resize(Mode_OUTPUT_SIZE);
 // 设置输入的input desc
 atb::SVector<atb::TensorDesc> inTensorDescs;
 inTensorDescs.resize(Mode_INPUT_SIZE);
 for (size_t i = 0; i < model_inTensors_.size(); ++i) {
 inTensorDescs.at(i) = model_inTensors_.at(i).desc;
 }
 // 调用infer shape，推导出模型的输出
 InferShape(inTensorDescs, outtensorDescs);
 CreateOutTensors(model_outTensors_, outtensorDescs);
 LOG_INFO("CreateModelOutput end");
}
atb::Status Model::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
 atb::SVector<atb::TensorDesc> &outTensorDescs)
{
 // 输出的shape和输入时相同的。取第一个的输入即可
 outTensorDescs.at(0) = model_inTensors_.at(0).desc;
 return atb::NO_ERROR;
}
void Model::Execute()
{
 LOG_INFO(modelName_ + " Execute start");
 for (size_t nodeId = 0; nodeId < nodes_.size(); ++nodeId) {
 BuildNodeVariantPack(nodeId);
 atb::Status status = ExecuteNode(nodeId);
 CHECK_RET(status, "ExecuteNode " + std::to_string(nodeId) + " failed. status: " + std::to_string(status));
 }
 WaitFinish();
 LOG_INFO(modelName_ + " Execute end");
}
void Model::BuildNodeVariantPack(int nodeId)
{
 LOG_INFO("buildNodeVariantPack nodes[" + std::to_string(nodeId) + "] start");
 auto &node = nodes_.at(nodeId);
 atb::SVector<atb::TensorDesc> inTensorDescs;
 node.variantPack_.inTensors.resize(node.operation_->GetInputNum());
 inTensorDescs.resize(node.operation_->GetInputNum());
 // 获取node中operation_的输入tensor desc
 for (size_t i = 0; i < node.inTensors_.size(); ++i) {
 node.variantPack_.inTensors.at(i) = *node.inTensors_.at(i);
 inTensorDescs.at(i) = node.inTensors_.at(i)->desc;
 }
 atb::SVector<atb::TensorDesc> outTensorDescs;
 outTensorDescs.resize(node.operation_->GetOutputNum());
 // 调用operation_的InferShape，推导出out tensor的desc
 atb::Status st = node.operation_->InferShape(inTensorDescs, outTensorDescs);
 node.variantPack_.outTensors.resize(node.operation_->GetOutputNum());
 for (size_t i = 0; i < node.outTensors_.size(); ++i) {
 node.variantPack_.outTensors.at(i) = *node.outTensors_.at(i);
 if (node.outTensorTypes_.at(i) == TensorType::INTERNAL_TENSOR) {
 // 创建输出tensor的空间
 CreateTensorFromDesc(node.variantPack_.outTensors.at(i), outTensorDescs.at(i));
 *node.outTensors_.at(i) = node.variantPack_.outTensors.at(i);
 }
 }
 LOG_INFO("buildNodeVariantPack nodes[" + std::to_string(nodeId) + "] end");
}
atb::Status Model::ExecuteNode(int nodeId)
{
 auto &node = nodes_.at(nodeId);
 uint64_t workspaceSize = 0;
 atb::Status status = node.operation_->Setup(node.variantPack_, workspaceSize, mode_context_);
 CHECK_RET(status, "Setup node " + std::to_string(nodeId) + " failed. status: " + std::to_string(status));
 LOG_INFO("Get node[" + std::to_string(nodeId) + "] workspace size:" + std::to_string(workspaceSize));
#ifdef USE_MEMPOOL
 CreateWorkspaceBuffer(nodeId, workspaceSize);
#else
 if (workspaceSize != 0) {
 status = aclrtMalloc(&node.workspace_, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
 CHECK_RET(status, "alloc error!");
 }
#endif
 LOG_INFO("Execute node[" + std::to_string(nodeId) + "] start");
 status =
 node.operation_->Execute(node.variantPack_, (uint8_t *)(node.workspace_), workspaceSize, 
mode_context_);
 CHECK_RET(status, "Execute node " + std::to_string(nodeId) + " failed. status: " + std::to_string(status));
 LOG_INFO("Execute node[" + std::to_string(nodeId) + "] end");
 return atb::NO_ERROR;
}
void Model::CreateWorkspaceBuffer(int nodeId, int workspaceSizeNeeded)
{
 auto &node = nodes_.at(nodeId);
 if(workspaceSizeNeeded == 0) {
 LOG_INFO("skip the workspacebuffer for size 0");
 return ;
 }
 if (node.workspaceBlockId_ == -1 || node.workspaceSize_ == 0) {
 node.workspaceSize_ = workspaceSizeNeeded;
 GetMemoryManager().AllocateBlock(node.workspaceSize_, node.workspaceBlockId_);
 }
 if (node.workspaceSize_ < workspaceSizeNeeded) {
 GetMemoryManager().FreeBlock(node.workspaceBlockId_);
 GetMemoryManager().AllocateBlock(workspaceSizeNeeded, node.workspaceBlockId_);
 node.workspaceSize_ = workspaceSizeNeeded;
 }
 GetMemoryManager().GetBlockPtr(node.workspaceBlockId_, node.workspace_);
}
void Model::FreeResource()
{
 LOG_INFO("FreeResource start");
 auto status = aclrtDestroyStream(model_stream_); // 销毁stream
 CHECK_RET(status, "aclrtDestroyStream failed");
 status = atb::DestroyContext(mode_context_); // 销毁context
 CHECK_RET(status, "aclrtDestroyStream failed");
 // 释放operation
 for (auto &node : nodes_) {
 atb::DestroyOperation(node.operation_);
#ifdef USE_MEMPOOL
 GetMemoryManager().FreeBlock(node.workspaceBlockId_);
#endif
 }
 // 销毁输入tensor
 for (size_t i = 0; i < model_inTensors_.size(); i++) {
     aclrtFree(model_inTensors_.at(i).deviceData);
 }
 // 销毁输出tensor
 for (size_t i = 0; i < model_outTensors_.size(); i++) {
 aclrtFree(model_outTensors_.at(i).deviceData);
 }
 // 释放中间tensor
 for (size_t i = 0; i < internalTensors_.size(); i++) {
 aclrtFree(internalTensors_.at(i).deviceData);
 }
 aclrtResetDevice(deviceId_); // 重置deviceId
 LOG_INFO("FreeResource end");
}
void Model::WaitFinish()
{
 // step9：销毁创建的对象，释放内存
 // 流同步，作用是等待device侧任务计算完成
 auto ret = aclrtSynchronizeStream(model_stream_);
 CHECK_RET(ret, "sync error!");
}