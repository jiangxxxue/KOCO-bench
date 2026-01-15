#include "aclnn/aclnn_operation_base.h"
#include "utils/log.h"
AclnnBaseOperation::AclnnBaseOperation(const std::string &opName) : opName_(opName)
{
}
AclnnBaseOperation::~AclnnBaseOperation()
{
 aclExecutor_ = nullptr;
}
std::string AclnnBaseOperation::GetName() const
{
 return opName_;
}
atb::Status AclnnBaseOperation::Setup(
 const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context)
{
 LOG_INFO(opName_ + " setup start");
 // 调用子类，创建输入输出tensor，并存入VariantPack
 int ret = CreateAclnnVariantPack(variantPack);
 if (ret != 0)
 {
 LOG_ERROR(opName_ + " call CreateAclnnVariantPack fail, error: " + std::to_string(ret));
 return atb::ERROR_INVALID_PARAM;
 }
 // 调用子类，获取Executor和Workspace
 ret = SetAclnnWorkspaceExecutor();
 if (ret != 0)
 {
 LOG_ERROR(
 opName_ + " call CreateAclnnVaSetAclnnWorkspaceExecutorriantPack fail, error: " + 
std::to_string(ret));
 return atb::ERROR_INVALID_PARAM;
 }
 // 返回计算出的workspaceSize
 workspaceSize = workspaceSize_;
 LOG_INFO(opName_ + " setup end");
 return ret;
}
atb::Status AclnnBaseOperation::Execute(
 const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize, atb::Context *context)
{
 LOG_INFO(opName_ + " execute start");
 if (!context)
 {
 LOG_ERROR(opName_ + " execute fail, context param is null");
 return atb::ERROR_INVALID_PARAM;
 }
 aclrtStream stream = context->GetExecuteStream();
 if (!stream)
 {
 LOG_ERROR(opName_ + " execute fail, execute stream in context is null");
 return atb::ERROR_INVALID_PARAM;
 }
 // 更新数据传入的地址
 int ret = UpdateAclnnVariantPack(variantPack);
 if (ret != 0)
 {
 LOG_ERROR(opName_ + " call UpdateAclnnVariantPack fail, error: " + std::to_string(ret));
 return atb::ERROR_CANN_ERROR;
 }
 LOG_INFO("Input workspaceSize " + std::to_string(workspaceSize) + " localCache workspaceSize " +
 std::to_string(workspaceSize_));
 ret = ExecuteAclnnOp(workspace, stream);
 if (ret != 0)
 {
 LOG_ERROR(opName_ + " call ExecuteAclnnOp fail, error: " + std::to_string(ret));
 return atb::ERROR_CANN_ERROR;
 }
 LOG_INFO(opName_ + " execute start");
 return ret;
}
atb::Status AclnnBaseOperation::UpdateAclnnVariantPack(const atb::VariantPack &variantPack)
{
 for (size_t i = 0; i < aclInTensors_.size(); ++i)
 {
 int ret = -1;
 if (!aclInTensors_[i]->needUpdateTensorDataPtr)
 {
 continue;
 }
 aclInTensors_[i]->atbTensor = variantPack.inTensors.at(i);
 ret = aclSetInputTensorAddr(aclExecutor_,
 aclInTensors_[i]->tensorIdx,
 aclInTensors_[i]->tensor,
 aclInTensors_[i]->atbTensor.deviceData);
 if (ret != 0)
 {
 LOG_ERROR(
 "inTensor " + std::to_string(i) + " call UpdateAclTensorDataPtr fail, error: " + std::to_string(ret));
 return atb::ERROR_CANN_ERROR;
 }
 }
 for (size_t i = 0; i < aclOutTensors_.size(); ++i)
 {
 int ret = -1;
 if (!aclOutTensors_[i]->needUpdateTensorDataPtr)
 {
 continue;
 }
  aclOutTensors_[i]->atbTensor = variantPack.outTensors.at(i);
 ret = aclSetOutputTensorAddr(aclExecutor_,
 aclOutTensors_[i]->tensorIdx,
 aclOutTensors_[i]->tensor,
 aclOutTensors_[i]->atbTensor.deviceData);
 if (ret != 0)
 {
 LOG_ERROR(
 "outTensor " + std::to_string(i) + " call UpdateAclTensorDataPtr fail, error: " + std::to_string(ret));
 return atb::ERROR_CANN_ERROR;
 }
 }
 return atb::NO_ERROR;
}