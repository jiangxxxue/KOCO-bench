#include "aclnn_gelu_operation.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_gelu.h"
#include "aclnnop/aclnn_gelu_v2.h"
#include "utils/log.h"
#include "utils/utils.h"
const int DIM0 = 0;
const int DIM1 = 1;
const int DIM2 = 2;
const int DIM3 = 3;
GeluOperation::GeluOperation(const std::string &name, AclnnGeluParam param) : 
AclnnBaseOperation(name), param_(param)
{
}
atb::Status GeluOperation::InferShape(
 const atb::SVector<atb::TensorDesc> &inTensorDesc, atb::SVector<atb::TensorDesc> &outTensorDesc) const
{
 LOG_INFO(opName_ + " InferShape start");
 outTensorDesc.at(0).format = inTensorDesc.at(0).format;
 outTensorDesc.at(0).dtype = inTensorDesc.at(0).dtype;
 outTensorDesc.at(0).shape.dimNum = inTensorDesc.at(0).shape.dimNum;
 if (inTensorDesc.at(0).shape.dimNum == DIM3)
 {
 LOG_INFO("[input0 dimNum = 3] CHECK " + opName_ + " input shape: [input0] " +
 std::to_string(inTensorDesc.at(0).shape.dims[DIM0]) + ", " +
 std::to_string(inTensorDesc.at(0).shape.dims[DIM1]) + ", " +
 std::to_string(inTensorDesc.at(0).shape.dims[DIM2]));
 outTensorDesc.at(0).shape.dims[DIM0] = inTensorDesc.at(0).shape.dims[DIM0];
 outTensorDesc.at(0).shape.dims[DIM1] = inTensorDesc.at(0).shape.dims[DIM1];
 outTensorDesc.at(0).shape.dims[DIM2] = inTensorDesc.at(0).shape.dims[DIM2];
 }
 else if (inTensorDesc.at(0).shape.dimNum == DIM2)
 {
 LOG_INFO("[input0 dimNum = 2] CHECK " + opName_ + " input shape: [input0] " +
 std::to_string(inTensorDesc.at(0).shape.dims[DIM0]) + ", " +
 std::to_string(inTensorDesc.at(0).shape.dims[DIM1]));
 outTensorDesc.at(0).shape.dims[DIM0] = inTensorDesc.at(0).shape.dims[DIM0];
 outTensorDesc.at(0).shape.dims[DIM1] = inTensorDesc.at(0).shape.dims[DIM1];
 }
 else
 {
 LOG_ERROR(opName_ + " invalid dimNum = " + std::to_string(inTensorDesc.at(0).shape.dimNum));
 }
 LOG_INFO(opName_ + " InferShape end");
 return atb::NO_ERROR;
}
uint32_t GeluOperation::GetInputNum() const
{
 return 1; // gelu入参个数
}
uint32_t GeluOperation::GetOutputNum() const
{
 return 1; // gelu出参个数
}
// 重写父类方法, 创建输入输出tensor，并存入VariantPack
atb::Status GeluOperation::CreateAclnnVariantPack(const atb::VariantPack &variantPack)
{
 LOG_INFO(opName_ + " CreateAclnnVariantPack start");
 auto ret = CreateAclnnInTensor(variantPack);
 if (ret != 0)
 {
 LOG_ERROR(opName_ + " CreateAclnnInTensor fail");
 return atb::ERROR_INVALID_PARAM;
 }
 ret = CreateAclnnOutTensor(variantPack);
 if (ret != 0)
 {
 LOG_ERROR(opName_ + " CreateAclNNOutTensorVariantPack fail");
 return atb::ERROR_INVALID_PARAM;
 }
 LOG_INFO(opName_ + " CreateAclnnVariantPack end");
 return atb::NO_ERROR;
}
atb::Status GeluOperation::CreateAclnnInTensor(const atb::VariantPack &variantPack)
{
 aclInTensors_.resize(GetInputNum());
 for (size_t i = 0; i < aclInTensors_.size(); ++i)
 {
 auto aclnnTensor = CreateAclnnTensor(variantPack.inTensors.at(i), i);
 if (aclnnTensor->tensor == nullptr)
 {
 LOG_ERROR(opName_ + " InTensor aclCreateTensor index " + std::to_string(i) + " fail");
 return atb::ERROR_INTERNAL_ERROR;
 }
 aclInTensors_[i] = aclnnTensor;
 }
 return atb::NO_ERROR;
}
atb::Status GeluOperation::CreateAclnnOutTensor(const atb::VariantPack &variantPack)
{
 aclOutTensors_.resize(GetOutputNum());
 for (size_t i = 0; i < aclOutTensors_.size(); ++i)
 {
 auto aclnnTensor = CreateAclnnTensor(variantPack.outTensors.at(i), i);
 if (aclnnTensor->tensor == nullptr)
 {
 LOG_ERROR(opName_ + " outTensor aclCreateTensor index " + std::to_string(i) + " fail");
 return atb::ERROR_INTERNAL_ERROR;
 }
 LOG_INFO(opName_ + " input[" + std::to_string(i) + "] CreateAclnnTensor start");
 aclOutTensors_[i] = aclnnTensor;
 }
 return atb::NO_ERROR;
}
atb::SVector<int64_t> GetCopyTensorStride(atb::Dims &tensorDims)
{
 atb::SVector<int64_t> tmpStrides(tensorDims.dimNum, 1);
 if (tensorDims.dimNum > 8)
 { // 8: tensor最大维度数量
 LOG_ERROR("tensor's dimNum is larger than 8, GetCopyTensorStride failed.");
 return tmpStrides;
 }
 for (int64_t i = static_cast<int64_t>(tensorDims.dimNum) - 2; i >= 0; i--)
 {
 tmpStrides[i] = (tensorDims.dims[i + 1] * tmpStrides[i + 1]);
 }
 return tmpStrides;
}
std::shared_ptr<AclnnTensor> GeluOperation::CreateAclnnTensor(atb::Tensor atbTensor, size_t tensorIdx)
{
 auto aclnnTensor = std::make_shared<AclnnTensor>();
 aclnnTensor->tensorIdx = static_cast<int>(tensorIdx);
 aclnnTensor->needUpdateTensorDataPtr = true;
 aclnnTensor->atbTensor = atbTensor;
 aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
 // 创建Aclnn tensor
  aclnnTensor->tensor = aclCreateTensor(atbTensor.desc.shape.dims,
 atbTensor.desc.shape.dimNum,
 atbTensor.desc.dtype,
 aclnnTensor->strides.data(),
 0,
 atbTensor.desc.format,
 atbTensor.desc.shape.dims,
 atbTensor.desc.shape.dimNum,
 atbTensor.deviceData);
 return aclnnTensor;
}
// 重写父类方法, 创建workspace和aclexecutor
atb::Status GeluOperation::SetAclnnWorkspaceExecutor()
{
 LOG_INFO(opName_ + " SetAclnnWorkspaceExecutor start");
 if (param_.geluApproximate == -1)
 {
 auto ret = aclnnGeluGetWorkspaceSize(aclInTensors_.at(0)->tensor, // self
 aclOutTensors_.at(0)->tensor, // out
 &workspaceSize_,
 &aclExecutor_);
 CHECK_RET(ret, opName_ + " aclnnGeluGetWorkspaceSize failed, ret: " + std::to_string(ret));
 LOG_INFO(opName_ + " SetAclnnWorkspaceExecutor end, workspaceSize_: " + 
std::to_string(workspaceSize_));
 return ret;
 }
 auto ret = aclnnGeluV2GetWorkspaceSize(aclInTensors_.at(0)->tensor, // x
 param_.geluApproximate, // approximate
 aclOutTensors_.at(0)->tensor, // y
 &workspaceSize_,
 &aclExecutor_);
 CHECK_RET(ret, opName_ + " aclnnGeluV2GetWorkspaceSize failed, ret: " + std::to_string(ret));
 LOG_INFO(opName_ + " SetAclnnWorkspaceExecutor end, workspaceSize_: " + 
std::to_string(workspaceSize_));
 return ret;
}
// 重写父类方法, 执行aclnn算子
atb::Status GeluOperation::ExecuteAclnnOp(uint8_t *workspace, aclrtStream &stream)
{
 LOG_INFO(opName_ + " ExecuteAclnnOp start");
 if (param_.geluApproximate == -1)
 {
 auto ret = aclnnGelu(workspace, workspaceSize_, aclExecutor_, stream);
 CHECK_RET(ret, opName_ + " ExecuteAclnnOp failed, ret: " + std::to_string(ret));
 LOG_INFO(opName_ + " ExecuteAclnnOp end");
 return ret;
 }
 auto ret = aclnnGeluV2(workspace, workspaceSize_, aclExecutor_, stream);
 CHECK_RET(ret, opName_ + " aclnnGeluV2 failed, ret: " + std::to_string(ret));
 LOG_INFO(opName_ + " ExecuteAclnnOp end");
 return ret;
}