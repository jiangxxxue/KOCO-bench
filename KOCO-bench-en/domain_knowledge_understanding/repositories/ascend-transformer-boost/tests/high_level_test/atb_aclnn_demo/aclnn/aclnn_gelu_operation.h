#ifndef ACLNN_GELU_OPERATION_H
#define ACLNN_GELU_OPERATION_H
#include "aclnn/aclnn_operation_base.h"
struct AclnnGeluParam
{
 int64_t geluApproximate = -1; // gelu_v2计算的入参，指定高斯近似算法，0: "none", 1: "tanh" , -1: 不使用gelu_v2
};
class GeluOperation : public AclnnBaseOperation
{
public:
 GeluOperation(const std::string &name, AclnnGeluParam param);
 atb::Status InferShape(
 const atb::SVector<atb::TensorDesc> &inTensorDesc, atb::SVector<atb::TensorDesc> &outTensorDesc) 
const override;
 uint32_t GetInputNum() const override;
 uint32_t GetOutputNum() const override;
 atb::Status CreateAclnnVariantPack(const atb::VariantPack &variantPack) override;
 atb::Status SetAclnnWorkspaceExecutor() override;
 atb::Status ExecuteAclnnOp(uint8_t *workspace, aclrtStream &stream) override;
private:
 atb::Status CreateAclnnInTensor(const atb::VariantPack &variantPack);
 atb::Status CreateAclnnOutTensor(const atb::VariantPack &variantPack);
 std::shared_ptr<AclnnTensor> CreateAclnnTensor(atb::Tensor atbTensor, size_t tensorIdx);
 AclnnGeluParam param_;
};
#endif