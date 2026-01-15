/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_LASER_ATTENTION_GRAD_OPERATION_H
#define ATB_LASER_ATTENTION_GRAD_OPERATION_H
#include "atb/operation/operation_base.h"
#include "atb/train_op_params.h"

namespace atb {
class LaserAttentionGradOperation : public OperationBase {
public:
    explicit LaserAttentionGradOperation(const train::LaserAttentionGradParam &param);
    ~LaserAttentionGradOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    train::LaserAttentionGradParam GetParam() const;
    void SetParam(const train::LaserAttentionGradParam &param);

protected:
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    Status InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const override;
    Status SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const override;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;
    nlohmann::json GetParamJson() const override;
    SVector<bool> GetEmptyInTensorPermissions() const override;

private:
    struct TensorDims {
        int64_t batch = 0;
        int64_t qHeadNum = 0;
        int64_t seqSize = 0;
        int64_t qHeadDim = 0;
        int64_t kvHeadNum = 0;
        int64_t kvSize = 0;
        int64_t kHeadDim = 0;
        int64_t vHeadDim = 0;
    };
    Status InTensorDescsCheck(const SVector<TensorDesc> &inTensorDescs, TensorDims &dims) const;
    Status OutTensorCheck(const SVector<Tensor> &outTensors, const SVector<TensorDesc> &inTensorDescs) const;
    Status QueryTensorDescCheck(const TensorDesc &queryTensorDesc, TensorDims &dims) const;
    Status KeyTensorDescCheck(const TensorDesc &keyTensorDesc, TensorDims &dims) const;
    Status ValueTensorDescCheck(const TensorDesc &valueTensorDesc, TensorDims &dims) const;
    Status AttentionOutGradTensorDescCheck(const TensorDesc &attentionOutGradTensorDesc, const TensorDims &dims) const;
    Status PseShiftTensorDescCheck(const TensorDesc &pseShiftTensorDesc) const;
    Status DropMaskTensorDescCheck(const TensorDesc &dropMaskTensorDesc) const;
    Status PaddingMaskTensorDescCheck(const TensorDesc &paddingMaskTensorDesc) const;
    Status AttenMaskTensorDescCheck(const TensorDesc &attenMaskTensorDesc, const TensorDims &dims) const;
    Status SoftmaxMaxTensorDescCheck(const TensorDesc &softmaxMaxTensorDesc, const TensorDims &dims) const;
    Status SoftmaxSumTensorDescCheck(const TensorDesc &softmaxSumTensorDesc,
                                     const TensorDesc &softmaxMaxTensorDesc) const;
    Status SoftmaxInTensorDescCheck(const TensorDesc &softmaxInTensorDesc) const;
    Status AttentionInTensorDescCheck(const TensorDesc &attentionInDescTensor, const TensorDims &dims) const;
    Status PrefixTensorDescCheck(const TensorDesc &prefixTensorDesc) const;
    Status ActualSeqQLenTensorDescCheck(const TensorDesc &actualSeqQLenTensorDesc) const;
    Status ActualSeqKVLenTensorDescCheck(const TensorDesc &actualSeqKVLenTensorDesc) const;
    Status QueryGradTensorCheck(const Tensor &queryGradTensor, const TensorDesc &queryTensorDesc) const;
    Status KeyGradTensorCheck(const Tensor &keyGradTensor, const TensorDesc &keyTensorDesc) const;
    Status ValueGradTensorCheck(const Tensor &valueGradTensor, const TensorDesc &valueTensorDesc) const;
    Status DpseTensorCheck(const Tensor &dpseTensor) const;
    Status InTensorDescsCheckSBH(const SVector<TensorDesc> &inTensorDescs, TensorDims &dims) const;
    Status QueryTensorDescCheckSBH(const TensorDesc &queryTensorDesc, TensorDims &dims) const;
    Status KeyTensorDescCheckSBH(const TensorDesc &keyTensorDesc, TensorDims &dims) const;
    Status ValueTensorDescCheckSBH(const TensorDesc &valueTensorDesc, TensorDims &dims) const;
    Status AttentionOutGradTensorDescCheckSBH(const TensorDesc &attentionOutGradTensorDesc,
                                              const TensorDims &dims) const;
    Status AttenMaskTensorDescCheckSBH(const TensorDesc &attenMaskTensorDesc, const TensorDims &dims) const;
    Status SoftmaxMaxTensorDescCheckSBH(const TensorDesc &softmaxMaxTensorDesc, const TensorDims &dims) const;
    Status AttentionInTensorDescCheckSBH(const TensorDesc &attentionInDescTensor, const TensorDims &dims) const;

private:
    train::LaserAttentionGradParam param_;
};
} // namespace atb
#endif // ATB_LASER_ATTENTION_GRAD_OPERATION_H
