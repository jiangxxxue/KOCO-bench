/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_SELF_ATTENTION_FUSION_OPERATION_H
#define ATB_SELF_ATTENTION_FUSION_OPERATION_H

#include "atb/operation/operation_base.h"
#include "atb/infer_op_params.h"

namespace atb {

bool MlaParamCheck(const infer::SelfAttentionParam &opParam);
bool SWAParamCheck(const infer::SelfAttentionParam &opParam);
bool DeviceParamCheck(const infer::SelfAttentionParam &opParam);
bool PrefixEncoderParamCheck(const infer::SelfAttentionParam &opParam);
bool BNSDParamCheck(const infer::SelfAttentionParam &opParam);

class SelfAttentionOperation : public OperationBase {
public:
    explicit SelfAttentionOperation(const infer::SelfAttentionParam &param);
    ~SelfAttentionOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

protected:
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;
    Status InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const override;
    Status SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const override;
    nlohmann::json GetParamJson() const override;

private:
    Status BypassInferShapeImpl910B(const SVector<TensorDesc> &inTensorDescs,
                                    SVector<TensorDesc> &outTensorDescs) const;
    Status InferShapeImpl910B(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const;
    Status InferShapeDimCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapePADimNumCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapePADimCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapePADimCheckBNSD(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapePADimNumCheckBNSD(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapeHiddenSizeCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapeDimNumCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status HeadSizeDimCheck910B(const SVector<TensorDesc> &inTensorDescs) const;
    Status MaxHeadSizeCheck910B(const int64_t headSizeK, const int64_t headSizeV) const;
    Status HeadSizeDimCheck310P(const SVector<TensorDesc> &inTensorDescs) const;
    Status SWAMaskDimCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status PAMaskDimCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status PAMaskDimCheckNz(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapeMaskDimNumCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapePrefixDimNumCheck910B(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapePrefixDimCheck910B(const SVector<TensorDesc> &inTensorDescs) const;

    Status SetupOutTensorCheck(const SVector<TensorDesc> &inTensorDescs, const SVector<Tensor> &outTensors) const;

    Status InferShapeBypassDimCheckBNSD910B(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapeBypassDimCheckBNSD310P(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapeDimNumCheckBNSD(const SVector<TensorDesc> &inTensorDescs) const;

    Status InferQKVQuantDimCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferLogNCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status DtypeCheck(const SVector<TensorDesc> &inTensorDescs) const;
    uint32_t Bools2Int(bool hasScale, bool hasKV, bool hasMask, bool hasSlopes, bool hasBatch) const;
    void InitFaOpIni();
    void InitPaEncoderOpIni();
    void InitMlaFaOpIni();
    infer::SelfAttentionParam param_;
    bool isMla_ = false;
    bool hasMask_ = false;
    int64_t kcacheId_ = 0;
    int64_t maskId_ = 0;
    int64_t tokenOffsetId_ = 0;
    int32_t kvHeadNum_ = 0;
    bool hasSlopes_ = false;
};
} // namespace atb
#endif // ATB_SELF_ATTENTION_FUSION_OPERATION_H
