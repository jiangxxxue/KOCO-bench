/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PAGD_ATTENTION_OPERATION_H
#define PAGD_ATTENTION_OPERATION_H

#include "atb/operation/operation_base.h"
#include "atb/infer_op_params.h"

namespace atb {

class PagedAttentionOperation : public OperationBase {
public:
    explicit PagedAttentionOperation(const infer::PagedAttentionParam &param);
    ~PagedAttentionOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    struct AttentionFlags {
        bool useQuantOffset;
        bool useQuant;
        bool useBatchRunStatus;
        bool useMask;
        bool useQLens;
        bool useRazorOffset;
        bool useLogN;
        bool useQKVQuantOffline;
        bool useQKVQuantOnline;
    };

protected:
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;
    Status InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const override;
    Status SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const override;
    Status InferShapeDimCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status KVCacheDimCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status KVCacheDimCheck310P(const SVector<TensorDesc> &inTensorDescs) const;
    Status KVCacheDimCheck910B(const SVector<TensorDesc> &inTensorDescs) const;
    bool BlockDimCheck(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const;
    bool RazorDimCheck(const SVector<Tensor> &inTensors) const;
    bool LogNDimCheck(const SVector<Tensor> &inTensors) const;
    Status SetupDimCheck(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const;
    nlohmann::json GetParamJson() const override;
    uint32_t Bools2IntLogN(AttentionFlags inputB) const;
    uint32_t Bools2IntQKVQuant(AttentionFlags inputB) const;
    void InitOpIni();
    Status InferShapeDimCheckBNSD910B(const SVector<TensorDesc> &inTensorDescs) const;
    Status SetupDimCheckBNSD910B(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const;
    bool MlaBatchSizeCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status MaskFreeInferShapeCheck310P(const SVector<TensorDesc> &inTensorDescs) const;
    Status MaskFreeSetupCheck310P(const SVector<Tensor> &inTensor) const;

private:
    bool IsInMLAIncompatible() const;

private:
    infer::PagedAttentionParam param_;
};
} // namespace atb

#endif
