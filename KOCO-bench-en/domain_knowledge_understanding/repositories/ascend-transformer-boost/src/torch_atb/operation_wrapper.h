/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TORCH_ATB_OPERATION_WRAPPER_H
#define TORCH_ATB_OPERATION_WRAPPER_H
#include <vector>
#include <string>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#include <torch/torch.h>
#include <torch/extension.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#pragma GCC diagnostic pop
#include "atb/atb_infer.h"

namespace TorchAtb {
class OperationWrapper {
public:
    OperationWrapper(const OperationWrapper &other) = delete;
    OperationWrapper &operator=(const OperationWrapper &other) = delete;
    OperationWrapper(OperationWrapper &&other) noexcept : operation_(std::move(other.operation_)){};
    OperationWrapper &operator=(OperationWrapper &&other) noexcept;
    ~OperationWrapper() = default;
    explicit OperationWrapper(const atb::infer::LayerNormParam &param);
    explicit OperationWrapper(const atb::infer::ElewiseParam &param);
    explicit OperationWrapper(const atb::infer::LinearParam &param);
    explicit OperationWrapper(const atb::infer::SoftmaxParam &param);
    explicit OperationWrapper(const atb::infer::SelfAttentionParam &param);
    explicit OperationWrapper(const atb::infer::PagedAttentionParam &param);
    explicit OperationWrapper(const atb::infer::RopeParam &param);
    explicit OperationWrapper(const atb::infer::SplitParam &param);
    explicit OperationWrapper(const atb::infer::GatherParam &param);
    explicit OperationWrapper(const atb::infer::ActivationParam &param);
    explicit OperationWrapper(const atb::infer::RmsNormParam &param);
    explicit OperationWrapper(const atb::infer::AllGatherParam &param);
    explicit OperationWrapper(const atb::infer::AsStridedParam &param);
    explicit OperationWrapper(const atb::infer::CumsumParam &param);
    explicit OperationWrapper(const atb::infer::DynamicNTKParam &param);
    explicit OperationWrapper(const atb::infer::MultinomialParam &param);
    explicit OperationWrapper(const atb::infer::ConcatParam &param);
    explicit OperationWrapper(const atb::infer::SliceParam &param);
    explicit OperationWrapper(const atb::infer::TransposeParam &param);
    explicit OperationWrapper(const atb::infer::GatingParam &param);
    explicit OperationWrapper(const atb::infer::ReshapeAndCacheParam &param);
    explicit OperationWrapper(const atb::infer::FillParam &param);
    explicit OperationWrapper(const atb::infer::RazorFusionAttentionParam &param);
    explicit OperationWrapper(const atb::infer::AllReduceParam &param);
    explicit OperationWrapper(const atb::infer::BroadcastParam &param);
    explicit OperationWrapper(const atb::infer::ReduceScatterParam &param);
    explicit OperationWrapper(const atb::infer::ReduceScatterVParam &param);
    explicit OperationWrapper(const atb::infer::FaUpdateParam &param);
    explicit OperationWrapper(const atb::infer::LinearParallelParam &param);
    explicit OperationWrapper(const atb::infer::LinearSparseParam &param);
    explicit OperationWrapper(const atb::infer::RelayAttentionParam &param);
    explicit OperationWrapper(const atb::infer::TopkToppSamplingParam &param);
    explicit OperationWrapper(const atb::infer::AllToAllParam &param);
    explicit OperationWrapper(const atb::GraphParam &param);
    atb::Operation *ReleaseOperation();
    std::string GetName() const;
    uint32_t GetInputNum() const;
    uint32_t GetOutputNum() const;
    std::vector<torch::Tensor> Forward(std::vector<torch::Tensor> &inTensors);

private:
    template <typename OpParam> void CreateOpUniquePtr(const OpParam &param);
    atb::SVector<atb::TensorDesc> InferShape();
    void Setup(std::vector<torch::Tensor> &inTensors, std::vector<torch::Tensor> &outTensors);
    void Execute();
    void BuildInTensorVariantPack(std::vector<torch::Tensor> &inTensors);
    void BuildOutTensorVariantPack();

private:
    std::unique_ptr<atb::Operation> operation_;
    atb::VariantPack variantPack_;
    uint64_t workspaceSize_{0};
};
} // namespace TorchAtb
#endif // TORCH_ATB_OPERATION_WRAPPER_H