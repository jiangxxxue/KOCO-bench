/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "operation_wrapper.h"
#include <stdexcept>
#include <mki/utils/time/timer.h>
#include "atb/utils/log.h"
#include "resource/utils.h"
#include "resource/memory_manager.h"
#include "prof/prof_stats.h"

namespace TorchAtb {

using namespace atb;
using namespace atb::infer;

OperationWrapper &OperationWrapper::operator=(OperationWrapper &&other) noexcept
{
    if (this != &other) {
        operation_ = std::move(other.operation_);
    }
    return *this;
}

template <typename OpParam> void OperationWrapper::CreateOpUniquePtr(const OpParam &param)
{
    Operation *operation = nullptr;
    Status st = CreateOperation(param, &operation);
    if (st != NO_ERROR) {
        throw std::runtime_error("Failed to create operation");
    }
    operation_.reset(operation);
}

atb::Operation *OperationWrapper::ReleaseOperation()
{
    return operation_.release();
}

OperationWrapper::OperationWrapper(const LayerNormParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const ElewiseParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const LinearParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const SoftmaxParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const SelfAttentionParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const PagedAttentionParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const RopeParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const SplitParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const GatherParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const ActivationParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const RmsNormParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const AllGatherParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const AsStridedParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const CumsumParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const DynamicNTKParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const MultinomialParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const ConcatParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const SliceParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const TransposeParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const GatingParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const ReshapeAndCacheParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const FillParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const RazorFusionAttentionParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const AllReduceParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const BroadcastParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const ReduceScatterParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const ReduceScatterVParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const FaUpdateParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const LinearParallelParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const LinearSparseParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const RelayAttentionParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const TopkToppSamplingParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const AllToAllParam &param)
{
    CreateOpUniquePtr(param);
}

OperationWrapper::OperationWrapper(const GraphParam &param)
{
    CreateOpUniquePtr(param);
}

std::string OperationWrapper::GetName() const
{
    return operation_->GetName();
}

uint32_t OperationWrapper::GetInputNum() const
{
    return operation_->GetInputNum();
}

uint32_t OperationWrapper::GetOutputNum() const
{
    return operation_->GetOutputNum();
}

std::vector<torch::Tensor> OperationWrapper::Forward(std::vector<torch::Tensor> &inTensors)
{
    Mki::Timer runTimer;
    if (!operation_) {
        throw std::runtime_error("call Forward fail, operation is nullptr");
    }
    std::vector<torch::Tensor> outTensors;
    Setup(inTensors, outTensors);
    Execute();
    ProfStats::GetProfStats().SetRunTime(GetName(), runTimer.ElapsedMicroSecond());
    return outTensors;
}

atb::SVector<atb::TensorDesc> OperationWrapper::InferShape()
{
    if (!operation_) {
        throw std::runtime_error("call InferShape fail, operation is nullptr");
    }
    atb::SVector<atb::TensorDesc> inTensorDescs;
    inTensorDescs.resize(variantPack_.inTensors.size());
    for (size_t i = 0; i < inTensorDescs.size(); ++i) {
        inTensorDescs.at(i) = variantPack_.inTensors.at(i).desc;
    }
    atb::SVector<atb::TensorDesc> outTensorDescs;
    Status st = operation_->InferShape(inTensorDescs, outTensorDescs);
    if (st != NO_ERROR) {
        throw std::runtime_error("call operation_->InferShape fail");
    }
    return outTensorDescs;
}

void OperationWrapper::Setup(std::vector<torch::Tensor> &inTensors, std::vector<torch::Tensor> &outTensors)
{
    if (!operation_) {
        throw std::runtime_error("call Setup fail, operation is nullptr");
    }
    BuildInTensorVariantPack(inTensors);
    atb::SVector<atb::TensorDesc> outTensorDescs = InferShape();
    outTensors.resize(outTensorDescs.size());
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        outTensors.at(i) = Utils::CreateTorchTensorFromTensorDesc(outTensorDescs.at(i));
    }
    variantPack_.outTensors.resize(outTensors.size());
    for (size_t i = 0; i < outTensors.size(); ++i) {
        variantPack_.outTensors.at(i) = Utils::ConvertToAtbTensor(outTensors.at(i));
    }
    atb::Context *context = Utils::GetAtbContext();
    atb::Status st = operation_->Setup(variantPack_, workspaceSize_, context);
    if (st != NO_ERROR) {
        throw std::runtime_error("call operation_->Setup fail");
    }
}

void OperationWrapper::Execute()
{
    if (!operation_) {
        throw std::runtime_error("call Execute fail, operation is nullptr");
    }
    uint8_t *workspace = nullptr;
    ATB_LOG(INFO) << "workspaceSize_: " << workspaceSize_;
    if (workspaceSize_ > 0) {
        workspace = (uint8_t *)MemoryManager::GetMemoryManager().GetWorkspaceBuffer(workspaceSize_);
    }
    atb::Context *context = Utils::GetAtbContext();
    if (Utils::IsTaskQueueEnable()) {
        ATB_LOG(DEBUG) << "IsTaskQueueEnable";
        at_npu::native::OpCommand cmd;
        cmd.Name(operation_->GetName());
        cmd.SetCustomHandler([=]() { return operation_->Execute(variantPack_, workspace, workspaceSize_, context); });
        cmd.Run();
        return;
    }
    Status st = operation_->Execute(variantPack_, workspace, workspaceSize_, context);
    if (st != NO_ERROR) {
        throw std::runtime_error("call operation_->Execute fail");
    }
    int ret = aclrtSynchronizeStream(context->GetExecuteStream());
    if (ret != 0) {
        throw std::runtime_error("call aclrtSynchronizeStream fail");
    }
}

void OperationWrapper::BuildInTensorVariantPack(std::vector<torch::Tensor> &inTensors)
{
    variantPack_.inTensors.resize(inTensors.size());
    for (size_t i = 0; i < inTensors.size(); ++i) {
        variantPack_.inTensors.at(i) = Utils::ConvertToAtbTensor(inTensors.at(i));
    }
}
} // namespace TorchAtb
