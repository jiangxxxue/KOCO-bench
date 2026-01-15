/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/runner/mki_node_implement.h"
#include <securec.h>
#include <mki/utils/time/timer.h>
#include "atb/utils/log.h"
#include "atb/utils/config.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/statistic.h"
#include "atb/utils/store_util.h"
#include "atb/utils/singleton.h"
#include "atb/utils/probe.h"

namespace atb {
MkiNodeImplement::MkiNodeImplement(Mki::Operation *op, MkiInferShapePreFunc func)
    : operation_(op), mkiInferShapePreFunc_(func)
{
    if (operation_ == nullptr) {
        ATB_LOG(ERROR) << "Operation implement init incorrectly";
    } else {
        ATB_LOG(INFO) << operation_->GetName() << " MkiNodeImplement Init";
    }
}

MkiNodeImplement::~MkiNodeImplement() {}

std::string MkiNodeImplement::GetName() const
{
    return kernel_ == nullptr ? operation_->GetName() : kernel_->GetName();
}

void MkiNodeImplement::Reset()
{
    runInfo_.Reset();
    launchParam_.Reset();
    kernelCacheValid_ = false;
    tilingBufferFilled_ = false;
    kernel_.reset();
}

bool MkiNodeImplement::BuildLaunchParam(const SVector<Mki::Tensor *> &inTensors, SVector<ViewFunc> &inTensorViewFuncs,
                                        const Mki::OpDesc &opDesc, size_t outTensorNum)
{
    launchParam_.SetParam(opDesc.specificParam);
    launchParam_.GetInTensors().clear();
    launchParam_.GetOutTensors().clear();
    kernelCacheValid_ = false; // launchParam改变，kernel cache失效

    for (size_t i = 0; i < inTensors.size(); ++i) {
        Mki::Tensor *tensor = inTensors.at(i);
        if (i < inTensorViewFuncs.size() && inTensorViewFuncs.at(i)) {
            Mki::Tensor viewTensor = *tensor;
            viewTensor.desc.dims.clear();
            ATB_LOG(INFO) << GetLogPrefix() << " inTensorViewFuncs[" << i
                          << "], tensor->desc.dims:" << TensorUtil::AsdOpsDimsToString(tensor->desc.dims)
                          << ",  viewTensor.desc.dims:" << TensorUtil::AsdOpsDimsToString(viewTensor.desc.dims);
            inTensorViewFuncs.at(i)(tensor->desc.dims, viewTensor.desc.dims);
            if (tensor->desc.format != Mki::TENSOR_FORMAT_FRACTAL_NZ && viewTensor.Numel() != tensor->Numel()) {
                ATB_LOG(ERROR) << GetLogPrefix() << " mki node: inTensorViewFuncs[" << i
                               << "], viewTensor.Numel:" << viewTensor.Numel() << ", tensor.Numel:" << tensor->Numel();
                return false;
            }
            ATB_LOG(INFO) << GetLogPrefix() << " mki node: view inTensor[" << i
                          << "], old:" << TensorUtil::AsdOpsDimsToString(tensor->desc.dims)
                          << ", new:" << TensorUtil::AsdOpsDimsToString(viewTensor.desc.dims);
            launchParam_.AddInTensor(viewTensor);
        } else {
            launchParam_.AddInTensor(*tensor);
        }
    }
    for (size_t i = 0; i < outTensorNum; ++i) {
        Mki::Tensor tensor;
        launchParam_.AddOutTensor(tensor);
    }

    ATB_LOG(DEBUG) << GetLogPrefix() << " launchParam:\n" << launchParam_.ToString();
    return true;
}

bool MkiNodeImplement::OperationGetBestKernel()
{
    Mki::Kernel *kernel = operation_->GetBestKernel(launchParam_);
    if (kernel == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << " " << operation_->GetName()
                       << " get best kernel fail, kernel count:" << operation_->GetKernelList().size();
        return false;
    }
    kernel_.reset(kernel);
    ATB_LOG(DEBUG) << GetLogPrefix() << " best kernel:" << kernel_->GetName() << ", addr: " << kernel_;

    return true;
}

bool MkiNodeImplement::PlanKernelInferShape()
{
    ATB_LOG(INFO) << GetLogPrefix() << " " << operation_->GetName() << " infer shape start, launchParam:\n"
                  << launchParam_.ToString();
    if (mkiInferShapePreFunc_) {
        ATB_LOG(DEBUG) << GetLogPrefix() << " " << operation_->GetName()
                       << " call inferShapePreFunc, old launchParam:\n"
                       << launchParam_.ToString();
        mkiInferShapePreFunc_(launchParam_);
        ATB_LOG(DEBUG) << GetLogPrefix() << " " << operation_->GetName()
                       << " call inferShapePreFunc, new launchParam:\n"
                       << launchParam_.ToString();
    }

    Mki::Status st = operation_->InferShape(launchParam_);
    if (!st.Ok()) {
        ATB_LOG(ERROR) << GetLogPrefix() << " " << operation_->GetName()
                       << " mki node infer shape fail, error:" << st.Message();
        return false;
    }
    ATB_LOG(INFO) << GetLogPrefix() << " " << operation_->GetName() << " mki node infer shape success, launchParam:\n"
                  << launchParam_.ToString();

    return true;
}

bool MkiNodeImplement::UpdateBestKernel()
{
    if (kernelCacheValid_ && kernel_) {
        ATB_LOG(DEBUG) << GetLogPrefix() << " " << operation_->GetName()
                       << " has got best kernel: " << kernel_->GetName();
        return true;
    }
    return OperationGetBestKernel();
}

Mki::SVector<Mki::Tensor> &MkiNodeImplement::GetInTensors()
{
    return launchParam_.GetInTensors();
}

Mki::SVector<Mki::Tensor> &MkiNodeImplement::GetOutTensors()
{
    return launchParam_.GetOutTensors();
}

size_t MkiNodeImplement::GetTilingSize() const
{
    if (kernel_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << " kernel is null";
        return 0;
    }
    return kernel_->GetTilingSize(launchParam_);
}

int64_t MkiNodeImplement::GetWorkspaceSize() const
{
    if (kernel_ == nullptr) {
        ATB_LOG(WARN) << GetLogPrefix() << " kernel is null";
        return -1;
    }
    return kernel_->GetKernelInfo().GetTotalScratchSize();
}
Status MkiNodeImplement::InitKernelInfo(uint8_t *hostTilingBuffer, uint64_t tilingSize, bool isLaunchWithTiling)
{
    ATB_LOG(DEBUG) << GetLogPrefix() << " init kernel info";
    if (kernel_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << " kernel is null";
        return ERROR_INVALID_PARAM;
    }
    if (isLaunchWithTiling) {
        ATB_LOG(INFO) << GetLogPrefix() << " use tiling optimize";
        kernel_->SetLaunchWithTiling(true);
    } else {
        ATB_LOG(DEBUG) << GetLogPrefix() << " set tiling info, tilingSize: " << tilingSize;
        kernel_->SetLaunchWithTiling(false);
        kernel_->SetTilingHostAddr(hostTilingBuffer, tilingSize);
    }
    Mki::Status status = kernel_->Init(launchParam_);
    if (!status.Ok()) {
        ATB_LOG(ERROR) << GetLogPrefix() << " InitRunInfo failed!";
        return atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType(status.Code()))->second;
    }
    return NO_ERROR;
}

void MkiNodeImplement::SetWorkspaceDeviceAddr(uint8_t *deviceWorkspaceBuffer)
{
    runInfo_.SetScratchDeviceAddr(deviceWorkspaceBuffer);
}

void MkiNodeImplement::SetTilingDeviceAddr(uint8_t *deviceTilingBuffer)
{
    runInfo_.SetTilingDeviceAddr(deviceTilingBuffer);
}

Status MkiNodeImplement::BuildArgs()
{
    Mki::Status st = kernel_->BuildArgs(launchParam_, runInfo_, argsHostBuffer_);
    if (st.Ok()) {
        ATB_LOG(DEBUG) << GetLogPrefix() << " " << kernel_->GetName()
                       << " BuildArgs success, launchParam\n:" << launchParam_.ToString();
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << " " << kernel_->GetName()
                       << " BuildArgs fail, launchParam\n:" << launchParam_.ToString() << "\nst: " << st.ToString();
        return atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType(st.Code()))->second;
    }
    return NO_ERROR;
}

Status MkiNodeImplement::Run(aclrtStream stream)
{
    Mki::Status st;
    if (kernel_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << " kernel is null";
        return ERROR_INVALID_PARAM;
    }
    runInfo_.SetStream(stream);
    ATB_LOG(INFO) << GetLogPrefix() << " " << kernel_->GetName() << " run start, launchParam:\n"
                  << launchParam_.ToString() << " runInfo:\n"
                  << runInfo_.ToString();
    bool isDeviceAddr = true;
    if (argsDeviceBuffer_ != nullptr) {
        st = kernel_->RunWithArgs(argsDeviceBuffer_, stream, isDeviceAddr);
    } else {
        st = kernel_->Run(launchParam_, runInfo_);
    }
    if (st.Ok()) {
        ATB_LOG(INFO) << GetLogPrefix() << " " << kernel_->GetName()
                      << " run end, launchParam\n:" << launchParam_.ToString();
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << " " << kernel_->GetName()
                       << " run fail, launchParam\n:" << launchParam_.ToString() << "\nst: " << st.ToString();
        return atb::ATB_MKI_ERROR_HASH.find(Mki::ErrorType(st.Code()))->second;
    }
    return NO_ERROR;
}

bool MkiNodeImplement::GetCachedTiling(KernelCache &kernelCache, size_t kernelIndex, uint8_t *kernelHostTilingBuffer,
                                       uint64_t maxTilingSize, uint64_t &tilingSizeFetched, bool isLaunchWithTiling)
{
    tilingBufferFilled_ = false;
    Mki::Timer kernelCacheGetTilingTimer;
    Mki::Kernel *kernelCached = nullptr;
    TilingBufferPtr cachedTilingBuffer = kernelCache.GetTiling(kernelIndex, launchParam_, kernelCached);
    if (kernelCached != nullptr) {
        // 由于当前的kernel在设计上是带状态的，必须保证kernel状态与当前所需相同才能使用cache中的kernel
        bool cachedTilingLaunchStatus = kernelCached->GetKernelInfo().GetLaunchWithTiling();
        if (cachedTilingLaunchStatus != isLaunchWithTiling) {
            ATB_LOG(INFO) << "Cache miss because of status of tilingLaunch mismatch.";
            return false;
        }
        kernel_.reset(kernelCached);
        kernelCacheValid_ = true;
    }
    GetOpSetupStatistic().kernelCacheGetTilingTime += kernelCacheGetTilingTimer.ElapsedMicroSecond();
    if (cachedTilingBuffer == nullptr) {
        tilingSizeFetched = 0;
        return false;
    }
    tilingSizeFetched = cachedTilingBuffer->size();
    if (tilingSizeFetched > maxTilingSize) {
        ATB_LOG(ERROR) << GetLogPrefix() << " MkiNodeImplement do not have enough tiling buffer for cached tilnig";
        return false;
    }
    if (!isLaunchWithTiling || Probe::IsSaveTiling()) {
        int ret = memcpy_s(kernelHostTilingBuffer, maxTilingSize, cachedTilingBuffer->data(), tilingSizeFetched);
        if (ret != EOK) {
            ATB_LOG(ERROR) << GetLogPrefix() << " MkiNodeImplement memcpy_s cached tiling fail, error:" << ret;
            return false;
        }
    }
    tilingBufferFilled_ = true;
    return true;
}

void MkiNodeImplement::AddTiling(KernelCache &kernelCache, size_t kernelIndex, uint8_t *hostTilingBuffer,
                                 size_t tilingSize) const
{
    kernelCache.AddTiling(kernelIndex, hostTilingBuffer, tilingSize, launchParam_, kernel_.get());
    ATB_LOG(DEBUG) << GetLogPrefix() << " AddTiling end, runinfo\n:" << runInfo_.ToString();
}

void MkiNodeImplement::ResetLogPrefix(const std::string &prefix, size_t kernelId)
{
    std::stringstream ss;
    ss << prefix << "[" << kernelId << "]";
    logPrefix_ = ss.str();
}

bool MkiNodeImplement::GetTilingFilledFlag() const
{
    return tilingBufferFilled_;
}

void MkiNodeImplement::SetTilingFilledFlag(bool flag)
{
    tilingBufferFilled_ = flag;
}

std::string MkiNodeImplement::GetLogPrefix() const
{
    return logPrefix_;
}

void MkiNodeImplement::SaveLaunchParam(aclrtStream stream, const std::string &dirPath) const
{
    StoreUtil::SaveLaunchParam(stream, launchParam_, dirPath);
}

void *MkiNodeImplement::GetMsprofInfoKey() const
{
    if (kernel_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << " kernel is nullptr, get msprof key failed";
        return nullptr;
    }
    return *(reinterpret_cast<void **>(kernel_.get()));
}

void MkiNodeImplement::GetReportTensors(Mki::SVector<std::pair<bool, Mki::Tensor>> &allTensors) const
{
    for (std::size_t i = 0; i < launchParam_.GetInTensorCount(); i++) {
        allTensors.push_back(std::make_pair(true, launchParam_.GetInTensors().at(i)));
    }

    for (std::size_t i = 0; i < launchParam_.GetOutTensorCount(); i++) {
        allTensors.push_back(std::make_pair(false, launchParam_.GetOutTensors().at(i)));
    }
}

uint32_t MkiNodeImplement::GetOpType() const
{
    return kernel_->GetType();
}

uint32_t MkiNodeImplement::GetBlockDim() const
{
    const Mki::KernelInfo &kernelInfo = kernel_->GetKernelInfo();
    return kernelInfo.GetBlockDim();
}

void MkiNodeImplement::SetArgsDeviceBuffer(void *deviceBuffer)
{
    argsDeviceBuffer_ = deviceBuffer;
}

void MkiNodeImplement::SetArgsHostBuffer(void *hostBuffer)
{
    argsHostBuffer_ = hostBuffer;
}

void *MkiNodeImplement::GetArgsDeviceBuffer()
{
    return argsDeviceBuffer_;
}

void *MkiNodeImplement::GetArgsHostBuffer()
{
    return argsHostBuffer_;
}

uint64_t MkiNodeImplement::GetArgsSize()
{
    if (kernel_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << " kernel is null";
        return 0;
    }
    return kernel_->GetKernelInfo().GetArgsSize();
}

Status MkiNodeImplement::BuildLaunchParam(const SVector<Mki::Tensor *> &inTensors,
                                          const SVector<Mki::Tensor *> &outTensors, const Mki::OpDesc &opDesc)
{
    launchParam_.SetParam(opDesc.specificParam);
    launchParam_.GetInTensors().clear();
    launchParam_.GetOutTensors().clear();
    kernelCacheValid_ = false;

    for (size_t i = 0; i < inTensors.size(); ++i) {
        Mki::Tensor *tensor = inTensors.at(i);
        launchParam_.AddInTensor(*tensor);
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        Mki::Tensor *tensor = outTensors.at(i);
        launchParam_.AddOutTensor(*tensor);
    }
    return NO_ERROR;
}
} // namespace atb