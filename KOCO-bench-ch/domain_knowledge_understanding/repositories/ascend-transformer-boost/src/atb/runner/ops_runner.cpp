/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/runner/ops_runner.h"
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <utility>
#include <securec.h>
#include <unistd.h>
#include <acl/acl_rt.h>
#include <mki/utils/time/timer.h>
#include <mki/utils/file_system/file_system.h>
#include <asdops/params/params.h>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/stringify/stringify.h>
#include "atb/utils/param_compare.h"
#include "atb/utils/config.h"
#include "atb/utils/probe.h"
#include "atb/kernel_cache/kernel_cache.h"
#include "atb/utils/statistic.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/store_util.h"
#include "atb/utils/common_utils.h"
#include "atb/utils/mem_allocation_solver/mem_allocation_solver_creator.h"
#include "atb/utils/current_op_tiling.h"
#include "atb/utils/utils_internal.h"
#include "atb/utils/acl_kernel.h"
#include "atb/operation/operation_base.h"
#include "atb/utils/singleton.h"
#include "atb/utils/mstx_mem_register.h"

namespace atb {
const int ALIGN_INT = 512;
thread_local std::vector<KernelCache> g_globalKernelCaches(RUNNER_TYPE_MAX);
constexpr uint32_t K_TENSOR_INFO_BYTES = 44UL;
constexpr uint32_t K_TENSOR_INFO_BYTES_WITH_CAP = 56U;
constexpr uint64_t MAX_TILING_BUFFER_SIZE = 1048576000;
static const char *TENSOR_FILE_NAME_EXT = ".bin";
constexpr uint32_t IS_OVERFLOW = 1;
constexpr uint32_t NOT_OVERFLOW = 0;

enum OpType : int {
    OP_TYPE_AI_CORE = 0,
    OP_TYPE_AI_CPU,
    OP_TYPE_AIV,
    OP_TYPE_WRITE_BACK,
    OP_TYPE_MIX_AIC,
    OP_TYPE_MIX_AIV,
    OP_TYPE_FFTS_PLUS,
    OP_TYPE_DSA,
    OP_TYPE_DVPP,
    OP_TYPE_HCCL,
    OP_TYPE_INVALID
};

OpsRunner::OpsRunner(const std::string &name, RunnerType runnerType) : Runner(name), runnerType_(runnerType)
{
    if (GetSingleton<Config>().IsworkspaceMemAllocGlobal()) {
        memAllocationSolver_ = GetGlobalMemAllocationSolver();
    } else {
        memAllocationSolver_ = CreateMemAllocationSolver();
    }
}

OpsRunner::~OpsRunner() {}

void OpsRunner::InitTensorFromRunnerPack(const RunnerVariantPack &runnerVariantPack)
{
    TensorUtil::FastCopyTensors(opsTensorPack_.inTensors, kernelGraph_.inTensors);
    TensorUtil::FastCopyTensors(opsTensorPack_.outTensors, kernelGraph_.outTensors);
    const size_t kernelGraphInTensorsSize = kernelGraph_.inTensors.size();
    for (size_t i = 0; i < kernelGraphInTensorsSize; i++) {
        isInTensorCanFree_[&kernelGraph_.inTensors.at(i)] = runnerVariantPack.isInTensorCanFree.at(i);
    }
    const size_t kernelGraphOutTensorsSize = kernelGraph_.outTensors.size();
    for (size_t i = 0; i < kernelGraphOutTensorsSize; i++) {
        isOutTensorNeedMalloc_[&kernelGraph_.outTensors.at(i)] = runnerVariantPack.isOutTensorNeedMalloc.at(i);
    }
}

void OpsRunner::UpdateOutTensorDeviceData(RunnerVariantPack &runnerVariantPack)
{
    const size_t kernelGraphOutTensorsSize = kernelGraph_.outTensors.size();
    for (size_t i = 0; i < kernelGraphOutTensorsSize; i++) {
        Mki::Tensor *outTensor = &kernelGraph_.outTensors.at(i);
        auto it = isOutTensorNeedMalloc_.find(outTensor);
        if (it == isOutTensorNeedMalloc_.end()) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor[" << i << "] isOutTensorNeedMalloc not found";
            if (!runnerVariantPack.outTensors.at(i).deviceData) {
                runnerVariantPack.outTensors.at(i).deviceData = outTensor->data;
            }
        } else if (it->second) {
            runnerVariantPack.outTensors.at(i).deviceData = outTensor->data;
        }
    }
}

void OpsRunner::RunMallocCache(RunnerVariantPack &runnerVariantPack)
{
    InitTensorFromRunnerPack(runnerVariantPack);

    ATB_LOG(INFO) << GetLogPrefix() << "start run malloc cache";
    for (auto &it : mallocCache_) {
        Mki::Tensor *tensor = it.first;
        if (it.second) {
            tensor->data = memAllocationSolver_->GetOffset(TensorUtil::AlignInt(tensor->dataSize, ALIGN_INT));
            ATB_LOG(INFO) << GetLogPrefix() << "run malloc, dataSize: " << tensor->dataSize
                          << ", dataptr: " << tensor->data;
        } else {
            memAllocationSolver_->Free((uint8_t *)tensor->data);
            ATB_LOG(INFO) << GetLogPrefix() << "run free, dataSize: " << tensor->dataSize
                          << ", dataptr: " << tensor->data;
        }
    }
    ATB_LOG(INFO) << GetLogPrefix() << "run malloc cache success";

    UpdateOutTensorDeviceData(runnerVariantPack);
}

void OpsRunner::ReserveSvector(RunnerVariantPack &runnerPack)
{
    tilingSizes_.reserve(kernelGraph_.nodes.size());
    opsTensorPack_.inTensors.reserve(runnerPack.inTensors.size());
    opsTensorPack_.outTensors.reserve(runnerPack.outTensors.size());
}

void OpsRunner::SetupCacheGetCachedTiling(uint8_t *kernelHostTilingBuffer, uint64_t maxTilingSize,
                                          bool launchWithTiling)
{
    if (totalTilingSize_ > maxTilingSize) {
        ATB_LOG(ERROR) << GetLogPrefix() << " no enough buffer for tiling";
        return;
    }
    for (size_t nodeId = 0; nodeId < kernelGraph_.nodes.size(); ++nodeId) {
        uint64_t tilingSize = nodeId < tilingSizes_.size() ? tilingSizes_.at(nodeId) : 0;
        bool getTilingSuccess = GetCachedTiling(kernelGraph_.nodes.at(nodeId), nodeId, kernelHostTilingBuffer,
                                                tilingSize, tilingSize, launchWithTiling);
        if (getTilingSuccess) {
            ATB_LOG(DEBUG) << GetLogPrefix() << " get tiling success";
            kernelHostTilingBuffer += tilingSize;
        }
    }
}

bool OpsRunner::SetupCanReuse(RunnerVariantPack &runnerVariantPack, bool &kernelGraphTopoChanged)
{
    GetOpSetupStatistic().setupTotalCount++;
    kernelGraphTopoChanged = true;
    if (GetSingleton<Config>().IsOpsRunnerSetupCacheEnable()) {
        if (needKernelGraphModify_ && !skipSetUpKernelGraphWhenCacheHit_) {
            return false;
        }
        if (!isParamUpdated_ && setupCount_ != 0 &&
            TensorUtil::IsRunnerVariantPackInputEqual(runnerVariantPack, lastRunnerVariantPack_)) {
            ATB_LOG(INFO) << GetLogPrefix() << " runnerVariantPack input is not change, setup do nothing";
            kernelGraphTopoChanged = false;
            GetOpSetupStatistic().setupCacheHitCount++;
            if (GetSingleton<Config>().IsworkspaceMemAllocGlobal() && (!needKernelGraphModify_)) {
                RunMallocCache(runnerVariantPack);
            }
            if (!needKernelGraphModify_) {
                bool launchWithTiling = runnerVariantPack.context->GetLaunchWithTilingStatus();
                SetupCacheGetCachedTiling(runnerVariantPack.hostTilingBuffer, runnerVariantPack.tilingBufferSize,
                                          launchWithTiling);
                return true; // 组图不改，参数不改，直接返回
            }
        } else {
            ATB_LOG(INFO) << GetLogPrefix() << " runnerVariantPack input is change, setup do";
            lastRunnerVariantPack_ = runnerVariantPack;
            GetOpSetupStatistic().setupCacheMissCount++;
        }
    }
    return false;
}

Status OpsRunner::SetupImpl(RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << " setup start, runnerVariantPack:\n" << runnerVariantPack.ToString();
    if (GetSingleton<Config>().IsworkspaceMemAllocGlobal()) {
        memAllocationSolver_ = GetGlobalMemAllocationSolver();
    }
    InitOpsTensorPack(runnerVariantPack);
    ReserveSvector(runnerVariantPack);

    bool kernelGraphTopoChanged = true;
    if (SetupCanReuse(runnerVariantPack, kernelGraphTopoChanged)) {
        return ErrorType::NO_ERROR;
    }
    Reset();
    if (kernelGraphTopoChanged || !skipSetUpKernelGraphWhenCacheHit_) {
        InitTensorFromRunnerPack(runnerVariantPack);
        Status st = SetupKernelGraph(opsTensorPack_);
        if (st != NO_ERROR) {
            return st;
        }
    }
    Status st = needKernelGraphModify_ ? ModifyKernelGraph(opsTensorPack_) : NO_ERROR;
    if (st != NO_ERROR) {
        return st;
    }
    InitKernelGraph();
    InitKernelCache();
    InitTensorMaxNodeMap();
    ATB_LOG(INFO) << GetLogPrefix() << " Setup start, kernel graph:\n" << kernelGraph_.ToString();
    ATB_LOG(DEBUG) << GetLogPrefix() << " runnerVariantPack inTensor size:" << runnerVariantPack.inTensors.size()
                   << ", outTensor size:" << runnerVariantPack.outTensors.size();
    bool launchWithTiling = runnerVariantPack.context->GetLaunchWithTilingStatus();
    st = PlanKernelGraph(runnerVariantPack.hostTilingBuffer, runnerVariantPack.tilingBufferSize, launchWithTiling);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "PlanKernelGraph fail";
        return st;
    }

    if (GetSingleton<Config>().IsworkspaceMemAllocGlobal()) {
        UpdateOutTensorDeviceData(runnerVariantPack);
    }
    isParamUpdated_ = false;
    return ErrorType::NO_ERROR;
}

void OpsRunner::InitKernelGraph()
{
    if (initKernelGraphFlag_ && !isParamUpdated_) {
        return;
    }
    kernelGraph_.Init();
    SetNodesSaveTensorFlag();
    initKernelGraphFlag_ = true;
}

uint64_t OpsRunner::GetTilingBufferSizeImpl()
{
    return totalTilingSize_;
}

Status OpsRunner::FillHostTilingBufferImpl(uint8_t *hostTilingBuffer, uint64_t tilingBufferSize, ContextBase *context)
{
    if (tilingBufferSize < totalTilingSize_) {
        ATB_LOG(FATAL) << GetLogPrefix() << " FillHostTilingBufferImpl fail, tilingBufferSize:" << tilingBufferSize
                       << ", totalTilingSize:" << totalTilingSize_;
        return ERROR_OUT_OF_HOST_MEMORY;
    }

    ATB_LOG(DEBUG) << GetLogPrefix() << " FillHostTilingBufferImpl start,  tilingBufferSize:" << tilingBufferSize
                   << ", totalTilingSize:" << totalTilingSize_;

    uint64_t offset = 0;
    for (size_t nodeId = 0; nodeId < kernelGraph_.nodes.size(); ++nodeId) {
        KernelGraphNode &node = kernelGraph_.nodes.at(nodeId);

        uint64_t tilingSize = nodeId < tilingSizes_.size() ? tilingSizes_.at(nodeId) : 0;
        if (tilingSize == 0) {
            continue;
        }

        uint8_t *kernelHostTilingBuffer = hostTilingBuffer + offset;
        Status ret = FillSingleKernelHostTilingBuffer(node, nodeId, kernelHostTilingBuffer, tilingSize, *context);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << " node[" << nodeId << "] fill tiling buffer fail, error code:" << ret;
            return ret;
        }
        if (nodesSaveTensorFlag_.at(nodeId) && Probe::IsExecuteCountInRange(executeCount_) && Probe::IsSaveTiling()) {
            std::string fileDir = GetSaveTensorDir() + "/" + std::to_string(nodeId) + "_" + node.GetName();
            Probe::SaveTiling(kernelHostTilingBuffer + offset, tilingSize, fileDir + "/kernel_tilingdata.bin");
        }
        offset += tilingSize;
    }
    return NO_ERROR;
}

Status OpsRunner::FillSingleKernelHostTilingBuffer(KernelGraphNode &node, size_t nodeId,
                                                   uint8_t *kernelHostTilingBuffer, size_t tilingSize,
                                                   ContextBase &context)
{
    bool ifGraphLaunchNeedCalcTiling = needKernelGraphModify_ && (context.GetLaunchMode() == GRAPH_LAUNCH_MODE);
    if (node.impl->GetTilingFilledFlag() && !ifGraphLaunchNeedCalcTiling) {
        return NO_ERROR;
    }

    ATB_LOG(DEBUG) << GetLogPrefix() << " node[" << nodeId << "] InitHostLaunchBuffer start";
    GetOpSetupStatistic().tilingCacheMissCount += 1;
    Mki::Timer fillTimer;
    bool launchWithTiling = context.GetLaunchWithTilingStatus();
    Status status = node.impl->InitKernelInfo(kernelHostTilingBuffer, tilingSize, launchWithTiling);
    if (status != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << " node[" << nodeId << "] InitRunInfo failed!";
        return status;
    }
    uint64_t fillTime = fillTimer.ElapsedMicroSecond();
    GetOpSetupStatistic().opsInitLuanchBufferTime += fillTime;
    ATB_LOG(DEBUG) << GetLogPrefix() << " node[" << nodeId << "] InitHostLaunchBuffer end, time:" << fillTime;

    UpdateCacheTiling(node, nodeId, kernelHostTilingBuffer, tilingSize);
    if (context.GetLaunchMode() == GRAPH_LAUNCH_MODE) {
        // 整图下发模式下绝大部分算子tiling只需计算一次，少部分需要多次计算的用needKernelGraphModify_进行标记
        node.impl->SetTilingFilledFlag(true);
    }
    return NO_ERROR;
}

void OpsRunner::CalcKernelWorkspace()
{
    uint64_t maxKernelWorkspaceSize = 0;
    for (size_t nodeId = 0; nodeId < kernelGraph_.nodes.size(); ++nodeId) {
        KernelGraphNode &node = kernelGraph_.nodes.at(nodeId);
        if (!node.impl) {
            ATB_LOG(ERROR) << GetLogPrefix() << " node[" << nodeId << "] implement is empty";
            return;
        }

        uint64_t tilingSize = nodeId < tilingSizes_.size() ? tilingSizes_.at(nodeId) : 0;
        if (tilingSize == 0) {
            continue;
        }

        int64_t originWorkspaceSize = node.impl->GetWorkspaceSize();
        if (originWorkspaceSize < 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << " node[" << nodeId << "] get workspace size failed";
            continue;
        }
        ATB_LOG(DEBUG) << GetLogPrefix() << " node[" << nodeId << "] " << node.GetName()
                       << " workspaceSize:" << originWorkspaceSize;

        uint64_t kernelWorkspaceSize =
            static_cast<uint64_t>(TensorUtil::AlignInt(originWorkspaceSize, 32)); // kernelWorkspace按照32长度对齐
        ATB_LOG(DEBUG) << GetLogPrefix() << " node[" << nodeId << "] " << node.GetName()
                       << " kernelWorkspaceSize:" << kernelWorkspaceSize
                       << ", maxKernelWorkspaceSize:" << maxKernelWorkspaceSize;

        maxKernelWorkspaceSize = std::max(maxKernelWorkspaceSize, kernelWorkspaceSize);
    }
    workspaceSize_ = maxKernelWorkspaceSize;
}

uint64_t OpsRunner::GetWorkspaceBufferSizeImpl()
{
    CalcKernelWorkspace();
    return workspaceSize_;
}

uint64_t OpsRunner::GetIntermediateBufferSizeImpl()
{
    return intermediateSize_;
}

Status OpsRunner::UpdateDeviceRealAddr(const RunnerVariantPack &runnerVariantPack)
{
    uint8_t *deviceIntermediateBuffer = runnerVariantPack.intermediateBuffer;
    bool isLaunchKernelWithTiling = runnerVariantPack.context->GetLaunchWithTilingStatus();
    bool needSetTiling = !(isLaunchKernelWithTiling || (totalTilingSize_ == 0));
    bool needSetworkspace = (workspaceSize_ != 0);
    uint64_t tilingOffset = 0;
    uint64_t deviceArgsSizeOffset = 0;
    uint64_t hostArgsSizeOffset = 0;

    for (size_t nodeId = 0; nodeId < kernelGraph_.nodes.size(); ++nodeId) {
        KernelGraphNode &node = kernelGraph_.nodes.at(nodeId);
        UpdateRunInfoTensorData(node, nodeId, deviceIntermediateBuffer);
        if (needSetTiling) {
            ATB_LOG(DEBUG) << GetLogPrefix() << " node[" << nodeId << "] update kernel runinfo launch buffer";
            uint64_t tilingBufferSize = tilingSizes_.at(nodeId);
            node.impl->SetTilingDeviceAddr(runnerVariantPack.tilingBuffer + tilingOffset);
            tilingOffset += tilingBufferSize;
        }
        if (needSetworkspace) {
            ATB_LOG(DEBUG) << GetLogPrefix() << " node[" << nodeId << "] update kernel runinfo workspace";
            node.impl->SetWorkspaceDeviceAddr(runnerVariantPack.workspaceBuffer);
        }
        if (runnerVariantPack.argsDeviceBuffer != nullptr) {
            node.impl->SetArgsDeviceBuffer(runnerVariantPack.argsDeviceBuffer + deviceArgsSizeOffset);
            ATB_LOG(DEBUG) << GetLogPrefix() << "argsDeviceBuffer from workSpace is "
                           << reinterpret_cast<void *>(runnerVariantPack.argsDeviceBuffer + deviceArgsSizeOffset);
            deviceArgsSizeOffset += node.impl->GetArgsSize();
        }
        if (runnerVariantPack.argsHostBuffer != nullptr) {
            node.impl->SetArgsHostBuffer(runnerVariantPack.argsHostBuffer + hostArgsSizeOffset);
            ATB_LOG(DEBUG) << GetLogPrefix() << "argsHostBuffer from workSpace is "
                           << reinterpret_cast<void *>(runnerVariantPack.argsHostBuffer + hostArgsSizeOffset);
            hostArgsSizeOffset += node.impl->GetArgsSize();
        }
    }
    return ErrorType::NO_ERROR;
}

Status OpsRunner::PreExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    InitOpsTensorPack(runnerVariantPack);

    ATB_LOG(INFO) << GetLogPrefix() << " execute start, totalTilingSize:" << totalTilingSize_
                  << ", workspaceSize:" << workspaceSize_ << ", intermediateSize:" << intermediateSize_;
    TensorUtil::FastCopyTensors(opsTensorPack_.inTensors, kernelGraph_.inTensors);
    TensorUtil::FastCopyTensors(opsTensorPack_.outTensors, kernelGraph_.outTensors);

    Status st = UpdateDeviceRealAddr(runnerVariantPack);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "UpdateDeviceRealAddr failed! ret:" << st;
        return st;
    }
    return ErrorType::NO_ERROR;
}

Status OpsRunner::ExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    Status st = RunAllKernel(runnerVariantPack);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "RunAllKernel failed! ret:" << st;
        return st;
    }

    ATB_LOG(INFO) << GetLogPrefix() << " execute end";
    return ErrorType::NO_ERROR;
}

static std::string GetUpdateRunInfoTensorDataString(size_t nodeId, std::string tensorFlag, uint64_t tensorId,
                                                    std::string tensorType, const Mki::Tensor &tensor)
{
    std::stringstream ss;
    ss << " update node[" << nodeId << "] " << tensorFlag << "[" << tensorId << "] is " << tensorType << ".";
    ss << " dataSize:" << tensor.dataSize;
#ifdef _DEBUG
    ss << " Data:" << tensor.data;
#endif
    return ss.str();
}

void OpsRunner::UpdateRunInfoTensorData(KernelGraphNode &node, size_t nodeId, uint8_t *deviceIntermediateBuffer) const
{
    auto &inTensors = node.impl->GetInTensors();
    const uint64_t inTensorsSize = inTensors.size();
    for (uint64_t tensorId = 0; tensorId < inTensorsSize; tensorId++) {
        Mki::Tensor &tensor = inTensors.at(tensorId);
        if (node.inTensorsType.at(tensorId) == TensorType::INTERMEDIATE_TENSOR) {
            tensor.data = deviceIntermediateBuffer + reinterpret_cast<uint64_t>(node.inTensors.at(tensorId)->data);
            ATB_LOG(INFO) << GetLogPrefix()
                          << GetUpdateRunInfoTensorDataString(nodeId, "inTensors", tensorId, "intermeidate", tensor);
        } else {
            tensor.data = node.inTensors.at(tensorId)->data;
            ATB_LOG(DEBUG) << GetLogPrefix()
                           << GetUpdateRunInfoTensorDataString(nodeId, "inTensors", tensorId, "not intermeidate",
                                                               tensor);
        }
    }
    auto &outTensors = node.impl->GetOutTensors();
    const uint64_t outTensorsSize = outTensors.size();
    for (uint64_t tensorId = 0; tensorId < outTensorsSize; tensorId++) {
        Mki::Tensor &tensor = outTensors.at(tensorId);
        if (node.outTensorsType.at(tensorId) == TensorType::INTERMEDIATE_TENSOR) {
            tensor.data = deviceIntermediateBuffer + reinterpret_cast<uint64_t>(node.outTensors.at(tensorId)->data);
            ATB_LOG(INFO) << GetLogPrefix()
                          << GetUpdateRunInfoTensorDataString(nodeId, "outTensors", tensorId, "intermeidate", tensor);
        } else {
            tensor.data = node.outTensors.at(tensorId)->data;
            ATB_LOG(DEBUG) << GetLogPrefix()
                           << GetUpdateRunInfoTensorDataString(nodeId, "outTensors", tensorId, "not intermeidate",
                                                               tensor);
        }
    }
}

void OpsRunner::ReportLaunchInfo(const uint64_t beginTime, const char *opName, void const *key) const
{
    MsProfApi info = {};
    info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
    const size_t nameLen = strlen(opName);

    info.itemId = GetSingleton<Mki::ProfilingFuncs>().ProfGetHashId(opName, nameLen, key);

    info.level = MSPROF_REPORT_NODE_LEVEL;
    const uint64_t endTime = GetSingleton<Mki::ProfilingFuncs>().ProfSysCycleTime();
    long tid = UtilsInternal::GetCurrentThreadId();
    if (tid == -1) {
        return;
    }
    info.threadId = static_cast<uint32_t>(tid);
    info.beginTime = beginTime;
    info.endTime = endTime;
    info.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    info.reserve = 0;
    auto ret = GetSingleton<Mki::ProfilingFuncs>().ProfReportApi(true, &info);
    if (ret != 0) {
        ATB_LOG(ERROR) << "ProfReportApi error!";
    }
}

void OpsRunner::ReportAdditionalInfo(const uint64_t timeStamp, const char *opName, size_t nodeId, void const *key) const
{
    const size_t nameLen = strlen(opName);
    const uint64_t nameHash = GetSingleton<Mki::ProfilingFuncs>().ProfGetHashId(opName, nameLen, key);

    auto &node = kernelGraph_.nodes.at(nodeId);
    uint32_t taskType = node.impl->GetOpType();

    MsprofCompactInfo nodeBasicInfo = {};
    auto &profNodeBasicInfo = nodeBasicInfo.data.nodeBasicInfo;
    profNodeBasicInfo.opName = nameHash;
    profNodeBasicInfo.opType = nameHash;
    profNodeBasicInfo.taskType = taskType;
    uint32_t blockDim = node.impl->GetBlockDim();
    if (taskType == OpType::OP_TYPE_MIX_AIC) {
        constexpr uint32_t constTwo = 2U;
        blockDim = ((blockDim & 0x0000FFFU) | (constTwo << 16U));
    }
    profNodeBasicInfo.blockDim = blockDim;
    nodeBasicInfo.level = static_cast<uint16_t>(MSPROF_REPORT_NODE_LEVEL);
    nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
    nodeBasicInfo.timeStamp = timeStamp;
    long tid = UtilsInternal::GetCurrentThreadId();
    if (tid == -1) {
        return;
    }

    nodeBasicInfo.threadId = static_cast<uint32_t>(tid);
    auto ret = GetSingleton<Mki::ProfilingFuncs>().ProfReportCompactInfo(
        static_cast<uint32_t>(true), static_cast<void *>(&nodeBasicInfo),
        static_cast<uint32_t>(sizeof(MsprofCompactInfo)));
    if (ret != 0) {
        ATB_LOG(ERROR) << "ProfReportCompactInfo error!";
    }
}

void OpsRunner::ReportTensorInfo(const uint64_t timeStamp, const char *opName, const KernelGraphNode &node,
                                 void const *key) const
{
    // 所有tensor一起分批上报
    const size_t nameLen = strlen(opName);
    const uint64_t nameHash = GetSingleton<Mki::ProfilingFuncs>().ProfGetHashId(opName, nameLen, key);

    std::shared_ptr<Mki::SVector<std::pair<bool, Mki::Tensor>>> allTensors =
        std::make_shared<Mki::SVector<std::pair<bool, Mki::Tensor>>>();
    node.impl->GetReportTensors(*allTensors);
    Mki::SVector<std::pair<bool, Mki::Tensor>, ATB_MSPROF_TENSOR_DATA_SIZE> batchTensors;
    for (auto &tensor : *allTensors) {
        batchTensors.push_back(tensor);
        if (batchTensors.size() == ATB_MSPROF_TENSOR_DATA_SIZE) {
            MsprofAdditionalInfo additionInfo = {};
            additionInfo.timeStamp = timeStamp;
            BuildAdditionalInfo(nameHash, batchTensors, additionInfo);
            DoReportTensorInfo(additionInfo);
            batchTensors.clear();
        }
    }

    if (!batchTensors.empty()) {
        MsprofAdditionalInfo remainTensorInfo = {};
        remainTensorInfo.timeStamp = timeStamp;
        BuildAdditionalInfo(nameHash, batchTensors, remainTensorInfo);
        DoReportTensorInfo(remainTensorInfo);
    }
}

void OpsRunner::ReportContextInfo(const uint64_t timeStamp, const char *opName, void const *key) const
{
    // 上报contextInfo信息
    MsprofAdditionalInfo additionalInfo = {};
    additionalInfo.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    additionalInfo.level = MSPROF_REPORT_NODE_LEVEL;
    additionalInfo.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
    additionalInfo.timeStamp = timeStamp;
    long tid = UtilsInternal::GetCurrentThreadId();
    if (tid == -1) {
        return;
    }
    additionalInfo.threadId = static_cast<uint32_t>(tid);
    additionalInfo.dataLen = sizeof(MsprofContextIdInfo);

    const size_t nameLen = strlen(opName);
    const uint64_t nameHash = GetSingleton<Mki::ProfilingFuncs>().ProfGetHashId(opName, nameLen, key);

    MsprofContextIdInfo contextInfo = {};
    contextInfo.opName = nameHash;
    contextInfo.ctxIdNum = 1;
    contextInfo.ctxIds[0] = 0;

    int ret =
        memcpy_s(additionalInfo.data, MSPROF_ADDTIONAL_INFO_DATA_LENGTH, &contextInfo, sizeof(MsprofContextIdInfo));
    ATB_LOG_IF(ret != EOK, ERROR) << "memcpy_s Error! Error Code: " << ret;

    auto retReport = GetSingleton<Mki::ProfilingFuncs>().ProfReportAdditionalInfo(
        static_cast<uint32_t>(true), static_cast<void *>(&additionalInfo),
        static_cast<uint32_t>(sizeof(MsprofAdditionalInfo)));
    if (retReport != 0) {
        ATB_LOG(ERROR) << "ProfReportAdditionalInfo error!";
    }
}

void OpsRunner::ReportMsprofInfo(const uint64_t timeStamp, const char *opName, const KernelGraphNode &node,
                                 size_t nodeId) const
{
    if (GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel0Status()) {
        void const *key = node.impl->GetMsprofInfoKey();
        ReportLaunchInfo(timeStamp, opName, key);
        ReportAdditionalInfo(timeStamp + 1, opName, nodeId, key);
        uint32_t taskType = node.impl->GetOpType();
        if (taskType == OpType::OP_TYPE_MIX_AIC || taskType == OpType::OP_TYPE_MIX_AIV) {
            ReportContextInfo(timeStamp + 1, opName, key);
        }
        if (GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel1Status()) {
            ReportTensorInfo(timeStamp + 1, opName, node, key);
        }
    }
}

Status OpsRunner::RunAllKernel(RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(DEBUG) << GetLogPrefix() << " start run all kernel, kernel count:" << kernelGraph_.nodes.size();
    aclrtStream stream = GetExecuteStream(runnerVariantPack.context);
    for (size_t nodeId = 0; nodeId < kernelGraph_.nodes.size(); ++nodeId) {
        KernelGraphNode &node = kernelGraph_.nodes.at(nodeId);
        if (runnerVariantPack.mstxMemRegister != nullptr) {
            runnerVariantPack.mstxMemRegister->ClearMstxMemRegions();
            if (runnerVariantPack.workspaceBufferSize) {
                runnerVariantPack.workspaceBufferSize = static_cast<uint64_t>(TensorUtil::AlignInt(runnerVariantPack.workspaceBufferSize, ALIGN_INT));
                runnerVariantPack.mstxMemRegister->AddTensorMemRegions(runnerVariantPack.workspaceBuffer, runnerVariantPack.workspaceBufferSize);
            }
            auto &inTensors = node.impl->GetInTensors();
            const uint64_t inTensorsSize = inTensors.size();
            for (uint64_t tensorId = 0; tensorId < inTensorsSize; tensorId++) {
                Mki::Tensor &tensor = inTensors.at(tensorId);
                if (node.inTensorsType.at(tensorId) == TensorType::INTERMEDIATE_TENSOR) {
                    tensor.data = runnerVariantPack.intermediateBuffer + reinterpret_cast<uint64_t>(node.inTensors.at(tensorId)->data);
                    runnerVariantPack.mstxMemRegister->AddTensorMemRegions(tensor.data, static_cast<uint64_t>(TensorUtil::AlignInt(tensor.dataSize, ALIGN_INT)));
                }
            }
            auto &outTensors = node.impl->GetOutTensors();
            const uint64_t outTensorsSize = outTensors.size();
            for (uint64_t tensorId = 0; tensorId < outTensorsSize; tensorId++) {
                Mki::Tensor &tensor = outTensors.at(tensorId);
                if (node.outTensorsType.at(tensorId) == TensorType::INTERMEDIATE_TENSOR) {
                    tensor.data = runnerVariantPack.intermediateBuffer + reinterpret_cast<uint64_t>(node.outTensors.at(tensorId)->data);
                    runnerVariantPack.mstxMemRegister->AddTensorMemRegions(tensor.data, static_cast<uint64_t>(TensorUtil::AlignInt(tensor.dataSize, ALIGN_INT)));
                }
            }
            if (runnerVariantPack.mstxMemRegister->CheckTensorRange()) {
                runnerVariantPack.mstxMemRegister->MstxMemRegionsRegister();
            }
        }
        RunKernelPreProcess(node, nodeId, stream);
        Status st = RunKernel(node, nodeId, runnerVariantPack.context);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << "RunKernel failed! ret:" << st;
            return st;
        } else {
            ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] " << node.GetName() << " run end";
        }
        RunKernelPostProcess(node, nodeId, stream);
        if (runnerVariantPack.mstxMemRegister != nullptr && runnerVariantPack.mstxMemRegister->CheckTensorRange()) {
            runnerVariantPack.mstxMemRegister->MstxMemRegionsUnregister();
        }
    }
    ATB_LOG(INFO) << GetLogPrefix() << " finish run all kernel";
    return NO_ERROR;
}

Status OpsRunner::RunKernel(KernelGraphNode &node, size_t nodeId, ContextBase *context) const
{
    const uint64_t beginTime = GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel0Status() ?
                                   GetSingleton<Mki::ProfilingFuncs>().ProfSysCycleTime() :
                                   0;
    Mki::Timer kernelExecuteTimer;
    aclrtStream stream = GetExecuteStream(context);
    Status st = node.impl->Run(stream);
    GetOpExecuteStatistic().kernelExecuteTime += kernelExecuteTimer.ElapsedMicroSecond();
    if (st != NO_ERROR) {
        return st;
    }

    ReportMsprofInfo(beginTime, node.GetName().c_str(), node, nodeId);

    if (Probe::IsOverflowCheck()) {
        bool isOverflow = CheckOverflow(node, context);
        if (isOverflow) {
            ATB_LOG(ERROR) << node.GetName() << " overflow!";
            if (Probe::IsOverflowStop()) {
                ATB_LOG(WARN) << "stop because of overflow!";
                return ERROR_RT_FAIL; // stop process
            }
        }
    }
    if (Probe::ReportKernelIOTensorEnable()) {
        DumpKernelIOTensorInfo(node);
    }
    return NO_ERROR;
}

void OpsRunner::Reset()
{
    totalTilingSize_ = 0;
    tilingSizes_.clear();
    workspaceSize_ = 0;
    intermediateSize_ = 0;
    if (!GetSingleton<Config>().IsworkspaceMemAllocGlobal()) {
        memAllocationSolver_->Reset();
    }
    mallocCache_.clear();
    tensorMalloced_.clear();
}

Status OpsRunner::PlanKernelGraph(uint8_t *kernelHostTilingBuffer, uint64_t maxTilingSize, bool launchWithTiling)
{
    ATB_LOG(DEBUG) << GetLogPrefix() << " plan kernel graph start";
    const size_t kernelGraphNodesSize = kernelGraph_.nodes.size();
    tilingSizes_.resize(kernelGraphNodesSize);

    for (size_t nodeId = 0; nodeId < kernelGraphNodesSize; ++nodeId) {
        if (maxTilingSize < totalTilingSize_) {
            ATB_LOG(ERROR) << GetLogPrefix() << " node[" << nodeId << "] tiling buffer is not enough";
            return ERROR_OUT_OF_HOST_MEMORY;
        }
        uint64_t restTilingBufferSize = maxTilingSize - totalTilingSize_;
        Status st = PlanKernel(nodeId, kernelHostTilingBuffer, restTilingBufferSize, launchWithTiling);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << " node[" << nodeId << "] plan kernel graph fail";
            return st;
        }
        kernelHostTilingBuffer += tilingSizes_.at(nodeId);
    }

    intermediateSize_ = memAllocationSolver_->GetSize();
    ATB_LOG(DEBUG) << GetLogPrefix() << " MemAllocationSolver size:" << memAllocationSolver_->GetMallocSize()
                   << ", real size:" << intermediateSize_;

    return NO_ERROR;
}

bool OpsRunner::UpdateNodeBestKernelAndTiling(KernelGraphNode &node, size_t nodeId, uint8_t *kernelHostTilingBuffer,
                                              uint64_t maxTilingSize, bool launchWithTiling)
{
    uint64_t tilingSize = 0;
    bool getTilingSuccess = GetCachedTiling(node, nodeId, kernelHostTilingBuffer,
                                            maxTilingSize, tilingSize, launchWithTiling);
    if (!node.impl->UpdateBestKernel()) {
        ATB_LOG(ERROR) << GetLogPrefix() << " node[" << nodeId << "] " << node.GetName()
                       << " update best kernel failed";
        return false;
    }

    tilingSize = getTilingSuccess ? tilingSize : GetNodeAlignedTilingBufferSize(node, nodeId);
    tilingSizes_.at(nodeId) = tilingSize;
    totalTilingSize_ += tilingSize;
    return true;
}

Status OpsRunner::PlanKernel(size_t nodeId, uint8_t *kernelHostTilingBuffer, uint64_t maxTilingSize,
                             bool launchWithTiling)
{
    KernelGraphNode &node = kernelGraph_.nodes.at(nodeId);

    if (!node.impl || isParamUpdated_) {
        if (!node.CreateImplement()) {
            ATB_LOG(ERROR) << GetLogPrefix() << " node[" << nodeId << "] create implement fail";
            return ERROR_INVALID_PARAM;
        }
        node.impl->ResetLogPrefix(logPrefix_, nodeId);
        ATB_LOG(DEBUG) << GetLogPrefix() << " node[" << nodeId << "] create implement success";
    }

    if (!node.impl->BuildLaunchParam(node.inTensors, node.inTensorViewFuncs, node.opDesc, node.outTensors.size())) {
        ATB_LOG(ERROR) << GetLogPrefix() << " node[" << nodeId << "] build launchParam fail";
        return ERROR_INVALID_PARAM;
    }

    if (!node.impl->PlanKernelInferShape()) {
        ATB_LOG(ERROR) << GetLogPrefix() << " node[" << nodeId << "] plan kernel inferShape fail";
        return ERROR_RT_FAIL;
    }

    if (!UpdateNodeBestKernelAndTiling(node, nodeId, kernelHostTilingBuffer, maxTilingSize, launchWithTiling)) {
        ATB_LOG(ERROR) << GetLogPrefix() << " node[" << nodeId << "] get best node's kernel fail";
        return ERROR_INVALID_PARAM;
    }

    MallocInternalTensor(node, nodeId);

    FreeInternalTensor(node, nodeId);

    return NO_ERROR;
}

void OpsRunner::FreeInternalTensor(KernelGraphNode &node, size_t nodeId)
{
    auto it = maxNodeIdTensorMap_.find(nodeId);
    if (it != maxNodeIdTensorMap_.end()) {
        for (auto tensorIt : it->second) {
            memAllocationSolver_->Free((uint8_t *)tensorIt->data);
            mallocCache_.push_back({tensorIt, false});
            ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] " << node.GetName() << " mem free.";
#ifdef _DEBUG
            ATB_LOG(INFO) << "data :" << tensorIt->data;
#endif
        }
    }
}

void OpsRunner::MallocLocalInternalTensor(const KernelGraphNode &node, size_t nodeId, size_t tensorId,
                                          const Mki::Tensor &infershapedOutTensor, Mki::Tensor *outTensor)
{
    if (tensorMalloced_.find(outTensor) == tensorMalloced_.end()) {
        tensorMalloced_.insert(outTensor);
        if (infershapedOutTensor.desc.dims.size() != 0) {
            outTensor->desc = infershapedOutTensor.desc;
        } else {
            ATB_LOG(ERROR) << GetLogPrefix() << " node[" << nodeId << "] " << node.GetName() << " outTensors["
                           << tensorId << "] is internal tensor, infer shape wrong, not use infer shape desc";
        }
        outTensor->dataSize = TensorUtil::CalcTensorDataSize(outTensor->desc);
        outTensor->data = memAllocationSolver_->GetOffset(TensorUtil::AlignInt(outTensor->dataSize, ALIGN_INT));
        mallocCache_.push_back({outTensor, true});
        ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] " << node.GetName() << " outTensors[" << tensorId
                      << "] is internal tensor, dataSize: " << outTensor->dataSize << ", mem solve.";
#ifdef _DEBUG
        ATB_LOG(INFO) << "data :" << outTensor->data;
#endif
    } else {
        ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] " << node.GetName() << " outTensors[" << tensorId
                      << "] is internal tensor, write in place";
    }
}

void OpsRunner::MallocGlobalInternalTensor(const KernelGraphNode &node, size_t nodeId, size_t tensorId,
                                           const Mki::Tensor &infershapedOutTensor, Mki::Tensor *outTensor)
{
    auto it = isOutTensorNeedMalloc_.find(outTensor);
    if (it != isOutTensorNeedMalloc_.end()) {
        if (it->second && tensorMalloced_.find(outTensor) == tensorMalloced_.end()) {
            tensorMalloced_.insert(outTensor);
            outTensor->data = memAllocationSolver_->GetOffset(TensorUtil::AlignInt(outTensor->dataSize, ALIGN_INT));
            mallocCache_.push_back({outTensor, true});
            ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] " << node.GetName() << " outTensors["
                          << tensorId << "] is outtensor, need malloc, dataSize: " << outTensor->dataSize
                          << ", mem solve.";
#ifdef _DEBUG
            ATB_LOG(INFO) << "data :" << outTensor->data;
#endif
        } else {
            ATB_LOG(DEBUG) << GetLogPrefix() << " node[" << nodeId << "] " << node.GetName() << " outTensors["
                           << tensorId << "] is outtensor, has malloced";
        }
    }
    if (!TensorUtil::AsdOpsTensorDescEqual(outTensor->desc, infershapedOutTensor.desc)) {
        ATB_LOG(WARN) << GetLogPrefix() << " node[" << nodeId
                      << "] outTensor->desc:" << TensorUtil::AsdOpsTensorDescToString(outTensor->desc)
                      << " != infershapedOutTensor.desc:"
                      << TensorUtil::AsdOpsTensorDescToString(infershapedOutTensor.desc);
    }
}

void OpsRunner::MallocInternalTensor(KernelGraphNode &node, size_t nodeId)
{
    auto &infershapedOutTensors = node.impl->GetOutTensors();
    for (size_t i = 0; i < node.outTensors.size(); ++i) {
        Mki::Tensor *outTensor = node.outTensors.at(i);
        Mki::Tensor &infershapedOutTensor = infershapedOutTensors.at(i);
        if (node.outTensorsType.at(i) == TensorType::INTERMEDIATE_TENSOR) {
            MallocLocalInternalTensor(node, nodeId, i, infershapedOutTensor, outTensor);
        } else {
            MallocGlobalInternalTensor(node, nodeId, i, infershapedOutTensor, outTensor);
        }
        infershapedOutTensor.data = outTensor->data;
        infershapedOutTensor.dataSize = outTensor->dataSize;

        ATB_LOG(DEBUG) << GetLogPrefix() << " node[" << nodeId << "] after mem solve, launchParam outtensor[" << i
                       << "]:\n"
                       << infershapedOutTensor.ToString();
    }
}


size_t OpsRunner::GetNodeAlignedTilingBufferSize(const KernelGraphNode &node, size_t nodeId) const
{
    size_t orgTilingSize = node.impl->GetTilingSize();
    size_t tilingSize = static_cast<size_t>(TensorUtil::AlignInt(static_cast<int64_t>(orgTilingSize), 32));
    ATB_LOG(DEBUG) << GetLogPrefix() << " node[" << nodeId << "] " << node.GetName()
                   << " orgTilingSize:" << orgTilingSize << ", tilingSize:" << tilingSize;
    return tilingSize;
}

void OpsRunner::SearchTensorInNodeInTensor(const Mki::Tensor *tensor, uint64_t &maxNodeId, uint64_t &dependNodeCount)
{
    for (size_t nodeId = 0; nodeId < kernelGraph_.nodes.size(); ++nodeId) {
        auto &node = kernelGraph_.nodes.at(nodeId);
        for (auto inTensorIt : node.inTensors) {
            if (tensor == inTensorIt) {
                maxNodeId = nodeId;
                dependNodeCount++;
            }
        }
    }
}

bool OpsRunner::SearchTensorInNodeOutTensor(const Mki::Tensor *tensor, uint64_t &maxNodeId)
{
    for (size_t nodeId = 0; nodeId < kernelGraph_.nodes.size(); ++nodeId) {
        auto &node = kernelGraph_.nodes.at(nodeId);
        for (auto outTensorIt : node.outTensors) {
            if (tensor == outTensorIt) {
                maxNodeId = nodeId;
                return true;
            }
        }
    }
    return false;
}

void OpsRunner::InitTensorMaxNodeMap()
{
    if (!tensorMaxNodeIdMap_.empty()) {
        ATB_LOG(INFO) << GetLogPrefix() << " InitTensorMaxNodeMap call once";
        return;
    }

    if (GetSingleton<Config>().IsworkspaceMemAllocGlobal()) {
        const size_t kernelGraphInTensorsSize = kernelGraph_.inTensors.size();
        for (size_t i = 0; i < kernelGraphInTensorsSize; ++i) {
            Mki::Tensor *tensor = &kernelGraph_.inTensors.at(i);
            auto it = isInTensorCanFree_.find(tensor);
            if (it == isInTensorCanFree_.end() || !it->second) {
                continue; // 若intensor的isInTensorCanFree为false，不参与内存释放
            }
            uint64_t maxNodeId = 0;
            uint64_t dependNodeCount = 0;

            SearchTensorInNodeInTensor(tensor, maxNodeId, dependNodeCount);
            if (dependNodeCount == 0) {
                ATB_LOG(WARN) << GetLogPrefix() << "intensor[" << i << "] dependNodeCount is 0, graph wrong";
                memAllocationSolver_->Free((uint8_t *)tensor->data); // 当intensor在graph内未被使用时，立即释放
                mallocCache_.push_back({tensor, false});
            } else {
                ATB_LOG(INFO) << GetLogPrefix() << "intensor[" << i << "] maxNodeId: " << maxNodeId
                              << ", dependNodeCount: " << dependNodeCount;
                tensorMaxNodeIdMap_[tensor] = maxNodeId;
                maxNodeIdTensorMap_[maxNodeId].insert(tensor);
            }
        }
    }
    const size_t kernelGraphInternalTensorsSize = kernelGraph_.internalTensors.size();
    for (size_t i = 0; i < kernelGraphInternalTensorsSize; ++i) {
        Mki::Tensor *tensor = &kernelGraph_.internalTensors[i];
        uint64_t maxNodeId = 0;
        uint64_t dependNodeCount = 0;

        SearchTensorInNodeInTensor(tensor, maxNodeId, dependNodeCount);
        if (dependNodeCount == 0) {
            ATB_LOG(WARN) << GetLogPrefix() << "internal tensor[" << i << "] dependNodeCount is 0, graph wrong";

            if (!SearchTensorInNodeOutTensor(tensor, maxNodeId)) {
                ATB_LOG(WARN) << GetLogPrefix() << "internal tensor[" << i << "] is not valued, graph wrong";
                continue; // 当中间tensor未在graph内使用时，若有被赋值，则赋值结束立刻释放；若未被赋值，则无需释放
            }
        } else {
            ATB_LOG(INFO) << GetLogPrefix() << "internal tensor[" << i << "] maxNodeId: " << maxNodeId
                          << ", dependNodeCount: " << dependNodeCount;
        }
        tensorMaxNodeIdMap_[tensor] = maxNodeId;
        maxNodeIdTensorMap_[maxNodeId].insert(tensor);
    }
}

Status OpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;
    return ErrorType::NO_ERROR;
}

Status OpsRunner::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;
    overrideModifyGraph_ = false;
    return NO_ERROR;
}

void OpsRunner::InitKernelCache()
{
    if (kernelCacheInited_) {
        return;
    }

    uint32_t localCacheCount = GetSingleton<Config>().GetLocalKernelCacheCount();
    uint32_t globalCacheCount = GetSingleton<Config>().GetGlobalKernelCacheCount();

    kernelCaches_.clear();

    localKernelCache_.Init(kernelGraph_.nodes.size(), localCacheCount);
    kernelCaches_.push_back(std::make_pair(&localKernelCache_, true));

    g_globalKernelCaches.at(runnerType_).Init(kernelGraph_.nodes.size(), globalCacheCount);
    kernelCaches_.push_back(std::make_pair(&(g_globalKernelCaches.at(runnerType_)), false));

    kernelCacheInited_ = true;
}

void OpsRunner::RunKernelPreProcess(KernelGraphNode &node, size_t nodeId, aclrtStream stream)
{
    if (nodesSaveTensorFlag_.at(nodeId) && Probe::IsExecuteCountInRange(executeCount_) && Probe::IsSaveTensorBefore()) {
        std::string dirPath = GetSaveTensorDir() + "/" + std::to_string(nodeId) + "_" + node.GetName() + "/before";
        node.impl->SaveLaunchParam(stream, dirPath);
        ATB_LOG(INFO) << GetLogPrefix() << " " << node.GetName() << " SaveRunInfo " << dirPath;
    }
    if (GetSingleton<Config>().IsCompareTilingEveryKernelEnable()) {
        SyncStream(node, nodeId, stream);
        std::string dirPath = GetSaveTensorDir() + "/" + std::to_string(nodeId) + "_" + node.GetName() + "/before";
        SaveGlobalDeviceTiling(dirPath, globalTilingBeforeKernelRun_);
    }
}

void OpsRunner::RunKernelPostProcess(KernelGraphNode &node, size_t nodeId, aclrtStream stream)
{
    if (GetSingleton<Config>().IsStreamSyncEveryKernelEnable()) {
        SyncStream(node, nodeId, stream);
    }
    if (nodesSaveTensorFlag_.at(nodeId) && Probe::IsExecuteCountInRange(executeCount_) && Probe::IsSaveTensorAfter()) {
        std::string dirPath = GetSaveTensorDir() + "/" + std::to_string(nodeId) + "_" + node.GetName() + "/after";
        node.impl->SaveLaunchParam(stream, dirPath);
        ATB_LOG(INFO) << GetLogPrefix() << " " << node.GetName() << " SaveLaunchParam " << dirPath;
    }
    if (GetSingleton<Config>().IsCompareTilingEveryKernelEnable()) {
        SyncStream(node, nodeId, stream);
        std::string dirPath = GetSaveTensorDir() + "/" + std::to_string(nodeId) + "_" + node.GetName() + "/after";
        SaveGlobalDeviceTiling(dirPath, globalTilingAfterKernelRun_);
        ATB_LOG_IF(globalTilingBeforeKernelRun_ != globalTilingAfterKernelRun_, FATAL)
            << GetLogPrefix() << " node[" << nodeId << "] " << node.GetName() << " change global device tiling";
    }
}

void OpsRunner::SyncStream(KernelGraphNode &node, size_t nodeId, aclrtStream stream) const
{
    Mki::Timer streamSyncTimer;
    ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] " << node.GetName() << " aclrtSynchronizeStream.";
    int ret = aclrtSynchronizeStream(stream);
    GetOpExecuteStatistic().streamSyncTime += streamSyncTimer.ElapsedMicroSecond();
    ATB_LOG_IF(ret != 0, ERROR) << GetLogPrefix() << " node[" << nodeId << "] aclrtSynchronizeStream fail, ret:" << ret;
}

void OpsRunner::SaveGlobalDeviceTiling(const std::string &dirPath, std::vector<uint8_t> &tilingData) const
{
    void *tilingDeviceBuffer = nullptr;
    uint64_t tilingBufferSize = 0;
    GetCurrentOpTiling(tilingDeviceBuffer, tilingBufferSize);
    if (tilingDeviceBuffer == nullptr || tilingBufferSize == 0) {
        ATB_LOG(INFO) << GetLogPrefix() << " global op tiling is empty, not save it";
        return;
    }

    if (tilingBufferSize >= MAX_TILING_BUFFER_SIZE) { // 1048576000 max global op tiling size
        ATB_LOG(WARN) << GetLogPrefix() << " global op tiling is too large, not save it";
        return;
    }

    tilingData.resize(tilingBufferSize, 0);

    ATB_LOG(INFO) << GetLogPrefix() << " copy device tiling to host tilingBufferSize:" << tilingBufferSize;
    int st = aclrtMemcpy(tilingData.data(), tilingData.size(), tilingDeviceBuffer, tilingBufferSize,
                         ACL_MEMCPY_DEVICE_TO_HOST);
    ATB_LOG_IF(st != 0, ERROR) << GetLogPrefix() << "aclrtMemcpy device to host fail for save tiling, ret:" << st;
    StoreUtil::WriteFile(tilingData.data(), tilingData.size(), dirPath + "/global_tilingdata.bin");
}

void OpsRunner::IncreaseStatisticCacheHitCount(bool localCache) const
{
    GetOpSetupStatistic().tilingCacheHitCount += 1;
    if (localCache) {
        GetOpSetupStatistic().tilingLocalCacheHitCount += 1;
    } else {
        GetOpSetupStatistic().tilingGlobalCacheHitCount += 1;
    }
}

bool OpsRunner::GetCachedTiling(KernelGraphNode &node, size_t nodeId, uint8_t *kernelHostTilingBuffer,
                                uint64_t maxTilingSize, uint64_t &tilingSizeFetched, bool launchWithTiling)
{
    if (!node.tilingCacheEnable) {
        return false;
    }
    const size_t kernelCachesSize = kernelCaches_.size();
    for (size_t i = 0; i < kernelCachesSize; ++i) {
        KernelCache *kernelCache = kernelCaches_.at(i).first;
        bool isLocalCache = kernelCaches_.at(i).second;
        bool getTilingSuccess = node.impl->GetCachedTiling(*kernelCache, nodeId, kernelHostTilingBuffer,
                                                           maxTilingSize, tilingSizeFetched, launchWithTiling);
        if (getTilingSuccess) {
            ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] kernel cache get last tiling";
            IncreaseStatisticCacheHitCount(isLocalCache);
            return true;
        }
    }
    return false;
}

void OpsRunner::UpdateCacheTiling(KernelGraphNode &node, size_t nodeId, uint8_t *kernelHostTilingBuffer,
                                  size_t tilingSize)
{
    if (!node.tilingCacheEnable) { // 更新缓存里的RunInfo和tiling
        return;
    }

    const size_t kernelCachesSize = kernelCaches_.size();
    for (size_t i = 0; i < kernelCachesSize; ++i) {
        KernelCache *kernelCache = kernelCaches_.at(i).first;
        node.impl->AddTiling(*kernelCache, nodeId, kernelHostTilingBuffer, tilingSize);
    }
}

void OpsRunner::BuildAdditionalInfo(
    const uint64_t nameHash,
    const Mki::SVector<std::pair<bool, Mki::Tensor>, ATB_MSPROF_TENSOR_DATA_SIZE> &batchTensors,
    MsprofAdditionalInfo &additionInfo) const
{
    // profiling的tensorInfo数据构造
    additionInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
    additionInfo.level = MSPROF_REPORT_NODE_LEVEL;
    long tid = UtilsInternal::GetCurrentThreadId();
    if (tid == -1) {
        return;
    }
    additionInfo.threadId = static_cast<uint32_t>(tid);
    additionInfo.dataLen = K_TENSOR_INFO_BYTES_WITH_CAP + (batchTensors.size() - 1) * K_TENSOR_INFO_BYTES;
    // 数据指针类型转换
    auto profTensorData = reinterpret_cast<MsprofTensorInfo *>(additionInfo.data);
    profTensorData->opName = nameHash;
    profTensorData->tensorNum = batchTensors.size();

    BuildTensorData(batchTensors, (*profTensorData));
}

void OpsRunner::BuildTensorData(
    const Mki::SVector<std::pair<bool, Mki::Tensor>, ATB_MSPROF_TENSOR_DATA_SIZE> &batchTensors,
    MsprofTensorInfo &tensorDataInfo) const
{
    // 依次构造本批次内tensor数据
    const size_t batchTensorsSize = batchTensors.size();
    for (size_t i = 0; i < batchTensorsSize; i++) {
        const auto &tensor = batchTensors.at(i);
        tensorDataInfo.tensorData[i].tensorType =
            tensor.first ? MSPROF_GE_TENSOR_TYPE_INPUT : MSPROF_GE_TENSOR_TYPE_OUTPUT;
        tensorDataInfo.tensorData[i].format = tensor.second.desc.format;
        tensorDataInfo.tensorData[i].dataType = tensor.second.desc.dtype;
        const size_t shapeSize = tensor.second.desc.dims.size();
        const size_t limitedSize = std::min(static_cast<size_t>(MSPROF_GE_TENSOR_DATA_SHAPE_LEN), shapeSize);
        for (size_t j = 0; j < limitedSize; j++) {
            tensorDataInfo.tensorData[i].shape[j] = static_cast<uint32_t>(tensor.second.desc.dims.at(j));
        }

        for (size_t j = limitedSize; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {
            tensorDataInfo.tensorData[i].shape[j] = 0;
        }
    }
}

void OpsRunner::DoReportTensorInfo(MsprofAdditionalInfo &additionInfo) const
{
    auto ret = GetSingleton<Mki::ProfilingFuncs>().ProfReportAdditionalInfo(
        static_cast<uint32_t>(true), &additionInfo, static_cast<uint32_t>(sizeof(MsprofAdditionalInfo)));
    ATB_LOG(INFO) << GetLogPrefix() << "ProfReportAdditionalInfo additionInfo: dataLen: " << additionInfo.dataLen
                  << "level: " << additionInfo.level << "magicNumber: " << additionInfo.magicNumber
                  << " threadId: " << additionInfo.threadId << " timeStamp: " << additionInfo.timeStamp
                  << " type: " << additionInfo.type;
    if (ret != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "ProfReportAdditionalInfo error!";
    }
}

void OpsRunner::SetNodesSaveTensorFlag()
{
    std::vector<int64_t> nodeIds(runnerIds_);
    nodeIds.push_back(0);
    size_t idx = nodeIds.size() - 1;
    const size_t kernelGraphNodesSize = kernelGraph_.nodes.size();
    nodesSaveTensorFlag_.resize(kernelGraphNodesSize);
    for (size_t i = 0; i < kernelGraphNodesSize; i++) {
        nodeIds.at(idx) = static_cast<int64_t>(i);
        nodesSaveTensorFlag_.at(i) = Probe::IsTensorNeedSave(nodeIds, GetOperationName());
    }
}

void OpsRunner::DumpKernelIOTensorInfo(KernelGraphNode &node) const
{
    const size_t nodeInTensorsSize = node.inTensors.size();
    std::vector<Probe::Tensor> inTensors(nodeInTensorsSize);
    std::string tensorDir = tensorDir_ + "/" + (Probe::IsSaveTensorBefore() ? "before" : "after");
    for (size_t i = 0; i < nodeInTensorsSize; ++i) {
        std::string fileName = "intensor" + std::to_string(i) + TENSOR_FILE_NAME_EXT;
        std::string tensorPath = Mki::FileSystem::Join({tensorDir, fileName});
        Probe::AsdopsTensorToProbeTensor(*node.inTensors[i], inTensors[i], tensorPath);
    }
    const size_t nodeOutTensorsSize = node.outTensors.size();
    std::vector<Probe::Tensor> outTensors(nodeOutTensorsSize);
    for (size_t i = 0; i < nodeOutTensorsSize; ++i) {
        std::string fileName = "outtensor" + std::to_string(i) + TENSOR_FILE_NAME_EXT;
        std::string tensorPath = Mki::FileSystem::Join({tensorDir, fileName});
        Probe::AsdopsTensorToProbeTensor(*node.outTensors[i], outTensors[i], tensorPath);
    }
    Probe::ReportKernelIOTensor(executeCount_, node.opDesc.opName, Mki::Stringify::ToString(node.opDesc.specificParam),
                                inTensors, outTensors);
}

bool OpsRunner::CheckOverflow(const KernelGraphNode &node, ContextBase *context) const
{
    OperationBase *opBase = dynamic_cast<OperationBase *>(operation_);
    if (!opBase) {
        ATB_LOG(ERROR) << GetLogPrefix() << "operation is not inherit from OperationBase";
        return false;
    }
    const std::vector<int64_t> &operationBaseIds = opBase->GetOperationBaseIds();
    std::string opName = GenerateOperationName(GetOperationName(), operationBaseIds);
    opName = opName + ":" + node.GetName();
    aclrtStream stream = GetExecuteStream(context);
    aclrtFloatOverflowMode satMode = ACL_RT_OVERFLOW_MODE_UNDEF;

    Status st = aclrtSynchronizeStream(stream);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "aclrtSynchronizeStream failed! ret:" << st;
        return false;
    }

    st = aclrtGetDeviceSatMode(&satMode);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "aclrtGetDeviceSatMode failed! ret:" << st;
        return false;
    }
    if (satMode == ACL_RT_OVERFLOW_MODE_SATURATION) {
        ATB_LOG(DEBUG) << "satMode is ACL_RT_OVERFLOW_MODE_SATURATION";
        return ExecuteOverFlowCheckKernel(opName, context);
    }
    if (satMode == ACL_RT_OVERFLOW_MODE_INFNAN) {
        ATB_LOG(DEBUG) << "satMode is ACL_RT_OVERFLOW_MODE_INFNAN";
        return CheckOverFlowByTensor(opName);
    }

    return false;
}

bool OpsRunner::ExecuteOverFlowCheckKernel(const std::string &opName, ContextBase *context) const
{
    AclKernel overflowKernel;
    SVector<Tensor> inTensors;
    SVector<Tensor> outTensors;
    int32_t hostBuffer = 0;
    aclrtStream stream = GetExecuteStream(context);
    outTensors.push_back(context->GetOverflowKernelOutTensor());
    Status st = overflowKernel.Run("NPUGetFloatStatusV2", inTensors, outTensors, stream);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "overflowKernel run failed! ret:" << st;
        return false;
    }
    st = aclrtSynchronizeStream(stream);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "aclrtSynchronizeStream failed! ret:" << st;
        return false;
    }

    st = aclrtMemcpy(&hostBuffer, sizeof(int32_t), outTensors.at(0).deviceData, sizeof(int32_t),
                     ACL_MEMCPY_DEVICE_TO_HOST);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "aclrtMemcpy failed! ret:" << st;
        return false;
    }

    if (hostBuffer == IS_OVERFLOW) {
        Probe::ReportOverflowKernel(opName);
        return true;
    } else {
        if (hostBuffer != NOT_OVERFLOW) {
            ATB_LOG(ERROR) << "overflow kernel result error!";
        }
        return false;
    }
    return false;
}

bool OpsRunner::CheckOverFlowByTensor(const std::string &opName) const
{
    const size_t opsTensorPackOutTensorsSize = opsTensorPack_.outTensors.size();
    for (size_t i = 0; i < opsTensorPackOutTensorsSize; i++) {
        const Mki::Tensor &tensor = opsTensorPack_.outTensors.at(i);
        std::vector<uint8_t> hostBuffer(tensor.dataSize);
        Status st =
            aclrtMemcpy(hostBuffer.data(), tensor.dataSize, tensor.data, tensor.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << "aclrtMemcpy failed! ret:" << st;
            return false;
        }

        if (JudgeOverflowTensor(opName, tensor, hostBuffer.data())) {
            return true;
        }
    }
    return false;
}

bool OpsRunner::JudgeOverflowTensor(const std::string &opName, const Mki::Tensor &tensor, uint8_t *hostBuffer) const
{
    if (hostBuffer == nullptr) {
        ATB_LOG(ERROR) << "HostBuffer of " << opName << " is nullptr!";
        return false;
    }
    if (tensor.desc.dtype == Mki::TensorDType::TENSOR_DTYPE_FLOAT16 ||
        tensor.desc.dtype == Mki::TensorDType::TENSOR_DTYPE_BF16) {
        Mki::fp16_t *bufferAddr = reinterpret_cast<Mki::fp16_t *>(hostBuffer);
        const size_t counter = tensor.dataSize / sizeof(Mki::fp16_t);
        for (size_t i = 0; i < counter; i++) {
            if (bufferAddr[i].IsInf()) {
                Probe::ReportOverflowKernel(opName);
                return true;
            }
        }
    }

    if (tensor.desc.dtype == Mki::TensorDType::TENSOR_DTYPE_FLOAT) {
        float *bufferAddr = reinterpret_cast<float *>(hostBuffer);
        const size_t counter = tensor.dataSize / sizeof(float);
        for (size_t i = 0; i < counter; i++) {
            if (std::isinf(bufferAddr[i])) {
                Probe::ReportOverflowKernel(opName);
                return true;
            }
        }
    }
    return false;
}

void OpsRunner::InitOpsTensorPack(const RunnerVariantPack &runnerPack)
{
    size_t opsTensorPackInTensorsSize = runnerPack.inTensors.size();
    size_t opsTensorPackOutTensorsSize = runnerPack.outTensors.size();
    opsTensorPack_.inTensors.resize(opsTensorPackInTensorsSize);
    opsTensorPack_.outTensors.resize(opsTensorPackOutTensorsSize);
    for (size_t i = 0; i < opsTensorPackInTensorsSize; i++) {
        TensorUtil::ConvertAtbTensor2OpsTensor(runnerPack.inTensors.at(i), opsTensorPack_.inTensors.at(i));
    }
    for (size_t i = 0; i < opsTensorPackOutTensorsSize; i++) {
        TensorUtil::ConvertAtbTensor2OpsTensor(runnerPack.outTensors.at(i), opsTensorPack_.outTensors.at(i));
    }
}

bool OpsRunner::IsSupportGlbWorkspace()
{
    return true;
}

uint64_t OpsRunner::GetArgsSize()
{
    uint64_t argsSize = 0;
    for (size_t i = 0; i < kernelGraph_.nodes.size(); i++) {
        ATB_LOG(DEBUG) << GetLogPrefix() << kernelGraph_.nodes.at(i).impl->GetName() << " argsSize from GetArgsSize is "
                       << kernelGraph_.nodes.at(i).impl->GetArgsSize();
        argsSize += kernelGraph_.nodes.at(i).impl->GetArgsSize();
    }
    return argsSize;
}

Status OpsRunner::BuildArgs()
{
    Status st = NO_ERROR;
    for (size_t i = 0; i < kernelGraph_.nodes.size(); i++) {
        st = kernelGraph_.nodes.at(i).impl->BuildArgs();
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "node [" << i << "] BuildArgs failed!";
            return st;
        }
    }
    return NO_ERROR;
}

Status OpsRunner::UpdateTensorAddr(RunnerVariantPack &runnerVariantPack)
{
    const size_t kernelGraphInTensorsSize = kernelGraph_.inTensors.size();
    for (size_t i = 0; i < kernelGraphInTensorsSize; i++) {
        TensorUtil::ConvertAtbTensor2OpsTensor(runnerVariantPack.inTensors.at(i), kernelGraph_.inTensors.at(i));
    }
    const size_t kernelGraphOutTensorsSize = kernelGraph_.outTensors.size();
    for (size_t i = 0; i < kernelGraphOutTensorsSize; i++) {
        TensorUtil::ConvertAtbTensor2OpsTensor(runnerVariantPack.outTensors.at(i), kernelGraph_.outTensors.at(i));
    }

    opsTensorPack_.inTensors = kernelGraph_.inTensors;
    opsTensorPack_.outTensors = kernelGraph_.outTensors;

    Status st = ModifyKernelGraph(opsTensorPack_);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "ModifyKernelGraph failed!";
        return st;
    }

    for (size_t i = 0; i < kernelGraph_.nodes.size(); i++) {
        KernelGraphNode &node = kernelGraph_.nodes.at(i);
        UpdateRunInfoTensorData(node, i, runnerVariantPack.intermediateBuffer);
        st = node.impl->BuildLaunchParam(node.inTensors, node.outTensors, node.opDesc);
        if (st != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "build launchParam fail";
            return st;
        }
    }
    return NO_ERROR;
}

Status OpsRunner::UpdateWorkspaceBuffer(RunnerVariantPack &runnerVariantPack)
{
    bool needSetworkspace = (workspaceSize_ != 0);
    for (size_t nodeId = 0; nodeId < kernelGraph_.nodes.size(); ++nodeId) {
        KernelGraphNode &node = kernelGraph_.nodes.at(nodeId);
        if (needSetworkspace) {
#ifdef _DEBUG
        ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] update kernel runinfo workspaceBuffer, and new workspaceBuffer is "
                      << static_cast<void *>(runnerVariantPack.workspaceBuffer);
#else
        ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] update kernel runinfo workspaceBuffer";
#endif
            node.impl->SetWorkspaceDeviceAddr(runnerVariantPack.workspaceBuffer);
        }
    }
    return NO_ERROR;
}
} // namespace atb