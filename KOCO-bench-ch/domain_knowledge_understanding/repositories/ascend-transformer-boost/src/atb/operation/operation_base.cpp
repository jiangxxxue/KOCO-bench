/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/operation/operation_base.h"
#include <sstream>
#include <atomic>
#include <unistd.h>
#include <cstring>
#include <acl/acl_rt.h>
#include <mki/utils/time/timer.h>
#include <mki/utils/SVector/SVector.h>
#include <mki/utils/profiling/profiling_funcs.h>
#include "atb/utils/config.h"
#include "atb/utils/probe.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/log.h"
#include "atb/utils.h"
#include "atb/utils/statistic.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/current_op_tiling.h"
#include "atb/utils/mem_allocation_solver/mem_allocation_solver_creator.h"
#include "atb/utils/utils_internal.h"
#include "atb/utils/common_utils.h"
#include "atb/utils/singleton.h"
#include "atb/utils/mstx_mem_register.h"

namespace atb {
static std::atomic_int64_t g_operationBaseId(0);
constexpr size_t WORKSPACE_ALIGN = 512;

int64_t GetNewOperationBaseId()
{
    int64_t operationId = g_operationBaseId++;
    return operationId;
}

static uint32_t GetTypeIdFromHash(uint64_t hashId)
{
    uint32_t typeId = static_cast<uint32_t>(hashId); // 获取 uint64_t 哈希值的低 32 位
    typeId &= 0x0000FFFF;                            // 清除高 16 位
    typeId |= 0x000B0000;                            // 设置高 16 位为 0x000B
    return typeId;
}

OperationBase::OperationBase(const std::string &name) : name_(name)
{
    operationBaseIds_.clear();
    operationBaseIds_.push_back(-1);
    logPrefix_ = name_ + " ";
}

OperationBase::~OperationBase()
{
    if (isGraphLaunchMode_) {
        // 只有整图下沉的情况下需要销毁Args的buffer，规避context析构和operation析构顺序问题
        // 对于GraphOperation来说，里面的子Op都不会有runnerVariantPack_
        if (runnerVariantPack_.context) {
            ATB_LOG(INFO) << GetLogPrefix() << "will free deviceArgsBuffer_ and hostArgsBuffer_";
            // 此处如果先destroy了context再destroy operation，里面调用aclrtFree接口会报错
            runnerVariantPack_.context->FreeArgsDeviceBuffer(deviceArgsBuffer_);
            runnerVariantPack_.context->FreeArgsHostBuffer(hostArgsBuffer_);
        }
    }
}

Status OperationBase::SetOperationBaseIds(const std::vector<int64_t> &operationBaseIds, const int64_t nodeId)
{
    operationBaseIds_ = operationBaseIds;
    operationBaseIds_.push_back(nodeId);
    ResetLogPrefix();

    return SetNodeOperationIds();
}

Status OperationBase::SetNodeOperationIds()
{
    return NO_ERROR;
}

std::string OperationBase::GetName() const
{
    return name_;
}

void OperationBase::InitEmptyInTensorPerms() const
{
    if (!operationIr_) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "operationIr_ is null.";
        return;
    }
    if (!operationIr_->IsValid()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "operationIr_ is invalid";
        return;
    }
    ATB_LOG(DEBUG) << GetLogPrefix() << "operationIr_ : " << operationIr_->ToString();
    const Mki::SVector<Mki::TensorInfoIr> &inTensorInfoIrs = operationIr_->GetInTensorInfoIrs();
    if (inTensorInfoIrs.size() != GetInputNum()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "GetInTensorInfoIrs size: " << inTensorInfoIrs.size()
                       << " is not equal with GetInputNum  : " << GetInputNum();
        return;
    }
    emptyInTensorPerms_.reserve(GetInputNum());
    emptyInTensorPerms_.resize(GetInputNum());
    for (size_t inTensorId = 0; inTensorId < inTensorInfoIrs.size(); inTensorId++) {
        if (inTensorInfoIrs[inTensorId].isOptional) {
            emptyInTensorPerms_.at(inTensorId) = true;
            ATB_LOG(INFO) << GetLogPrefix() << "emptyInTensorPerms init inTensor[" << inTensorId << "] is isOptional";
        } else {
            emptyInTensorPerms_.at(inTensorId) = false;
        }
    }
    ATB_LOG(INFO) << GetLogPrefix() << "InitEmptyInTensorPerms finished:" << emptyInTensorPerms_;
}

SVector<bool> OperationBase::GetEmptyInTensorPermissions() const
{
    if (emptyInTensorPerms_.size() == 0) {
        InitEmptyInTensorPerms();
    }
    return emptyInTensorPerms_;
}

void OperationBase::InitEmptyOutTensorPerms() const
{
    if (!operationIr_) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "operationIr_ is null.";
        return;
    }
    if (!operationIr_->IsValid()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "operationIr_ is invalid";
        return;
    }
    ATB_LOG(DEBUG) << GetLogPrefix() << "operationIr_ : " << operationIr_->ToString();
    const Mki::SVector<Mki::TensorInfoIr> &outTensorInfoIrs = operationIr_->GetOutTensorInfoIrs();
    if (GetOutputNum() != 0 && outTensorInfoIrs.size() != GetOutputNum()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "GetOutTensorInfoIrs size: " << outTensorInfoIrs.size()
                       << " which is not equals to GetOutputNum: " << GetOutputNum();
        return;
    }
    emptyOutTensorPerms_.reserve(outTensorInfoIrs.size());
    emptyOutTensorPerms_.resize(outTensorInfoIrs.size());
    for (size_t outTensorId = 0; outTensorId < outTensorInfoIrs.size(); outTensorId++) {
        if (outTensorInfoIrs[outTensorId].isOptional) {
            emptyOutTensorPerms_.at(outTensorId) = true;
            ATB_LOG(INFO) << GetLogPrefix() << "emptyOutTensorPerms init outTensor[" << outTensorId
                          << "] is isOptional";
        } else {
            emptyOutTensorPerms_.at(outTensorId) = false;
        }
    }
    ATB_LOG(INFO) << GetLogPrefix() << "InitEmptyOutTensorPerms finished:" << emptyOutTensorPerms_;
}
 
SVector<bool> OperationBase::GetEmptyOutTensorPermissions() const
{
    if (emptyOutTensorPerms_.size() == 0) {
        InitEmptyOutTensorPerms();
    }
    return emptyOutTensorPerms_;
}

template <typename TensorType> Status OperationBase::CheckInTensor(const SVector<TensorType> &inTensors) const
{
    Status st = NO_ERROR;
    SVector<bool> emptyTensorPerms = GetEmptyInTensorPermissions();
    for (size_t inTensorId = 0; inTensorId < inTensors.size(); inTensorId++) {
        const auto &inTensor = inTensors.at(inTensorId);
        if (inTensorId < emptyTensorPerms.size() && emptyTensorPerms.at(inTensorId) &&
            TensorCheck::IsEmptyTensor(inTensor)) {
            ATB_LOG(INFO) << GetLogPrefix() << "inTensor [" << inTensorId << "] is allowed empty";
            continue;
        }
        st = TensorCheck::CheckTensorShape(inTensor);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensor [" << inTensorId
                           << "] CheckTensorShape failed. ErrorType: " << st;
            return st;
        }
    }
    return NO_ERROR;
}

template <typename TensorType> Status OperationBase::CheckOutTensor(const SVector<TensorType> &outTensors) const
{
    Status st = NO_ERROR;
    SVector<bool> emptyTensorPerms = GetEmptyOutTensorPermissions();
    for (size_t outTensorId = 0; outTensorId < outTensors.size(); outTensorId++) {
        const auto &outTensor = outTensors.at(outTensorId);
        if (outTensorId < emptyTensorPerms.size() && emptyTensorPerms.at(outTensorId) &&
            TensorCheck::IsEmptyTensor(outTensor)) {
            ATB_LOG(INFO) << GetLogPrefix() << "outTensor [" << outTensorId << "] is allowed empty";
            continue;
        }
        st = TensorCheck::CheckTensorShape(outTensor);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor [" << outTensorId
                           << "] CheckTensorShape failed. ErrorType: " << st;
            return st;
        }
    }
    return NO_ERROR;
}

template <typename TensorType>
static bool CheckIniMatchSupportIdx(const SVector<TensorType> &tensors,
                                    const Mki::SVector<Mki::TensorInfoIr> &tensorInfoIrs, const size_t &supportIdx)
{
    for (size_t tensorIdx = 0; tensorIdx < tensors.size(); tensorIdx++) {
        if (TensorCheck::IsEmptyTensor(tensors[tensorIdx])) {
            continue;
        }
        aclDataType tensorDType = static_cast<aclDataType>(tensorInfoIrs[tensorIdx].supportedDtypes[supportIdx]);
        aclFormat tensorFormat = static_cast<aclFormat>(tensorInfoIrs[tensorIdx].supportedFormats[supportIdx]);
        if ((!TensorCheck::IsTensorDType(tensors[tensorIdx], tensorDType)) ||
            (!TensorCheck::IsTensorFormat(tensors[tensorIdx], tensorFormat))) {
            return false;
        }
    }
    return true;
}

bool OperationBase::CheckIniMatch(const SVector<TensorDesc> &inTensorDescs) const
{
    if (!operationIr_) {
        return true;
    }
    if (!operationIr_->IsValid()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "operationIr_ is invalid";
        return false;
    }
    const Mki::SVector<Mki::TensorInfoIr> &inTensorInfoIrs = operationIr_->GetInTensorInfoIrs();
    if (inTensorDescs.size() != inTensorInfoIrs.size()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensorDescs size: " << inTensorDescs.size() << " is not equal "
                       << "inTensorInfoIrs size : " << inTensorInfoIrs.size();
        return false;
    }
    size_t supportSize = operationIr_->GetSupportSize();
    for (size_t supportIdx = 0; supportIdx < supportSize; supportIdx++) {
        if (CheckIniMatchSupportIdx(inTensorDescs, inTensorInfoIrs, supportIdx)) {
            ATB_LOG(INFO) << GetLogPrefix() << "dType and format matched. index: " << supportIdx;
            return true;
        }
    }
    return false;
}

bool OperationBase::CheckIniMatch(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    if (!operationIr_) {
        return true;
    }
    if (!operationIr_->IsValid()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "operationIr_ is invalid";
        return false;
    }
    const Mki::SVector<Mki::TensorInfoIr> &inTensorInfoIrs = operationIr_->GetInTensorInfoIrs();
    const Mki::SVector<Mki::TensorInfoIr> &outTensorInfoIrs = operationIr_->GetOutTensorInfoIrs();
    if (inTensors.size() != inTensorInfoIrs.size()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensors size: " << inTensors.size() << " is not equal "
                       << "inTensorInfoIrs size : " << inTensorInfoIrs.size();
        return false;
    }
    if (outTensors.size() != 0 && outTensors.size() != outTensorInfoIrs.size()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensors size: " << outTensors.size() << " is not equal "
                       << "outTensorInfoIrs size : " << outTensorInfoIrs.size();
        return false;
    }
    size_t supportSize = operationIr_->GetSupportSize();
    for (size_t supportIdx = 0; supportIdx < supportSize; supportIdx++) {
        if (CheckIniMatchSupportIdx(inTensors, inTensorInfoIrs, supportIdx) &&
            CheckIniMatchSupportIdx(outTensors, outTensorInfoIrs, supportIdx)) {
            ATB_LOG(INFO) << GetLogPrefix() << "dType and format matched. index: " << supportIdx;
            return true;
        }
    }
    return false;
}

Status OperationBase::InferShapeCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    ATB_LOG(INFO) << GetLogPrefix() << "infer shape start";
    for (size_t i = 0; i < inTensorDescs.size(); ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "infer shape inTensorDescs[" << i
                      << "]:" << TensorUtil::TensorDescToString(inTensorDescs.at(i));
    }
    if (inTensorDescs.size() != GetInputNum()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensorDescs.size: " << inTensorDescs.size()
                       << " is not equal GetInputNum: " << GetInputNum();
        return ERROR_INVALID_IN_TENSOR_NUM;
    }
    Status st = CheckInTensor(inTensorDescs);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Check in tensor desc fail, ErrorType: " << st;
        return st;
    }
    if (!CheckIniMatch(inTensorDescs)) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "CheckIniMatch Failed! Supported "
                          "Combs: \n"
                       << operationIr_->GetCombString();
        return ERROR_INVALID_TENSOR_INI_MATCH;
    }
    if (!TensorCheck::CheckBF16(inTensorDescs)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atlas inference products can not support bf16.";
        return ERROR_INVALID_TENSOR_DTYPE;
    }
    st = InferShapeCheckImpl(inTensorDescs);
    return st;
}

Status OperationBase::InferShapeThrow(const SVector<TensorDesc> &inTensorDescs,
                                      SVector<TensorDesc> &outTensorDescs) const
{
    uint64_t outTensorNum = GetOutputNum();
    outTensorDescs.reserve(outTensorNum);
    outTensorDescs.resize(outTensorNum);

    Status st = InferShapeImpl(inTensorDescs, outTensorDescs);

    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "infer shape outTensorDescs[" << i
                      << "]:" << TensorUtil::TensorDescToString(outTensorDescs.at(i));
    }
    return st;
}

Status OperationBase::InferShape(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const
{
    Status st = NO_ERROR;
    try {
        st = InferShapeCheck(inTensorDescs);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "infer shape check fail, error code: " << st;
            return st;
        }
        ATB_LOG(DEBUG) << GetLogPrefix() << "infer shape check success";
        st = InferShapeThrow(inTensorDescs, outTensorDescs);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "infer shape fail, error code: " << st;
            return st;
        }
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) << GetLogPrefix() << "infer shape throw an exception: " << e.what();
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return st;
}

Status OperationBase::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    ATB_LOG(INFO) << GetLogPrefix() << "InTensorDesc Size:" << inTensorDescs.size();
    return NO_ERROR;
}

Status OperationBase::CheckVariantPack(const VariantPack &variantPack) const
{
    if (variantPack.inTensors.size() != GetInputNum()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "variantPack.inTensors.size: " << variantPack.inTensors.size()
                       << "is not equal GetInputNum: " << GetInputNum();
        return ERROR_INVALID_IN_TENSOR_NUM;
    }
    if (variantPack.outTensors.size() != GetOutputNum()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "variantPack.outTensors.size: " << variantPack.outTensors.size()
                       << "is not equal GetOutputNum: " << GetOutputNum();
        return ERROR_INVALID_IN_TENSOR_NUM;
    }
    Status st = NO_ERROR;
    st = CheckInTensor(variantPack.inTensors);
    if (st != NO_ERROR) {
        return st;
    }
    st = CheckOutTensor(variantPack.outTensors);
    if (st != NO_ERROR) {
        return st;
    }
    if (!CheckIniMatch(variantPack.inTensors, variantPack.outTensors)) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "CheckIniMatch Failed! Supported "
                          "Combs: \n"
                       << operationIr_->GetCombString();
        return ERROR_INVALID_TENSOR_INI_MATCH;
    }
    return NO_ERROR;
}

Status OperationBase::SetupCheck(const VariantPack &variantPack, Context *context)
{
    Status st = CheckVariantPack(variantPack);
    if (st != NO_ERROR) {
        return st;
    }
    SVector<TensorDesc> inTensorDescs = {};
    SVector<TensorDesc> outTensorDescs = {};
    inTensorDescs.reserve(variantPack.inTensors.size());
    outTensorDescs.reserve(variantPack.outTensors.size());
    OperationUtil::InTensorsToInTensorDescs(variantPack.inTensors, inTensorDescs);
    OperationUtil::InTensorsToInTensorDescs(variantPack.outTensors, outTensorDescs);
    if (!TensorCheck::CheckBF16(inTensorDescs) || !TensorCheck::CheckBF16(outTensorDescs)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atlas inference products can not support bf16.";
        return ERROR_INVALID_TENSOR_DTYPE;
    }
    st = SetupCheckImpl(variantPack.inTensors, variantPack.outTensors);
    if (st != NO_ERROR) {
        return st;
    }

    if (!context) {
        ATB_LOG(ERROR) << GetLogPrefix() << "context is null, setup fail";
        return ERROR_INVALID_PARAM;
    }

    runnerVariantPack_.context = dynamic_cast<ContextBase *>(context);
    if (!runnerVariantPack_.context) {
        ATB_LOG(ERROR) << GetLogPrefix() << "context is not ContextBase, setup fail";
        return ERROR_INVALID_PARAM;
    }

    return NO_ERROR;
}

Status OperationBase::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    ATB_LOG(INFO) << GetLogPrefix() << "inTensorsSize:" << inTensors.size() << "outTensorsSize:" << outTensors.size();
    return NO_ERROR;
}

void OperationBase::SetSaveTensorDir()
{
    if (operationBaseIds_.size() == 0) {
        ATB_LOG(FATAL) << GetLogPrefix() << "operationBaseIds is empty!";
        return;
    }
    std::string runnerSaveTensorDir = GetSaveTensorRootDir() + "/" + std::to_string(executeCount_) + "/" +
                                      std::to_string(operationBaseIds_.at(operationBaseIds_.size() - 1)) + "_" + name_;
    runner_->SetSaveTensorDir(runnerSaveTensorDir);
}

Status OperationBase::CreateRunnerFunc(Context *context)
{
    if (!runner_) {
        if (context == nullptr) {
            ATB_LOG(ERROR) << "context is nullptr!";
            return ERROR_INVALID_CONTEXT_ADDR;
        }
        runner_ = CreateRunner(*context);
        if (!runner_) {
            return ERROR_OPERATION_NULL_RUNNER;
        }
        if (Probe::ReportOperationGraphEnable()) {
            Probe::ReportOperationGraph(GenerateOperationName(name_, operationBaseIds_), GetGraphInfo().dump());
        }
    }
    runner_->SetRunnerOperation(this);
    runner_->SetRunnerInfo(name_, operationBaseIds_);
    return NO_ERROR;
}

Status OperationBase::SetupThrowPrepare(uint64_t &workspaceSize, Context *context)
{
    workspaceSize = 0;

    Status st = CreateRunnerFunc(context);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "CreateRunnerFunc failed";
        return st;
    }

    Reset();
    return NO_ERROR;
}

void OperationBase::InitRunnerVariantPack(const VariantPack &variantPack)
{
    runnerVariantPack_.inTensors.reserve(variantPack.inTensors.size());
    TensorUtil::FastCopyTensors(variantPack.inTensors, runnerVariantPack_.inTensors);
    TensorUtil::FastCopyTensors(variantPack.outTensors, runnerVariantPack_.outTensors);

    runnerVariantPack_.isInTensorCanFree.reserve(variantPack.inTensors.size());
    runnerVariantPack_.isInTensorCanFree.resize(variantPack.inTensors.size());
    for (size_t i = 0; i < variantPack.inTensors.size(); i++) {
        runnerVariantPack_.isInTensorCanFree.at(i) = false;
    }
    runnerVariantPack_.isOutTensorNeedMalloc.reserve(variantPack.outTensors.size());
    runnerVariantPack_.isOutTensorNeedMalloc.resize(variantPack.outTensors.size());
    for (size_t i = 0; i < variantPack.outTensors.size(); i++) {
        runnerVariantPack_.isOutTensorNeedMalloc.at(i) = false;
    }
}

Status OperationBase::SetupThrow(const VariantPack &variantPack, uint64_t &workspaceSize)
{
    Mki::Timer runnerCreateTimer;
    Status st = SetupThrowPrepare(workspaceSize, runnerVariantPack_.context);
    if (st != NO_ERROR) {
        return st;
    }
    GetOpSetupStatistic().runnerCreateTime += runnerCreateTimer.ElapsedMicroSecond();

    InitRunnerVariantPack(variantPack);
    // step1 runner setup
    Mki::Timer runnerSetupTimer;
    if (!runner_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "runner is null.";
        return ERROR_OPERATION_NULL_RUNNER;
    }

    hostTilingBuffer_ = runnerVariantPack_.context->GetHostTilingBuffer();
    if (!hostTilingBuffer_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "get host tiling buffer from contextbase is null";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    runnerVariantPack_.hostTilingBuffer = hostTilingBuffer_;
    runnerVariantPack_.tilingBufferSize = runnerVariantPack_.context->GetTilingBufferBlockSize();
#ifdef _DEBUG
    ATB_LOG(INFO) << GetLogPrefix()
                  << "get host tiling buffer from contextbase is:" << static_cast<void *>(hostTilingBuffer_)
                  << ", size:" << runnerVariantPack_.context->GetTilingBufferBlockSize();
#endif

    st = runner_->Setup(runnerVariantPack_);
    GetOpSetupStatistic().runnerSetupTime += runnerSetupTimer.ElapsedMicroSecond();
    if (st != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "runner setup fail";
        return st;
    }

    // step2, get runner tiling buffer size
    runnerVariantPack_.tilingBufferSize =
        static_cast<uint64_t>(TensorUtil::AlignInt(runner_->GetTilingBufferSize(), WORKSPACE_ALIGN));

    // step3, fill runner host tiling buffer
    FillHostTilingBuffer();

    // step4, get runner workspasce and inter
    // 计算所有流的workspaceBufferSize
    runnerVariantPack_.workspaceBufferSize =
        static_cast<uint64_t>(TensorUtil::AlignInt(GetTotalWorkspaceBufferSize(), WORKSPACE_ALIGN));

    runnerVariantPack_.intermediateBufferSize =
        static_cast<uint64_t>(TensorUtil::AlignInt(runner_->GetIntermediateBufferSize(), WORKSPACE_ALIGN));

    workspaceSize = runnerVariantPack_.workspaceBufferSize + runnerVariantPack_.intermediateBufferSize;
    workspaceSize_ = workspaceSize;

    ATB_LOG(INFO) << GetLogPrefix() << "setup success, workspaceSize:" << workspaceSize
                  << ", runner.tilingBufferSize:" << runnerVariantPack_.tilingBufferSize
                  << ", runner.workspaceBufferSize:" << runnerVariantPack_.workspaceBufferSize
                  << ", runner.intermediateBufferSize:" << runnerVariantPack_.intermediateBufferSize;

    return st;
}

void OperationBase::ProfilingPrepare()
{
    if (GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel0Status() && !isProfArrayInited_) {
        isProfArrayInited_ = true;
        hashIdArray_.resize(OPERATION_MAX);
        typeIdArray_.resize(OPERATION_MAX);
        std::string setupName = name_ + "::Setup";
        std::string executeName = name_ + "::Execute";
        std::string preLaunchName = name_ + "::PreLaunch";
        std::string launchName = name_ + "::Launch";
        RegProfArray(OPERATION_SETUP, setupName);
        RegProfArray(OPERATION_EXECUTE, executeName);
        RegProfArray(OPERATION_PRELAUNCH, preLaunchName);
        RegProfArray(OPERATION_LAUNCH, launchName);
    }
}

Status OperationBase::Setup(const VariantPack &variantPack, uint64_t &workspaceSize, Context *context)
{
    Status st = NO_ERROR;
    ProfilingPrepare();
    const uint64_t beginTime = GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel0Status() ?
                                   GetSingleton<Mki::ProfilingFuncs>().ProfSysCycleTime() :
                                   0;
    if (!context) {
        ATB_LOG(ERROR) << GetLogPrefix() << "context is null, setup fail";
        return ERROR_INVALID_PARAM;
    }
    if (context->GetLaunchMode() == GRAPH_LAUNCH_MODE) {
        ATB_LOG(INFO) << GetLogPrefix() << "run in GRAPH_LAUNCH_MODE";
        isGraphLaunchMode_ = true;
        st = GraphModeSetup(variantPack, workspaceSize, context);
    } else {
        ATB_LOG(INFO) << GetLogPrefix() << "run in KERNEL_LAUNCH_MODE";
        st = EagerModeSetup(variantPack, workspaceSize, context);
    }
    if (Probe::ReportOperationStatisticEnable()) {
        Probe::ReportOperationSetupStatistic(executeCount_, GenerateOperationName(name_, operationBaseIds_),
                                             GetOpSetupStatistic().ToString());
    }
    if (GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel0Status()) {
        ReportApiInfo(beginTime, OPERATION_SETUP);
    }
    return st;
}

Status OperationBase::EagerModeSetup(const VariantPack &variantPack, uint64_t &workspaceSize, Context *context)
{
    Mki::Timer totalTimer;
    Status st = SetupPrepare();
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "setup fail, error code: " << st;
        return st;
    }

#ifdef _DEBUG
    ATB_LOG(INFO) << GetLogPrefix() << "setup start, context:" << context << ", variantPack:\n"
                  << VariantPackToString(variantPack);
#else
    ATB_LOG(INFO) << GetLogPrefix() << "setup start, variantPack:\n" << VariantPackToString(variantPack);
#endif

    try {
        st = SetupCheck(variantPack, context);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "invalid param, setup check fail, error code: " << st;
            return st;
        }
        st = SetupThrow(variantPack, workspaceSize);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "setup fail, error code: " << st;
            return st;
        }
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) << GetLogPrefix() << "setup throw an exception: " << e.what();
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    executeCount_++;
    if (st == 0) {
        setUpSuccess_ = true;
    }
    GetOpSetupStatistic().totalTime += totalTimer.ElapsedMicroSecond();
    ATB_LOG(INFO) << GetLogPrefix() << "setup statistic:" << GetOpSetupStatistic().ToString();
    GetOpSetupStatistic().Reset();
    return st;
}

Status OperationBase::GraphModeSetup(const VariantPack &variantPack, uint64_t &workspaceSize, Context *context)
{
    if (!isCaptured_) {
        Status st = EagerModeSetup(variantPack, workspaceSize, context);
        return st;
    } else {
        if (!TensorUtil::IsRunnerVariantPackEqual(variantPack, runnerVariantPack_)) {
            ATB_LOG(ERROR) << "Tensor shape is not support to change in GRAPH_MODE";
            return ERROR_INVALID_TENSOR_DIM;
        }
        InitRunnerVariantPack(variantPack);
        return NO_ERROR;
    }
}

void OperationBase::RegProfArray(ProfilingFuncName profFuncType, std::string profName)
{
    if (profFuncType <= OPERATION_UNDEFINED || profFuncType >= OPERATION_MAX) {
        ATB_LOG(ERROR) << "type: " << profFuncType << "is not invalid ProfilingFuncName";
        return;
    }
    uint64_t profHashId = GetSingleton<Mki::ProfilingFuncs>().ProfGetHashId(profName.c_str(), profName.size());
    hashIdArray_.at(profFuncType) = profHashId;
    typeIdArray_.at(profFuncType) = GetTypeIdFromHash(profHashId);
    GetSingleton<Mki::ProfilingFuncs>().ProfReportTypeInfo(MSPROF_REPORT_ACL_LEVEL, typeIdArray_.at(profFuncType),
                                                           profName);
}

Status OperationBase::SetupPrepare()
{
    setUpSuccess_ = false;
    if (GetSingleton<Config>().IsworkspaceMemAllocGlobal()) {
        GetGlobalMemAllocationSolver()->Reset();
    }
    if (operationBaseIds_.at(0) == -1) {
        operationBaseIds_.at(0) = GetNewOperationBaseId();
        ResetLogPrefix();
        Status st = SetNodeOperationIds();
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "set node operation ids fail";
            return st;
        }
    }
    return NO_ERROR;
}

Status OperationBase::CopyTilingToDevice()
{
    ContextBase *contextBase = runnerVariantPack_.context;
    if (runnerVariantPack_.tilingBufferSize > contextBase->GetTilingBufferBlockSize()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "tilingSize is bigger than tilingBufferSize!";
        return ERROR_OUT_OF_HOST_MEMORY;
    }

    if (runnerVariantPack_.tilingBufferSize == 0) {
        return NO_ERROR;
    }

    aclrtStream executeStream = GetExecuteStream(runnerVariantPack_.context);
    aclrtStream tilingCopyStream = contextBase->GetAsyncTilingCopyStream();
    aclrtEvent tilingCopyEvent = nullptr;
    if (tilingCopyStream) {
        tilingCopyEvent = contextBase->GetAsyncTilingCopyEvent();
        ATB_LOG_IF(!tilingCopyEvent, ERROR) << GetLogPrefix() << "get copy event from contextbase fail";
    }

    Status st = NO_ERROR;
    if (tilingCopyStream && tilingCopyEvent) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "tiling copy stream is valid, use it to copy tiling to device";
        st = CopyHostTilingToDevice(tilingCopyStream);
        st = aclrtRecordEvent(tilingCopyEvent, tilingCopyStream) + st;
        st = aclrtStreamWaitEvent(executeStream, tilingCopyEvent) + st;
        st = aclrtResetEvent(tilingCopyEvent, executeStream) + st;
    } else {
        ATB_LOG(DEBUG) << GetLogPrefix() << "use execute stream to copy tiling to device";
        st = CopyHostTilingToDevice(executeStream);
    }

    return st;
}

template <typename TensorType>
Status OperationBase::ExecuteVariantPackInTensorCheck(const SVector<TensorType> &inTensors) const
{
    std::string Prefix = GetLogPrefix();
    if (inTensors.size() != runnerVariantPack_.inTensors.size()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "execute inTensors.size:" << inTensors.size()
                       << " != setup inTensors.size:" << runnerVariantPack_.inTensors.size();
        return ERROR_INVALID_PARAM;
    }
    SVector<bool> emptyInTensorPerms = GetEmptyInTensorPermissions();
    for (size_t i = 0; i < inTensors.size(); i++) {
        const Tensor &variantPackInTensor = inTensors.at(i);
        if (Prefix.find("WithStride") == std::string::npos && // "WithStride" indicates non continuous tensors
            variantPackInTensor.dataSize != Utils::GetTensorSize(runnerVariantPack_.inTensors.at(i).desc)) {
            ATB_LOG(ERROR) << GetLogPrefix() << "execute variantPack.inTensors(" << i
                           << ").dataSize is Not equal to the setup dataSize";
            return ERROR_INVALID_PARAM;
        }
        if (i < emptyInTensorPerms.size() && emptyInTensorPerms.at(i) &&
            TensorCheck::IsEmptyTensor(variantPackInTensor)) {
            continue;
        }
        if (!variantPackInTensor.deviceData && !variantPackInTensor.hostData) {
            return ERROR_INVALID_PARAM;
        }
    }
    return NO_ERROR;
}

template <typename TensorType>
Status OperationBase::ExecuteVariantPackOutTensorCheck(const SVector<TensorType> &outTensors) const
{
    std::string Prefix = GetLogPrefix();
    if (outTensors.size() != runnerVariantPack_.outTensors.size()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "execute outTensors.size:" << outTensors.size()
                       << " != setup outTensors.size:" << runnerVariantPack_.outTensors.size();
        return ERROR_INVALID_PARAM;
    }
    SVector<bool> emptyOutTensorPerms = GetEmptyOutTensorPermissions();
    for (size_t i = 0; i < outTensors.size(); i++) {
        const Tensor &variantPackOutTensor = outTensors.at(i);
        if (variantPackOutTensor.dataSize != Utils::GetTensorSize(runnerVariantPack_.outTensors.at(i).desc)) {
            ATB_LOG(ERROR) << GetLogPrefix() << "execute variantPack.outTensors(" << i
                           << ").dataSize is Not equal to the setup dataSize";
            return ERROR_INVALID_PARAM;
        }
        if (i < emptyOutTensorPerms.size() && emptyOutTensorPerms.at(i) &&
            TensorCheck::IsEmptyTensor(variantPackOutTensor)) {
            continue;
        }
        if (!variantPackOutTensor.deviceData && !variantPackOutTensor.hostData) {
            ATB_LOG(ERROR) << GetLogPrefix() << "execute variantPack.outTensors(" << i
                           << ") deviceData&hostData is null";
            return ERROR_INVALID_PARAM;
        }
    }
    return NO_ERROR;
}

Status OperationBase::ExecuteVariantPackCheck(const VariantPack &variantPack) const
{
    Status st = NO_ERROR;
    st = ExecuteVariantPackInTensorCheck(variantPack.inTensors);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "ExecuteVariantPackCheck for inTensor failed, error code: " << st;
        return st;
    }
    st = ExecuteVariantPackOutTensorCheck(variantPack.outTensors);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "ExecuteVariantPackCheck for outTensor failed, error code: " << st;
        return st;
    }
    return st;
}

Status OperationBase::ExecuteCheck(const VariantPack &variantPack, const uint8_t *workspace, uint64_t workspaceSize,
                                   Context *context)
{
    if (GetExecuteStream(runnerVariantPack_.context) == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "execute stream is null";
        return ERROR_INVALID_PARAM;
    }
    if (context != runnerVariantPack_.context) {
        ATB_LOG(ERROR) << GetLogPrefix() << "execute context is same with setup context";
        return ERROR_INVALID_PARAM;
    }

    if (!runner_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "runner is null.";
        return ERROR_OPERATION_NULL_RUNNER;
    }

    if (workspaceSize < workspaceSize_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "execute workspaceSize:" << workspaceSize
                       << " < setup workspaceSize:" << workspaceSize_;
        return ERROR_INVALID_PARAM;
    }

    if (workspaceSize_ != 0 && workspace == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "execute workspace is can't be null when workspaceSize > 0";
        return ERROR_INVALID_PARAM;
    }
    if (variantPack.inTensors.size() > 0 || variantPack.outTensors.size() > 0) {
        Status st = ExecuteVariantPackCheck(variantPack);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "CheckExecuteVariantPack failed!";
            return st;
        }
    }

    return NO_ERROR;
}

Status OperationBase::PreExecuteThrow(const VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize)
{
    ATB_LOG(INFO) << GetLogPrefix() << "execute start, workspaceSize:" << workspaceSize
                  << ", variantPack:" << VariantPackToString(variantPack);
#ifdef _DEBUG
    ATB_LOG(INFO) << "workspace:" << static_cast<void *>(workspace);
#endif

    UpdateTensorData(variantPack, workspace);

    Status st = NO_ERROR;
    if (!(runnerVariantPack_.context->GetLaunchWithTilingStatus())) {
        st = CopyTilingToDevice();
        if (st != 0) {
            return st;
        }
    }
#ifdef _DEBUG
    ATB_LOG(INFO) << GetLogPrefix() << "PreExecute " << runner_->GetName() << "_" << runner_.get() << " start";
#else
    ATB_LOG(INFO) << GetLogPrefix() << "PreExecute " << runner_->GetName() << " start";
#endif
    st = runner_->PreExecute(runnerVariantPack_);
    if (st != 0) {
#ifdef _DEBUG
        ATB_LOG(ERROR) << GetLogPrefix() << "PreExecute " << runner_->GetName() << "_" << runner_.get() << " fail";
#else
        ATB_LOG(ERROR) << GetLogPrefix() << "PreExecute " << runner_->GetName() << " fail";
#endif
        return st;
    }
    return NO_ERROR;
}

Status OperationBase::PreLaunch(const VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                                Context *context)
{
    if (!context) {
        ATB_LOG(ERROR) << GetLogPrefix() << "context is null, PreLaunch fail";
        return ERROR_INVALID_PARAM;
    }
    if (context->GetLaunchMode() == GRAPH_LAUNCH_MODE) {
        isGraphLaunchMode_ = true;
        return GraphModePreLaunch(variantPack, workspace, workspaceSize, context);
    } else {
        return EagerModePreLaunch(variantPack, workspace, workspaceSize, context);
    }
}

Status OperationBase::EagerModePreLaunch(const VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                                         Context *context)
{
    if (!setUpSuccess_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "setup failed, execute exit.";
        return ERROR_INVALID_PARAM;
    }
    Mki::Timer preLaunchTime;
    Status st = NO_ERROR;
    try {
        st = ExecuteCheck(variantPack, workspace, workspaceSize, context);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "invalid param, execute check fail, error code: " << st;
            return st;
        }
        st = PreExecuteThrow(variantPack, workspace, workspaceSize);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "execute fail, error code: " << st;
            return st;
        }
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) << GetLogPrefix() << "execute throw an exception: " << e.what();
        return ERROR_RT_FAIL;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "PreExecute " << runner_->GetName() << " end";
    GetOpExecuteStatistic().preLaunchTime += preLaunchTime.ElapsedMicroSecond();
    return st;
}

Status OperationBase::GraphModePreLaunch(const VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                                         Context *context)
{
    Status st = NO_ERROR;
    lastWorkspaceAddr_ = reinterpret_cast<void *>(workspace);
    if (!isCaptured_) {
        argsBufferSize_ = runner_->GetArgsSize();
        if (deviceArgsBuffer_ == nullptr) {
            // 调用Context分配deviceArgsBuffer_
            deviceArgsBuffer_ = runnerVariantPack_.context->GetArgsDeviceBuffer(argsBufferSize_);
        }

        if (hostArgsBuffer_ == nullptr) {
            // 调用Context分配hostArgsBuffer_
            hostArgsBuffer_ = runnerVariantPack_.context->GetArgsHostBuffer(argsBufferSize_);
        }

        runnerVariantPack_.argsDeviceBuffer = reinterpret_cast<uint8_t *>(deviceArgsBuffer_);
        runnerVariantPack_.argsHostBuffer = reinterpret_cast<uint8_t *>(hostArgsBuffer_);
        st = EagerModePreLaunch(variantPack, workspace, workspaceSize, context);
        ATB_CHECK(st == 0, "EagerModePreLaunch failed!", return st);
        // 上述的preLaunch当前只做了地址更新及tiling拷贝，后续考虑改名。当前为保证已有代码的质量不修改原有流程，后续需要考虑EAGER_MODE与GRAPH_MODE的流程归纳与复用。
    } else {
        ATB_LOG(INFO) << GetLogPrefix() << "begin update tensor addr.";
        if (workspace == runnerVariantPack_.workspaceBuffer) {
            // 如果workspace地址没有变化，intermediateBuffer作为偏移量为0
            runnerVariantPack_.intermediateBuffer = nullptr;
        } else if (workspace > runnerVariantPack_.workspaceBuffer) {
            // 如果workspace发生了变化，计算workspace变化带来的偏移量时需要再加上workspaceBufferSize才是中间tensor对应内存的起始地址
            runnerVariantPack_.intermediateBuffer = workspace -
            reinterpret_cast<uint64_t>(runnerVariantPack_.workspaceBuffer) + runnerVariantPack_.workspaceBufferSize;
#ifdef _DEBUG
            ATB_LOG(INFO) << GetLogPrefix() << "changing the old workspace: " << static_cast<void *>(runnerVariantPack_.workspaceBuffer)
                          << " to new workspace: " << static_cast<void *>(workspace) << ", and the runnerVariantPack_.intermediateBuffer: "
                          << static_cast<void *>(runnerVariantPack_.intermediateBuffer);
#endif
            runnerVariantPack_.workspaceBuffer = workspace;
            // 刷新图中所有node的workpsaceBuffer
            st = runner_->UpdateWorkspaceBuffer(runnerVariantPack_);
        } else {
            runnerVariantPack_.intermediateBuffer = runnerVariantPack_.workspaceBuffer -
            reinterpret_cast<uint64_t>(workspace) + runnerVariantPack_.workspaceBufferSize;
#ifdef _DEBUG
            ATB_LOG(INFO) << GetLogPrefix() << "changing the old workspace: " << static_cast<void *>(runnerVariantPack_.workspaceBuffer)
                          << " to new workspace: " << static_cast<void *>(workspace) << ", and the runnerVariantPack_.intermediateBuffer: "
                          << static_cast<void *>(runnerVariantPack_.intermediateBuffer);
#endif
            runnerVariantPack_.workspaceBuffer = workspace;
            st = runner_->UpdateWorkspaceBuffer(runnerVariantPack_);
        }
        st = runner_->UpdateTensorAddr(runnerVariantPack_);
        ATB_CHECK(st == 0, GetLogPrefix() + "UpdateTensorAddr failed!", return st);
        FillHostTilingBuffer();
        st = CopyTilingToDevice();
        ATB_CHECK(st == 0, GetLogPrefix() + "CopyTilingToDevice failed!", return st);
    }
    st = runner_->BuildArgs();
    ATB_CHECK(st == 0, GetLogPrefix() + "BuildArgs failed!", return st);

    st = CopyArgsToDevice(context);
    ATB_CHECK(st == 0, GetLogPrefix() + "CopyArgsToDevice failed!", return st);

    return st;
}

Status OperationBase::Launch()
{
    Status st = NO_ERROR;
    if (runnerVariantPack_.context->GetLaunchMode() == GRAPH_LAUNCH_MODE) {
        isGraphLaunchMode_ = true;
        aclmdlRI tmpModel = nullptr;
        st = aclmdlRICaptureGetInfo(GetExecuteStream(runnerVariantPack_.context), &streamStatus_, &tmpModel);
        if (tmpModel != nullptr) {
            model_ = tmpModel;
        }
        ATB_LOG_IF(st != 0, ERROR) << GetLogPrefix() << "aclmdlRICaptureGetInfo failed! ret:" << st;
        return GraphModeLaunch();
    } else {
        return EagerModeLaunch();
    }
}

Status OperationBase::EagerModeLaunch()
{
    Mki::Timer ExecuteTime;
    void *executeStream = GetExecuteStream(runnerVariantPack_.context);
#ifdef _DEBUG
    ATB_LOG(INFO) << GetLogPrefix() << "execute " << runner_->GetName() << "_" << runner_.get() << " start";
    ATB_LOG(INFO) << "stream: " << executeStream;
#else
    ATB_LOG(INFO) << GetLogPrefix() << "execute " << runner_->GetName() << " start";
#endif
    Status st = NO_ERROR;
    try {
        st = runner_->Execute(runnerVariantPack_);
        if (st != NO_ERROR) {
#ifdef _DEBUG
            ATB_LOG(ERROR) << GetLogPrefix() << "execute " << runner_->GetName() << "_" << runner_.get() << " fail";
#else
            ATB_LOG(ERROR) << GetLogPrefix() << "execute " << runner_->GetName() << " fail";
#endif
        }
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) << GetLogPrefix() << "execute throw an exception: " << e.what();
        return ERROR_RT_FAIL;
    }
#ifdef _DEBUG
    ATB_LOG(INFO) << GetLogPrefix() << "execute " << runner_->GetName() << "_" << runner_.get() << " success";
#else
    ATB_LOG(INFO) << GetLogPrefix() << "execute " << runner_->GetName() << " success";
#endif
    if (GetSingleton<Config>().IsStreamSyncEveryOperationEnable()) {
        int ret = aclrtSynchronizeStream(executeStream);
        ATB_LOG_IF(ret != 0, ERROR) << GetLogPrefix() << "stream sync fail, ret:" << ret;
    }
    GetOpExecuteStatistic().launchTime += ExecuteTime.ElapsedMicroSecond();
    GetOpExecuteStatistic().totalTime += GetOpExecuteStatistic().preLaunchTime + GetOpExecuteStatistic().launchTime;
    ATB_LOG(INFO) << GetLogPrefix() << "execute statistic:" << GetOpExecuteStatistic().ToString();
    return st;
}

Status OperationBase::GraphModeLaunch()
{
    Status st = NO_ERROR;
    aclrtStream executeStream = GetExecuteStream(runnerVariantPack_.context);
    if (streamStatus_ == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
        isCaptured_ = true;
        st = EagerModeLaunch();
        ATB_LOG_IF(st != 0, ERROR) << GetLogPrefix() << "EagerModeLaunch failed! ret:" << st;
        return st;
    }

    if (!isCaptured_) {
        st = aclmdlRICaptureBegin(executeStream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL);
        ATB_LOG_IF(st != 0, ERROR) << GetLogPrefix() << "aclmdlRICaptureBegin failed! ret:" << st;
        st = EagerModeLaunch();
        ATB_LOG_IF(st != 0, ERROR) << GetLogPrefix() << "EagerModeLaunch failed! ret:" << st;
        st = aclmdlRICaptureEnd(executeStream, &model_);
        ATB_LOG_IF(st != 0, ERROR) << GetLogPrefix() << "aclmdlRICaptureEnd failed! ret:" << st;
    }

    isCaptured_ = true;
    st = aclmdlRIExecuteAsync(model_, executeStream);
    ATB_LOG_IF(st != 0, ERROR) << GetLogPrefix() << "aclmdlRIExecuteAsync failed! ret:" << st;
    return st;
}

Status OperationBase::Execute(const VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                              Context *context)
{
    const uint64_t beginTime = GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel0Status() ?
                                   GetSingleton<Mki::ProfilingFuncs>().ProfSysCycleTime() :
                                   0;
    ExecuteType executeType = context->GetExecuteType();
    ProfilingFuncName profType = executeType == EXECUTE_NORMAL ?
                                     OPERATION_EXECUTE :
                                     (executeType == EXECUTE_PRELAUNCH ? OPERATION_PRELAUNCH : OPERATION_LAUNCH);
    std::shared_ptr<MstxMemRegister> mstxMemRegister{nullptr};
    if (workspaceSize && MstxMemRegister::IsMstxEnable()) {
        mstxMemRegister = std::make_shared<MstxMemRegister>();
        if (mstxMemRegister->MstxHeapRegister(workspace, workspaceSize) == NO_ERROR) {
            runnerVariantPack_.mstxMemRegister = mstxMemRegister.get();
            ATB_LOG(INFO) << GetLogPrefix() << "mstxMemHeapRegister success ";
        }
    }
    Status st = NO_ERROR;
    if (executeType == EXECUTE_NORMAL || executeType == EXECUTE_PRELAUNCH) {
        st = PreLaunch(variantPack, workspace, workspaceSize, context);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "PreLaunch fail, error code: " << st;
            return st;
        }
    }
    if (executeType == EXECUTE_NORMAL || executeType == EXECUTE_LAUNCH) {
        st = Launch();
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Launch fail, error code: " << st;
            return st;
        }
    }
    if (GetSingleton<Mki::ProfilingFuncs>().GetProfilingLevel0Status()) {
        ReportApiInfo(beginTime, profType);
    }
    GetOpExecuteStatistic().Reset();
    return st;
}

void OperationBase::Reset()
{
    workspaceSize_ = 0;
    if (Probe::IsSaveTensorDesc()) {
        SetSaveTensorDir();
    }
    runnerVariantPack_.hostTilingBuffer = nullptr;
    runnerVariantPack_.tilingBuffer = nullptr;
    runnerVariantPack_.tilingBufferSize = 0;
    runnerVariantPack_.workspaceBuffer = nullptr;
    runnerVariantPack_.workspaceBufferSize = 0;
    runnerVariantPack_.intermediateBuffer = nullptr;
    runnerVariantPack_.intermediateBufferSize = 0;
}

Status OperationBase::CopyHostTilingToDevice(aclrtStream stream)
{
    if (runnerVariantPack_.tilingBufferSize != 0) {
        UpdateCurrentOpTiling(runnerVariantPack_.tilingBuffer, runnerVariantPack_.tilingBufferSize);

        ATB_LOG(DEBUG) << GetLogPrefix() << "copy host tiling to device start, totalTilingBufferSize:"
                       << runnerVariantPack_.tilingBufferSize;
        Mki::Timer timer;
        if (hostTilingBuffer_ == nullptr) {
            ATB_LOG(ERROR) << GetLogPrefix() << "host tiling buffer is null!";
            return ERROR_OUT_OF_HOST_MEMORY;
        }
        int ret = 0;
        if (runnerVariantPack_.context->GetLaunchMode() == GRAPH_LAUNCH_MODE && !isCaptured_) {
            ret = aclrtMemcpy(runnerVariantPack_.tilingBuffer, runnerVariantPack_.tilingBufferSize, hostTilingBuffer_,
                              runnerVariantPack_.tilingBufferSize, ACL_MEMCPY_HOST_TO_DEVICE);
        } else {
            ret = aclrtMemcpyAsync(runnerVariantPack_.tilingBuffer, runnerVariantPack_.tilingBufferSize,
                                   hostTilingBuffer_, runnerVariantPack_.tilingBufferSize, ACL_MEMCPY_HOST_TO_DEVICE,
                                   stream);
        }

        GetOpExecuteStatistic().tillingCopyTime += timer.ElapsedMicroSecond();
        if (ret != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "copy host tiling to device fail, ret:" << ret;
            return ERROR_RT_FAIL;
        }
        ATB_LOG(DEBUG) << GetLogPrefix() << "copy host tiling to device success";
    } else {
        UpdateCurrentOpTiling(nullptr, 0);
    }
    return NO_ERROR;
}

void OperationBase::InitProInfo()
{
    if (!isProfArrayInited_) {
        isProfArrayInited_ = true;
        hashIdArray_.resize(OPERATION_MAX);
        typeIdArray_.resize(OPERATION_MAX);
        std::string setupName = name_ + "::Setup";
        std::string executeName = name_ + "::Execute";
        std::string preLaunchName = name_ + "::PreLaunch";
        std::string launchName = name_ + "::Launch";
        RegProfArray(OPERATION_SETUP, setupName);
        RegProfArray(OPERATION_EXECUTE, executeName);
        RegProfArray(OPERATION_PRELAUNCH, preLaunchName);
        RegProfArray(OPERATION_LAUNCH, launchName);
    }
}

void OperationBase::ReportApiInfo(const uint64_t beginTime, ProfilingFuncName type)
{
    const uint64_t endTime = GetSingleton<Mki::ProfilingFuncs>().ProfSysCycleTime();
    InitProInfo();
    MsProfApi info{};
    if (type <= OPERATION_UNDEFINED || type >= OPERATION_MAX) {
        ATB_LOG(ERROR) << "type: " << type << "is not invalid ProfilingFuncName";
        return;
    }
    info.type = typeIdArray_.at(type);
    info.itemId = hashIdArray_.at(type);
    info.level = MSPROF_REPORT_ACL_LEVEL;
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
        ATB_LOG(ERROR) << "ProfReportApi error! Return value:" << ret;
    }
}

std::string OperationBase::GetSaveTensorRootDir() const
{
    int32_t deviceId = -1;
    aclError aclRet = aclrtGetDevice(&deviceId);
    if (aclRet != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "get device id error!";
        deviceId = -1;
    }
    int32_t pid = UtilsInternal::GetCurrentProcessId();
    return std::to_string(deviceId) + "_" + std::to_string(pid);
}

std::string OperationBase::GetLogPrefix() const
{
    return logPrefix_;
}

void OperationBase::FillHostTilingBuffer()
{
    if (runnerVariantPack_.tilingBufferSize != 0) {
        if (runnerVariantPack_.tilingBufferSize > runnerVariantPack_.context->GetTilingBufferBlockSize()) {
            ATB_LOG(ERROR) << GetLogPrefix() << "runner's tiling size:" << runnerVariantPack_.tilingBufferSize
                           << " is bigger than tiling block size:"
                           << runnerVariantPack_.context->GetTilingBufferBlockSize();
            return;
        }
        if (!hostTilingBuffer_) {
            ATB_LOG(ERROR) << GetLogPrefix() << " tiling buffer filled is null";
            return;
        }

        Mki::Timer runnerFillHostTilingTimer;
        Status st = runner_->FillHostTilingBuffer(hostTilingBuffer_, runnerVariantPack_.tilingBufferSize, runnerVariantPack_.context);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "fill host tiling buffer fail";
            return;
        }
        GetOpSetupStatistic().runnerFillHostTilingTime += runnerFillHostTilingTimer.ElapsedMicroSecond();
    }
}

uint64_t OperationBase::GetTotalWorkspaceBufferSize()
{
    uint64_t totalWorkspaceBufferSize = 0;
    for (size_t i = 0; i < runner_->GetWorkspaceBufferSize().size(); i++) {
        totalWorkspaceBufferSize += runner_->GetWorkspaceBufferSize().at(i);
        ATB_LOG(INFO) << GetLogPrefix() << "runner.workspaceBufferSize: " << runner_->GetWorkspaceBufferSize().at(i)
                      << " at streamId: " << i;
    }
    return totalWorkspaceBufferSize;
}

void OperationBase::UpdateTensorData(const VariantPack &variantPack, uint8_t *workspace)
{
    uint8_t *deviceTilingBuffer = runnerVariantPack_.context->GetDeviceTilingBuffer();
    if (!deviceTilingBuffer) {
        ATB_LOG(ERROR) << GetLogPrefix() << "get device tiling buffer from contextbase fail";
        return;
    }
#ifdef _DEBUG
    ATB_LOG(INFO) << GetLogPrefix() << "get device tiling buffer from contextbase success, buffer:"
                  << static_cast<void *>(deviceTilingBuffer);
#endif

    runnerVariantPack_.tilingBuffer = deviceTilingBuffer;
    runnerVariantPack_.workspaceBuffer = workspace;
    runnerVariantPack_.intermediateBuffer = workspace + runnerVariantPack_.workspaceBufferSize;
    if (variantPack.inTensors.size() > 0 || variantPack.outTensors.size() > 0) {
        TensorUtil::FastCopyTensorsData(variantPack.inTensors, runnerVariantPack_.inTensors);
        TensorUtil::FastCopyTensorsData(variantPack.outTensors, runnerVariantPack_.outTensors);
    }
}

std::string OperationBase::VariantPackToString(const VariantPack &variantPack) const
{
    std::stringstream ss;
    for (size_t i = 0; i < variantPack.inTensors.size(); ++i) {
        ss << "inTensors[" << i << "]: " << TensorUtil::TensorToString(variantPack.inTensors.at(i)) << std::endl;
    }

    for (size_t i = 0; i < variantPack.outTensors.size(); ++i) {
        ss << "outTensors[" << i << "]: " << TensorUtil::TensorToString(variantPack.outTensors.at(i)) << std::endl;
    }

    return ss.str();
}

void OperationBase::ResetLogPrefix()
{
    std::stringstream ss;
    ss << name_;
    for (size_t i = 0; i < operationBaseIds_.size(); i++) {
        ss << "_" << operationBaseIds_.at(i);
    }
    ss << " ";
    logPrefix_ = ss.str();
}

nlohmann::json OperationBase::GetParamJson() const
{
    nlohmann::json emptyJson;
    return emptyJson;
}

nlohmann::json OperationBase::GetGraphInfo() const
{
    nlohmann::json graphJson;
    graphJson["opType"] = name_;
    graphJson["opName"] = GenerateOperationName(name_, operationBaseIds_);
    graphJson["inTensorNum"] = GetInputNum();
    graphJson["outTensorNum"] = GetOutputNum();
    GetGraphInfoImpl(graphJson);

    return graphJson;
}

void OperationBase::GetGraphInfoImpl(nlohmann::json &graphJson) const
{
    graphJson["param"] = GetParamJson();
}

const std::vector<int64_t> &OperationBase::GetOperationBaseIds()
{
    return operationBaseIds_;
}

void OperationBase::SetExecuteStreamId(uint32_t streamId)
{
    streamId_ = streamId;
}

uint32_t OperationBase::GetExecuteStreamId() const
{
    return streamId_;
}

aclrtStream OperationBase::GetExecuteStream(Context *context) const
{
    if (context == nullptr) {
        ATB_LOG(ERROR) << "context is nullptr";
        return nullptr;
    }
    std::vector<aclrtStream> streams = context->GetExecuteStreams();
    if (streams.size() < (streamId_ + 1)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "streamId is bigger than actual stream number,"
                       << "actual stream number is " << streams.size() << ", streamId is " << streamId_;
        return nullptr;
    }
    return streams.at(streamId_);
}

Status OperationBase::CopyArgsToDevice(Context *context) const
{
    Status st = NO_ERROR;
    if (hostArgsBuffer_ == nullptr) {
        ATB_LOG(INFO) << "hostArgsBuffer is nullptr, no need to copy args";
        return st;
    }
#ifdef _DEBUG
    ATB_LOG(DEBUG) << GetLogPrefix() << "args in graphMode is:";
    const size_t counter =  argsBufferSize_ / sizeof(void *);
    for (size_t i = 0; i < counter; i++) {
        ATB_LOG(DEBUG) << ((void **)(hostArgsBuffer_))[i];
    }
#endif
    if (deviceArgsBuffer_ == nullptr) {
        ATB_LOG(INFO) << "deviceArgsBuffer is nullptr, no need to copy args";
        return st;
    }
    if (!isCaptured_) {
        st = aclrtMemcpy(deviceArgsBuffer_, argsBufferSize_, hostArgsBuffer_, argsBufferSize_,
                         ACL_MEMCPY_HOST_TO_DEVICE);
        ATB_LOG_IF(st != 0, ERROR) << GetLogPrefix() << "aclrtMemcpy failed! ret:" << st;
    } else {
        st = aclrtMemcpyAsync(deviceArgsBuffer_, argsBufferSize_, hostArgsBuffer_, argsBufferSize_,
                              ACL_MEMCPY_HOST_TO_DEVICE, GetExecuteStream(context));
        ATB_LOG_IF(st != 0, ERROR) << GetLogPrefix() << "aclrtMemcpyAsync failed! ret:" << st;
    }
    return st;
}
} // namespace atb
