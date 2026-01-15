/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_OPERATIONBASE_H
#define ATB_OPERATIONBASE_H
#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <nlohmann/json.hpp>
#include <acl/acl_mdl.h>
#include "mki/utils/operationir/operation_ir_cfg.h"
#include "atb/operation.h"
#include "atb/runner/runner.h"
#include "atb/utils/runner_variant_pack.h"
#include "atb/context.h"

namespace atb {
enum ProfilingFuncName : int {
    OPERATION_UNDEFINED = -1,
    OPERATION_SETUP,
    OPERATION_EXECUTE,
    OPERATION_PRELAUNCH,
    OPERATION_LAUNCH,
    OPERATION_MAX
};

class OperationBase : public Operation {
public:
    explicit OperationBase(const std::string &name);
    ~OperationBase() override;
    std::string GetName() const override;
    Status InferShape(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    Status Setup(const VariantPack &variantPack, uint64_t &workspaceSize, Context *context) override;
    Status Execute(const VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                   Context *context) override;
    Status SetOperationBaseIds(const std::vector<int64_t> &operationBaseIds, const int64_t nodeId);
    virtual nlohmann::json GetParamJson() const;
    const std::vector<int64_t> &GetOperationBaseIds();
    virtual std::shared_ptr<Runner> CreateRunner(Context &context) const = 0;
    virtual void SetExecuteStreamId(uint32_t streamId);
    virtual uint32_t GetExecuteStreamId() const;
    aclrtStream GetExecuteStream(Context *context) const;

protected:
    virtual Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                  SVector<TensorDesc> &outTensorDescs) const = 0;
    virtual Status InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const;
    virtual Status SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const;
    void InitEmptyInTensorPerms() const;
    void InitEmptyOutTensorPerms() const;
    virtual SVector<bool> GetEmptyInTensorPermissions() const;
    virtual SVector<bool> GetEmptyOutTensorPermissions() const;
    std::string GetLogPrefix() const;
    virtual Status SetNodeOperationIds();
    nlohmann::json GetGraphInfo() const;
    virtual void GetGraphInfoImpl(nlohmann::json &graphJson) const;
    friend class GraphOperation;

protected:
    std::string name_;
    std::vector<int64_t> operationBaseIds_;
    Mki::OperationIr *operationIr_ = nullptr;
    mutable SVector<bool> emptyInTensorPerms_;
    mutable SVector<bool> emptyOutTensorPerms_;
    RunnerVariantPack runnerVariantPack_;
    std::shared_ptr<Runner> runner_;

private:
    void Reset();
    Status InferShapeCheck(const SVector<TensorDesc> &inTensorDescs) const;
    Status InferShapeThrow(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const;
    Status SetupCheck(const VariantPack &variantPack, Context *context);
    Status SetupPrepare();
    Status SetupThrowPrepare(uint64_t &workspaceSize, Context *context);
    Status PreExecuteThrow(const VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize);
    Status PreLaunch(const VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize, Context *context);
    Status Launch();
    Status SetupThrow(const VariantPack &variantPack, uint64_t &workspaceSize);
    Status ExecuteCheck(const VariantPack &variantPack, const uint8_t *workspace, uint64_t workspaceSize,
                        Context *context);
    template <typename TensorType> Status ExecuteVariantPackInTensorCheck(const SVector<TensorType> &inTensors) const;
    template <typename TensorType> Status ExecuteVariantPackOutTensorCheck(const SVector<TensorType> &outTensors) const;
    Status ExecuteVariantPackCheck(const VariantPack &variantPack) const;
    void InitRunnerVariantPack(const VariantPack &variantPack);
    Status CopyHostTilingToDevice(aclrtStream stream);
    Status CopyTilingToDevice();
    void InitProInfo();
    void ReportApiInfo(const uint64_t beginTime, ProfilingFuncName type);
    std::string GetSaveTensorRootDir() const;
    void FillHostTilingBuffer();
    Status CheckVariantPack(const VariantPack &variantPack) const;
    template <typename TensorType> Status CheckInTensor(const SVector<TensorType> &inTensors) const;
    template <typename TensorType> Status CheckOutTensor(const SVector<TensorType> &outTensors) const;
    bool CheckIniMatch(const SVector<TensorDesc> &inTensorDescs) const;
    bool CheckIniMatch(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const;
    void SetSaveTensorDir();
    void UpdateTensorData(const VariantPack &variantPack, uint8_t *workspace);
    std::string VariantPackToString(const VariantPack &variantPack) const;
    Status CreateRunnerFunc(Context *context);
    void ResetLogPrefix();
    void RegProfArray(ProfilingFuncName profFuncType, std::string profName);
    uint64_t GetTotalWorkspaceBufferSize();
    Status EagerModeSetup(const VariantPack &variantPack, uint64_t &workspaceSize, Context *context);
    Status GraphModeSetup(const VariantPack &variantPack, uint64_t &workspaceSize, Context *context);
    Status GraphModePreLaunch(const VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                                Context *context);
    Status EagerModePreLaunch(const VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                                Context *context);
    Status EagerModeLaunch();
    Status GraphModeLaunch();
    void ProfilingPrepare();
    Status CopyArgsToDevice(Context *context) const;

private:
    std::string logPrefix_;
    std::atomic_bool setUpSuccess_{false};
    uint8_t *hostTilingBuffer_ = nullptr;
    std::vector<uint64_t> hashIdArray_;
    std::vector<uint32_t> typeIdArray_;
    size_t executeCount_ = 0;
    uint64_t workspaceSize_ = 0;
    bool isProfArrayInited_ = false;
    uint32_t streamId_ = 0;
    aclmdlRI model_ = nullptr;
    uint64_t argsBufferSize_ = 0;
    void *deviceArgsBuffer_ = nullptr;
    void *hostArgsBuffer_ = nullptr;
    aclmdlRICaptureStatus streamStatus_ = ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED;
    void *lastWorkspaceAddr_ = nullptr;
    bool isCaptured_ = false;
    bool isGraphLaunchMode_ = false;  // 规避先调用DestroyContext再调用DestroyOperation的core问题
};
} // namespace atb
#endif