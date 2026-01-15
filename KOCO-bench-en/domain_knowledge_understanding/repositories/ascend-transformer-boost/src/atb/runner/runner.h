/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_RUNNER_H
#define ATB_RUNNER_H
#include <string>
#include <mki/utils/any/any.h>
#include "atb/utils/runner_variant_pack.h"

namespace atb {
class Runner {
public:
    explicit Runner(const std::string &name);
    virtual ~Runner();
    std::string GetName() const;
    std::string GetOperationName() const;
    Status Setup(RunnerVariantPack &runnerVariantPack);
    uint64_t GetTilingBufferSize();
    Status FillHostTilingBuffer(uint8_t *hostTilingBuffer, uint64_t tilingBufferSize, ContextBase *context);
    virtual std::vector<uint64_t> &GetWorkspaceBufferSize();
    uint64_t GetIntermediateBufferSize();
    Status Execute(RunnerVariantPack &runnerVariantPack);
    Status PreExecute(RunnerVariantPack &runnerVariantPack);
    void SetRunnerInfo(const std::string &operationName, const std::vector<int64_t> &operationIds);
    void SetRunnerOperation(Operation *operation);
    virtual void SetParam(const Mki::Any &param);
    virtual bool IsSupportGlbWorkspace();
    aclrtStream GetExecuteStream(Context *context) const;
    virtual uint64_t GetArgsSize();
    virtual Status BuildArgs();
    virtual Status UpdateTensorAddr(RunnerVariantPack &runnerVariantPack);
    virtual Status UpdateWorkspaceBuffer(RunnerVariantPack &runnerVariantPack);

protected:
    virtual void SetSaveTensorDir(const std::string &tensorDir);
    std::string GetSaveTensorDir() const;
    bool IsSaveTensor() const;
    std::string GetLogPrefix() const;
    virtual void ChangeWorkspaceBufferByExecuteStream(RunnerVariantPack &runnerVariantPack);
    friend class GraphRunner;
    friend class OperationBase;

protected:
    std::vector<int64_t> runnerIds_;
    size_t executeCount_ = 0;
    size_t setupCount_ = 0;
    std::string tensorDir_;
    bool isParamUpdated_ = false;
    Operation *operation_ = nullptr;
    std::string logPrefix_;
    std::vector<uint64_t> multiStreamWorkspaceSizes_;

private:
    virtual Status SetupImpl(RunnerVariantPack &runnerVariantPack);
    virtual uint64_t GetTilingBufferSizeImpl();
    virtual Status FillHostTilingBufferImpl(uint8_t *hostTilingBuffer, uint64_t tilingBufferSize,
                                            ContextBase *context);
    virtual uint64_t GetWorkspaceBufferSizeImpl();
    virtual uint64_t GetIntermediateBufferSizeImpl();
    virtual Status ExecuteImpl(RunnerVariantPack &runnerVariantPack);
    virtual Status PreExecuteImpl(RunnerVariantPack &runnerVariantPack);
    void DumpIOTensorInfo(RunnerVariantPack &runnerVariantPack) const;

private:
    std::string name_;
    std::string operationName_;
    bool saveTensorFlag_ = false;
};
} // namespace atb
#endif