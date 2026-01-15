/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_HCCL_RUNNER_H
#define ATB_HCCL_RUNNER_H
#include <hccl/hccl_types.h>
#include <mki/utils/share_memory/share_memory.h>
#include "atb/runner/runner_type.h"
#include "atb/runner/runner.h"
#include "atb/infer_op_params.h"

namespace atb {
struct CommInitInfo {
    int signal = 0;
    HcclRootInfo hcclRootInfo = {};
    bool barrier[1]; // Flexible array member
};

using HcclCommSharedPtr = std::shared_ptr<void>;

class HcclRunner : public Runner {
public:
    explicit HcclRunner(const std::string &name, RunnerType runnerType = RUNNER_TYPE_UNDEFINED, int rank = 0,
                        int rankSize = 0, int rankRoot = 0, const std::string &commDomain = "");
    explicit HcclRunner(const std::string &name, RunnerType runnerType = RUNNER_TYPE_UNDEFINED, int rank = 0,
                        const std::string &rankTableFile = "", const std::string &commDomain = "");
    HcclRunner(const std::string &name, HcclComm hcclComm, RunnerType runnerType = RUNNER_TYPE_UNDEFINED);
    ~HcclRunner() override;
    HcclCommSharedPtr GetHcclCommSharedPtr() const;

protected:
    Status ExecuteImpl(RunnerVariantPack &runnerVariantPack) override;

protected:
    RunnerType runnerType_ = RUNNER_TYPE_UNDEFINED;
    int rank_ = 0;
    int rankSize_ = 0;
    int rankRoot_ = 0;
    std::string allReduceType_ = "sum";
    HcclCommSharedPtr hcclComm_;
    HcclRootInfo hcclRootInfo_ = {};
    std::string rankTableFile_;
    bool useRankTableFile_ = false;
    std::string commDomain_;

private:
    void Init();
    HcclCommSharedPtr CreateHcclComm();
    HcclCommSharedPtr CreateHcclCommInMulitProcess();
    HcclCommSharedPtr CreateHcclCommInMulitProcessByRootInfo();
    HcclCommSharedPtr CreateHcclCommInMulitProcessByRankFile() const;
    bool CreateHcclRootInfo();
    void ShmGetHcclRootInfo(Mki::ShareMemory &shm, const CommInitInfo &shmInfo);
    void ShmSetHcclRootInfo(Mki::ShareMemory &shm, CommInitInfo &shmInfo);
    bool ShmBarrier(Mki::ShareMemory &shm, CommInitInfo &shmInfo);
    void ShmSetReady(Mki::ShareMemory &shm, CommInitInfo &shmInfo) const;
    HcclResult HcclExecute(RunnerVariantPack &runnerVariantPack);
};
} // namespace atb
#endif
