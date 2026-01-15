/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_OPS_RUNNER_H
#define ATB_OPS_RUNNER_H

#include <functional>
#include <map>
#include <set>
#include <memory>
#include <mki/utils/profiling/profiling_funcs.h>
#include <mki/op_desc.h>
#include <mki/launch_param.h>
#include "atb/runner/runner.h"
#include "atb/utils/log.h"
#include "atb/runner/runner_type.h"
#include "atb/types.h"
#include "atb/utils/runner_variant_pack.h"
#include "atb/svector.h"
#include "atb/utils/mem_allocation_solver/mem_allocation_solver.h"
#include "atb/kernel_cache/kernel_cache.h"
#include "atb/runner/atb_kernel_method.h"
#include "atb/runner/kernel_graph.h"


namespace atb {
struct OpsTensorPack {
    SVector<Mki::Tensor> inTensors;
    SVector<Mki::Tensor> outTensors;
};

class OpsRunner : public Runner {
public:
    explicit OpsRunner(const std::string &name, RunnerType runnerType = RUNNER_TYPE_UNDEFINED);
    ~OpsRunner() override;
    void ReserveSvector(RunnerVariantPack &runnerPack);
    bool IsSupportGlbWorkspace() override;
    uint64_t GetArgsSize() override;
    Status BuildArgs() override;
    Status UpdateTensorAddr(RunnerVariantPack &runnerVariantPack) override;
    Status UpdateWorkspaceBuffer(RunnerVariantPack &runnerVariantPack) override;

protected:
    virtual Status SetupKernelGraph(const OpsTensorPack &opsTensorPack);

protected:
    Status SetupImpl(RunnerVariantPack &runnerVariantPack) override;
    uint64_t GetTilingBufferSizeImpl() override;
    Status FillHostTilingBufferImpl(uint8_t *hostTilingBuffer, uint64_t tilingBufferSize,
                                    ContextBase *context) override;
    uint64_t GetWorkspaceBufferSizeImpl() override;
    uint64_t GetIntermediateBufferSizeImpl() override;
    Status ExecuteImpl(RunnerVariantPack &runnerVariantPack) override;
    Status PreExecuteImpl(RunnerVariantPack &runnerVariantPack) override;
    virtual Status ModifyKernelGraph(const OpsTensorPack &opsTensorPack);

protected:
    KernelGraph kernelGraph_;
    bool needKernelGraphModify_ = false;
    bool skipSetUpKernelGraphWhenCacheHit_ = true;

private:
    void Reset();
    bool SetupCanReuse(RunnerVariantPack &runnerVariantPack, bool &kernelGraphTopoChanged);
    void SetupCacheGetCachedTiling(uint8_t *kernelHostTilingBuffer, uint64_t maxTilingSize, bool launchWithTiling);
    Status PlanKernelGraph(uint8_t *kernelHostTilingBuffer, uint64_t maxTilingSize, bool launchWithTiling);
    Status PlanKernel(size_t nodeId, uint8_t *kernelHostTilingBuffer, uint64_t maxTilingSize, bool launchWithTiling);
    bool UpdateNodeBestKernelAndTiling(KernelGraphNode &node, size_t nodeId, uint8_t *kernelHostTilingBuffer,
                                       uint64_t maxTilingSize, bool launchWithTiling);
    void RunMallocCache(RunnerVariantPack &runnerVariantPack);
    void UpdateOutTensorDeviceData(RunnerVariantPack &runnerVariantPack);
    size_t GetNodeAlignedTilingBufferSize(const KernelGraphNode &node, size_t nodeId) const;
    void SearchTensorInNodeInTensor(const Mki::Tensor *tensor, uint64_t &maxNodeId, uint64_t &dependNodeCount);
    bool SearchTensorInNodeOutTensor(const Mki::Tensor *tensor, uint64_t &maxNodeId);
    void InitTensorMaxNodeMap();
    bool IsInternalTensor(const Mki::Tensor *tensor);
    void WriteTilingData(const uint8_t *tilingData, size_t len, const std::string &filePath) const;
    void UpdateRunInfoTensorData(KernelGraphNode &node, size_t nodeId, uint8_t *deviceIntermediateBuffer) const;
    Status UpdateRunInfoTiling(RunnerVariantPack &runnerVariantPack);
    void UpdateRunInfoWorkspace(RunnerVariantPack &runnerVariantPack);
    Status RunAllKernel(RunnerVariantPack &runnerVariantPack);
    void InitTensorFromRunnerPack(const RunnerVariantPack &runnerVariantPack);
    void InitKernelGraph();
    void CalcKernelWorkspace();

    void ReportLaunchInfo(const uint64_t beginTime, const char *opName, const void *key) const;
    void ReportAdditionalInfo(const uint64_t timeStamp, const char *opName, size_t nodeId, const void *key) const;
    void ReportTensorInfo(const uint64_t timeStamp, const char *opName, const KernelGraphNode &node,
                          const void *key) const;
    void ReportContextInfo(const uint64_t timeStamp, const char *opName, const void *key) const;
    void ReportMsprofInfo(const uint64_t timeStamp, const char *opName, const KernelGraphNode &node,
                          size_t nodeId) const;
    Status UpdateDeviceRealAddr(const RunnerVariantPack &runnerVariantPack);
    Status RunKernel(KernelGraphNode &node, size_t nodeId, ContextBase *context) const;
    Status FillSingleKernelHostTilingBuffer(KernelGraphNode &node, size_t nodeId, uint8_t *kernelHostTilingBuffer,
                                            size_t tilingSize, ContextBase &context);
    void MallocLocalInternalTensor(const KernelGraphNode &node, size_t nodeId, size_t tensorId,
                                   const Mki::Tensor &infershapedOutTensor, Mki::Tensor *outTensor);
    void MallocGlobalInternalTensor(const KernelGraphNode &node, size_t nodeId, size_t tensorId,
                                    const Mki::Tensor &infershapedOutTensor, Mki::Tensor *outTensor);
    void MallocInternalTensor(KernelGraphNode &node, size_t nodeId);
    void FreeInternalTensor(KernelGraphNode &node, size_t nodeId);
    void InitKernelCache();
    void SaveGlobalDeviceTiling(const std::string &dirPath, std::vector<uint8_t> &tilingData) const;
    void RunKernelPreProcess(KernelGraphNode &node, size_t nodeId, aclrtStream stream);
    void RunKernelPostProcess(KernelGraphNode &node, size_t nodeId, aclrtStream stream);
    void SyncStream(KernelGraphNode &node, size_t nodeId, aclrtStream stream) const;
    void SaveGlobalDeviceTiling(const std::string &dirPath, std::vector<char> &tilingData) const;
    void IncreaseStatisticCacheHitCount(bool localCache) const;
    bool GetCachedTiling(KernelGraphNode &node, size_t nodeId, uint8_t *kernelHostTilingBuffer, uint64_t maxTilingSize,
                         uint64_t &tilingSizeFetched, bool launchWithTiling);
    void UpdateCacheTiling(KernelGraphNode &node, size_t nodeId, uint8_t *kernelHostTilingBuffer, size_t tilingSize);
    void SetNodesSaveTensorFlag();
    void SetNodeIds();
    void DumpKernelIOTensorInfo(KernelGraphNode &node) const;
    bool CheckOverflow(const KernelGraphNode &node, ContextBase *context) const;
    bool ExecuteOverFlowCheckKernel(const std::string &opName, ContextBase *context) const;
    bool CheckOverFlowByTensor(const std::string &opName) const;
    bool JudgeOverflowTensor(const std::string &opName, const Mki::Tensor &tensor, uint8_t *hostBuffer) const;
    void InitOpsTensorPack(const RunnerVariantPack &runnerPack);

private:
    void
    BuildAdditionalInfo(const uint64_t nameHash,
                        const Mki::SVector<std::pair<bool, Mki::Tensor>, ATB_MSPROF_TENSOR_DATA_SIZE> &batchTensors,
                        MsprofAdditionalInfo &additionInfo) const;
    void BuildTensorData(const Mki::SVector<std::pair<bool, Mki::Tensor>, ATB_MSPROF_TENSOR_DATA_SIZE> &batchTensors,
                         MsprofTensorInfo &tensorDataInfo) const;
    void DoReportTensorInfo(MsprofAdditionalInfo &additionInfo) const;
    void DoReportContextInfo(MsprofContextIdInfo &mixTensorInfo) const;
    uint64_t totalTilingSize_ = 0;
    SVector<uint64_t> tilingSizes_;
    uint64_t workspaceSize_ = 0;
    uint64_t intermediateSize_ = 0;
    std::map<Mki::Tensor *, uint64_t> tensorMaxNodeIdMap_;
    std::map<uint64_t, std::set<Mki::Tensor *>> maxNodeIdTensorMap_;
    std::map<Mki::Tensor *, bool> isInTensorCanFree_;
    std::map<Mki::Tensor *, bool> isOutTensorNeedMalloc_;
    std::set<Mki::Tensor *> tensorMalloced_;
    std::shared_ptr<MemAllocationSolver> memAllocationSolver_;
    RunnerType runnerType_ = RUNNER_TYPE_UNDEFINED;
    RunnerVariantPack lastRunnerVariantPack_;
    OpsTensorPack opsTensorPack_;
    std::vector<std::pair<Mki::Tensor *, bool>> mallocCache_; // true: malloc, false: free
    bool initKernelGraphFlag_ = false;
    KernelCache localKernelCache_;
    bool kernelCacheInited_ = false;
    std::vector<uint8_t> globalTilingBeforeKernelRun_;
    std::vector<uint8_t> globalTilingAfterKernelRun_;
    std::vector<std::pair<KernelCache *, bool>> kernelCaches_;
    std::vector<bool> nodesSaveTensorFlag_;
    bool isVariantPackEqual_ = false;
    bool overrideModifyGraph_ = true;
};
} // namespace atb
#endif
