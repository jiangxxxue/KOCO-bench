/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "linear_parallel_aclnn_runner.h"
#include <hccl/hccl.h>
#include "atb/utils/dl_manager.h"
#include "atb/utils/aclnn_util.h"


namespace atb {

static const uint32_t LINEAR_REDUCE_SCATTER_IN_TENSOR_NUM = 6;
static const uint32_t LINEAR_REDUCE_SCATTER_OUT_TENSOR_NUM = 2;

aclnnStatus (*LinearParallelAclnnRunner::aclnnMatmulReduceScatterV2GetWorkspaceSizeFunc_)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    int64_t, const char *, const char *, int64_t, int64_t, int64_t, const char *, const aclTensor *, const aclTensor *,
    uint64_t *, aclOpExecutor **) = nullptr;

aclnnStatus (*LinearParallelAclnnRunner::aclnnMatmulReduceScatterV2Func_)(void *, uint64_t, aclOpExecutor *,
                                                                          aclrtStream) = nullptr;

LinearParallelAclnnRunner::LinearParallelAclnnRunner(const infer::LinearParallelParam &param, bool useRankTableFile)
    : AclnnRunner("LinearParallelAclnnRunner", RUNNER_TYPE_LINEAR_PARALLEL),
      hcclRunner_(!useRankTableFile ? HcclRunner("LinearParallelAclnnRunner", RUNNER_TYPE_LINEAR_PARALLEL, param.rank,
                                                 param.rankSize, param.rankRoot, param.commDomain) :
                                      HcclRunner("LinearParallelAclnnRunner", RUNNER_TYPE_LINEAR_PARALLEL, param.rank,
                                                 param.rankTableFile, param.commDomain)),
      param_(param)
{
    ATB_LOG(INFO) << "LinearParallelAclnnRunner::LinearParallelAclnnRunner called";
}

LinearParallelAclnnRunner::LinearParallelAclnnRunner(const infer::LinearParallelParam &param, HcclComm hcclComm)
    : AclnnRunner("LinearParallelAclnnRunner", RUNNER_TYPE_LINEAR_PARALLEL),
      hcclRunner_("LinearParallelAclnnRunner", hcclComm, RUNNER_TYPE_LINEAR_PARALLEL), param_(param)
{
    ATB_LOG(INFO) << "LinearParallelAclnnRunner::LinearParallelAclnnRunner ext called";
}

LinearParallelAclnnRunner::~LinearParallelAclnnRunner() {}

Status LinearParallelAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack called";
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    this->aclnnVariantPack_.aclInTensors.reserve(LINEAR_REDUCE_SCATTER_IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(LINEAR_REDUCE_SCATTER_IN_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        if (i > 1) {
            this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
            continue;
        }
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        atb::Dims viewDims = atbTensor.desc.shape;
        if (i == 1 && param_.transWeight) {
            aclnnTensorPtr->strides = GetTransposeTensorStride(viewDims);
            viewDims.dims[0] = atbTensor.desc.shape.dims[1];
            viewDims.dims[1] = atbTensor.desc.shape.dims[0];
        } else {
            aclnnTensorPtr->strides = GetCopyTensorStride(viewDims);
        }
        ret = CallAclCreateTensor(viewDims, atbTensor.desc.shape, atbTensor, aclnnTensorPtr);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = i;
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
    }

    this->aclnnVariantPack_.aclOutTensors.reserve(LINEAR_REDUCE_SCATTER_OUT_TENSOR_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(LINEAR_REDUCE_SCATTER_OUT_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        if (i >= 1) {
            this->aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
            continue;
        }
        atb::Tensor atbTensor = runnerVariantPack.outTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = i;
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        this->aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
    }
    return ret;
}

aclnnStatus LinearParallelAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "SetAclNNWorkspaceExecutor called";
    Status status = LinearParallelAclnnRunner::LoadMethodMatmulReduceScatter();
    if (status != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "load getWorkspace function from aclnn failed! Consider upgrade CANN first";
        return 561003; // ACLNN_ERR_INNER_FIND_KERNEL_ERROR
    }
    ATB_LOG(INFO) << GetLogPrefix() << "aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
    size_t inTensorStart = 0;
    aclTensor *x1 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *x2 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *bias = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *x1Scale = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *x2Scale = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *quantScale = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;

    size_t outTensorStart = 0;
    aclTensor *output = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
    aclTensor *amaxOutOptional = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;

    int64_t blockSize = 0;
    char group[128] = {0};
    aclnnStatus ret = HcclGetCommName(hcclRunner_.GetHcclCommSharedPtr().get(), group);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "HcclGetCommName error: " << ret;
        return ret;
    }
    char reduceOp[128] = "sum";
    int64_t commTurn = 0;
    int64_t streamMode = 1;
    int64_t groupSize = 0;
    char commMode[128] = "aiv";

    aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();
    ATB_LOG(INFO) << GetLogPrefix() << "&(this->aclnnExecutor_): " << &(this->aclnnExecutor_)
                  << ", addr of this->aclnnExecutor_: " << this->aclnnExecutor_
                  << ", raw ptr from it: " << raw_executor_ptr
                  << ", then take the address of the raw ptr: " << &raw_executor_ptr;

    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize addr: " << &(this->atbVariantPack_.workspaceBufferSize);

    ret = LinearParallelAclnnRunner::aclnnMatmulReduceScatterV2GetWorkspaceSizeFunc_(
        x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, group, reduceOp, commTurn, streamMode, groupSize,
        commMode, output, amaxOutOptional, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "SetAclNNWorkspaceExecutor error: " << ret;
        return ret;
    }
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}


Status LinearParallelAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel called";
    Status status = NO_ERROR;
    aclnnStatus ret = 0;
    void *executeStream = GetExecuteStream(this->atbVariantPack_.context);
    status = LinearParallelAclnnRunner::LoadMethodMatmulReduceScatter();
    if (status != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "load getWorkspace function from aclnn failed! Consider upgrade CANN first";
        return status;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "aclnnMatmulReduceScatterV2 execute start.";
    ret = LinearParallelAclnnRunner::aclnnMatmulReduceScatterV2Func_(this->atbVariantPack_.workspaceBuffer,
                                                                     this->atbVariantPack_.workspaceBufferSize,
                                                                     this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn error: " << ret;
        return ERROR_CANN_ERROR;
    }
    return NO_ERROR;
}

Status LinearParallelAclnnRunner::LoadMethodMatmulReduceScatter()
{
    ATB_LOG(INFO) << "LinearParallelAclnnRunner LoadMethod";
    if (LinearParallelAclnnRunner::aclnnMatmulReduceScatterV2GetWorkspaceSizeFunc_ != nullptr &&
        LinearParallelAclnnRunner::aclnnMatmulReduceScatterV2Func_ != nullptr) {
        return NO_ERROR;
    }
    static DlManager dlManager = DlManager(std::string(std::getenv("ASCEND_HOME_PATH")) + "/lib64/libopapi.so");
    Status ret =
        dlManager.getSymbol("aclnnMatmulReduceScatterV2GetWorkspaceSize",
                            (void *&)LinearParallelAclnnRunner::aclnnMatmulReduceScatterV2GetWorkspaceSizeFunc_);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << "load aclnnMatmulReduceScatterV2GetWorkspaceSize failed! Consider upgrade the CANN first!";
        return ret;
    }
    ret = dlManager.getSymbol("aclnnMatmulReduceScatterV2",
                              (void *&)LinearParallelAclnnRunner::aclnnMatmulReduceScatterV2Func_);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << "load aclnnMatmulReduceScatterV2 failed! Consider upgrade the CANN first!";
        return ret;
    }
    ATB_LOG(INFO) << "load aclnnMatmulReduceScatterV2 two-staged method success!";
    return NO_ERROR;
}

} // namespace atb
