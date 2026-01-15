/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "operation_torch.h"
#include <vector>
#include <mki/utils/time/timer.h>
#include <asdops/ops.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/config.h"
#include "atb/utils/statistic.h"
#include "atb/context.h"
#include "atb_torch/utils/utils.h"
#include "test_utils/context/memory_context.h"
#include "operation_funcs.h"
#include "hosttensor_binder_creator.h"
#include "atb/types.h"
#include "atb/utils/singleton.h"
#include "acl/acl.h"

uint64_t GetNewOpId()
{
    static uint64_t opId = 0;
    uint64_t newOpId = opId++;
    return newOpId;
}

static atb::Context *GetAtbContext()
{
    static atb::Context *context = nullptr;
    if (context) {
        context->SetExecuteStream(Utils::GetCurrentStream());
        return context;
    }

    atb::CreateContext(&context);
    if (context) {
        context->SetExecuteStream(Utils::GetCurrentStream());
        const char *env = std::getenv("ATB_USE_TILING_COPY_STREAM");
        if (env && std::string(env) == "1") {
            ATB_LOG(WARN) << "ATB_USE_TILING_COPY_STREAM is 1, call SetAsyncTilingCopyStatus";
            context->SetAsyncTilingCopyStatus(true);
        } else {
            ATB_LOG(WARN) << "ATB_USE_TILING_COPY_STREAM is not 1, not call SetAsyncTilingCopyStatus";
        }
    }

    return context;
}

OperationTorch::OperationTorch(std::string opName) : opName_(opName), name_(opName)
{
    opId_ = GetNewOpId();
    nodeId_ = std::to_string(opId_);
    ATB_LOG(INFO) << "OperationTorch::OperationTorch:" << " opName:" << opName << ", opId:" << opId_;
    std::vector<Mki::Operation *> ops = AsdOps::Ops::Instance().GetAllOperations();
}

OperationTorch::~OperationTorch() {}

void OperationTorch::SetName(std::string name)
{
    name_ = name;
}

std::string OperationTorch::SetParam(std::string param)
{
    ATB_LOG(INFO) << name_ << " set param start, param:" << param;
    param_ = param;

    atb::Operation *operation = nullptr;
    atb::Status st = CreateOperation(opName_, param_, &operation);
    nlohmann::json setParamStat = {};
    setParamStat["result"] = st;
    if (st != 0) {
        ATB_LOG(ERROR) << name_ << " create operation fail, error code: " << st;
        return setParamStat.dump();
    }
    if (operation == nullptr) {
        ATB_LOG(FATAL) << name_ << " create operation fail, opName:" << opName_ << ", param:" << param_;
        return setParamStat.dump();
    }

    operation_.reset(operation);

    HostTensorBinder *binder = CreateHostTensorBinder(opName_);
    hostTensorBinder_.reset(binder);

    ATB_LOG(DEBUG) << name_ << " set param end";
    return setParamStat.dump();
}

std::string OperationTorch::UpdateParam(std::string newParam)
{
    nlohmann::json updateParamStat = {};
    if (operation_ == nullptr) {
        ATB_LOG(ERROR) << name_ << " operation is null, opName:" << opName_ << ", param:" << param_;
        updateParamStat["result"] = -1;
        return updateParamStat.dump();
    }
    atb::Status st = UpdateOperationParam(opName_, newParam, operation_.get());
    updateParamStat["result"] = st;
    if (st != 0) {
        ATB_LOG(ERROR) << name_ << " update operation fail, error code: " << st;
        return updateParamStat.dump();
    }
    param_ = newParam;
    ATB_LOG(DEBUG) << name_ << " update param end";
    return updateParamStat.dump();
}

void OperationTorch::SetVaraintPackParam(std::string varaintPackParam)
{
    ATB_LOG(INFO) << name_ << " set varaint pack param start, param:" << varaintPackParam;

    if (hostTensorBinder_) {
        try {
            nlohmann::json paramJson = nlohmann::json::parse(varaintPackParam);
            hostTensorBinder_->ParseParam(paramJson);
        } catch (const std::exception &e) {
            ATB_LOG(ERROR) << "parse json fail, error:" << e.what();
        }
    } else {
        ATB_LOG(ERROR) << "hostTensorBinder is nullptr";
    }
    ATB_LOG(INFO) << name_ << " set varaint pack param end";
}

void OperationTorch::SetTensorList(std::vector<torch::Tensor> atInTensors, std::string tensorListName)
{
    ATB_LOG(INFO) << name_ << " SetTensorList start, tensorListName:" << tensorListName;
    if (hostTensorBinder_) {
        try {
            std::vector<atb::Tensor> tensorList;
            tensorList.resize(atInTensors.size());
            for (size_t i = 0; i < atInTensors.size(); i++) {
                tensorList.at(i) = Utils::AtTensor2Tensor(atInTensors.at(i));
            }
            hostTensorBinder_->SetTensorList(tensorList, tensorListName);
        } catch (const std::exception &e) {
            ATB_LOG(ERROR) << "SetTensorList fail, error:" << e.what();
        }
    } else {
        ATB_LOG(ERROR) << "hostTensorBinder is nullptr";
    }
    ATB_LOG(INFO) << name_ << " SetTensorList end";
}

atb::Status OperationTorch::InferShapeOutTensorDesc(std::vector<torch::Tensor> &atInTensors,
                                                    atb::SVector<atb::TensorDesc> &outTensorDescs)
{
    Utils::ContiguousAtTensor(atInTensors);
    atb::SVector<atb::TensorDesc> inTensorDescs;
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        if (atInTensors.at(i).sizes().size() > atb::MAX_DIM) {
            ATB_LOG(ERROR) << "tensor dimNum " << atInTensors.at(i).sizes().size()
                           << " is invalid, should >0 && <= MAX_DIM(8)";
            return atb::ERROR_INVALID_TENSOR_DIM;
        }
        auto &atInTensor = atInTensors.at(i);
        atb::Tensor inTensor = Utils::AtTensor2Tensor(atInTensor);
        if (inTensor.desc.format == ACL_FORMAT_NCHW) {
            inTensor.desc.format = ACL_FORMAT_ND;
        }
        inTensorDescs.push_back(inTensor.desc);
        ATB_LOG(INFO) << name_ << " infer shape inTensors[" << i << "]:" << atb::TensorUtil::TensorToString(inTensor);
    }
    return operation_->InferShape(inTensorDescs, outTensorDescs);
}

std::string OperationTorch::InferShape(std::vector<torch::Tensor> atInTensors)
{
    ATB_LOG(INFO) << " infershape start";
    if (!operation_) {
        ATB_LOG(FATAL) << name_ << " infershape fail, operation is null";
    }

    atb::SVector<atb::TensorDesc> outTensorDescs;
    atb::Status st = InferShapeOutTensorDesc(atInTensors, outTensorDescs);
    nlohmann::json inferShapeStat = {};
    inferShapeStat["result"] = st;
    if (st != 0) {
        ATB_LOG(ERROR) << name_ << " infer shape fail, error code: " << st;
    } else {
        inferShapeStat["num"] = outTensorDescs.size();
        for (size_t i = 0; i < outTensorDescs.size(); i++) {
            const auto &desc = outTensorDescs.at(i);
            inferShapeStat["dtype"].push_back(desc.dtype);
            inferShapeStat["format"].push_back(desc.format);
            std::vector<int> dims;
            int dimNum = desc.shape.dimNum;
            for (int i = 0; i < dimNum; ++i) {
                dims.push_back(desc.shape.dims[i]);
            }
            inferShapeStat["shape"].push_back(dims);
        }
    }
    ATB_LOG(INFO) << " infershape end";
    return inferShapeStat.dump();
}

std::string OperationTorch::Setup(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors)
{
    ATB_LOG(INFO) << " Setup start";

    BuildVariantPack(atInTensors, atOutTensors);
    atb::Context *context = GetAtbContext();

    uint64_t workspaceSize = 0;
    Mki::Timer setupTimer;
    atb::Status st = operation_->Setup(variantPack_, workspaceSize, context);
    auto setupTime = setupTimer.ElapsedMicroSecond();

    nlohmann::json setupStat = {};
    setupStat["result"] = st;
    setupStat["workspace_size"] = workspaceSize;
    setupStat["setup_time"] = setupTime;
    if (st != 0) {
        ATB_LOG(ERROR) << name_ << " setup fail, not call execute, error code: " << st;
    }
    ATB_LOG(INFO) << " Setup end";
    return setupStat.dump();
}

std::string OperationTorch::ExecuteSync(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors,
                                        int64_t workspaceSize)
{
    ATB_LOG(INFO) << " ExecuteSync start";

    BuildVariantPack(atInTensors, atOutTensors);

    uint8_t *workspace = nullptr;
    if (workspaceSize > 0) {
        aclError ret = aclrtMalloc((void **)(&workspace), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << "overflowworkspace malloc failed!";
        }
    }

    nlohmann::json executeSyncStat = {};
    atb::Context *context = GetAtbContext();

    Mki::Timer executeTimer;
    atb::Status st = operation_->Execute(variantPack_, workspace, workspaceSize, context);
    auto executeTime = executeTimer.ElapsedMicroSecond();

    executeSyncStat["result"] = st;
    if (st != 0) {
        ATB_LOG(ERROR) << name_ << " ExecuteSync fail, error code: " << st;
    } else {
        executeSyncStat["execute_time"] = executeTime;
        Mki::Timer syncTimer;
        aclrtSynchronizeStream(context->GetExecuteStream());
        auto syncTime = syncTimer.ElapsedMicroSecond();
        executeSyncStat["sync_time"] = syncTime;
    }
    if (workspace){
        aclError ret = aclrtFree(workspace);
    	if (ret != ACL_SUCCESS) {
        	ATB_LOG(ERROR) << "workspace free failed!";
    	}
    }
    ATB_LOG(INFO) << " ExecuteSync end";
    return executeSyncStat.dump();
}

std::vector<torch::Tensor> OperationTorch::ExecuteSplitThread(std::vector<torch::Tensor> atInTensors)
{
    ATB_LOG(DEBUG) << name_ << " execute start";
    if (!operation_) {
        ATB_LOG(FATAL) << name_ << " execute fail, operation is null";
    }

    std::vector<torch::Tensor> atOutTensors;
    CreateAtOutTensors(atInTensors, atOutTensors);

    ATB_LOG(DEBUG) << name_ << " execute impl execCount:" << executeCount_;

    BuildVariantPack(atInTensors, atOutTensors);

    uint64_t workspaceSize = 0;
    atb::Context *context = GetAtbContext();

    atb::Status st = operation_->Setup(variantPack_, workspaceSize, context);
    if (st != 0) {
        ATB_LOG(ERROR) << name_ << " setup fail, not call execute, error code: " << st;
        return atOutTensors;
    }

    ATB_LOG(INFO) << name_ << " get plan workspace size:" << workspaceSize;

    uint8_t *workspace = nullptr;
    if (workspaceSize > 0) {
        workspace = (uint8_t *)atb::GetSingleton<atb::MemoryContext>().GetWorkspaceBuffer(workspaceSize);
    }
    context->SetExecuteType(atb::EXECUTE_PRELAUNCH);
    st = operation_->Execute(variantPack_, workspace, workspaceSize, context);
    context->SetExecuteType(atb::EXECUTE_LAUNCH);
    st = operation_->Execute(variantPack_, workspace, workspaceSize, context);
    context->SetExecuteType(atb::EXECUTE_NORMAL);
    ATB_LOG_IF(st != 0, ERROR) << name_ << " execute plan fail, error code: " << st;

    ATB_LOG(INFO) << name_ << " execute end";
    return atOutTensors;
}

std::vector<torch::Tensor> OperationTorch::Execute(std::vector<torch::Tensor> atInTensors)
{
    ATB_LOG(DEBUG) << name_ << " execute start";
    if (!operation_) {
        ATB_LOG(FATAL) << name_ << " execute fail, operation is null";
    }

    std::vector<torch::Tensor> atOutTensors;
    CreateAtOutTensors(atInTensors, atOutTensors);
    ExecuteOutImpl(atInTensors, atOutTensors);
    ATB_LOG(INFO) << name_ << " execute end";
    return atOutTensors;
}

void OperationTorch::ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors)
{
    ATB_LOG(INFO) << name_ << " execute out start";
    if (!operation_) {
        ATB_LOG(FATAL) << name_ << " execute out fail, operation is null";
    }
    ExecuteOutImpl(atInTensors, atOutTensors);
}

void OperationTorch::ExecuteOutImpl(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors)
{
    ATB_LOG(DEBUG) << name_ << " execute impl execCount:" << executeCount_;

    BuildVariantPack(atInTensors, atOutTensors);

    uint64_t workspaceSize = 0;
    atb::Context *context = GetAtbContext();

    atb::Status st = operation_->Setup(variantPack_, workspaceSize, context);
    if (st != 0) {
        ATB_LOG(ERROR) << name_ << " setup fail, not call execute, error code: " << st;
        return;
    }

    ATB_LOG(INFO) << name_ << " get plan workspace size:" << workspaceSize;

    uint8_t *workspace = nullptr;
    if (workspaceSize > 0) {
        workspace = (uint8_t *)atb::GetSingleton<atb::MemoryContext>().GetWorkspaceBuffer(workspaceSize);
    }

    st = operation_->Execute(variantPack_, workspace, workspaceSize, context);
    ATB_LOG_IF(st != 0, ERROR) << name_ << " execute plan fail, error code: " << st;

    executeCount_++;
}

void OperationTorch::CreateAtOutTensors(std::vector<torch::Tensor> &atInTensors,
                                        std::vector<torch::Tensor> &atOutTensors)
{
    atb::SVector<atb::TensorDesc> outTensorDescs;
    atb::Status st = InferShapeOutTensorDesc(atInTensors, outTensorDescs);
    if (st != 0) {
        ATB_LOG(ERROR) << name_ << " infer shape fail, error code: " << st;
        return;
    }
    atOutTensors.resize(outTensorDescs.size());
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        ATB_LOG(INFO) << name_ << " infer shape outTensorDescs[" << i
                      << "]:" << atb::TensorUtil::TensorDescToString(outTensorDescs.at(i));
        Mki::Timer timer;
        at::Tensor newTensor = Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(i));
        atOutTensors.at(i) = newTensor;
    }
}

void OperationTorch::BuildVariantPack(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors)
{
    Utils::ContiguousAtTensor(atInTensors);
    Utils::ContiguousAtTensor(atOutTensors);
    variantPack_.inTensors.resize(atInTensors.size());
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        ATB_LOG(DEBUG) << name_ << " execute start, atInTensors[" << i << "].options:" << atInTensors.at(i).options()
                       << ", data:" << atInTensors.at(i).data_ptr()
                       << ", storage_offset:" << atInTensors.at(i).storage_offset()
                       << ", format:" << Utils::GetTensorNpuFormat(atInTensors.at(i));
        atInTensors.at(i) = Utils::NpuFormatCast(atInTensors.at(i));
        variantPack_.inTensors.at(i) = Utils::AtTensor2Tensor(atInTensors.at(i));
        if (variantPack_.inTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
            variantPack_.inTensors.at(i).desc.format = ACL_FORMAT_ND;
        }
    }

    variantPack_.outTensors.resize(atOutTensors.size());
    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        ATB_LOG(DEBUG) << name_ << " execute start, atOutTensors[" << i << "].options:" << atOutTensors.at(i).options()
                       << ", data:" << atOutTensors.at(i).data_ptr()
                       << ", storage_offset:" << atOutTensors.at(i).storage_offset()
                       << ", format:" << Utils::GetTensorNpuFormat(atOutTensors.at(i));
        variantPack_.outTensors.at(i) = Utils::AtTensor2Tensor(atOutTensors.at(i));
        if (variantPack_.outTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
            variantPack_.outTensors.at(i).desc.format = ACL_FORMAT_ND;
        }
    }

    if (hostTensorBinder_) {
        hostTensorBinder_->BindTensor(variantPack_);
    }
}

std::string OperationTorch::GetSaveTensorDir()
{
    const char *envStr = std::getenv("AIT_CMP_TASK_ID");
    std::string dir = envStr ? std::string(envStr) : std::to_string(executeCount_);
    return "atb_temp/tensors/" + dir + "/" + std::to_string(opId_) + "_OperationTorch";
}

bool OperationTorch::IsCopyStreamValid()
{
    const char *env = std::getenv("ATB_USE_TILING_COPY_STREAM");
    if (!env) {
        return false;
    }
    bool isValid = std::string(env) == "1";
    return isValid;
}

TORCH_LIBRARY(OperationTorch, m)
{
    m.class_<OperationTorch>("OperationTorch")
        .def(torch::init<std::string>())
        .def("set_param", &OperationTorch::SetParam)
        .def("update_param", &OperationTorch::UpdateParam)
        .def("set_varaintpack_param", &OperationTorch::SetVaraintPackParam)
        .def("infer_shape", &OperationTorch::InferShape)
        .def("setup", &OperationTorch::Setup)
        .def("execute_sync", &OperationTorch::ExecuteSync)
        .def("execute", &OperationTorch::Execute)
        .def("set_tensor_list", &OperationTorch::SetTensorList)
        .def("execute_out", &OperationTorch::ExecuteOut)
        .def("execute_split_thread", &OperationTorch::ExecuteSplitThread);
}
