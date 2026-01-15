/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "operation_test.h"
#include <random>
#include <cstring>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <mki/types.h>
#include "atb/utils/log.h"
#include "atb/utils.h"
#include "atb/utils/tensor_util.h"
#include "context/memory_context.h"
#include "mki/utils/fp16/fp16_t.h"
#include "atb/utils/singleton.h"
#include <iomanip>

const uint64_t MAX_NUMEL = 300000000000;

namespace atb {
OperationTest::OperationTest()
{
    const char *envStr = std::getenv("SET_NPU_DEVICE");
    deviceId_ = (envStr != nullptr) ? atoi(envStr) : 0;
}

OperationTest::~OperationTest()
{
    Cleanup();
}

void OperationTest::Cleanup()
{
    for (auto tensor : goldenContext_.hostInTensors) {
        if (tensor.deviceData) {
            free(tensor.deviceData);
        }
        if (tensor.hostData) {
            free(tensor.hostData);
        }
    }
    goldenContext_.hostInTensors.clear();

    for (auto tensor : goldenContext_.hostOutTensors) {
        if (tensor.deviceData) {
            free(tensor.deviceData);
        }
        if (tensor.hostData) {
            free(tensor.hostData);
        }
    }
    goldenContext_.hostOutTensors.clear();

    for (auto tensor : goldenContext_.deviceInTensors) {
        if (tensor.deviceData) {
            aclrtFree(tensor.deviceData);
        }
        if (tensor.hostData) {
            aclrtFree(tensor.hostData);
        }
    }
    goldenContext_.deviceInTensors.clear();

    for (auto tensor : goldenContext_.deviceOutTensors) {
        if (tensor.deviceData) {
            aclrtFree(tensor.deviceData);
        }
        if (tensor.hostData) {
            aclrtFree(tensor.hostData);
        }
    }
    goldenContext_.deviceOutTensors.clear();

    if (context_ != nullptr && context_->GetExecuteStream()) {
        aclrtDestroyStream(context_->GetExecuteStream());
    }
    if (context_) {
        atb::DestroyContext(context_);
    }
}

void OperationTest::SetDeviceId(int deviceId)
{
    deviceId_ = deviceId;
}

Status OperationTest::Init()
{
    ATB_LOG(DEBUG) << "aclrtSetDevice " << deviceId_;
    int ret = aclrtSetDevice(deviceId_);
    if (ret != 0) {
        ATB_LOG(ERROR) << "aclrtSetDevice fail";
        return ret;
    }
    ret = atb::CreateContext(&context_);
    if (ret != 0) {
        ATB_LOG(ERROR) << "CreateContext fail";
        return ret;
    }
    aclrtStream stream = nullptr;
    ATB_LOG(DEBUG) << "aclrtCreateStream call";

    ret = aclrtCreateStream(&stream);
    if (ret != 0) {
        ATB_LOG(ERROR) << "aclrtCreateStream fail";
        return ret;
    }

    context_->SetExecuteStream(stream);
    return ErrorType::NO_ERROR;
}

Status OperationTest::Prepare(atb::Operation *operation, const SVector<Tensor> &inTensorLists)
{
    operation_ = operation;

    if (inTensorLists.size() != operation_->GetInputNum()) {
        return ErrorType::ERROR_INVALID_IN_TENSOR_NUM;
    }
    return ErrorType::NO_ERROR;
}

Tensor OperationTest::CreateHostTensor(const Tensor &tensorIn)
{
    Tensor tensor;
    tensor.desc = tensorIn.desc;
    tensor.dataSize = tensorIn.dataSize;

    tensor.deviceData = malloc(tensor.dataSize);
    memcpy(tensor.deviceData, tensorIn.deviceData, tensor.dataSize);
    tensor.hostData = malloc(tensor.dataSize);
    memcpy(tensor.hostData, tensorIn.hostData, tensor.dataSize);

    return tensor;
}

Tensor OperationTest::HostTensor2DeviceTensor(const Tensor &hostTensor)
{
    Tensor deviceTensor;
    deviceTensor.desc = hostTensor.desc;
    deviceTensor.dataSize = hostTensor.dataSize;
    int st = aclrtMalloc(&deviceTensor.deviceData, deviceTensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (st != ErrorType::NO_ERROR) {
        ATB_LOG(ERROR) << "malloc device tensor fail";
        return deviceTensor;
    }
    st = aclrtMemcpy(deviceTensor.deviceData, deviceTensor.dataSize, hostTensor.deviceData, hostTensor.dataSize,
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (st != ErrorType::NO_ERROR) {
        ATB_LOG(ERROR) << "copy host tensor to device tensor";
    }

    st = aclrtMalloc(&deviceTensor.hostData, deviceTensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (st != ErrorType::NO_ERROR) {
        ATB_LOG(ERROR) << "malloc device tensor fail";
        return deviceTensor;
    }
    st = aclrtMemcpy(deviceTensor.hostData, deviceTensor.dataSize, hostTensor.hostData, hostTensor.dataSize,
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (st != ErrorType::NO_ERROR) {
        ATB_LOG(ERROR) << "copy host tensor to device tensor";
    }
    return deviceTensor;
}

std::string OperationTest::TensorToString(const Tensor &tensor)
{
    return "";
}

Tensor OperationTest::CreateHostZeroTensor(const TensorDesc &tensorDesc)
{
    Tensor tensor;
    tensor.desc = tensorDesc;

    std::random_device rd;
    std::default_random_engine eng(rd());
    if (tensorDesc.dtype == ACL_FLOAT || tensorDesc.dtype == ACL_INT32) {
        tensor.dataSize = Utils::GetTensorNumel(tensor) * sizeof(float);
    } else if (tensorDesc.dtype == ACL_FLOAT16 || tensorDesc.dtype == ACL_BF16) {
        tensor.dataSize = Utils::GetTensorNumel(tensor) * sizeof(Mki::fp16_t);
    } else if (tensorDesc.dtype == ACL_INT64) {
        tensor.dataSize = Utils::GetTensorNumel(tensor) * sizeof(int64_t);
    } else if (tensorDesc.dtype == ACL_INT32) {
        tensor.dataSize = Utils::GetTensorNumel(tensor) * sizeof(int32_t);
    } else if (tensorDesc.dtype == ACL_UINT32) {
        tensor.dataSize = Utils::GetTensorNumel(tensor) * sizeof(uint32_t);
    } else if (tensorDesc.dtype == ACL_INT8) {
        tensor.dataSize = Utils::GetTensorNumel(tensor) * sizeof(int8_t);
    } else {
        ATB_LOG(ERROR) << "not support";
        return tensor;
    }

    tensor.hostData = malloc(tensor.dataSize);
    memset(tensor.hostData, 0, tensor.dataSize);
    tensor.deviceData = malloc(tensor.dataSize);
    memset(tensor.deviceData, 0, tensor.dataSize);
    return tensor;
}

void OperationTest::BuildVariantPack(const SVector<Tensor> &inTensorLists)
{
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.resize(operation_->GetInputNum());
    for (size_t i = 0; i < inTensorLists.size(); i++) {
        inTensorDescs.at(i) = inTensorLists.at(i).desc;
    }
    SVector<TensorDesc> outTensorDescs;
    outTensorDescs.resize(operation_->GetOutputNum());

    Status st = operation_->InferShape(inTensorDescs, outTensorDescs);
    if (st != ErrorType::NO_ERROR) {
        ATB_LOG(ERROR) << "InferShape fail, error!";
        return;
    }

    variantPack_.inTensors.resize(operation_->GetInputNum());
    variantPack_.outTensors.resize(operation_->GetOutputNum());

    for (size_t i = 0; i < inTensorLists.size(); ++i) {
        Tensor &deviceTensor = variantPack_.inTensors.at(i);
        Tensor hostTensor = CreateHostTensor(inTensorLists.at(i));
        deviceTensor.dataSize = hostTensor.dataSize;
        deviceTensor = HostTensor2DeviceTensor(hostTensor);
        ATB_LOG(DEBUG) << "InTensor[" << i << "] = " << TensorToString(hostTensor);
        goldenContext_.hostInTensors.push_back(hostTensor);
        goldenContext_.deviceInTensors.push_back(deviceTensor);
    }

    for (size_t i = 0; i < variantPack_.outTensors.size(); ++i) {
        variantPack_.outTensors.at(i).desc = outTensorDescs.at(i);
        Tensor &deviceTensor = variantPack_.outTensors.at(i);
        Tensor hostTensor = CreateHostZeroTensor(deviceTensor.desc);
        deviceTensor.dataSize = hostTensor.dataSize;
        int st = aclrtMalloc(&deviceTensor.deviceData, deviceTensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
        st = aclrtMalloc(&deviceTensor.hostData, deviceTensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ATB_LOG_IF(st != ErrorType::NO_ERROR, ERROR) << "malloc device tensor fail";
        goldenContext_.hostOutTensors.push_back(hostTensor);
        goldenContext_.deviceOutTensors.push_back(deviceTensor);
    }
}

Status OperationTest::RunOperation()
{
    Status st = operation_->Setup(variantPack_, workSpaceSize_, context_);
    if (st != ErrorType::NO_ERROR) {
        ATB_LOG(ERROR) << "Operation SetUp fail, error!";
        return st;
    }
    ATB_LOG(DEBUG) << "get Operation workspace size:" << workSpaceSize_;

    if (workSpaceSize_ > 0) {
        workSpace_ = GetSingleton<atb::MemoryContext>().GetWorkspaceBuffer(workSpaceSize_);
    }

    st = operation_->Execute(variantPack_, (uint8_t *)workSpace_, workSpaceSize_, context_);
    if (st != ErrorType::NO_ERROR) {
        ATB_LOG(ERROR) << "Plan Execute fail, error!";
        return st;
    }

    return ErrorType::NO_ERROR;
}

Status OperationTest::CopyDeviceTensorToHostTensor()
{
    for (size_t i = 0; i < variantPack_.outTensors.size(); ++i) {
        Tensor &deivceTensor = variantPack_.outTensors.at(i);
        Tensor &hostTensor = goldenContext_.hostOutTensors.at(i);
        ATB_LOG(DEBUG) << "aclrtMemcpy start, hostTensor.data:" << hostTensor.deviceData << ", hostTensor.dataSize:" <<
            hostTensor.dataSize << ", deivceTensor.data:" << deivceTensor.deviceData << ", deivceTensor.dataSize:" <<
            deivceTensor.dataSize;
        int st = aclrtMemcpy(hostTensor.deviceData, hostTensor.dataSize, deivceTensor.deviceData, deivceTensor.dataSize,
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (st != ErrorType::NO_ERROR) {
            ATB_LOG(ERROR) << "copy memory from device to host fail";
            return ErrorType::ERROR_RT_FAIL;
        }

        ATB_LOG(DEBUG) << "aclrtMemcpy start, hostTensor.data:" << hostTensor.hostData << ", hostTensor.dataSize:" <<
            hostTensor.dataSize << ", deivceTensor.data:" << deivceTensor.hostData << ", deivceTensor.dataSize:" <<
            deivceTensor.dataSize;
        st = aclrtMemcpy(hostTensor.hostData, hostTensor.dataSize, deivceTensor.hostData, deivceTensor.dataSize,
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (st != ErrorType::NO_ERROR) {
            ATB_LOG(ERROR) << "copy memory from device to host fail";
            return ErrorType::ERROR_RT_FAIL;
        }
        ATB_LOG(DEBUG) << "OutTensor[" << i << "] = " << TensorToString(hostTensor);
    }
    return ErrorType::NO_ERROR;
}

void OperationTest::FreeInTensorList(SVector<Tensor> &hostTensors)
{
    for (size_t i = 0; i < hostTensors.size(); i++) {
        if (hostTensors.at(i).hostData) {
            free(hostTensors.at(i).hostData);
        }
        if (hostTensors.at(i).deviceData) {
            free(hostTensors.at(i).deviceData);
        }
    }
}

Status OperationTest::Run(atb::Operation *operation, const SVector<TensorDesc> &inTensorDescs)
{
    SVector<Tensor> hostInTensors;
    hostInTensors.resize(inTensorDescs.size());
    ATB_LOG(DEBUG) << "GenerateRandomTensors Start!";
    Status st = GenerateRandomTensors(inTensorDescs, hostInTensors);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "GenerateRandomTensors failed!";
        return st;
    }
    ATB_LOG(DEBUG) << "RunImpl Start!";
    st = RunImpl(operation, hostInTensors);
    FreeInTensorList(hostInTensors);
    return st;
}

Status OperationTest::RunPerf(atb::Operation *operation, const SVector<Tensor> &inTensorLists)
{
    return RunPrefImpl(operation, inTensorLists);
}

Status OperationTest::RunPerf(atb::Operation *operation, const SVector<TensorDesc> &inTensorDescs)
{
    SVector<Tensor> hostInTensors;
    hostInTensors.resize(inTensorDescs.size());
    ATB_LOG(DEBUG) << "GenerateRandomTensors Start!";
    Status st = GenerateRandomTensors(inTensorDescs, hostInTensors);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "GenerateRandomTensors failed!";
        return st;
    }
    ATB_LOG(DEBUG) << "RunImpl Start!";
    st = RunPrefImpl(operation, hostInTensors);
    FreeInTensorList(hostInTensors);
    return st;
}

Status OperationTest::RunPrefImpl(atb::Operation *operation, const SVector<Tensor> &inTensorLists)
{
    Cleanup();
    ATB_LOG(DEBUG) << "Cleanup success!";
    Status status = Init();
    if (status != ErrorType::NO_ERROR) {
        ATB_LOG(ERROR) << "Init error!";
        return status;
    }
    ATB_LOG(DEBUG) << "Init success!";

    status = Prepare(operation, inTensorLists);
    ATB_LOG(DEBUG) << "Prepare success!";
    if (mockFlag_) {
        return status;
    }

    if (status != ErrorType::NO_ERROR) {
        return status;
    }

    BuildVariantPack(inTensorLists);
    ATB_LOG(DEBUG) << "BuildVariantPack success!";
    // warmup 消除波动
    for (int i = 0; i < 10; i++) {
        status = RunOperation();
        if (status != ErrorType::NO_ERROR) {
            return status;
        }
    }
    int ret = aclrtSynchronizeStream(context_->GetExecuteStream());
    if (ret != 0) {
        ATB_LOG(ERROR) << "aclrtSynchronizeStream fail, ret:" << ret;
        return ret;
    }
    for (int i = 0; i < 1000; i++) {
        status = RunOperation();
        if (status != ErrorType::NO_ERROR) {
            return status;
        }
    }
    ATB_LOG(DEBUG) << "RunOperation success!";
    std::chrono::duration<double, std::micro> avgSyncTime;
    auto start = std::chrono::high_resolution_clock::now();
    ret = aclrtSynchronizeStream(context_->GetExecuteStream());
    auto end = std::chrono::high_resolution_clock::now();
    avgSyncTime = (end - start) / 1000.0;
    ATB_LOG(ERROR) << "avgSyncTime: " << std::fixed << std::setprecision(2) << avgSyncTime.count() << " us";
    if (ret != 0) {
        ATB_LOG(ERROR) << "aclrtSynchronizeStream fail, ret:" << ret;
        return ret;
    }
    status = CopyDeviceTensorToHostTensor();
    if (status != ErrorType::NO_ERROR) {
        return status;
    }
    return status;
}

Status OperationTest::Run(atb::Operation *operation, const SVector<Tensor> &inTensorLists)
{
    return RunImpl(operation, inTensorLists);
}

Mki::Status OperationTest::Run(atb::Operation *operation, const Mki::SVector<Mki::TensorDesc> &inTensorDescs)
{
    SVector<TensorDesc> inAtbTensorDescs;
    inAtbTensorDescs.resize(inTensorDescs.size());
    for (size_t i = 0; i < inAtbTensorDescs.size(); i++) {
        TensorUtil::OpsTensorDesc2AtbTensorDesc(inTensorDescs.at(i), inAtbTensorDescs.at(i));
    }
    SVector<Tensor> hostInTensors;
    hostInTensors.resize(inAtbTensorDescs.size());
    GenerateRandomTensors(inAtbTensorDescs, hostInTensors);
    Status st = RunImpl(operation, hostInTensors);
    FreeInTensorList(hostInTensors);
    if (st != ErrorType::NO_ERROR) {
        return Mki::Status::FailStatus(st, "RunImpl error!");
    }
    return Mki::Status::OkStatus();
}
 
Mki::Status OperationTest::Run(atb::Operation *operation, const Mki::SVector<Mki::Tensor> &inTensorLists)
{
    SVector<Tensor> inAtbTensorLists;
    inAtbTensorLists.resize(inTensorLists.size());
    for (size_t i = 0; i < inTensorLists.size(); i++) {
        TensorUtil::ConvertOpsTensor2AtbTensor(inTensorLists.at(i), inAtbTensorLists.at(i));
    }
    Status st = RunImpl(operation, inAtbTensorLists);
    if (st != ErrorType::NO_ERROR) {
        return Mki::Status::FailStatus(-1, "RunImpl error!");
    }
    return Mki::Status::OkStatus();
}
 
Mki::Status OperationTest::Setup(atb::Operation *operation, const Mki::SVector<Mki::TensorDesc> &inTensorDescs)
{
    SVector<TensorDesc> inAtbTensorDescs;
    inAtbTensorDescs.resize(inTensorDescs.size());
    for (size_t i = 0; i < inAtbTensorDescs.size(); i++) {
        TensorUtil::OpsTensorDesc2AtbTensorDesc(inTensorDescs.at(i), inAtbTensorDescs.at(i));
    }
    SVector<Tensor> hostInTensors;
    hostInTensors.resize(inAtbTensorDescs.size());
    GenerateRandomTensors(inAtbTensorDescs, hostInTensors);
 
    Cleanup();
    Status st = Init();
    if (st != ErrorType::NO_ERROR) {
        return Mki::Status::FailStatus(-1, "Setup init error!");
    }
 
    Status status = Prepare(operation, hostInTensors);
    if (mockFlag_) {
        return Mki::Status::FailStatus(-1, "Setup error!");
    }
 
    if (status != ErrorType::NO_ERROR) {
        return Mki::Status::FailStatus(-1, "Setup error!");
    }
 
    BuildVariantPack(hostInTensors);
 
    st = operation_->Setup(variantPack_, workSpaceSize_, context_);
    FreeInTensorList(hostInTensors);
    if (st != ErrorType::NO_ERROR) {
        return Mki::Status::FailStatus(-1, "Setup error!");
    }
    return Mki::Status::OkStatus();
}
 

Status OperationTest::RunImpl(atb::Operation *operation, const SVector<Tensor> &inTensorLists)
{
    Cleanup();
    ATB_LOG(DEBUG) << "Cleanup success!";
    Status status = Init();
    if (status != ErrorType::NO_ERROR) {
        ATB_LOG(ERROR) << "Init error!";
        return status;
    }
    ATB_LOG(DEBUG) << "Init success!";

    status = Prepare(operation, inTensorLists);
    ATB_LOG(DEBUG) << "Prepare success!";
    if (mockFlag_) {
        return status;
    }

    if (status != ErrorType::NO_ERROR) {
        return status;
    }

    BuildVariantPack(inTensorLists);
    ATB_LOG(DEBUG) << "BuildVariantPack success!";

    status = RunOperation();
    if (status != ErrorType::NO_ERROR) {
        return status;
    }
    ATB_LOG(DEBUG) << "RunOperation success!";

    int ret = aclrtSynchronizeStream(context_->GetExecuteStream());
    if (ret != 0) {
        ATB_LOG(ERROR) << "aclrtSynchronizeStream fail, ret:" << ret;
        return ret;
    }

    status = CopyDeviceTensorToHostTensor();
    if (status != ErrorType::NO_ERROR) {
        return status;
    }

    Mki::Status st = RunGolden();
    ATB_LOG(DEBUG) << "RunGolden success!";
    if (!st.Ok()) {
        ATB_LOG(ERROR) << "golden fail, error!";
        return ErrorType::ERROR_INTERNAL_ERROR;
    }
    return status;
}

Mki::Status OperationTest::RunGolden()
{
    if (golden_) {
        OpsGoldenContext opsGoldenContext;
        opsGoldenContext.deviceInTensors.resize(goldenContext_.deviceInTensors.size());
        for (size_t i = 0; i < opsGoldenContext.deviceInTensors.size(); i++) {
            TensorUtil::ConvertAtbTensor2OpsTensor(goldenContext_.deviceInTensors.at(i),
                opsGoldenContext.deviceInTensors.at(i));
        }
        opsGoldenContext.hostInTensors.resize(goldenContext_.hostInTensors.size());
        for (size_t i = 0; i < opsGoldenContext.hostInTensors.size(); i++) {
            TensorUtil::ConvertAtbTensor2OpsTensor(goldenContext_.hostInTensors.at(i),
                opsGoldenContext.hostInTensors.at(i));
        }
        opsGoldenContext.deviceOutTensors.resize(goldenContext_.deviceOutTensors.size());
        for (size_t i = 0; i < opsGoldenContext.deviceOutTensors.size(); i++) {
            TensorUtil::ConvertAtbTensor2OpsTensor(goldenContext_.deviceOutTensors.at(i),
                opsGoldenContext.deviceOutTensors.at(i));
        }
        opsGoldenContext.hostOutTensors.resize(goldenContext_.hostOutTensors.size());
        for (size_t i = 0; i < opsGoldenContext.hostOutTensors.size(); i++) {
            TensorUtil::ConvertAtbTensor2OpsTensor(goldenContext_.hostOutTensors.at(i),
                opsGoldenContext.hostOutTensors.at(i));
        }
        return golden_(opsGoldenContext);
    }
    return Mki::Status::OkStatus();
}

Status OperationTest::CreateHostRandTensor(const TensorDesc &tensorDesc, Tensor &tensor)
{
    tensor.desc = tensorDesc;

    std::random_device rd;
    std::default_random_engine eng(rd());
    uint64_t tensorNumel = Utils::GetTensorNumel(tensor);
    ATB_LOG(DEBUG) << "Create Tensor, TensorSize:" << tensorNumel;
    if (tensorNumel == 0 || tensorNumel > MAX_NUMEL) {
        ATB_LOG(ERROR) << "tensorNumel is inValid!";
        return ERROR_INVALID_TENSOR_SIZE;
    }
    if (tensorDesc.dtype == ACL_FLOAT) {
        tensor.dataSize = tensorNumel * sizeof(float);
        tensor.hostData = malloc(tensor.dataSize);
        tensor.deviceData = malloc(tensor.dataSize);
        std::uniform_real_distribution<float> distr(randFloatMin_, randFloatMax_);
        float *tensorData = static_cast<float *>(tensor.hostData);
        for (uint64_t i = 0; i < tensorNumel; i++) {
            tensorData[i] = static_cast<float>(distr(eng));
        }
        tensorData = static_cast<float *>(tensor.deviceData);
        for (uint64_t i = 0; i < tensorNumel; i++) {
            tensorData[i] = static_cast<float>(distr(eng));
        }
    } else if (tensorDesc.dtype == ACL_FLOAT16) {
        tensor.dataSize = tensorNumel * sizeof(Mki::fp16_t);
        tensor.hostData = malloc(tensor.dataSize);
        tensor.deviceData = malloc(tensor.dataSize);
        std::uniform_real_distribution<float> distr(randFloatMin_, randFloatMax_);
        Mki::fp16_t *tensorData = static_cast<Mki::fp16_t *>(tensor.hostData);
        for (uint64_t i = 0; i < tensorNumel; i++) {
            tensorData[i] = static_cast<Mki::fp16_t>(distr(eng));
        }
        tensorData = static_cast<Mki::fp16_t *>(tensor.deviceData);
        for (uint64_t i = 0; i < tensorNumel; i++) {
            tensorData[i] = static_cast<Mki::fp16_t>(distr(eng));
        }
    } else if (tensorDesc.dtype == ACL_INT32) {
        tensor.dataSize = tensorNumel * sizeof(int32_t);
        tensor.hostData = malloc(tensor.dataSize);
        tensor.deviceData = malloc(tensor.dataSize);
        std::uniform_int_distribution<int32_t> distr(randIntMin_, randIntMax_);
        int32_t *tensorData = static_cast<int32_t *>(tensor.hostData);
        for (uint64_t i = 0; i < tensorNumel; i++) {
            tensorData[i] = static_cast<int32_t>(distr(eng));
        }
        tensorData = static_cast<int32_t *>(tensor.deviceData);
        for (uint64_t i = 0; i < tensorNumel; i++) {
            tensorData[i] = static_cast<int32_t>(distr(eng));
        }
    } else if (tensorDesc.dtype == ACL_INT64) {
        tensor.dataSize = tensorNumel * sizeof(int64_t);
        tensor.hostData = malloc(tensor.dataSize);
        tensor.deviceData = malloc(tensor.dataSize);
        std::uniform_int_distribution<int64_t> distr(randLongMin_, randLongMax_);
        int64_t *tensorData = static_cast<int64_t *>(tensor.hostData);
        for (uint64_t i = 0; i < tensorNumel; i++) {
            tensorData[i] = static_cast<int64_t>(distr(eng));
        }
        tensorData = static_cast<int64_t *>(tensor.deviceData);
        for (uint64_t i = 0; i < tensorNumel; i++) {
            tensorData[i] = static_cast<int64_t>(distr(eng));
        }
    } else if (tensorDesc.dtype == ACL_UINT32) {
        tensor.dataSize = tensorNumel * sizeof(uint32_t);
        tensor.hostData = malloc(tensor.dataSize);
        tensor.deviceData = malloc(tensor.dataSize);
        std::uniform_int_distribution<uint32_t> distr(randLongMin_, randLongMax_);
        uint32_t *tensorData = static_cast<uint32_t *>(tensor.hostData);
        for (uint64_t i = 0; i < tensorNumel; i++) {
            tensorData[i] = static_cast<uint32_t>(distr(eng));
        }
        tensorData = static_cast<uint32_t *>(tensor.deviceData);
        for (uint64_t i = 0; i < tensorNumel; i++) {
            tensorData[i] = static_cast<uint32_t>(distr(eng));
        }
    } else if (tensorDesc.dtype == ACL_INT8) {
        tensor.dataSize = tensorNumel * sizeof(int8_t);
        tensor.hostData = malloc(tensor.dataSize);
        tensor.deviceData = malloc(tensor.dataSize);
        std::uniform_int_distribution<int8_t> distr(randInt8Min_, randInt8Max_);
        int8_t *tensorData = static_cast<int8_t *>(tensor.hostData);
        for (uint64_t i = 0; i < tensorNumel; i++) {
            tensorData[i] = static_cast<int8_t>(distr(eng));
        }
        tensorData = static_cast<int8_t *>(tensor.deviceData);
        for (uint64_t i = 0; i < tensorNumel; i++) {
            tensorData[i] = static_cast<int8_t>(distr(eng));
        }
    } else {
        ATB_LOG(ERROR) << "dtype not support in CreateHostRandTensor!";
        return ERROR_INVALID_TENSOR_DTYPE;
    }
    return NO_ERROR;
}

Status OperationTest::GenerateRandomTensors(const SVector<TensorDesc> &inTensorDescs, SVector<Tensor> &inTensors)
{
    if (inTensorDescs.size() != inTensors.size()) {
        ATB_LOG(ERROR) << "TensorDescs Num not equal Tensors Num!";
        return ERROR_INVALID_TENSOR_SIZE;
    }
    for (size_t i = 0; i < inTensorDescs.size(); i++) {
        Status st = CreateHostRandTensor(inTensorDescs.at(i), inTensors.at(i));
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << "CreateHostRandTensor failed!";
            return st;
        }
        ATB_LOG(DEBUG) << "inTensor dataSize " << i << ":" << inTensors.at(i).dataSize;
    }
    return NO_ERROR;
}

void OperationTest::Golden(OpTestGolden golden)
{
    golden_ = golden;
}

void OperationTest::FloatRand(float min, float max)
{
    randFloatMin_ = min;
    randFloatMax_ = max;
    ATB_LOG(DEBUG) << "randFloatMin:" << randFloatMin_ << ", randFloatMax:" << randFloatMax_;
}

void OperationTest::Int8Rand(int8_t min, int8_t max)
{
    randInt8Min_ = min;
    randInt8Max_ = max;
    ATB_LOG(DEBUG) << "randIntMin:" << randInt8Min_ << ", randIntMax:" << randInt8Max_;
}

void OperationTest::IntRand(int32_t min, int32_t max)
{
    randIntMin_ = min;
    randIntMax_ = max;
    ATB_LOG(DEBUG) << "randIntMin:" << randIntMin_ << ", randIntMax:" << randIntMax_;
}

void OperationTest::LongRand(int64_t min, int64_t max)
{
    randLongMin_ = min;
    randLongMax_ = max;
    ATB_LOG(DEBUG) << "randIntMin:" << randLongMin_ << ", randIntMax:" << randLongMax_;
}

void OperationTest::SetMockFlag(bool flag)
{
    mockFlag_ = flag;
}

bool OperationTest::GetMockFlag()
{
    return mockFlag_;
}
} // namespace atb