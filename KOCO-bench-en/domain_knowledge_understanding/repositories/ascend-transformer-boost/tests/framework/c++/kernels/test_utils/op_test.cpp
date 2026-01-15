/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "op_test.h"

#include <fstream>
#include <iterator>
#include <random>
#include <securec.h>
#include <sstream>

#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include <mki/utils/rt/rt.h>
#include <mki/utils/time/timer.h>
#include "asdops/params/params.h"
#include "atbops/params/params.h"
#include "float_util.h"

namespace Mki {
namespace Test {
constexpr float ATOL = 0.00001;
constexpr float RTOL = 0.00001;

std::string OpTestStatistic::ToString() const
{
    std::stringstream ss;
    ss << "total:" << total << ", init:" << init << ", prepare:" << prepare << ", kernelRun:" << kernelRun
       << ", streamSync:" << streamSync << ", runGolden:" << runGolden;
    return ss.str();
}

MkiOpTest::MkiOpTest() : launchWithTiling_(true), deviceId_(0) {}

MkiOpTest::~MkiOpTest()
{
    Cleanup();
    int ret = MkiRtDeviceResetCurrent(deviceId_);
    MKI_LOG_IF(ret != 0, ERROR) << "MkiRtDeviceResetCurrent fail";
}

void MkiOpTest::Golden(OpTestGolden golden) { golden_ = golden; }

void MkiOpTest::FloatRand(float min, float max)
{
    randFloatMin_ = min;
    randFloatMax_ = max;
    MKI_LOG(INFO) << "randFloatMin:" << randFloatMin_ << ", randFloatMax:" << randFloatMax_;
}

void MkiOpTest::Int8Rand(int8_t min, int8_t max)
{
    randInt8Min_ = min;
    randInt8Max_ = max;
    MKI_LOG(INFO) << "randIntMin:" << randInt8Min_ << ", randIntMax:" << randInt8Max_;
}

void MkiOpTest::IntRand(int32_t min, int32_t max)
{
    randIntMin_ = min;
    randIntMax_ = max;
    MKI_LOG(INFO) << "randIntMin:" << randIntMin_ << ", randIntMax:" << randIntMax_;
}

void MkiOpTest::LongRand(int64_t min, int64_t max)
{
    randLongMin_ = min;
    randLongMax_ = max;
    MKI_LOG(INFO) << "randIntMin:" << randLongMin_ << ", randIntMax:" << randLongMax_;
}

Status MkiOpTest::Run(const UtOpDesc &opDesc, const TensorDesc &inTensorDesc, const std::string &kernelName)
{
    SVector<TensorDesc> inTensorDescs = {inTensorDesc};
    return Run(opDesc, inTensorDescs, kernelName);
}

Status MkiOpTest::Run(const UtOpDesc &opDesc, const SVector<TensorDesc> &inTensorDescs, const std::string &kernelName)
{
    Timer timer;
    Status status = RunImpl(opDesc, inTensorDescs, kernelName);
    statistic_.total = timer.ElapsedMicroSecond();
    MKI_LOG(INFO) << "statistic:" << statistic_.ToString();
    return status;
}

Status MkiOpTest::Run(const UtOpDesc &opDesc, const SVector<Tensor> &inTensorLists, const std::string &kernelName)
{
    Timer timer;
    Status status = RunImpl(opDesc, inTensorLists, kernelName);
    statistic_.total = timer.ElapsedMicroSecond();
    MKI_LOG(INFO) << "statistic:" << statistic_.ToString();
    return status;
}

Status MkiOpTest::RunWithDataFile(const UtOpDesc &opDesc, const SVector<TensorDesc> &inTensorDescs,
                                  const SVector<std::string> &files, const std::string &kernelName)
{
    Timer timer;
    dataFiles_ = files;
    Status status = RunImpl(opDesc, inTensorDescs, kernelName);
    statistic_.total = timer.ElapsedMicroSecond();
    MKI_LOG(INFO) << "statistic:" << statistic_.ToString();
    return status;
}

Status MkiOpTest::RunWithDataFileKVPtr(const UtOpDesc &opDesc, const SVector<TensorDesc> &inTensorDescs,
                                       const SVector<std::string> &files, const std::string &kernelName)
{
    Timer timer;
    dataFiles_ = files;
    MKI_LOG(INFO) << "kvFile size is: " << kvFile_.size();
    Status status = RunImplKVPtr(opDesc, inTensorDescs, kernelName);
    statistic_.total = timer.ElapsedMicroSecond();
    MKI_LOG(INFO) << "statistic:" << statistic_.ToString();
    return status;
}

Status MkiOpTest::RunImplKVPtr(const UtOpDesc &opDesc, const SVector<TensorDesc> &inTensorDescs,
                               const std::string &kernelName)
{
    Timer timer;
    statistic_.init = timer.ElapsedMicroSecond();
    
    timer.Reset();
    Status status = Prepare(opDesc, inTensorDescs);
    statistic_.prepare = timer.ElapsedMicroSecond();
    if (!status.Ok()) {
        return status;
    }
    status = RunKernel(kernelName);
    if (!status.Ok()) {
        return status;
    }

    status = CopyDeviceTensorToHostTensor();
    if (!status.Ok()) {
        return status;
    }

    timer.Reset();
    if (isRelay) {
        status = RunGolden();
        statistic_.runGolden = timer.ElapsedMicroSecond();
        if (!status.Ok()) {
            MKI_LOG(ERROR) << "golden fail, error:" << status.ToString();
            return status;
        }
    }
    return Status::OkStatus();
}

Status MkiOpTest::RunImpl(const UtOpDesc &opDesc, const SVector<TensorDesc> &inTensorDescs,
                          const std::string &kernelName)
{
    Timer timer;
    Cleanup();
    Init();
    statistic_.init = timer.ElapsedMicroSecond();

    timer.Reset();
    Status status = Prepare(opDesc, inTensorDescs);
    statistic_.prepare = timer.ElapsedMicroSecond();
    if (!status.Ok()) {
        return status;
    }

    status = RunKernel(kernelName);
    if (!status.Ok()) {
        return status;
    }

    status = CopyDeviceTensorToHostTensor();
    if (!status.Ok()) {
        return status;
    }

    timer.Reset();
    status = RunGolden();
    statistic_.runGolden = timer.ElapsedMicroSecond();
    if (!status.Ok()) {
        MKI_LOG(ERROR) << "golden fail, error:" << status.ToString();
        return status;
    }
    return Status::OkStatus();
}

Status MkiOpTest::RunImpl(const UtOpDesc &opDesc, const SVector<Tensor> &inTensorLists, const std::string &kernelName)
{
    Timer timer;
    Cleanup();
    Init();
    statistic_.init = timer.ElapsedMicroSecond();

    timer.Reset();
    Status status = Prepare(opDesc, inTensorLists);
    statistic_.prepare = timer.ElapsedMicroSecond();
    if (!status.Ok()) {
        return status;
    }

    status = RunKernel(kernelName);
    if (!status.Ok()) {
        return status;
    }

    status = CopyDeviceTensorToHostTensor();
    if (!status.Ok()) {
        return status;
    }

    timer.Reset();
    status = RunGolden();
    statistic_.runGolden = timer.ElapsedMicroSecond();
    if (!status.Ok()) {
        MKI_LOG(ERROR) << "golden fail, error:" << status.ToString();
        return status;
    }
    return Status::OkStatus();
}

std::vector<std::vector<uint64_t>> MkiOpTest::PrepareKVshapeInfos(uint32_t batchNum, const TensorDesc &kvTensorDesc)
{
    SVector<int64_t> shape = kvTensorDesc.dims;
    std::vector<std::vector<uint64_t>> shapeInfos(batchNum, std::vector<uint64_t>(kvTensorDesc.Numel(), 0));
    for (auto &s : shapeInfos) {
        for (int64_t i = 0; i < kvTensorDesc.Numel(); ++i) {
            s[i] = shape.at(i);
        }
    }
    return shapeInfos;
}

void MkiOpTest::PrepareKVcacheBatchwiseTensors(uint32_t batchNum, const TensorDesc &kvTensorDescIn,
                                               const std::vector<std::string> &kvfiles, uint32_t share_len)
{
    Cleanup();
    Init();
    kvFile_ = kvfiles;
    std::vector<std::uint8_t*> kvCacheBatchwisePtr;
    for (size_t i = 0; i < batchNum; ++i) {
        Tensor tensor = {kvTensorDescIn, nullptr, 0};
        kCacheBatchWise.push_back(tensor);
        vCacheBatchWise.push_back(tensor);
    }
    for (size_t i = 0; i < batchNum; ++i) {
        Tensor &deviceTensor = kCacheBatchWise[i];
        Tensor hostTensor;
        hostTensor = CreateHostTensorFromFile(deviceTensor.desc, kvFile_[i]);
        deviceTensor.dataSize = hostTensor.dataSize;
        deviceTensor = HostTensor2DeviceTensor(hostTensor);
    }
    for (size_t i = 0; i < batchNum; ++i) {
        Tensor &deviceTensor = vCacheBatchWise[i];
        Tensor hostTensor;
        hostTensor = CreateHostTensorFromFile(deviceTensor.desc, kvFile_[i + batchNum]);
        deviceTensor.dataSize = hostTensor.dataSize;
        deviceTensor = HostTensor2DeviceTensor(hostTensor);
    }
    if (isRelay) {
        for (size_t i = 0; i < share_len; ++i) {
            TensorDesc kvTensorDesc = kvTensorDescIn;
            kvTensorDesc.dims[0] = shareLen[i];
            Tensor tensor = {kvTensorDesc, nullptr, 0};
            kSepCacheBatchWise.push_back(tensor);
            vSepCacheBatchWise.push_back(tensor);
        }
        for (size_t i = 0; i < share_len; ++i) {
            Tensor &deviceTensor = kSepCacheBatchWise[i];
            Tensor hostTensor;
            hostTensor = CreateHostTensorFromFile(deviceTensor.desc, kvFile_[i + batchNum * 2]);
            deviceTensor.dataSize = hostTensor.dataSize;
            deviceTensor = HostTensor2DeviceTensor(hostTensor);
        }
        for (size_t i = 0; i < share_len; ++i) {
            Tensor &deviceTensor = vSepCacheBatchWise[i];
            Tensor hostTensor;
            hostTensor = CreateHostTensorFromFile(deviceTensor.desc, kvFile_[i + batchNum * 2 + share_len]);
            deviceTensor.dataSize = hostTensor.dataSize;
            deviceTensor = HostTensor2DeviceTensor(hostTensor);
        }
    }
}

Status MkiOpTest::Prepare(const UtOpDesc &opDesc, const SVector<TensorDesc> &inTensorDescs)
{
    op_ = Mki::AutoGen::GetOpByName(opDesc.opName);
    if (op_ == nullptr) {
        MKI_LOG(ERROR) << opDesc.opName << " not exists";
        return Status::FailStatus(-1, opDesc.opName + " not exists");
    }

    goldenContext_.opDesc = opDesc;
    launchParam_.SetParam(opDesc.specificParam);

    for (auto &tensorDesc : inTensorDescs) {
        Tensor tensor = {tensorDesc, nullptr, 0};
        launchParam_.AddInTensor(tensor);
    }

    int outTensorNum = GetOutputNum(opDesc);
    for (int i = 0; i < outTensorNum; ++i) {
        Tensor tensor;
        launchParam_.AddOutTensor(tensor);
    }

    if (kCacheBatchWise.empty()) {
        MallocInTensor();
    } else {
        MallocInTensorKVPtr();
    }

    Status status = op_->InferShape(launchParam_);
    if (!status.Ok()) {
        MKI_LOG(ERROR) << opDesc.opName << " infer shape fail, error:" << status.ToString();
        return status;
    }

    MallocOutTensor();

    if (!op_->IsConsistent(launchParam_)) {
        MKI_LOG(ERROR) << op_->GetName() << " check launchParam is consistent fail";
        return Status::FailStatus(-1, op_->GetName() + " check launchParam is consistent fail");
    }

    return Status::OkStatus();
}


Status MkiOpTest::Prepare(const UtOpDesc &opDesc, const SVector<Tensor> &inTensorLists)
{
    op_ = Mki::AutoGen::GetOpByName(opDesc.opName);
    if (op_ == nullptr) {
        MKI_LOG(ERROR) << opDesc.opName << " not exists";
        return Status::FailStatus(-1, opDesc.opName + " not exists");
    }

    goldenContext_.opDesc = opDesc;
    launchParam_.SetParam(opDesc.specificParam);

    for (auto tensor : inTensorLists) {
        launchParam_.AddInTensor(tensor);
    }

    int outTensorNum = GetOutputNum(opDesc);
    for (int i = 0; i < outTensorNum; ++i) {
        Tensor tensor;
        launchParam_.AddOutTensor(tensor);
    }

    MallocInTensor(inTensorLists);

    Status status = op_->InferShape(launchParam_);
    if (!status.Ok()) {
        MKI_LOG(ERROR) << opDesc.opName << " infer shape fail, error:" << status.ToString();
        return status;
    }

    MallocOutTensor();

    if (!op_->IsConsistent(launchParam_)) {
        MKI_LOG(ERROR) << op_->GetName() << " check launchParam is consistent fail";
        return Status::FailStatus(-1, op_->GetName() + " check launchParam is consistent fail");
    }

    return Status::OkStatus();
}

void MkiOpTest::MallocInTensor()
{
    for (size_t i = 0; i < launchParam_.GetInTensorCount(); ++i) {
        Tensor &deviceTensor = launchParam_.GetInTensor(i);
        Tensor hostTensor;
        if (dataFiles_.size() > i) {
            hostTensor = CreateHostTensorFromFile(deviceTensor.desc, dataFiles_[i]);
        } else {
            hostTensor = CreateHostRandTensor(deviceTensor.desc);
        }
        deviceTensor.dataSize = hostTensor.dataSize;
        deviceTensor = HostTensor2DeviceTensor(hostTensor);
        MKI_LOG(INFO) << "InTensor[" << i << "] = " << TensorToString(hostTensor);
        goldenContext_.hostInTensors.push_back(hostTensor);
        goldenContext_.deviceInTensors.push_back(deviceTensor);
    }
}

void MkiOpTest::CreateTensorListForKVPtr(const uint32_t batchNum, std::vector<uint8_t *>& inTensorPtrs,
                                         Tensor &deviceTensor)
{
    int st = MkiRtMemMallocDevice(&deviceTensor.data, deviceTensor.dataSize, MKIRT_MEM_DEFAULT);
    if (st != 0) {
        MKI_LOG(ERROR) << "malloc device tensor fail";
    }
    std::vector<uint64_t> listTensorDesc;
    listTensorDesc.push_back(3 * sizeof(uint64_t));
    listTensorDesc.push_back((uint64_t)batchNum << 32);
    listTensorDesc.push_back(0xffffffff);
    for (size_t i = 0; i < batchNum; ++i) {
        listTensorDesc.push_back(reinterpret_cast<uint64_t>(inTensorPtrs[i]));
    }
    st = MkiRtMemCopy(deviceTensor.data, deviceTensor.dataSize, listTensorDesc.data(), deviceTensor.dataSize,
                      MKIRT_MEMCOPY_HOST_TO_DEVICE);
    if (st != 0) {
        MKI_LOG(ERROR) << "copy listTensorDesc to device as a tensor failed";
    }
}

void MkiOpTest::MallocInTensorKVPtr()
{
    for (size_t i = 0; i < launchParam_.GetInTensorCount(); ++i) {
        Tensor &deviceTensor = launchParam_.GetInTensor(i);      
        Tensor hostTensor;
        if (isRelay) {
            if (dataFiles_.size() > i) {
                hostTensor = CreateHostTensorFromFile(deviceTensor.desc, dataFiles_[i]);              
            } else {
                hostTensor = CreateHostRandTensor(deviceTensor.desc);
            }
            deviceTensor.dataSize = hostTensor.dataSize;
            deviceTensor = HostTensor2DeviceTensor(hostTensor);   
        } else {
            if (i != 1 && i != 2) {
                if (dataFiles_.size() > i) {
                    hostTensor = CreateHostTensorFromFile(deviceTensor.desc, dataFiles_[i]);              
                } else {
                    hostTensor = CreateHostRandTensor(deviceTensor.desc);
                }
                deviceTensor.dataSize = hostTensor.dataSize;
                deviceTensor = HostTensor2DeviceTensor(hostTensor);       
            }
        }
        goldenContext_.hostInTensors.push_back(hostTensor);
        goldenContext_.deviceInTensors.push_back(deviceTensor);
    }
}

void MkiOpTest::MallocInTensor(const SVector<Tensor> &inTensorLists)
{
    if (launchParam_.GetInTensorCount() != inTensorLists.size()) {
        MKI_LOG(ERROR) << "MallocInTensor ERROR";
        return;
    }
    for (size_t i = 0; i < launchParam_.GetInTensorCount(); ++i) {
        Tensor &deviceTensor = launchParam_.GetInTensor(i);
        Tensor hostTensor = CreateHostTensor(inTensorLists.at(i));
        deviceTensor.dataSize = hostTensor.dataSize;
        deviceTensor = HostTensor2DeviceTensor(hostTensor);
        MKI_LOG(INFO) << "InTensor[" << i << "] = " << TensorToString(hostTensor);
        goldenContext_.hostInTensors.push_back(hostTensor);
        goldenContext_.deviceInTensors.push_back(deviceTensor);
    }
}

void MkiOpTest::MallocOutTensor()
{
    for (size_t i = 0; i < launchParam_.GetOutTensorCount(); ++i) {
        Tensor &deviceTensor = launchParam_.GetOutTensor(i);
        Tensor hostTensor = CreateHostZeroTensor(deviceTensor.desc);
        deviceTensor.dataSize = hostTensor.dataSize;
        if (outUseInputdata_.find(i) != outUseInputdata_.end()) {
            deviceTensor.data = launchParam_.GetInTensor(outUseInputdata_[i]).data;
        } else {
            int st = MkiRtMemMallocDevice(&deviceTensor.data, deviceTensor.dataSize, MKIRT_MEM_DEFAULT);
            MKI_LOG_IF(st != MKIRT_SUCCESS, ERROR) << "malloc device tensor fail";

            // init deviceTensor zero
            st = MkiRtMemCopy(deviceTensor.data, deviceTensor.dataSize, hostTensor.data, hostTensor.dataSize,
                              MKIRT_MEMCOPY_HOST_TO_DEVICE);
            MKI_LOG_IF(st != MKIRT_SUCCESS, ERROR) << "init device tensor 0 fail";
        }
        goldenContext_.hostOutTensors.push_back(hostTensor);
        goldenContext_.deviceOutTensors.push_back(deviceTensor);
    }
}

void MkiOpTest::SetOutdataUseInputData(size_t outIndex, size_t inIndex) { outUseInputdata_[outIndex] = inIndex; }

Status MkiOpTest::RunKernel(const std::string &kernelName)
{
    std::unique_ptr<Kernel> kernel(GetKernel(kernelName));
    if (kernel == nullptr) {
        return Status::FailStatus(-1, "get kernel fail");
    }
    MKI_LOG(INFO) << "use kernel " << kernel->GetName();

    uint8_t *deviceLaunchBuffer = nullptr;

    if (launchWithTiling_) {
        kernel->SetLaunchWithTiling(true);
        kernel->Init(launchParam_);
    } else {
        kernel->SetLaunchWithTiling(false);
        uint32_t launchBufferSize = kernel->GetTilingSize(launchParam_);
        if (launchBufferSize == 0) {
            MKI_LOG(ERROR) << "empty tiling size";
            return Status::FailStatus(-1, "empty tiling size");
        }
        uint8_t hostLaunchBuffer[launchBufferSize];
        kernel->SetTilingHostAddr(hostLaunchBuffer, launchBufferSize);
        kernel->Init(launchParam_);

        int st = MkiRtMemMallocDevice(reinterpret_cast<void **>(&deviceLaunchBuffer), launchBufferSize, MKIRT_MEM_DEFAULT);
        if (st != MKIRT_SUCCESS) {
            MKI_LOG(ERROR) << "malloc device memory fail";
            return Status::FailStatus(-1, "malloc device memory fail");
        }
        st = MkiRtMemCopy(deviceLaunchBuffer, launchBufferSize,
            hostLaunchBuffer, launchBufferSize, MKIRT_MEMCOPY_HOST_TO_DEVICE);
        if (st != MKIRT_SUCCESS) {
            MkiRtMemFreeDevice(deviceLaunchBuffer);
            deviceLaunchBuffer = nullptr;
            MKI_LOG(ERROR) << "copy host memory to device fail";
            return Status::FailStatus(-1, "copy host memory to device fail");
        }
        runInfo_.SetTilingDeviceAddr(deviceLaunchBuffer);
    }
    const KernelInfo &kernelInfo = kernel->GetKernelInfo();
    AddWorkspace(kernelInfo);

    MKI_LOG(INFO) << kernel->GetName() << " run start, LaunchParam:\n" << launchParam_.ToString();
    MKI_LOG(INFO) << "RunInfo:\n" << runInfo_.ToString();

    Timer timer;
    Status status = kernel->Run(launchParam_, runInfo_);
    statistic_.kernelRun = timer.ElapsedMicroSecond();
    if (status.Ok()) {
        MKI_LOG(INFO) << kernel->GetName() << " run success";
    } else {
        MKI_LOG(ERROR) << kernel->GetName() << " run fail, error:" << status.ToString();
    }

    timer.Reset();
    int ret = MkiRtStreamSynchronize(runInfo_.GetStream());
    statistic_.streamSync = timer.ElapsedMicroSecond();
    MKI_LOG(INFO) << "sync ret " << ret;
    MKI_LOG_IF(ret != 0, ERROR) << "MkiRtStreamSynchronize fail";

    if (deviceLaunchBuffer != nullptr) {
        MkiRtMemFreeDevice(deviceLaunchBuffer);
    }

    FreeWorkspace(kernelInfo);

    return status;
}

Kernel *MkiOpTest::GetKernel(const std::string &kernelName)
{
    Kernel *kernel = nullptr;
    if (kernelName != "") {
        kernel = op_->GetKernelByName(kernelName);
        if (kernel == nullptr) {
            MKI_LOG(ERROR) << "kernel " << kernelName << "not exists";
            return nullptr;
        }
        MKI_LOG(INFO) << "get kernel by name:" << kernelName << " success";
    } else {
        kernel = op_->GetBestKernel(launchParam_);
        if (kernel == nullptr) {
            MKI_LOG(ERROR) << op_->GetName() << " get best kernel fail";
            return nullptr;
        }
    }

    return kernel;
}

Status MkiOpTest::CopyDeviceTensorToHostTensor()
{
    for (size_t i = 0; i < launchParam_.GetOutTensorCount(); ++i) {
        Tensor &deivceTensor = launchParam_.GetOutTensor(i);
        Tensor &hostTensor = goldenContext_.hostOutTensors.at(i);
        MKI_LOG(INFO) << "MkiRtMemCopy start, hostTensor.data:" << hostTensor.data
                      << ", hostTensor.dataSize:" << hostTensor.dataSize << ", deivceTensor.data:" << deivceTensor.data
                      << ", deivceTensor.dataSize:" << deivceTensor.dataSize;
        int st = MkiRtMemCopy(hostTensor.data, hostTensor.dataSize, deivceTensor.data, deivceTensor.dataSize,
                              MKIRT_MEMCOPY_DEVICE_TO_HOST);
        if (st != 0) {
            MKI_LOG(ERROR) << "copy memory from device to host fail";
            return Status::FailStatus(-1, "copy memory from device to host fail");
        }
        MKI_LOG(INFO) << "OutTensor[" << i << "] = " << TensorToString(hostTensor);
    }
    return Status::OkStatus();
}

void MkiOpTest::SetOutputNum(int64_t outputNum) { outputNum_ = outputNum; }

int64_t MkiOpTest::GetOutputNum(const UtOpDesc &opDesc)
{
    if (outputNum_ > 0) {
        return outputNum_;
    }
    return op_->GetOutputNum(opDesc.specificParam);
}

Status MkiOpTest::RunGolden()
{
    if (golden_) {
        return golden_(goldenContext_);
    }
    return Status::OkStatus();
}

Tensor MkiOpTest::CreateHostRandTensor(const TensorDesc &tensorDesc)
{
    Tensor tensor;
    tensor.desc = tensorDesc;

    std::random_device rd;
    std::default_random_engine eng(rd());
    if (tensorDesc.dtype == TENSOR_DTYPE_FLOAT) {
        tensor.dataSize = tensor.Numel() * sizeof(float);
        tensor.data = malloc(tensor.dataSize);
        std::uniform_real_distribution<float> distr(randFloatMin_, randFloatMax_);
        float *tensorData = static_cast<float *>(tensor.data);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = static_cast<float>(distr(eng));
        }
    } else if (tensorDesc.dtype == TENSOR_DTYPE_FLOAT16) {
        tensor.dataSize = tensor.Numel() * sizeof(fp16_t);
        tensor.data = malloc(tensor.dataSize);
        std::uniform_real_distribution<float> distr(randFloatMin_, randFloatMax_);
        fp16_t *tensorData = static_cast<fp16_t *>(tensor.data);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = static_cast<fp16_t>(distr(eng));
        }
    } else if (tensorDesc.dtype == TENSOR_DTYPE_INT32) {
        tensor.dataSize = tensor.Numel() * sizeof(int32_t);
        tensor.data = malloc(tensor.dataSize);
        std::uniform_int_distribution<int32_t> distr(randIntMin_, randIntMax_);
        int32_t *tensorData = static_cast<int32_t *>(tensor.data);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = static_cast<int32_t>(distr(eng));
        }
    } else if (tensorDesc.dtype == TENSOR_DTYPE_INT64) {
        tensor.dataSize = tensor.Numel() * sizeof(int64_t);
        tensor.data = malloc(tensor.dataSize);
        std::uniform_int_distribution<int64_t> distr(randLongMin_, randLongMax_);
        int64_t *tensorData = static_cast<int64_t *>(tensor.data);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = static_cast<int64_t>(distr(eng));
        }
    } else if (tensorDesc.dtype == TENSOR_DTYPE_UINT32) {
        tensor.dataSize = tensor.Numel() * sizeof(uint32_t);
        tensor.data = malloc(tensor.dataSize);
        std::uniform_int_distribution<uint32_t> distr(randIntMin_, randIntMax_);
        uint32_t *tensorData = static_cast<uint32_t *>(tensor.data);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = static_cast<uint32_t>(distr(eng));
        }
    } else if (tensorDesc.dtype == TENSOR_DTYPE_INT8) {
        tensor.dataSize = tensor.Numel() * sizeof(int8_t);
        tensor.data = malloc(tensor.dataSize);
        std::uniform_int_distribution<int8_t> distr(randInt8Min_, randInt8Max_);
        int8_t *tensorData = static_cast<int8_t *>(tensor.data);
        for (int64_t i = 0; i < tensor.Numel(); i++) {
            tensorData[i] = static_cast<int8_t>(distr(eng));
        }
    } else {
        MKI_LOG(ERROR) << "dtype not support in CreateHostRandTensor!";
    }

    return tensor;
}

Tensor MkiOpTest::CreateHostTensor(const Tensor &tensorIn)
{
    Tensor tensor;
    tensor.desc = tensorIn.desc;
    tensor.dataSize = tensorIn.dataSize;

    tensor.data = malloc(tensor.dataSize);
    auto ret = memcpy_s(tensor.data, tensor.dataSize, tensorIn.data, tensor.dataSize);
    MKI_LOG_IF(ret != EOK, ERROR) << "CreateHostTensor fail";

    return tensor;
}

Tensor MkiOpTest::CreateHostZeroTensor(const TensorDesc &tensorDesc)
{
    Tensor tensor;
    tensor.desc = tensorDesc;

    std::random_device rd;
    std::default_random_engine eng(rd());
    if (tensorDesc.dtype == TENSOR_DTYPE_FLOAT) {
        tensor.dataSize = tensor.Numel() * sizeof(float);
    } else if (tensorDesc.dtype == TENSOR_DTYPE_FLOAT16) {
        tensor.dataSize = tensor.Numel() * sizeof(fp16_t);
    } else if (tensorDesc.dtype == TENSOR_DTYPE_INT64) {
        tensor.dataSize = tensor.Numel() * sizeof(int64_t);
    } else if (tensorDesc.dtype == TENSOR_DTYPE_INT32) {
        tensor.dataSize = tensor.Numel() * sizeof(int32_t);
    } else if (tensorDesc.dtype == TENSOR_DTYPE_UINT32) {
        tensor.dataSize = tensor.Numel() * sizeof(uint32_t);
    } else if (tensorDesc.dtype == TENSOR_DTYPE_INT8) {
        tensor.dataSize = tensor.Numel() * sizeof(int8_t);
    } else if (tensorDesc.dtype == TENSOR_DTYPE_UINT8) {
        tensor.dataSize = tensor.Numel() * sizeof(uint8_t);
    } else if (tensorDesc.dtype == TENSOR_DTYPE_BF16) {
        tensor.dataSize = tensor.Numel() * sizeof(uint16_t);
    } else {
        MKI_LOG(ERROR) << "not support";
        return tensor;
    }

    tensor.data = malloc(tensor.dataSize);
    auto ret = memset_s(tensor.data, tensor.dataSize, 0, tensor.dataSize);
    MKI_LOG_IF(ret != EOK, ERROR) << "CreateHostZeroTensor fail";
    return tensor;
}

Tensor MkiOpTest::CreateHostTensorFromFile(const TensorDesc &tensorDesc, const std::string &dataFile)
{
    Tensor tensor;
    tensor.desc = tensorDesc;
    if (tensorDesc.dtype == TENSOR_DTYPE_FLOAT || tensorDesc.dtype == TENSOR_DTYPE_INT32) {
        tensor.dataSize = tensor.Numel() * sizeof(float);
    } else if (tensorDesc.dtype == TENSOR_DTYPE_FLOAT16) {
        tensor.dataSize = tensor.Numel() * sizeof(fp16_t);
    } else if (tensorDesc.dtype == TENSOR_DTYPE_INT64) {
        tensor.dataSize = tensor.Numel() * sizeof(int64_t);
    } else if (tensorDesc.dtype == TENSOR_DTYPE_UINT32) {
        tensor.dataSize = tensor.Numel() * sizeof(uint32_t);
    } else if (tensorDesc.dtype == TENSOR_DTYPE_BF16) {
        tensor.dataSize = tensor.Numel() * sizeof(uint16_t);
    } else {
        MKI_LOG(ERROR) << "not support";
        return tensor;
    }

    tensor.data = malloc(tensor.dataSize);
    memset(tensor.data, 0, tensor.dataSize);
    ReadFile(tensor.data, tensor.dataSize, dataFile);

    return tensor;
}

Tensor MkiOpTest::HostTensor2DeviceTensor(const Tensor &hostTensor)
{
    Tensor deviceTensor;
    deviceTensor.desc = hostTensor.desc;
    deviceTensor.dataSize = hostTensor.dataSize;
    int st = MkiRtMemMallocDevice(&deviceTensor.data, deviceTensor.dataSize, MKIRT_MEM_DEFAULT);
    if (st != 0) {
        MKI_LOG(ERROR) << "malloc device tensor fail";
        return deviceTensor;
    }
    st = MkiRtMemCopy(deviceTensor.data, deviceTensor.dataSize, hostTensor.data, hostTensor.dataSize,
                      MKIRT_MEMCOPY_HOST_TO_DEVICE);
    if (st != 0) {
        MKI_LOG(ERROR) << "copy host tensor to device tensor";
    }
    return deviceTensor;
}

void MkiOpTest::Init()
{
    MKI_LOG(INFO) << "MkiRtDeviceSetCurrent " << deviceId_;
    int ret = MkiRtDeviceSetCurrent(deviceId_);
    MKI_LOG_IF(ret != 0, ERROR) << "MkiRtDeviceSetCurrent fail";

    MkiRtStream stream = nullptr;
    MKI_LOG(INFO) << "MkiRtStreamCreate call";
    ret = MkiRtStreamCreate(&stream, 0);
    MKI_LOG_IF(ret != 0, ERROR) << "MkiRtStreamCreate fail";

    runInfo_.SetStream(stream);
}

void MkiOpTest::AddWorkspace(const KernelInfo &kernelInfo)
{
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    if (bufferSize == 0) {
        MKI_LOG(INFO) << "no workspace";
        return;
    }
    MKI_LOG(INFO) << "Workspace size: " << bufferSize;
    uint8_t *deviceBuffer = nullptr;
    int ret = MkiRtMemMallocDevice(reinterpret_cast<void **>(&deviceBuffer), bufferSize, MKIRT_MEM_DEFAULT);
    if (ret != MKIRT_SUCCESS) {
        MKI_LOG(ERROR) << "MkiRtMemMallocDevice fail, errCode:" << ret << ", errName:" << MkiRtErrorName(ret)
                        << "errDesc:" << MkiRtErrorDesc(ret);
        return;
    }
    runInfo_.SetScratchDeviceAddr(deviceBuffer);
}

void MkiOpTest::FreeWorkspace(const KernelInfo &kernelInfo)
{
    uint8_t *deviceBuffer = runInfo_.GetScratchDeviceAddr();
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    if (bufferSize == 0) {
        return;
    }
    if (deviceBuffer != nullptr) {
        MkiRtMemFreeDevice(deviceBuffer);
    }
}

void MkiOpTest::Cleanup()
{
    for (auto tensor : goldenContext_.hostInTensors) {
        if (tensor.data) {
            free(tensor.data);
            tensor.data = nullptr;
        }
    }
    goldenContext_.hostInTensors.clear();

    for (auto tensor : goldenContext_.hostOutTensors) {
        if (tensor.data) {
            free(tensor.data);
            tensor.data = nullptr;
        }
    }
    goldenContext_.hostOutTensors.clear();

    for (auto tensor : goldenContext_.deviceInTensors) {
        if (tensor.data) {
            MkiRtMemFreeDevice(tensor.data);
        }
    }
    goldenContext_.deviceInTensors.clear();

    size_t i = 0;
    for (auto tensor : goldenContext_.deviceOutTensors) {
        if (outUseInputdata_.find(i++) != outUseInputdata_.end()) {
            continue;
        }
        if (tensor.data) {
            MkiRtMemFreeDevice(tensor.data);
        }
    }
    goldenContext_.deviceOutTensors.clear();

    MkiRtStream stream = runInfo_.GetStream();
    if (stream) {
        MkiRtStreamDestroy(stream);
    }

    launchParam_.Reset();
    runInfo_.Reset();
}

OpTestStatistic MkiOpTest::GetRunStatistic() const { return statistic_; }

void MkiOpTest::ReadFile(void *data, size_t dataSize, const std::string &dataFile)
{
    std::ifstream input(dataFile, std::ios::binary);
    if (!input.is_open()) {
        MKI_LOG(ERROR) << "Failed to open file " << dataFile;
        return;
    }

    // copies all data into buffer
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
    if (buffer.size() != dataSize) {
        MKI_FLOG_ERROR("File size %zu is invalid, expect %zu", buffer.size(), dataSize);
        return;
    }
    MKI_FLOG_INFO("File %s read, total size %zu", dataFile.c_str(), buffer.size());
    auto ret = memcpy_s(data, dataSize, buffer.data(), dataSize);
    MKI_LOG_IF(ret != EOK, ERROR) << "ReadFile fail";
}

void MkiOpTest::LogInferShapeResult()
{
    for (size_t i = 0; i < launchParam_.GetInTensorCount(); ++i) {
        Tensor &tensor = launchParam_.GetInTensor(i);
        MKI_LOG(INFO) << "InferShape result InTensor[" << i << "] = " << TensorToString(tensor);
    }

    for (size_t i = 0; i < launchParam_.GetOutTensorCount(); ++i) {
        Tensor &tensor = launchParam_.GetOutTensor(i);
        MKI_LOG(INFO) << "InferShape result OutTensor[" << i << "] = " << TensorToString(tensor);
    }
}

std::string MkiOpTest::TensorToString(const Tensor &tensor)
{
    const int64_t printMaxCount = 10;
    std::ostringstream ss;
    ss << "dtype:" << GetStrWithDType(tensor.desc.dtype) << ", format:" << GetStrWithFormat(tensor.desc.format)
       << ", numel:" << tensor.Numel() << ", dataSize:" << tensor.dataSize << ", data:[";

    if (tensor.data) {
        for (int64_t i = 0; i < tensor.Numel(); ++i) {
            if (i == printMaxCount) {
                ss << "...";
                break;
            }

            if (tensor.desc.dtype == TENSOR_DTYPE_FLOAT16) {
                fp16_t *tensorData = static_cast<fp16_t *>(tensor.data);
                ss << tensorData[i] << ",";
            } else if (tensor.desc.dtype == TENSOR_DTYPE_FLOAT) {
                float *tensorData = static_cast<float *>(tensor.data);
                ss << tensorData[i] << ",";
            } else if (tensor.desc.dtype == TENSOR_DTYPE_INT32) {
                int32_t *tensorData = static_cast<int32_t *>(tensor.data);
                ss << tensorData[i] << ",";
            } else if (tensor.desc.dtype == TENSOR_DTYPE_INT64) {
                int64_t *tensorData = static_cast<int64_t *>(tensor.data);
                ss << tensorData[i] << ",";
            } else if (tensor.desc.dtype == TENSOR_DTYPE_INT8) {
                int8_t *tensorData = static_cast<int8_t *>(tensor.data);
                ss << static_cast<int>(tensorData[i]) << ",";
            } else if (tensor.desc.dtype == TENSOR_DTYPE_UINT32) {
                uint32_t *tensorData = static_cast<uint32_t *>(tensor.data);
                ss << tensorData[i] << ",";
            } else {
                ss << "N,";
            }
        }
    } else {
        ss << "null";
    }

    ss << "]";
    return ss.str();
}
} // namespace Test
} // namespace Mki