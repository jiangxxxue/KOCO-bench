/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MKIOPTESTS_OPTEST_H
#define MKIOPTESTS_OPTEST_H

#include <array>
#include <float.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <mki/kernel.h>
#include <mki/launch_param.h>
#include <mki/operation.h>
#include <mki/kernel_info.h>
#include <mki/run_info.h>
#include <mki/utils/any/any.h>
#include <mki/utils/SVector/SVector.h>
#include "op_desc_json.h"

namespace Mki {
namespace Test {
const int32_t MAX_INPUT_TENSOR = 16;

struct UtOpDesc {
    std::string opName;
    Any specificParam;
};

struct GoldenContext {
    UtOpDesc opDesc;
    SVector<Tensor> hostInTensors;
    SVector<Tensor> hostOutTensors;
    SVector<Tensor> deviceInTensors;
    SVector<Tensor> deviceOutTensors;
};

using OpTestGolden = std::function<Status(const GoldenContext &context)>;

struct OpTestStatistic {
    uint64_t total = 0;
    uint64_t init = 0;
    uint64_t prepare = 0;
    uint64_t kernelRun = 0;
    uint64_t streamSync = 0;
    uint64_t runGolden = 0;
    std::string ToString() const;
};

class MkiOpTest {
public:
    explicit MkiOpTest();
    ~MkiOpTest();
    void Golden(OpTestGolden golden);
    void FloatRand(float min, float max);
    void Int8Rand(int8_t min, int8_t max);
    void IntRand(int32_t min, int32_t max);
    void LongRand(int64_t min, int64_t max);
    Status Run(const UtOpDesc &opDesc, const TensorDesc &inTensorDesc, const std::string &kernelName = "");
    Status Run(const UtOpDesc &opDesc, const SVector<TensorDesc> &inTensorDescs, const std::string &kernelName = "");
    Status Run(const UtOpDesc &opDesc, const SVector<Tensor> &inTensorLists, const std::string &kernelName = "");
    Status RunWithDataFile(const UtOpDesc &opDesc, const SVector<TensorDesc> &inTensorDescs,
                           const SVector<std::string> &files, const std::string &kernelName = "");
    Status RunWithDataFileKVPtr(const UtOpDesc &opDesc, const SVector<TensorDesc> &inTensorDescs, 
                               const SVector<std::string> &files,
                               const std::string &kernelName = "");
    OpTestStatistic GetRunStatistic() const;

    void ReadFile(void *data, size_t dataSize, const std::string &dataFile);

    void SetOutdataUseInputData(size_t outIndex, size_t inIndex);
    void SetOutputNum(int64_t outputNum);
    void PrepareKVcacheBatchwiseTensors(uint32_t batchNum, const TensorDesc &kvTensorDesc, const std::vector<std::string> &kvfiles, uint32_t share_len = 0);
    std::vector<std::vector<uint64_t>> PrepareKVshapeInfos(const std::vector<TensorDesc> &kvTensorDescs);
    std::vector<std::vector<uint64_t>> PrepareKVshapeInfos(uint32_t batchNum, const TensorDesc &kvTensorDesc);
    void CreateTensorListForKVPtr(const uint32_t batchNum, std::vector<uint8_t *>& inTensorPtrs, Tensor &deviceTensor);
    std::vector<Tensor> kCacheBatchWise;
    std::vector<Tensor> vCacheBatchWise;
    std::vector<Tensor> kSepCacheBatchWise;
    std::vector<Tensor> vSepCacheBatchWise;
    std::vector<int> shareIdx;
    std::vector<int> shareLen;
    bool isRelay = false;

private:
    Status RunImplKVPtr(const UtOpDesc &opDesc, const SVector<TensorDesc> &inTensorDescs, 
                        const std::string &kernelName);
    Status RunImpl(const UtOpDesc &opDesc, const SVector<TensorDesc> &inTensorDescs, const std::string &kernelName);
    Status RunImpl(const UtOpDesc &opDesc, const SVector<Tensor> &inTensorLists, const std::string &kernelName);
    Tensor CreateHostRandTensor(const TensorDesc &tensorDesc);
    Tensor CreateHostZeroTensor(const TensorDesc &tensorDesc);
    Tensor CreateHostTensorFromFile(const TensorDesc &tensorDesc, const std::string &dataFile);
    Tensor CreateHostTensor(const Tensor &tensorIn);
    Tensor HostTensor2DeviceTensor(const Tensor &hostTensor);
    Status Prepare(const UtOpDesc &opDesc, const SVector<TensorDesc> &inTensorDescs);
    Status Prepare(const UtOpDesc &opDesc, const SVector<Tensor> &inTensorLists);
    Kernel *GetKernel(const std::string &kernelName);
    Status RunKernel(const std::string &kernelName);
    Status CopyDeviceTensorToHostTensor();
    Status RunGolden();
    void Init();
    void AddWorkspace(const KernelInfo &kernelInfo);
    void FreeWorkspace(const KernelInfo &kernelInfo);
    int64_t GetOutputNum(const UtOpDesc &opDesc);
    void Cleanup();
    void LogInferShapeResult();
    void MallocInTensor();
    void MallocInTensorKVPtr();
    void MallocInTensor(const SVector<Tensor> &inTensorLists);
    void MallocOutTensor();
    std::string TensorToString(const Tensor &tensor);
    void StoreKVPtr(std::vector<std::uint8_t*> &kvCachePtr);
    

private:
    bool launchWithTiling_ = true;
    int deviceId_ = 0;
    OpTestGolden golden_;
    GoldenContext goldenContext_;
    SVector<std::string> dataFiles_;
    std::vector<std::string> kvFile_;
    std::map<size_t, size_t> outUseInputdata_; // <0, 2> = outtensor(0).data = intensor(2).data
    LaunchParam launchParam_;
    RunInfo runInfo_;
    Operation *op_ = nullptr;
    int64_t outputNum_ = 0;
    float randFloatMin_ = FLT_MIN;
    float randFloatMax_ = FLT_MAX;
    int8_t randInt8Min_ = 3;
    int8_t randInt8Max_ = 3;
    int32_t randIntMin_ = 2;
    int32_t randIntMax_ = 2;
    int64_t randLongMin_ = 1;
    int64_t randLongMax_ = 1;
    OpTestStatistic statistic_;
};
} // namespace Test
} // namespace Mki

#endif