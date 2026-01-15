/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ATB_OPTEST_H
#define ATB_OPTEST_H
#include <functional>
#include <string>
#include <mki/tensor.h>
#include <mki/utils/status/status.h>
#include <mki/utils/SVector/SVector.h>
#include "atb/operation.h"
#include "atb/types.h"
#include "atb/svector.h"
#include "atb/context.h"

namespace atb {
struct GoldenContext {
    SVector<Tensor> hostInTensors;
    SVector<Tensor> deviceInTensors;
    SVector<Tensor> hostOutTensors;
    SVector<Tensor> deviceOutTensors;
};

struct OpsGoldenContext {
    SVector<Mki::Tensor> hostInTensors;
    SVector<Mki::Tensor> deviceInTensors;
    SVector<Mki::Tensor> hostOutTensors;
    SVector<Mki::Tensor> deviceOutTensors;
};

using OpTestGolden = std::function<Mki::Status(const OpsGoldenContext &context)>;

class OperationTest {
public:
    explicit OperationTest();
    ~OperationTest();
    void Golden(OpTestGolden golden);
    void FloatRand(float min, float max);
    void Int8Rand(int8_t min, int8_t max);
    void IntRand(int32_t min, int32_t max);
    void LongRand(int64_t min, int64_t max);
    Status Run(atb::Operation *operation, const SVector<TensorDesc> &inTensorDescs);
    Status Run(atb::Operation *operation, const SVector<Tensor> &inTensorLists);
    Status RunPerf(atb::Operation *operation,  const SVector<Tensor> &inTensorLists);
    Status RunPerf(atb::Operation *operation, const SVector<TensorDesc> &inTensorDescs);
    Mki::Status Run(atb::Operation *operation, const Mki::SVector<Mki::TensorDesc> &inTensorDescs);
    Mki::Status Run(atb::Operation *operation, const Mki::SVector<Mki::Tensor> &inTensorLists);
    Mki::Status Setup(atb::Operation *operation, const Mki::SVector<Mki::TensorDesc> &inTensorDescs);
    void SetMockFlag(bool flag);
    bool GetMockFlag();
    void SetDeviceId(int deviceId);

private:
    Status GenerateRandomTensors(const SVector<TensorDesc> &inTensorDescs, SVector<Tensor> &inTensors);
    Status Init();
    void Cleanup();
    Status Prepare(atb::Operation *operation, const SVector<Tensor> &inTensorLists);
    Tensor CreateHostTensor(const Tensor &tensorIn);
    Status CopyDeviceTensorToHostTensor();
    Tensor HostTensor2DeviceTensor(const Tensor &hostTensor);
    std::string TensorToString(const Tensor &tensor);
    Tensor CreateHostZeroTensor(const TensorDesc &tensorDesc);
    void BuildVariantPack(const SVector<Tensor> &inTensorLists);
    Status RunOperation();
    Mki::Status RunGolden();
    Status CreateHostRandTensor(const TensorDesc &tensorDesc, Tensor &tensor);
    Status RunImpl(atb::Operation *operation, const SVector<Tensor> &inTensorLists);
    Status RunPrefImpl(atb::Operation *operation, const SVector<Tensor> &inTensorLists);
    void FreeInTensorList(SVector<Tensor> &hostInTensors);

private:
    int deviceId_ = 0;
    OpTestGolden golden_;
    GoldenContext goldenContext_;
    atb::Operation *operation_;
    atb::Context *context_ = nullptr;
    atb::VariantPack variantPack_;
    uint64_t workSpaceSize_ = 0;
    void *workSpace_ = nullptr;
    bool mockFlag_ = false;
    float randFloatMin_ = 1;
    float randFloatMax_ = 1;
    int8_t randInt8Min_ = 1;
    int8_t randInt8Max_ = 1;
    int32_t randIntMin_ = 1;
    int32_t randIntMax_ = 1;
    int64_t randLongMin_ = 1;
    int64_t randLongMax_ = 1;
};
} // namespace atb
#endif