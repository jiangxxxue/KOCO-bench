/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <numeric>
#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include "mki_loader/op_register.h"
#include "mki/utils/fp16/fp16_t.h"
#include "mki/utils/log/log.h"
#include "atbops/params/params.h"
#include "mixkernels/rope/tiling/rope_tiling.h"
#include "test_common.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"

using namespace AtbOps;
using namespace Mki;
using namespace Mki;

static constexpr float ATOL = 0.001;
static constexpr float RTOL = 0.001;
static constexpr float HALF_FLOAT_MIN = -1.0;
static constexpr float HALF_FLOAT_MAX = 1.0;

TEST(ropeNd, RopeKernelCanSupport)
{
    int64_t ntokens = 32;
    int64_t batch = 32;
    int64_t hiddenSize = 4096;
    int64_t headDim = 128;
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, headDim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, headDim}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {batch}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    OpParam::Rope opParam;
    Mki::Test::UtOpDesc opDesc = {"RopeOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    std::unique_ptr<Kernel> kernel{nullptr};
    auto &kernelCreators = AtbOps::KernelRegister::GetKernelCreators();
    for (const auto &iter : kernelCreators) {
        if (iter.kernelName == "AtbRopeKernel") {
            const std::string &opName = iter.opName;
            Operation *op = AutoGen::GetOpByName(opName);
            kernel.reset(op->GetKernelByName("AtbRopeKernel"));
            break;
        }
    }
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

TEST(ropeNd, ropeNdTiling)
{
    int64_t ntokens = 32;
    int64_t batch = 32;
    int64_t hiddenSize = 4096;
    int64_t headDim = 128;
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, headDim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, headDim}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {batch}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    OpParam::Rope opParam;
    Mki::Test::UtOpDesc opDesc = {"RopeOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
 
    Operation *operation = AutoGen::GetOpByName(opDesc.opName);
    std::unique_ptr<Kernel> kernel(operation->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
    uint32_t launchBufferSize = kernel->GetTilingSize(launchParam);
    if (launchBufferSize == 0) {
        MKI_LOG(ERROR) << "empty tiling size";
    }
    KernelInfo &kernelInfo = const_cast<KernelInfo &>(kernel->GetKernelInfo());
    kernelInfo.AllocTilingHost(launchBufferSize);
    Status status = RopeTiling(launchParam, kernelInfo);
    ASSERT_EQ(status.Ok(), true);
}

TEST(ropeNd, ropeNdGetBestKernel)
{
    int64_t ntokens = 32;
    int64_t batch = 32;
    int64_t hiddenSize = 4096;
    int64_t headDim = 128;
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, headDim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, headDim}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {batch}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    OpParam::Rope opParam;
    Mki::Test::UtOpDesc opDesc = {"RopeOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    Operation *operation = AutoGen::GetOpByName(opDesc.opName);
    ASSERT_NE(operation, nullptr);

    std::unique_ptr<Kernel> kernel(operation->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

TEST(ropeNd, ropeNdInferShape)
{
    int64_t ntokens = 32;
    int64_t batch = 32;
    int64_t hiddenSize = 4096;
    int64_t headDim = 128;
    AtbOps::SVector<AtbOps::Tensor> outTensor;
    outTensor.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    outTensor.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, headDim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, headDim}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {batch}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    OpParam::Rope opParam;
    Mki::Test::UtOpDesc opDesc = {"RopeOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    Operation *operation = AutoGen::GetOpByName(opDesc.opName);
    ASSERT_EQ(operation != nullptr, true);
    Status status = operation->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);

    bool flag = true;
    for (size_t i = 0; i < launchParam.GetOutTensorCount(); ++i) {
        flag &= outTensor[i].desc.dtype == launchParam.GetOutTensor(i).desc.dtype;
        flag &= outTensor[i].desc.format == launchParam.GetOutTensor(i).desc.format;
        flag &= outTensor[i].desc.dims.size() == launchParam.GetOutTensor(i).desc.dims.size();
        for (size_t j = 0; j < outTensor[i].desc.dims.size(); ++j) {
            flag &= outTensor[i].desc.dims[j] == launchParam.GetOutTensor(i).desc.dims[j];
        }
    }
    ASSERT_EQ(flag, true);
}

TEST(ropeNd, ropeNdInferShapeFailed)
{
    int64_t ntokens = 32;
    int64_t batch = 32;
    int64_t hiddenSize = 4096;
    int64_t headDim = 128;
    AtbOps::SVector<AtbOps::Tensor> outTensor;
    outTensor.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    outTensor.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, headDim}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, headDim}});
    launchParam.AddInTensor({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {batch}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {ntokens, hiddenSize}});
    OpParam::Rope opParam;
    Mki::Test::UtOpDesc opDesc = {"RopeOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    Operation *operation = AutoGen::GetOpByName(opDesc.opName);
    ASSERT_EQ(operation != nullptr, true);
    Status status = operation->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);

    bool flag = true;
    for (size_t i = 0; i < launchParam.GetOutTensorCount(); ++i) {
        flag &= outTensor[i].desc.dtype == launchParam.GetOutTensor(i).desc.dtype;
        flag &= outTensor[i].desc.format == launchParam.GetOutTensor(i).desc.format;
        flag &= outTensor[i].desc.dims.size() == launchParam.GetOutTensor(i).desc.dims.size();
        for (size_t j = 0; j < outTensor[i].desc.dims.size(); ++j) {
            flag &= outTensor[i].desc.dims[j] == launchParam.GetOutTensor(i).desc.dims[j];
        }
    }
    ASSERT_EQ(flag, false);
}
