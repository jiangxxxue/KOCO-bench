/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include <climits>
#include <fstream>
#include <sys/stat.h>
#include <mki/base/operation_base.h>
#include <mki/types.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/env/env.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/strings/str_replace.h>
#include <mki_loader/op_register.h>
#include "atbops/params/params.h"

namespace AtbOps {
using namespace Mki;
const uint64_t INPUT_NUM_TWO = 2;
const uint64_t INPUT_NUM_THREE = 3;
const size_t DYNAMICSIZE = 126976;
const size_t DYNAMICNULLSIZE = 0;
class FusionOperation : public OperationBase {
    using MmType = OpParam::Fusion;

public:
    explicit FusionOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        std::string kernelName = "FusionMatmulAddKernel";
        OpParam::Fusion fusionType = launchParam.GetParam<OpParam::Fusion>();
        if (fusionType.fusionType == OpParam::Fusion::MATMUL_GELU) {
            kernelName = "FusionMatmulGeluKernel";
        } else if (fusionType.fusionType == OpParam::Fusion::MATMUL_SIGMOID) {
            kernelName = "FusionMatmulSigmoidKernel";
        } else if (fusionType.fusionType == OpParam::Fusion::MATMUL_SWIGLU) {
            kernelName = "FusionMatmulSwiGluKernel";
        } else if (fusionType.fusionType == OpParam::Fusion::NON_FUSION) {
            kernelName = "FusionErasedKernel";
        }
        MKI_LOG(INFO) << "getBestKernel " << kernelName;
        return GetKernelByName(kernelName);
    }

    void MatMulAddFusion()
    {
        static uint8_t matMulAddFusionKernelBinData[DYNAMICSIZE];
        std::string path = std::string(std::getenv("HOME")) + "/.atb_auto_fusion/bishengir_bin/matmul_add.cpp";
        if (IsSoftLink(path.c_str())) {
            MKI_LOG(ERROR) << "MatMulAddFusion CPP SHOULD NOT be a symbolic link ";
            return;
        }
        std::ifstream cpp(path.c_str());
        std::string line;
        uint32_t counter = 0;
        while (std::getline(cpp, line)) {
            char c = static_cast<char>(std::stoi(line, nullptr, 16));
            matMulAddFusionKernelBinData[counter++] = c;
        }
        cpp.close();
        std::string deviceVersion = PlatformInfo::Instance().GetPlatformName();
        std::string kernelName = "FusionMatmulAddKernel";
        if (deviceVersion == "ascend910b") {
            static KernelBinaryRegister binFusionMatmulAddKernelascend910bregister =
                KernelBinaryRegister(deviceVersion.c_str(), kernelName.c_str(), matMulAddFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionMatmulAddKernelascend910bregister);
        } else if (deviceVersion == "ascend910") {
            static KernelBinaryRegister binFusionMatmulAddKernelascend910register =
                KernelBinaryRegister(deviceVersion.c_str(), kernelName.c_str(), matMulAddFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionMatmulAddKernelascend910register);
        } else if (deviceVersion == "ascend310b") {
            static KernelBinaryRegister binFusionMatmulAddKernelascend310bregister =
                KernelBinaryRegister(deviceVersion.c_str(), kernelName.c_str(), matMulAddFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionMatmulAddKernelascend310bregister);
        } else if (deviceVersion == "ascend310p") {
            static KernelBinaryRegister binFusionMatmulAddKernelascend310pregister =
                KernelBinaryRegister(deviceVersion.c_str(), kernelName.c_str(), matMulAddFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionMatmulAddKernelascend310pregister);
        } else {
            MKI_LOG(ERROR) << "MatMulAddFusion operation add failed ";
        }
        return;
    }

    void MatMulGeluFusion()
    {
        static uint8_t matMulGeluFusionKernelBinData[DYNAMICSIZE];
        std::string path = std::string(std::getenv("HOME")) + "/.atb_auto_fusion/bishengir_bin/matmul_gelu.cpp";
        if (IsSoftLink(path.c_str())) {
            MKI_LOG(ERROR) << "MatMulGeluFusion CPP SHOULD NOT be a symbolic link ";
            return;
        }
        std::ifstream cpp(path.c_str());
        std::string line;
        uint32_t counter = 0;
        while (std::getline(cpp, line)) {
            char c = static_cast<char>(std::stoi(line, nullptr, 16));
            matMulGeluFusionKernelBinData[counter++] = c;
        }
        cpp.close();
        std::string deviceVersion = PlatformInfo::Instance().GetPlatformName();
        std::string kernelName = "FusionMatmulGeluKernel";
        if (deviceVersion == "ascend910b") {
            static KernelBinaryRegister binFusionMatmulGeluKernelascend910bregister =
                KernelBinaryRegister(deviceVersion.c_str(), kernelName.c_str(), matMulGeluFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionMatmulGeluKernelascend910bregister);
        } else if (deviceVersion == "ascend910") {
            static KernelBinaryRegister binFusionMatmulGeluKernelascend910register =
                KernelBinaryRegister(deviceVersion.c_str(), kernelName.c_str(), matMulGeluFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionMatmulGeluKernelascend910register);
        } else if (deviceVersion == "ascend310b") {
            static KernelBinaryRegister binFusionMatmulGeluKernelascend310bregister =
                KernelBinaryRegister(deviceVersion.c_str(), kernelName.c_str(), matMulGeluFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionMatmulGeluKernelascend310bregister);
        } else if (deviceVersion == "ascend310p") {
            static KernelBinaryRegister binFusionMatmulGeluKernelascend310pregister =
                KernelBinaryRegister(deviceVersion.c_str(), kernelName.c_str(), matMulGeluFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionMatmulGeluKernelascend310pregister);
        } else {
            MKI_LOG(ERROR) << "MatMulGeluFusion operation add failed ";
        }
        return;
    }

    void MatMulSigmoidFusion()
    {
        static uint8_t matMulSigmoidFusionKernelBinData[DYNAMICSIZE];
        std::string path = std::string(std::getenv("HOME")) + "/.atb_auto_fusion/bishengir_bin/matmul_sigmoid.cpp";
        if (IsSoftLink(path.c_str())) {
            MKI_LOG(ERROR) << "MatMulSigmoidFusion CPP SHOULD NOT be a symbolic link ";
            return;
        }
        std::ifstream cpp(path.c_str());
        std::string line;
        uint32_t counter = 0;
        while (std::getline(cpp, line)) {
            char c = static_cast<char>(std::stoi(line, nullptr, 16));
            matMulSigmoidFusionKernelBinData[counter++] = c;
        }
        cpp.close();
        std::string deviceVersion = PlatformInfo::Instance().GetPlatformName();
        std::string kernelName = "FusionMatmulSigmoidKernel";
        if (deviceVersion == "ascend910b") {
            static KernelBinaryRegister binFusionMatmulSigmoidKernelascend910bregister = KernelBinaryRegister(
                deviceVersion.c_str(), kernelName.c_str(), matMulSigmoidFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionMatmulSigmoidKernelascend910bregister);
        } else if (deviceVersion == "ascend910") {
            static KernelBinaryRegister binFusionMatmulSigmoidKernelascend910register = KernelBinaryRegister(
                deviceVersion.c_str(), kernelName.c_str(), matMulSigmoidFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionMatmulSigmoidKernelascend910register);
        } else if (deviceVersion == "ascend310b") {
            static KernelBinaryRegister binFusionMatmulSigmoidKernelascend310bregister = KernelBinaryRegister(
                deviceVersion.c_str(), kernelName.c_str(), matMulSigmoidFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionMatmulSigmoidKernelascend310bregister);
        } else if (deviceVersion == "ascend310p") {
            static KernelBinaryRegister binFusionMatmulSigmoidKernelascend310pregister = KernelBinaryRegister(
                deviceVersion.c_str(), kernelName.c_str(), matMulSigmoidFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionMatmulSigmoidKernelascend310pregister);
        } else {
            MKI_LOG(ERROR) << "MatMulSigmoidFusion operation add failed ";
        }
        return;
    }

    void MatMulSwigluFusion()
    {
        static uint8_t matMulSwigluFusionKernelBinData[DYNAMICSIZE];
        std::string path = std::string(std::getenv("HOME")) + "/.atb_auto_fusion/bishengir_bin/matmul_swiglu.cpp";
        if (IsSoftLink(path.c_str())) {
            MKI_LOG(ERROR) << "MatMulSwigluFusion CPP SHOULD NOT be a symbolic link ";
            return;
        }
        std::ifstream cpp(path.c_str());
        std::string line;
        uint32_t counter = 0;
        while (std::getline(cpp, line)) {
            char c = static_cast<char>(std::stoi(line, nullptr, 16));
            matMulSwigluFusionKernelBinData[counter++] = c;
        }
        cpp.close();
        std::string deviceVersion = PlatformInfo::Instance().GetPlatformName();
        std::string kernelName = "FusionMatmulSwiGluKernel";
        if (deviceVersion == "ascend910b") {
            static KernelBinaryRegister binFusionMatmulSwiGluKernelascend910bregister = KernelBinaryRegister(
                deviceVersion.c_str(), kernelName.c_str(), matMulSwigluFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionMatmulSwiGluKernelascend910bregister);
        } else if (deviceVersion == "ascend910") {
            static KernelBinaryRegister binFusionMatmulSwiGluKernelascend910register = KernelBinaryRegister(
                deviceVersion.c_str(), kernelName.c_str(), matMulSwigluFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionMatmulSwiGluKernelascend910register);
        } else if (deviceVersion == "ascend310b") {
            static KernelBinaryRegister binFusionMatmulSwiGluKernelascend310bregister = KernelBinaryRegister(
                deviceVersion.c_str(), kernelName.c_str(), matMulSwigluFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionMatmulSwiGluKernelascend310bregister);
        } else if (deviceVersion == "ascend310p") {
            static KernelBinaryRegister binFusionMatmulSwiGluKernelascend310pregister = KernelBinaryRegister(
                deviceVersion.c_str(), kernelName.c_str(), matMulSwigluFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionMatmulSwiGluKernelascend310pregister);
        } else {
            MKI_LOG(ERROR) << "MatMulSwigluFusion operation add failed ";
        }
        return;
    }

    void ErasedFusion() const
    {
        static uint8_t erasedFusionKernelBinData[DYNAMICNULLSIZE];
        uint32_t counter = 0;
        std::string deviceVersion = PlatformInfo::Instance().GetPlatformName();
        std::string kernelName = "FusionErasedKernel";
        if (deviceVersion == "ascend910b") {
            static KernelBinaryRegister binFusionErasedKernelascend910bregister =
                KernelBinaryRegister(deviceVersion.c_str(), kernelName.c_str(), erasedFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionErasedKernelascend910bregister);
        } else if (deviceVersion == "ascend910") {
            static KernelBinaryRegister binFusionErasedKernelascend910register =
                KernelBinaryRegister(deviceVersion.c_str(), kernelName.c_str(), erasedFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionErasedKernelascend910register);
        } else if (deviceVersion == "ascend310b") {
            static KernelBinaryRegister binFusionErasedKernelascend310bregister =
                KernelBinaryRegister(deviceVersion.c_str(), kernelName.c_str(), erasedFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionErasedKernelascend310bregister);
        } else if (deviceVersion == "ascend310p") {
            static KernelBinaryRegister binFusionErasedKernelascend310pregister =
                KernelBinaryRegister(deviceVersion.c_str(), kernelName.c_str(), erasedFusionKernelBinData, counter);
            UNUSED_VALUE(binFusionErasedKernelascend310pregister);
        } else {
            MKI_LOG(ERROR) << "ErasedFusion operation add failed ";
        }
        return;
    }

    bool DynamicRegisterKernelByName(const LaunchParam &launchParam, const std::string &opName) override
    {
        OpParam::Fusion fusionType = launchParam.GetParam<OpParam::Fusion>();
        std::string deviceVersion = PlatformInfo::Instance().GetPlatformName();
        if (fusionType.fusionType == OpParam::Fusion::MATMUL_ADD) {
            MatMulAddFusion();
        } else if (fusionType.fusionType == OpParam::Fusion::MATMUL_GELU) {
            MatMulGeluFusion();
        } else if (fusionType.fusionType == OpParam::Fusion::MATMUL_SIGMOID) {
            MatMulSigmoidFusion();
        } else if (fusionType.fusionType == OpParam::Fusion::MATMUL_SWIGLU) {
            MatMulSwigluFusion();
        } else {
            ErasedFusion();
            MKI_LOG(DEBUG) << "DynamicRegisterKernelByName null kernel ";
        }
        return true;
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        OpParam::Fusion fusionParam = AnyCast<OpParam::Fusion>(specificParam);
        if (fusionParam.fusionType == OpParam::Fusion::MATMUL_ADD) {
            return INPUT_NUM_THREE;
        }
        return INPUT_NUM_TWO;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        OpParam::Fusion fusionType = launchParam.GetParam<OpParam::Fusion>();
        if (fusionType.fusionType == OpParam::Fusion::MATMUL_ADD) {
            auto inTensorDescA = launchParam.GetInTensor(2).desc;
            TensorDesc &tensorDescOut = outTensors[0].desc;
            tensorDescOut.dtype = TENSOR_DTYPE_FLOAT16;
            tensorDescOut.format = inTensorDescA.format;
            auto &outDims = tensorDescOut.dims;
            outDims.emplace_back(inTensorDescA.dims[DIM_0]);
            outDims.emplace_back(inTensorDescA.dims[DIM_1]);
        } else {
            auto inTensorDescA = launchParam.GetInTensor(0).desc;
            auto inTensorDescB = launchParam.GetInTensor(1).desc;

            TensorDesc &tensorDescOut = outTensors[0].desc;
            tensorDescOut.dtype = TENSOR_DTYPE_FLOAT16;
            tensorDescOut.format = inTensorDescA.format;
            auto &outDims = tensorDescOut.dims;
            outDims.emplace_back(inTensorDescA.dims[DIM_0]);
            outDims.emplace_back(inTensorDescB.dims[DIM_0]);
        }
        return Status::OkStatus();
    }
    
    bool IsSoftLink(const char *path) const
    {
        struct stat fileStat;
        if (lstat(path, &fileStat) != 0) {
            return false;
        }
        return S_ISLNK(fileStat.st_mode);
    }
};
REG_OPERATION(FusionOperation);
} // namespace AtbOps
