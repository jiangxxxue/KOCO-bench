/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <mki/base/operation_base.h>
#include <mki/utils/log/log.h>
#include <mki/utils/const/op_const.h>
#include <mki_loader/op_register.h>
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/params.h"

namespace {
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t CONST_20 = 20;
constexpr uint32_t CONST_24 = 24;
constexpr uint32_t CONST_32 = 32;
constexpr uint32_t CONST_48 = 48;
constexpr uint32_t CONST_64 = 64;
constexpr uint32_t CONST_96 = 96;
constexpr uint32_t CONST_128 = 128;
constexpr uint32_t CONST_144 = 144;
constexpr uint32_t CONST_192 = 192;
constexpr uint32_t CONST_256 = 256;
constexpr uint32_t CONST_7168 = 7168;
}

namespace AsdOps {

using namespace Mki;
class MoeGateCorrOperation : public OperationBase {
public:
    explicit MoeGateCorrOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        const int64_t inputNum = 2;
        MKI_CHECK(specificParam.Type() == typeid(OpParam::MoeGateCorr), "OpParam is invalid", return 0);
        return inputNum;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        const int64_t outputNum = 1;
        MKI_CHECK(specificParam.Type() == typeid(OpParam::MoeGateCorr), "OpParam is invalid", return 0);
        return outputNum;
    }

    bool CheckMoeGateCorr(const LaunchParam &launchParam) const
    {
        return true;
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MoeGateCorr),
            "OpParam is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        MKI_CHECK(CheckMoeGateCorr(launchParam), "Failed to check run info",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check run info"));

        const auto &inputADim = launchParam.GetInTensor(0).desc.dims;
        const auto &inputBDim = launchParam.GetInTensor(1).desc.dims;

        MKI_CHECK(inputADim.size() == 2, "inTensor0 dims invalid, should be 2",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "inTensor0 dims invalid, should be 2"));
        MKI_CHECK(inputBDim.size() == 2, "inTensor1 dims invalid, should be 2",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "inTensor1 dims invalid, should be 2"));

        // NT
        int64_t m = inputADim.at(0);
        int64_t n = inputBDim.at(0);

        auto &outDesc = outTensors[0].desc;
        outDesc.dtype = TensorDType::TENSOR_DTYPE_FLOAT;
        outDesc.format = TensorFormat::TENSOR_FORMAT_ND;
        outDesc.dims = {m, n};
        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MoeGateCorr),
            "OpParam is invalid", return nullptr);
        auto opParam = AnyCast<OpParam::MoeGateCorr>(launchParam.GetParam());
        MKI_CHECK(!opParam.transposeA && opParam.transposeB,
            "Only supported no transposed A_matrix and transposed B_matrix", return nullptr);

        uint32_t coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);

        const auto &inputADim = launchParam.GetInTensor(0).desc.dims;
        const auto &inputBDim = launchParam.GetInTensor(1).desc.dims;
        // NT
        int64_t m = inputADim.at(0);
        int64_t n = inputBDim.at(0);
        int64_t k = inputADim.at(1);
        if (n == CONST_256 && k == CONST_7168) {
            if (m <= CONST_16) {
                MKI_LOG(INFO) << "use MoeGateCorrM16N16K2048Kernel, m:" << m << ", n:" << n << ", k:" << k;
                return GetKernelByName("MoeGateCorrM16N16K2048Kernel");
            } else if ((coreNum== CONST_20 && m <= CONST_32) || (coreNum== CONST_24 && m <= CONST_48)) {
                MKI_LOG(INFO) << "use MoeGateCorrM16N32K1024Kernel, m:" << m << ", n:" << n << ", k:" << k;
                return GetKernelByName("MoeGateCorrM16N32K1024Kernel");
            } else if ((coreNum== CONST_20 && m <= CONST_64) || (coreNum== CONST_24 && m <= CONST_96)) {
                MKI_LOG(INFO) << "use MoeGateCorrM32N32K1024Kernel, m:" << m << ", n:" << n << ", k:" << k;
                return GetKernelByName("MoeGateCorrM32N32K1024Kernel");
            } else if ((coreNum== CONST_20 && m <= CONST_96) || (coreNum== CONST_24 && m <= CONST_128)) {
                MKI_LOG(INFO) << "use MoeGateCorrM32N48K640Kernel, m:" << m << ", n:" << n << ", k:" << k;
                return GetKernelByName("MoeGateCorrM32N48K640Kernel");
            } else if ((coreNum== CONST_20 && m <= CONST_144) || (coreNum== CONST_24 && m <= CONST_192)) {
                MKI_LOG(INFO) << "use MoeGateCorrM48N48K640Kernel, m:" << m << ", n:" << n << ", k:" << k;
                return GetKernelByName("MoeGateCorrM48N48K640Kernel");
            } else if ((coreNum== CONST_20 && m <= CONST_192) || (coreNum== CONST_24 && m <= CONST_256)) {
                MKI_LOG(INFO) << "use MoeGateCorrM48N64K512Kernel, m:" << m << ", n:" << n << ", k:" << k;
                return GetKernelByName("MoeGateCorrM48N64K512Kernel");
            } else if (coreNum== CONST_20 && m <= CONST_256) {
                MKI_LOG(INFO) << "use MoeGateCorrM64N64K512Kernel, m:" << m << ", n:" << n << ", k:" << k;
                return GetKernelByName("MoeGateCorrM64N64K512Kernel");
            } else {
                MKI_LOG(INFO) << "use MoeGateCorrM64N64K512Kernel, m:" << m << ", n:" << n << ", k:" << k;
                return GetKernelByName("MoeGateCorrM64N64K512Kernel");
            }
        } else {
            MKI_LOG(INFO) << "use MoeGateCorrM64N64K512Kernel, m:" << m << ", n:" << n << ", k:" << k;
            return GetKernelByName("MoeGateCorrM64N64K512Kernel");
        }
    }
};

REG_OPERATION(MoeGateCorrOperation);
} // namespace AsdOps