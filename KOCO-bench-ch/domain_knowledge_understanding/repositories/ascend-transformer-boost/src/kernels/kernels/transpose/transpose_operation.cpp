/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/base/operation_base.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki_loader/op_register.h>

#include "asdops/params/params.h"

namespace {
void Heapify(Mki::SVector<int32_t> &inputDims, int n, int i)
{
    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;

    if (l < n && inputDims[l] > inputDims[largest]) {
        largest = l;
    }

    if (r < n && inputDims[r] > inputDims[largest]) {
        largest = r;
    }

    if (largest != i) {
        std::swap(inputDims[i], inputDims[largest]);
        Heapify(inputDims, n, largest);
    }
}

Mki::SVector<int32_t> HeapSort(Mki::SVector<int32_t> inputDims)
{
    int n = static_cast<int>(inputDims.size());
    for (int i = n / 2 - 1; i >= 0; --i) {
        Heapify(inputDims, n, i);
    }

    for (int i = n - 1; i >= 0; --i) {
        std::swap(inputDims[0], inputDims[i]);
        Heapify(inputDims, i, 0);
    }
    return inputDims;
}

int32_t CheckSVector(Mki::SVector<int32_t> sortDims)
{
    for (size_t i = 0; i < sortDims.size() - 1; ++i) {
        if (sortDims[i + 1] == sortDims[i]) {
            MKI_LOG(INFO) << i << sortDims[i + 1] << sortDims[i];
            return -1;
        }
    }
    return 0;
}
} // namespace

namespace AsdOps {
using namespace Mki;
class TransposeOperation : public OperationBase {
public:
    explicit TransposeOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Fail to check consistent", return nullptr);
        auto inDtype = launchParam.GetInTensor(0).desc.dtype;
        auto platformType = PlatformInfo::Instance().GetPlatformType();
        if (inDtype == TENSOR_DTYPE_FLOAT16 || inDtype == TENSOR_DTYPE_BF16) {
            return GetKernelByName("Transpose16Kernel");
        } else if (inDtype == TENSOR_DTYPE_FLOAT || inDtype == TENSOR_DTYPE_INT32) {
            MKI_CHECK(platformType == PlatformType::ASCEND_910B || platformType == PlatformType::ASCEND_310P,
                      "INT32/FP32 is not supported by current Soc", return nullptr);
            return GetKernelByName("Transpose32Kernel");
        } else if (inDtype == TENSOR_DTYPE_INT64) {
            return GetKernelByName("Transpose64Kernel");
        } else if (inDtype == TENSOR_DTYPE_INT8) {
            MKI_CHECK(platformType == PlatformType::ASCEND_910B || platformType == PlatformType::ASCEND_310P,
                      "INT8 is not supported by current Soc", return nullptr);
            return GetKernelByName("Transpose8Kernel");
        }
        return nullptr;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Transpose), "transpose: no match param type",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "no match specificParam type."));
        OpParam::Transpose param = AnyCast<OpParam::Transpose>(launchParam.GetParam());
        SVector<int32_t> perm = param.perm;

        outTensors[0].desc.dtype = launchParam.GetInTensor(0).desc.dtype;
        outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;

        Mki::SVector<int64_t> inputDims = launchParam.GetInTensor(0).desc.dims;

        if (perm.empty() || inputDims.empty() || perm.size() != inputDims.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "Wrong input perm");
        }
        Mki::SVector<int32_t> sortDims = HeapSort(perm);

        if (sortDims[0] != 0 || sortDims[sortDims.size() - 1] != static_cast<int64_t>(inputDims.size() - 1) ||
            CheckSVector(sortDims) != 0) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "Wrong idx of perm");
        }

        Mki::SVector<int64_t> outputDims;
        for (size_t i = 0; i < perm.size(); ++i) {
            size_t j = static_cast<size_t>(perm[i]);
            if (j >= inputDims.size()) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "Wrong perm or input");
            }
            outputDims.push_back(inputDims[j]);
        }
        outTensors[0].desc.dims = outputDims;

        return Status::OkStatus();
    }
};
REG_OPERATION(TransposeOperation);
} //    namespace AsdOps