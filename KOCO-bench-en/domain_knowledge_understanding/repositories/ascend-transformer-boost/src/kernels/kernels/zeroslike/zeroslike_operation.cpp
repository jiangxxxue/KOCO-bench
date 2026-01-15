/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendOpCommonLib is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "asdops/params/zeroslike.h"
#include "mki/base/operation_base.h"
#include "mki/utils/log/log.h"
#include "mki_loader/op_register.h"

namespace AsdOps {
using namespace Mki;

class ZerosLikeOperation : public OperationBase {
public:
    explicit ZerosLikeOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        auto dtype = launchParam.GetInTensor(0).desc.dtype;
        if (dtype == TENSOR_DTYPE_FLOAT) {
            return GetKernelByName("ZerosLikeF32Kernel");
        } else {
            return nullptr;
        }
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        TensorDType dtype = launchParam.GetInTensor(0).desc.dtype;
        TensorFormat format = launchParam.GetInTensor(0).desc.format;
        SVector<int64_t> dims = launchParam.GetInTensor(0).desc.dims;
        MKI_CHECK(dtype == TENSOR_DTYPE_FLOAT, "input tensor dtype is invalid: " << dtype,
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR,
                                            "input tensor dtype should be float32"));
        MKI_CHECK(format == TENSOR_FORMAT_ND, "input tensor format is invalid: " << format,
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "input tensor format should be ND"));
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::ZerosLike), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));

        outTensors[0].desc = {dtype, format, dims, {}, 0};
        // check tensor num
        return Status::OkStatus();
    }
};
REG_OPERATION(ZerosLikeOperation);
} //    namespace AsdOps