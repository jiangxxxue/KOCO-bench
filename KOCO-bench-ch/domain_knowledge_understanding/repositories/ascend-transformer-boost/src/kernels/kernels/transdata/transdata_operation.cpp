/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include <mki/base/operation_base.h>
#include "asdops/params/transdata.h"
#include "kernels/matmul/common/common_tiling.h"

static const size_t IDX_0 = 0;
static const size_t IDX_1 = 1;
static const size_t IDX_2 = 2;
static const size_t IDX_3 = 3;
static const size_t BUFFER_SIZE_4 = 4;
static const size_t BUFFER_SIZE_3 = 3;
static const int64_t DEFAULT_ALIGN = 16;
static const int64_t DEFAULT_INT8_ALIGN = 32;

namespace AsdOps {
using namespace Mki;
class TransdataOperation : public OperationBase {
public:
    explicit TransdataOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Fail to check consistent", return nullptr);
        OpParam::Transdata param = AnyCast<OpParam::Transdata>(launchParam.GetParam());
        TensorDType dtype = launchParam.GetInTensor(0).desc.dtype;
        if (param.transdataType == OpParam::Transdata::FRACTAL_NZ_TO_ND) {
            return GetKernelByName("TransdataNzToNdKernel");
        } else if (param.transdataType == OpParam::Transdata::ND_TO_FRACTAL_NZ) {
            if (dtype == TENSOR_DTYPE_INT8) {
                return GetKernelByName("TransdataNdToNzInt8Kernel");
            }
            return GetKernelByName("TransdataNdToNzKernel");
        } else {
            return nullptr;
        }
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Transdata), "transdata: no match param type",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "no match specificParam type."));
            
        auto param = AnyCast<OpParam::Transdata>(launchParam.GetParam());
        const SVector<int64_t> &outCrops = param.outCrops;
        TensorDType dtype = launchParam.GetInTensor(0).desc.dtype;
        TensorFormat format = launchParam.GetInTensor(0).desc.format;
        TensorDesc &outDesc = outTensors[0].desc;
        SVector<int64_t> inDims = launchParam.GetInTensor(0).desc.dims;
        int64_t nzAlign = DEFAULT_ALIGN;
        if (dtype == TENSOR_DTYPE_INT8) {
            nzAlign = DEFAULT_INT8_ALIGN;
        }

        if (param.transdataType == OpParam::Transdata::FRACTAL_NZ_TO_ND && format == TENSOR_FORMAT_FRACTAL_NZ) {
            if (inDims.size() < BUFFER_SIZE_4) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "inDims is not support");
            }
            MKI_CHECK(outCrops.size() >= 2, "transdata: size of outCrops is invalid",
                return Status::FailStatus(ERROR_INVALID_VALUE, "size of outCrops is invalid"));

            if (outCrops[0] > 0 && outCrops[1] > 0) {
                // the out shape is specified by param.outCrops
                outDesc.dims = SVector<int64_t>({inDims[0], outCrops[0], outCrops[1]});
            } else {
                // the out shape is given by inference.
                // [N, W1, H, W0] -> [N, H, W]
                outDesc.dims =
                    SVector<int64_t>({inDims[0], inDims[2], inDims[1] * inDims[3]}); // inDims[0, 1, 2, 3]:[N,W1,H,W0]
            }
            outDesc.dtype = dtype;
            outDesc.format = TENSOR_FORMAT_ND;
            return AsdOps::Status::OkStatus();
        } else if (param.transdataType == OpParam::Transdata::ND_TO_FRACTAL_NZ && format == TENSOR_FORMAT_ND) {
            if (inDims.size() < BUFFER_SIZE_3) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "inDims is not support");
            }
            // inference aux dims: [N, H, W] -> [N, H', W'/16, 16]
            SVector<int64_t> auxDims{ 0, 0, 0, 0 };
            auxDims[IDX_0] = inDims[IDX_0];
            auxDims[IDX_1] = AsdOps::RoundUp(inDims[IDX_1], DEFAULT_ALIGN);
            auxDims[IDX_2] = AsdOps::RoundUp(inDims[IDX_2], nzAlign) / nzAlign;
            auxDims[IDX_3] = nzAlign;

            // inference output dims: [N, H', W'/16, 16] -> [N, W'/16, H', 16]
            outDesc.dims = SVector<int64_t>({ 0, 0, 0, 0 });
            outDesc.dims[IDX_0] = auxDims[IDX_0];
            outDesc.dims[IDX_1] = auxDims[IDX_2];
            outDesc.dims[IDX_2] = auxDims[IDX_1];
            outDesc.dims[IDX_3] = auxDims[IDX_3];
            outDesc.dtype = dtype;
            outDesc.format = TENSOR_FORMAT_FRACTAL_NZ;
            return AsdOps::Status::OkStatus();
        } else {
            MKI_LOG(ERROR) << "Failed to check consistent";
        }

        return AsdOps::Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to infershape.");
    }
};
REG_OPERATION(TransdataOperation);
} // namespace AsdOps