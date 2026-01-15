/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

#include <mki/base/operation_base.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki_loader/op_register.h>

#include "asdops/params/elewise.h"

namespace AsdOps {
using namespace Mki;
static const std::unordered_map<TensorDType, std::string> DTYPE_SHORT_NAME = {
    {TENSOR_DTYPE_FLOAT, "F32"}, {TENSOR_DTYPE_FLOAT16, "F16"}, {TENSOR_DTYPE_BF16, "BF16"},
    {TENSOR_DTYPE_INT32, "I32"}, {TENSOR_DTYPE_INT64, "I64"},   {TENSOR_DTYPE_INT8, "I8"}};
std::optional<std::string> GetDTypeShortName(TensorDType dtype)
{
    auto iter = DTYPE_SHORT_NAME.find(dtype);
    if (iter == DTYPE_SHORT_NAME.end()) {
        MKI_LOG(ERROR) << "No kernel for ELEWISE_CAST dtype: " << GetStrWithDType(dtype);
        return std::nullopt;
    }
    return iter->second;
}

class ElewiseOperation : public OperationBase {
public:
    explicit ElewiseOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetCastKernel(const LaunchParam &launchParam) const
    {
        TensorDType inDType = launchParam.GetInTensor(0).desc.dtype;
        OpParam::Elewise param = AnyCast<OpParam::Elewise>(launchParam.GetParam());
        TensorDType outDType = param.outTensorType;
        if (outDType == TENSOR_DTYPE_UNDEFINED) {
            outDType = launchParam.GetOutTensor(0).desc.dtype;
        }

        std::stringstream ss;
        ss << "Cast";
        auto inDTypeShortName = GetDTypeShortName(inDType);
        if (!inDTypeShortName) {
            MKI_LOG(ERROR) << "Cast does not support input type: " << GetStrWithDType(inDType);
            return nullptr;
        }
        ss << *inDTypeShortName;
        auto outDTypeShortName = GetDTypeShortName(outDType);
        if (!outDTypeShortName) {
            MKI_LOG(ERROR) << "Cast does not support input type: " << GetStrWithDType(outDType);
            return nullptr;
        }
        ss << *outDTypeShortName;
        ss << "Kernel";
        std::string castKernelName = ss.str();
        Kernel *castKernel = GetKernelByName(castKernelName);
        if (castKernel == nullptr) {
            MKI_LOG(ERROR) << "No such cast kernel: " << castKernelName;
        }
        return castKernel;
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        auto inDtype = launchParam.GetInTensor(0).desc.dtype;
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Elewise), "OpParam is invalid", return nullptr);
        OpParam::Elewise param = AnyCast<OpParam::Elewise>(launchParam.GetParam());
        switch (param.elewiseType) {
            case OpParam::Elewise::ELEWISE_CAST:
                return GetCastKernel(launchParam);
            case OpParam::Elewise::ELEWISE_MULS:
                if (inDtype == TENSOR_DTYPE_FLOAT) {
                    return GetKernelByName("MulsF32Kernel");
                } else if (inDtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("MulsF16Kernel");
                } else if (inDtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("MulsBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_MULS inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_COS:
                if (inDtype == TENSOR_DTYPE_FLOAT) {
                    return GetKernelByName("CosF32Kernel");
                } else if (inDtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("CosF16Kernel");
                } else if (inDtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("CosBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_COS inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_SIN:
                if (inDtype == TENSOR_DTYPE_FLOAT) {
                    return GetKernelByName("SinF32Kernel");
                } else if (inDtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("SinF16Kernel");
                } else if (inDtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("SinBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_SIN inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_NEG:
                if (inDtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("NegF16Kernel");
                } else if (inDtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("NegBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_NEG inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_QUANT:
                if (inDtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("QuantF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_QUANT inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_DYNAMIC_QUANT:
                if (inDtype == TENSOR_DTYPE_FLOAT16 || inDtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("DynamicQuantKernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_DYNAMIC_QUANT inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_LOGICAL_NOT:
                if (inDtype == TENSOR_DTYPE_INT8) {
                    return GetKernelByName("LogicalNotKernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_LOGICAL_NOT inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_ADD:
                if (inDtype == TENSOR_DTYPE_FLOAT) {
                    return GetKernelByName("AddF32Kernel");
                } else if (inDtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("AddF16Kernel");
                } else if (inDtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("AddBF16Kernel");
                } else if (inDtype == TENSOR_DTYPE_INT32) {
                    return GetKernelByName("AddI32Kernel");
                } else if (inDtype == TENSOR_DTYPE_INT64) {
                    return GetKernelByName("AddI64Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_ADD inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_SUB:
                if (inDtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("SubF16Kernel");
                } else if (inDtype == TENSOR_DTYPE_INT64) {
                    return GetKernelByName("SubInt64Kernel");
                } else if (inDtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("SubBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_SUB inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_MUL:
                if (inDtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("MulF16Kernel");
                } else if (inDtype == TENSOR_DTYPE_FLOAT) {
                    return GetKernelByName("MulF32Kernel");
                } else if (inDtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("MulBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_MUL inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_GREATER:
                if (inDtype == TENSOR_DTYPE_INT64) {
                    return GetKernelByName("GreaterInt64Kernel");
                } else if (inDtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("GreaterF16Kernel");
                } else if (inDtype == TENSOR_DTYPE_FLOAT) {
                    return GetKernelByName("GreaterF32Kernel");
                } else if (inDtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("GreaterBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_GREATER inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_REALDIV:
                if (inDtype == TENSOR_DTYPE_FLOAT) {
                    return GetKernelByName("RealDivF32Kernel");
                } else if (inDtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("RealDivF16Kernel");
                } else if (inDtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("RealDivBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_REALDIV inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_LOGICAL_AND:
                if (inDtype == TENSOR_DTYPE_INT8) {
                    return GetKernelByName("LogicalAndInt8Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_LOGICAL_AND inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_LOGICAL_OR:
                if (inDtype == TENSOR_DTYPE_INT8) {
                    return GetKernelByName("LogicalOrInt8Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_LOGICAL_OR inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_LESS:
                if (inDtype == TENSOR_DTYPE_INT64) {
                    return GetKernelByName("LessInt64Kernel");
                } else if (inDtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("LessF16Kernel");
                } else if (inDtype == TENSOR_DTYPE_FLOAT) {
                    return GetKernelByName("LessF32Kernel");
                } else if (inDtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("LessBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_LESS inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_TANH:
                if (inDtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("TanhF16Kernel");
                }
                if (inDtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("TanhBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_TANH inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_EQUAL:
                if (inDtype == TENSOR_DTYPE_FLOAT16) {
                    return GetKernelByName("EqualF16Kernel");
                } else if (inDtype == TENSOR_DTYPE_FLOAT) {
                    return GetKernelByName("EqualF32Kernel");
                } else if (inDtype == TENSOR_DTYPE_BF16) {
                    return GetKernelByName("EqualBF16Kernel");
                }
                MKI_LOG(ERROR) << "No kernel for ELEWISE_EQUAL inDtype " << GetStrWithDType(inDtype);
                return nullptr;
            case OpParam::Elewise::ELEWISE_QUANT_PER_CHANNEL:
                return GetKernelByName("AtbQuantPerChannelKernel");
            case OpParam::Elewise::ELEWISE_DEQUANT_PER_CHANNEL:
                if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_310P) {
                    MKI_LOG(ERROR) << "No kernel for dequantperchannel";
                    return nullptr;
                }
                return GetKernelByName("AtbDequantPerChannelKernel");
            default:
                return nullptr;
        }
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::Elewise), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::Elewise>(specificParam);
        const int64_t inTensorNumOne = 1;
        const int64_t inTensorNumTwo = 2;
        switch (param.elewiseType) {
            case OpParam::Elewise::ELEWISE_ADD:
            case OpParam::Elewise::ELEWISE_EQUAL:
            case OpParam::Elewise::ELEWISE_GREATER:
            case OpParam::Elewise::ELEWISE_LESS:
            case OpParam::Elewise::ELEWISE_LOGICAL_AND:
            case OpParam::Elewise::ELEWISE_LOGICAL_OR:
            case OpParam::Elewise::ELEWISE_MUL:
            case OpParam::Elewise::ELEWISE_REALDIV:
            case OpParam::Elewise::ELEWISE_SUB:
                return inTensorNumTwo;
            case OpParam::Elewise::ELEWISE_CAST:
            case OpParam::Elewise::ELEWISE_COS:
            case OpParam::Elewise::ELEWISE_MULS:
            case OpParam::Elewise::ELEWISE_LOGICAL_NOT:
            case OpParam::Elewise::ELEWISE_NEG:
            case OpParam::Elewise::ELEWISE_QUANT:
            case OpParam::Elewise::ELEWISE_DYNAMIC_QUANT:
            case OpParam::Elewise::ELEWISE_SIN:
            case OpParam::Elewise::ELEWISE_TANH:
                return inTensorNumOne;
            case OpParam::Elewise::ELEWISE_QUANT_PER_CHANNEL:
            case OpParam::Elewise::ELEWISE_DEQUANT_PER_CHANNEL:
                return 3; // 3: x + scale + offset
            default:
                return 0;
        }
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::Elewise), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::Elewise>(specificParam);
        switch (param.elewiseType) {
            case OpParam::Elewise::ELEWISE_DYNAMIC_QUANT:
                return 3; // 3 is out tensor num
            default:
                return 1;
        }
    }

protected:
    Status InferShapeCommon(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        TensorDType dtype0 = launchParam.GetInTensor(0).desc.dtype;
        TensorDType dtype1 = launchParam.GetInTensor(1).desc.dtype;
        if (dtype0 != dtype1) {
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Unsupported input descriptor.");
        }

        TensorFormat format = launchParam.GetInTensor(0).desc.format;
        SVector<int64_t> dims0 = launchParam.GetInTensor(0).desc.dims;
        SVector<int64_t> dims1 = launchParam.GetInTensor(1).desc.dims;
        int64_t dims0Size = static_cast<int64_t>(dims0.size());
        int64_t dims1Size = static_cast<int64_t>(dims1.size());
        int64_t largerSize = (dims0.size() >= dims1.size()) ? dims0Size : dims1Size;
        MKI_LOG(INFO) << "largestSize = " << largerSize;
        SVector<int64_t> dimsOut(largerSize);
        int64_t idx = 0;
        int64_t dimDefault = 1;
        while (idx < largerSize) {
            auto dimIdxOut = largerSize - idx - 1;
            auto dimIdx0 = dims0Size - idx - 1;
            auto dimIdx1 = dims1Size - idx - 1;
            auto dim0 = dimIdx0 < 0 ? dimDefault : dims0[dimIdx0];
            auto dim1 = dimIdx1 < 0 ? dimDefault : dims1[dimIdx1];
            MKI_LOG(INFO) << "broadcast idx = " << idx << " dim0 = " << dim0 << " dim1 = " << dim1;
            if (dim0 != 1 && dim1 != 1 && dim0 != dim1) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "cannot broadcast this dim");
            }
            auto dimOut = dim0 >= dim1 ? dim0 : dim1;
            dimsOut[dimIdxOut] = dimOut;
            idx++;
        }
        outTensors[0].desc = {dtype0, format, dimsOut, {}, 0};
        return Status::OkStatus();
    }

    Status SimplyBroadcastInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        size_t inTensorCount = launchParam.GetInTensorCount();
        const SVector<int64_t> &srcDims = launchParam.GetInTensor(1).desc.dims;
        for (size_t i = 2; i < inTensorCount; i++) {
            // 可选
            const Tensor &tensor = launchParam.GetInTensor(i);
            if (CheckEmptyTensor(tensor)) {
                continue;
            }

            if (srcDims != tensor.desc.dims) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "src dims not same");
            }
        }

        if (launchParam.GetInTensor(1).desc.Numel() == 1) {
            outTensors[0].desc = launchParam.GetInTensor(0).desc;
            return Status::OkStatus();
        }

        const SVector<int64_t> &dstDims = launchParam.GetInTensor(0).desc.dims;
        size_t dstDimsSize = dstDims.size();
        size_t srcDimsSize = srcDims.size();
        if (dstDimsSize < srcDimsSize) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "src dims larger than dst");
        }
        for (size_t i = 0; i < srcDimsSize; i++) {
            if (dstDims[dstDimsSize - i - 1] != srcDims[srcDimsSize - i - 1]) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "cannot broadcast this dim");
            }
        }
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        return Status::OkStatus();
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        TensorDType dtype = launchParam.GetInTensor(0).desc.dtype;
        TensorFormat format = launchParam.GetInTensor(0).desc.format;
        SVector<int64_t> dims = launchParam.GetInTensor(0).desc.dims;
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Elewise), "transpose: no match param type",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        OpParam::Elewise param = AnyCast<OpParam::Elewise>(launchParam.GetParam());
        OpParam::Elewise::ElewiseType type = param.elewiseType;
        switch (type) {
            case OpParam::Elewise::ELEWISE_CAST:
                if (param.outTensorType == TENSOR_DTYPE_UNDEFINED) {
                    // for Compatibility
                    if (dtype == TENSOR_DTYPE_FLOAT) {
                        outTensors[0].desc = {TENSOR_DTYPE_FLOAT16, format, dims, {}, 0};
                    } else if (dtype == TENSOR_DTYPE_FLOAT16) {
                        outTensors[0].desc = {TENSOR_DTYPE_FLOAT, format, dims, {}, 0};
                    }
                    return Status::OkStatus();
                } else {
                    outTensors[0].desc = {param.outTensorType, format, dims, {}, 0};
                    return Status::OkStatus();
                }
            case OpParam::Elewise::ELEWISE_MULS:
            case OpParam::Elewise::ELEWISE_TANH:
            case OpParam::Elewise::ELEWISE_COS:
            case OpParam::Elewise::ELEWISE_SIN:
            case OpParam::Elewise::ELEWISE_NEG:
            case OpParam::Elewise::ELEWISE_LOGICAL_NOT:
                outTensors[0].desc = launchParam.GetInTensor(0).desc;
                return Status::OkStatus();
            case OpParam::Elewise::ELEWISE_QUANT:
                if (dtype == TENSOR_DTYPE_FLOAT16) {
                    outTensors[0].desc = launchParam.GetInTensor(0).desc;
                    outTensors[0].desc.dtype = TENSOR_DTYPE_INT8;
                    return Status::OkStatus();
                } else {
                    return Status::FailStatus(ERROR_INVALID_VALUE, "no matched input desc");
                }
            case OpParam::Elewise::ELEWISE_DYNAMIC_QUANT: {
                if (dims.size() < 2) { // 2 is the min allowed dim of intensor
                    return Status::FailStatus(ERROR_INVALID_VALUE, "no matched input desc.dims.size");
                }
                if (dtype == TENSOR_DTYPE_FLOAT16 || dtype == TENSOR_DTYPE_BF16) {
                    outTensors[0].desc.format = format;
                    outTensors[0].desc.dtype = TENSOR_DTYPE_INT8;
                    outTensors[0].desc.dims = dims;
                    outTensors[1].desc.format = format;
                    outTensors[1].desc.dtype = TENSOR_DTYPE_FLOAT;
                    SVector<int64_t> outTensorDims;
                    for (size_t i = 0; i < dims.size() - 1; ++i) {
                        outTensorDims.push_back(dims[i]);
                    }
                    outTensors[1].desc.dims = outTensorDims;

                    outTensors[2].desc = outTensors[1].desc; // 2 is outtensor
                    return Status::OkStatus();
                } else {
                    return Status::FailStatus(ERROR_INVALID_VALUE, "no matched input desc.dtype");
                }
            }
            case OpParam::Elewise::ELEWISE_ADD:
            case OpParam::Elewise::ELEWISE_MUL:
            case OpParam::Elewise::ELEWISE_REALDIV:
            case OpParam::Elewise::ELEWISE_SUB:
                return InferShapeCommon(launchParam, outTensors);
            case OpParam::Elewise::ELEWISE_LESS:
            case OpParam::Elewise::ELEWISE_GREATER:
            case OpParam::Elewise::ELEWISE_EQUAL:
            case OpParam::Elewise::ELEWISE_LOGICAL_AND:
            case OpParam::Elewise::ELEWISE_LOGICAL_OR: {
                Status status = InferShapeCommon(launchParam, outTensors);
                if (status.Ok()) {
                    outTensors[0].desc.dtype = TENSOR_DTYPE_INT8;
                }
                return status;
            }
            case OpParam::Elewise::ELEWISE_QUANT_PER_CHANNEL: {
                Status status = SimplyBroadcastInferShape(launchParam, outTensors);
                outTensors[0].desc.dtype = TENSOR_DTYPE_INT8;
                return status;
            }
            case OpParam::Elewise::ELEWISE_DEQUANT_PER_CHANNEL: {
                Status status = SimplyBroadcastInferShape(launchParam, outTensors);
                outTensors[0].desc.dtype = TENSOR_DTYPE_FLOAT16;
                return status;
            }
            default:
                return Status::FailStatus(ERROR_INVALID_VALUE, "no matched elewiseType");
        }
    }
};
REG_OPERATION(ElewiseOperation);
} // namespace AsdOps
