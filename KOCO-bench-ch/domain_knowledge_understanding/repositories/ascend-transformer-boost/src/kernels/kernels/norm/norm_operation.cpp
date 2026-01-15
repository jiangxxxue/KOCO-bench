/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include <mki/utils/SVector/SVector.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/const/op_const.h>
#include "asdops/params/params.h"

namespace {
constexpr size_t LAYERNORM_TENSOR_COUNT = 3;
constexpr size_t COHERE_LAYER_NORM_GAMMA_DIM_SIZE = 2;
constexpr size_t ADA_LAYER_NORM_GAMMA_DIM_SIZE = 3;
constexpr size_t LAYERNORMQUANT_TENSOR_IN_COUNT = 5;
constexpr size_t POSTLAYERNORM_TENSOR_IN_COUNT = 4;
constexpr size_t POSTLAYERNORM_TENSOR_OUT_COUNT = 1;
constexpr size_t POSTLAYERNORMQUANT_TENSOR_IN_COUNT = 6;
constexpr size_t POSTLAYERNORMQUANT_NORES_TENSOR_IN_COUNT = 3;
constexpr size_t POSTLAYERNORMQUANT_TENSOR_OUT_COUNT = 2;
constexpr size_t RMSPRENORMQUANT_TENSOR_IN_COUNT = 6;
constexpr size_t RMSPRENORMQUANT_TENSOR_OUT_COUNT = 2;
constexpr size_t RMSPOSTNORMQUANT_TENSOR_IN_COUNT = 5;
constexpr size_t RMSPOSTNORMQUANT_TENSOR_OUT_COUNT = 2;
constexpr size_t RMSNORMQUANT_TENSOR_IN_COUNT = 5;
constexpr size_t RMSNORMQUANT_TENSOR_OUT_COUNT = 1;
constexpr size_t TENSOR_THE_FIRST = 0;
constexpr size_t RMSNORM_TENSOR_IN_COUNT = 2;
constexpr size_t RMSNORM_TENSOR_OUT_COUNT = 1;
constexpr size_t TENSOR_THE_SECOND = 1;
constexpr size_t TENSOR_THE_THIRD = 2;
constexpr size_t TENSOR_THE_FOURTH = 3;
constexpr size_t TENSOR_THE_FIFTH = 4;
constexpr size_t TENSOR_THE_SIXTH = 5;
constexpr size_t RMSNORMFORWARD_TENSOR_IN_COUNT = 2;
constexpr size_t RMSNORMFORWARD_TENSOR_OUT_COUNT = 2;
constexpr size_t RMSNORM_BACKWARD_IN_COUNT = 4;
constexpr size_t RMSNORM_BACKWARD_OUT_COUNT = 2;
constexpr size_t QUANT_TENSOR_IN_COUNT = 2;
constexpr size_t NORM_DYNAMIC_QUANT_IN_COUNT = 3;
constexpr size_t NORM_DYNAMIC_QUANT_OUT_COUNT = 3;
constexpr size_t GATHER_PRERMSNORM_IN_COUNT = 4;
constexpr size_t GATHER_PRERMSNORM_OUT_COUNT = 2;
constexpr size_t GATHER_PRERMSNORM_MAX_COLNUM = 7680;

constexpr size_t FP16_NUM_IN_ONE_BLOCK = 16;
constexpr size_t FP32_NUM_IN_ONE_BLOCK = 32;
} // namespace

namespace AsdOps {
using namespace Mki;

bool checkNCT(const LaunchParam &launchParam)
{
    uint32_t rowStrideNum = 2;
    const auto& xStrides = launchParam.GetInTensor(0).desc.strides;
    if (xStrides.empty()) {
        return false;
    }
    const SVector<int64_t>& xShapes = launchParam.GetInTensor(0).desc.dims;
    uint32_t dimNum = xStrides.size();
    MKI_CHECK(dimNum >= rowStrideNum, "noncontiguous tensor dimNum must lager than two", return false);
    return (xStrides[dimNum - rowStrideNum] == xShapes[dimNum - 1]) ? false : true;
}

class NormOperation : public OperationBase {
public:
    explicit NormOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        auto dtype = launchParam.GetInTensor(0).desc.dtype;
        OpParam::Norm param = AnyCast<OpParam::Norm>(launchParam.GetParam());
        switch (param.normType) {
            case OpParam::Norm::LAYER_NORM:
                if (param.inGamma && param.inBeta && !param.inRes && !param.outRes && !param.outMean &&
                    !param.outVarience && !param.outResQuant) {
                    if (checkNCT(launchParam)) {
                        return GetKernelByName("LayerNormStrideKernel");
                    } else {
                        return GetKernelByName("AdaLayerNormKernel");
                    }
                } else if (param.inGamma && param.inBeta && !param.inRes && !param.outRes && param.outMean &&
                    param.outVarience && !param.outResQuant) {
                    if (dtype == TENSOR_DTYPE_FLOAT) {
                        return GetKernelByName("LayernormF32Kernel");
                    } else if (dtype == TENSOR_DTYPE_BF16) {
                        return GetKernelByName("LayernormBF16Kernel");
                    } else if (dtype == TENSOR_DTYPE_FLOAT16) {
                        return GetKernelByName("LayernormF16Kernel");
                    } else {
                        return nullptr;
                    }
                } else if (param.inGamma && !param.inBeta && !param.inRes && !param.outRes && !param.outMean &&
                          !param.outVarience && !param.outResQuant) {
                    return GetKernelByName("CohereLayernormKernel");
                } else if (param.inGamma && param.inBeta && !param.inRes && !param.outRes && !param.outMean &&
                          !param.outVarience && param.outResQuant) {
                    return !param.isDynamicQuant ? GetKernelByName("LayerNormF16QuantKernel") :
                                                   GetKernelByName("LayerNormDynamicQuantKernel");
                } else if (param.inGamma && param.inBeta && param.inRes && !param.outRes && !param.outMean &&
                           !param.outVarience && !param.outResQuant) {
                    return GetKernelByName("PostLayerNormF16Kernel");
                } else if (param.inGamma && param.inBeta && param.inRes && param.outRes && !param.outMean &&
                           !param.outVarience && !param.outResQuant) {
                    return GetKernelByName("PreLayerNormKernel");
                } else if (param.inGamma && param.inBeta && param.inRes && !param.outRes && !param.outMean &&
                           !param.outVarience && param.outResQuant) {
                    return GetKernelByName("PostLayerNormF16QuantKernel");
                } else {
                    return nullptr;
                }
            case OpParam::Norm::RMS_NORM:
                if (param.inGamma && !param.inBeta && !param.outRes && !param.inRes && !param.inNormBias) {
                    return GetKernelByName("RmsNormKernel");
                } else if (param.inGamma && param.inBeta && param.outRes && param.inRes && !param.inNormBias) {
                    return GetKernelByName("PreRmsNormKernel");
                }  else if (param.inGamma && param.inBeta && !param.outRes && param.inRes && !param.inNormBias) {
                    return GetKernelByName("PostRmsNormKernel");
                } else if (param.inGamma && param.inBeta && !param.outRes && !param.inRes && !param.inNormBias) {
                    return !param.isDynamicQuant ? GetKernelByName("RmsNormQuantKernel") :
                                                   GetKernelByName("RmsNormDynamicQuantKernel");
                } else if (param.inGamma && !param.inBeta && param.outRes && param.inRes && param.inNormBias) {
                    return GetKernelByName("RmsPreNormQuantKernel");
                } else if (param.inGamma && !param.inBeta && param.outRes && param.inRes && !param.inNormBias) {
                    return GetKernelByName("RmsPostNormQuantKernel");
                } else {
                    return nullptr;
                }
            case OpParam::Norm::RMS_NORM_FORWARD:
                return GetKernelByName("RmsNormForwardKernel");
            case OpParam::Norm::RMS_NORM_BACKWARD:
                return GetKernelByName("RmsNormBackwardKernel");
            case OpParam::Norm::GATHER_PRE_RMS_NORM:
                return GetKernelByName("GatherPreRmsNormKernel");
            default: return nullptr;
        }
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        int64_t inputNum = 1;
        MKI_CHECK(specificParam.Type() == typeid(OpParam::Norm), "OpParam is invalid", return 0);
        OpParam::Norm param = AnyCast<OpParam::Norm>(specificParam);
        inputNum += param.inGamma ? 1 : 0;
        inputNum += param.inBeta ? 1 : 0;
        inputNum += param.inRes ? 1 : 0;
        inputNum += param.inNormBias ? 1 : 0;

        if (param.isDynamicQuant) {
            return NORM_DYNAMIC_QUANT_IN_COUNT;
        }
        if (param.normType == OpParam::Norm::LAYER_NORM && param.outResQuant) {
            inputNum += QUANT_TENSOR_IN_COUNT;
        }
        if (param.normType == OpParam::Norm::RMS_NORM) {
            bool rmsNormQuant = param.inGamma && param.inBeta && !param.outRes && !param.inRes && !param.inNormBias;
            bool rmsPreNormQuant = param.inGamma && !param.inBeta && param.outRes && param.inRes && param.inNormBias;
            bool rmsPostNormQuant = param.inGamma && !param.inBeta && param.outRes && param.inRes && !param.inNormBias;
            inputNum = (rmsNormQuant) ? RMSNORMQUANT_TENSOR_IN_COUNT : inputNum;
            inputNum = (rmsPreNormQuant) ? RMSPRENORMQUANT_TENSOR_IN_COUNT : inputNum;
            inputNum = (rmsPostNormQuant) ? RMSPOSTNORMQUANT_TENSOR_IN_COUNT : inputNum;
        }
        if (param.normType == OpParam::Norm::RMS_NORM_FORWARD) {
            inputNum = RMSNORMFORWARD_TENSOR_IN_COUNT;
        }
        if (param.normType == OpParam::Norm::RMS_NORM_BACKWARD) {
            inputNum = RMSNORM_BACKWARD_IN_COUNT;
        }
        if (param.normType == OpParam::Norm::GATHER_PRE_RMS_NORM) {
            inputNum = GATHER_PRERMSNORM_IN_COUNT;
        }
        return inputNum;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        int64_t outputNum = 1;
        MKI_CHECK(specificParam.Type() == typeid(OpParam::Norm), "OpParam is invalid", return 0);
        OpParam::Norm param = AnyCast<OpParam::Norm>(specificParam);
        outputNum += param.outMean ? 1 : 0;
        outputNum += param.outVarience ? 1 : 0;
        outputNum += param.outResQuant ? 1 : 0;
        outputNum += param.outRes ? 1 : 0;

        if (param.isDynamicQuant) {
            return NORM_DYNAMIC_QUANT_OUT_COUNT;
        }
        bool layerNormQuant =
            param.inGamma && param.inBeta && !param.inRes && !param.outMean && !param.outVarience && param.outResQuant;
        if (param.normType == OpParam::Norm::LAYER_NORM && layerNormQuant) {
            outputNum = 1;
        }
        if (param.normType == OpParam::Norm::RMS_NORM_FORWARD) {
            outputNum = RMSNORMFORWARD_TENSOR_OUT_COUNT;
        }
        if (param.normType == OpParam::Norm::RMS_NORM_BACKWARD) {
            outputNum = RMSNORM_BACKWARD_OUT_COUNT;
        }
        if (param.normType == OpParam::Norm::GATHER_PRE_RMS_NORM) {
            outputNum = GATHER_PRERMSNORM_OUT_COUNT;
        }
        return outputNum;
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Norm), "OpParam is invalid",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        OpParam::Norm param = AnyCast<OpParam::Norm>(launchParam.GetParam());
        OpParam::Norm::NormType type = param.normType;

        if (type == OpParam::Norm::LAYER_NORM) {
            if (param.inGamma && param.inBeta && !param.inRes && !param.outRes && !param.outMean && !param.outVarience
                && !param.outResQuant) {
                if (checkNCT(launchParam)) {
                    return LayerNormStrideInferShape(launchParam, outTensors);
                } else {
                    return AdaLayernormInferShape(launchParam, outTensors);
                }
            } else if (param.inGamma && param.inBeta && !param.inRes && !param.outRes && param.outMean &&
                       param.outVarience && !param.outResQuant) {
                return LayernormInferShape(launchParam, outTensors);
            } else if (param.inGamma && !param.inBeta && !param.inRes && !param.outRes && !param.outMean &&
                          !param.outVarience && !param.outResQuant) {
                return CohereLayernormInferShape(launchParam, outTensors);
            } else if (param.inGamma && param.inBeta && !param.inRes && !param.outRes && !param.outMean &&
                       !param.outVarience && param.outResQuant) {
                return !param.isDynamicQuant ? LayernormF16QuantInferShape(launchParam, outTensors) :
                                               NormDynamicQuantInferShape(launchParam, outTensors);
            } else if (param.inGamma && param.inBeta && param.inRes && !param.outRes && !param.outMean &&
                       !param.outVarience && !param.outResQuant) {
                return PostLayernormF16InferShape(launchParam, outTensors);
            } else if (param.inGamma && param.inBeta && param.inRes && param.outRes && !param.outMean &&
                       !param.outVarience && !param.outResQuant) {
                return PreLayernormF16InferShape(launchParam, outTensors);
            } else if (param.inGamma && param.inBeta && param.inRes && !param.outRes && !param.outMean &&
                       !param.outVarience && param.outResQuant) {
                return PostLayernormF16QuantInferShape(launchParam, outTensors);
            } else {
                return Status::FailStatus(ERROR_INVALID_VALUE, "inferShape failed");
            }
        } else if (type == OpParam::Norm::RMS_NORM) {
            if (param.inGamma && !param.inBeta && !param.outRes && !param.inRes && !param.inNormBias) {
                return RmsNormInferShape(launchParam, outTensors);
            } else if (param.inGamma && param.inBeta && !param.outRes && !param.inRes && !param.inNormBias) {
                return !param.isDynamicQuant ? RmsNormQuantInferShape(launchParam, outTensors) :
                                               NormDynamicQuantInferShape(launchParam, outTensors);
            } else if (param.inGamma && !param.inBeta && param.outRes && param.inRes && param.inNormBias) {
                return RmsPreNormQuantInferShape(launchParam, outTensors);
            } else if (param.inGamma && !param.inBeta && param.outRes && param.inRes && !param.inNormBias) {
                return RmsPostNormQuantInferShape(launchParam, outTensors);
            } else if (param.inGamma && param.inBeta && !param.outRes && param.inRes && !param.inNormBias) {
                return PostRmsNormInferShape(launchParam, outTensors);
            } else if (param.inGamma && param.inBeta && param.outRes && param.inRes && !param.inNormBias) {
                return PreRmsNormInferShape(launchParam, outTensors);
            } else {
                return Status::FailStatus(ERROR_INVALID_VALUE, "inferShape failed");
            }
        } else if (type == OpParam::Norm::RMS_NORM_FORWARD) {
            return RmsNormForwardInferShape(launchParam, outTensors);
        } else if (type == OpParam::Norm::RMS_NORM_BACKWARD) {
            return RmsNormBackwardInferShape(launchParam, outTensors);
        } else if (type == OpParam::Norm::GATHER_PRE_RMS_NORM) {
            return GatherPreRmsNormInferShape(launchParam, outTensors);
        } else {
            return Status::FailStatus(ERROR_INVALID_VALUE, "inferShape failed");
        }
    }

private:
    Status RmsNormInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        MKI_CHECK(CheckRmsNormLaunchParam(launchParam), "Failed to Check RmsNormInferShape",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR,
                                               "Failed to Check RmsNormInferShape"));
        const SVector<int64_t> &inDims0 = launchParam.GetInTensor(0).desc.dims;
        const SVector<int64_t> &inDims1 = launchParam.GetInTensor(1).desc.dims;
        if (inDims0.empty() || inDims1.empty() || inDims0[inDims0.size() - 1] != inDims1[inDims1.size() - 1]) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "inDims is empty or the last dimension should be the same");
        }
        outTensors[0].desc = launchParam.GetInTensor(0).desc;

        return Status::OkStatus();
    }

    Status PostRmsNormInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        return Status::OkStatus();
    }

    Status PreRmsNormInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        outTensors[1].desc = launchParam.GetInTensor(0).desc;
        return Status::OkStatus();
    }

    Status RmsNormForwardInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        MKI_CHECK(CheckRmsNorm(launchParam), "Failed to Check RmsNormForwardInferShape",
                     return Status::FailStatus(ERROR_INFERSHAPE_ERROR,
                                               "Failed to Check RmsNormForwardInferShape"));
        outTensors[TENSOR_THE_FIRST].desc = launchParam.GetInTensor(TENSOR_THE_FIRST).desc;
        outTensors[TENSOR_THE_SECOND].desc = launchParam.GetInTensor(TENSOR_THE_FIRST).desc;
        outTensors[TENSOR_THE_SECOND].desc.dtype = TENSOR_DTYPE_FLOAT;
        auto gammaDesc = launchParam.GetInTensor(TENSOR_THE_SECOND).desc;
        size_t xDimNum = launchParam.GetInTensor(TENSOR_THE_FIRST).desc.dims.size();
        size_t gammaDimNum = gammaDesc.dims.size();
        for (size_t i = 0; i < xDimNum; i++) {
            if (i >= xDimNum - gammaDimNum) {
            outTensors[TENSOR_THE_SECOND].desc.dims[i] = 1;
            }
        }
        return AsdOps::Status::OkStatus();
    }

    bool CheckRmsNorm(const LaunchParam &launchParam) const
    {
        auto xDim = launchParam.GetInTensor(TENSOR_THE_FIRST).desc.dims;
        auto gammaDim = launchParam.GetInTensor(TENSOR_THE_SECOND).desc.dims;

        int32_t xDimNum = static_cast<int32_t>(launchParam.GetInTensor(TENSOR_THE_FIRST).desc.dims.size());
        size_t gammaDimNum = launchParam.GetInTensor(TENSOR_THE_SECOND).desc.dims.size();
        int32_t startDim = xDimNum - static_cast<int32_t>(gammaDimNum);
        MKI_CHECK(startDim >= 0, "gamma shape invalid", return false);
        for (int32_t i = startDim; i < xDimNum; i++) {
            MKI_CHECK(xDim[i] == gammaDim[i - startDim],
                         "gamma shape invalid", return false);
        }
        return true;
    }

    bool CheckRmsNormLaunchParam(const LaunchParam &launchParam) const
    {
        const Mki::SVector<int64_t> &shape1 = launchParam.GetInTensor(1).desc.dims;
        MKI_CHECK(!CheckEmptyTensor(launchParam.GetInTensor(0)),
                     "shape1 should not be empty", return false);
        MKI_CHECK(!CheckEmptyTensor(launchParam.GetInTensor(1)),
                     "shape1 should not be empty", return false);
        if (shape1.size() >= DIM_2) {
            for (size_t i = 0; i < shape1.size() - DIM_1; i++) {
                MKI_CHECK(shape1[i] == DIM_1, "The shape dimension should be preceded by 1", return false);
            }
        }
        return true;
    }
    
    Status RmsNormBackwardInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        outTensors[TENSOR_THE_FIRST].desc = launchParam.GetInTensor(TENSOR_THE_SECOND).desc;
        outTensors[TENSOR_THE_SECOND].desc = launchParam.GetInTensor(TENSOR_THE_FOURTH).desc;
        outTensors[TENSOR_THE_SECOND].desc.dtype = TENSOR_DTYPE_FLOAT;
        return Status::OkStatus();
    }

    Status LayernormInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        OpParam::Norm param = AnyCast<OpParam::Norm>(launchParam.GetParam());
        int64_t beginNormAxis = param.beginNormAxis;
        const SVector<int64_t> &inDims = launchParam.GetInTensor(0).desc.dims;
        const SVector<int64_t> &inDimsGamma = launchParam.GetInTensor(TENSOR_THE_SECOND).desc.dims;
        const SVector<int64_t> &inDimsBeta = launchParam.GetInTensor(TENSOR_THE_THIRD).desc.dims;
        MKI_CHECK(inDims.size() > 0, "inDims is empty",
            return Status::FailStatus(ERROR_INVALID_VALUE, "inDims is empty"));
        if (beginNormAxis < 0) {
            beginNormAxis += static_cast<int64_t>(inDims.size());
        }

        MKI_CHECK_NO_LOG(beginNormAxis >= 0 && beginNormAxis < static_cast<int64_t>(inDims.size()),
            return Status::FailStatus(ERROR_INVALID_VALUE, "beginNormAxis should be less than inDims.size()"));

        if (static_cast<size_t>(beginNormAxis) > inDims.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "wrong beginNormAxis > inDims");
        }
        if (inDimsGamma.size() > inDims.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "wrong inDimsGamma larger than inDims");
        }
        if (inDimsGamma.size() + static_cast<size_t>(beginNormAxis) != inDims.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "wrong inDimsGamma or inDims");
        }
        if (inDimsGamma.size() != inDimsBeta.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsGamma are not equal to inDimsBeta");
        }
        for (size_t i = 0; i < inDimsGamma.size(); ++i) {
            if (inDimsGamma[i] != inDimsBeta[i]) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsGamma are not equal to inDimsBeta");
            }
        }
        for (size_t i = 0; i < inDimsGamma.size(); ++i) {
            if (inDimsGamma[i] != inDims[i + static_cast<size_t>(beginNormAxis)]) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "wrong dims");
            }
        }

        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        outTensors[TENSOR_THE_SECOND].desc = launchParam.GetInTensor(0).desc;
        outTensors[TENSOR_THE_THIRD].desc = launchParam.GetInTensor(0).desc;
        for (size_t i = 0; i < outTensors[0].desc.dims.size(); i++) {
            if (i >= (size_t)beginNormAxis) {
                outTensors[TENSOR_THE_SECOND].desc.dims[i] = 1;
                outTensors[TENSOR_THE_THIRD].desc.dims[i] = 1;
            }
        }

        return Status::OkStatus();
    }

    Status CohereLayernormInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        const SVector<int64_t> &inDims = launchParam.GetInTensor(0).desc.dims;
        const SVector<int64_t> &inDimsGamma = launchParam.GetInTensor(TENSOR_THE_SECOND).desc.dims;
 
        if (inDimsGamma.size() != COHERE_LAYER_NORM_GAMMA_DIM_SIZE) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "The dims of inDimsGamma should be equal to 2");
        }
        for (size_t i = 0; i < inDimsGamma.size(); ++i) {
            int64_t dimsSize = inDims.size();
            if (inDimsGamma[i] != inDims[dimsSize - COHERE_LAYER_NORM_GAMMA_DIM_SIZE + i]) {
                return Status::FailStatus(ERROR_INVALID_VALUE,
                                          "The dims of inDimsGamma should be equal to the last two dims of inDims");
            }
        }
 
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
 
        return Status::OkStatus();
    }

    Status AdaLayernormInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        const SVector<int64_t> &inDimsX = launchParam.GetInTensor(0).desc.dims;
        const SVector<int64_t> &inDimsGamma = launchParam.GetInTensor(TENSOR_THE_SECOND).desc.dims;
        const SVector<int64_t> &inDimsBeta = launchParam.GetInTensor(TENSOR_THE_THIRD).desc.dims;
        size_t inDimsSizeX = inDimsX.size();
        size_t inDimsSizeGamma = inDimsGamma.size();
        if (inDimsX[inDimsSizeX - 1] % FP16_NUM_IN_ONE_BLOCK != 0) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "the lastdim of X is not a multiple of 32-byte");
        }
        if (inDimsGamma[inDimsSizeGamma - 1] != inDimsX[inDimsSizeX - 1]) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "the lastdim of Gamma is not equal to X");
        }
        MKI_CHECK((inDimsX[0] % inDimsGamma[0]) == 0, "the batch of X is not multiple of Gamma", return
            Status::FailStatus(ERROR_INVALID_VALUE));
        if (inDimsSizeGamma == ADA_LAYER_NORM_GAMMA_DIM_SIZE) { // 3d adalayernorm
            MKI_CHECK((inDimsX[1] % inDimsGamma[1]) == 0, "the seqlen of X is not multiple of Gamma", return
                Status::FailStatus(ERROR_INVALID_VALUE));
        }
        if (inDimsGamma.size() != inDimsBeta.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "the number of Gamma dims are not equal to Beta");
        }
        for (size_t i = 0; i < inDimsGamma.size(); ++i) {
            if (inDimsGamma[i] != inDimsBeta[i]) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "the dims of Gamma are not equal to Beta");
            }
        }
        outTensors[0].desc = launchParam.GetInTensor(0).desc;

        return Status::OkStatus();
    }

    Status PreLayernormF16InferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        Status status = PostLayernormF16InferShape(launchParam, outTensors);
        MKI_CHECK(status.Ok(), "Fail check PreLayernormInfershape", return
            Status::FailStatus(ERROR_INVALID_VALUE));
        outTensors[1].desc.format = launchParam.GetInTensor(1).desc.format;
        outTensors[1].desc.dtype = launchParam.GetInTensor(1).desc.dtype;
        outTensors[1].desc.dims = launchParam.GetInTensor(1).desc.dims;
        return Status::OkStatus();
    }

    Status PostLayernormF16InferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        const SVector<int64_t> &inDimsInpX = launchParam.GetInTensor(0).desc.dims;
        const SVector<int64_t> &inDimsResIn = launchParam.GetInTensor(TENSOR_THE_SECOND).desc.dims;
        const SVector<int64_t> &inDimsGamma = launchParam.GetInTensor(TENSOR_THE_THIRD).desc.dims;
        const SVector<int64_t> &inDimsBeta = launchParam.GetInTensor(TENSOR_THE_FOURTH).desc.dims;
        if (inDimsInpX.size() != inDimsResIn.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsInpX are not equal inDimsResIn");
        }
        for (size_t i = 0; i < inDimsInpX.size(); i++) {
            if (inDimsInpX[i] != inDimsResIn[i]) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsInpX are not equal inDimsResIn");
            }
        }
        if (inDimsGamma.size() != inDimsBeta.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsGamma are not equal inDimsBeta");
        }
        for (size_t i = 0; i < inDimsGamma.size(); i++) {
            if (inDimsGamma[i] != inDimsBeta[i]) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsGamma are not equal inDimsBeta");
            }
        }
        outTensors[0].desc.format = launchParam.GetInTensor(0).desc.format;
        outTensors[0].desc.dtype = launchParam.GetInTensor(0).desc.dtype;
        outTensors[0].desc.dims = launchParam.GetInTensor(0).desc.dims;
        return Status::OkStatus();
    }

    Status PostLayernormF16QuantInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        const Mki::SVector<int64_t> &inDimsInpX = launchParam.GetInTensor(TENSOR_THE_FIRST).desc.dims;
        const Mki::SVector<int64_t> &inDimsGamma = launchParam.GetInTensor(TENSOR_THE_SECOND).desc.dims;
        const Mki::SVector<int64_t> &inDimsBeta = launchParam.GetInTensor(TENSOR_THE_THIRD).desc.dims;
        const Mki::SVector<int64_t> &inDimsResIn = launchParam.GetInTensor(TENSOR_THE_FOURTH).desc.dims;
        const Tensor &inputScale = launchParam.GetInTensor(TENSOR_THE_FIFTH);
        const Tensor &inputOffset = launchParam.GetInTensor(TENSOR_THE_SIXTH);
        if (inDimsInpX.size() != inDimsResIn.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsInpX are not equal to inDimsResIn");
        }
        for (size_t i = 0; i < inDimsResIn.size(); i++) {
            if (inDimsInpX[i] != inDimsResIn[i]) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "single inDimsInpX are not equal to inDimsResIn");
            }
        }
        if (inDimsGamma.size() != inDimsBeta.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsGamma are not equal inDimsBeta");
        }
        for (size_t i = 0; i < inDimsGamma.size(); i++) {
            if (inDimsGamma[i] != inDimsBeta[i]) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsGamma are not equal inDimsBeta");
            }
        }
        if (inputScale.Numel() != 1) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "InputScale shape is not [1]");
        }
        if (inputOffset.Numel() != 1) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "InputOffset shape is not [1]");
        }
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        outTensors[0].desc.dtype = TENSOR_DTYPE_INT8;
        outTensors[1].desc = launchParam.GetInTensor(0).desc;
        return Status::OkStatus();
    }

    Status RmsNormQuantInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        const Tensor &inInpB = launchParam.GetInTensor(TENSOR_THE_THIRD);
        const Mki::SVector<int64_t> &inDimsInpG = launchParam.GetInTensor(TENSOR_THE_SECOND).desc.dims;
        const Mki::SVector<int64_t> &inDimsInpB = launchParam.GetInTensor(TENSOR_THE_THIRD).desc.dims;
        const Tensor &inputScale = launchParam.GetInTensor(TENSOR_THE_FOURTH);
        const Tensor &inputOffset = launchParam.GetInTensor(TENSOR_THE_FIFTH);
        if (!CheckEmptyTensor(inInpB)) {
            if (inDimsInpG.size() != inDimsInpB.size()) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsInpG are not equal to inDimsInpB");
            }
            for (size_t i = 0; i < inDimsInpG.size(); i++) {
                if (inDimsInpG[i] != inDimsInpB[i]) {
                    return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsInpG are not equal to inDimsInpB");
                }
            }
        }

        if (inputScale.Numel() != 1) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "InputScale shape is not [1]");
        }
        if (inputOffset.Numel() != 1) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "InputOffset shape is not [1]");
        }
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        outTensors[0].desc.dtype = TENSOR_DTYPE_INT8;
        return Status::OkStatus();
    }

    Status NormDynamicQuantInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        const Tensor &inInpB = launchParam.GetInTensor(TENSOR_THE_THIRD);
        const Mki::SVector<int64_t> &inDimsInpG = launchParam.GetInTensor(TENSOR_THE_SECOND).desc.dims;
        const Mki::SVector<int64_t> &inDimsInpB = launchParam.GetInTensor(TENSOR_THE_THIRD).desc.dims;
        if (!CheckEmptyTensor(inInpB)) {
            if (inDimsInpG.size() != inDimsInpB.size()) {
                return Status::FailStatus(ERROR_INVALID_VALUE,
                    "NormDynamicQuant inDimsInpG are not equal to inDimsInpB");
            }
            for (size_t i = 0; i < inDimsInpG.size(); i++) {
                if (inDimsInpG[i] != inDimsInpB[i]) {
                    return Status::FailStatus(ERROR_INVALID_VALUE,
                        "NormDynamicQuant inDimsInpG are not equal to inDimsInpB");
                }
            }
        }

        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        outTensors[0].desc.dtype = TENSOR_DTYPE_INT8;

        Mki::SVector<int64_t> dims;
        for (size_t i = 0; i < launchParam.GetInTensor(0).desc.dims.size() - 1; ++i) {
            dims.push_back(launchParam.GetInTensor(0).desc.dims[i]);
        }
        outTensors[TENSOR_THE_SECOND].desc = outTensors[0].desc;
        outTensors[TENSOR_THE_SECOND].desc.dims = dims;
        outTensors[TENSOR_THE_SECOND].desc.dtype = TENSOR_DTYPE_FLOAT;

        outTensors[TENSOR_THE_THIRD].desc = outTensors[TENSOR_THE_SECOND].desc;

        return Status::OkStatus();
    }

    Status LayerNormStrideInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        const Mki::SVector<int64_t> &inDimsGamma = launchParam.GetInTensor(TENSOR_THE_SECOND).desc.dims;
        const Mki::SVector<int64_t> &inDimsBeta = launchParam.GetInTensor(TENSOR_THE_THIRD).desc.dims;
        if (inDimsGamma.size() != inDimsBeta.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsGamma are not equal to inDimsBeta");
        }
        for (size_t i = 0; i < inDimsGamma.size(); i++) {
            if (inDimsGamma[i] != inDimsBeta[i]) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsGamma are not equal to inDimsBeta");
            }
        }
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        return Status::OkStatus();
    }

    Status LayernormF16QuantInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        const Mki::SVector<int64_t> &inDimsGamma = launchParam.GetInTensor(TENSOR_THE_SECOND).desc.dims;
        const Mki::SVector<int64_t> &inDimsBeta = launchParam.GetInTensor(TENSOR_THE_THIRD).desc.dims;
        const Tensor &inputScale = launchParam.GetInTensor(TENSOR_THE_FOURTH);
        const Tensor &inputOffset = launchParam.GetInTensor(TENSOR_THE_FIFTH);
        if (inDimsGamma.size() != inDimsBeta.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsGamma are not equal to inDimsBeta");
        }
        for (size_t i = 0; i < inDimsGamma.size(); i++) {
            if (inDimsGamma[i] != inDimsBeta[i]) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsGamma are not equal to inDimsBeta");
            }
        }
        if (inputScale.Numel() != 1) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "InputScale shape is not [1]");
        }
        if (inputOffset.Numel() != 1) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "InputOffset shape is not [1]");
        }
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        outTensors[0].desc.dtype = TENSOR_DTYPE_INT8;
        return Status::OkStatus();
    }

    Status RmsPreNormQuantInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        const Mki::SVector<int64_t> &inDimsInpX = launchParam.GetInTensor(TENSOR_THE_FIRST).desc.dims;
        const Mki::SVector<int64_t> &inDimsResIn = launchParam.GetInTensor(TENSOR_THE_SECOND).desc.dims;
        const Mki::SVector<int64_t> &inDimsGamma = launchParam.GetInTensor(TENSOR_THE_THIRD).desc.dims;
        const Mki::SVector<int64_t> &inDimsNormBias = launchParam.GetInTensor(TENSOR_THE_FOURTH).desc.dims;
        const Tensor &inputScale = launchParam.GetInTensor(TENSOR_THE_FIFTH);
        const Tensor &inputOffset = launchParam.GetInTensor(TENSOR_THE_SIXTH);
        if (inDimsInpX.size() != inDimsResIn.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsInpX are not equal to inDimsResIn");
        }
        for (size_t i = 0; i < inDimsInpX.size(); i++) {
            if (inDimsInpX[i] != inDimsResIn[i]) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsInpX are not equal to inDimsResIn");
            }
        }
        if (inDimsGamma.size() != inDimsNormBias.size()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsGamma are not equal to inDimsNormBias");
        }
        for (size_t i = 0; i < inDimsGamma.size(); i++) {
            if (inDimsGamma[i] != inDimsNormBias[i]) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsGamma are not equal to inDimsNormBias");
            }
        }
        if (inputScale.Numel() != 1) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "InputScale shape is not [1]");
        }
        if (inputOffset.Numel() != 1) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "InputOffset shape is not [1]");
        }
        outTensors[0].desc = launchParam.GetInTensor(0).desc;
        outTensors[0].desc.dtype = TENSOR_DTYPE_INT8;
        outTensors[1].desc = launchParam.GetInTensor(0).desc;
        return Status::OkStatus();
    }
    
    Status RmsPostNormQuantInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        if (launchParam.GetInTensorCount() != RMSPOSTNORMQUANT_TENSOR_IN_COUNT) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "input num invalid");
        }
        if (launchParam.GetOutTensorCount() != RMSPOSTNORMQUANT_TENSOR_OUT_COUNT) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "output num invalid");
        }

        const Mki::SVector<int64_t> &inDimsInpX = launchParam.GetInTensor(TENSOR_THE_FIRST).desc.dims;
        const Mki::SVector<int64_t> &inDimsResIn = launchParam.GetInTensor(TENSOR_THE_SECOND).desc.dims;
        const Mki::SVector<int64_t> &inDimsGamma = launchParam.GetInTensor(TENSOR_THE_THIRD).desc.dims;
        const Tensor &inputScale = launchParam.GetInTensor(TENSOR_THE_FOURTH);
        const Tensor &inputOffset = launchParam.GetInTensor(TENSOR_THE_FIFTH);

        if (inDimsInpX.empty() || inDimsResIn.empty() || inDimsGamma.empty()) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsInpx or inDimsResIn or inDimsGamma is empty");
        }

        size_t dimsSizeX = inDimsInpX.size();
        size_t dimsSizeG = inDimsGamma.size();
        if (dimsSizeX == 0 || dimsSizeG == 0) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "dimsSize is invalid");
        }

        uint64_t lastDimX = inDimsInpX[dimsSizeX - 1];
        if (lastDimX % FP32_NUM_IN_ONE_BLOCK != 0) {
            return Status::FailStatus(ERROR_INVALID_VALUE,
                "the last dimension of the input or the output is not 32-byte aligned");
        }

        if (inDimsInpX != inDimsResIn) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "inDimsInpX are not equal to inDimsResIn");
        }

        if (inDimsInpX[inDimsInpX.size() - 1] != inDimsGamma[inDimsGamma.size() - 1]) {
            return Status::FailStatus(ERROR_INVALID_VALUE,
                "the last dimension of inDimsInpX and inDimsGamma should be the same");
        }

        if (inputScale.Numel() != 1) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "InputScale shape is not [1]");
        }

        if (inputOffset.Numel() != 1) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "InputOffset shape is not [1]");
        }

        outTensors[TENSOR_THE_FIRST].desc = launchParam.GetInTensor(TENSOR_THE_FIRST).desc;
        outTensors[TENSOR_THE_FIRST].desc.dtype = TENSOR_DTYPE_INT8;
        outTensors[TENSOR_THE_SECOND].desc = launchParam.GetInTensor(TENSOR_THE_FIRST).desc;
        
        return Status::OkStatus();
    }

    Status GatherPreRmsNormInferShape(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        if (launchParam.GetInTensorCount() != GATHER_PRERMSNORM_IN_COUNT) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "input num invalid");
        }
        if (launchParam.GetOutTensorCount() != GATHER_PRERMSNORM_OUT_COUNT) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "output num invalid");
        }

        const Mki::SVector<int64_t> &inDimsInpX = launchParam.GetInTensor(TENSOR_THE_FIRST).desc.dims;
        const Mki::SVector<int64_t> &inDimsResIn = launchParam.GetInTensor(TENSOR_THE_SECOND).desc.dims;
        const Mki::SVector<int64_t> &inDimsInpIndices = launchParam.GetInTensor(TENSOR_THE_THIRD).desc.dims;
        const Mki::SVector<int64_t> &inDimsGamma = launchParam.GetInTensor(TENSOR_THE_FOURTH).desc.dims;
        if (inDimsInpX.empty() || inDimsInpIndices.empty() || inDimsResIn.empty() || inDimsGamma.empty()) {
            return Status::FailStatus(ERROR_INVALID_VALUE,
                                      "inDimsInpx or inDimsInpIndices or inDimsResIn or inDimsGamma is empty");
        }

        size_t dimsSizeX = inDimsInpX.size();
        size_t dimsSizeIndices = inDimsInpIndices.size();
        size_t dimsSizeResIn = inDimsResIn.size();
        size_t dimsSizeG = inDimsGamma.size();
        if (dimsSizeX != DIM_2 || dimsSizeIndices != DIM_1 || dimsSizeResIn != DIM_2 ||
            (dimsSizeG != DIM_2 && dimsSizeG != DIM_1)) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "dimsSize is invalid");
        }

        uint64_t lastDimX = inDimsInpX[dimsSizeX - 1];
        if (lastDimX > GATHER_PRERMSNORM_MAX_COLNUM) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "the last dimension of the input is greater than 7680");
        }

        if (lastDimX % FP16_NUM_IN_ONE_BLOCK != 0) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "the last dimension of the input is not 32-byte aligned");
        }

        if (inDimsInpX[dimsSizeX - 1] != inDimsResIn[dimsSizeResIn - 1] ||
            inDimsInpX[dimsSizeX - 1] != inDimsGamma[dimsSizeG - 1]) {
            return Status::FailStatus(ERROR_INVALID_VALUE,
                                      "the last dimension of the input X, ResIn and Gamma must be equal");
        }

        if (inDimsInpIndices[0] != inDimsInpX[0]) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "the first dimension of indices and input X must be equal");
        }

        outTensors[TENSOR_THE_FIRST].desc = launchParam.GetInTensor(TENSOR_THE_FIRST).desc;
        outTensors[TENSOR_THE_FIRST].desc.dtype = TENSOR_DTYPE_FLOAT;
        outTensors[TENSOR_THE_SECOND].desc = launchParam.GetInTensor(TENSOR_THE_FIRST).desc;

        return Status::OkStatus();
    }
};
REG_OPERATION(NormOperation);
} //    namespace AsdOps
