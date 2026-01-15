/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>

#include <mki/base/operation_base.h>
#include <mki/types.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/env/env.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/strings/str_replace.h>
#include <mki_loader/op_register.h>

#include "asdops/params/matmul.h"

namespace {
constexpr uint32_t DTYPE_BIT_COUNT = 2;
constexpr uint32_t FORMAT_BIT_COUNT = 1;
constexpr uint32_t INPUT_BIT_COUNT = 2;
constexpr uint32_t PP_MATMUL_I8_BF16_KERNEL_KEY = 0b1'00'00'10'0'0'0'11;
constexpr uint32_t PP_MATMUL_I8_FP16_KERNEL_KEY = 0b1'00'00'01'0'0'0'11;
constexpr uint32_t PP_MATMUL_I8_BF16_WEIGHT_NZ_KERNEL_KEY = 0b1'00'00'10'0'1'0'11;
constexpr uint32_t PP_MATMUL_I8_FP16_WEIGHT_NZ_KERNEL_KEY = 0b1'00'00'01'0'1'0'11;
constexpr uint32_t PP_MATMUL_I8_KERNEL_KEY = 0b1'00'00'01'0'0'0'11;
constexpr uint32_t PP_MATMUL_I8_WEIGHT_NZ_KERNEL_KEY = 0b1'00'00'01'0'1'0'11;
constexpr uint32_t PP_MATMUL_F16_KERNEL_KEY = 0b1'01'01'01'0'0'0'00;
constexpr uint32_t PP_MATMUL_F16_MIX_KERNEL_KEY = 0b1'01'01'01'0'0'0'01;
constexpr uint32_t PP_MATMUL_BF16_KERNEL_KEY = 0b1'10'10'10'0'0'0'00;
constexpr uint32_t PP_MATMUL_F16_OPT_KERNEL_KEY = 0b1'01'01'01'0'1'0'00;
constexpr uint32_t PP_MATMUL_BF16_ND_NZ_ND_KERNEL_KEY = 0b1'10'10'10'0'1'0'00;
constexpr uint32_t PP_MATMUL_NZ_F16_KERNEL_KEY = 0b0'01'01'01'1'1'1'00;
constexpr uint32_t PP_MATMUL_I8_NZ_KERNEL_KEY = 0b0'00'00'01'1'1'1'11;
constexpr uint32_t PP_MATMUL_I8_NZ_COMPRESS_KERNEL_KEY = 0b0'00'00'01'1'1'1'11;
constexpr uint32_t PP_MATMUL_ACCUM_KERNEL_KEY = 0b1'10'10'11'0'0'0'00;
constexpr uint32_t PP_MATMUL_FP_ND_ND_KERNEL_KEY = 0b0'01'01'01'0'0'0'00;
constexpr uint32_t PP_MATMUL_I8_NZ_M300_KERNEL_KEY = 0b0'00'00'01'0'1'0'11;
constexpr uint32_t PP_MATMUL_I8_ND_M300_KERNEL_KEY = 0b0'00'00'01'0'0'0'11;
constexpr uint32_t PP_MATMUL_F16_NZ_M300_KERNEL_KEY = 0b0'01'01'01'0'1'0'00;

} // namespace

namespace AsdOps {
using namespace Mki;
class MatMulOperation : public OperationBase {
    using MmType = OpParam::MatMul::MatMulType;

public:
    explicit MatMulOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check IsConsistent", return nullptr);
        auto inTensorDescA = launchParam.GetInTensor(0).desc;
        auto inTensorDescB = launchParam.GetInTensor(1).desc;
        auto outTensorDescC = launchParam.GetOutTensor(0).desc;
        size_t inTensorCount = launchParam.GetInTensorCount();
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MatMul), "Invalid specificParam type.",
                  return nullptr);
        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        const TensorFormat formatA = inTensorDescA.format;
        const TensorFormat formatB = inTensorDescB.format;
        const TensorFormat formatC = outTensorDescC.format;
        const TensorDType dtypeA = inTensorDescA.dtype;
        const TensorDType dtypeB = inTensorDescB.dtype;
        const TensorDType dtypeC = outTensorDescC.dtype;
        bool isSparseDequant = (opParam.enDequant && opParam.tilingK > 0 && opParam.tilingN > 0);
        if (!isSparseDequant) {
            // Validate input tensor format, excluding PpMatMulI8NzCompressKernel.
            MKI_CHECK((formatA == TENSOR_FORMAT_ND) || ValidNzFormat(launchParam.GetInTensor(0)),
                      "Invalid format of matrix A.", return nullptr);
            MKI_CHECK((formatB == TENSOR_FORMAT_ND) || ValidNzFormat(launchParam.GetInTensor(1)),
                      "Invalid format of matrix B.", return nullptr);
        }
        PlatformType platform = PlatformInfo::Instance().GetPlatformType();
        std::unordered_map<PlatformType, uint32_t> archTypeMap = {{PlatformType::ASCEND_310P, 0},
                                                                  {PlatformType::ASCEND_910A, 0},
                                                                  {PlatformType::ASCEND_310B, 0},
                                                                  {PlatformType::ASCEND_910B, 1}};
        std::unordered_map<TensorFormat, uint32_t> tensorFormatMap = {{TENSOR_FORMAT_ND, 0},
                                                                      {TENSOR_FORMAT_FRACTAL_NZ, 1}};
        std::unordered_map<TensorDType, uint32_t> tensorDtypeMap = {
            {TENSOR_DTYPE_INT8, 0}, {TENSOR_DTYPE_FLOAT16, 1}, {TENSOR_DTYPE_BF16, 2}, {TENSOR_DTYPE_FLOAT, 3}};

        MKI_CHECK(archTypeMap.find(platform) != archTypeMap.end(), "Unsupported platform.", return nullptr);
        MKI_CHECK(tensorFormatMap.find(formatA) != tensorFormatMap.end(), "Unsupported format of matrix A.",
                  return nullptr);
        MKI_CHECK(tensorFormatMap.find(formatB) != tensorFormatMap.end(), "Unsupported format of matrix B.",
                  return nullptr);
        MKI_CHECK(tensorDtypeMap.find(dtypeA) != tensorDtypeMap.end(), "Unsupported dtype of matrix A.",
                  return nullptr);
        MKI_CHECK(tensorDtypeMap.find(dtypeB) != tensorDtypeMap.end(), "Unsupported dtype of matrix B.",
                  return nullptr);

        uint32_t kernelKey = archTypeMap[platform];                             // ArchType
        kernelKey = (kernelKey << DTYPE_BIT_COUNT) + tensorDtypeMap[dtypeA];    // DtypeA
        kernelKey = (kernelKey << DTYPE_BIT_COUNT) + tensorDtypeMap[dtypeB];    // DtypeB
        kernelKey = (kernelKey << DTYPE_BIT_COUNT) + tensorDtypeMap[dtypeC];    // DtypeC
        kernelKey = (kernelKey << FORMAT_BIT_COUNT) + tensorFormatMap[formatA]; // FormatA
        kernelKey = (kernelKey << FORMAT_BIT_COUNT) + tensorFormatMap[formatB]; // FormatB
        kernelKey = (kernelKey << FORMAT_BIT_COUNT) + tensorFormatMap[formatC]; // FormatC
        kernelKey = (kernelKey << INPUT_BIT_COUNT) + (inTensorCount - DIM_2);
        MKI_LOG(INFO) << "kernelKey: " << kernelKey;
        MKI_LOG(INFO) << ">>> PpMatmulType:" << static_cast<uint32_t>(opParam.matmulType);
        if (opParam.matmulType == MmType::MATMUL_ACCUM_ATOMIC) {
            return GetKernelByName("PpMatmulAccumAtomicKernel");
        }
        if (opParam.matmulType == MmType::MATMUL_WITH_BIAS) {
            return GetKernelByName("PpMatmulWithBiasKernel");
        }
        if (opParam.matmulType == MmType::MATMUL_EIN_SUM) {
            MKI_CHECK(!opParam.transposeA, "Unsupported transposed A_matrix", return nullptr);
            return GetKernelByName("PpMatmulEinSumKernel");
        }
        // 先判断w8a8compress
        if (isSparseDequant) {
            switch (kernelKey) {
                case PP_MATMUL_I8_NZ_COMPRESS_KERNEL_KEY: return GetKernelByName("PpMatMulI8NzCompressKernel");
                default: MKI_LOG(ERROR) << "No matched kernel for matmul operation."; return nullptr;
            }
        }

        if (opParam.quantMode == OpParam::MatMul::QuantMode::PER_TOKEN_SYMM) {
            switch (kernelKey) {
                case PP_MATMUL_I8_BF16_KERNEL_KEY:
                case PP_MATMUL_I8_FP16_KERNEL_KEY:
                case PP_MATMUL_I8_BF16_WEIGHT_NZ_KERNEL_KEY:
                case PP_MATMUL_I8_FP16_WEIGHT_NZ_KERNEL_KEY:
                    return GetKernelByName("PpMatMulI8Bf16Kernel");
                default: MKI_LOG(ERROR) << "No matched kernel for matmul operation."; return nullptr;
            }
        }

        // 判断w8a8
        switch (kernelKey) {
            case PP_MATMUL_I8_BF16_KERNEL_KEY:
            case PP_MATMUL_I8_BF16_WEIGHT_NZ_KERNEL_KEY:
                return GetKernelByName("PpMatMulI8Bf16Kernel");
            case PP_MATMUL_I8_KERNEL_KEY: return GetKernelByName("PpMatMulI8Kernel");
            case PP_MATMUL_I8_WEIGHT_NZ_KERNEL_KEY: return GetKernelByName("PpMatMulI8WeightNzKernel");
            case PP_MATMUL_F16_KERNEL_KEY: return GetKernelByName("PpMatMulF16Kernel");
            case PP_MATMUL_F16_MIX_KERNEL_KEY: return GetKernelByName("PpMatMulF16MixKernel");
            case PP_MATMUL_BF16_KERNEL_KEY: return GetKernelByName("PpMatMulBf16Kernel");
            case PP_MATMUL_F16_OPT_KERNEL_KEY: return GetKernelByName("PpMatMulF16OptKernel");
            case PP_MATMUL_BF16_ND_NZ_ND_KERNEL_KEY: return GetKernelByName("PpMatMulBf16NdNzNdKernel");
            case PP_MATMUL_NZ_F16_KERNEL_KEY: return GetKernelByName("PpMatMulNzF16Kernel");
            case PP_MATMUL_I8_NZ_KERNEL_KEY: return GetKernelByName("PpMatMulI8NzKernel");
            case PP_MATMUL_FP_ND_ND_KERNEL_KEY: return GetKernelByName("PpMatmulF16NdM300Kernel");
            case PP_MATMUL_I8_ND_M300_KERNEL_KEY: return GetKernelByName("PpMatMulI8Kernel");
            case PP_MATMUL_I8_NZ_M300_KERNEL_KEY: return GetKernelByName("PpMatMulI8NdNzKernel");
            case PP_MATMUL_F16_NZ_M300_KERNEL_KEY: return GetKernelByName("PpMatmulF16NzM300Kernel");
            default: MKI_LOG(ERROR) << "No matched kernel for matmul operation."; return nullptr;
        }
        return nullptr;
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::MatMul), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::MatMul>(specificParam);
        if (param.tilingN > 0 && param.tilingK > 0) {
            return 5; // There're 5 inputs if int8NzSparse
        }
        if (param.enDequant) {
            return 5; // There're 5 inputs if enable post dequant.
        }
        bool fuseAdd = (param.withBias || param.matmulType == MmType::MATMUL_ACCUM_ATOMIC ||
                        param.matmulType == MmType::MATMUL_WITH_BIAS);
        if (fuseAdd) {
            return 3; // 3 withBias matmul
        }
        return 2; // matmul has 2 inputs
    }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(InferShapeCheck(launchParam).Ok(), "InferShape launch param check failed.",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        auto inTensorDescA = launchParam.GetInTensor(0).desc;
        auto inTensorDescB = launchParam.GetInTensor(1).desc;
        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        auto mmType = opParam.matmulType;
        int64_t batchA = GetTensorBatchA(inTensorDescA, mmType);
        int64_t batchB = GetTensorBatchB(inTensorDescB);
        MKI_LOG(INFO) << "OpParam::MatMul::MatmulType: " << static_cast<uint32_t>(mmType);
        if (batchA > 1 && batchB > 1) {
            MKI_CHECK(batchB == batchA,
                      "batchA(" + std::to_string(batchA) + ") != batchB(" + std::to_string(batchB) + ")",
                      return AsdOps::Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        }

        TensorDesc &tensorDescOut = outTensors[0].desc;
        if (inTensorDescA.dtype == TENSOR_DTYPE_INT8) {
            tensorDescOut.dtype = opParam.outDtype;
        } else if (mmType == MmType::MATMUL_ACCUM_ATOMIC) {
            tensorDescOut.dtype = TENSOR_DTYPE_FLOAT;
        } else {
            tensorDescOut.dtype = inTensorDescA.dtype;
        }
        tensorDescOut.format = inTensorDescA.format;
        auto &outDims = tensorDescOut.dims;

        if (mmType == MmType::MATMUL_EIN_SUM) {
            return InferShapeMatmulEinSum(launchParam, outTensors);
        }

        switch (inTensorDescA.format) {
            case TENSOR_FORMAT_ND: {
                if (inTensorDescB.format == TENSOR_FORMAT_FRACTAL_NZ) {
                    return GetOutDimsWeightNz(batchA, opParam, outDims);
                }
                return GetOutDimsNd(inTensorDescA, inTensorDescB, opParam, outDims);
            }
            case TENSOR_FORMAT_FRACTAL_NZ: {
                return GetOutDimsNz(inTensorDescA, inTensorDescB, opParam, outDims);
            }
            default: return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Invalid infershape format");
        }

        // check tensor num
        return Status::FailStatus(ERROR_INFERSHAPE_ERROR);
    }

    int64_t GetTensorBatchA(const TensorDesc &tensorDesc, const MmType mmType = MmType::MATMUL_DEFAULT) const
    {
        switch (tensorDesc.format) {
            case TENSOR_FORMAT_ND: {
                // make sure dims.size() == 2 or 3
                if (tensorDesc.dims.size() == DIM_2) {
                    return DIM_1;
                }
                if (tensorDesc.dims.size() == DIM_3 && mmType == MmType::MATMUL_EIN_SUM) {
                    return tensorDesc.dims[DIM_1];
                }
                return tensorDesc.dims[DIM_0];
            }
            case TENSOR_FORMAT_FRACTAL_NZ: {
                return tensorDesc.dims[DIM_0];
            }
            default: MKI_LOG(ERROR) << "tensor format error"; return DIM_1;
        }
    }

    int64_t GetTensorBatchB(const TensorDesc &tensorDesc) const { return GetTensorBatchA(tensorDesc); }

    Status InferShapeMatmulEinSum(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        const auto &inTensorDims0 = launchParam.GetInTensor(0).desc.dims;
        const auto &inTensorDims1 = launchParam.GetInTensor(1).desc.dims;
        auto &outTensorDesc = outTensors.at(0).desc;
        auto inTensorDescA = launchParam.GetInTensor(0).desc;
        auto inTensorDescB = launchParam.GetInTensor(1).desc;
        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        outTensorDesc.dtype = launchParam.GetInTensor(0).desc.dtype;
        auto &outTensorDims = outTensorDesc.dims;
        uint64_t bSize = GetTensorBatchB(launchParam.GetInTensor(1).desc);
        uint64_t mSize = opParam.transposeA ? inTensorDims0.at(DIM_2) : inTensorDims0.at(DIM_0);
        uint64_t kSizeA = opParam.transposeA ? inTensorDims0.at(DIM_0) : inTensorDims0.at(DIM_2);
        uint64_t kSizeB = 0;
        uint64_t nSize = 0;
        if (inTensorDescB.format == TENSOR_FORMAT_FRACTAL_NZ) {
            kSizeB = opParam.transposeB ? inTensorDims1.at(DIM_1) *  inTensorDims1.at(DIM_3): inTensorDims1.at(DIM_2);
            nSize = opParam.transposeB ? inTensorDims1.at(DIM_2) : inTensorDims1.at(DIM_1) *  inTensorDims1.at(DIM_3);
        } else {
            kSizeB = opParam.transposeB ? inTensorDims1.at(DIM_2) : inTensorDims1.at(DIM_1);
            nSize = opParam.transposeB ? inTensorDims1.at(DIM_1) : inTensorDims1.at(DIM_2);
        }
        switch (inTensorDescB.format) {
            case TENSOR_FORMAT_ND: {
                MKI_CHECK(kSizeA == kSizeB, "Invalid input shape: KA != KB.",
                  return AsdOps::Status::FailStatus(ERROR_INFERSHAPE_ERROR));
                outTensorDims.clear();
                outTensorDims.emplace_back(mSize); // m
                outTensorDims.emplace_back(bSize); // batch
                outTensorDims.emplace_back(nSize); // n
                return AsdOps::Status::OkStatus();
            }
            case TENSOR_FORMAT_FRACTAL_NZ: {
                outTensorDims.clear();
                outTensorDims.emplace_back(opParam.oriShape[0]); // 0: M
                outTensorDims.emplace_back(bSize);
                outTensorDims.emplace_back(opParam.oriShape[2]); // 2: N
                return Status::OkStatus();
            }
            default: return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Invalid infershape format");
        }
    }

    Status InferShapeCheck(const LaunchParam &launchParam) const
    {
        auto inTensorDescA = launchParam.GetInTensor(0).desc;
        auto inTensorDescB = launchParam.GetInTensor(1).desc;
        auto inputADim = DIM_2;
        auto batchInputADim = DIM_3;
        auto inputBDim = DIM_2;
        auto batchInputBDim = DIM_3;
        if (inTensorDescA.format == TENSOR_FORMAT_FRACTAL_NZ) {
            inputADim = NZ_DIM;
            batchInputADim = NZ_DIM;
        }
        if (inTensorDescB.format == TENSOR_FORMAT_FRACTAL_NZ) {
            inputBDim = NZ_DIM;
            batchInputBDim = NZ_DIM;
        }
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MatMul), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(inTensorDescA.dims.size() == inputADim || inTensorDescA.dims.size() == batchInputADim,
                  "Tensor0 Invalid input dims: " << inTensorDescA.dims.size(),
                  return AsdOps::Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        MKI_CHECK(opParam.oriShape.size() == DIM_3, "size of oriShape is invalid: " << opParam.oriShape.size(),
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_LOG(INFO) << ">>> inTensorDescA.dims = " << inTensorDescA.dims;
        MKI_LOG(INFO) << ">>> inTensorDescB.dims = " << inTensorDescB.dims;
        if (opParam.tilingK > 0 && opParam.tilingN > 0) {
            MKI_CHECK(inTensorDescB.dims.size() == inputBDim || inTensorDescB.dims.size() == DIM_2,
                      "Tensor1 input dims", return AsdOps::Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        } else {
            MKI_CHECK(inTensorDescB.dims.size() == inputBDim || inTensorDescB.dims.size() == batchInputBDim,
                      "Tensor1 input dims", return AsdOps::Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        }
        return Status::OkStatus();
    }

    Status GetOutDimsWeightNz(const int64_t numBatch, const OpParam::MatMul opParam, SVector<int64_t> &outDims) const
    {
        outDims.clear();
        outDims.emplace_back(numBatch);
        outDims.emplace_back(opParam.oriShape[0]); // 0: M
        outDims.emplace_back(opParam.oriShape[2]); // 2: N
        return Status::OkStatus();
    }

    Status GetOutDimsNd(const TensorDesc &inTensorDescA, const TensorDesc &inTensorDescB, const OpParam::MatMul opParam,
                        SVector<int64_t> &outDims) const
    {
        int64_t kA = 0;
        int64_t kB = 0;
        outDims.clear();
        if (inTensorDescA.dims.size() == DIM_3 && inTensorDescB.dims.size() == DIM_3) {
            outDims.emplace_back(GetTensorBatchB(inTensorDescB));
            if (opParam.transposeA) {
                kA = inTensorDescA.dims.at(DIM_1);
                outDims.emplace_back(inTensorDescA.dims[DIM_2]); // A: B,K,M
            } else {
                kA = inTensorDescA.dims.at(DIM_2);
                outDims.emplace_back(inTensorDescA.dims[DIM_1]); // A: B,M,K
            }
            if (opParam.transposeB) {
                kB = inTensorDescB.dims.at(DIM_2);
                outDims.emplace_back(inTensorDescB.dims[DIM_1]); // B: B,N,K
            } else {
                kB = inTensorDescB.dims.at(DIM_1);
                outDims.emplace_back(inTensorDescB.dims[DIM_2]); // B: B,K,N
            }
        } else {
            if (opParam.transposeA) {
                kA = inTensorDescA.dims.at(DIM_0);
                outDims.emplace_back(inTensorDescA.dims[DIM_1]); // A: K,M
            } else {
                kA = inTensorDescA.dims.at(DIM_1);
                outDims.emplace_back(inTensorDescA.dims[DIM_0]); // A: M,K
            }
            if (opParam.transposeB) {
                kB = inTensorDescB.dims.at(DIM_1);
                outDims.emplace_back(inTensorDescB.dims[DIM_0]); // B: N,K
            } else {
                kB = inTensorDescB.dims.at(DIM_0);
                outDims.emplace_back(inTensorDescB.dims[DIM_1]); // B: K,N
            }
        }
        if (kA != kB) {
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "kA != kB");
        }
        return Status::OkStatus();
    }

    Status GetOutDimsNz(const TensorDesc &inTensorDescA, const TensorDesc &inTensorDescB, const OpParam::MatMul opParam,
                        SVector<int64_t> &outDims) const
    {
        outDims.clear();                                 // Out: B,N1,M,N0
        outDims.emplace_back(inTensorDescA.dims[DIM_0]); // B
        if (opParam.transposeB) {
            if (opParam.tilingK > 0 && opParam.tilingN > 0) {
                outDims.emplace_back((opParam.oriShape[DIM_2] + FP16_ALIGN_NUM - 1) / FP16_ALIGN_NUM);
            } else {
                if (inTensorDescB.dims.size() < DIM_3) {
                    return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Tensor2 input dims");
                }
                outDims.emplace_back(inTensorDescB.dims[DIM_2] / FP16_ALIGN_NUM); // B: B,K1,N,K0
            }
        } else {
            outDims.emplace_back(inTensorDescB.dims[DIM_1]); // B: B,N1,K,N0
        }
        if (opParam.transposeA) {
            outDims.emplace_back(inTensorDescA.dims[DIM_1] * FP16_ALIGN_NUM); // A: B,M1,K,M0
        } else {
            if (inTensorDescA.dims.size() < DIM_3) {
                return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Tensor1 input dims");
            }
            outDims.emplace_back(inTensorDescA.dims[DIM_2]); // A: B,K1,M,K0
        }
        outDims.emplace_back(FP16_ALIGN_NUM);
        return Status::OkStatus();
    }

    bool ValidNzFormat(const Tensor &tensor) const
    {
        if (tensor.desc.format != TENSOR_FORMAT_FRACTAL_NZ) {
            return false;
        }

        const uint32_t kValidDimSize = 4;
        const SVector<int64_t> &dims = tensor.desc.dims;
        if (dims.size() != kValidDimSize) {
            return false;
        }

        const uint32_t kAlignBase16 = 16;
        const uint32_t kAlignBase32 = 32;
        switch (tensor.desc.dtype) {
            case TENSOR_DTYPE_INT8: return (dims.at(DIM_3) == kAlignBase32) && (dims.at(DIM_2) % kAlignBase16 == 0);
            case TENSOR_DTYPE_BF16:
            case TENSOR_DTYPE_FLOAT16: return (dims.at(DIM_3) == kAlignBase16) && (dims.at(DIM_2) % kAlignBase16 == 0);
            default: return false;
        }
    }
};
REG_OPERATION(MatMulOperation);
} // namespace AsdOps