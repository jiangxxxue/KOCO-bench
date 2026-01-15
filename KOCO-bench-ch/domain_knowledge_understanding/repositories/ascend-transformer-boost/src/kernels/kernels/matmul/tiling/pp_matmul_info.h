/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_OPS_MATMUL_PP_INFO_H
#define ASCEND_OPS_MATMUL_PP_INFO_H


#include <array>
#include <iostream>
#include <map>
#include <mki/launch_param.h>
#include <mki/types.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/status/status.h>
#include "asdops/params/params.h"

namespace AsdOps {
using namespace Mki;
struct MatMulInfo {
    uint32_t batchSize{0};
    uint32_t m{0}; // 实际输入的 m
    uint32_t n{0}; // 实际输入的 n
    uint32_t k{0}; // 实际输入的 k
    TensorDType dtypeA{TENSOR_DTYPE_FLOAT16};
    TensorDType dtypeB{TENSOR_DTYPE_FLOAT16};
    TensorDType dtypeC{TENSOR_DTYPE_FLOAT16};
    TensorFormat formatA{TENSOR_FORMAT_ND};
    TensorFormat formatB{TENSOR_FORMAT_ND};
    TensorFormat formatC{TENSOR_FORMAT_ND};
    OpParam::MatMul::MatMulType mmType{OpParam::MatMul::MatMulType::MATMUL_DEFAULT};
    bool transA{0};   // false: 0, true: 1
    bool transB{0};   // false: 0, true: 1
    bool biasFlag{0}; // false: 0, true: 1
    bool isInt8{0};   // 是否shi int8融合
    float inDtype{0};
    float outDtype{0};
    OpParam::MatMul::QuantMode quantMode{OpParam::MatMul::QuantMode::PER_CHANNEL_SYMM};
    MatMulInfo() {}
    explicit MatMulInfo(const LaunchParam &launchParam)
    {
        const auto &attrs = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        const auto &inputDType = launchParam.GetInTensor(0).desc.dtype;
        const auto &inputADim = launchParam.GetInTensor(0).desc.dims;
        formatA = launchParam.GetInTensor(0).desc.format;
        formatB = launchParam.GetInTensor(1).desc.format;
        formatC = launchParam.GetOutTensor(0).desc.format;
        transA = attrs.transposeA;
        transB = attrs.transposeB;
        isInt8 = (inputDType == TENSOR_DTYPE_INT8);
        dtypeA = launchParam.GetInTensor(0).desc.dtype;
        dtypeB = launchParam.GetInTensor(1).desc.dtype;
        dtypeC = launchParam.GetOutTensor(0).desc.dtype;
        biasFlag = attrs.withBias;
        mmType = attrs.matmulType;
        quantMode = attrs.quantMode;
        if (formatA == TENSOR_FORMAT_ND && formatB == TENSOR_FORMAT_FRACTAL_NZ) {
            if (inputADim.size() == 2) { // 2: [M, K]
                batchSize = 1;
            } else if (inputADim.size() == 3) { // 3: [B, M, K]
                if (mmType == OpParam::MatMul::MatMulType::MATMUL_EIN_SUM) {
                    batchSize = static_cast<uint32_t>(inputADim[1]);
                } else {
                    batchSize = static_cast<uint32_t>(inputADim[0]);
                }
            }
            m = attrs.oriShape.at(0); // attrs.oriShape[0]: m
            k = attrs.oriShape.at(1); // attrs.oriShape[1]: k
            n = attrs.oriShape.at(2); // attrs.oriShape[2]: n
        } else {
            // NDxND
            SVector<SVector<int64_t>> oriShapeTable;
            SetShapeTable(oriShapeTable, launchParam);
            uint32_t oriShapeIdx = (static_cast<uint32_t>(transA) << 1) + static_cast<uint32_t>(transB);
            batchSize = static_cast<uint32_t>(oriShapeTable.at(oriShapeIdx).at(0));
            m = static_cast<uint32_t>(oriShapeTable.at(oriShapeIdx).at(1));
            k = static_cast<uint32_t>(oriShapeTable.at(oriShapeIdx).at(2)); // dim 2
            n = static_cast<uint32_t>(oriShapeTable.at(oriShapeIdx).at(3)); // dim 3
        }
    }

    void SetShapeTable(SVector<SVector<int64_t>> &oriShapeTable, const LaunchParam &launchParam) const
    {
        const auto &inputADim = launchParam.GetInTensor(0).desc.dims;
        const auto &inputBDim = launchParam.GetInTensor(1).desc.dims;
        if (inputBDim.size() > 2 && inputADim.size() > 2) { // Input dimsize greater than 2
            if (mmType == OpParam::MatMul::MatMulType::MATMUL_EIN_SUM) {
                oriShapeTable = {
                    // （m, b, k）* （b, k, n）
                    {inputADim.at(1), inputADim.at(0), inputADim.at(2), inputBDim.at(2)}, // NN  (m, b, k) * (b, k, n)
                    {inputADim.at(1), inputADim.at(0), inputADim.at(2), inputBDim.at(1)}, // NT  (m, b, k) * (b, n, k)
                    {inputADim.at(1), inputADim.at(2), inputADim.at(0), inputBDim.at(2)}, // TN  (k, b, m) * (b, k, n)
                    {inputADim.at(1), inputADim.at(2), inputADim.at(0), inputBDim.at(1)}  // TT  (k, b, m) * (b, n, k)
                };
            } else {
                oriShapeTable = {
                    {inputADim.at(0), inputADim.at(1), inputADim.at(2), inputBDim.at(2)}, // NN
                    {inputADim.at(0), inputADim.at(1), inputADim.at(2), inputBDim.at(1)}, // NT
                    {inputADim.at(0), inputADim.at(2), inputADim.at(1), inputBDim.at(2)}, // TN
                    {inputADim.at(0), inputADim.at(2), inputADim.at(1), inputBDim.at(1)}  // TT
                };
            }
        } else {
            oriShapeTable = {
                {1, inputADim.at(0), inputADim.at(1), inputBDim.at(1)}, // NN
                {1, inputADim.at(0), inputADim.at(1), inputBDim.at(0)}, // NT
                {1, inputADim.at(1), inputADim.at(0), inputBDim.at(1)}, // TN
                {1, inputADim.at(1), inputADim.at(0), inputBDim.at(0)}  // TT
            };
        }
    }
};

struct HardwareInfo {
    uint32_t coreNum{0};
    uint32_t l2Size{0};
    uint32_t l1Size{0};
    uint32_t l0aSize{0};
    uint32_t l0bSize{0};
    uint32_t l0cSize{0};
    uint32_t hbmBandWidth{0};
    uint32_t l2BandWidth{0};
    HardwareInfo()
    {
        auto platform = PlatformInfo::Instance();
        coreNum = platform.GetCoreNum(CoreType::CORE_TYPE_CUBE);
        l2Size = platform.GetL2Size();
        l1Size = platform.GetL1Size();
        l0aSize = platform.GetL0ASize();
        l0bSize = platform.GetL0BSize();
        l0cSize = platform.GetL0CSize();
        hbmBandWidth = 1;
        l2BandWidth = 5; // 5x faster than hbm.
    }
};

struct OpShape {
    uint32_t batchSize{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
};

struct PpMixTilingData {
    OpShape opShape;
    uint32_t mLoop{0};
    uint32_t kLoop{0};
    uint32_t nLoop{0};
    uint32_t coreLoop{0};
    uint32_t swizzlCount{0};
    uint32_t tilingKey{0};
    uint32_t blockDim{0};
    uint32_t swizzlDirect{0};
    uint32_t splitk{0};
    void SetIdentityMatrix();
    void SetBaseShape(uint32_t batchSize, uint32_t m, uint32_t k, uint32_t n);
    void SetBaseOp(uint32_t coreNum, uint32_t mBase, uint32_t nBase);
    void SetTilingKey(const MatMulInfo &mmInfo, uint32_t isSwizlDirect, uint32_t isSplitk);
    uint32_t End(const MatMulInfo &mmInfo);
};

struct PpTilingData {
    OpShape opShape{};
    uint32_t mLoop{1};
    uint32_t kLoop{1};
    uint32_t nLoop{1};
    uint32_t coreLoop{1};
    uint32_t swizzlCount{1};
    uint32_t tilingKey{0};
    uint32_t blockDim{1};
    uint32_t swizzlDirect{0};
    uint32_t splitk{0};
    uint32_t enShuffleK{0};
    uint32_t quantMode{0};

    void SetBaseShape(uint32_t batchSize, uint32_t m, uint32_t k, uint32_t n);
    void SetBaseOp(uint32_t coreNum, uint32_t mBase, uint32_t nBase, const MatMulInfo &mmInfo);
    void SetTilingKey(const MatMulInfo &mmInfo, uint32_t swizzleDirect, uint32_t enSplitK);
    uint32_t End(const MatMulInfo &mmInfo);
};
} // namespace AsdOps
#endif // ASCEND_OPS_MATMUL_PP_INFO_H