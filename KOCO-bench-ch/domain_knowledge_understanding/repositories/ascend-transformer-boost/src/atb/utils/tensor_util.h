/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_TENSOR_UTIL_H
#define ATB_TENSOR_UTIL_H
#include <vector>
#include <string>
#include <map>
#include <mki/tensor.h>
#include <mki/utils/SVector/SVector.h>
#include <mki/run_info.h>
#include <mki/types.h>
#include "atb/utils/runner_variant_pack.h"

namespace atb {
class TensorUtil {
public:
    static uint64_t CalcTensorDataSize(const Mki::Tensor &tensor);
    static uint64_t CalcTensorDataSize(const Mki::TensorDesc &tensorDesc);
    static std::string AsdOpsTensorToString(const Mki::Tensor &tensor);
    static std::string AsdOpsTensorDescToString(const Mki::TensorDesc &tensorDesc);
    static bool AsdOpsTensorDescEqual(const Mki::TensorDesc &tensorDescA, const Mki::TensorDesc &tensorDescB);
    static std::string AsdOpsDimsToString(const Mki::SVector<int64_t> &dims);
    static int64_t AlignInt(int64_t value, int align);
    static void ConvertAtbTensor2OpsTensor(const Tensor &atbTensor, Mki::Tensor &opsTensor);
    static void ConvertOpsTensor2AtbTensor(const Mki::Tensor &opsTensor, Tensor &atbTensor);
    static void OpsTensorDesc2AtbTensorDesc(const Mki::TensorDesc &opsTensorDesc, TensorDesc &atbTensorDesc);
    static void AtbTensorDesc2OpsTensorDesc(const TensorDesc &atbTensorDesc, Mki::TensorDesc &opsTensorDesc);
    static void OpsTensorDescs2AtbTensorDescs(const Mki::SVector<Mki::TensorDesc> &opsTensorDescs,
                                              SVector<TensorDesc> &atbTensorDescs);
    static void OpsTensorDescs2AtbTensorDescs(const SVector<Mki::TensorDesc> &opsTensorDescs,
                                              SVector<TensorDesc> &atbTensorDescs);
    static void AtbTensorDescs2OpsTensorDescs(const SVector<TensorDesc> &atbTensorDescs,
                                              SVector<Mki::TensorDesc> &opsTensorDescs);
    // new api
    static uint64_t CalcTensorDataSize(const Tensor &tensor);
    static uint64_t CalcTensorDataSize(const TensorDesc &tensorDesc);
    static std::string TensorToString(const Tensor &tensor);
    static std::string TensorDescToString(const TensorDesc &tensorDesc);

    static bool TensorShapeEqual(const Dims &shape0, const Dims &shape1);
    static bool TensorDescEqual(const TensorDesc &tensorDescA, const TensorDesc &tensorDescB);
    static std::string ShapeToString(const Dims &dims);
    template <typename T> static void AtbSVector2OpsSVector(const SVector<T> &atbSVector, Mki::SVector<T> &opsSVector)
    {
        opsSVector.resize(atbSVector.size());
        for (size_t i = 0; i < opsSVector.size(); i++) {
            opsSVector.at(i) = atbSVector.at(i);
        }
    }
    template <typename T> static Mki::SVector<T> AtbSVector2OpsSVector(const SVector<T> &atbSVector)
    {
        Mki::SVector<T> opsSVector;
        AtbSVector2OpsSVector(atbSVector, opsSVector);
        return opsSVector;
    }
    static void FastCopyTensors(const SVector<Mki::Tensor> &srcTensors, SVector<Mki::Tensor> &destTensors);
    static void FastCopyTensors(const SVector<Tensor> &srcTensors, SVector<Tensor> &destTensors);
    static void FastCopyTensorsData(const SVector<Tensor> &srcTensors, SVector<Tensor> &destTensors);
    static bool TensorDescsEqual(const SVector<Tensor> &tensors1, const SVector<TensorDesc> &tensorDescs2);
    static bool IsRunnerVariantPackEqual(const VariantPack &runnerVariantPack1,
                                         const RunnerVariantPack &runnerVariantPack2);
    static bool IsTensorAddrEqual(const VariantPack &runnerVariantPack1, const RunnerVariantPack &runnerVariantPack2);
    static bool IsRunnerVariantPackInputEqual(const RunnerVariantPack &runnerVariantPack1,
                                              const RunnerVariantPack &runnerVariantPack2);
};
} // namespace atb
#endif