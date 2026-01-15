/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/tensor_util.h"
#include <map>
#include <limits>
#include <sstream>
#include <fstream>
#include <string>
#include <acl/acl.h>
#include <atb/utils/log.h>
#include <mki/utils/file_system/file_system.h>
#include "atb/types.h"
#include "atb/utils.h"

namespace atb {
uint64_t TensorUtil::CalcTensorDataSize(const Mki::Tensor &tensor)
{
    return CalcTensorDataSize(tensor.desc);
}

uint64_t TensorUtil::CalcTensorDataSize(const Mki::TensorDesc &tensorDesc)
{
    if (tensorDesc.dims.size() == 0) {
        return 0;
    }
    uint64_t dataItemSize = static_cast<uint64_t>(Mki::GetTensorElementSize(tensorDesc.dtype));
    if (dataItemSize == 0) {
        ATB_LOG(ERROR) << "not support dtype:" << Mki::GetStrWithDType(tensorDesc.dtype);
        return 0;
    }
    uint64_t elementCount = 1;
    uint64_t maxVal = std::numeric_limits<uint64_t>::max();
    for (auto dim : tensorDesc.dims) {
        if (dim <= 0) {
            return 0;
        }
        if (static_cast<uint64_t>(maxVal / static_cast<uint64_t>(dim)) < elementCount) {
            ATB_LOG(ERROR) << "CalcTensorDataSize Overflow!";
            return 0;
        }
        elementCount *= static_cast<uint64_t>(dim);
    }
    if (elementCount == 0) {
        return 0;
    }
    if (std::numeric_limits<uint64_t>::max() / dataItemSize < elementCount) {
        ATB_LOG(ERROR) << "CalcTensorDataSize Overflow!";
        return 0;
    }
    return dataItemSize * elementCount;
}

std::string TensorUtil::AsdOpsTensorToString(const Mki::Tensor &tensor)
{
    std::stringstream ss;
    ss << AsdOpsTensorDescToString(tensor.desc);
#ifdef _DEBUG
    ss << ", data:" << tensor.data;
#endif
    ss << ", dataSize:" << tensor.dataSize;
    return ss.str();
}

std::string TensorUtil::AsdOpsTensorDescToString(const Mki::TensorDesc &tensorDesc)
{
    std::stringstream ss;
    ss << "dtype:" << Mki::GetStrWithDType(tensorDesc.dtype) << ", format:" << Mki::GetStrWithFormat(tensorDesc.format)
       << ", dims:[";
    for (size_t i = 0; i < tensorDesc.dims.size(); ++i) {
        if (i == 0) {
            ss << tensorDesc.dims.at(i);
        } else {
            ss << ", " << tensorDesc.dims.at(i);
        }
    }
    ss << "]";

    return ss.str();
}

bool TensorUtil::AsdOpsTensorDescEqual(const Mki::TensorDesc &tensorDescA, const Mki::TensorDesc &tensorDescB)
{
    return tensorDescA.dtype == tensorDescB.dtype && tensorDescA.format == tensorDescB.format &&
           tensorDescA.dims == tensorDescB.dims;
}

std::string TensorUtil::AsdOpsDimsToString(const Mki::SVector<int64_t> &dims)
{
    std::string str;
    for (size_t i = 0; i < dims.size(); ++i) {
        str.append(std::to_string(dims.at(i)));
        if (i != dims.size() - 1) {
            str.append(",");
        }
    }
    return str;
}

int64_t TensorUtil::AlignInt(int64_t value, int align)
{
    if (align == 0) {
        return -1;
    }
    return (value + (align - 1)) / align * align;
}

void TensorUtil::ConvertAtbTensor2OpsTensor(const Tensor &atbTensor, Mki::Tensor &opsTensor)
{
    Mki::SVector<int64_t> dims;
    for (size_t i = 0; i < atbTensor.desc.shape.dimNum; i++) {
        dims.push_back(atbTensor.desc.shape.dims[i]);
    }
    opsTensor.desc.dtype = static_cast<Mki::TensorDType>(atbTensor.desc.dtype);
    opsTensor.desc.format = static_cast<Mki::TensorFormat>(atbTensor.desc.format);
    opsTensor.desc.dims = dims;
    opsTensor.data = atbTensor.deviceData;
    opsTensor.hostData = atbTensor.hostData;
    opsTensor.dataSize = atbTensor.dataSize;
}

void TensorUtil::ConvertOpsTensor2AtbTensor(const Mki::Tensor &opsTensor, Tensor &atbTensor)
{
    atbTensor.desc.dtype = static_cast<aclDataType>(opsTensor.desc.dtype);
    atbTensor.desc.format = static_cast<aclFormat>(opsTensor.desc.format);
    atbTensor.desc.shape.dimNum = opsTensor.desc.dims.size();
    for (size_t i = 0; i < opsTensor.desc.dims.size(); i++) {
        atbTensor.desc.shape.dims[i] = opsTensor.desc.dims.at(i);
    }
    atbTensor.deviceData = opsTensor.data;
    atbTensor.hostData = opsTensor.hostData;
    atbTensor.dataSize = opsTensor.dataSize;
}

void TensorUtil::OpsTensorDesc2AtbTensorDesc(const Mki::TensorDesc &opsTensorDesc, TensorDesc &atbTensorDesc)
{
    atbTensorDesc.dtype = static_cast<aclDataType>(opsTensorDesc.dtype);
    atbTensorDesc.format = static_cast<aclFormat>(opsTensorDesc.format);
    atbTensorDesc.shape.dimNum = opsTensorDesc.dims.size();
    for (size_t i = 0; i < opsTensorDesc.dims.size(); i++) {
        atbTensorDesc.shape.dims[i] = opsTensorDesc.dims.at(i);
    }
}

void TensorUtil::AtbTensorDesc2OpsTensorDesc(const TensorDesc &atbTensorDesc, Mki::TensorDesc &opsTensorDesc)
{
    opsTensorDesc.dtype = static_cast<Mki::TensorDType>(atbTensorDesc.dtype);
    opsTensorDesc.format = static_cast<Mki::TensorFormat>(atbTensorDesc.format);
    opsTensorDesc.dims.resize(atbTensorDesc.shape.dimNum);
    for (size_t i = 0; i < atbTensorDesc.shape.dimNum; ++i) {
        opsTensorDesc.dims.at(i) = atbTensorDesc.shape.dims[i];
    }
}

void TensorUtil::OpsTensorDescs2AtbTensorDescs(const Mki::SVector<Mki::TensorDesc> &opsTensorDescs,
                                               SVector<TensorDesc> &atbTensorDescs)
{
    atbTensorDescs.resize(opsTensorDescs.size());
    for (size_t i = 0; i < atbTensorDescs.size(); i++) {
        TensorUtil::OpsTensorDesc2AtbTensorDesc(opsTensorDescs.at(i), atbTensorDescs.at(i));
    }
}

void TensorUtil::OpsTensorDescs2AtbTensorDescs(const SVector<Mki::TensorDesc> &opsTensorDescs,
                                               SVector<TensorDesc> &atbTensorDescs)
{
    atbTensorDescs.resize(opsTensorDescs.size());
    for (size_t i = 0; i < opsTensorDescs.size(); ++i) {
        OpsTensorDesc2AtbTensorDesc(opsTensorDescs.at(i), atbTensorDescs.at(i));
    }
}

void TensorUtil::AtbTensorDescs2OpsTensorDescs(const SVector<TensorDesc> &atbTensorDescs,
                                               SVector<Mki::TensorDesc> &opsTensorDescs)
{
    opsTensorDescs.resize(atbTensorDescs.size());
    for (size_t i = 0; i < atbTensorDescs.size(); ++i) {
        AtbTensorDesc2OpsTensorDesc(atbTensorDescs.at(i), opsTensorDescs.at(i));
    }
}

uint64_t TensorUtil::CalcTensorDataSize(const Tensor &tensor)
{
    return Utils::GetTensorSize(tensor.desc);
}

uint64_t TensorUtil::CalcTensorDataSize(const TensorDesc &tensorDesc)
{
    return Utils::GetTensorSize(tensorDesc);
}

std::string TensorUtil::TensorToString(const Tensor &tensor)
{
    std::stringstream ss;
    ss << TensorDescToString(tensor.desc);
#ifdef _DEBUG
    ss << ", deviceData:" << tensor.deviceData << ", hostData:" << tensor.hostData;
#endif
    ss << ", dataSize:" << tensor.dataSize;
    return ss.str();
}

std::string TensorUtil::TensorDescToString(const TensorDesc &tensorDesc)
{
    std::stringstream ss;
    ss << "dtype: " << Mki::GetStrWithDType(tensorDesc.dtype)
       << ", format: " << Mki::GetStrWithFormat(tensorDesc.format) << ", shape:[";
    for (size_t i = 0; i < tensorDesc.shape.dimNum; ++i) {
        if (i == 0) {
            ss << tensorDesc.shape.dims[i];
        } else {
            ss << ", " << tensorDesc.shape.dims[i];
        }
    }
    ss << "]";

    return ss.str();
}

bool TensorUtil::TensorShapeEqual(const Dims &shape0, const Dims &shape1)
{
    if (shape0.dimNum != shape1.dimNum) {
        return false;
    }
    for (size_t i = 0; i < shape0.dimNum; i++) {
        if (shape0.dims[i] != shape1.dims[i]) {
            return false;
        }
    }
    return true;
}

bool TensorUtil::TensorDescEqual(const TensorDesc &tensorDescA, const TensorDesc &tensorDescB)
{
    if (tensorDescA.dtype != tensorDescB.dtype || tensorDescA.format != tensorDescB.format) {
        return false;
    }
    return TensorShapeEqual(tensorDescA.shape, tensorDescB.shape);
}

std::string TensorUtil::ShapeToString(const Dims &dims)
{
    std::string str;
    for (size_t i = 0; i < dims.dimNum; ++i) {
        str.append(std::to_string(dims.dims[i]));
        if (i != dims.dimNum - 1) {
            str.append(",");
        }
    }
    return str;
}

void TensorUtil::FastCopyTensors(const SVector<Mki::Tensor> &srcTensors, SVector<Mki::Tensor> &destTensors)
{
    destTensors.resize(srcTensors.size());
    for (size_t i = 0; i < srcTensors.size(); ++i) {
        const Mki::Tensor &srcTensor = srcTensors.at(i);
        Mki::Tensor &destTensor = destTensors.at(i);
        destTensor.dataSize = srcTensor.dataSize;
        destTensor.data = srcTensor.data;
        destTensor.hostData = srcTensor.hostData;
        destTensor.desc = srcTensor.desc;
    }
}

void TensorUtil::FastCopyTensors(const SVector<Tensor> &srcTensors, SVector<Tensor> &destTensors)
{
    destTensors.resize(srcTensors.size());
    for (size_t i = 0; i < srcTensors.size(); ++i) {
        const Tensor &srcTensor = srcTensors.at(i);
        Tensor &destTensor = destTensors.at(i);
        destTensor.dataSize = srcTensor.dataSize;
        destTensor.deviceData = srcTensor.deviceData;
        destTensor.hostData = srcTensor.hostData;
        destTensor.desc = srcTensor.desc;
    }
}

void TensorUtil::FastCopyTensorsData(const SVector<Tensor> &srcTensors, SVector<Tensor> &destTensors)
{
    destTensors.resize(srcTensors.size());
    for (size_t i = 0; i < srcTensors.size(); ++i) {
        const Tensor &srcTensor = srcTensors.at(i);
        Tensor &destTensor = destTensors.at(i);
        destTensor.dataSize = srcTensor.dataSize;
        destTensor.deviceData = srcTensor.deviceData;
        destTensor.hostData = srcTensor.hostData;
    }
}

bool TensorUtil::TensorDescsEqual(const SVector<Tensor> &tensors1, const SVector<TensorDesc> &tensorDescs2)
{
    if (tensors1.size() != tensorDescs2.size()) {
        return false;
    }
    for (size_t i = 0; i < tensors1.size(); i++) {
        if (!TensorDescEqual(tensors1.at(i).desc, tensorDescs2.at(i))) {
            return false;
        }
    }
    return true;
}

bool TensorUtil::IsRunnerVariantPackEqual(const VariantPack &runnerVariantPack1,
                                          const RunnerVariantPack &runnerVariantPack2)
{
    if (runnerVariantPack1.inTensors.size() != runnerVariantPack2.inTensors.size()) {
        return false;
    }
    for (size_t i = 0; i < runnerVariantPack1.inTensors.size(); ++i) {
        if (!TensorUtil::TensorDescEqual(runnerVariantPack1.inTensors.at(i).desc,
                                         runnerVariantPack2.inTensors.at(i).desc)) {
            return false;
        }
    }

    if (runnerVariantPack1.outTensors.size() != runnerVariantPack2.outTensors.size()) {
        return false;
    }
    for (size_t i = 0; i < runnerVariantPack1.outTensors.size(); ++i) {
        if (!TensorUtil::TensorDescEqual(runnerVariantPack1.outTensors.at(i).desc,
                                         runnerVariantPack2.outTensors.at(i).desc)) {
            return false;
        }
    }
    return true;
}

bool TensorUtil::IsTensorAddrEqual(const VariantPack &runnerVariantPack1, const RunnerVariantPack &runnerVariantPack2)
{
    if (runnerVariantPack1.inTensors.size() != runnerVariantPack2.inTensors.size()) {
        return false;
    }
    for (size_t i = 0; i < runnerVariantPack1.inTensors.size(); ++i) {
        auto &tensor1 = runnerVariantPack1.inTensors.at(i);
        auto &tensor2 = runnerVariantPack2.inTensors.at(i);
        if (tensor1.deviceData != tensor2.deviceData) {
            return false;
        }
    }

    if (runnerVariantPack1.outTensors.size() != runnerVariantPack2.outTensors.size()) {
        return false;
    }
    for (size_t i = 0; i < runnerVariantPack1.outTensors.size(); ++i) {
        auto &tensor1 = runnerVariantPack1.outTensors.at(i);
        auto &tensor2 = runnerVariantPack2.outTensors.at(i);
        if (tensor1.deviceData != tensor2.deviceData) {
            return false;
        }
    }
    return true;
}

bool TensorUtil::IsRunnerVariantPackInputEqual(const RunnerVariantPack &runnerVariantPack1,
                                               const RunnerVariantPack &runnerVariantPack2)
{
    if (runnerVariantPack1.inTensors.size() != runnerVariantPack2.inTensors.size()) {
        return false;
    }
    for (size_t i = 0; i < runnerVariantPack1.inTensors.size(); ++i) {
        if (!TensorUtil::TensorDescEqual(runnerVariantPack1.inTensors.at(i).desc,
                                         runnerVariantPack2.inTensors.at(i).desc)) {
            return false;
        }
    }
    return true;
}
} // namespace atb
