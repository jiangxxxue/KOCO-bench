/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_util.h"

#include <sstream>
#include <cstring>
#include <securec.h>

#include "log.h"

namespace {
/// Constant to represent index
const int DIM0 = 0;
const int DIM1 = 1;
const int DIM2 = 2;
const int DIM3 = 3;
}  // namespace


namespace atb {
template <typename T, typename U>
typename std::common_type<T, U>::type CheckIntMulOverFlow(const T a, const U b)
{
    if (std::is_signed<T>::value != std::is_signed<U>::value) {
        throw std::runtime_error("Multiplication between signed and unsigned integer not supported, it's not safe");
    }
    using PromotedType = typename std::common_type<T, U>::type;
    if (a == 0 || b == 0) {
        return 0;
    }

    PromotedType pa = static_cast<PromotedType>(a);
    PromotedType pb = static_cast<PromotedType>(b);

    if constexpr (std::is_signed<PromotedType>::value) {
        const PromotedType maxVal = std::numeric_limits<PromotedType>::max();
        const PromotedType minVal = std::numeric_limits<PromotedType>::min();
        if (pa > 0 && pb > 0) {
            if (pa > maxVal / pb) {
                throw std::overflow_error("Integer overflow detected.");
            }
        } else if (pa < 0 && pb < 0) {
            if (pa < maxVal / pb) {
                throw std::overflow_error("Integer overflow detected.");
            }
        } else if (pa > 0 && pb < 0) {
            if (pa > minVal / pb) {
                throw std::overflow_error("Integer overflow detected.");
            }
        } else if (pa < minVal / pb) {
            throw std::overflow_error("Integer overflow detected.");
        }
    } else {
        const PromotedType maxVal = std::numeric_limits<PromotedType>::max();
        if (pa > maxVal / pb) {
            throw std::overflow_error("Integer overflow detected.");
        }
    }
    return pa * pb;
}

atb::SVector<int64_t> GetCopyTensorStride(atb::Dims &tensorDims)
{
    atb::SVector<int64_t> tmpStrides(tensorDims.dimNum, 1);
    if (tensorDims.dimNum > 8) {  // 8: tensor最大维度数量
        ATB_LOG(ERROR) << "Tensor's dimNum is larger than 8, `GetCopyTensorStride` failed.";
        return tmpStrides;
    }
    for (int64_t i = static_cast<int64_t>(tensorDims.dimNum) - 2; i >= 0; i--) {
        tmpStrides[i] = CheckIntMulOverFlow(tensorDims.dims[i + 1], tmpStrides[i + 1]);
    }
    return tmpStrides;
}

atb::SVector<int64_t> GetTransposeTensorStride(atb::Dims &tensorDims)
{
    atb::SVector<int64_t> tmptransposeStrides(tensorDims.dimNum, 1);
    tmptransposeStrides[tensorDims.dimNum - 1] = tensorDims.dims[tensorDims.dimNum - 1];
    if (tensorDims.dimNum == 3) {                      // 3: 维度
        tmptransposeStrides[0] = CheckIntMulOverFlow(  // 0: 第0维
            tensorDims.dims[1],
            tensorDims.dims[2]);  // 1, 2: 跳过第1维和第2维的大小
    }
    return tmptransposeStrides;
}

atb::Status CallAclCreateTensor(
    atb::Dims &viewDims, atb::Dims &storageDims, atb::Tensor &atbTensor, std::shared_ptr<AclNNTensor> aclnnTensor)
{
    aclnnTensor->tensor = aclCreateTensor(viewDims.dims,
        viewDims.dimNum,
        atbTensor.desc.dtype,
        aclnnTensor->strides.data(),
        0,
        atbTensor.desc.format,
        storageDims.dims,
        storageDims.dimNum,
        atbTensor.deviceData);
    if (aclnnTensor->tensor == nullptr) {
        return atb::ERROR_INTERNAL_ERROR;
    }
    return atb::NO_ERROR;
}

atb::Tensor SqueezeBatchSeq(atb::Tensor atbTensor)
{
    if (atbTensor.desc.shape.dimNum == DIM3) {
        atbTensor.desc.shape.dimNum = DIM2;
        atbTensor.desc.shape.dims[DIM0] =
            CheckIntMulOverFlow(atbTensor.desc.shape.dims[DIM0], atbTensor.desc.shape.dims[DIM1]);
        atbTensor.desc.shape.dims[DIM1] = atbTensor.desc.shape.dims[DIM2];
    }
    return atbTensor;
}

std::string PrintAclNNVariankPack(const AclNNVariantPack &aclnnVariantPack)
{
    std::stringstream ss;
    ss << "ATB aclnn Op Cache: AclNNVariantPack ";
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); i++) {
        const atb::TensorDesc &tensorDesc = aclnnVariantPack.aclInTensors[i]->atbTensor.desc;
        ss << "index " << i << " dtype " << tensorDesc.dtype << " format " << tensorDesc.format << " dimNum "
           << tensorDesc.shape.dimNum;
        for (uint64_t j = 0; j < std::min(tensorDesc.shape.dimNum, static_cast<uint64_t>(8));
             j++) {  // 8: tensor最大维度数量
            ss << "dim[" << j << "]=" << tensorDesc.shape.dims[j] << " ";
        }
    }
    return ss.str();
}

std::string PrintATBVariankPack(const atb::VariantPack &atbVariantPack)
{
    std::stringstream ss;
    ss << "ATB aclnn Op Cache: ATBVariantPack ";
    for (size_t i = 0; i < atbVariantPack.inTensors.size(); i++) {
        const atb::TensorDesc &tensorDesc = atbVariantPack.inTensors[i].desc;
        ss << "index " << i << " dtype " << tensorDesc.dtype << " format " << tensorDesc.format << " dimNum "
           << tensorDesc.shape.dimNum;
        for (uint64_t j = 0; j < std::min(tensorDesc.shape.dimNum, static_cast<uint64_t>(8));
             j++) {  // 8: tensor最大维度数量
            ss << "dim[" << j << "]=" << tensorDesc.shape.dims[j] << " ";
        }
    }
    return ss.str();
}

bool IsHostDataEqual(const std::shared_ptr<AclNNTensor> tensorA, const atb::Tensor &tensorB, int tensorIdx)
{
    if (tensorA->intArrayHostData.intArray != nullptr && tensorB.hostData == nullptr) {
        ATB_LOG(DEBUG) << "ATB aclnn Op Cache: tensor index " << tensorIdx
                       << " aclnnVariantPack hostData is not null but atbVariantPack hostData is";
        return false;
    }
    if (tensorA->intArrayHostData.intArray == nullptr && tensorB.hostData != nullptr) {
        ATB_LOG(DEBUG) << "ATB aclnn Op Cache: tensor index " << tensorIdx
                       << " aclnnVariantPack hostData is null but atbVariantPack hostData is not";
        return false;
    }
    if (tensorA->intArrayHostData.intArray != nullptr && tensorB.hostData != nullptr) {
        if (tensorA->intArrayHostData.dataOri.size() * 4 != tensorB.dataSize) {  // 8: int64_t in bytes
            ATB_LOG(DEBUG) << "ATB aclnn Op Cache: tensor index " << tensorIdx << " dataSize not equal";
            return false;
        }
        if (memcmp(tensorA->intArrayHostData.dataOri.data(), tensorB.hostData, tensorB.dataSize) != 0) {
            ATB_LOG(DEBUG) << "ATB aclnn Op Cache: tensor index " << tensorIdx << " hostData not equal";
            return false;
        }
    }
    return true;
}

bool IsTensorDescEqual(const atb::TensorDesc &tensorDescA, const atb::TensorDesc &tensorDescB, int tensorIdx)
{
    if (tensorDescA.dtype != tensorDescB.dtype) {
        ATB_LOG(DEBUG) << "ATB aclnn Op Cache: tensor index " << tensorIdx
                       << " dtype not equal, aclnnVariantPack dtype " << tensorDescA.dtype << " atbVariantPack dtype "
                       << tensorDescB.dtype;
        return false;
    }
    if (tensorDescA.format != tensorDescB.format) {
        ATB_LOG(DEBUG) << "ATB aclnn Op Cache: tensor index " << tensorIdx
                       << " format not equal, aclnnVariantPack format " << tensorDescA.format
                       << " atbVariantPack format " << tensorDescB.format;
        return false;
    }
    if (tensorDescA.shape.dimNum != tensorDescB.shape.dimNum || tensorDescA.shape.dimNum > 8 ||
        tensorDescA.shape.dimNum <= 0) {  // 8: tensor最大维度数量
        ATB_LOG(DEBUG) << "ATB aclnn Op Cache: tensor index " << tensorIdx
                       << " dimNum not equal, aclnnVariantPack dimNum " << tensorDescA.shape.dimNum
                       << " atbVariantPack dimNum " << tensorDescB.shape.dimNum;
        return false;
    }
    for (uint64_t j = 0; j < tensorDescA.shape.dimNum; j++) {
        if (tensorDescA.shape.dims[j] != tensorDescB.shape.dims[j]) {
            ATB_LOG(DEBUG) << "ATB aclnn Op Cache: : tensor index " << tensorIdx << " shape.dims " << j
                           << " not equal, aclnnVariantPack value " << tensorDescA.shape.dims[j]
                           << " atbVariantPack value " << tensorDescB.shape.dims[j];
            return false;
        }
    }
    return true;
}

bool AreTensorVectorsEqual(
    const atb::SVector<std::shared_ptr<AclNNTensor>> &aclnnTensors, const atb::SVector<atb::Tensor> &atbTensors)
{
    // Check the size of two vectors
    if (aclnnTensors.size() != atbTensors.size()) {
        ATB_LOG(DEBUG) << "ATB aclnn Op Cache: size not equal, aclnnVariantPack size " << aclnnTensors.size()
                       << " atbVariantPack size " << atbTensors.size();
        return false;
    }

    // Check if every tensor in each vector has consistent data type, format, shape and host data.
    for (size_t i = 0; i < aclnnTensors.size(); i++) {
        const std::shared_ptr<AclNNTensor> tensorA = aclnnTensors[i];
        const atb::Tensor &tensorB = atbTensors[i];

        if (!IsHostDataEqual(tensorA, tensorB, i)) {
            return false;
        }

        if (!IsTensorDescEqual(tensorA->atbTensor.desc, tensorB.desc, i)) {
            return false;
        }
    }

    return true;
}

bool IsAclnnRunnerVariankPackEqual(const AclNNVariantPack &aclnnVariantPack, const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << "Compare AclNNVariantPack with RunnerVariantPack:";
    ATB_LOG(INFO) << PrintAclNNVariankPack(aclnnVariantPack);
    ATB_LOG(INFO) << runnerVariantPack.ToString();
    if (!AreTensorVectorsEqual(aclnnVariantPack.aclInTensors, runnerVariantPack.inTensors)) {
        return false;
    }

    if (!AreTensorVectorsEqual(aclnnVariantPack.aclOutTensors, runnerVariantPack.outTensors)) {
        return false;
    }
    ATB_LOG(INFO) << "ATB aclnn Op Cache: TensorDesc match";
    return true;
}

bool IsAclnnAtbVariankPackEqual(const AclNNVariantPack &aclnnVariantPack, const atb::VariantPack &atbVariantPack)
{
    ATB_LOG(INFO) << PrintAclNNVariankPack(aclnnVariantPack);
    ATB_LOG(INFO) << PrintATBVariankPack(atbVariantPack);

    if (!AreTensorVectorsEqual(aclnnVariantPack.aclInTensors, atbVariantPack.inTensors)) {
        return false;
    }

    if (!AreTensorVectorsEqual(aclnnVariantPack.aclOutTensors, atbVariantPack.outTensors)) {
        return false;
    }
    ATB_LOG(INFO) << "ATB aclnn Op Cache: TensorDesc match";
    return true;
}

std::shared_ptr<AclNNTensor> CreateTensor(atb::Tensor atbTensor, int tensorIdx)
{
    std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
    aclnnTensor->needUpdateTensorDataPtr = true;
    aclnnTensor->atbTensor = atbTensor;
    aclnnTensor->tensorIdx = tensorIdx;
    aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
    CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensor);
    return aclnnTensor;
}

int ConvertTensorToSeqLengths(atb::Tensor &tensor, aclIntArray *&actualSeqLengths)
{
    static std::vector<int64_t> seqLenCache;
    size_t dataSize = tensor.dataSize / 8;  // 8: int64 size
    if (seqLenCache.size() < dataSize) {
        seqLenCache.resize(dataSize);
    }
    if (memcpy_s(seqLenCache.data(), dataSize * 8, tensor.hostData, dataSize * 8) != 0) {  // 8: int64 size
        ATB_LOG(ERROR) << "memcpy_s failed!";
        return atb::ERROR_INTERNAL_ERROR;
    }
    if (actualSeqLengths != nullptr) {
        aclDestroyIntArray(actualSeqLengths);
        actualSeqLengths = nullptr;
    }
    actualSeqLengths = aclCreateIntArray(static_cast<int64_t *>(seqLenCache.data()), dataSize);
    return atb::NO_ERROR;
}
}  // namespace atb
