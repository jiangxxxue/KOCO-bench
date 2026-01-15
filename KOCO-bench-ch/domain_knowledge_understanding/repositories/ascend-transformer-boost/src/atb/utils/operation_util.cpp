/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/operation_util.h"
#include <sstream>
#include "atb/infer_op_params.h"
#include "atb/utils/log.h"
#include "atb/operation/operation_base.h"
#include "atb/utils/tensor_check.h"

static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;
static constexpr size_t DIM_4 = 4;
static constexpr size_t INDEX_2 = 2;
static constexpr int64_t BYTE_NUM = 8;

namespace atb {
int64_t OperationUtil::RoundUp(int64_t val, int64_t align)
{
    if (align == 0) {
        return -1;
    }
    if (align >= (std::numeric_limits<int64_t>::max() / align)) {
        ATB_LOG(ERROR) << "align * align Overflow!";
        return -1;
    }
    if (val >= (std::numeric_limits<int64_t>::max() - align + 1)) {
        ATB_LOG(ERROR) << "val + align -1 Overflow!";
        return -1;
    }
    return (val + align - 1) / align * align;
}

/**
 * 获取矩阵batch，若为2维，则为1；若为3维/4维，则为第0维维度；否则返回0
 * @param tensorDesc
 * @return
 */
int64_t OperationUtil::GetTensorBatch(const TensorDesc &tensorDesc, const infer::LinearParam::MatmulType matmulType)
{
    uint64_t dimNum = tensorDesc.shape.dimNum;
    if (matmulType == infer::LinearParam::MATMUL_EIN_SUM) {
        if (dimNum == DIM_3) {
            return tensorDesc.shape.dims[1];
        } else {
            ATB_LOG(ERROR) << "tensor dim num error!";
            return 0;
        }
    } else if (dimNum == DIM_2) {
        return 1;
    } else if (dimNum == DIM_3 || dimNum == DIM_4) {
        return tensorDesc.shape.dims[0];
    } else {
        ATB_LOG(ERROR) << "tensor dim num error!";
        return 0;
    }
}

/**
 * 获取A矩阵中m值
 * @param xTensorDesc
 * @param transposeA
 * @return
 */
int64_t OperationUtil::GetXTensorM(const TensorDesc &xTensorDesc, bool transposeA,
                                   const infer::LinearParam::MatmulType matmulType)
{
    if (matmulType == infer::LinearParam::MATMUL_EIN_SUM) {
        if (xTensorDesc.shape.dimNum == DIM_3) {
            return xTensorDesc.shape.dims[0];
        } else {
            ATB_LOG(ERROR) << "tensor dim num error!";
            return 0;
        }
    } else if (xTensorDesc.shape.dimNum == DIM_2) {
        return transposeA ? xTensorDesc.shape.dims[1] : xTensorDesc.shape.dims[0];
    } else if (xTensorDesc.shape.dimNum == DIM_3) {
        return transposeA ? xTensorDesc.shape.dims[DIM_2] : xTensorDesc.shape.dims[1];
    }
    return 0;
}

/**
 * 获取A矩阵中k值
 * @param xTensorDesc
 * @param transposeA
 * @return
 */
int64_t OperationUtil::GetXTensorK(const TensorDesc &xTensorDesc, bool transposeA)
{
    if (xTensorDesc.shape.dimNum == DIM_2) {
        return transposeA ? xTensorDesc.shape.dims[0] : xTensorDesc.shape.dims[1];
    } else if (xTensorDesc.shape.dimNum == DIM_3) {
        return transposeA ? xTensorDesc.shape.dims[1] : xTensorDesc.shape.dims[DIM_2];
    }
    return 0;
}

/**
 * 获取B矩阵中n值
 * @param yTensorDesc
 * @param transposeB
 * @return
 */
int64_t OperationUtil::GetYTensorN(const TensorDesc &yTensorDesc, bool transposeB)
{
    if (yTensorDesc.shape.dimNum == DIM_2) {
        return transposeB ? yTensorDesc.shape.dims[0] : yTensorDesc.shape.dims[1];
    } else if (yTensorDesc.shape.dimNum == DIM_3) {
        return transposeB ? yTensorDesc.shape.dims[1] : yTensorDesc.shape.dims[DIM_2];
    } else if (yTensorDesc.shape.dimNum == DIM_4) {
        return transposeB ? yTensorDesc.shape.dims[DIM_2] : yTensorDesc.shape.dims[1] * yTensorDesc.shape.dims[DIM_3];
    }
    return 0;
}

/**
 * 获取outTensor的m值
 * @param outTensorDesc
 * @return
 */
int64_t OperationUtil::GetOutTensorM(const TensorDesc &outTensorDesc, const infer::LinearParam::MatmulType matmulType)
{
    uint64_t dimNum = outTensorDesc.shape.dimNum;
    if (dimNum != DIM_2 && dimNum != DIM_3) {
        ATB_LOG(ERROR) << "outTensor dimNum error!";
        return 0;
    }
    if (matmulType == infer::LinearParam::MATMUL_EIN_SUM) {
        if (outTensorDesc.shape.dimNum == DIM_3) {
            return outTensorDesc.shape.dims[0];
        } else {
            ATB_LOG(ERROR) << "tensor dim num error!";
            return 0;
        }
    } else {
        return outTensorDesc.shape.dimNum == DIM_2 ? outTensorDesc.shape.dims[0] : outTensorDesc.shape.dims[1];
    }
}

/**
 * 获取outTensor的n值
 * @param outTensorDesc
 * @return
 */
int64_t OperationUtil::GetOutTensorN(const TensorDesc &outTensorDesc)
{
    uint64_t dimNum = outTensorDesc.shape.dimNum;
    if (dimNum != DIM_2 && dimNum != DIM_3) {
        ATB_LOG(ERROR) << "outTensor dimNum error!";
        return 0;
    }
    return outTensorDesc.shape.dimNum == DIM_2 ? outTensorDesc.shape.dims[1] : outTensorDesc.shape.dims[DIM_2];
}

/**
 * matmul相关算子通用InferShape
 * @param inTensorDescs
 * @param outTensorDescs
 * @param param
 * @return
 */
Status OperationUtil::MatmulInferShape(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs,
                                       MatmulCommonCheckParam param)
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).dtype = param.outDataType == ACL_DT_UNDEFINED ? outTensorDescs.at(0).dtype : param.outDataType;
    if (param.enAccum) {
        outTensorDescs.at(0).dtype = ACL_FLOAT;
    }
    int64_t m = GetXTensorM(inTensorDescs.at(0), param.transposeA);
    int64_t n = GetYTensorN(inTensorDescs.at(1), param.transposeB);
    // linear_sparse
    if (param.matmulOpEnum == LINEAR_SPARSE) {
        n = inTensorDescs.at(INDEX_2).shape.dims[inTensorDescs.at(INDEX_2).shape.dimNum - 1];
    }
    if (inTensorDescs.at(0).shape.dimNum == DIM_2) {
        outTensorDescs.at(0).shape.dims[0] = m;
        outTensorDescs.at(0).shape.dims[1] = n;
    } else if (inTensorDescs.at(0).shape.dimNum == DIM_3) {
        outTensorDescs.at(0).shape.dims[1] = m;
        outTensorDescs.at(0).shape.dims[DIM_2] = n;
    }
    return NO_ERROR;
}

static int64_t GetBiasDeqNeedFirstDim(int64_t k, int32_t quantGroupSize)
{
    if (quantGroupSize == 0) {
        return 1;
    }
    return (k % quantGroupSize == 0) ? k / quantGroupSize : k / quantGroupSize + 1;
}

/**
 * Matmul输入tensorDesc校验
 * @param inTensorDescs
 * @param logPrefix
 * @param param
 * @return
 */
bool OperationUtil::MatmulInTensorDescsCheck(const SVector<TensorDesc> &inTensorDescs, const std::string &logPrefix,
                                             MatmulCommonCheckParam param)
{
    if (param.matmulOpEnum >= MATMUL_OP_BUTT) {
        ATB_LOG(ERROR) << logPrefix << "matmulOpEnum input error!";
        return false;
    }
    size_t inTensorId = 0;
    const TensorDesc &inputTensorDesc = inTensorDescs.at(inTensorId++);
    const TensorDesc &weightTensorDesc = inTensorDescs.at(inTensorId++);
    if (!MatmulInputWeightDimNumCheck(inTensorDescs, logPrefix, param) ||
        !MatmulInputWeightShapeCheck(inTensorDescs, logPrefix, param)) {
        return false;
    }
    int64_t n = GetYTensorN(weightTensorDesc, param.transposeB);
    if (param.matmulOpEnum == LINEAR_SPARSE) {
        n = inTensorDescs.at(inTensorId).shape.dims[inTensorDescs.at(inTensorId).shape.dimNum - 1];
    }
    int64_t k = GetXTensorK(inputTensorDesc, param.transposeA);
    int64_t biasDeqNeedLastDim = param.isPerTensor ? 1 : n;
    int64_t biasDeqNeedFirstDim = GetBiasDeqNeedFirstDim(k, param.quantGroupSize);
    if (param.hasBias) {
        size_t biasTensorId = inTensorId++;
        const TensorDesc &biasTensorDesc = inTensorDescs.at(biasTensorId);
        if (inTensorDescs.at(INDEX_2).dtype != ACL_FLOAT) {
            if (!TensorCheck::IsEmptyTensor(biasTensorDesc) &&
                !LinearBiasDeqCheck(biasTensorDesc, logPrefix, biasDeqNeedLastDim, biasDeqNeedFirstDim, biasTensorId)) {
                return false;
            }
        } else {
            if (biasTensorDesc.shape.dimNum != DIM_2) {
                ATB_LOG(ERROR) << logPrefix << "bias tensor's dims should be 2";
                return false;
            }
            return biasTensorDesc.shape.dims[0] == GetTensorBatch(weightTensorDesc) &&
                   biasTensorDesc.shape.dims[1] == n;
        }
    }
    if (param.isQuant) {
        size_t deqTensorId = inTensorId++;
        const TensorDesc &deqScaleTensorDesc = inTensorDescs.at(deqTensorId);
        if (!LinearBiasDeqCheck(deqScaleTensorDesc, logPrefix, biasDeqNeedLastDim, biasDeqNeedFirstDim, deqTensorId)) {
            return false;
        }
    }
    if (param.enAccum) {
        const TensorDesc &cTensorDesc = inTensorDescs.at(inTensorId++);
        if (cTensorDesc.shape.dimNum != inputTensorDesc.shape.dimNum) {
            ATB_LOG(ERROR) << logPrefix << "the dimNum of inTensor0 and inTensor2 should be the same";
            return false;
        }
        int64_t m = GetXTensorM(inputTensorDesc, param.transposeA);
        if (cTensorDesc.shape.dimNum == DIM_2) {
            return cTensorDesc.shape.dims[0] == m && cTensorDesc.shape.dims[1] == n;
        } else {
            auto batch = inputTensorDesc.shape.dims[0];
            return cTensorDesc.shape.dims[0] == batch && cTensorDesc.shape.dims[1] == m &&
                   cTensorDesc.shape.dims[INDEX_2] == n;
        }
    }
    if (param.matmulOpEnum == LINEAR_SPARSE) {
        int64_t k1 = RoundUp(k, INT8_ALIGN);
        int64_t n1 = RoundUp(n, DEFAULT_ALIGN);
        int64_t weightDim = weightTensorDesc.shape.dims[0];
        if (weightDim > k1 * n1) {
            ATB_LOG(ERROR) << logPrefix << "weight dim should be <= " << k1 * n1 << " but get [" << weightDim << "]";
            return false;
        }
        size_t idxTensorId = inTensorId++;
        const TensorDesc &idxTensorDesc = inTensorDescs.at(idxTensorId);
        if (!LinearSparseIdxCheck(idxTensorDesc, logPrefix, param, k, n)) {
            return false;
        }
    }
    return true;
}

/**
 * 将inTensors转换成inTensorDescs
 * @param inTensors
 * @param inTensorDescs
 */
void OperationUtil::InTensorsToInTensorDescs(const SVector<Tensor> &inTensors, SVector<TensorDesc> &inTensorDescs)
{
    inTensorDescs.clear();
    for (Tensor inTensor : inTensors) {
        inTensorDescs.push_back(inTensor.desc);
    }
}

/**
 * Matmul输出矩阵校验
 * @param outTensorDesc
 * @param inTensorDescs
 * @param logPrefix
 * @param param
 * @return
 */
bool OperationUtil::MatmulOutTensorCheck(const TensorDesc &outTensorDesc, const SVector<TensorDesc> &inTensorDescs,
                                         const std::string &logPrefix, MatmulCommonCheckParam param)
{
    const TensorDesc &inputTensorDesc = inTensorDescs.at(0);
    const TensorDesc &weightTensorDesc = inTensorDescs.at(1);
    if (inputTensorDesc.shape.dimNum != outTensorDesc.shape.dimNum) {
        ATB_LOG(ERROR) << logPrefix << "outTensor dimNum [" << outTensorDesc.shape.dimNum << "] and inTensor0 dimNum ["
                       << inputTensorDesc.shape.dimNum << "] should be equal";
        return false;
    }
    int64_t xBatch = GetTensorBatch(inputTensorDesc);
    int64_t outBatch = GetTensorBatch(outTensorDesc);
    if (xBatch != outBatch) {
        ATB_LOG(ERROR) << logPrefix << "outTensor batch [" << outBatch << "] and inTensor0 batch [" << xBatch
                       << "] should be equal";
        return false;
    }
    int64_t xTensorM = GetXTensorM(inputTensorDesc, param.transposeA);
    int64_t outTensorM = GetOutTensorM(outTensorDesc);
    if (xTensorM != outTensorM) {
        ATB_LOG(ERROR) << logPrefix << "inTensor0 m [" << xTensorM << "] and outTensor m [" << outTensorM
                       << "] should be equal";
        return false;
    }
    int64_t yTensorN = GetYTensorN(weightTensorDesc, param.transposeB);
    if (param.matmulOpEnum == LINEAR_SPARSE) {
        yTensorN = inTensorDescs.at(INDEX_2).shape.dims[inTensorDescs.at(INDEX_2).shape.dimNum - 1];
    }
    int64_t outTensorN = GetOutTensorN(outTensorDesc);
    if (yTensorN != outTensorN) {
        ATB_LOG(ERROR) << logPrefix << "inTensor1 n [" << yTensorN << "] and outTensor n [" << outTensorN
                       << "] should be equal";
        return false;
    }
    return true;
}

std::string OperationUtil::VectorToString(const std::vector<int32_t> &vec)
{
    if (vec.empty()) {
        return "";
    }
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size(); i++) {
        ss << vec.at(i);
        if (i == vec.size() - 1) {
            ss << "]";
        } else {
            ss << ",";
        }
    }
    return ss.str();
}

/**
 * 获取B矩阵中k值
 * @param yTensorDesc
 * @param transposeB
 * @return
 */
int64_t OperationUtil::GetYTensorK(const TensorDesc &yTensorDesc, bool transposeB)
{
    if (yTensorDesc.shape.dimNum == DIM_2) {
        return transposeB ? yTensorDesc.shape.dims[1] : yTensorDesc.shape.dims[0];
    } else if (yTensorDesc.shape.dimNum == DIM_3) {
        return transposeB ? yTensorDesc.shape.dims[DIM_2] : yTensorDesc.shape.dims[1];
    } else if (yTensorDesc.shape.dimNum == DIM_4) {
        return transposeB ? yTensorDesc.shape.dims[1] * yTensorDesc.shape.dims[DIM_3] : yTensorDesc.shape.dims[DIM_2];
    }
    return 0;
}

/**
 * input/weight TensorDesc维度大小校验
 * @param inputTensorDesc
 * @param weightTensorDesc
 * @param matmulOpEnum
 * @param logPrefix
 * @param transposeA
 * @return
 */
bool OperationUtil::MatmulInputWeightDimNumCheck(const SVector<TensorDesc> &inTensorDescs, const std::string &logPrefix,
                                                 MatmulCommonCheckParam param)
{
    const auto &inputTensorDesc = inTensorDescs.at(0);
    const auto &weightTensorDesc = inTensorDescs.at(1);
    uint64_t xDimNum = inputTensorDesc.shape.dimNum;
    uint64_t yDimNum = weightTensorDesc.shape.dimNum;
    if (xDimNum != DIM_2 && xDimNum != DIM_3) {
        ATB_LOG(ERROR) << logPrefix << "inTensor0 dimNum should be 2 or 3, but get [" << xDimNum << "]";
        return false;
    }
    if (xDimNum == DIM_3 && param.transposeA && yDimNum == DIM_2) {
        ATB_LOG(ERROR) << logPrefix << "transposeA is true, 3 * 2 is not supported";
        return false;
    }
    if (param.enAccum || (param.hasBias && inTensorDescs.at(INDEX_2).dtype == ACL_FLOAT)) {
        return (xDimNum == DIM_3 && yDimNum == DIM_3) || yDimNum == DIM_2;
    }
    SVector<uint64_t> validDimNums = {};
    if (param.matmulOpEnum == LINEAR_SPARSE) {
        validDimNums = {1};
    } else {
        if (weightTensorDesc.format == ACL_FORMAT_ND) {
            validDimNums = {DIM_2, DIM_3};
        } else if (weightTensorDesc.format == ACL_FORMAT_FRACTAL_NZ) {
            validDimNums = {DIM_2, DIM_4};
        }
    }
    for (uint64_t dimNum : validDimNums) {
        if (yDimNum == dimNum) {
            return true;
        }
    }
    ATB_LOG(ERROR) << logPrefix << "inTensor1 dimNum support [" << validDimNums << "], but get [" << yDimNum << "]";
    return false;
}

/**
 * input/weight TensorDesc batch校验
 * @param inTensorDescs
 * @param logPrefix
 * @param param
 * @param isPass
 * @return
 */
bool CheckBatch(const SVector<TensorDesc> &inTensorDescs, const std::string &logPrefix, MatmulCommonCheckParam param,
                bool &isPass)
{
    const auto &inputTensorDesc = inTensorDescs.at(0);
    const auto &weightTensorDesc = inTensorDescs.at(1);
    if (param.enAccum || (param.hasBias && inTensorDescs.at(INDEX_2).dtype == ACL_FLOAT)) {
        if (inputTensorDesc.shape.dimNum == DIM_3 && weightTensorDesc.shape.dimNum == DIM_3) {
            auto inputTensorBatch = inputTensorDesc.shape.dims[0];
            auto weightTensorBatch = weightTensorDesc.shape.dims[0];
            if (inputTensorBatch != weightTensorBatch) {
                ATB_LOG(ERROR) << logPrefix << "inTensor0's batch " << inputTensorBatch << " should be equal to "
                               << "inTensor1's batch " << weightTensorBatch;
                isPass = false;
                return true;
            }
        }
    } else {
        int64_t weightBatch = OperationUtil::GetTensorBatch(weightTensorDesc);
        if (weightBatch != 1) {
            ATB_LOG(ERROR) << logPrefix << "inTensor1 batch [" << weightBatch << "] should be 1";
            isPass = false;
            return true;
        }
    }
    return false;
}

/**
 * input/weight TensorDesc shape校验
 * @param inTensorDescs
 * @param logPrefix
 * @param param
 * @return
 */
bool OperationUtil::MatmulInputWeightShapeCheck(const SVector<TensorDesc> &inTensorDescs, const std::string &logPrefix,
                                                MatmulCommonCheckParam param)
{
    const auto &inputTensorDesc = inTensorDescs.at(0);
    const auto &weightTensorDesc = inTensorDescs.at(1);
    if (param.matmulOpEnum == LINEAR_SPARSE) {
        return true;
    }
    // check batch
    if (!param.isMoe) {
        bool isPass;
        if (CheckBatch(inTensorDescs, logPrefix, param, isPass))
            return isPass;
    }

    // check k
    int64_t xTensorK = GetXTensorK(inputTensorDesc, param.transposeA);
    int64_t yTensorK = GetYTensorK(weightTensorDesc, param.transposeB);
    if (xTensorK != yTensorK) {
        ATB_LOG(ERROR) << logPrefix << "inTensor0 k [" << xTensorK << "] and inTensor1 k [" << yTensorK
                       << "] should be equal";
        return false;
    }
    // check 4d nz dims
    int64_t lastAlign = param.isQuant ? INT8_ALIGN : DEFAULT_ALIGN;
    if (weightTensorDesc.format == ACL_FORMAT_FRACTAL_NZ && weightTensorDesc.shape.dimNum == DIM_4) {
        int64_t penultimateDim = weightTensorDesc.shape.dims[weightTensorDesc.shape.dimNum - DIM_2];
        if (penultimateDim % DEFAULT_ALIGN != 0) {
            ATB_LOG(ERROR) << logPrefix << "inTensor1 penultimateDim should be divisible by 16, but get ["
                           << penultimateDim << "]";
            return false;
        }
        int64_t lastDim = weightTensorDesc.shape.dims[weightTensorDesc.shape.dimNum - 1];
        if (lastDim != lastAlign) {
            ATB_LOG(ERROR) << logPrefix << "inTensor1 last dim should be " << lastAlign << ", but get [ << " << lastDim
                           << "]";
            return false;
        }
    }
    return true;
}

/**
 * Linear bias/deqScale矩阵校验
 * @param biasDeqTensorDesc biasTensorDesc/deqScaleTensorDesc
 * @param logPrefix
 * @param n
 * @param inTensorId
 * @return
 */
bool OperationUtil::LinearBiasDeqCheck(const TensorDesc &biasDeqTensorDesc, const std::string &logPrefix,
                                       const int64_t needLastDim, const int64_t needFirstDim, size_t inTensorId)
{
    if (biasDeqTensorDesc.shape.dimNum != 1 && biasDeqTensorDesc.shape.dimNum != DIM_2) {
        ATB_LOG(ERROR) << logPrefix << "inTensor" << inTensorId << " dimNum should be 1 or 2, but get ["
                       << biasDeqTensorDesc.shape.dimNum << "]";
        return false;
    }
    int64_t lastDim = biasDeqTensorDesc.shape.dims[biasDeqTensorDesc.shape.dimNum - 1];
    if (lastDim != needLastDim) {
        ATB_LOG(ERROR) << logPrefix << "inTensor" << inTensorId << " last dim should be [" << needLastDim
                       << "], but get [" << lastDim << "]";
        return false;
    }
    if (biasDeqTensorDesc.shape.dimNum == DIM_2 && biasDeqTensorDesc.shape.dims[0] != needFirstDim) {
        ATB_LOG(ERROR) << logPrefix << "inTensor" << inTensorId << " first dim should be " << needFirstDim
                       << " , but get [" << biasDeqTensorDesc.shape.dims[0] << "]";
        return false;
    }
    return true;
}

/**
 * linear_sparse compressIdx矩阵校验
 * @param idxTensorDesc
 * @param logPrefix
 * @param param
 * @param k
 * @param n
 * @return
 */
bool OperationUtil::LinearSparseIdxCheck(const TensorDesc &idxTensorDesc, const std::string &logPrefix,
                                         MatmulCommonCheckParam param, int64_t k, int64_t n)
{
    if (idxTensorDesc.shape.dimNum != 1) {
        ATB_LOG(ERROR) << logPrefix << "inTensor4 dimNum should be 1 but get [" << idxTensorDesc.shape.dimNum << "]";
        return false;
    }
    if (param.tilingK == 0 || param.tilingN == 0) {
        return false;
    }
    int64_t k1 = RoundUp(k, INT8_ALIGN);
    int64_t n1 = RoundUp(n, DEFAULT_ALIGN);
    int64_t kIndex = RoundUp(k1 / INT8_ALIGN, param.tilingK) / param.tilingK;
    int64_t nIndex = RoundUp(n1 / DEFAULT_ALIGN, param.tilingN) / param.tilingN;
    int64_t x = kIndex * nIndex * BYTE_NUM;
    if (idxTensorDesc.shape.dims[0] != x) {
        ATB_LOG(ERROR) << logPrefix << "inTensor4 dim0 should be " << x << ", but get [" << idxTensorDesc.shape.dims[0]
                       << "]";
        return false;
    }
    return true;
}
} // namespace atb
