/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPERATION_UTIL_H
#define OPERATION_UTIL_H
#include <mki/utils/file_system/file_system.h>
#include <sstream>
#include "atb/infer_op_params.h"
#include "atb/utils/log.h"
#include "atb/operation/operation_base.h"

namespace atb {
constexpr uint32_t MAX_STRING_LEN = 128;

enum MatmulOpEnum : int {
    LINEAR = 0,
    LINEAR_PARALLEL,
    LINEAR_SPARSE,
    MATMUL_OP_BUTT
};

struct MatmulCommonCheckParam {
    MatmulOpEnum matmulOpEnum = MATMUL_OP_BUTT;
    bool transposeA = false;
    bool transposeB = true;
    bool hasBias = false;
    aclDataType outDataType = ACL_DT_UNDEFINED;
    bool isQuant = false;
    bool isPerTensor = false;
    int32_t quantGroupSize = 0;
    int64_t tilingK = 0;
    int64_t tilingN = 0;
    bool isMoe = false;
    bool enAccum = false;

    MatmulCommonCheckParam &operator=(const infer::LinearParam &linearParam)
    {
        this->matmulOpEnum = LINEAR;
        this->transposeA = linearParam.transposeA;
        this->transposeB = linearParam.transposeB;
        this->hasBias = linearParam.hasBias;
        this->outDataType = linearParam.outDataType;
        this->enAccum = linearParam.enAccum;
        isQuant = linearParam.outDataType != ACL_DT_UNDEFINED; // 临时方案
        return *this;
    }

    MatmulCommonCheckParam &operator=(const infer::LinearParallelParam &linearParallelParam)
    {
        this->matmulOpEnum = LINEAR_PARALLEL;
        this->transposeA = false;
        this->transposeB = linearParallelParam.transWeight;
        this->isMoe =
            (linearParallelParam.type == atb::infer::LinearParallelParam::ParallelType::ALLTOALLVC_ALL_GATHER_GMM ||
             linearParallelParam.type == atb::infer::LinearParallelParam::ParallelType::GMM_REDUCE_SCATTER_ALLTOALLVC);
        this->isQuant =
            (linearParallelParam.backend == "lcoc" || linearParallelParam.backend == "mc2") &&
            (linearParallelParam.type == atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE ||
             linearParallelParam.type == atb::infer::LinearParallelParam::ParallelType::PURE_LINEAR ||
             linearParallelParam.type == atb::infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER ||
             linearParallelParam.type == atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR ||
             linearParallelParam.type == atb::infer::LinearParallelParam::ParallelType::ALLTOALLVC_ALL_GATHER_GMM ||
             linearParallelParam.type ==
                 atb::infer::LinearParallelParam::ParallelType::GMM_REDUCE_SCATTER_ALLTOALLVC) &&
            (linearParallelParam.quantType > atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_UNDEFINED &&
             linearParallelParam.quantType < atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_MAX);
        this->hasBias = this->isQuant;
        this->isPerTensor = this->isQuant && linearParallelParam.quantType ==
                                                 atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_TENSOR;
        this->quantGroupSize = linearParallelParam.quantGroupSize;
        this->outDataType = linearParallelParam.outDataType;
        return *this;
    }

    MatmulCommonCheckParam &operator=(const infer::LinearSparseParam &linearSparseParam)
    {
        this->matmulOpEnum = LINEAR_SPARSE;
        this->transposeA = linearSparseParam.transposeA;
        this->transposeB = linearSparseParam.transposeB;
        this->hasBias = true;
        this->outDataType = ACL_FLOAT16;
        this->isQuant = true;
        this->tilingK = linearSparseParam.tilingK;
        this->tilingN = linearSparseParam.tilingN;
        return *this;
    }
};

constexpr int64_t DEFAULT_ALIGN = 16;
constexpr int64_t INT8_ALIGN = 32;

class OperationUtil {
public:
    static int64_t RoundUp(int64_t val, int64_t align);
    static int64_t
    GetTensorBatch(const TensorDesc &tensorDesc,
                   const atb::infer::LinearParam::MatmulType matmulType = atb::infer::LinearParam::MATMUL_UNDEFINED);
    static int64_t
    GetXTensorM(const TensorDesc &xTensorDesc, bool transposeA = false,
                const atb::infer::LinearParam::MatmulType matmulType = atb::infer::LinearParam::MATMUL_UNDEFINED);
    static int64_t GetXTensorK(const TensorDesc &xTensorDesc, bool transposeA = false);
    static int64_t GetYTensorK(const TensorDesc &yTensorDesc, bool transposeB = true);
    static int64_t GetYTensorN(const TensorDesc &yTensorDesc, bool transposeB = true);
    static int64_t
    GetOutTensorM(const TensorDesc &outTensorDesc,
                  const atb::infer::LinearParam::MatmulType matmulType = atb::infer::LinearParam::MATMUL_UNDEFINED);
    static int64_t GetOutTensorN(const TensorDesc &outTensorDesc);
    static Status MatmulInferShape(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs,
                                   MatmulCommonCheckParam param);
    static bool MatmulInTensorDescsCheck(const SVector<TensorDesc> &inTensorDescs, const std::string &logPrefix,
                                         MatmulCommonCheckParam param);
    static void InTensorsToInTensorDescs(const SVector<Tensor> &inTensors, SVector<TensorDesc> &inTensorDescs);
    static bool LinearBiasDeqCheck(const TensorDesc &biasDeqTensorDesc, const std::string &logPrefix,
                                   const int64_t needLastDim, const int64_t needFirstDim, size_t inTensorId);
    static bool MatmulOutTensorCheck(const TensorDesc &outTensorDesc, const SVector<TensorDesc> &inTensorDescs,
                                     const std::string &logPrefix, MatmulCommonCheckParam param);
    template <typename OpParam> static Status DistributedInitCheck(const OpParam &opParam)
    {
        if (opParam.commMode != infer::CommMode::COMM_MULTI_PROCESS &&
            opParam.commMode != infer::CommMode::COMM_MULTI_THREAD) {
            ATB_LOG(ERROR) << "commMode: " << opParam.commMode
                           << " is invalid, only support COMM_MULTI_PROCESS, COMM_MULTI_THREAD";
            return ERROR_INVALID_PARAM;
        }
        if (opParam.commDomain.size() > MAX_STRING_LEN) {
            ATB_LOG(ERROR) << " len(commDomain) is illegal ,should less than or equal to 128 ";
            return ERROR_INVALID_PARAM;
        }
        if (!opParam.rankTableFile.empty()) {
            ATB_LOG(INFO) << " if you want use ranktableFile to hcclcomminit, you should set correct"
                             " rankTableFilePath, otherwise you should set ranktableFilePath to empty";
            if (Mki::FileSystem::Exists(opParam.rankTableFile)) {
                return NO_ERROR;
            } else {
                ATB_LOG(ERROR) << " rankTableFile path: " << opParam.rankTableFile << " does not exist.";
                return ERROR_INVALID_PARAM;
            }
        }
        if (opParam.hcclComm != nullptr) { // hccl指针传入，无需校验
            ATB_LOG(INFO) << "hcclComm is not null. Skip checking";
            return NO_ERROR;
        }
        uint32_t deviceCount = 0;
        aclError ret = aclrtGetDeviceCount(&deviceCount);
        if (ret != 0) {
            ATB_LOG(ERROR) << "get device count failed";
            return ERROR_RT_FAIL;
        }
        if (opParam.rankSize <= 0 || opParam.rankSize > static_cast<int>(deviceCount)) {
            ATB_LOG(ERROR) << "rankSize [" << opParam.rankSize << "] is invalid";
            return ERROR_INVALID_PARAM;
        }
        if (opParam.rank < 0 || opParam.rank >= opParam.rankSize) {
            ATB_LOG(ERROR) << "rank [" << opParam.rank << "] should be >=0 and smaller than rankSize ["
                           << opParam.rankSize << "]";
            return ERROR_INVALID_PARAM;
        }
        if (opParam.rankRoot < 0 || opParam.rankRoot >= opParam.rankSize) {
            ATB_LOG(ERROR) << " rankRoot must be greater or equal to 0"
                              "and must be smaller than ranksize";
            return ERROR_INVALID_PARAM;
        }
        return NO_ERROR;
    }

    template <typename QSeqLenList> static bool QSeqLenCheck(const QSeqLenList &qSeqLen, int maxSeqLen = -1)
    {
        constexpr size_t maxSeqLenSize = 32;
        if (qSeqLen.size() == 0) {
            ATB_LOG(ERROR) << "qSeqLen list should not be empty!";
            return false;
        }
        if (qSeqLen.size() > maxSeqLenSize) {
            ATB_LOG(ERROR) << "qSeqLen list size should be less than 32!";
            return false;
        }
        for (auto sampleSeqLen : qSeqLen) {
            if (sampleSeqLen <= 0 || (maxSeqLen != -1 && sampleSeqLen > maxSeqLen)) {
                ATB_LOG(ERROR) << "Invalid qSeqLen: " << sampleSeqLen;
                return false;
            }
        }
        return true;
    }
    static std::string VectorToString(const std::vector<int32_t> &vec);
    template <typename... Args> static std::string ConcatInfo(const Args &...args)
    {
        std::stringstream ss;
        ((ss << args), ...);
        return ss.str();
    }

private:
    static bool MatmulInputWeightDimNumCheck(const SVector<TensorDesc> &inTensorDescs, const std::string &logPrefix,
                                             MatmulCommonCheckParam param);
    static bool MatmulInputWeightShapeCheck(const SVector<TensorDesc> &inTensorDescs, const std::string &logPrefix,
                                            MatmulCommonCheckParam param);
    static bool LinearSparseIdxCheck(const TensorDesc &idxTensorDesc, const std::string &logPrefix,
                                     MatmulCommonCheckParam param, int64_t k, int64_t n);
};
} // namespace atb
#endif
