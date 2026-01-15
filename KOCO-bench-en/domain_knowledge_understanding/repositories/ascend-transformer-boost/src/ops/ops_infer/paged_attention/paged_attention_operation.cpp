/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "paged_attention_operation.h"
#include <map>
#include "paged_attention_ops_runner.h"
#include "paged_attention_ops_runner_910a.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/config.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 5;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint32_t DIM_ALIGN_16_NZ = 16;
static const uint32_t MAX_BATCH_SIZE_8192 = 8192;
static const int QUANTOFFSET_BIT = 0x00001;
static const int USEQUANT_BIT = 0x00002;
static const int BATCHSTATUS_BIT = 0x00004;
static const int MASK_BIT = 0x00008;
static const int QLENS_BIT = 0x00010;
static const int RAZOROFFSET_BIT = 0x00020;
static const int LOGN_BIT = 0x00040;
static const int QKVQUANTOFFLINE_BIT = 0x00040;
static const int QKVQUANTONLINE_BIT = 0x00080;
} // namespace

namespace atb {

static bool DeviceParamCheck(const infer::PagedAttentionParam &opParam);
static bool CompressParamCheck(const infer::PagedAttentionParam &opParam);
static bool CalcParamCheck(const infer::PagedAttentionParam &opParam);
static bool QuantParamCheck(const infer::PagedAttentionParam &opParam);
static bool LogNParamCheck(const infer::PagedAttentionParam &opParam);
static bool BNSDParamCheck(const infer::PagedAttentionParam &opParam);
static bool MlaParamCheck(const infer::PagedAttentionParam &opParam);

template <> Status CreateOperation(const infer::PagedAttentionParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (opParam.headNum <= 0) {
        ATB_LOG(ERROR) << "headNum should be greater than zero!";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.kvHeadNum < 0) {
        ATB_LOG(ERROR) << "kvHeadNum should be no less than zero!";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.kvHeadNum != 0) {
        if (opParam.headNum % opParam.kvHeadNum != 0) {
            ATB_LOG(ERROR) << "headNum mod kvHeadNum should be zero";
            return ERROR_INVALID_PARAM;
        }
    }
    if (!DeviceParamCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    if (!CompressParamCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    if (!CalcParamCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    if (!QuantParamCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    if (!LogNParamCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    if (!BNSDParamCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    if (!MlaParamCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) PagedAttentionOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

bool DeviceParamCheck(const infer::PagedAttentionParam &opParam)
{
    if (!GetSingleton<Config>().Is910B()) {
        if (opParam.batchRunStatusEnable) {
            ATB_LOG(ERROR) << "dynamic batch only support Atlas 800I A2 inference product";
            return false;
        }
        if (opParam.compressType != infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_UNDEFINED) {
            ATB_LOG(ERROR) << "head compress only support Atlas 800I A2 inference product";
            return false;
        }
        if (opParam.mlaVHeadSize > 0) {
            ATB_LOG(ERROR) << "mla mode only support Atlas 800I A2 inference product";
            return false;
        }
        if (opParam.quantType != atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_UNQUANT) {
            ATB_LOG(ERROR) << "quant feature only support Atlas 800I A2 inference product";
            return false;
        }
    }
    if (GetSingleton<Config>().Is910A()) {
        if (opParam.maskType == atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_SPEC ||
            opParam.maskType == atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_MASK_FREE) {
            ATB_LOG(ERROR) << "SPEC_MASK and MASK_FREE does not support Atlas 800 training product";
            return false;
        }
        if (opParam.calcType != atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_UNDEFINED) {
            ATB_LOG(ERROR) << "SPEC feature does not support Atlas 800 training product";
            return false;
        }
        if (opParam.scaleType != atb::infer::PagedAttentionParam::ScaleType::SCALE_TYPE_TOR) {
            ATB_LOG(ERROR) << "logN feature does not support Atlas 800 training product";
            return false;
        }
        if (opParam.inputLayout != atb::infer::InputLayout::TYPE_BSND) {
            ATB_LOG(ERROR) << "BNSD feature does not support Atlas 800 training product";
            return false;
        }
    }
    return true;
}

bool CompressParamCheck(const infer::PagedAttentionParam &opParam)
{
    if (opParam.compressType >= infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_MAX ||
        opParam.compressType < infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_UNDEFINED) {
        ATB_LOG(ERROR) << "compressType should be in the range of its enum value";
        return false;
    }
    if (opParam.compressType == infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD_ROPE &&
        opParam.maskType != infer::PagedAttentionParam::MaskType::UNDEFINED) {
        ATB_LOG(ERROR) << "When compressType is set to COMPRESS_TYPE_KVHead_ROPE, maskType must be set to UNDEFINED.";
        return false;
    }
    if (opParam.compressType == infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD_ROPE &&
        opParam.batchRunStatusEnable) {
        ATB_LOG(ERROR) << "When compressType is set to COMPRESS_TYPE_KVHEAD_ROPE,"
                       << "batchRunStatusEnable must be set to false.";
        return false;
    }
    return true;
}

bool CalcParamCheck(const infer::PagedAttentionParam &opParam)
{
    if (opParam.calcType != infer::PagedAttentionParam::CalcType::CALC_TYPE_UNDEFINED) {
        if (opParam.batchRunStatusEnable) {
            ATB_LOG(ERROR) << "SPEC func does not support dynamic batch";
            return false;
        }
        if (opParam.compressType != infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_UNDEFINED) {
            ATB_LOG(ERROR) << "SPEC func only support when compressType is COMPRESS_TYPE_UNDEFINED";
            return false;
        }
        if (GetSingleton<Config>().Is910B()) {
            if (opParam.mlaVHeadSize == 0 &&
                opParam.maskType != atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM &&
                opParam.maskType != atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_SPEC) {
                ATB_LOG(ERROR) << "SPEC func only support norm mask and spec mask";
                return false;
            }
        } else {
            if (opParam.maskType != atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_SPEC &&
                opParam.maskType != atb::infer::PagedAttentionParam::MaskType::UNDEFINED &&
                opParam.maskType != atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_MASK_FREE) {
                ATB_LOG(ERROR) << "SPEC func only support no mask,spec mask or mask free";
                return false;
            }
        }
    }
    return true;
}

bool QuantParamCheck(const infer::PagedAttentionParam &opParam)
{
    if (opParam.quantType == infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION &&
        opParam.calcType != infer::PagedAttentionParam::CalcType::CALC_TYPE_UNDEFINED) {
        ATB_LOG(ERROR) << "Dequant only support when calcType is CALC_TYPE_UNDEFINED";
        return false;
    }
    if (opParam.quantType == infer::PagedAttentionParam::TYPE_QUANT_QKV_OFFLINE ||
        opParam.quantType == infer::PagedAttentionParam::TYPE_QUANT_QKV_ONLINE) {
        if (opParam.outDataType != ACL_FLOAT16 && opParam.outDataType != ACL_BF16) {
            ATB_LOG(ERROR) << "outDataType only support ACL_FLOAT16 and ACL_BF16";
            return false;
        }
        if (opParam.hasQuantOffset) {
            ATB_LOG(ERROR) << "QKVQuant only support when hasQuantOffset is False";
            return false;
        }
        if (opParam.compressType != infer::PagedAttentionParam::COMPRESS_TYPE_UNDEFINED) {
            ATB_LOG(ERROR) << "QKVQuant only support when compressType is COMPRESS_TYPE_UNDEFINED";
            return false;
        }
        if (opParam.calcType != infer::PagedAttentionParam::CalcType::CALC_TYPE_UNDEFINED) {
            ATB_LOG(ERROR) << "QKVQuant only support when calcType is CALC_TYPE_UNDEFINED";
            return false;
        }
    }
    return true;
}

bool LogNParamCheck(const infer::PagedAttentionParam &opParam)
{
    if (opParam.scaleType >= infer::PagedAttentionParam::ScaleType::SCALE_TYPE_MAX ||
        opParam.scaleType < infer::PagedAttentionParam::ScaleType::SCALE_TYPE_TOR) {
        ATB_LOG(ERROR) << "scaleType should be in the range of its enum value";
        return false;
    }
    if (opParam.scaleType == infer::PagedAttentionParam::SCALE_TYPE_LOGN) {
        if (opParam.quantType != infer::PagedAttentionParam::TYPE_QUANT_UNQUANT) {
            ATB_LOG(ERROR) << "logN func does not support quant feature";
            return false;
        }
        if (opParam.calcType != infer::PagedAttentionParam::CALC_TYPE_UNDEFINED) {
            ATB_LOG(ERROR) << "logN func does not support calcType feature";
            return false;
        }
        if (opParam.compressType != atb::infer::PagedAttentionParam::COMPRESS_TYPE_UNDEFINED) {
            ATB_LOG(ERROR) << "logN func does not support compressType feature";
            return false;
        }
    }
    return true;
}

bool BNSDParamCheck(const infer::PagedAttentionParam &opParam)
{
    if (opParam.inputLayout == infer::InputLayout::TYPE_BNSD) {
        if (opParam.calcType == atb::infer::PagedAttentionParam::CALC_TYPE_SPEC) {
            ATB_LOG(ERROR) << "BNSD feature and calcType feature cannot coexist";
            return false;
        }
        if (opParam.compressType != atb::infer::PagedAttentionParam::COMPRESS_TYPE_UNDEFINED) {
            ATB_LOG(ERROR) << "BNSD feature and compressType feature cannot coexist";
            return false;
        }
        if (opParam.quantType != atb::infer::PagedAttentionParam::TYPE_QUANT_UNQUANT) {
            ATB_LOG(ERROR) << "BNSD feature and quantType feature cannot coexist";
            return false;
        }
        if (opParam.scaleType != atb::infer::PagedAttentionParam::SCALE_TYPE_TOR) {
            ATB_LOG(ERROR) << "BNSD feature and scaleType feature cannot coexist";
            return false;
        }
    }
    return true;
}

bool MlaParamCheck(const infer::PagedAttentionParam &opParam)
{
    if (opParam.mlaVHeadSize > 0) {
        if (opParam.maskType == infer::PagedAttentionParam::MaskType::MASK_TYPE_ALIBI) {
            ATB_LOG(ERROR) << "mla mode does not support alibi mask";
            return false;
        }
        if (opParam.calcType == infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC &&
            (opParam.maskType == infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM ||
             opParam.quantType != infer::PagedAttentionParam::QuantType::TYPE_QUANT_UNQUANT ||
             opParam.batchRunStatusEnable)) {
            ATB_LOG(ERROR) << "spec does not support norm mask, quant and dynamic batch";
            return false;
        }
        if (opParam.compressType != infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_UNDEFINED) {
            ATB_LOG(ERROR) << "mla mode does not support compress mask";
            return false;
        }
        if (opParam.quantType == infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION) {
            ATB_LOG(ERROR) << "mla mode does not support dequant_fusion";
            return false;
        }
        if (opParam.scaleType != infer::PagedAttentionParam::ScaleType::SCALE_TYPE_TOR) {
            ATB_LOG(ERROR) << "mla mode does not support logN scale";
            return false;
        }
        if (opParam.inputLayout != infer::InputLayout::TYPE_BSND) {
            ATB_LOG(ERROR) << "mla mode only support BSND InputLayout";
            return false;
        }
        if (opParam.kvHeadNum != 1) {
            ATB_LOG(ERROR) << "kvHeadNum should be 1, mla mode only support MQA";
            return false;
        }
        if (opParam.mlaVHeadSize > 576) { // 576: MLA大小限制
            ATB_LOG(ERROR) << "mlaVHeadSize should be no greater than 576";
            return false;
        }
    }
    return true;
}

PagedAttentionOperation::PagedAttentionOperation(const infer::PagedAttentionParam &param)
    : OperationBase("PagedAttentionOperation"), param_(param)
{
    if (param_.mlaVHeadSize > 0) {
        std::stringstream opIrKeySs;
        opIrKeySs << "PagedAttentionOperationMla";
        if (param_.maskType != infer::PagedAttentionParam::MaskType::UNDEFINED) {
            opIrKeySs << "Mask";
        }
        if (param_.batchRunStatusEnable) {
            opIrKeySs << "Batch";
        }
        if (param_.quantType == infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_OFFLINE) {
            opIrKeySs << "QuantOffline";
        } else if (param_.quantType == infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
            opIrKeySs << "QuantOnline";
        }
        if (param_.calcType == infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC) {
            opIrKeySs << "Qlens";
        }
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(opIrKeySs.str());
    } else {
        InitOpIni();
    }

    ATB_LOG(INFO) << GetLogPrefix() << "PagedAttentionParam headNum:" << param.headNum << ", qkScale:" << param.qkScale
                  << ", kvHeadNum:" << param.kvHeadNum << ", maskType:" << param.maskType
                  << ", batchRunStatusEnable:" << param.batchRunStatusEnable << ", quantType:" << param.quantType
                  << ", outDataType:" << param.outDataType << ", hasQuantOffset:" << param.hasQuantOffset
                  << ", compressType:" << param.compressType << ", calcType:" << param.calcType
                  << ", scaleType:" << param.scaleType << ", inputLayout:" << param.inputLayout;
}

PagedAttentionOperation::~PagedAttentionOperation() {}

uint32_t PagedAttentionOperation::GetInputNum() const
{
    uint32_t inputNumBase = IN_TENSOR_NUM;
    if (param_.maskType != atb::infer::PagedAttentionParam::UNDEFINED) {
        inputNumBase += 1; // need to input mask
    }
    if (param_.batchRunStatusEnable) {
        inputNumBase += 1; // need to input batchRunStatus
    }
    if (param_.quantType == infer::PagedAttentionParam::TYPE_DEQUANT_FUSION) {
        inputNumBase += 2; // 2: kDescale, vDescale
        if (param_.hasQuantOffset) {
            inputNumBase += 2; // 2: kOffset, vOffset
        }
    }
    if (param_.calcType == infer::PagedAttentionParam::CALC_TYPE_SPEC) {
        inputNumBase += 1; // 1: qSeqLen
    }
    if (param_.compressType == infer::PagedAttentionParam::COMPRESS_TYPE_KVHEAD_ROPE) {
        inputNumBase += 1; // 1: razorOffset
    }
    bool needQKVQuant = (param_.quantType == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_OFFLINE ||
                         param_.quantType == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_ONLINE);
    if (needQKVQuant) {
        inputNumBase += 2; // 2: kDescale, vDescale
        if (param_.quantType == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_OFFLINE) {
            inputNumBase += 1; // pScale
        }
    }
    if (param_.scaleType == infer::PagedAttentionParam::SCALE_TYPE_LOGN) {
        inputNumBase += 1; // 1: logN
    }
    if (param_.mlaVHeadSize > 0) {
        inputNumBase--;
    }
    return inputNumBase;
}

uint32_t PagedAttentionOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status PagedAttentionOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                               SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    bool needQKVQuant = (param_.quantType == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_OFFLINE ||
                         param_.quantType == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_ONLINE);
    if (needQKVQuant) {
        outTensorDescs.at(0).dtype = param_.outDataType;
    }
    if (GetSingleton<Config>().Is910B()) {
        int64_t hiddenSizeValue = 0;
        if (param_.mlaVHeadSize > 0) {
            hiddenSizeValue = param_.mlaVHeadSize;
        } else {
            uint32_t hiddenSizeValuePos = inTensorDescs.at(2).shape.dimNum - 1;
            hiddenSizeValue = inTensorDescs.at(2).shape.dims[hiddenSizeValuePos]; // 2: valueTensor
        }
        uint32_t hiddenSizeOut = outTensorDescs.at(0).shape.dimNum - 1;
        outTensorDescs.at(0).shape.dims[hiddenSizeOut] = hiddenSizeValue;
    }
    return NO_ERROR;
}

Status PagedAttentionOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    Status st = NO_ERROR;
    if (param_.inputLayout == infer::InputLayout::TYPE_BNSD && GetSingleton<Config>().Is910B()) {
        st = InferShapeDimCheckBNSD910B(inTensorDescs);
    } else {
        st = InferShapeDimCheck(inTensorDescs);
    }

    if (st != NO_ERROR) {
        return st;
    }
    return MaskFreeInferShapeCheck310P(inTensorDescs);
}

Status PagedAttentionOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                               const SVector<Tensor> &outTensors) const
{
    if (inTensors.at(1).desc.shape.dimNum != 4) { // 4: 必须是四维
        ATB_LOG(ERROR) << "ErrorCode: " << ERROR_INVALID_TENSOR_DIM
                       << ". the keyCache dimNum is:" << inTensors.at(1).desc.shape.dimNum
                       << ". keyCache should be 4 dims";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    Status st = NO_ERROR;
    if (param_.inputLayout == infer::InputLayout::TYPE_BSND) {
        st = SetupDimCheck(inTensors, outTensors);
    } else if (param_.inputLayout == infer::InputLayout::TYPE_BNSD && GetSingleton<Config>().Is910B()) {
        st = SetupDimCheckBNSD910B(inTensors, outTensors);
    }

    if (st != NO_ERROR) {
        return st;
    }
    return MaskFreeSetupCheck310P(inTensors);
}

Status PagedAttentionOperation::KVCacheDimCheck310P(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(1).shape.dims[3] != DIM_ALIGN_16_NZ) { // 1: keyCache 3: last dim
        ATB_LOG(ERROR) << "lastDim of KVCache should be 16";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(2).shape.dims[3] != DIM_ALIGN_16_NZ) { // 2: valueCache 3: last dim
        ATB_LOG(ERROR) << "lastDim of KVCache should be 16";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t blockSize = inTensorDescs.at(2).shape.dims[2]; // 2: valueCache 2: blockSize
    if (blockSize != inTensorDescs.at(1).shape.dims[2]) {  // 1: keyCache 2: blockSize
        ATB_LOG(ERROR) << "blocksize of KVCache should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t kvHeadNum = param_.kvHeadNum > 0 ? param_.kvHeadNum : param_.headNum;
    // kvHeadNum is checked > 0 in CreateOperation
    int64_t headSize = inTensorDescs.at(1).shape.dims[1] * DIM_ALIGN_16_NZ / kvHeadNum;
    if (headSize % DIM_ALIGN_16_NZ != 0) {
        ATB_LOG(ERROR) << "head_size should align 16 when format of keycache is NZ";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (headSize > 256 || headSize * blockSize > 128 * 128) { // 256: 310p headSize大小限制  // 128: 大小限制
        ATB_LOG(ERROR) << "head_size of keyCache should be no greater than 256 and "
                       << "block_size * head_size should be no greater than 128 * 128";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (headSize != inTensorDescs.at(0).shape.dims[2]) { // 2: headSize
        ATB_LOG(ERROR) << "headSizes of query and keyCache should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

bool PagedAttentionOperation::IsInMLAIncompatible() const
{
    bool needQKVQuant = (param_.quantType == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_OFFLINE ||
                         param_.quantType == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_ONLINE);
    if (param_.quantType == infer::PagedAttentionParam::TYPE_DEQUANT_FUSION ||
        (param_.calcType == infer::PagedAttentionParam::CALC_TYPE_SPEC && param_.mlaVHeadSize == 0) ||
        (needQKVQuant && param_.mlaVHeadSize == 0) || param_.scaleType == infer::PagedAttentionParam::SCALE_TYPE_LOGN ||
        param_.compressType != infer::PagedAttentionParam::COMPRESS_TYPE_UNDEFINED) {
        return true;
    }
    return false;
}

bool PagedAttentionOperation::MlaBatchSizeCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t batchSize = inTensorDescs.at(3).shape.dims[0]; // 3: contextLens
    int64_t maxBatchSize = MAX_BATCH_SIZE_8192;
    if (batchSize > maxBatchSize) {
        ATB_LOG(ERROR) << "batchSize should <= " << maxBatchSize;
        return false;
    }
    return true;
}

Status PagedAttentionOperation::KVCacheDimCheck910B(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t headSize = inTensorDescs.at(1).shape.dims[3]; // 1: keyCache 3: headSize
    if (headSize != inTensorDescs.at(0).shape.dims[2]) {  // 2: headSize
        ATB_LOG(ERROR) << "headSize of keyCache and query should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t blockSize = inTensorDescs.at(1).shape.dims[1];   // 1: keyCache 1: 1st dim
    if (IsInMLAIncompatible()) {                             // 非mla情况
        if (headSize != inTensorDescs.at(2).shape.dims[3]) { // 2: valueCache dim 3: headSize
            ATB_LOG(ERROR) << "headSize of keyCache and valueCache should be same";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (headSize > 256 || headSize * blockSize > 128 * 128) { // 256: 310p headSize大小限制  // 128: 大小限制
            ATB_LOG(ERROR) << "head_size of keyCache should be no greater than 256 and "
                           << "block_size * head_size should be no greater than 128 * 128";
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else {
        if (param_.mlaVHeadSize > headSize) {
            ATB_LOG(ERROR) << "mlaVHeadSize should be no greater than headSizeK";
            return ERROR_INVALID_TENSOR_DIM;
        }
        int64_t headSizeV = param_.mlaVHeadSize > 0 ? param_.mlaVHeadSize :
                                                      inTensorDescs.at(2).shape.dims[3]; // 2: valueCache 3: headSize
        if (headSize > 576 || headSizeV > 576) { // 576: 910b headSize大小限制 // 576: headSize大小限制
            ATB_LOG(ERROR) << "head_size of keyCache and ValueCache should be no greater than 576";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (param_.mlaVHeadSize > 0 && !MlaBatchSizeCheck(inTensorDescs)) {
            return ERROR_INVALID_TENSOR_DIM;
        }
        // 特殊场景支持blocksize 256
        bool blockSize256Check =
            param_.mlaVHeadSize > 0 && blockSize == 256 && param_.kvHeadNum == 1 && // 256: blockSize
            (param_.headNum == 16 || param_.headNum == 32) && headSize == 576 &&    // 16 32: headNum 576: headSize
            headSizeV == 512 &&                                                     // 512: headSizeV
            param_.calcType != infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC;
        if (blockSize256Check) {
            return NO_ERROR;
        }
        if (((headSize > 256 || headSizeV > 256) && blockSize > 128)) { // 128: mla blockSize大小限制 256：headsize阈值
            ATB_LOG(ERROR) << "blockSize should be no greater than 128 if headSize > 256";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

Status PagedAttentionOperation::KVCacheDimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t numBlocks = inTensorDescs.at(1).shape.dims[0]; // 1: keyCache
    int64_t blockSize = inTensorDescs.at(1).shape.dims[1]; // 1: keyCache 1: blockSize
    int64_t headNum = inTensorDescs.at(1).shape.dims[2];   // 1: keyCache 2: headNum
    if (param_.mlaVHeadSize == 0) {
        if (numBlocks != inTensorDescs.at(2).shape.dims[0]) { // 2: valueCache
            ATB_LOG(ERROR) << "numBlocks should be same";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (blockSize != inTensorDescs.at(2).shape.dims[1]) { // 2: valueCache 1: 1st dim
            ATB_LOG(ERROR) << "2nd dim of KVCache should be same";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (headNum != inTensorDescs.at(2).shape.dims[2]) { // 2: valueCache 1: 2: headNum
            ATB_LOG(ERROR) << "3rd dim of KVCache should be same";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    Status st = NO_ERROR;
    if (!GetSingleton<Config>().Is910B()) {
        st = KVCacheDimCheck310P(inTensorDescs);
    } else {
        st = KVCacheDimCheck910B(inTensorDescs);
    }
    return st;
}

Status PagedAttentionOperation::InferShapeDimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    uint32_t blockTablesPos = 3; // 3: blockTables
    uint32_t contextLensPos = 4; // 4: contextLens
    if (param_.mlaVHeadSize > 0) {
        blockTablesPos--;
        contextLensPos--;
    } else if (inTensorDescs.at(2).shape.dimNum != 4) { // 2: valueCache 4: 4 dims
        ATB_LOG(ERROR) << "invalid intensor2 dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(0).shape.dimNum != 3 ||              // 0: query 3: 3 dims
        inTensorDescs.at(1).shape.dimNum != 4 ||              // 1: keyCache 4: 4 dims
        inTensorDescs.at(blockTablesPos).shape.dimNum != 2 || // 2: 2 dims
        inTensorDescs.at(contextLensPos).shape.dimNum != 1) { // 1: 1 dim
        ATB_LOG(ERROR) << "invalid intensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    int64_t numTokens = inTensorDescs.at(0).shape.dims[0];
    if (param_.batchRunStatusEnable) {
        if (numTokens > inTensorDescs.at(blockTablesPos).shape.dims[0] || // 3: blockTables
            numTokens > inTensorDescs.at(contextLensPos).shape.dims[0]) { // 4: contextLens
            ATB_LOG(ERROR) << "numTokens in q should be no greater than blockTables and contextLens"
                           << " , and should be same as output";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }

    Status st = KVCacheDimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

Status PagedAttentionOperation::InferShapeDimCheckBNSD910B(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(0).shape.dimNum != 3 || // 0: query 3: 3rd dims
        inTensorDescs.at(1).shape.dimNum != 4 || // 1: keyCache 4: 4th dims
        inTensorDescs.at(2).shape.dimNum != 4 || // 2: valueCache 4: 4th dims
        inTensorDescs.at(3).shape.dimNum != 2 || // 3: blockTables 2: 2nd dims
        inTensorDescs.at(4).shape.dimNum != 1) { // 4: contestLens 1: 1st dim
        ATB_LOG(ERROR) << "invalid intensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    int64_t headSize = inTensorDescs.at(0).shape.dims[2];  // 0: query 2: 2nd dim
    int64_t numBlocks = inTensorDescs.at(1).shape.dims[0]; // 1: keyCache 0: 0th dim
    int64_t blockSize = inTensorDescs.at(1).shape.dims[2]; // 1: keyCache 2: 2nd dim
    if (headSize != inTensorDescs.at(1).shape.dims[3] ||   // 1: keyCache 3: 3rd dim
        headSize != inTensorDescs.at(2).shape.dims[3]) {   // 2: valueCache 3: 3rd dim
        ATB_LOG(ERROR) << "headSize should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (numBlocks != inTensorDescs.at(2).shape.dims[0]) { // 2: valueCache 0: 0th dim
        ATB_LOG(ERROR) << "numBlocks should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (blockSize != inTensorDescs.at(2).shape.dims[2]) { // 2: valueCache 2: 2nd dim
        ATB_LOG(ERROR) << "blockSizes should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status PagedAttentionOperation::MaskFreeInferShapeCheck310P(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.maskType == atb::infer::PagedAttentionParam::MASK_TYPE_MASK_FREE) {
        if (GetSingleton<Config>().Is310P()) {
            if (inTensorDescs.at(5).shape.dimNum != 4) {
                ATB_LOG(ERROR) << "When maskType is mask free on Altas 300I Duo inference products, mask dim num should be 4";
                return ERROR_INVALID_TENSOR_DIM;
            }
            if (inTensorDescs.at(5).shape.dims[0] != 1 || inTensorDescs.at(5).shape.dims[1] != 8 || inTensorDescs.at(5).shape.dims[2] != 128 || inTensorDescs.at(5).shape.dims[3] != 16) {
                ATB_LOG(ERROR) << "When maskType is mask free on Altas 300I Duo inference products, mask dims should be [1,8,128,16]";
                return ERROR_INVALID_TENSOR_DIM;
            }
            size_t kBlockSize = inTensorDescs.at(1).shape.dims[2];
            size_t vBlockSize = inTensorDescs.at(2).shape.dims[2];
            if (kBlockSize != 128 || vBlockSize != 128) {
                ATB_LOG(ERROR) << "PagedAttentionOperation intensor1 and intensor2 dim2 should be 128.";
                return ERROR_INVALID_PARAM;
            }
        } else {
            ATB_LOG(ERROR) << "Only Altas 300I Duo inference products support mask free";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

Status PagedAttentionOperation::MaskFreeSetupCheck310P(const SVector<Tensor> &inTensor) const
{
    if (param_.maskType == atb::infer::PagedAttentionParam::MASK_TYPE_MASK_FREE) {
        if (GetSingleton<Config>().Is310P()) {
            if (GetSingleton<Config>().Is310P() && param_.maskType == atb::infer::PagedAttentionParam::MASK_TYPE_MASK_FREE) {
                if (inTensor.at(5).desc.shape.dimNum != 4) {
                    ATB_LOG(ERROR) << "When maskType is mask free on Altas 300I Duo inference products, mask dim num should be 4";
                    return ERROR_INVALID_TENSOR_DIM;
                }
                if (inTensor.at(5).desc.shape.dims[0] != 1 || inTensor.at(5).desc.shape.dims[1] != 8 || inTensor.at(5).desc.shape.dims[2] != 128 || inTensor.at(5).desc.shape.dims[3] != 16) {
                    ATB_LOG(ERROR) << "When maskType is mask free on Altas 300I Duo inference products, mask dims should be [1,8,128,16]";
                    return ERROR_INVALID_TENSOR_DIM;
                }
            }
            if (inTensor.at(4).desc.shape.dimNum == 1) {
                size_t batch = inTensor.at(4).desc.shape.dims[0];
                int *k_seqlen_list = static_cast<int *>(inTensor[4].hostData);
                int *q_seqlen_list = static_cast<int *>(inTensor[6].hostData);
        
                for (size_t i = 0; i < batch; i++) {
                    if (k_seqlen_list[i] < q_seqlen_list[i]) {
                        ATB_LOG(ERROR) << "PagedAttentionOperation intensor4[i] should bigger than intensor6[i].";
                        return ERROR_INVALID_PARAM;
                    }
                    if ((k_seqlen_list[i] - q_seqlen_list[i]) % 128 != 0) {
                        ATB_LOG(ERROR) << "PagedAttentionOperation (intensor4[i] - item in intensor6[i]) % 128 should be 0. ";
                        return ERROR_INVALID_PARAM;
                    }
                }
            } else {
                ATB_LOG(ERROR) << "PagedAttentionOperation k_seqlen_list dims should be 1.";
                return ERROR_INVALID_PARAM;
            }
        
            size_t kBlockSize = inTensor.at(1).desc.shape.dims[2];
            size_t vBlockSize = inTensor.at(2).desc.shape.dims[2];
            if (kBlockSize != 128 || vBlockSize != 128) {
                ATB_LOG(ERROR) << "PagedAttentionOperation intensor1 and intensor2 dim2 should be 128.";
                return ERROR_INVALID_PARAM;
            }
        } else {
            ATB_LOG(ERROR) << "Only Altas 300I Duo inference products support mask free";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

bool PagedAttentionOperation::BlockDimCheck(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    int64_t numBlocks = inTensors.at(1).desc.shape.dims[0];                            // 1: keyCache
    int64_t numHeads = inTensors.at(0).desc.shape.dims[1];                             // 1: 1st dim
    int64_t blockSize = inTensors.at(1).desc.shape.dims[1];                            // 1: keyCache 1: 1st dim
    if (param_.mlaVHeadSize == 0 && numBlocks != inTensors.at(2).desc.shape.dims[0]) { // 2: valueCache
        ATB_LOG(ERROR) << GetLogPrefix() << "numBlocks should be same";
        return false;
    }

    if (numHeads != outTensors.at(0).desc.shape.dims[1]) { // 1: 1st dim
        ATB_LOG(ERROR) << GetLogPrefix() << "numHeads should be same";
        return false;
    }
    if (param_.mlaVHeadSize == 0 && blockSize != inTensors.at(2).desc.shape.dims[1]) { // 2: valueCache 1: 1st dim
        ATB_LOG(ERROR) << GetLogPrefix() << "blockSizes should be same";
        return false;
    }
    return true;
}

bool PagedAttentionOperation::RazorDimCheck(const SVector<Tensor> &inTensors) const
{
    int64_t numBlocks = inTensors.at(1).desc.shape.dims[0]; // 1: keyCache
    int64_t blockSize = inTensors.at(1).desc.shape.dims[1]; // 1: keyCache 1: 1st dim
    uint64_t dimRazor = 2;
    uint64_t indexReverse = (param_.scaleType == infer::PagedAttentionParam::SCALE_TYPE_LOGN) ? 2 : 1;
    if (param_.compressType == infer::PagedAttentionParam::COMPRESS_TYPE_KVHEAD_ROPE) {
        if (inTensors.at(inTensors.size() - indexReverse).desc.shape.dimNum != dimRazor) {
            ATB_LOG(ERROR) << GetLogPrefix() << "invalid intensor dimNum";
            return false;
        }
        if (numBlocks != inTensors.at(inTensors.size() - indexReverse).desc.shape.dims[0]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "numBlocks should be same";
            return false;
        }
        if (blockSize != inTensors.at(inTensors.size() - indexReverse).desc.shape.dims[1]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "blockSizes should be same";
            return false;
        }
    }
    return true;
}

bool PagedAttentionOperation::LogNDimCheck(const SVector<Tensor> &inTensors) const
{
    if (param_.scaleType == infer::PagedAttentionParam::SCALE_TYPE_LOGN) {
        if (inTensors.at(inTensors.size() - 1).desc.shape.dimNum != 1) {
            ATB_LOG(ERROR) << GetLogPrefix() << "invalid logN intensor dimNum";
            return false;
        }
        uint32_t inputNumBase = IN_TENSOR_NUM;
        uint32_t batchLogNNum = inTensors.at(inTensors.size() - 1).desc.shape.dims[0];
        if (inTensors.at(inputNumBase - 1).desc.shape.dims[0] != batchLogNNum) {
            ATB_LOG(ERROR) << "intensor contextLens and intensor logn has different batch size";
            return false;
        }
        if (param_.maskType != atb::infer::PagedAttentionParam::UNDEFINED) {
            inputNumBase += 1; // need to input mask
        }
        if (param_.batchRunStatusEnable) {
            if (inTensors.at(inputNumBase).desc.shape.dims[0] != batchLogNNum) {
                ATB_LOG(ERROR) << "intensor batchRunStatus and intensor logn has different batch size";
                return false;
            }
        }
    }
    return true;
}

Status PagedAttentionOperation::SetupDimCheck(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    Status st = InferShapeCheckImpl(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    int64_t numTokens = inTensors.at(0).desc.shape.dims[0];
    if (param_.batchRunStatusEnable) {
        if (numTokens != outTensors.at(0).desc.shape.dims[0]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "numTokens of outTensor should be the same as q";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    int64_t targetHeadSize =
        param_.mlaVHeadSize > 0 ? param_.mlaVHeadSize : inTensors.at(2).desc.shape.dims[3]; // 2: valueCache 3: 3rd dim
    if (!GetSingleton<Config>().Is910B()) {
        targetHeadSize = inTensors.at(0).desc.shape.dims[2]; // 2: 2nd dim
    }
    if (targetHeadSize != outTensors.at(0).desc.shape.dims[2]) { // 2: 2nd dim
        ATB_LOG(ERROR) << "headSize of attnOut error! It should equal to " << targetHeadSize;
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (!BlockDimCheck(inTensors, outTensors)) {
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (!RazorDimCheck(inTensors)) {
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (!LogNDimCheck(inTensors)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status PagedAttentionOperation::SetupDimCheckBNSD910B(const SVector<Tensor> &inTensors,
                                                      const SVector<Tensor> &outTensors) const
{
    if (inTensors.at(0).desc.shape.dimNum != 3 || // 0: query 3: 3 dims
        inTensors.at(1).desc.shape.dimNum != 4 || // 1: keyCache 4: 4 dims
        inTensors.at(2).desc.shape.dimNum != 4 || // 2: valueCache 4: 4 dims
        inTensors.at(3).desc.shape.dimNum != 2 || // 3: blockTables 2: 2 dims
        inTensors.at(4).desc.shape.dimNum != 1) { // 4:contestLens 1: 1 dim
        ATB_LOG(ERROR) << "invalid intensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    int64_t headSize = inTensors.at(0).desc.shape.dims[2];  // 0: query 2: 2nd dim
    int64_t numBlocks = inTensors.at(1).desc.shape.dims[0]; // 1: keyCache 0: 0th dim
    int64_t blockSize = inTensors.at(1).desc.shape.dims[2]; // 1: keyCache 2: 2nd dim
    if (headSize != inTensors.at(1).desc.shape.dims[3] ||   // 1: keyCache 3: 3rd dim
        headSize != inTensors.at(2).desc.shape.dims[3] ||   // 2: valueCache 3: 3rd dim
        headSize != outTensors.at(0).desc.shape.dims[2]) {  // 3: 3rd dim 2: 2nd dim
        ATB_LOG(ERROR) << "headSizes should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (numBlocks != inTensors.at(2).desc.shape.dims[0]) { // 2: valueCache 0: 0th dim
        ATB_LOG(ERROR) << "numBlocks should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (blockSize != inTensors.at(2).desc.shape.dims[2]) { // 2: valueCache 2: 2nd dim
        ATB_LOG(ERROR) << "blockSize should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

uint32_t PagedAttentionOperation::Bools2IntQKVQuant(AttentionFlags inputB) const
{
    uint32_t ret = 0;
    ret = inputB.useQuantOffset ? (ret | QUANTOFFSET_BIT) : ret;
    ret = inputB.useQuant ? (ret | USEQUANT_BIT) : ret;
    ret = inputB.useBatchRunStatus ? (ret | BATCHSTATUS_BIT) : ret;
    ret = inputB.useMask ? (ret | MASK_BIT) : ret;
    ret = inputB.useQLens ? (ret | QLENS_BIT) : ret;
    ret = inputB.useRazorOffset ? (ret | RAZOROFFSET_BIT) : ret;
    ret = inputB.useQKVQuantOffline ? (ret | QKVQUANTOFFLINE_BIT) : ret;
    ret = inputB.useQKVQuantOnline ? (ret | QKVQUANTONLINE_BIT) : ret;
    return ret;
}

uint32_t PagedAttentionOperation::Bools2IntLogN(AttentionFlags inputB) const
{
    uint32_t ret = 0;
    ret = inputB.useQuantOffset ? (ret | QUANTOFFSET_BIT) : ret;
    ret = inputB.useQuant ? (ret | USEQUANT_BIT) : ret;
    ret = inputB.useBatchRunStatus ? (ret | BATCHSTATUS_BIT) : ret;
    ret = inputB.useMask ? (ret | MASK_BIT) : ret;
    ret = inputB.useQLens ? (ret | QLENS_BIT) : ret;
    ret = inputB.useRazorOffset ? (ret | RAZOROFFSET_BIT) : ret;
    ret = inputB.useLogN ? (ret | LOGN_BIT) : ret;
    return ret;
}

void PagedAttentionOperation::InitOpIni()
{
    AttentionFlags inputB2I;
    inputB2I.useQuantOffset = param_.hasQuantOffset;
    inputB2I.useQuant = (param_.quantType == infer::PagedAttentionParam::TYPE_DEQUANT_FUSION);
    inputB2I.useBatchRunStatus = param_.batchRunStatusEnable;
    inputB2I.useMask = (param_.maskType != atb::infer::PagedAttentionParam::UNDEFINED);
    inputB2I.useQLens = (param_.calcType == atb::infer::PagedAttentionParam::CALC_TYPE_SPEC);
    inputB2I.useRazorOffset = (param_.compressType == infer::PagedAttentionParam::COMPRESS_TYPE_KVHEAD_ROPE);
    inputB2I.useLogN = (param_.scaleType == infer::PagedAttentionParam::SCALE_TYPE_LOGN);
    inputB2I.useQKVQuantOffline = (param_.quantType == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_OFFLINE);
    inputB2I.useQKVQuantOnline = (param_.quantType == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_ONLINE);
    uint32_t caseCodeLogN = Bools2IntLogN(inputB2I);
    uint32_t caseCodeQKVQuant = Bools2IntQKVQuant(inputB2I);

    static std::map<uint32_t, std::string> opIniTableLogN = {
        {99, "PagedAttentionOperationLogN1RazorOffset1QLens0Mask0Batch0Quant1Offset1"},
        {98, "PagedAttentionOperationLogN1RazorOffset1QLens0Mask0Batch0Quant1Offset0"},
        {96, "PagedAttentionOperationLogN1RazorOffset1QLens0Mask0Batch0Quant0Offset0"},
        {88, "PagedAttentionOperationLogN1RazorOffset0QLens1Mask1Batch0Quant0Offset0"},
        {79, "PagedAttentionOperationLogN1RazorOffset0QLens0Mask1Batch1Quant1Offset1"},
        {78, "PagedAttentionOperationLogN1RazorOffset0QLens0Mask1Batch1Quant1Offset0"},
        {76, "PagedAttentionOperationLogN1RazorOffset0QLens0Mask1Batch1Quant0Offset0"},
        {75, "PagedAttentionOperationLogN1RazorOffset0QLens0Mask1Batch0Quant1Offset1"},
        {74, "PagedAttentionOperationLogN1RazorOffset0QLens0Mask1Batch0Quant1Offset0"},
        {72, "PagedAttentionOperationLogN1RazorOffset0QLens0Mask1Batch0Quant0Offset0"},
        {71, "PagedAttentionOperationLogN1RazorOffset0QLens0Mask0Batch1Quant1Offset1"},
        {70, "PagedAttentionOperationLogN1RazorOffset0QLens0Mask0Batch1Quant1Offset0"},
        {68, "PagedAttentionOperationLogN1RazorOffset0QLens0Mask0Batch1Quant0Offset0"},
        {67, "PagedAttentionOperationLogN1RazorOffset0QLens0Mask0Batch0Quant1Offset1"},
        {66, "PagedAttentionOperationLogN1RazorOffset0QLens0Mask0Batch0Quant1Offset0"},
        {64, "PagedAttentionOperationLogN1RazorOffset0QLens0Mask0Batch0Quant0Offset0"},
    };

    static std::map<uint32_t, std::string> opIniTableQKVQuant = {
        {160, "PagedAttentionOperationQKVQuantOnline1QKVQuantOffline0RazorOffset1QLens0Mask0Batch0Quant0Offset0"},
        {140, "PagedAttentionOperationQKVQuantOnline1QKVQuantOffline0RazorOffset0QLens0Mask1Batch1Quant0Offset0"},
        {136, "PagedAttentionOperationQKVQuantOnline1QKVQuantOffline0RazorOffset0QLens0Mask1Batch0Quant0Offset0"},
        {132, "PagedAttentionOperationQKVQuantOnline1QKVQuantOffline0RazorOffset0QLens0Mask0Batch1Quant0Offset0"},
        {128, "PagedAttentionOperationQKVQuantOnline1QKVQuantOffline0RazorOffset0QLens0Mask0Batch0Quant0Offset0"},
        {96, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline1RazorOffset1QLens0Mask0Batch0Quant0Offset0"},
        {76, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline1RazorOffset0QLens0Mask1Batch1Quant0Offset0"},
        {72, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline1RazorOffset0QLens0Mask1Batch0Quant0Offset0"},
        {68, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline1RazorOffset0QLens0Mask0Batch1Quant0Offset0"},
        {64, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline1RazorOffset0QLens0Mask0Batch0Quant0Offset0"},

        {35, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline0RazorOffset1QLens0Mask0Batch0Quant1Offset1"},
        {34, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline0RazorOffset1QLens0Mask0Batch0Quant1Offset0"},
        {32, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline0RazorOffset1QLens0Mask0Batch0Quant0Offset0"},
        {24, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline0RazorOffset0QLens1Mask1Batch0Quant0Offset0"},
        {15, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline0RazorOffset0QLens0Mask1Batch1Quant1Offset1"},
        {14, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline0RazorOffset0QLens0Mask1Batch1Quant1Offset0"},
        {12, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline0RazorOffset0QLens0Mask1Batch1Quant0Offset0"},
        {11, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline0RazorOffset0QLens0Mask1Batch0Quant1Offset1"},
        {10, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline0RazorOffset0QLens0Mask1Batch0Quant1Offset0"},
        {8, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline0RazorOffset0QLens0Mask1Batch0Quant0Offset0"},
        {7, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline0RazorOffset0QLens0Mask0Batch1Quant1Offset1"},
        {6, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline0RazorOffset0QLens0Mask0Batch1Quant1Offset0"},
        {4, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline0RazorOffset0QLens0Mask0Batch1Quant0Offset0"},
        {3, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline0RazorOffset0QLens0Mask0Batch0Quant1Offset1"},
        {2, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline0RazorOffset0QLens0Mask0Batch0Quant1Offset0"},
        {0, "PagedAttentionOperationQKVQuantOnline0QKVQuantOffline0RazorOffset0QLens0Mask0Batch0Quant0Offset0"},
    };
    std::map<uint32_t, std::string>::const_iterator itLogN = opIniTableLogN.find(caseCodeLogN);
    if (itLogN != opIniTableLogN.end()) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(itLogN->second);
        return;
    }

    std::map<uint32_t, std::string>::const_iterator itQKVQuant = opIniTableQKVQuant.find(caseCodeQKVQuant);
    if (itQKVQuant != opIniTableQKVQuant.end()) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(itQKVQuant->second);
        return;
    }

    if (operationIr_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "No matched param for op ini";
    }
}

std::shared_ptr<Runner> PagedAttentionOperation::CreateRunner(Context &context) const
{
    ContextBase *contextBase = dynamic_cast<ContextBase *>(&context);
    if (!contextBase) {
        ATB_LOG(DEBUG) << "context cast to contextBase failed!";
        return nullptr;
    }
    RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_PAGED_ATTENTION);
    if (!GetSingleton<Config>().Is910B()) {
        Runner *runner = pool.MallocRunner<PagedAttentionOpsRunner910A, infer::PagedAttentionParam>(param_);
        return runner ? std::shared_ptr<Runner>(runner, [&pool](Runner *runner) { pool.FreeRunner(runner); }) :
                        std::make_shared<PagedAttentionOpsRunner910A>(param_);
    }
    return std::make_shared<PagedAttentionOpsRunner>(param_);
}

nlohmann::json PagedAttentionOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb
