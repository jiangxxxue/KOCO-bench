/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "self_attention_operation.h"
#include <cfloat>
#include "atb/utils/config.h"
#include "self_attention_fusion_ops_runner.h"
#include "self_attention_fusion_bypass_ops_runner.h"
#include "self_attention_fusion_bypass_ops_runner_BNSD.h"
#include "self_attention_fusion_ops_runner_910a.h"
#include "self_attention_fusion_bypass_ops_runner_910a.h"
#include "self_attention_fusion_bypass_ops_runner_BNSD_910a.h"
#include "self_attention_prefix_encoder_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/operation_util.h"
#include "self_attention_encoder_fusion_ops_runner.h"
#include "self_attention_encoder_fusion_ops_runner_910a.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/singleton.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static constexpr uint32_t FUSION_IN_TENSOR_NUM = 8;
static constexpr uint32_t FUSION_BYPASS_IN_TENSOR_NUM = 6;
// query, key, value, blockTables, mask, seqlen, kvSeqLen, slopes
static constexpr uint32_t PREFIX_ENCODER_IN_TENSOR_NUM = 8;
static constexpr uint32_t PREFIX_128MASK_ENCODER_IN_TENSOR_NUM = 7;
static constexpr uint32_t FUSION_OUT_TENSOR_NUM = 1;
static constexpr uint32_t NOFUSION_OUT_TENSOR_NUM = 3;

static constexpr uint32_t MAX_HEAD_SIZE_MLA = 576;
static constexpr uint32_t MAX_HEAD_SIZE = 256;
static const int BATCH_BIT = 0x00001;
static const int SLOPES_BIT = 0x00002;
static const int MASK_BIT = 0x00004;
static const int BYPASS_BIT = 0x00008;
static const int SCALE_BIT = 0x00010;

template <> Status CreateOperation(const infer::SelfAttentionParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    ATB_LOG(INFO) << "SelfAttentionParam headNum: " << opParam.headNum << ", qScale: " << opParam.qScale
                  << ", qkScale: " << opParam.qkScale << ", maskType: " << opParam.maskType
                  << ", kvHeadNum: " << opParam.kvHeadNum << ", calcType: " << opParam.calcType
                  << ", clampType: " << opParam.clampType << ", kvcacheCfg: " << opParam.kvcacheCfg
                  << ", scaleType: " << opParam.scaleType << ", inputLayout: " << opParam.inputLayout
                  << ", mlaVHeadSize: " << opParam.mlaVHeadSize << ", windowSize: " << opParam.windowSize
                  << ", cacheType: " << opParam.cacheType << ", kernelType: " << opParam.kernelType
                  << ", quantType: " << opParam.quantType << ", outDataType: " << opParam.outDataType
                  << ", batchRunStatusEnable: " << opParam.batchRunStatusEnable << ", isTriuMask: "
                  << opParam.isTriuMask << ", clampMin: " << opParam.clampMin << ", clampMax: " << opParam.clampMax;
    if (opParam.headNum <= 0) {
        ATB_LOG(ERROR) << "headNum should be greater than zero!";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.kvHeadNum < 0) {
        ATB_LOG(ERROR) << "kvHeadNum should be no less than zero!";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.kvcacheCfg != atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS &&
        opParam.kvcacheCfg != atb::infer::SelfAttentionParam::K_CACHE_V_CACHE) {
        ATB_LOG(ERROR) << "kvcacheCfg is invalid, should be K_CACHE_V_CACHE or K_BYPASS_V_BYPASS";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.quantType == atb::infer::SelfAttentionParam::TYPE_DEQUANT_FUSION) {
        ATB_LOG(ERROR) << "quantType can not be TYPE_DEQUANT_FUSION";
        return ERROR_INVALID_PARAM;
    }
    bool needQKVQuant = (opParam.quantType == atb::infer::SelfAttentionParam::TYPE_QUANT_QKV_OFFLINE ||
                         opParam.quantType == atb::infer::SelfAttentionParam::TYPE_QUANT_QKV_ONLINE);
    if (needQKVQuant && (opParam.outDataType != ACL_FLOAT16 && opParam.outDataType != ACL_BF16)) {
        ATB_LOG(ERROR) << "outDataType only support ACL_FLOAT16 and ACL_BF16";
        return ERROR_INVALID_PARAM;
    }
    if (needQKVQuant && opParam.calcType != atb::infer::SelfAttentionParam::PA_ENCODER) {
        ATB_LOG(ERROR) << "QKVQuant only support when calcType is PA_ENCODER";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.scaleType != infer::SelfAttentionParam::SCALE_TYPE_TOR && needQKVQuant) {
        ATB_LOG(ERROR) << "QKVQuant only support when scaleType is SCALE_TYPE_TOR";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.scaleType >= infer::SelfAttentionParam::SCALE_TYPE_MAX ||
        opParam.scaleType < infer::SelfAttentionParam::SCALE_TYPE_TOR) {
        ATB_LOG(ERROR) << "scaleType should be in the range of its enum value";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.calcType == infer::SelfAttentionParam::PA_ENCODER &&
        opParam.kvcacheCfg == atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS) {
        ATB_LOG(ERROR) << "when calcType is PA_ENCODER, kvcacheCfg should not be K_BYPASS_V_BYPASS";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.calcType != infer::SelfAttentionParam::PA_ENCODER &&
        opParam.calcType != infer::SelfAttentionParam::PREFIX_ENCODER &&
        (opParam.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
         opParam.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
         opParam.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN)) {
        ATB_LOG(ERROR) << "only PA_ENCODER and PREFIX_ENCODER supports alibi compress mask";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.maskType == infer::SelfAttentionParam::MASK_TYPE_UNDEFINED && opParam.isTriuMask == 1) {
        ATB_LOG(ERROR) << "when maskType is MASK_TYPE_UNDEFINED, isTriuMask should not be 1";
        return ERROR_INVALID_PARAM;
    }
    if (!BNSDParamCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    if (!MlaParamCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    if (!SWAParamCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    if (!DeviceParamCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    if (opParam.calcType == infer::SelfAttentionParam::PREFIX_ENCODER && !PrefixEncoderParamCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) SelfAttentionOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

bool BNSDParamCheck(const infer::SelfAttentionParam &opParam)
{
    if (opParam.inputLayout == atb::infer::InputLayout::TYPE_BNSD) {
        if (opParam.scaleType != atb::infer::SelfAttentionParam::SCALE_TYPE_TOR) {
            ATB_LOG(ERROR) << "BNSD feature and scaleType feature cannot coexist";
            return false;
        }
        if (opParam.quantType != atb::infer::SelfAttentionParam::TYPE_QUANT_UNQUANT) {
            ATB_LOG(ERROR) << "BNSD feature and quantType feature cannot coexist";
            return false;
        }
        if (opParam.calcType == infer::SelfAttentionParam::PREFIX_ENCODER) {
            ATB_LOG(ERROR) << "BNSD feature does not support prefix encoder";
            return false;
        } else if (opParam.calcType == infer::SelfAttentionParam::PA_ENCODER) {
            if (opParam.mlaVHeadSize > 0) {
                ATB_LOG(ERROR) << "BNSD feature does not support mla mode";
                return false;
            }
            if (opParam.maskType != infer::SelfAttentionParam::MASK_TYPE_UNDEFINED) {
                ATB_LOG(ERROR) << "when inputLayout is TYPE_BNSD, maskType should be MASK_TYPE_UNDEFINED";
                return false;
            }
            if (!GetSingleton<Config>().Is910B()) {
                ATB_LOG(ERROR) << "PA_ENCODER BNSD only supports Atlas 800I A2 inference product";
                return false;
            }
            if (opParam.kernelType == infer::SelfAttentionParam::KERNELTYPE_HIGH_PRECISION) {
                ATB_LOG(ERROR) << "PA_ENCODER BNSD does not support KERNELTYPE_HIGH_PRECISION";
                return false;
            }
        } else {
            if (opParam.kvcacheCfg != atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS) {
                ATB_LOG(ERROR) << "when inputLayout is TYPE_BNSD, kvcacheCfg should be K_BYPASS_V_BYPASS";
                return false;
            }
            if (opParam.maskType == infer::SelfAttentionParam::MASK_TYPE_UNDEFINED) {
                ATB_LOG(ERROR) << "when inputLayout is TYPE_BNSD, maskType should not be MASK_TYPE_UNDEFINED";
                return false;
            }
        }
    }
    return true;
}

bool DeviceParamCheck(const infer::SelfAttentionParam &opParam)
{
    if (!GetSingleton<Config>().Is910B()) {
        if (opParam.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN) {
            ATB_LOG(ERROR) << "MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN only support Atlas 800I A2 inference product";
            return false;
        }
        if (opParam.clampType != infer::SelfAttentionParam::ClampType::CLAMP_TYPE_UNDEFINED) {
            ATB_LOG(ERROR) << "clamp only support Atlas 800I A2 inference product";
            return false;
        }
        if (opParam.quantType == atb::infer::SelfAttentionParam::TYPE_QUANT_QKV_OFFLINE ||
            opParam.quantType == atb::infer::SelfAttentionParam::TYPE_QUANT_QKV_ONLINE) {
            ATB_LOG(ERROR) << "QKVQuant only support Atlas 800I A2 inference product";
            return false;
        }
        if (opParam.batchRunStatusEnable) {
            ATB_LOG(ERROR) << "Dynamic batch only support Atlas 800I A2 inference product";
            return false;
        }
    }
    if (GetSingleton<Config>().Is910A()) {
        if (opParam.calcType != infer::SelfAttentionParam::PA_ENCODER) {
            ATB_LOG(ERROR) << "Atlas 800 product only supports PA ENCODER";
            return false;
        }
        if (opParam.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
            opParam.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
            opParam.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN) {
            ATB_LOG(ERROR) << "Atlas 800 product does not support alibi compress mask";
            return false;
        }
        if (opParam.windowSize > 0) {
            ATB_LOG(ERROR) << "Atlas 800 product does not support sliding window attention";
            return false;
        }
        if (opParam.scaleType == infer::SelfAttentionParam::SCALE_TYPE_LOGN) {
            ATB_LOG(ERROR) << "Atlas 800 product does not support logN";
            return false;
        }
    }
    return true;
}

bool MlaParamCheck(const infer::SelfAttentionParam &opParam)
{
    if (opParam.mlaVHeadSize > 0) {
        if (opParam.calcType != infer::SelfAttentionParam::PA_ENCODER) {
            ATB_LOG(ERROR) << "mla mode only support PA ENCODER";
            return false;
        }
        if (!GetSingleton<Config>().Is910B()) {
            ATB_LOG(ERROR) << "mla mode only support 800I A2 inference product";
            return false;
        }
        if (opParam.maskType != infer::SelfAttentionParam::MASK_TYPE_UNDEFINED &&
            opParam.maskType != infer::SelfAttentionParam::MASK_TYPE_NORM &&
            opParam.maskType != infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_NORM &&
            opParam.maskType != infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_COMPRESS) {
            ATB_LOG(ERROR) << "mla mode does not support alibi mask";
            return false;
        }
        if (opParam.scaleType != infer::SelfAttentionParam::SCALE_TYPE_TOR) {
            ATB_LOG(ERROR) << "mla mode does not support logN scale";
            return false;
        }
        if (opParam.clampType != infer::SelfAttentionParam::CLAMP_TYPE_UNDEFINED) {
            ATB_LOG(ERROR) << "mla mode does not support clamp";
            return false;
        }
        if (opParam.mlaVHeadSize > MAX_HEAD_SIZE_MLA) {
            ATB_LOG(ERROR) << "mlaVHeadSize should be no greater than 576";
            return false;
        }
    }
    return true;
}

bool SWAParamCheck(const infer::SelfAttentionParam &opParam)
{
    if (opParam.windowSize == 0 &&
        (opParam.maskType != infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_NORM &&
         opParam.maskType != infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_COMPRESS)) { // 不开启swa
        if (opParam.cacheType == infer::SelfAttentionParam::CACHE_TYPE_SWA) {
            ATB_LOG(ERROR) << "cacheType should not be CACHE_TYPE_SWA if Sliding Window Attention is not used";
            return false;
        }
    } else if (opParam.windowSize > 0 &&
               (opParam.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_NORM ||
                opParam.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_COMPRESS)) { // 开启swa
        if (opParam.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_COMPRESS &&
            opParam.calcType == infer::SelfAttentionParam::DECODER) {
            ATB_LOG(ERROR) << "maskType should not be MASK_TYPE_SLIDING_WINDOW_COMPRESS for decoder";
            return false;
        }
        if (opParam.batchRunStatusEnable || opParam.kernelType != infer::SelfAttentionParam::KERNELTYPE_DEFAULT ||
            opParam.clampType != infer::SelfAttentionParam::CLAMP_TYPE_UNDEFINED ||
            opParam.quantType != infer::SelfAttentionParam::TYPE_QUANT_UNQUANT ||
            opParam.scaleType != infer::SelfAttentionParam::SCALE_TYPE_TOR || opParam.inputLayout != infer::TYPE_BSND) {
            ATB_LOG(ERROR) << "Sliding Window Attention does not support dynamic batch, high precision kernel, "
                              "clamp, qkvquant, logN func and BNSD feature";
            return false;
        }
    } else {
        ATB_LOG(ERROR) << "windowSize should greater than 0 and maskType should be MASK_TYPE_SLIDING_WINDOW_NORM "
                          "or MASK_TYPE_SLIDING_WINDOW_COMPRESS if Sliding Window Attention is used";
        return false;
    }
    return true;
}

bool PrefixEncoderParamCheck(const infer::SelfAttentionParam &opParam)
{
    if (opParam.kvHeadNum > 0 && opParam.headNum % opParam.kvHeadNum != 0) {
        ATB_LOG(ERROR) << "PREFIX_ENCODER expects headNum is divisible by kvHeadNum";
        return false;
    }
    std::vector<std::pair<bool, const char *>> paramCheckList = {
        {GetSingleton<Config>().Is910B(), "PREFIX_ENCODER only support 800I A2/A3 inference product"},
        {opParam.quantType == infer::SelfAttentionParam::TYPE_QUANT_UNQUANT,
         "PREFIX_ENCODER doesn't support quantification, use default quantType"},
        {opParam.outDataType == ACL_DT_UNDEFINED,
         "PREFIX_ENCODER doesn't support quantification, use default outDataType"},
        {opParam.headNum >= opParam.kvHeadNum, "PREFIX_ENCODER expects headNum >= kvHeadNum"},
        {std::abs(opParam.qScale - 1.0f) < FLT_EPSILON, "PREFIX_ENCODER expects qScale to be 1.0f"},
        {!opParam.batchRunStatusEnable, "PREFIX_ENCODER doesn't support dynamic batch"},
        {opParam.isTriuMask == 1, "PREFIX_ENCODER expects isTriuMask to be 1 for alibi mask"},
        {opParam.kernelType == infer::SelfAttentionParam::KERNELTYPE_HIGH_PRECISION,
         "PREFIX_ENCODER expects kernelType to be KERNELTYPE_HIGH_PRECISION for alibi mask"},
        {opParam.clampType == infer::SelfAttentionParam::CLAMP_TYPE_UNDEFINED, "PREFIX_ENCODER doesn't support clamp"},
        {std::abs(opParam.clampMin - 0.0f) < FLT_EPSILON, "PREFIX_ENCODER doesn't support clamp"},
        {std::abs(opParam.clampMax - 0.0f) < FLT_EPSILON, "PREFIX_ENCODER doesn't support clamp"},
        {opParam.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
             opParam.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
             opParam.maskType == infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS ||
             opParam.maskType == infer::SelfAttentionParam::MASK_TYPE_CAUSAL_MASK,
         "PREFIX_ENCODER only supports alibi compress mask, norm compress mask and alibi compress sqrt mask"},
        {opParam.kvcacheCfg == atb::infer::SelfAttentionParam::K_CACHE_V_CACHE,
         "PREFIX_ENCODER doesn't support key value bypass"},
        {opParam.scaleType == infer::SelfAttentionParam::SCALE_TYPE_TOR, "PREFIX_ENCODER doesn't support logN scale"},
        {opParam.inputLayout == infer::TYPE_BSND, "PREFIX_ENCODER only supports when inputLayout is TYPE_BSND"},
        {opParam.cacheType == atb::infer::SelfAttentionParam::CACHE_TYPE_NORM,
         "PREFIX_ENCODER only supports normal cache type"},
        {opParam.mlaVHeadSize == 0, "PREFIX_ENCODER doesn't support mla"},
        {opParam.windowSize == 0, "PREFIX_ENCODER doesn't support swa"}};
    for (const std::pair<bool, const char *> &paramCheckItem : paramCheckList) {
        if (!paramCheckItem.first) {
            ATB_LOG(ERROR) << paramCheckItem.second;
            return false;
        }
    }
    return true;
}

void SelfAttentionOperation::InitMlaFaOpIni()
{
    std::stringstream opIrKeySs;
    opIrKeySs << "SelfAttentionOperationMLA";
    if (param_.maskType != infer::SelfAttentionParam::MaskType::MASK_TYPE_UNDEFINED) {
        opIrKeySs << "Mask";
    }
    if (param_.quantType == infer::SelfAttentionParam::QuantType::TYPE_QUANT_QKV_OFFLINE) {
        opIrKeySs << "QuantOffline";
    } else if (param_.quantType == infer::SelfAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
        opIrKeySs << "QuantOnline";
    }
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(opIrKeySs.str());
}

SelfAttentionOperation::SelfAttentionOperation(const infer::SelfAttentionParam &param)
    : OperationBase("SelfAttentionOperation"), param_(param)
{
    isMla_ = param_.mlaVHeadSize > 0;
    hasMask_ = (param_.maskType != infer::SelfAttentionParam::MASK_TYPE_UNDEFINED
                && param_.maskType != infer::SelfAttentionParam::MASK_TYPE_CAUSAL_MASK) &&
               !(param_.calcType == infer::SelfAttentionParam::DECODER &&
                 param_.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_NORM);
    kvHeadNum_ = (param_.kvHeadNum > 0) ? param_.kvHeadNum : param_.headNum;
    if (param_.calcType == infer::SelfAttentionParam::PA_ENCODER) {
        kcacheId_ = 1;
        maskId_ = 3;                       // 3: mask
        tokenOffsetId_ = hasMask_ ? 4 : 3; // tokenoffset 4: with mask 3: no mask
        if (isMla_) {
            tokenOffsetId_--;
            maskId_--;
        }
        hasSlopes_ = param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
                     param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
                     param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN;
        InitPaEncoderOpIni();
    } else if (param_.calcType == atb::infer::SelfAttentionParam::PREFIX_ENCODER) {
        maskId_ = 4; // 4: mask: query, key, value, blockTables, mask, qSeqLen, kvSeqLen, slopes
        kcacheId_ = 1;
        hasSlopes_ = true;
        if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS) {
            hasSlopes_ = false;
            operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("SelfAttentionOperationPrefixEncoder");
        } else if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_CAUSAL_MASK) {
            hasSlopes_ = false;
            operationIr_ =
                GetSingleton<AtbOperationIrCfg>().GetOperationIr("SelfAttentionOperationPrefixEncoderCausalMask");
        } else {
            operationIr_ =
                GetSingleton<AtbOperationIrCfg>().GetOperationIr("SelfAttentionOperationPrefixEncoderSlopes");
        }
    } else {
        kcacheId_ = 3;                     // 3: kcache
        maskId_ = 5;                       // 5: mask
        tokenOffsetId_ = hasMask_ ? 6 : 5; // tokenoffset 6: with mask 5: no mask
        if (param_.kvcacheCfg == atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS) {
            maskId_ -= 2;        // 2: key, value
            tokenOffsetId_ -= 2; // 2: key, value
            kcacheId_ -= 2;      // 2: key, value
        }
        InitFaOpIni();
    }
}

uint32_t SelfAttentionOperation::Bools2Int(bool hasScale, bool hasKV, bool hasMask, bool hasSlopes, bool hasBatch) const
{
    uint32_t ret = 0;
    ret = hasScale ? (ret | SCALE_BIT) : ret;
    ret = !hasKV ? (ret | BYPASS_BIT) : ret;
    ret = hasMask ? (ret | MASK_BIT) : ret;
    ret = hasSlopes ? (ret | SLOPES_BIT) : ret;
    ret = hasBatch ? (ret | BATCH_BIT) : ret;
    return ret;
}

void SelfAttentionOperation::InitPaEncoderOpIni()
{
    if (param_.scaleType == infer::SelfAttentionParam::SCALE_TYPE_LOGN) {
        operationIr_ =
            param_.maskType == infer::SelfAttentionParam::MASK_TYPE_UNDEFINED ?
                GetSingleton<AtbOperationIrCfg>().GetOperationIr("SelfAttentionOperationEncoderLogn1Mask0") :
                (hasSlopes_ ?
                     GetSingleton<AtbOperationIrCfg>().GetOperationIr("SelfAttentionOperationEncoderLogn1Slopes1") :
                     GetSingleton<AtbOperationIrCfg>().GetOperationIr("SelfAttentionOperationEncoderLogn1"));
    } else if (isMla_) {
        InitMlaFaOpIni();
    } else if (param_.inputLayout == infer::TYPE_BNSD) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("SelfAttentionOperationEncoderBNSD");
    } else if (param_.quantType == infer::SelfAttentionParam::QuantType::TYPE_QUANT_QKV_OFFLINE) {
        operationIr_ =
            param_.maskType == infer::SelfAttentionParam::MASK_TYPE_UNDEFINED ?
                GetSingleton<AtbOperationIrCfg>().GetOperationIr("SelfAttentionOperationEncoderQKVQuentOfflineMask0") :
                (hasSlopes_ ?
                     GetSingleton<AtbOperationIrCfg>().GetOperationIr(
                         "SelfAttentionOperationEncoderQKVQuantOfflineSlopes1") :
                     GetSingleton<AtbOperationIrCfg>().GetOperationIr("SelfAttentionOperationEncoderQKVQuantOffline"));
    } else if (param_.quantType == infer::SelfAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
        operationIr_ =
            param_.maskType == infer::SelfAttentionParam::MASK_TYPE_UNDEFINED ?
                GetSingleton<AtbOperationIrCfg>().GetOperationIr("SelfAttentionOperationEncoderQKVQuentOnlineMask0") :
                (hasSlopes_ ?
                     GetSingleton<AtbOperationIrCfg>().GetOperationIr(
                         "SelfAttentionOperationEncoderQKVQuentOnlineSlopes1") :
                     GetSingleton<AtbOperationIrCfg>().GetOperationIr("SelfAttentionOperationEncoderQKVQuentOnline"));
    } else if (param_.quantType == infer::SelfAttentionParam::QuantType::TYPE_QUANT_UNQUANT) {
        operationIr_ =
            param_.maskType == infer::SelfAttentionParam::MASK_TYPE_UNDEFINED ?
                GetSingleton<AtbOperationIrCfg>().GetOperationIr("SelfAttentionOperationEncoderMask0") :
                (hasSlopes_ ? GetSingleton<AtbOperationIrCfg>().GetOperationIr("SelfAttentionOperationEncoderSlopes1") :
                              GetSingleton<AtbOperationIrCfg>().GetOperationIr("SelfAttentionOperationEncoder"));
    }
}

void SelfAttentionOperation::InitFaOpIni()
{
    uint32_t caseCode = Bools2Int(param_.scaleType == infer::SelfAttentionParam::SCALE_TYPE_LOGN,
                                  param_.kvcacheCfg != infer::SelfAttentionParam::K_BYPASS_V_BYPASS, hasMask_,
                                  (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
                                   param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT),
                                  param_.batchRunStatusEnable);

    static std::map<uint32_t, std::string> opIniTable = {
        {31, "SelfAttentionOperationLogn1Kv0Mask1Slopes1Batch1"},
        {30, "SelfAttentionOperationLogn1Kv0Mask1Slopes1Batch0"},
        {29, "SelfAttentionOperationLogn1Kv0Mask1Slopes0Batch1"},
        {28, "SelfAttentionOperationLogn1Kv0Mask1Slopes0Batch0"},
        {25, "SelfAttentionOperationLogn1Kv0Mask0Slopes0Batch1"},
        {24, "SelfAttentionOperationLogn1Kv0Mask0Slopes0Batch0"},
        {23, "SelfAttentionOperationLogn1Kv1Mask1Slopes1Batch1"},
        {22, "SelfAttentionOperationLogn1Kv1Mask1Slopes1Batch0"},
        {21, "SelfAttentionOperationLogn1Kv1Mask1Slopes0Batch1"},
        {20, "SelfAttentionOperationLogn1Kv1Mask1Slopes0Batch0"},
        {17, "SelfAttentionOperationLogn1Kv1Mask0Slopes0Batch1"},
        {16, "SelfAttentionOperationLogn1Kv1Mask0Slopes0Batch0"},
        {15, "SelfAttentionOperationLogn0Kv0Mask1Slopes1Batch1"},
        {14, "SelfAttentionOperationLogn0Kv0Mask1Slopes1Batch0"},
        {13, "SelfAttentionOperationLogn0Kv0Mask1Slopes0Batch1"},
        {12, "SelfAttentionOperationLogn0Kv0Mask1Slopes0Batch0"},
        {9, "SelfAttentionOperationLogn0Kv0Mask0Slopes0Batch1"},
        {8, "SelfAttentionOperationLogn0Kv0Mask0Slopes0Batch0"},
        {7, "SelfAttentionOperationLogn0Kv1Mask1Slopes1Batch1"},
        {6, "SelfAttentionOperationLogn0Kv1Mask1Slopes1Batch0"},
        {5, "SelfAttentionOperationLogn0Kv1Mask1Slopes0Batch1"},
        {4, "SelfAttentionOperationLogn0Kv1Mask1Slopes0Batch0"},
        {1, "SelfAttentionOperationLogn0Kv1Mask0Slopes0Batch1"},
        {0, "SelfAttentionOperationLogn0Kv1Mask0Slopes0Batch0"},
    };
    std::map<uint32_t, std::string>::const_iterator it = opIniTable.find(caseCode);
    if (it != opIniTable.end()) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(it->second);
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << "No matched param for op ini";
    }
}

SelfAttentionOperation::~SelfAttentionOperation() {}

uint32_t SelfAttentionOperation::GetInputNum() const
{
    bool hasKV = (param_.kvcacheCfg != atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS);
    uint32_t inputNumBase = hasKV ? FUSION_IN_TENSOR_NUM : FUSION_BYPASS_IN_TENSOR_NUM; // 6或8
    if (param_.calcType == infer::SelfAttentionParam::PA_ENCODER) {
        uint32_t inputNumBasePa =
            (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_UNDEFINED ? 4 : // 4: no mask
                 (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
                  param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
                  // 6: mask and slopes, 5: mask
                  param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN) ?
                                                                                 6 :
                                                                                 5) +
            (param_.scaleType == infer::SelfAttentionParam::SCALE_TYPE_LOGN ? 1 : 0);
        bool needQKVOnlineQuant = (param_.quantType == atb::infer::SelfAttentionParam::TYPE_QUANT_QKV_ONLINE);
        bool needQKVOfflineQuant = (param_.quantType == atb::infer::SelfAttentionParam::TYPE_QUANT_QKV_OFFLINE);
        if (needQKVOnlineQuant) {
            inputNumBasePa += 4; // 4: qkDescale、qkOffset、vpvDescale、vpvOffset
        }
        if (needQKVOfflineQuant) {
            inputNumBasePa += 5; // 5: qkDescale、qkOffset、vpvDescale、vpvOffset、pScale
        }
        if (isMla_) {
            inputNumBasePa--; // 没有value
        }
        return inputNumBasePa;
    } else if (param_.calcType == infer::SelfAttentionParam::PREFIX_ENCODER) {
        if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS) {
            return PREFIX_128MASK_ENCODER_IN_TENSOR_NUM;
        } else if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_CAUSAL_MASK) {
            return FUSION_BYPASS_IN_TENSOR_NUM;
        } else {
            return PREFIX_ENCODER_IN_TENSOR_NUM;
        }
    }
    if (param_.batchRunStatusEnable) {
        inputNumBase += 1;
    }
    if (param_.maskType != infer::SelfAttentionParam::MASK_TYPE_UNDEFINED &&
        !(param_.calcType == infer::SelfAttentionParam::DECODER &&
          param_.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_NORM)) {
        inputNumBase += 1; // need mask
    }
    if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
        param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT) {
        inputNumBase += 1; // need slopes
    }
    if (param_.scaleType == infer::SelfAttentionParam::SCALE_TYPE_LOGN) {
        inputNumBase += 1; // need logn scale
    }
    return inputNumBase;
}

uint32_t SelfAttentionOperation::GetOutputNum() const
{
    if (param_.calcType == infer::SelfAttentionParam::PA_ENCODER) {
        return 1;
    } else {
        return FUSION_OUT_TENSOR_NUM;
    }
}

Status SelfAttentionOperation::BypassInferShapeImpl910B(const SVector<TensorDesc> &inTensorDescs,
                                                        SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    if (inTensorDescs.at(0).shape.dimNum == 4) { // 4: 总维度数量
        outTensorDescs.at(0).shape.dimNum = 3;   // 3: 表示输出维度
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] =   // 2: outTensor的最后一维
            inTensorDescs.at(2).shape.dims[3]; // 2: valueCache, 3: hiddenSize
    } else {
        uint64_t outHiddenSizePos = inTensorDescs.at(0).shape.dimNum - 1;     // outTensor的最后一维
        uint64_t vHiddenSizePos = inTensorDescs.at(2).shape.dimNum - 1;       // valueTensor的最后一维
        int64_t vHiddenSize = inTensorDescs.at(2).shape.dims[vHiddenSizePos]; // 2: valueTensor
        outTensorDescs.at(0).shape.dims[outHiddenSizePos] = vHiddenSize;
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferShapeImpl910B(const SVector<TensorDesc> &inTensorDescs,
                                                  SVector<TensorDesc> &outTensorDescs) const
{
    if (param_.inputLayout == atb::infer::InputLayout::TYPE_BNSD) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return NO_ERROR;
    }
    if (param_.kvcacheCfg == atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS) {
        return BypassInferShapeImpl910B(inTensorDescs, outTensorDescs);
    }
    if (param_.calcType == atb::infer::SelfAttentionParam::PREFIX_ENCODER) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return NO_ERROR;
    }
    if (inTensorDescs.at(0).shape.dimNum == 4) { // 4: 总维度数量
        outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
        outTensorDescs.at(0).format = inTensorDescs.at(0).format;
        outTensorDescs.at(0).shape.dimNum = 3; // 3: 表示输出维度
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1]; // batch == 1
        int64_t vHiddenSize =
            isMla_ ? param_.mlaVHeadSize : inTensorDescs.at(2).shape.dims[3]; // 2, 3: 设置第三维度的大小
        outTensorDescs.at(0).shape.dims[2] =                                  // 2: 设置第二维度的大小
            inTensorDescs.at(0).shape.dims[2] * vHiddenSize;                  // 2: 设置第三维度的大小
    } else if (inTensorDescs.at(0).shape.dimNum == 2) { // outTensor需要合轴 2: [nTokens, hiddenSize]
        outTensorDescs.at(0) = inTensorDescs.at(0);
        int32_t qHeadNum = param_.headNum;
        int64_t vHiddenSize = 0;
        if (isMla_) {
            vHiddenSize = param_.mlaVHeadSize;
        } else if (inTensorDescs.at(2).shape.dimNum == 2) { // 2: valueTensor 2: [nTokens, hiddenSize]
            // kvHeadNum_ is checked to be > 0 in CreateOperation()
            vHiddenSize = inTensorDescs.at(2).shape.dims[1] / kvHeadNum_; // 2: valueTensor
        } else {
            uint32_t hiddenSizePos = inTensorDescs.at(2).shape.dimNum - 1; // 2: valueTensor
            vHiddenSize = inTensorDescs.at(2).shape.dims[hiddenSizePos];   // 2: valueTensor
        }
        outTensorDescs.at(0).shape.dims[1] = qHeadNum * vHiddenSize;
    } else { // pa encoder q: [nTokens, head_num, head_size]
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dims[2] = isMla_ ? param_.mlaVHeadSize :         // 2: head_size
                                                      inTensorDescs.at(2).shape.dims[2]; // 2: value
    }
    if (param_.quantType == atb::infer::SelfAttentionParam::TYPE_QUANT_QKV_OFFLINE ||
        param_.quantType == atb::infer::SelfAttentionParam::TYPE_QUANT_QKV_ONLINE) {
        outTensorDescs.at(0).dtype = param_.outDataType;
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                              SVector<TensorDesc> &outTensorDescs) const
{
    if (GetSingleton<Config>().Is910B()) {
        return InferShapeImpl910B(inTensorDescs, outTensorDescs);
    } else {
        if (param_.inputLayout == atb::infer::InputLayout::TYPE_BNSD) {
            outTensorDescs.at(0) = inTensorDescs.at(0);
            return NO_ERROR;
        }
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (inTensorDescs.at(0).shape.dimNum == 4) { // 4: 总维度数量
            outTensorDescs.at(0).shape.dimNum = 3;   // 3: 表示输出维度
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];    // batch == 1
            outTensorDescs.at(0).shape.dims[2] =                                       // 2: 设置第二维度的大小
                inTensorDescs.at(0).shape.dims[2] * inTensorDescs.at(0).shape.dims[3]; // 2, 3: 设置第三维度的大小
        }
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::DtypeCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    aclFormat targetFormat = ACL_FORMAT_ND;
    if (!GetSingleton<Config>().Is910B()) {
        targetFormat = ACL_FORMAT_FRACTAL_NZ;
    }
    if (param_.calcType != infer::SelfAttentionParam::PA_ENCODER) {
        if (inTensorDescs.at(kcacheId_).format != targetFormat ||
            inTensorDescs.at(kcacheId_ + 1).format != targetFormat) { // +1 : vcache
            ATB_LOG(ERROR) << "kvcache dtype should be ACL_FORMAT_ND on 800I A2 inference product, "
                           << "and ACL_FORMAT_FRACTAL_NZ on Atlas 800 product "
                           << "and Atlas inference products (with Ascend 310P AI Processors)";
            return ERROR_INVALID_TENSOR_DTYPE;
        }
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    Status st = NO_ERROR;
    st = DtypeCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    st = InferLogNCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    bool needQKVQuant = (param_.quantType == atb::infer::SelfAttentionParam::TYPE_QUANT_QKV_OFFLINE ||
                         param_.quantType == atb::infer::SelfAttentionParam::TYPE_QUANT_QKV_ONLINE);
    if (needQKVQuant) {
        st = InferQKVQuantDimCheck(inTensorDescs);
        if (st != NO_ERROR) {
            return st;
        }
    }
    if (param_.inputLayout == atb::infer::InputLayout::TYPE_BNSD) {
        if (param_.calcType == infer::SelfAttentionParam::PA_ENCODER) {
            st = InferShapePADimCheckBNSD(inTensorDescs);
        } else {
            st = GetSingleton<Config>().Is910B() ? InferShapeBypassDimCheckBNSD910B(inTensorDescs) :
                                                   InferShapeBypassDimCheckBNSD310P(inTensorDescs);
        }
        if (st != NO_ERROR) {
            return st;
        }
    } else { // BSND
        st =
            GetSingleton<Config>().Is910B() ? HeadSizeDimCheck910B(inTensorDescs) : HeadSizeDimCheck310P(inTensorDescs);
        if (st != NO_ERROR) {
            return st;
        }
        if (param_.windowSize > 0) {
            st = SWAMaskDimCheck(inTensorDescs);
            if (st != NO_ERROR) {
                return st;
            }
        }
        switch (param_.calcType) {
            case infer::SelfAttentionParam::PA_ENCODER:
                st = InferShapePADimCheck(inTensorDescs);
                break;
            case infer::SelfAttentionParam::PREFIX_ENCODER:
                st = InferShapePrefixDimCheck910B(inTensorDescs);
                break;
            default:
                st = InferShapeDimCheck(inTensorDescs);
        }
        if (st != NO_ERROR) {
            return st;
        }
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    Status st = NO_ERROR;
    SVector<TensorDesc> inTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    st = DtypeCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    if (param_.inputLayout == atb::infer::InputLayout::TYPE_BNSD) {
        if (param_.calcType == infer::SelfAttentionParam::PA_ENCODER) {
            st = InferShapePADimCheckBNSD(inTensorDescs);
        } else {
            st = GetSingleton<Config>().Is910B() ? InferShapeBypassDimCheckBNSD910B(inTensorDescs) :
                                                   InferShapeBypassDimCheckBNSD310P(inTensorDescs);
        }
    } else {
        switch (param_.calcType) {
            case infer::SelfAttentionParam::PA_ENCODER:
                st = InferShapePADimCheck(inTensorDescs);
                break;
            case infer::SelfAttentionParam::PREFIX_ENCODER:
                st = InferShapePrefixDimCheck910B(inTensorDescs);
                break;
            default:
                st = InferShapeDimCheck(inTensorDescs);
        }
    }
    if (st != NO_ERROR) {
        return st;
    }
    if (param_.inputLayout == atb::infer::InputLayout::TYPE_BSND) {
        st =
            GetSingleton<Config>().Is910B() ? HeadSizeDimCheck910B(inTensorDescs) : HeadSizeDimCheck310P(inTensorDescs);
        if (st != NO_ERROR) {
            return st;
        }
        st = SetupOutTensorCheck(inTensorDescs, outTensors);
        if (st != NO_ERROR) {
            return st;
        }
    }
    if (param_.windowSize > 0) {
        st = SWAMaskDimCheck(inTensorDescs);
        if (st != NO_ERROR) {
            return st;
        }
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::SetupOutTensorCheck(const SVector<TensorDesc> &inTensorDescs,
                                                   const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> outTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(outTensors, outTensorDescs);
    if (GetSingleton<Config>().Is910B()) {
        SVector<TensorDesc> targetOutTensorDescs = {};
        targetOutTensorDescs.reserve(1);
        targetOutTensorDescs.resize(1);
        InferShapeImpl910B(inTensorDescs, targetOutTensorDescs);
        if (!TensorUtil::TensorDescEqual(outTensorDescs.at(0), targetOutTensorDescs.at(0))) {
            ATB_LOG(ERROR) << "invalid outTensor shape";
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else {
        if (inTensorDescs.at(0).shape.dimNum == 4) {       // 4: q dimNum
            if (outTensors.at(0).desc.shape.dimNum != 3) { // 3: out dimNum
                ATB_LOG(ERROR) << "invalid outTensor dimNum, should be 3";
                return ERROR_INVALID_TENSOR_SIZE;
            }
            if (outTensors.at(0).desc.shape.dims[0] != inTensorDescs.at(0).shape.dims[0] ||
                outTensors.at(0).desc.shape.dims[1] != inTensorDescs.at(0).shape.dims[1] ||
                outTensors.at(0).desc.shape.dims[2] != // 2: out第二维度的大小
                    inTensorDescs.at(0).shape.dims[2] * inTensorDescs.at(0).shape.dims[3]) { // 2, 3: q第二三维度的大小
                ATB_LOG(ERROR) << "invalid outTensor shape";
                return ERROR_INVALID_TENSOR_DIM;
            }
        } else if (!TensorUtil::TensorDescEqual(outTensorDescs.at(0), inTensorDescs.at(0))) {
            ATB_LOG(ERROR) << "invalid outTensor shape";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferQKVQuantDimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t headNum = param_.headNum;
    if (param_.quantType != atb::infer::SelfAttentionParam::TYPE_QUANT_QKV_OFFLINE) {
        // DimNum check
        if (inTensorDescs.at(inTensorDescs.size() - 2).shape.dimNum != 1 || // 2: vpvDescale 1: 1dim
            inTensorDescs.at(inTensorDescs.size() - 4).shape.dimNum != 1) { // 4: qkDescale 1: 1dim
            ATB_LOG(ERROR) << "invalid intensor dimNum";
            return ERROR_INVALID_TENSOR_DIM;
        }
        // Dim check
        if (inTensorDescs.at(inTensorDescs.size() - 2).shape.dims[0] != headNum || // 2: vpvDescale
            inTensorDescs.at(inTensorDescs.size() - 4).shape.dims[0] != headNum) { // 4: qkDescale
            ATB_LOG(ERROR) << "invalid intensor dim";
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else {
        // DimNum check
        if (inTensorDescs.at(inTensorDescs.size() - 1).shape.dimNum != 1 || // 1: pScale 1: 1dim
            inTensorDescs.at(inTensorDescs.size() - 3).shape.dimNum != 1 || // 3: vpvDescale 1: 1dim
            inTensorDescs.at(inTensorDescs.size() - 5).shape.dimNum != 1) { // 5: qkDescale 1: 1dim
            ATB_LOG(ERROR) << "invalid intensor dimNum";
            return ERROR_INVALID_TENSOR_DIM;
        }
        // Dim check
        if (inTensorDescs.at(inTensorDescs.size() - 1).shape.dims[0] != headNum || // 1: pScale
            inTensorDescs.at(inTensorDescs.size() - 3).shape.dims[0] != headNum || // 3: vpvDescale
            inTensorDescs.at(inTensorDescs.size() - 5).shape.dims[0] != headNum) { // 5: qkDescale
            ATB_LOG(ERROR) << "invalid intensor dim";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferShapeDimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    Status st = InferShapeDimNumCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    if (param_.kvcacheCfg != atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS) {
        st = InferShapeHiddenSizeCheck(inTensorDescs);
        if (st != NO_ERROR) {
            return st;
        }
    }
    // nd格式kvcache shape除最后一维外相同，nz格式kvcache shape相同
    int64_t kvCacheRange =
        GetSingleton<Config>().Is910B() ? inTensorDescs.at(kcacheId_).shape.dimNum - 1 : 5; // 5: nz kvcache dimNum
    for (int64_t i = 0; i < kvCacheRange; i++) {
        if (inTensorDescs.at(kcacheId_).shape.dims[i] != inTensorDescs.at(kcacheId_ + 1).shape.dims[i]) {
            ATB_LOG(ERROR) << "invalid kvcache shape";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    int64_t batchPos = param_.batchRunStatusEnable ? 0 : 1;
    int64_t batch = inTensorDescs.at(kcacheId_).shape.dims[batchPos];  // 3: cackeK 1: 1st dim
    if (batch != inTensorDescs.at(tokenOffsetId_).shape.dims[0] ||     // tokenOffset
        batch != inTensorDescs.at(tokenOffsetId_ + 1).shape.dims[0] || // tokenOffsetId_ + 1: seqLen
        (param_.batchRunStatusEnable &&
         batch != inTensorDescs.at(tokenOffsetId_ + 3).shape.dims[0])) { // 3: batchStatus
        ATB_LOG(ERROR) << "batch should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferShapePADimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    Status st = InferShapePADimNumCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    if (!isMla_) {
        for (size_t i = 0; i < inTensorDescs.at(kcacheId_).shape.dimNum - 1; i++) { // kvcache shape除最后一维外相同
            if (inTensorDescs.at(kcacheId_).shape.dims[i] != inTensorDescs.at(kcacheId_ + 1).shape.dims[i]) {
                ATB_LOG(ERROR) << "invalid kvcache shape";
                return ERROR_INVALID_TENSOR_DIM;
            }
        }
    }
    if (hasMask_) {
        st = GetSingleton<Config>().Is910B() ? PAMaskDimCheck(inTensorDescs) : PAMaskDimCheckNz(inTensorDescs);
        if (st != NO_ERROR) {
            return st;
        }
    }
    if (hasSlopes_ && inTensorDescs.at(tokenOffsetId_ + 1).shape.dims[0] != param_.headNum) {
        ATB_LOG(ERROR) << "shape of slopes should be the same as headNum";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(tokenOffsetId_).shape.dimNum == 2 &&   // 2: [2, batch]
        inTensorDescs.at(tokenOffsetId_).shape.dims[0] != 2) {  // 2: [2, batch]
        ATB_LOG(ERROR) << "shape of seqlen should be [batch] or [2, batch]";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferShapePADimCheckBNSD(const SVector<TensorDesc> &inTensorDescs) const
{
    Status st = InferShapePADimNumCheckBNSD(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    if (!TensorUtil::TensorDescEqual(inTensorDescs.at(kcacheId_), inTensorDescs.at(kcacheId_ + 1))) {
        ATB_LOG(ERROR) << "shape of key and value should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    uint32_t batch = inTensorDescs.at(tokenOffsetId_).shape.dims[0];
    if (inTensorDescs.at(tokenOffsetId_).shape.dimNum == 2) { // 2: [2, batch]
        batch = inTensorDescs.at(tokenOffsetId_).shape.dims[1];
        if (inTensorDescs.at(tokenOffsetId_).shape.dims[0] != 2) { // 2: [2, batch]
            ATB_LOG(ERROR) << "shape of seqlen should be [batch] or [2, batch]";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (batch != inTensorDescs.at(0).shape.dims[0] || batch != inTensorDescs.at(kcacheId_).shape.dims[0]) {
        ATB_LOG(ERROR) << "batch of seqlen, query, key and value should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(0).shape.dims[1] != param_.headNum) {
        ATB_LOG(ERROR) << "2nd dim of query should be same as param headNum when inputLayout is TYPE_BNSD";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(1).shape.dims[1] != kvHeadNum_) {
        ATB_LOG(ERROR) << "2nd dim of key should be same as kvHeadNum when inputLayout is TYPE_BNSD";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(0).shape.dims[3] != inTensorDescs.at(kcacheId_).shape.dims[3]) { // 3: headSize
        ATB_LOG(ERROR) << "headSize of key and query should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(0).shape.dims[3] > MAX_HEAD_SIZE) { // 3: headSize
        ATB_LOG(ERROR) << "headSize should be no greater than 256 when inputLayout is TYPE_BNSD";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::PAMaskDimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    if (hasSlopes_) {
        if (inTensorDescs.at(maskId_).dtype == aclDataType::ACL_FLOAT16 &&
            param_.kernelType == infer::SelfAttentionParam::KERNELTYPE_DEFAULT) {
            ATB_LOG(ERROR) << "kernelType should be KERNELTYPE_HIGH_PRECISION"
                           << "when using alibi compress mask with float16";
            return ERROR_INVALID_PARAM;
        }
        if (inTensorDescs.at(maskId_).shape.dimNum == 2 &&      // 2: [256,256]
            (inTensorDescs.at(maskId_).shape.dims[0] != 256 ||  // 256 : compress mask shape
             inTensorDescs.at(maskId_).shape.dims[1] != 256)) { // 256 : compress mask shape
            ATB_LOG(ERROR) << "invalid alibi compress mask shape, should be [256, 256]";
            return ERROR_INVALID_TENSOR_DIM;
        } else if (inTensorDescs.at(maskId_).shape.dimNum == 3) { // 3: [head_num, seqlen, 128]
            if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN) {
                ATB_LOG(ERROR) << " shape of left align alibi compress mask should be [256, 256]";
                return ERROR_INVALID_TENSOR_DIM;
            }
            if (inTensorDescs.at(maskId_).shape.dims[0] != param_.headNum &&
                inTensorDescs.at(maskId_).shape.dims[2] != 128) { // 2: last dim 128 : compress mask shape
                ATB_LOG(ERROR) << "invalid alibi compress mask shape, should be [head_num, seqlen, 128]";
                return ERROR_INVALID_TENSOR_DIM;
            }
        } else if (inTensorDescs.at(maskId_).shape.dimNum == 4) { // 4: wrong mask dimnum
            ATB_LOG(ERROR) << "invalid alibi compress mask dimNum";
            return ERROR_INVALID_TENSOR_SIZE;
        }
    } else if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS) {
        if (inTensorDescs.at(maskId_).shape.dimNum != 2) { // 2: [128,128]
            ATB_LOG(ERROR) << "invalid compress mask dimNum";
            return ERROR_INVALID_TENSOR_SIZE;
        }
        if (inTensorDescs.at(maskId_).shape.dims[0] != 128 || // 128 : compress mask shape
            inTensorDescs.at(maskId_).shape.dims[1] != 128) { // 128 : compress mask shape
            ATB_LOG(ERROR) << "invalid compress mask shape";
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else {
        if (inTensorDescs.at(maskId_).shape.dimNum != 2 && // 5: attnMask 2: 2 dims
            inTensorDescs.at(maskId_).shape.dimNum != 3 && // 5: attnMask 3: 3 dims
            inTensorDescs.at(maskId_).shape.dimNum != 4) { // 5: attnMask 4: 4 dims
            ATB_LOG(ERROR) << "invalid mask dimNum";
            return ERROR_INVALID_TENSOR_SIZE;
        }
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::PAMaskDimCheckNz(const SVector<TensorDesc> &inTensorDescs) const
{
    if (hasSlopes_) {
        if (inTensorDescs.at(maskId_).shape.dimNum != 4) { // 4: [head_num,128//16,maxSeqlen,16] or [1,256//16,256,16]
            ATB_LOG(ERROR) << "invalid mask dimNum";
            return ERROR_INVALID_TENSOR_SIZE;
        } else if (inTensorDescs.at(maskId_).shape.dimNum == 4 &&   // 4: compress mask dimNum
                   inTensorDescs.at(maskId_).shape.dims[3] != 16) { // 3: dim3 16: nz format
            ATB_LOG(ERROR) << "invalid alibi compress mask shape";
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS) {
        if (inTensorDescs.at(maskId_).shape.dimNum != 4) { // 4: [1,8,128,16]
            ATB_LOG(ERROR) << "invalid compress mask dimNum";
            return ERROR_INVALID_TENSOR_SIZE;
        }
        if (inTensorDescs.at(maskId_).shape.dims[0] != 1 ||   // 1 : compress mask shape
            inTensorDescs.at(maskId_).shape.dims[1] != 8 ||   // 8 : compress mask shape
            inTensorDescs.at(maskId_).shape.dims[2] != 128 || // 2: dim2 128 : compress mask shape
            inTensorDescs.at(maskId_).shape.dims[3] != 16) {  // 3: dim3 16 : compress mask shape
            ATB_LOG(ERROR) << "invalid compress mask shape";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::SWAMaskDimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.calcType == infer::SelfAttentionParam::DECODER) { // decoder无mask
        return NO_ERROR;
    }
    uint32_t maskId = 5; // 5: mask
    if (param_.kvcacheCfg == atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS ||
        param_.calcType == infer::SelfAttentionParam::PA_ENCODER) {
        maskId -= 2; // 2: key,value 不传
    }
    if (isMla_) {
        maskId = 2;
    }
    if (GetSingleton<Config>().Is910B()) {
        if (inTensorDescs.at(maskId).shape.dimNum != 2) { // 2: mask dimNum
            ATB_LOG(ERROR) << "dimNum of swa mask should be two";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_COMPRESS) {
            if (inTensorDescs.at(maskId).shape.dims[0] != 512 || // 512: swa compress mask shape
                inTensorDescs.at(maskId).shape.dims[1] != 512) { // 512: swa compress mask shape
                ATB_LOG(ERROR) << "shape of swa compress mask should be [512,512]";
                return ERROR_INVALID_TENSOR_DIM;
            }
        }
    } else {
        if (inTensorDescs.at(maskId).shape.dimNum != 4) { // 4: mask dimNum 310p
            ATB_LOG(ERROR) << "dimNum of swa mask should be four";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_COMPRESS) {
            if (inTensorDescs.at(maskId).shape.dims[0] != 1 ||   // [1, 32, 512, 16]: swa compress mask shape
                inTensorDescs.at(maskId).shape.dims[1] != 32 ||  // [1, 32, 512, 16]: swa compress mask shape
                inTensorDescs.at(maskId).shape.dims[2] != 512 || // [1, 32, 512, 16]: swa compress mask shape
                inTensorDescs.at(maskId).shape.dims[3] != 16) {  // [1, 32, 512, 16]: swa compress mask shape
                ATB_LOG(ERROR) << "shape of swa compress mask should be [1, 32, 512, 16]";
                return ERROR_INVALID_TENSOR_DIM;
            }
        }
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::HeadSizeDimCheck910B(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t headSizeK = 0;
    int64_t headSizeV = 0;
    uint32_t lastDimPosK = inTensorDescs.at(1).shape.dimNum - 1; // 1: key
    uint32_t lastDimPosV = inTensorDescs.at(2).shape.dimNum - 1; // 2: value

    if (param_.kvcacheCfg == atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS ||
        inTensorDescs.at(1).shape.dimNum == 2 || // 2: [nTokens, qHiddenSize]
        (param_.calcType == atb::infer::SelfAttentionParam::PREFIX_ENCODER &&
         inTensorDescs.at(1).shape.dimNum == 3)) { // PREFIX_ENCODER 3: [numBlocks, blockSize, kvHiddenSize]
        int64_t hiddenSizeK = inTensorDescs.at(1).shape.dims[lastDimPosK]; // 1: key
        if (hiddenSizeK % kvHeadNum_ != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "hiddenSizeK(" << hiddenSizeK << ") should be multiples of kvHeadNum("
                           << kvHeadNum_ << ")";
            return ERROR_INVALID_TENSOR_DIM;
        }
        int64_t hiddenSizeV = inTensorDescs.at(2).shape.dims[lastDimPosV]; // 2: value
        if (!isMla_ && (hiddenSizeV % kvHeadNum_ != 0)) {
            ATB_LOG(ERROR) << GetLogPrefix() << "hiddenSizeV(" << hiddenSizeV << ") should be multiples of kvHeadNum("
                           << kvHeadNum_ << ")";
            return ERROR_INVALID_TENSOR_DIM;
        }
        headSizeK = hiddenSizeK / kvHeadNum_;
        if (!isMla_) {
            headSizeV = hiddenSizeV / kvHeadNum_; // 2: cacheV
        }
    } else {
        headSizeK = inTensorDescs.at(1).shape.dims[lastDimPosK];
        headSizeV = inTensorDescs.at(2).shape.dims[lastDimPosV]; // 2: cacheV
    }
    if (isMla_) {
        headSizeV = param_.mlaVHeadSize;
        if (param_.mlaVHeadSize > headSizeK) {
            ATB_LOG(ERROR) << GetLogPrefix() << "param mlaVHeadSize(" << headSizeV
                           << ") should be no greater than headSizeK(" << headSizeK << ")";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return MaxHeadSizeCheck910B(headSizeK, headSizeV);
}

Status SelfAttentionOperation::MaxHeadSizeCheck910B(const int64_t headSizeK, const int64_t headSizeV) const
{
    int64_t maxHeadSize = MAX_HEAD_SIZE_MLA;
    if ((param_.windowSize > 0 && !isMla_) || param_.scaleType != infer::SelfAttentionParam::SCALE_TYPE_TOR ||
        param_.inputLayout == atb::infer::InputLayout::TYPE_BNSD ||
        (!isMla_ && param_.quantType != infer::SelfAttentionParam::QuantType::TYPE_QUANT_UNQUANT)) {
        maxHeadSize = 256; // 256: 不支持mla的场景headsize小于等于256，且headSizeK，headSizeV需要相等
        if (headSizeK != headSizeV) {
            ATB_LOG(ERROR) << GetLogPrefix() << "headSizeK(" << headSizeK << ") and headSizeV(" << headSizeV
                           << ") should be same";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
        param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
        param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN) {
        maxHeadSize = 128; // 128: 压缩alibi情况headsize小于等于128
    }
    if (headSizeK > maxHeadSize) {
        ATB_LOG(ERROR) << GetLogPrefix() << "headSizeK(" << headSizeK << ") should be no greater than " << maxHeadSize;
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (headSizeV > maxHeadSize) {
        ATB_LOG(ERROR) << GetLogPrefix() << "headSizeV(" << headSizeV << ") should be no greater than " << maxHeadSize;
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::HeadSizeDimCheck310P(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t headSizeK = 0;
    int64_t headSizeV = 0;
    if (!TensorUtil::TensorDescEqual(inTensorDescs.at(1), inTensorDescs.at(2))) { // 2: cacheV
        ATB_LOG(ERROR) << GetLogPrefix() << "shape of internsor1 and intensor2 should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (param_.kvcacheCfg == atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS) {
        if (inTensorDescs.at(1).shape.dims[2] * 16 % kvHeadNum_ != 0 || // 2: hiddenSize/16 16: 16对齐
            inTensorDescs.at(2).shape.dims[2] * 16 % kvHeadNum_ != 0) { // 2: value 2: hiddenSize/16 16: 16对齐
            ATB_LOG(ERROR) << GetLogPrefix() << "hiddenSize of key and value should be multiples of kvHeadNum";
            return ERROR_INVALID_TENSOR_DIM;
        }
        headSizeK = inTensorDescs.at(1).shape.dims[2] * 16 / kvHeadNum_; // 2: hiddenSize/16 16: 16对齐
        headSizeV = inTensorDescs.at(2).shape.dims[2] * 16 / kvHeadNum_; // 2: cacheV 2: hiddenSize/16 16: 16对齐
    } else if (inTensorDescs.at(1).shape.dimNum == 2) {                  // 2: [nTokens, hiddenSize]
        if (inTensorDescs.at(1).shape.dims[1] % kvHeadNum_ != 0 ||
            inTensorDescs.at(2).shape.dims[1] % kvHeadNum_ != 0) { // 2: value
            ATB_LOG(ERROR) << GetLogPrefix() << "hiddenSize of key and value should be multiples of kvHeadNum";
            return ERROR_INVALID_TENSOR_DIM;
        }
        headSizeK = inTensorDescs.at(1).shape.dims[1] / kvHeadNum_;
        headSizeV = inTensorDescs.at(2).shape.dims[1] / kvHeadNum_; // 2: value
    } else {
        uint32_t lastDimPos = inTensorDescs.at(1).shape.dimNum - 1;
        headSizeK = inTensorDescs.at(1).shape.dims[lastDimPos];
        lastDimPos = inTensorDescs.at(2).shape.dimNum - 1;      // 2: cacheV
        headSizeV = inTensorDescs.at(2).shape.dims[lastDimPos]; // 2: cacheV
    }
    if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
        param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
        param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN) {
        if (headSizeK > 128 || headSizeV > 128) { // 128: 压缩alibi情况headsize小于等于128
            ATB_LOG(ERROR) << GetLogPrefix()
                           << "headSize of key and value should be no greater than 128 with alibi compress mask";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (headSizeK > MAX_HEAD_SIZE || headSizeV > MAX_HEAD_SIZE) {
        ATB_LOG(ERROR) << GetLogPrefix() << "headSize of key and value should be no greater than 256";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (headSizeK % 16 != 0 || headSizeV % 16 != 0) { // 16: kvcache约束 16 ：kvcache约束
        ATB_LOG(ERROR) << GetLogPrefix() << "headSize of key and value should be multiples of 16.";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferShapeDimNumCheckBNSD(const SVector<TensorDesc> &inTensorDescs) const
{
    // check qkv
    if (inTensorDescs.at(0).shape.dimNum != 4) { // 0: q 4: 4 dims
        ATB_LOG(ERROR) << "dimNum of query should be 4";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(kcacheId_).shape.dimNum != 5 ||     // 5: 5 dims
        inTensorDescs.at(kcacheId_ + 1).shape.dimNum != 5) { // kcacheId_+1: cacheV 5: 5 dims
        ATB_LOG(ERROR) << "dimNum of cacheK or cacheV should be 5";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    // check mask
    if (hasMask_) {
        Status st = InferShapeMaskDimNumCheck(inTensorDescs);
        if (st != NO_ERROR) {
            return ERROR_INVALID_TENSOR_DIM_NUM;
        }
    }
    if (inTensorDescs.at(tokenOffsetId_).shape.dimNum != 1 ||     // tokenOffset 1: 1dim
        inTensorDescs.at(tokenOffsetId_ + 1).shape.dimNum != 1 || // seqLen 1: 1 dim
        inTensorDescs.at(tokenOffsetId_ + 2).shape.dimNum != 1) { // tokenOffsetId + 2: layerId 1: 1 dim
        ATB_LOG(ERROR) << "dimNum of tokenOffset, seqLen and layerId should be 1";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (param_.batchRunStatusEnable) {
        if (inTensorDescs.at(tokenOffsetId_ + 3).shape.dimNum != 1) { // tokenOffsetId + 3: batchStatusId 1: 1 dim
            ATB_LOG(ERROR) << "dimNum of batchStatus should be 1";
            return ERROR_INVALID_TENSOR_DIM_NUM;
        }
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferShapeBypassDimCheckBNSD910B(const SVector<TensorDesc> &inTensorDescs) const
{
    // InferShapeBypassDimNumCheckBNSD910B ENCODER AND DECODER
    Status st = InferShapeDimNumCheckBNSD(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    // InferShapeBypassDimCheckBNSD910B ENCODER AND DECODER
    if (!TensorUtil::TensorDescEqual(inTensorDescs.at(kcacheId_), inTensorDescs.at(kcacheId_ + 1))) {
        ATB_LOG(ERROR) << "shape of cacheK and cacheV should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t batch = inTensorDescs.at(0).shape.dims[0];        // 0: q 0: 0th dim
    int64_t headNum = inTensorDescs.at(0).shape.dims[1];      // 0: q 1: head_num
    int64_t headSize = inTensorDescs.at(0).shape.dims[3];     // 0: q 3: head_size
    if (batch != inTensorDescs.at(kcacheId_).shape.dims[1]) { // 1: cacheK 1: 1st dim
        ATB_LOG(ERROR) << "batch of query and cacheK should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(kcacheId_).shape.dims[2] != kvHeadNum_) { // 1: cacheK 2: head_num
        ATB_LOG(ERROR) << "Dim2 of cacheK and cacheV should equal to kvHeadNum";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (headNum != param_.headNum) {
        ATB_LOG(ERROR) << "Dim1 of query should equal to headNum";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (headSize > 256) { // 256: headsize limit
        ATB_LOG(ERROR) << "headSize of query should be no greater than 256";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(kcacheId_).shape.dims[4] != headSize) { // 1: cacheK 4: head_size
        ATB_LOG(ERROR) << "headSize of query should be same as cacheK";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (batch != inTensorDescs.at(tokenOffsetId_).shape.dims[0] ||     // tokenOffset
        batch != inTensorDescs.at(tokenOffsetId_ + 1).shape.dims[0] || // tokenOffsetId_ + 1: seqLen
        (param_.batchRunStatusEnable &&
         batch != inTensorDescs.at(tokenOffsetId_ + 3).shape.dims[0])) { // 3: batchStatus
        ATB_LOG(ERROR) << "batch of query, tokenOffset, seqLen and batchStatus should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferShapeBypassDimCheckBNSD310P(const SVector<TensorDesc> &inTensorDescs) const
{
    // InferShapeBypassDimNumCheck
    Status st = InferShapeDimNumCheckBNSD(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    // InferShapeBypassDimCheckBNSD310P
    if (!TensorUtil::TensorDescEqual(inTensorDescs.at(kcacheId_), inTensorDescs.at(kcacheId_ + 1))) {
        ATB_LOG(ERROR) << "shape of cacheK and cacheV should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t batch = inTensorDescs.at(0).shape.dims[0];    // 0: q 0: 0th dim
    int64_t headNum = inTensorDescs.at(0).shape.dims[1];  // 0: q 1: head_num
    int64_t headSize = inTensorDescs.at(0).shape.dims[3]; // 0: q 3: head_size
    if (batch != inTensorDescs.at(kcacheId_).shape.dims[1] / kvHeadNum_) {
        ATB_LOG(ERROR) << "2nd dim of cacheK should be batch * kvHeadNum";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (headNum != param_.headNum) {
        ATB_LOG(ERROR) << "Dim1 of query should equal to headNum";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (headSize > 256) { // 256: headsize limit
        ATB_LOG(ERROR) << "headSize of query should be no greater than 256";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(kcacheId_).shape.dims[4] != 16) { // 4: 4th dim 16: nz format
        ATB_LOG(ERROR) << "last dim of nz cacheK should be 16";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(kcacheId_).shape.dims[2] * 16 != headSize) { // 2:  headSize / 16, 16: nz format
        ATB_LOG(ERROR) << "headSize of query should be same as cacheK";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (batch != inTensorDescs.at(tokenOffsetId_).shape.dims[0] ||     // tokenOffset
        batch != inTensorDescs.at(tokenOffsetId_ + 1).shape.dims[0] || // tokenOffsetId_ + 1: seqLen
        (param_.batchRunStatusEnable &&
         batch != inTensorDescs.at(tokenOffsetId_ + 3).shape.dims[0])) { // 3: batchStatus
        ATB_LOG(ERROR) << "batch of query, tokenOffset, seqLen and batchStatus should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}
Status SelfAttentionOperation::InferShapeHiddenSizeCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t hiddenSizeK = 0;
    int64_t hiddenSizeV = 0;
    int64_t hiddenSizeKcacahe = 0;
    int64_t hiddenSizeVcacahe = 0;
    if (inTensorDescs.at(1).shape.dimNum == 4) {          // 4: 4 dims
        hiddenSizeK = inTensorDescs.at(1).shape.dims[2] * // 1: k 2: 2nd dim
                      inTensorDescs.at(1).shape.dims[3];  // 1: k 3: 3rd dim
    } else {
        hiddenSizeK = inTensorDescs.at(1).shape.dims[1]; // 1: k 1: 1st dim
    }
    if (inTensorDescs.at(2).shape.dimNum == 4) {          // 2: cacheV 4: 4 dims
        hiddenSizeV = inTensorDescs.at(2).shape.dims[2] * // 2: v 2: 2nd dim
                      inTensorDescs.at(2).shape.dims[3];  // 2: v 3: 3rd dim
    } else {
        hiddenSizeV = inTensorDescs.at(2).shape.dims[1]; // 2: v 1: 1st dim
    }
    if (GetSingleton<Config>().Is910B()) {
        uint64_t hiddenSizePos = inTensorDescs.at(3).shape.dimNum - 1;     // 3: kcache
        hiddenSizeKcacahe = inTensorDescs.at(3).shape.dims[hiddenSizePos]; // 3: kcache 3: khiddenSize
        hiddenSizeVcacahe = inTensorDescs.at(4).shape.dims[hiddenSizePos]; // 4: vcache 3: khiddenSize
    } else {
        hiddenSizeKcacahe = inTensorDescs.at(3).shape.dims[2] * // 3: kcache 2: embedim / 16
                            inTensorDescs.at(3).shape.dims[4];  // 3: kcache 4: 16
        hiddenSizeVcacahe = inTensorDescs.at(4).shape.dims[2] * // 4: vcache 2: embedim / 16
                            inTensorDescs.at(4).shape.dims[4];  // 4: vcache 4: 16
    }
    if (hiddenSizeK != hiddenSizeKcacahe) { // 3: cacheK 3: 3rd dim
        ATB_LOG(ERROR) << "hiddenSize of k should be the same as cacheK";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (hiddenSizeV != hiddenSizeVcacahe) { // 4: cacheV 3: 3rd dim
        ATB_LOG(ERROR) << "hiddenSize of v should be the same as cacheV";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (hiddenSizeV % 16 != 0 || hiddenSizeK % 16 != 0) { // 16: kvcache约束 16 ：kvcache约束
        ATB_LOG(ERROR) << "hiddenSize of key and value should be multiples of 16.";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferShapeMaskDimNumCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    if (GetSingleton<Config>().Is910B()) {
        if (inTensorDescs.at(maskId_).shape.dimNum != 2 && // 5: attnMask 2: 2 dims
            inTensorDescs.at(maskId_).shape.dimNum != 3 && // 5: attnMask 3: 3 dims
            inTensorDescs.at(maskId_).shape.dimNum != 4) { // 5: attnMask 4: 4 dims
            ATB_LOG(ERROR) << "invalid mask dimNum";
            return ERROR_INVALID_TENSOR_SIZE;
        }
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferShapeDimNumCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    uint64_t kvcacheDimNum = param_.batchRunStatusEnable ? 3 : 4; // DimNum of kvcache 3: dynamic batch 4: normal case
    if (!GetSingleton<Config>().Is910B()) {
        kvcacheDimNum = 5; // 5: nz format kvcache dimNum
    }
    // check qkv
    if ((inTensorDescs.at(0).shape.dimNum != 2 &&  // 0: q 2: 2 dims
         inTensorDescs.at(0).shape.dimNum != 4)) { // 0: q 4: 4 dims
        ATB_LOG(ERROR) << "dimNum of query should be 2 or 4";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (param_.kvcacheCfg != atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS) {
        if ((inTensorDescs.at(1).shape.dimNum != 2 &&  // 1: k 2: 2 dims
             inTensorDescs.at(1).shape.dimNum != 4) || // 1: k 4: 4 dims
            (inTensorDescs.at(2).shape.dimNum != 2 &&  // 2: v 2: 2 dims
             inTensorDescs.at(2).shape.dimNum != 4)) { // 2: v 4: 4 dims
            ATB_LOG(ERROR) << "invalid intensor dimNum";
            return ERROR_INVALID_TENSOR_DIM_NUM;
        }
    }
    // check kvcache
    if (inTensorDescs.at(kcacheId_).shape.dimNum != kvcacheDimNum ||
        inTensorDescs.at(kcacheId_ + 1).shape.dimNum != kvcacheDimNum) {
        ATB_LOG(ERROR) << "Invalid dimNum of cacheK and cacheV";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    // check mask
    if (hasMask_) {
        Status st = InferShapeMaskDimNumCheck(inTensorDescs);
        if (st != NO_ERROR) {
            return st;
        }
    }
    if (inTensorDescs.at(tokenOffsetId_).shape.dimNum != 1 ||     // tokenOffset 1: 1dim
        inTensorDescs.at(tokenOffsetId_ + 1).shape.dimNum != 1 || // seqLen 1: 1 dim
        inTensorDescs.at(tokenOffsetId_ + 2).shape.dimNum != 1) { // tokenOffsetId + 2: layerId 1: 1 dim
        ATB_LOG(ERROR) << "invalid intensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (param_.batchRunStatusEnable) {
        if (inTensorDescs.at(tokenOffsetId_ + 3).shape.dimNum != 1) { // tokenOffsetId + 3: batchStatusId 1: 1 dim
            ATB_LOG(ERROR) << "invalid batchStatus dimNum";
            return ERROR_INVALID_TENSOR_DIM_NUM;
        }
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferShapePADimNumCheckBNSD(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(0).shape.dimNum != 4) { // 4: [batch, head_num, seq_len, head_size]
        ATB_LOG(ERROR) << GetLogPrefix() << "dimNum of query should be 4, bot got: "
                       << inTensorDescs.at(0).shape.dimNum;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(kcacheId_).shape.dimNum != 4) { // 4: [batch, head_num, seq_len, head_size]
        ATB_LOG(ERROR) << GetLogPrefix() << "dimNum of key should be 4, bot got: "
                       << inTensorDescs.at(kcacheId_).shape.dimNum;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    uint64_t seqLenDimNum = inTensorDescs.at(tokenOffsetId_).shape.dimNum;
    if (seqLenDimNum != 1 && seqLenDimNum != 2) { // 2: seqlen: [2, batch]
        ATB_LOG(ERROR) << GetLogPrefix() << "dimNum of seqlen should be 1 or 2, bot got: " << seqLenDimNum;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferShapePADimNumCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    if (isMla_) {
        if ((inTensorDescs.at(0).shape.dimNum != 2 &&  // 0: q 2: 2 [nTokens, hiddenSize]
             inTensorDescs.at(0).shape.dimNum != 3 &&  // 0: q 3: [nTokens, head_num, head_size]
             inTensorDescs.at(0).shape.dimNum != 4)) { // 0: q 4: [batch, seq_len, head_num, head_size]
            ATB_LOG(ERROR) << "dimNum of query should be 2, 3 or 4";
            return ERROR_INVALID_TENSOR_SIZE;
        }
        if ((inTensorDescs.at(1).shape.dimNum != 2 &&  // 1: k 2: 2 [nTokens, hiddenSize]
             inTensorDescs.at(1).shape.dimNum != 3 &&  // 1: k 3: [nTokens, head_num, head_size]
             inTensorDescs.at(1).shape.dimNum != 4)) { // 1: k 4: [batch, seq_len, head_num, head_size]
            ATB_LOG(ERROR) << "dimNum of key should be 2, 3 or 4";
            return ERROR_INVALID_TENSOR_SIZE;
        }
    } else {
        if (inTensorDescs.at(0).shape.dimNum != 2 && // 0: q 2: 2 [nTokens, hiddenSize]
            inTensorDescs.at(0).shape.dimNum != 3) { // 0: q 3: [nTokens, head_num, head_size]
            ATB_LOG(ERROR) << "dimNum of query should be 2, 3";
            return ERROR_INVALID_TENSOR_SIZE;
        }
        if (inTensorDescs.at(0).shape.dimNum != inTensorDescs.at(1).shape.dimNum) { // 0: q 1: K
            ATB_LOG(ERROR) << "dimNum of key and query should be the same";
            return ERROR_INVALID_TENSOR_SIZE;
        }
        if (inTensorDescs.at(2).shape.dimNum != inTensorDescs.at(1).shape.dimNum) { // 1: K 2: V
            ATB_LOG(ERROR) << "dimNum of key and value should be the same";
            return ERROR_INVALID_TENSOR_SIZE;
        }
    }
    if (hasSlopes_ && inTensorDescs.at(tokenOffsetId_ + 1).shape.dimNum != 1) { // tokenOffsetId + 1: Slopes 1: 1 dim
        ATB_LOG(ERROR) << "invalid intensor dimNum";
        return ERROR_INVALID_TENSOR_SIZE;
    }
    if (inTensorDescs.at(tokenOffsetId_).shape.dimNum != 1 &&
        inTensorDescs.at(tokenOffsetId_).shape.dimNum != 2) { // 2: seqlen: [2, batch]
        ATB_LOG(ERROR) << "dimNum of seqlen should be 1 or 2";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferLogNCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.scaleType == infer::SelfAttentionParam::SCALE_TYPE_LOGN) {
        if (inTensorDescs.at(inTensorDescs.size() - 1).shape.dimNum != 1) {
            ATB_LOG(ERROR) << GetLogPrefix() << "invalid logN intensor dimNum";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (param_.calcType != infer::SelfAttentionParam::PA_ENCODER &&
            param_.calcType != infer::SelfAttentionParam::DECODER) {
            ATB_LOG(ERROR) << GetLogPrefix() << "When use logN func, calcType need be PA_ENCODER or DECODER";
            return ERROR_INVALID_PARAM;
        }
        if (GetSingleton<Config>().Is910B()) {
            if (param_.calcType == infer::SelfAttentionParam::PA_ENCODER &&
                param_.kernelType != infer::SelfAttentionParam::KERNELTYPE_HIGH_PRECISION) {
                ATB_LOG(ERROR) << GetLogPrefix() << "PA ENCODER enable logN func need KERNELTYPE_HIGH_PRECISION";
                return ERROR_INVALID_PARAM;
            }
            if (inTensorDescs.at(inTensorDescs.size() - 1).dtype == ACL_FLOAT16) {
                ATB_LOG(ERROR) << GetLogPrefix() << "dtype of logN in Atlas 800I A2 inference product should be float";
                return ERROR_INVALID_TENSOR_DTYPE;
            }
        } else {
            if (inTensorDescs.at(inTensorDescs.size() - 1).dtype == ACL_FLOAT) {
                ATB_LOG(ERROR) << GetLogPrefix() << "dtype of logN in Atlas inference products should be float16";
                return ERROR_INVALID_TENSOR_DTYPE;
            }
        }
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferShapePrefixDimNumCheck910B(const SVector<TensorDesc> &inTensorDescs) const
{
    // IR: query, key, value, blockTables, mask, slopes
    const std::size_t kQueryIndex = 0;
    const std::size_t kKeyIndex = 1;
    const std::size_t kValueIndex = 2;
    const std::size_t kBlockTablesIndex = 3;
    const std::size_t kSeqLenIndex = param_.maskType != infer::SelfAttentionParam::MASK_TYPE_CAUSAL_MASK ? 5 : 4;
    const std::size_t kSlopesIndex = 7;
    if ((inTensorDescs.at(kQueryIndex).shape.dimNum != 2 &&  // 2: query: [batch * qSeqLen, qHiddenSize]
         inTensorDescs.at(kQueryIndex).shape.dimNum != 3)) { // 3: [batch * seqLen, headNum, headSize]
        ATB_LOG(ERROR) << "DimNum of query should be 2 or 3";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if ((inTensorDescs.at(kKeyIndex).shape.dimNum != 3 &&  // 3: key, value: [numBlocks, blockSize, kvHiddenSize]
         inTensorDescs.at(kKeyIndex).shape.dimNum != 4)) { // 4: [numBlocks, blockSize, headNum, headSize]
        ATB_LOG(ERROR) << "DimNum of key should be 3 or 4";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if ((inTensorDescs.at(kKeyIndex).shape.dimNum != inTensorDescs.at(kValueIndex).shape.dimNum)) {
        ATB_LOG(ERROR) << "DimNum of key and value must be the same";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(kBlockTablesIndex).shape.dimNum != 2) { // 2: blockTables: [batch, maxBlockNum]
        ATB_LOG(ERROR) << "DimNum of blockTables should be 2";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(kSeqLenIndex).shape.dimNum != 1) { // 1: seqLen: [batch]
        ATB_LOG(ERROR) << "DimNum of seqlen should be 1";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(kSeqLenIndex + 1).shape.dimNum != 1) { // 1: kvSeqLen: [batch]
        ATB_LOG(ERROR) << "DimNum of kvSeqLen should be 1";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (hasSlopes_ && inTensorDescs.at(kSlopesIndex).shape.dimNum != 1) { // 1: kvSeqLen: [head]
        ATB_LOG(ERROR) << "DimNum of seqlen should be 1";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    return NO_ERROR;
}

Status SelfAttentionOperation::InferShapePrefixDimCheck910B(const SVector<TensorDesc> &inTensorDescs) const
{
    const std::size_t kBlockTablesIndex = 3;
    const std::size_t kSeqLenIndex = param_.maskType != infer::SelfAttentionParam::MASK_TYPE_CAUSAL_MASK ? 5 : 4;
    const std::size_t kBlockSizeIndex = 1;
    const int32_t kMaxHeadSize = 128;
    Status st = InferShapePrefixDimNumCheck910B(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    if (hasMask_) {
        st = PAMaskDimCheck(inTensorDescs); // check mask
    }
    if (st != NO_ERROR) {
        return st;
    }
    // blockSize * headSize <= kMaxHeadSize ​* kMaxHeadSize
    int32_t blockSize = static_cast<int32_t>(inTensorDescs.at(1).shape.dims[kBlockSizeIndex]);
    uint64_t kvDimNum = inTensorDescs.at(1).shape.dimNum;
    int32_t headSize = static_cast<int32_t>(inTensorDescs.at(1).shape.dims[kvDimNum - 1]);
    if (inTensorDescs.at(1).shape.dimNum == 3) { // dimNum = 3, [numBlocks, blockSize, kvHiddenSize]
        headSize /= kvHeadNum_;                  // kvHiddenSize = headSize * kvHeadNum
    }
    if (headSize > kMaxHeadSize) {
        ATB_LOG(ERROR) << "invalid key, value HeadSize, maximum: " << kMaxHeadSize << ", but got " << headSize;
        return ERROR_INVALID_TENSOR_DIM;
    }
    int32_t qHeadSize = static_cast<int32_t>(inTensorDescs.at(0).shape.dims[inTensorDescs.at(0).shape.dimNum - 1]);
    // dimNum = 2, [batch*qSeqLen, qHiddenSize]
    qHeadSize /= inTensorDescs.at(0).shape.dimNum == 2 ? param_.headNum : 1;
    if (qHeadSize > kMaxHeadSize) {
        ATB_LOG(ERROR) << "invalid query HeadSize, maximum: " << kMaxHeadSize << ", but got " << qHeadSize;
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (qHeadSize % 16 != 0 || headSize % 16 != 0) { // 16: query headSize约束 16: kv headSize约束
        ATB_LOG(ERROR) << "headSize of query, key and value should be multiples of 16";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (blockSize * headSize > kMaxHeadSize * kMaxHeadSize) {
        ATB_LOG(ERROR) << "invalid blockSize * headSize, maximum: 128 * 128"
                       << ", but got " << blockSize * headSize;
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t seqlenBatch = inTensorDescs.at(kSeqLenIndex).shape.dims[0];           // 0: seqLen: [batch]
    int64_t kvSeqLenBatch = inTensorDescs.at(kSeqLenIndex + 1).shape.dims[0];     // 0: kvSeqLen: [batch]
    int64_t blockTablesBatch = inTensorDescs.at(kBlockTablesIndex).shape.dims[0]; // 0: blockTables: [batch, BlockNum]
    if (seqlenBatch != blockTablesBatch || kvSeqLenBatch != blockTablesBatch) {
        ATB_LOG(ERROR) << "invalid seqlenBatch batch size" << seqlenBatch << ", kv: " << kvSeqLenBatch
                       << ", expect: " << blockTablesBatch;
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> SelfAttentionOperation::CreateRunner(Context &context) const
{
    ContextBase *contextBase = dynamic_cast<ContextBase *>(&context);
    if (!contextBase) {
        ATB_LOG(DEBUG) << "context cast to contextBase failed!";
        return nullptr;
    }
    if (GetSingleton<Config>().Is910B()) {
        if (param_.calcType == infer::SelfAttentionParam::PA_ENCODER) {
            return std::make_shared<SelfAttentionEncoderFusionOpsRunner>(param_);
        } else if (param_.calcType == infer::SelfAttentionParam::PREFIX_ENCODER) {
            return std::make_shared<SelfAttentionPrefixEncoderOpsRunner>(param_);
        } else if (param_.kvcacheCfg == atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS) {
            if (param_.inputLayout == atb::infer::InputLayout::TYPE_BNSD) {
                return std::make_shared<SelfAttentionFusionBypassOpsRunnerBNSD>(param_);
            } else {
                return std::make_shared<SelfAttentionFusionBypassOpsRunner>(param_);
            }
        } else {
            return std::make_shared<SelfAttentionFusionOpsRunner>(param_);
        }
    } else {
        if (param_.calcType == infer::SelfAttentionParam::PA_ENCODER) {
            RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_SELF_ATTENTION_PA_ENCODER);
            Runner *runner =
                pool.MallocRunner<SelfAttentionEncoderFusionOpsRunner910A, infer::SelfAttentionParam>(param_);
            return runner ? std::shared_ptr<Runner>(runner, [&pool](Runner *runner) { pool.FreeRunner(runner); }) :
                            std::make_shared<SelfAttentionEncoderFusionOpsRunner910A>(param_);
        } else if (param_.kvcacheCfg == atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS) {
            if (param_.inputLayout == atb::infer::InputLayout::TYPE_BNSD) {
                RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_SELF_ATTENTION_KV_BYPASS_BNSD);
                Runner *runner =
                    pool.MallocRunner<SelfAttentionFusionBypassOpsRunnerBNSD910A, infer::SelfAttentionParam>(param_);
                return runner ? std::shared_ptr<Runner>(runner, [&pool](Runner *runner) { pool.FreeRunner(runner); }) :
                                std::make_shared<SelfAttentionFusionBypassOpsRunnerBNSD910A>(param_);
            } else {
                RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_SELF_ATTENTION_KV_BYPASS);
                Runner *runner =
                    pool.MallocRunner<SelfAttentionFusionBypassOpsRunner910A, infer::SelfAttentionParam>(param_);
                return runner ? std::shared_ptr<Runner>(runner, [&pool](Runner *runner) { pool.FreeRunner(runner); }) :
                                std::make_shared<SelfAttentionFusionBypassOpsRunner910A>(param_);
            }
        }
        RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_SELF_ATTENTION);
        Runner *runner = pool.MallocRunner<SelfAttentionFusionOpsRunner910A, infer::SelfAttentionParam>(param_);
        return runner ? std::shared_ptr<Runner>(runner, [&pool](Runner *runner) { pool.FreeRunner(runner); }) :
                        std::make_shared<SelfAttentionFusionOpsRunner910A>(param_);
    }
    return std::shared_ptr<Runner>();
}

nlohmann::json SelfAttentionOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb
