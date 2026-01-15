/*

Copyright (c) 2025 Huawei Technologies Co., Ltd.
This file is a part of the CANN Open Software.
Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include "atb/atb_acl.h"
#include "atb/utils/atb_acl_util.h"
#include "atb/operation/operation_base.h"
#ifdef __cplusplus
extern "C" {
#endif

const size_t g_SELF_ATTENTION_PREFIX_ENCODER_INTENSOR_NUM = 6;
const size_t g_SELF_ATTENTION_PREFIX_ENCODER_OUTTENSOR_NUM = 1;

atb::Status AtbSelfAttentionPrefixEncoderGetWorkspaceSize(const aclTensor *query, const aclTensor *key,
                                                          const aclTensor *value, const aclTensor *blockTables,
                                                          const aclTensor *mask, const aclTensor *seqLen,
                                                          const aclTensor *kvSeqLen, const aclTensor *slopes,
                                                          int maskType, int32_t headNum, int32_t kvHeadNum,
                                                          float qkScale, aclTensor *attnOut, uint64_t *workspaceSize,
                                                          atb::Operation **op, atb::Context *context)
{
    atb::infer::SelfAttentionParam param;
    param.maskType = atb::infer::SelfAttentionParam::MaskType(maskType);
    param.headNum = headNum;
    param.kvHeadNum = kvHeadNum;
    param.qkScale = qkScale;
    param.quantType = atb::infer::SelfAttentionParam::QuantType::TYPE_QUANT_UNDEFINED;
    param.outDataType = ACL_DT_UNDEFINED;
    param.qScale = 1;
    param.batchRunStatusEnable = false;
    param.isTriuMask = 1;
    param.calcType = atb::infer::SelfAttentionParam::CalcType::PREFIX_ENCODER;
    param.kernelType = atb::infer::SelfAttentionParam::KernelType::KERNELTYPE_HIGH_PRECISION;
    param.clampType = atb::infer::SelfAttentionParam::ClampType::CLAMP_TYPE_UNDEFINED;
    param.clampMin = 0;
    param.clampMax = 0;
    param.kvcacheCfg = atb::infer::SelfAttentionParam::KvCacheCfg::K_CACHE_V_CACHE;
    param.scaleType = atb::infer::SelfAttentionParam::ScaleType::SCALE_TYPE_TOR;
    param.inputLayout = atb::infer::InputLayout::TYPE_BSND;

    if (op != nullptr && *op == nullptr) {
        auto st = CreateOperation(param, op);
        if (st != atb::NO_ERROR) {
            ATB_LOG(ERROR) << "Create SelfAttention Operation Prefix Encoder failed!";
            return st;
        }
    }
    atb::VariantPack pack;
    size_t index = 0;
    bool isAlibiMask = param.maskType == atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI_COMPRESS ||
                       param.maskType == atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI_COMPRESS_SQRT;
    if (isAlibiMask) {
        pack.inTensors.resize(g_SELF_ATTENTION_PREFIX_ENCODER_INTENSOR_NUM + 2); // 2: mask, slopes
    } else if (param.maskType == atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_CAUSAL_MASK) {
        pack.inTensors.resize(g_SELF_ATTENTION_PREFIX_ENCODER_INTENSOR_NUM); // mask auto-generated
    } else {
        pack.inTensors.resize(g_SELF_ATTENTION_PREFIX_ENCODER_INTENSOR_NUM + 1); // 1: mask
    }

    auto status = aclTensorToAtbTensor(query, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "query create failed!", return status);
    status = aclTensorToAtbTensor(key, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "key create failed!", return status);
    status = aclTensorToAtbTensor(value, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "value create failed!", return status);
    status = aclTensorToAtbTensor(blockTables, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "blockTables create failed!", return status);
    if (param.maskType != atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_CAUSAL_MASK) {
        status = aclTensorToAtbTensor(mask, &(pack.inTensors[index++]));
        ATB_CHECK(status == atb::NO_ERROR, "mask create failed!", return status);
    }
    status = aclTensorToAtbTensorHost(seqLen, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "seqLen create failed!", return status);
    status = aclTensorToAtbTensorHost(kvSeqLen, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "kvSeqLen create failed!", return status);
    if (isAlibiMask) {
        status = aclTensorToAtbTensor(slopes, &(pack.inTensors[index++]));
        ATB_CHECK(status == atb::NO_ERROR, "slopes create failed!", return status);
    }

    index = 0;
    pack.outTensors.resize(g_SELF_ATTENTION_PREFIX_ENCODER_OUTTENSOR_NUM);
    status = aclTensorToAtbTensor(attnOut, &(pack.outTensors[index]));
    ATB_CHECK(status == atb::NO_ERROR, "attnOut create failed!", return status);

    if (op == nullptr || *op == nullptr) {
        ATB_LOG(ERROR) << "AtbSelfAttentionPrefixEncoderGetWorkspaceSize opeartion pointer is nullptr!";
        return atb::ERROR_INVALID_OPERATION_ADDR;
    }
    status = (*op)->Setup(pack, *workspaceSize, context);
    ATB_CHECK(status == atb::NO_ERROR, "AtbSelfAttentionPrefixEncoder Setup failed!", return status);
    return atb::NO_ERROR;
}

atb::Status AtbSelfAttentionPrefixEncoder(void *workspace, uint64_t workspaceSize, atb::Operation *op,
                                          atb::Context *context)
{
    ATB_CHECK(op != nullptr, "AtbSelfAttentionPrefixEncoder expect op pointer not to be null!",
              return atb::ERROR_INVALID_OPERATION_ADDR);
    atb::VariantPack pack;
    atb::Status st = op->Execute(pack, (uint8_t *)(workspace), workspaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbSelfAttentionPrefixEncoder Execute failed!", return st);
    return st;
}

#ifdef __cplusplus
}
#endif