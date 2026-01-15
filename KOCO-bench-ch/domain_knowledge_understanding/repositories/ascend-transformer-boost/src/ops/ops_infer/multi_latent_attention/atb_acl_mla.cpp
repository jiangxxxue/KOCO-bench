/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/atb_acl.h"
#include "atb/utils/atb_acl_util.h"
#include "atb/operation/operation_base.h"

#ifdef __cplusplus
extern "C" {
#endif

const size_t g_MLAINTENSORNUMINT8NOMASK = 9;
const size_t g_MLAINTENSORNUMINT8MASK = 10;
const size_t g_MLAINTENSORNUMNOMASK = 7;
const size_t g_MLAINTENSORNUMMASK = 8;
const size_t g_MLAOUTTENSORNUMCALCRING = 2;
const size_t g_MLAOUTTENSORNUMNOCALCRING = 1;

atb::Status AtbMLAGetWorkspaceSize(const aclTensor *qNope, const aclTensor *qRope, const aclTensor *ctKV,
                                   const aclTensor *kRope, const aclTensor *blockTables, const aclTensor *contextLens,
                                   const aclTensor *mask, const aclTensor *qSeqLen, const aclTensor *qkDescale,
                                   const aclTensor *pvDescale, int32_t headNum, float qkScale, int32_t kvHeadNum,
                                   int maskType, int calcType, uint8_t cacheMode, aclTensor *attenOut, aclTensor *lse,
                                   uint64_t *workspaceSize, atb::Operation **op, atb::Context *context)
{
    atb::infer::MultiLatentAttentionParam param;
    param.headNum = headNum;
    param.qkScale = qkScale;
    param.kvHeadNum = kvHeadNum;
    param.maskType = atb::infer::MultiLatentAttentionParam::MaskType(maskType);
    param.calcType = atb::infer::MultiLatentAttentionParam::CalcType(calcType);
    param.cacheMode = atb::infer::MultiLatentAttentionParam::CacheMode(cacheMode);
    if (op != nullptr && *op == nullptr) {
        auto st = CreateOperation(param, op);
        if (st != atb::NO_ERROR) {
            ATB_LOG(ERROR) << "Create MLA Operation failed!";
            return st;
        }
    }
    atb::VariantPack pack;
    size_t i = 0;
    size_t counter = 0;
    if (param.cacheMode == atb::infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
        if (param.maskType == atb::infer::MultiLatentAttentionParam::MaskType::UNDEFINED) {
            pack.inTensors.resize(g_MLAINTENSORNUMINT8NOMASK);
            counter = g_MLAINTENSORNUMINT8NOMASK;
        } else {
            pack.inTensors.resize(g_MLAINTENSORNUMINT8MASK);
            counter = g_MLAINTENSORNUMINT8MASK;
        }
    } else {
        if (param.maskType == atb::infer::MultiLatentAttentionParam::MaskType::UNDEFINED) {
            pack.inTensors.resize(g_MLAINTENSORNUMNOMASK);
            counter = g_MLAINTENSORNUMNOMASK;
        } else {
            pack.inTensors.resize(g_MLAINTENSORNUMMASK);
            counter = g_MLAINTENSORNUMMASK;
        }
    }
    if (param.calcType != atb::infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC) {
        pack.inTensors.resize(counter - 1);
    }
    auto status = aclTensorToAtbTensor(qNope, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "qNope create failed!", return status);
    status = aclTensorToAtbTensor(qRope, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "qRope create failed!", return status);
    status = aclTensorToAtbTensor(ctKV, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "ctKV create failed!", return status);
    status = aclTensorToAtbTensor(kRope, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "kRope create failed!", return status);
    status = aclTensorToAtbTensor(blockTables, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "blockTables create failed!", return status);
    status = aclTensorToAtbTensorHost(contextLens, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "contextLens create failed!", return status);

    if (param.maskType != atb::infer::MultiLatentAttentionParam::MaskType::UNDEFINED) {
        status = aclTensorToAtbTensor(mask, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "mask create failed!", return status);
    }
    if (param.calcType == atb::infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC) {
        status = aclTensorToAtbTensorHost(qSeqLen, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "qSeqLen create failed!", return status);
    }
    if (param.cacheMode == atb::infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
        status = aclTensorToAtbTensor(qkDescale, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "qkDescale create failed!", return status);
        status = aclTensorToAtbTensor(pvDescale, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "pvDescale create failed!", return status);
    }
    i = 0;
    if (param.calcType != atb::infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_RING) {
        pack.outTensors.resize(g_MLAOUTTENSORNUMNOCALCRING);
        status = aclTensorToAtbTensor(attenOut, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "attenOut create failed!", return status);
    } else {
        pack.outTensors.resize(g_MLAOUTTENSORNUMCALCRING);
        status = aclTensorToAtbTensor(attenOut, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "calc_type_ring attenOut create failed!", return status);
        status = aclTensorToAtbTensor(lse, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "calc_type_ring lse create failed!", return status);
    }
    if (op == nullptr || *op == nullptr) {
        ATB_LOG(ERROR) << "AtbMLAGetWorkspaceSize opeartion pointer is nullptr!";
        return atb::ERROR_INVALID_OPERATION_ADDR;
    }
    atb::Status st = (*op)->Setup(pack, *workspaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbMLA Setup failed!", return st);
    return atb::NO_ERROR;
}

atb::Status AtbMLA(void *workSpcace, uint64_t workspaceSize, atb::Operation *op, atb::Context *context)
{
    atb::VariantPack pack;
    atb::Status st = op->Execute(pack, (uint8_t *)(workSpcace), workspaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbMLA Execute failed!", return st);
    return st;
}


atb::Status AtbMLAPreFillGetWorkspaceSize(const aclTensor *q, const aclTensor *qRope, const aclTensor *k,
                                          const aclTensor *kRope, const aclTensor *v, const aclTensor *qSeqLen,
                                          const aclTensor *kvSeqLen, const aclTensor *mask, int32_t headNum,
                                          float qkScale, int32_t kvHeadNum, int maskType, uint8_t cacheMode,
                                          aclTensor *attenOut, uint64_t *workspaceSize, atb::Operation **op,
                                          atb::Context *context)
{
    atb::infer::MultiLatentAttentionParam param;
    param.headNum = headNum;
    param.qkScale = qkScale;
    param.kvHeadNum = kvHeadNum;
    param.maskType = atb::infer::MultiLatentAttentionParam::MaskType(maskType);
    param.calcType = atb::infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_PREFILL;
    param.cacheMode = atb::infer::MultiLatentAttentionParam::CacheMode(cacheMode);
    if (op != nullptr && *op == nullptr) {
        auto st = CreateOperation(param, op);
        if (st != atb::NO_ERROR) {
            ATB_LOG(ERROR) << "Create MLA Operation prefill failed!";
            return st;
        }
    }
    atb::VariantPack pack;
    size_t i = 0;

    if (param.maskType == atb::infer::MultiLatentAttentionParam::MaskType::UNDEFINED) {
        pack.inTensors.resize(g_MLAINTENSORNUMNOMASK);
    } else {
        pack.inTensors.resize(g_MLAINTENSORNUMMASK);
    }

    auto status = aclTensorToAtbTensor(q, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "qNope create failed!", return status);
    status = aclTensorToAtbTensor(qRope, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "qRope create failed!", return status);
    status = aclTensorToAtbTensor(k, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "key create failed!", return status);
    status = aclTensorToAtbTensor(kRope, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "kRope create failed!", return status);
    status = aclTensorToAtbTensor(v, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "value create failed!", return status);
    status = aclTensorToAtbTensorHost(qSeqLen, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "qSeqLen create failed!", return status);
    status = aclTensorToAtbTensorHost(kvSeqLen, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "kvSeqLen create failed!", return status);

    if (param.maskType == atb::infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_MASK_FREE) {
        status = aclTensorToAtbTensor(mask, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "mask create failed!", return status);
    }

    pack.outTensors.resize(g_MLAOUTTENSORNUMNOCALCRING);
    status = aclTensorToAtbTensor(attenOut, &(pack.outTensors[0]));
    ATB_CHECK(status == atb::NO_ERROR, "attenOut create failed!", return status);

    if (op == nullptr || *op == nullptr) {
        ATB_LOG(ERROR) << "AtbMLAPreFillGetWorkspaceSize opeartion pointer is nullptr!";
        return atb::ERROR_INVALID_OPERATION_ADDR;
    }
    atb::Status st = (*op)->Setup(pack, *workspaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbMLAPreFill Setup failed!", return st);
    return atb::NO_ERROR;
}

atb::Status AtbMLAPreFill(void *workspace, uint64_t workspaceSize, atb::Operation *op, atb::Context *context)
{
    ATB_CHECK(op != nullptr, "AtbMLAPreFill expect op pointer not to be null!",
              return atb::ERROR_INVALID_OPERATION_ADDR);
    atb::VariantPack pack;
    atb::Status st = op->Execute(pack, (uint8_t *)(workspace), workspaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbMLAPreFill Execute failed!", return st);
    return st;
}

#ifdef __cplusplus
}
#endif
