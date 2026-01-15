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

const size_t g_MLAPPINTENSORNUM = 24;
const size_t g_MLAPPOUTTENSORNUMCACHEMODE = 4;
const size_t g_MLAPPOUTTENSORNUM = 2;

atb::Status AtbMLAPreprocessGetWorkspaceSize(
    const aclTensor *input, const aclTensor *gamma0, const aclTensor *beta0, const aclTensor *quantScale0,
    const aclTensor *quantOffset0, const aclTensor *wdqkv, const aclTensor *deScale0, const aclTensor *bias0,
    const aclTensor *gamma1, const aclTensor *beta1, const aclTensor *quantScale1, const aclTensor *quantOffset1,
    const aclTensor *wuq, const aclTensor *deScale1, const aclTensor *bias1, const aclTensor *gamma2,
    const aclTensor *cos, const aclTensor *sin, const aclTensor *wuk, const aclTensor *kvCache,
    const aclTensor *kvCacheRope, const aclTensor *slotmapping, const aclTensor *ctkvScale, const aclTensor *qNopeScale,
    uint32_t wdqDim, uint32_t qRopeDim, uint32_t kRopeDim, float epsilon, uint32_t qRotaryCoeff, uint32_t kRotaryCoeff,
    bool transposeWdq, bool transposeWuq, bool transposeWuk, uint8_t cacheMode, uint16_t quantMode, aclTensor *qOut0,
    aclTensor *kvCacheOut0, aclTensor *qOut1, aclTensor *kvCacheOut1, uint64_t *workspaceSize, atb::Operation **op,
    atb::Context *context)
{
    atb::infer::MlaPreprocessParam param;
    param.wdqDim = wdqDim;
    param.qRopeDim = qRopeDim;
    param.kRopeDim = kRopeDim;
    param.epsilon = epsilon;
    param.qRotaryCoeff = static_cast<int32_t>(qRotaryCoeff);
    param.kRotaryCoeff = static_cast<int32_t>(kRotaryCoeff);
    param.transposeWdq = transposeWdq;
    param.transposeWuq = transposeWuq;
    param.transposeWuk = transposeWuk;
    param.cacheMode = atb::infer::MlaPreprocessParam::CacheMode(cacheMode);
    param.quantMode = atb::infer::MlaPreprocessParam::QuantMode(quantMode);

    if (op != nullptr && *op == nullptr) {
        auto st = CreateOperation(param, op);
        if (st != atb::NO_ERROR) {
            ATB_LOG(ERROR) << "Create MLAPreprocess Operation failed!";
            return st;
        }
    }
    atb::VariantPack pack;
    size_t i = 0;
    pack.inTensors.resize(g_MLAPPINTENSORNUM);
    auto status = aclTensorToAtbTensor(input, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "input create failed!", return status);
    status = aclTensorToAtbTensor(gamma0, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "gamma0 create failed!", return status);
    status = aclTensorToAtbTensor(beta0, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "beta0 create failed!", return status);
    if (param.quantMode == atb::infer::MlaPreprocessParam::QuantMode::PER_TENSOR_QUANT_ASYMM) {
        status = aclTensorToAtbTensor(quantScale0, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "quantScale0 create failed!", return status);
        status = aclTensorToAtbTensor(quantOffset0, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "quantOffset0 create failed!", return status);
    } else {
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
    }
    status = aclTensorToAtbTensor(wdqkv, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "wdqkv create failed!", return status);
    status = aclTensorToAtbTensor(deScale0, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "deScale0 create failed!", return status);
    if (param.quantMode != atb::infer::MlaPreprocessParam::QuantMode::PER_TOKEN_QUANT_SYMM &&
        param.quantMode != atb::infer::MlaPreprocessParam::QuantMode::UNQUANT) {
        status = aclTensorToAtbTensor(bias0, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "bias0 create failed!", return status);
    } else {
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
    }
    status = aclTensorToAtbTensor(gamma1, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "gamma1 create failed!", return status);
    status = aclTensorToAtbTensor(beta1, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "beta1 create failed!", return status);

    if (param.quantMode == atb::infer::MlaPreprocessParam::QuantMode::PER_TENSOR_QUANT_ASYMM) {
        status = aclTensorToAtbTensor(quantScale1, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "quantScale1 create failed!", return status);
        status = aclTensorToAtbTensor(quantOffset1, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "quantOffset1 create failed!", return status);
    } else {
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
    }
    status = aclTensorToAtbTensor(wuq, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "wuq create failed!", return status);
    status = aclTensorToAtbTensor(deScale1, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "deScale1 create failed!", return status);
    if (param.quantMode != atb::infer::MlaPreprocessParam::QuantMode::PER_TOKEN_QUANT_SYMM &&
        param.quantMode != atb::infer::MlaPreprocessParam::QuantMode::UNQUANT) {
        status = aclTensorToAtbTensor(bias1, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "bias1 create failed!", return status);
    } else {
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
    }
    status = aclTensorToAtbTensor(gamma2, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "gamma2 create failed!", return status);

    status = aclTensorToAtbTensor(cos, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "cos create failed!", return status);

    status = aclTensorToAtbTensor(sin, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "sin create failed!", return status);

    status = aclTensorToAtbTensor(wuk, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "wuk create failed!", return status);

    status = aclTensorToAtbTensor(kvCache, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "kvCache create failed!", return status);

    if (param.cacheMode != atb::infer::MlaPreprocessParam::CacheMode::KVCACHE) {
        status = aclTensorToAtbTensor(kvCacheRope, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "kvCacheRope create failed!", return status);
    } else {
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
    }
    status = aclTensorToAtbTensor(slotmapping, &(pack.inTensors[i++]));
    if (param.cacheMode == atb::infer::MlaPreprocessParam::CacheMode::INT8_NZCACHE) {
        status = aclTensorToAtbTensor(ctkvScale, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "ctkvScale create failed!", return status);
        status = aclTensorToAtbTensor(qNopeScale, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "qNopeScale create failed!", return status);
    } else {
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
        status = aclTensorToAtbTensor(nullptr, &(pack.inTensors[i++]));
    }

    i = 0;
    if (param.cacheMode != atb::infer::MlaPreprocessParam::CacheMode::KVCACHE) {
        pack.outTensors.resize(g_MLAPPOUTTENSORNUMCACHEMODE);
        status = aclTensorToAtbTensor(qOut0, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "qOut0 create failed!", return status);
        status = aclTensorToAtbTensor(kvCacheOut0, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "kvCacheOut0 create failed!", return status);
        status = aclTensorToAtbTensor(qOut1, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "qOut1 create failed!", return status);
        status = aclTensorToAtbTensor(kvCacheOut1, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "kvCacheOut1 create failed!", return status);
    } else {
        pack.outTensors.resize(g_MLAPPOUTTENSORNUM);
        status = aclTensorToAtbTensor(qOut0, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "qOut0 create failed!", return status);
        status = aclTensorToAtbTensor(kvCacheOut0, &(pack.outTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "kvCacheOut0 create failed!", return status);
    }
    if (op == nullptr || *op == nullptr) {
        ATB_LOG(ERROR) << "AtbMLAPreprocessGetWorkspaceSize opeartion pointer is nullptr!";
        return atb::ERROR_INVALID_OPERATION_ADDR;
    }
    atb::Status st = (*op)->Setup(pack, *workspaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbMLAPreprocess Setup failed!", return st);
    return atb::NO_ERROR;
}

atb::Status AtbMLAPreprocess(void *workspace, uint64_t workspaceSize, atb::Operation *op, atb::Context *context)
{
    ATB_CHECK(op != nullptr, "AtbMLAPreprocess expect op pointer not to be null!",
              return atb::ERROR_INVALID_OPERATION_ADDR);
    atb::VariantPack pack;
    atb::Status st = op->Execute(pack, (uint8_t *)(workspace), workspaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbMLAPreprocess Execute failed!", return st);
    return st;
}

#ifdef __cplusplus
}
#endif
