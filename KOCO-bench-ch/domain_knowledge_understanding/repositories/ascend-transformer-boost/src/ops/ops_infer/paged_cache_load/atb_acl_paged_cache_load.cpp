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

const size_t g_PAGED_CACHE_LOAD_INTENSOR_NUM = 6;
const size_t g_PAGED_CACHE_LOAD_OUTTENSOR_NUM = 2;

atb::Status AtbPagedCacheLoadGetWorkspaceSize(const aclTensor *keyCache, const aclTensor *valueCache,
                                              const aclTensor *blockTables, const aclTensor *contextLens,
                                              const aclTensor *key, const aclTensor *value, const aclTensor *seqStarts,
                                              int8_t kvCacheCfg, bool isSeqLensCumsumType, bool hasSeqStarts,
                                              uint64_t *workspaceSize, atb::Operation **op, atb::Context *context)
{
    atb::infer::PagedCacheLoadParam param;
    param.kvCacheCfg = atb::infer::PagedCacheLoadParam::KvCacheCfg(kvCacheCfg);
    param.isSeqLensCumsumMode = isSeqLensCumsumType;
    param.hasSeqStarts = hasSeqStarts;

    if (op != nullptr && *op == nullptr) {
        auto st = CreateOperation(param, op);
        if (st != atb::NO_ERROR) {
            ATB_LOG(ERROR) << "Create PagedCacheLoad Operation failed!";
            return st;
        }
    }
    atb::VariantPack pack;
    size_t i = 0;
    if (param.hasSeqStarts) {
        pack.inTensors.resize(g_PAGED_CACHE_LOAD_INTENSOR_NUM + 1);
    } else {
        pack.inTensors.resize(g_PAGED_CACHE_LOAD_INTENSOR_NUM);
    }

    auto status = aclTensorToAtbTensor(keyCache, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "keyCache create failed!", return status);
    status = aclTensorToAtbTensor(valueCache, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "valueCache create failed!", return status);
    status = aclTensorToAtbTensor(blockTables, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "blockTables create failed!", return status);
    status = aclTensorToAtbTensor(contextLens, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "contextLens create failed!", return status);
    status = aclTensorToAtbTensor(key, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "key create failed!", return status);
    status = aclTensorToAtbTensor(value, &(pack.inTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "value create failed!", return status);
    if (param.hasSeqStarts) {
        status = aclTensorToAtbTensor(seqStarts, &(pack.inTensors[i++]));
        ATB_CHECK(status == atb::NO_ERROR, "seqStarts create failed!", return status);
    }

    i = 0;
    pack.outTensors.resize(g_PAGED_CACHE_LOAD_OUTTENSOR_NUM);
    status = aclTensorToAtbTensor(key, &(pack.outTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "key create failed!", return status);
    status = aclTensorToAtbTensor(value, &(pack.outTensors[i++]));
    ATB_CHECK(status == atb::NO_ERROR, "value create failed!", return status);

    if (op == nullptr || *op == nullptr) {
        ATB_LOG(ERROR) << "AtbPagedCacheLoadGetWorkspaceSize opeartion pointer is nullptr!";
        return atb::ERROR_INVALID_OPERATION_ADDR;
    }
    atb::Status st = (*op)->Setup(pack, *workspaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbPagedCacheLoad Setup failed!", return st);
    return atb::NO_ERROR;
}


atb::Status AtbPagedCacheLoad(void *workspace, uint64_t workspaceSize, atb::Operation *op, atb::Context *context)
{
    ATB_CHECK(op != nullptr, "AtbPagedCacheLoad expect op pointer not to be null!",
              return atb::ERROR_INVALID_OPERATION_ADDR);
    atb::VariantPack pack;
    atb::Status st = op->Execute(pack, (uint8_t *)(workspace), workspaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbPagedCacheLoad Execute failed!", return st);
    return st;
}

#ifdef __cplusplus
}
#endif
