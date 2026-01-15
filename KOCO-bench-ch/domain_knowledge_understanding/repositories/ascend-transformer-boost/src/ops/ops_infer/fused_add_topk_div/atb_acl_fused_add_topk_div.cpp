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
const size_t g_FUSED_ADD_TOPK_INTENSOR_NUM = 2;
const size_t g_FUSED_ADD_TOPK_OUTTENSOR_NUM = 2;

atb::Status AtbFusedAddTopkDivGetWorkspaceSize(const aclTensor *x, const aclTensor *addNum, const aclTensor *mappingNum,
                                               const aclTensor *mappingTable, uint32_t groupNum, uint32_t groupTopk,
                                               uint32_t n, uint32_t k, int activationType, bool isNorm, float scale,
                                               bool enableExpertMapping, aclTensor *y, aclTensor *indices,
                                               uint64_t *workspaceSize, atb::Operation **op, atb::Context *context)
{
    atb::infer::FusedAddTopkDivParam param;
    param.groupNum = groupNum;
    param.groupTopk = groupTopk;
    param.n = n;
    param.k = k;
    param.isNorm = isNorm;
    param.scale = scale;
    param.enableExpertMapping = enableExpertMapping;
    param.activationType = atb::infer::ActivationType(activationType);

    if (op != nullptr && *op == nullptr) {
        auto st = CreateOperation(param, op);
        if (st != atb::NO_ERROR) {
            ATB_LOG(ERROR) << "Create FusedAddTopkDiv Operation failed!";
            return st;
        }
    }
    atb::VariantPack pack;

    size_t intensorNum = g_FUSED_ADD_TOPK_INTENSOR_NUM;
    if (enableExpertMapping) {
        intensorNum += 2; // 2: mappingNum, mappingTable
    }
    pack.inTensors.resize(intensorNum);
    size_t index = 0;
    auto status = aclTensorToAtbTensor(x, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "x create failed!", return status);
    status = aclTensorToAtbTensor(addNum, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "addNum create failed!", return status);
    if (enableExpertMapping) {
        status = aclTensorToAtbTensor(mappingNum, &(pack.inTensors[index++]));
        ATB_CHECK(status == atb::NO_ERROR, "mappingNum create failed!", return status);
        status = aclTensorToAtbTensor(mappingTable, &(pack.inTensors[index++]));
        ATB_CHECK(status == atb::NO_ERROR, "mappingTable create failed!", return status);
    }

    index = 0;
    pack.outTensors.resize(g_FUSED_ADD_TOPK_OUTTENSOR_NUM);
    status = aclTensorToAtbTensor(y, &(pack.outTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "y create failed!", return status);
    status = aclTensorToAtbTensor(indices, &(pack.outTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "indices create failed!", return status);
    if (op == nullptr || *op == nullptr) {
        ATB_LOG(ERROR) << "AtbFusedAddTopkDivGetWorkspaceSize opeartion pointer is nullptr!";
        return atb::ERROR_INVALID_OPERATION_ADDR;
    }
    status = (*op)->Setup(pack, *workspaceSize, context);
    ATB_CHECK(status == atb::NO_ERROR, "AtbFusedAddTopkDiv Setup failed!", return status);
    return atb::NO_ERROR;
}

atb::Status AtbFusedAddTopkDiv(void *workspace, uint64_t workspaceSize, atb::Operation *op, atb::Context *context)
{
    ATB_CHECK(op != nullptr, "AtbFusedAddTopkDiv expect op pointer not to be null!",
              return atb::ERROR_INVALID_OPERATION_ADDR);
    atb::VariantPack pack;
    atb::Status st = op->Execute(pack, (uint8_t *)(workspace), workspaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbFusedAddTopkDiv Execute failed!", return st);
    return st;
}

#ifdef __cplusplus
}
#endif
