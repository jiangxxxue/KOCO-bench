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

const size_t g_RING_MLA_INTENSOR_NUM = 7;
const size_t g_RING_MLA_OUTTENSOR_NUM = 2;

atb::Status AtbRingMLAGetWorkspaceSize(const aclTensor *querySplit1, const aclTensor *querySplit2,
                                       const aclTensor *keySplit1, const aclTensor *keySplit2, const aclTensor *value,
                                       const aclTensor *mask, const aclTensor *seqLen, const aclTensor *prevOut,
                                       const aclTensor *prevLse, int32_t headNum, int32_t kvHeadNum, float qkScale,
                                       int kernelType, int maskType, int inputLayout, int calcType, aclTensor *output,
                                       aclTensor *softmaxLse, uint64_t *workspaceSize, atb::Operation **op,
                                       atb::Context *context)
{
    atb::infer::RingMLAParam param;
    param.headNum = headNum;
    param.kvHeadNum = kvHeadNum;
    param.qkScale = qkScale;
    param.kernelType = atb::infer::RingMLAParam::KernelType(kernelType);
    param.maskType = atb::infer::RingMLAParam::MaskType(maskType);
    param.inputLayout = atb::infer::InputLayout(inputLayout);
    param.calcType = atb::infer::RingMLAParam::CalcType(calcType);
    if (op != nullptr && *op == nullptr) {
        auto st = CreateOperation(param, op);
        if (st != atb::NO_ERROR) {
            ATB_LOG(ERROR) << "Create RingMLA Operation failed!";
            return st;
        }
    }
    atb::VariantPack pack;
    size_t index = 0;
    if (param.calcType == atb::infer::RingMLAParam::CalcType::CALC_TYPE_DEFAULT) {
        pack.inTensors.resize(g_RING_MLA_INTENSOR_NUM + 2); // 2: prevOut, prevLse
    } else {
        pack.inTensors.resize(g_RING_MLA_INTENSOR_NUM);
    }

    auto status = aclTensorToAtbTensor(querySplit1, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "querySplit1 create failed!", return status);
    status = aclTensorToAtbTensor(querySplit2, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "querySplit2 create failed!", return status);
    status = aclTensorToAtbTensor(keySplit1, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "keySplit1 create failed!", return status);
    status = aclTensorToAtbTensor(keySplit2, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "keySplit2 create failed!", return status);
    status = aclTensorToAtbTensor(value, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "value create failed!", return status);
    status = aclTensorToAtbTensor(mask, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "mask create failed!", return status);
    status = aclTensorToAtbTensorHost(seqLen, &(pack.inTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "seqLen create failed!", return status);
    if (param.calcType == atb::infer::RingMLAParam::CalcType::CALC_TYPE_DEFAULT) {
        status = aclTensorToAtbTensor(prevOut, &(pack.inTensors[index++]));
        ATB_CHECK(status == atb::NO_ERROR, "prevOut create failed!", return status);
        status = aclTensorToAtbTensor(prevLse, &(pack.inTensors[index++]));
        ATB_CHECK(status == atb::NO_ERROR, "prevLse create failed!", return status);
    }

    index = 0;
    pack.outTensors.resize(g_RING_MLA_OUTTENSOR_NUM);
    status = aclTensorToAtbTensor(output, &(pack.outTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "output create failed!", return status);
    status = aclTensorToAtbTensor(softmaxLse, &(pack.outTensors[index++]));
    ATB_CHECK(status == atb::NO_ERROR, "softmaxLse create failed!", return status);
    if (op == nullptr || *op == nullptr) {
        ATB_LOG(ERROR) << "AtbRingMLAGetWorkspaceSize opeartion pointer is nullptr!";
        return atb::ERROR_INVALID_OPERATION_ADDR;
    }
    status = (*op)->Setup(pack, *workspaceSize, context);
    ATB_CHECK(status == atb::NO_ERROR, "AtbRingMLA Setup failed!", return status);
    return atb::NO_ERROR;
}

atb::Status AtbRingMLA(void *workspace, uint64_t workspaceSize, atb::Operation *op, atb::Context *context)
{
    ATB_CHECK(op != nullptr, "AtbRingMLA expect op pointer not to be null!", return atb::ERROR_INVALID_OPERATION_ADDR);
    atb::VariantPack pack;
    atb::Status st = op->Execute(pack, (uint8_t *)(workspace), workspaceSize, context);
    ATB_CHECK(st == atb::NO_ERROR, "AtbRingMLA Execute failed!", return st);
    return st;
}

#ifdef __cplusplus
}
#endif
