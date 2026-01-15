/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "transdata_ops_runner.h"
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/runner_util.h"
#include "atb/utils/tensor_util.h"

namespace atb {
TransdataOpsRunner::TransdataOpsRunner(const infer::TransdataParam &param)
    : OpsRunner("TransdataOpsRunner", RUNNER_TYPE_TRANSDATA), param_(param)
{
    ATB_LOG(INFO) << "TransdataOpsRunner::TransdataOpsRunner called, param_.transdataType:" << param_.transdataType;
    kernelGraph_.inTensors.resize(1);
    kernelGraph_.outTensors.resize(1);

    Mki::Tensor &operationInTensor = kernelGraph_.inTensors.at(0);
    Mki::Tensor &operationOutTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &transdataNode = kernelGraph_.nodes.at(0);

    Mki::SVector<int64_t> outCrops;
    TensorUtil::AtbSVector2OpsSVector(param_.outCrops, outCrops);
    switch (param_.transdataType) {
        case atb::infer::TransdataParam::FRACTAL_NZ_TO_ND:
            transdataNode.opDesc = {
                0, "TransdataOperation",
                AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND, outCrops})};
            break;
        case atb::infer::TransdataParam::ND_TO_FRACTAL_NZ:
            transdataNode.opDesc = {
                0, "TransdataOperation",
                AsdOps::OpParam::Transdata({AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ, outCrops})};
            break;
        default:
            ATB_LOG(WARN) << "Unsupported transdata type!";
            break;
    }
    transdataNode.inTensors = {&operationInTensor};
    transdataNode.outTensors = {&operationOutTensor};
    transdataNode.inTensorViewFuncs.resize(transdataNode.inTensors.size());
    transdataNode.inTensorViewFuncs.at(0) = [=](const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims) {
        InTensorViewFunc(oldDims, newDims);
    };
}

void TransdataOpsRunner::InTensorViewFunc(const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims)
{
    newDims = oldDims;
    if (param_.transdataType == atb::infer::TransdataParam::ND_TO_FRACTAL_NZ) {
        const int64_t dim2 = 2;
        if (oldDims.size() == dim2) {
            newDims = {1, oldDims.at(0), oldDims.at(1)};
        }
    }
}

TransdataOpsRunner::~TransdataOpsRunner() {}
} // namespace atb