/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "paged_cache_load_ops_runner.h"
#include "atb/utils/log.h"

namespace atb {
PagedCacheLoadOpsRunner::PagedCacheLoadOpsRunner(const infer::PagedCacheLoadParam &param)
    : OpsRunner("PagedCacheLoadOpsRunner", RUNNER_TYPE_PAGED_CACHE_LOAD), param_(param)
{
    const std::size_t intensorSize = 7;
    const std::size_t outtensorSize = 2;
    kernelGraph_.inTensors.resize(intensorSize);
    kernelGraph_.outTensors.resize(outtensorSize);

    size_t inTensorStart = 0;
    Mki::Tensor &keyCacheTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &valueCacheTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &blockTablesTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &contextLens = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &key = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &value = kernelGraph_.inTensors.at(inTensorStart++);

    if (param_.kvCacheCfg == infer::PagedCacheLoadParam::KvCacheCfg::K_CACHE_V_CACHE_NZ) {
        Mki::Tensor &seq_starts = kernelGraph_.inTensors.at(3);

        size_t outTensorStart = 0;
        Mki::Tensor &outKeyTensor = kernelGraph_.outTensors.at(outTensorStart++);
        Mki::Tensor &outValueTensor = kernelGraph_.outTensors.at(outTensorStart++);

        kernelGraph_.nodes.resize(1);
        auto &pagedCacheLoadNode = kernelGraph_.nodes.at(0);
        AtbOps::OpParam::PagedCacheLoad pagedCacheLoadParam;

        pagedCacheLoadParam.type = AtbOps::OpParam::PagedCacheLoad::PAGED_CACHE_LOAD_NZ;
        pagedCacheLoadParam.cuSeqLens = param_.isSeqLensCumsumMode;
        pagedCacheLoadParam.hasSeqStarts = param_.hasSeqStarts;

        pagedCacheLoadNode.opDesc = {0, "PagedCacheLoadOperation", pagedCacheLoadParam};
        pagedCacheLoadNode.inTensors = {&keyCacheTensor, &valueCacheTensor, &blockTablesTensor, &contextLens, &key, &value, &seq_starts};
        pagedCacheLoadNode.outTensors = {&outKeyTensor, &outValueTensor};
        pagedCacheLoadNode.inferShapePreFunc = [](Mki::LaunchParam &launchParam) {
            for (size_t i = 0; i < launchParam.GetInTensorCount(); i++) {
                if (i == 0 || i == 1) { // 2, 3: intensor index
                    launchParam.GetInTensor(i).desc.format = Mki::TENSOR_FORMAT_FRACTAL_NZ;
                } else {
                    launchParam.GetInTensor(i).desc.format = Mki::TENSOR_FORMAT_ND;
                }
            }
        };
    } else {
        Mki::Tensor &seq_starts = kernelGraph_.inTensors.at(inTensorStart++);

        size_t outTensorStart = 0;
        Mki::Tensor &outKeyTensor = kernelGraph_.outTensors.at(outTensorStart++);
        Mki::Tensor &outValueTensor = kernelGraph_.outTensors.at(outTensorStart++);

        kernelGraph_.nodes.resize(1);
        auto &pagedCacheLoadNode = kernelGraph_.nodes.at(0);
        AtbOps::OpParam::PagedCacheLoad pagedCacheLoadParam;
        pagedCacheLoadParam.type = AtbOps::OpParam::PagedCacheLoad::PAGED_CACHE_LOAD_ND;
        pagedCacheLoadParam.cuSeqLens = param_.isSeqLensCumsumMode;
        pagedCacheLoadParam.hasSeqStarts = param_.hasSeqStarts;

        pagedCacheLoadNode.opDesc = {0, "PagedCacheLoadOperation", pagedCacheLoadParam};
        pagedCacheLoadNode.inTensors = {&keyCacheTensor, &valueCacheTensor, &blockTablesTensor, &contextLens, &key, &value, &seq_starts};
        pagedCacheLoadNode.outTensors = {&outKeyTensor, &outValueTensor};
        pagedCacheLoadNode.inferShapePreFunc = [](Mki::LaunchParam &launchParam) {
            for (size_t i = 0; i < launchParam.GetInTensorCount(); i++) {
                launchParam.GetInTensor(i).desc.format = Mki::TENSOR_FORMAT_ND;
            }
        };
    }
}

PagedCacheLoadOpsRunner::~PagedCacheLoadOpsRunner() {}

} // namespace atb