/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "stridedbatchmatmul_tiling.h"
#include <tiling/tiling_api.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "tiling_data.h"

using namespace matmul_tiling;
using namespace std;

namespace AtbOps {
Status StridedBatchMatmulOpParamCheck(const OpParam::StridedBatchMatmul &opParam)
{
    MKI_CHECK(opParam.batch > 0, "batch is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(opParam.headNum > 0, "headNum is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(static_cast<int64_t>(opParam.batch) * static_cast<int64_t>(opParam.headNum) < INT32_MAX, "batch * headNum is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    size_t b = static_cast<size_t>(opParam.batch);
    MKI_CHECK(opParam.m.size() == b, "size of m is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(opParam.k.size() == b, "size of k is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(opParam.n.size() == b, "size of n is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(opParam.lda.size() == b, "size of lda is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(opParam.ldb.size() == b, "size of ldb is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(opParam.ldc.size() == b, "size of ldc is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(opParam.strideA.size() == b, "size of strideA is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(opParam.strideB.size() == b, "size of strideB is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(opParam.strideC.size() == b, "size of strideC is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));

    return Status::OkStatus();
}

void StridedBatchMatmulSampleTiling(StridedBatchMatmulTilingData *tilingdata,
                                    const OpParam::StridedBatchMatmul &opParam)
{
    int batchOffsetA = 0;
    int batchOffsetB = 0;
    int batchOffsetC = 0;
    for (int i = 0; i < opParam.batch; i++) {
        auto sampleTilingDataPtr = reinterpret_cast<StridedBatchMatmulSampleTilingData *>(
            reinterpret_cast<uint8_t *>(tilingdata) + sizeof(StridedBatchMatmulTilingData) +
            i * sizeof(StridedBatchMatmulSampleTilingData));
        if (i == 0) {
            sampleTilingDataPtr->batchOffsetA = 0;
            sampleTilingDataPtr->batchOffsetB = 0;
            sampleTilingDataPtr->batchOffsetC = 0;
        } else {
            sampleTilingDataPtr->batchOffsetA = batchOffsetA + opParam.m[i - 1] * opParam.k[i - 1] * opParam.headNum;
            sampleTilingDataPtr->batchOffsetB = batchOffsetB + opParam.n[i - 1] * opParam.k[i - 1] * opParam.headNum;
            sampleTilingDataPtr->batchOffsetC = batchOffsetC + opParam.m[i - 1] * opParam.n[i - 1] * opParam.headNum;
            batchOffsetA = sampleTilingDataPtr->batchOffsetA;
            batchOffsetB = sampleTilingDataPtr->batchOffsetB;
            batchOffsetC = sampleTilingDataPtr->batchOffsetC;
        }
        sampleTilingDataPtr->strideA = opParam.strideA[i];
        sampleTilingDataPtr->strideB = opParam.strideB[i];
        sampleTilingDataPtr->strideC = opParam.strideC[i];

        MKI_LOG(INFO) << "StridedBatchMatmul Sample Tiling Data: batchOffsetA " << sampleTilingDataPtr->batchOffsetA
            << ", batchOffsetB " << sampleTilingDataPtr->batchOffsetB
            << ", batchOffsetC " << sampleTilingDataPtr->batchOffsetC
            << ", strideA " << sampleTilingDataPtr->strideA
            << ", strideB " << sampleTilingDataPtr->strideB
            << ", strideC " << sampleTilingDataPtr->strideC;
    }
}

void TCubeTiling(StridedBatchMatmulTilingData *tilingdata, const OpParam::StridedBatchMatmul &opParam)
{
    int isBias = 0;
    for (int i = 0; i < opParam.batch; i++) {
        MatmulApiTiling tilingApi;
        optiling::TCubeTiling tCubeTiling;
        tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, opParam.transA);
        tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, opParam.transB);
        tilingApi.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
        tilingApi.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);

        int orgM = (opParam.transA != 0) ? opParam.lda[i] : opParam.m[i];
        int orgK = (opParam.transA != 0) ? opParam.k[i] : opParam.lda[i];
        int orgN = (opParam.transB != 0) ? opParam.n[i] : opParam.ldb[i];

        tilingApi.SetShape(opParam.m[i], opParam.n[i], opParam.k[i]);
        tilingApi.SetOrgShape(orgM, orgN, orgK);
        tilingApi.SetBias(bool(isBias));
        tilingApi.SetBufferSpace(-1, -1, -1);

        int64_t res = tilingApi.GetTiling(tCubeTiling);
        if (res == -1) {
            MKI_LOG(ERROR) << "gen tiling failed";
            return;
        }
        uint32_t tilingSize = tCubeTiling.GetDataSize();
        tCubeTiling.SaveToBuffer(reinterpret_cast<uint8_t *>(tilingdata) + sizeof(StridedBatchMatmulTilingData) +
            opParam.batch * sizeof(StridedBatchMatmulSampleTilingData) + i * tilingSize, tilingSize);
    }
}

Status StridedBatchMatmulTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto opParam = AnyCast<OpParam::StridedBatchMatmul>(launchParam.GetParam());
    Status ret = StridedBatchMatmulOpParamCheck(opParam);
    OP_TILING_CHECK_STATUS_RETURN(ret);

    void *tilingdata = kernelInfo.GetTilingHostAddr();
    StridedBatchMatmulTilingData *tilingDataPtr =
        reinterpret_cast<StridedBatchMatmulTilingData *>(tilingdata);
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE));

    tilingDataPtr->transA = opParam.transA;
    tilingDataPtr->transB = opParam.transB;
    tilingDataPtr->batch = opParam.batch;
    tilingDataPtr->headNum = opParam.headNum;
    int32_t coreNum = static_cast<int32_t>(Mki::PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));
    if (coreNum > opParam.batch * opParam.headNum) {
        tilingDataPtr->blockNum = opParam.batch * opParam.headNum;
    } else {
        tilingDataPtr->blockNum = coreNum;
    }
    MKI_LOG(INFO) << "StridedBatchMatmul Tiling Data: transA " << tilingDataPtr->transA
        << ", transB " << tilingDataPtr->transB << ", batch " << tilingDataPtr->batch
        << ", headNum " << tilingDataPtr->headNum << ", blockNum " << tilingDataPtr->blockNum;

    StridedBatchMatmulSampleTiling(tilingDataPtr, opParam);
    TCubeTiling(tilingDataPtr, opParam);

    kernelInfo.SetBlockDim(tilingDataPtr->blockNum);
    return Status::OkStatus();
}
} // namespace AtbOps
