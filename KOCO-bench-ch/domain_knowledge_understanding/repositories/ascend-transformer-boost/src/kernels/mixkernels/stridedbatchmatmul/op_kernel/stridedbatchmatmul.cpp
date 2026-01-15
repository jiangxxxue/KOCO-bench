/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "kernel_operator.h"
#include "lib/matrix/matmul/matmul.h"
#include "mixkernels/stridedbatchmatmul/tiling/tiling_data.h"

using namespace AscendC;
using namespace matmul;

__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM, int32_t batch, int32_t batchIdx)
{
    int32_t *ptr = reinterpret_cast<int32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ int32_t *>(tilingGM + sizeof(AtbOps::StridedBatchMatmulTilingData) +
        batch * sizeof(AtbOps::StridedBatchMatmulSampleTilingData));

    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(int32_t); i++, ptr++) {
        *ptr = *(tiling32 + sizeof(TCubeTiling) / sizeof(int32_t) * batchIdx + i);
    }

    return;
}

extern "C" __global__ __aicore__ void stridedbatchmatmul(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tilingGm)
{
    int32_t transA =  (*(const __gm__ int32_t *)(tilingGm));
    int32_t transB = (*(const __gm__ int32_t *)(tilingGm + sizeof(int32_t)));
    int32_t batch =  (*(const __gm__ int32_t *)(tilingGm + 2 * sizeof(int32_t)));
    int32_t headNum = (*(const __gm__ int32_t *)(tilingGm + 3 * sizeof(int32_t)));
    int32_t blockNum = (*(const __gm__ int32_t *)(tilingGm + 4 * sizeof(int32_t)));

    using A_T = half;
    using B_T = half;
    using C_T = half;
    using BiasT = half;
    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobal;

    // cube core cases, ignore vector core
    if (g_coreType == AIV) {
        return;
    }

    int numsPerCore = batch * headNum / blockNum;
    int tailNum = batch * headNum % blockNum;
    int currentNums;
    int gmNumsIdx;
    if (GetBlockIdx() < tailNum) {
        currentNums = numsPerCore + 1;
        gmNumsIdx = GetBlockIdx() * currentNums;
    } else {
        currentNums = numsPerCore;
        gmNumsIdx = tailNum + GetBlockIdx() * numsPerCore;
    }

    int batchIdx = gmNumsIdx / headNum;

    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm, batch, batchIdx);
    int32_t tilingOffset = sizeof(AtbOps::StridedBatchMatmulTilingData) +
        batchIdx * sizeof(AtbOps::StridedBatchMatmulSampleTilingData);
    int32_t batchOffsetA = (*(const __gm__ int32_t *)(tilingGm + tilingOffset));
    int32_t batchOffsetB = (*(const __gm__ int32_t *)(tilingGm + tilingOffset + sizeof(int32_t)));
    int32_t batchOffsetC = (*(const __gm__ int32_t *)(tilingGm + tilingOffset + 2 * sizeof(int32_t)));
    int32_t strideA = (*(const __gm__ int32_t *)(tilingGm + tilingOffset + 3 * sizeof(int32_t)));
    int32_t strideB = (*(const __gm__ int32_t *)(tilingGm + tilingOffset + 4 * sizeof(int32_t)));
    int32_t strideC = (*(const __gm__ int32_t *)(tilingGm + tilingOffset + 5 * sizeof(int32_t)));

    TPipe que;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(c), tiling.M * tiling.N);
    auto gmA = aGlobal;
    auto gmB = bGlobal;
    auto gmC = cGlobal;

    using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T, true>;
    using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T, true>;
    using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_T, true>;
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasT>;
    MatmulImpl<aType, bType, cType, biasType> mm;
    mm.SetSubBlockIdx(0);

    mm.Init(&tiling, &que);

    for (int j = 0; j < currentNums; j++) {
        int gmIdx = gmNumsIdx % numsPerCore;
        if (gmNumsIdx / headNum > batchIdx) {
            pipe_barrier(PIPE_ALL);
            batchIdx = gmNumsIdx / headNum;
            CopyTiling(&tiling, tilingGm, batch, batchIdx);
            tilingOffset = sizeof(AtbOps::StridedBatchMatmulTilingData) +
                batchIdx * sizeof(AtbOps::StridedBatchMatmulSampleTilingData);
            mm.Init(&tiling, &que);
            batchOffsetA = (*(const __gm__ int32_t *)(tilingGm + tilingOffset));
            batchOffsetB = (*(const __gm__ int32_t *)(tilingGm + tilingOffset + sizeof(int32_t)));
            batchOffsetC = (*(const __gm__ int32_t *)(tilingGm + tilingOffset + 2 * sizeof(int32_t)));
            strideA = (*(const __gm__ int32_t *)(tilingGm + tilingOffset + 3 * sizeof(int32_t)));
            strideB = (*(const __gm__ int32_t *)(tilingGm + tilingOffset + 4 * sizeof(int32_t)));
            strideC = (*(const __gm__ int32_t *)(tilingGm + tilingOffset + 5 * sizeof(int32_t)));
        }
        int offsetA = batchOffsetA + strideA * (gmNumsIdx - batchIdx * headNum);
        int offsetB = batchOffsetB + strideB * (gmNumsIdx - batchIdx * headNum);
        int offsetC = batchOffsetC + strideC * (gmNumsIdx - batchIdx * headNum);
        gmA = aGlobal[offsetA];
        gmB = bGlobal[offsetB];
        gmC = cGlobal[offsetC];
        mm.SetTensorA(gmA, bool(transA));
        mm.SetTensorB(gmB, bool(transB));
        mm.IterateAll(gmC);
        mm.End();
        gmNumsIdx += 1;
    }
}

