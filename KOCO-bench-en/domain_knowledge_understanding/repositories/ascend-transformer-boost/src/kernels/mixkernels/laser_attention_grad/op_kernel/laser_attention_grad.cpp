/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mixkernels/laser_attention_grad/tiling/tiling_data.h"
#include "mixkernels/utils/common/kernel/kernel_utils.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "cube_backward.h"
#include "vector_backward.h"
#include "cube_backward_band_op.h"
#include "cube_backward_band_op_192.h"
#include "vector_backward_band_op.h"
#include "lag_post.h"
#include "TransposeCustom.h"
#include "TransposeWithDtype.h"
#include "TransposeGrad.h"

using namespace AscendC;

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata,
                                      AtbOps::LaserAttentionGradTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->batchSize = (*(const __gm__ int32_t *)(p_tilingdata + 0));
    tilingdata->headNum = (*(const __gm__ int32_t *)(p_tilingdata + 4));
    tilingdata->seqSize = (*(const __gm__ int32_t *)(p_tilingdata + 8));
    tilingdata->headDim = (*(const __gm__ int32_t *)(p_tilingdata + 12));
    tilingdata->blockDim = (*(const __gm__ int32_t *)(p_tilingdata + 16));
    tilingdata->blockNumPerCore = (*(const __gm__ int32_t *)(p_tilingdata + 20));
    tilingdata->isTriangle = (*(const __gm__ int32_t *)(p_tilingdata + 24));
    tilingdata->attenType = (*(const __gm__ int32_t *)(p_tilingdata + 28));
    tilingdata->sparseMode = (*(const __gm__ int32_t *)(p_tilingdata + 32));
    tilingdata->headGroupSize = (*(const __gm__ int32_t *)(p_tilingdata + 36));
    tilingdata->windowLen = (*(const __gm__ int32_t *)(p_tilingdata + 40));
    tilingdata->qSeqLength = (*(const __gm__ int32_t *)(p_tilingdata + 44));
    tilingdata->kSeqLength = (*(const __gm__ int32_t *)(p_tilingdata + 48));
    tilingdata->vSeqLength = (*(const __gm__ int32_t *)(p_tilingdata + 52));
    tilingdata->isHighPrecision = (*(const __gm__ int32_t *)(p_tilingdata + 56));
    tilingdata->scale = (*(const __gm__ float *)(p_tilingdata + 60));
    tilingdata->keepProb = (*(const __gm__ float *)(p_tilingdata + 64));
    tilingdata->preTokens = (*(const __gm__ int32_t *)(p_tilingdata + 68));
    tilingdata->nextTokens = (*(const __gm__ int32_t *)(p_tilingdata + 72));
    tilingdata->maskSeqLength = (*(const __gm__ int32_t *)(p_tilingdata + 76));
    tilingdata->queryShape = (*(const __gm__ int64_t *)(p_tilingdata + 80));
    tilingdata->keyShape = (*(const __gm__ int64_t *)(p_tilingdata + 88));
    tilingdata->valueShape = (*(const __gm__ int64_t *)(p_tilingdata + 96));
    tilingdata->postBaseNum = (*(const __gm__ int64_t *)(p_tilingdata + 104));
    tilingdata->postDqFrontCoreNum = (*(const __gm__ int64_t *)(p_tilingdata + 112));
    tilingdata->postDqTailCoreNum = (*(const __gm__ int64_t *)(p_tilingdata + 120));
    tilingdata->postDqFrontDataNum = (*(const __gm__ int64_t *)(p_tilingdata + 128));
    tilingdata->postDqTailDataNum = (*(const __gm__ int64_t *)(p_tilingdata + 136));
    tilingdata->postDkFrontCoreNum = (*(const __gm__ int64_t *)(p_tilingdata + 144));
    tilingdata->postDkTailCoreNum = (*(const __gm__ int64_t *)(p_tilingdata + 152));
    tilingdata->postDkFrontDataNum = (*(const __gm__ int64_t *)(p_tilingdata + 160));
    tilingdata->postDkTailDataNum = (*(const __gm__ int64_t *)(p_tilingdata + 168));
    tilingdata->postDvFrontCoreNum = (*(const __gm__ int64_t *)(p_tilingdata + 176));
    tilingdata->postDvTailCoreNum = (*(const __gm__ int64_t *)(p_tilingdata + 184));
    tilingdata->postDvFrontDataNum = (*(const __gm__ int64_t *)(p_tilingdata + 192));
    tilingdata->postDvTailDataNum = (*(const __gm__ int64_t *)(p_tilingdata + 200));
    tilingdata->outputWorkspaceOffset = (*(const __gm__ int64_t *)(p_tilingdata + 208));
    tilingdata->inputLayout = (*(const __gm__ int32_t *)(p_tilingdata + 216));

#else
    __ubuf__ uint8_t *tilingdata_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    int32_t tilingBlockNum = sizeof(AtbOps::LaserAttentionGradTilingData) / 32 + 1;
    copy_gm_to_ubuf(((__ubuf__ uint8_t *)tilingdata_in_ub), p_tilingdata, 0, 1, tilingBlockNum, 0, 0);
    pipe_barrier(PIPE_ALL);
    tilingdata->batchSize = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 0));
    pipe_barrier(PIPE_ALL);
#endif
}

extern "C" __global__ __aicore__ void
laser_attention_grad(__gm__ uint8_t *__restrict__ ffts_addr, __gm__ uint8_t *__restrict__ query_gm,
                     __gm__ uint8_t *__restrict__ key_gm, __gm__ uint8_t *__restrict__ value_gm,
                     __gm__ uint8_t *__restrict__ attention_out_grad_gm, __gm__ uint8_t *__restrict__ alibi_mask_gm,
                     __gm__ uint8_t *__restrict__ drop_mask_gm, __gm__ uint8_t *__restrict__ atten_mask_gm,
                     __gm__ uint8_t *__restrict__ softmax_max_gm, __gm__ uint8_t *__restrict__ softmax_sum_gm,
                     __gm__ uint8_t *__restrict__ attention_in_gm, __gm__ uint8_t *__restrict__ query_grad_gm,
                     __gm__ uint8_t *__restrict__ key_grad_gm, __gm__ uint8_t *__restrict__ value_grad_gm,
                     __gm__ uint8_t *__restrict__ workspace, __gm__ uint8_t *__restrict__ tiling_para_gm)

{
    AtbOps::LaserAttentionGradTilingData tilingData;
    InitTilingData(tiling_para_gm, &(tilingData));
    const AtbOps::LaserAttentionGradTilingData *__restrict tiling_data = &tilingData;
    SetSysWorkspaceForce(workspace);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    set_ffts_base_addr((uint64_t)ffts_addr);

    int32_t batchSize = tiling_data->batchSize;
    int32_t headNum = tiling_data->headNum;
    int32_t seqSize = tiling_data->seqSize;
    int32_t qSize = tiling_data->qSeqLength;
    int32_t kSize = tiling_data->kSeqLength;
    int32_t isTriangle = tiling_data->isTriangle;
    int32_t headDim = tiling_data->headDim;
    int32_t gqaGroupSize = tiling_data->headGroupSize;
    int32_t sparseMode = tiling_data->sparseMode;
    int32_t windowLength = tiling_data->windowLen;

    int32_t postBaseNum = tiling_data->postBaseNum;
    int32_t postDqFrontCoreNum = tiling_data->postBaseNum;
    int32_t postDqFrontDataNum = tiling_data->postDqFrontDataNum;
    int32_t postDqTailDataNum = tiling_data->postDqTailDataNum;
    int32_t postDkFrontCoreNum = tiling_data->postDkFrontCoreNum;
    int32_t postDkFrontDataNum = tiling_data->postDkFrontDataNum;
    int32_t postDkTailDataNum = tiling_data->postDkTailDataNum;
    int32_t postDvFrontCoreNum = tiling_data->postDvFrontCoreNum;
    int32_t postDvFrontDataNum = tiling_data->postDvFrontDataNum;
    int32_t postDvTailDataNum = tiling_data->postDvTailDataNum;

    int32_t blockDim = tiling_data->blockDim;
    int32_t blockNumPerCore = tiling_data->blockNumPerCore;
    int32_t maskSeqLength = tiling_data->maskSeqLength;
    int32_t dpOffset = blockDim * blockNumPerCore * 128 * 128 * 2;
    float scale = tiling_data->scale;
    int32_t isTri = isTriangle;
    int32_t inputLayout = tiling_data->inputLayout;
    int32_t gqaHead = (headNum + gqaGroupSize - 1) / gqaGroupSize;

    int64_t dkOffset = tiling_data->queryShape;
    int64_t dvOffset = dkOffset + tiling_data->keyShape;
    int64_t sOffset = dvOffset + tiling_data->valueShape;

    if (TILING_KEY_IS(2)) {
        using TYPE = __bf16;
        __gm__ TYPE *__restrict__ gm_Q = (__gm__ TYPE *__restrict__)query_gm;
        __gm__ TYPE *__restrict__ gm_K = (__gm__ TYPE *__restrict__)key_gm;
        __gm__ TYPE *__restrict__ gm_V = (__gm__ TYPE *__restrict__)value_gm;
        __gm__ TYPE *__restrict__ gm_dO = (__gm__ TYPE *__restrict__)attention_out_grad_gm;
        __gm__ TYPE *__restrict__ gm_dQ = (__gm__ TYPE *__restrict__)query_grad_gm;
        __gm__ TYPE *__restrict__ gm_dK = (__gm__ TYPE *__restrict__)key_grad_gm;
        __gm__ TYPE *__restrict__ gm_dV = (__gm__ TYPE *__restrict__)value_grad_gm;
        __gm__ TYPE *__restrict__ gm_O = (__gm__ TYPE *__restrict__)attention_in_gm;
        __gm__ float *__restrict__ gm_rowmax = (__gm__ float *__restrict__)softmax_max_gm;
        __gm__ float *__restrict__ gm_rowsum = (__gm__ float *__restrict__)softmax_sum_gm;
        __gm__ TYPE *__restrict__ gm_mask = (__gm__ TYPE *__restrict__)atten_mask_gm;
        __gm__ float *__restrict__ workspace_dQ = (__gm__ float *__restrict__)user;
        __gm__ float *__restrict__ workspace_dK = (__gm__ float *__restrict__)(workspace_dQ + dkOffset);
        __gm__ float *__restrict__ workspace_dV = (__gm__ float *__restrict__)(workspace_dQ + dvOffset);
        __gm__ float *__restrict__ gm_S = (__gm__ float *__restrict__)(workspace_dQ + sOffset);
        __gm__ float *__restrict__ gm_dP = (__gm__ float *__restrict__)(gm_S + dpOffset);

#ifdef __DAV_C220_CUBE__
        CUBE_BACKWARD_FP32_OP::CubeBackward<TYPE, true> cube;
        cube.Init(gm_Q, gm_K, gm_V, gm_dO, gm_dP, gm_S, workspace_dQ, workspace_dK, workspace_dV, isTri, qSize, kSize,
                  batchSize, headNum, gqaGroupSize, headDim, sparseMode, windowLength, blockNumPerCore);
        cube.Run_mix();
        SyncAll<false>();
#elif __DAV_C220_VEC__
        VEC_BACKWARD_FP32_OP::VectorBackward<TYPE, true> Vector;
        Vector.Init(gm_dO, gm_O, gm_S, gm_rowmax, gm_rowsum, gm_dP, gm_mask, isTri, qSize, kSize, batchSize, headNum,
                    headDim, blockNumPerCore, sparseMode, windowLength, maskSeqLength, scale);
        Vector.Run();

        LAG_POST::LagPost<TYPE, true> post;
        post.Init(workspace_dQ, workspace_dK, workspace_dV, gm_dQ, gm_dK, gm_dV, postBaseNum, postDqFrontCoreNum,
                  postDqFrontDataNum, postDqTailDataNum, postDkFrontCoreNum, postDkFrontDataNum, postDkTailDataNum,
                  postDvFrontCoreNum, postDvFrontDataNum, postDvTailDataNum, scale);
        SyncAll<false>();
        post.Process();
#endif
    } else if (TILING_KEY_IS(3)) {
        using TYPE = __bf16;
        __gm__ TYPE *__restrict__ gm_Q = (__gm__ TYPE *__restrict__)query_gm;
        __gm__ TYPE *__restrict__ gm_K = (__gm__ TYPE *__restrict__)key_gm;
        __gm__ TYPE *__restrict__ gm_V = (__gm__ TYPE *__restrict__)value_gm;
        __gm__ TYPE *__restrict__ gm_dO = (__gm__ TYPE *__restrict__)attention_out_grad_gm;
        __gm__ TYPE *__restrict__ gm_dQ = (__gm__ TYPE *__restrict__)query_grad_gm;
        __gm__ TYPE *__restrict__ gm_dK = (__gm__ TYPE *__restrict__)key_grad_gm;
        __gm__ TYPE *__restrict__ gm_dV = (__gm__ TYPE *__restrict__)value_grad_gm;
        __gm__ TYPE *__restrict__ gm_O = (__gm__ TYPE *__restrict__)attention_in_gm;
        __gm__ float *__restrict__ gm_rowmax = (__gm__ float *__restrict__)softmax_max_gm;
        __gm__ float *__restrict__ gm_rowsum = (__gm__ float *__restrict__)softmax_sum_gm;
        __gm__ TYPE *__restrict__ gm_mask = (__gm__ TYPE *__restrict__)atten_mask_gm;
        __gm__ float *__restrict__ workspace_dQ = (__gm__ float *__restrict__)user;
        __gm__ float *__restrict__ workspace_dK = (__gm__ float *__restrict__)(workspace_dQ + dkOffset);
        __gm__ float *__restrict__ workspace_dV = (__gm__ float *__restrict__)(workspace_dQ + dvOffset);
        __gm__ float *__restrict__ gm_S = (__gm__ float *__restrict__)(workspace_dQ + sOffset);
        __gm__ float *__restrict__ gm_dP = (__gm__ float *__restrict__)(gm_S + dpOffset);

        uint32_t dpSize = blockDim * blockNumPerCore * 128 * 128 * 2;
        uint32_t gmQSize = batchSize * 192 * headNum * qSize;
        uint32_t gmKSize = batchSize * 256 * gqaHead * kSize;
        uint32_t gmVSize = batchSize * 128 * gqaHead * kSize;
        uint32_t gmOSize = batchSize * 128 * headNum * qSize;

        __gm__ TYPE *__restrict__ gm_Q_trans =
            (__gm__ TYPE *__restrict__)(gm_dP + blockDim * blockNumPerCore * 128 * 128 * 2);
        __gm__ TYPE *__restrict__ gm_K_trans = (__gm__ TYPE *__restrict__)(gm_Q_trans + gmQSize);
        __gm__ TYPE *__restrict__ gm_V_trans = (__gm__ TYPE *__restrict__)(gm_K_trans + gmKSize);
        __gm__ TYPE *__restrict__ gm_O_trans = (__gm__ TYPE *__restrict__)(gm_V_trans + gmVSize);
        __gm__ TYPE *__restrict__ gm_dO_trans = (__gm__ TYPE *__restrict__)(gm_O_trans + gmOSize);

#ifdef __DAV_C220_CUBE__
        if (headDim == 128) {
            CUBE_BACK_BAND_OP::CubeBackwardBandOp<TYPE, true> cube;
            cube.Init(gm_Q, gm_K, gm_V, gm_dO, gm_dP, gm_S, workspace_dQ, workspace_dK, workspace_dV, isTri, qSize,
                      kSize, batchSize, headNum, gqaGroupSize, headDim, sparseMode, windowLength, blockNumPerCore);
            cube.Run_op();
            SyncAll<false>();
        } else {
            CUBE_BACK_BAND_OP_192::CubeBackwardBandOp192<TYPE, true> cube;
            if (inputLayout != 3) {
                wait_flag_dev(AIV2AICFLAGID);
                cube.Init(gm_Q_trans, gm_K_trans, gm_V_trans, gm_dO_trans, gm_dP, gm_S, workspace_dQ, workspace_dK,
                          workspace_dV, isTri, qSize, kSize, batchSize, headNum, gqaGroupSize, headDim, sparseMode,
                          windowLength, blockNumPerCore);
            } else {
                cube.Init(gm_Q, gm_K, gm_V, gm_dO, gm_dP, gm_S, workspace_dQ, workspace_dK, workspace_dV, isTri, qSize,
                          kSize, batchSize, headNum, gqaGroupSize, headDim, sparseMode, windowLength, blockNumPerCore);
            }
            cube.Run_op();
            SyncAll<false>();
        }
#elif __DAV_C220_VEC__
        VEC_BACKWARD_BAND_OP::VectorBackwardBandOp<TYPE, true> Vector;

        TransposeGrad<TYPE, true> transpose_grad;
        transpose_grad.Init(gm_Q, gm_K, gm_V, gm_dO, gm_O, gm_Q_trans, gm_K_trans, gm_V_trans, gm_dO_trans, gm_O_trans,
                            batchSize, headNum, qSize, kSize, inputLayout, gqaHead);
        transpose_grad.Run();

        if (inputLayout != 3) {
            Vector.Init(gm_dO_trans, gm_O_trans, gm_S, gm_rowmax, gm_rowsum, gm_dP, gm_mask, isTri, qSize, kSize,
                        batchSize, headNum, headDim, blockNumPerCore, sparseMode, windowLength, maskSeqLength, scale);
        } else {
            Vector.Init(gm_dO, gm_O, gm_S, gm_rowmax, gm_rowsum, gm_dP, gm_mask, isTri, qSize, kSize, batchSize,
                        headNum, headDim, blockNumPerCore, sparseMode, windowLength, maskSeqLength, scale);
        }

        Vector.Run_op();

        if (inputLayout == 3) {
            LAG_POST::LagPost<TYPE, true> post;
            post.Init(workspace_dQ, workspace_dK, workspace_dV, gm_dQ, gm_dK, gm_dV, postBaseNum, postDqFrontCoreNum,
                      postDqFrontDataNum, postDqTailDataNum, postDkFrontCoreNum, postDkFrontDataNum, postDkTailDataNum,
                      postDvFrontCoreNum, postDvFrontDataNum, postDvTailDataNum, scale);
            SyncAll<false>();
            post.Process();
        } else {
            TransposeWithDtype<TYPE, true> op_transpose;
            SyncAll<false>();
            op_transpose.Init(workspace_dQ, nullptr, gm_dQ, batchSize, headNum, qSize, 192, inputLayout, 0, true,
                              scale);
            op_transpose.Run();

            op_transpose.Init(workspace_dK, nullptr, gm_dK, batchSize, gqaHead, kSize, 256, inputLayout, 0, true,
                              scale);
            op_transpose.Run();

            op_transpose.Init(workspace_dV, nullptr, gm_dV, batchSize, gqaHead, kSize, 128, inputLayout, 0, false,
                              scale);
            op_transpose.Run();
            // SyncAll<false>();
        }

#endif
    }
}
