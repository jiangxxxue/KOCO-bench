/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mixkernels/laser_attention/tiling/tiling_data.h"
#include "mixkernels/utils/common/kernel/kernel_utils.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "CubeForward.h"
#include "CubeForward_192.h"
#include "VectorForward.h"
#include "TransposeCustom.h"
#include "TransposeWithDtype.h"

using namespace AscendC;

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AtbOps::LaserAttentionTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->batchSize = (*(const __gm__ int32_t *)(p_tilingdata + 0));
    tilingdata->headNum = (*(const __gm__ int32_t *)(p_tilingdata + 4));
    tilingdata->seqSize = (*(const __gm__ int32_t *)(p_tilingdata + 8));
    tilingdata->headDim = (*(const __gm__ int32_t *)(p_tilingdata + 12));
    tilingdata->coreNumPerGroup = (*(const __gm__ int32_t *)(p_tilingdata + 16));
    tilingdata->coreGroupNum = (*(const __gm__ int32_t *)(p_tilingdata + 20));
    tilingdata->qSeqLength = (*(const __gm__ int32_t *)(p_tilingdata + 24));
    tilingdata->kSeqLength = (*(const __gm__ int32_t *)(p_tilingdata + 28));
    tilingdata->vSeqLength = (*(const __gm__ int32_t *)(p_tilingdata + 32));
    tilingdata->maskSeqLength = (*(const __gm__ int32_t *)(p_tilingdata + 36));
    tilingdata->scale = (*(const __gm__ float *)(p_tilingdata + 40));
    tilingdata->keepProb = (*(const __gm__ float *)(p_tilingdata + 44));
    tilingdata->preTokens = (*(const __gm__ int32_t *)(p_tilingdata + 48));
    tilingdata->nextTokens = (*(const __gm__ int32_t *)(p_tilingdata + 52));
    tilingdata->attenType = (*(const __gm__ int32_t *)(p_tilingdata + 56));
    tilingdata->sparseMode = (*(const __gm__ int32_t *)(p_tilingdata + 60));
    tilingdata->headGroupSize = (*(const __gm__ int32_t *)(p_tilingdata + 64));
    tilingdata->windowLen = (*(const __gm__ int32_t *)(p_tilingdata + 68));
    tilingdata->isTriangle = (*(const __gm__ int32_t *)(p_tilingdata + 72));
    tilingdata->isHighPrecision = (*(const __gm__ int32_t *)(p_tilingdata + 76));
    tilingdata->inputLayout = (*(const __gm__ int32_t *)(p_tilingdata + 80));
#else
    __ubuf__ uint8_t *tilingdata_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    int32_t tilingBlockNum = sizeof(AtbOps::LaserAttentionTilingData) / 32 + 1;
    copy_gm_to_ubuf(((__ubuf__ uint8_t *)tilingdata_in_ub), p_tilingdata, 0, 1, tilingBlockNum, 0, 0);
    pipe_barrier(PIPE_ALL);
    tilingdata->batchSize = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 0));
    pipe_barrier(PIPE_ALL);
#endif
}

extern "C" __global__ __aicore__ void
laser_attention(__gm__ uint8_t *__restrict__ ffts_addr, __gm__ uint8_t *__restrict__ q_gm,
                __gm__ uint8_t *__restrict__ k_gm, __gm__ uint8_t *__restrict__ v_gm,
                __gm__ uint8_t *__restrict__ alibi_mask_gm, __gm__ uint8_t *__restrict__ drop_mask_gm,
                __gm__ uint8_t *__restrict__ atten_mask_gm, __gm__ uint8_t *__restrict__ softmax_max_gm,
                __gm__ uint8_t *__restrict__ softmax_sum_gm, __gm__ uint8_t *__restrict__ attention_out_gm,
                __gm__ uint8_t *__restrict__ workspace, __gm__ uint8_t *__restrict__ tiling_para_gm)
{
    AtbOps::LaserAttentionTilingData tilingData;
    InitTilingData(tiling_para_gm, &(tilingData));
    const AtbOps::LaserAttentionTilingData *__restrict tiling_data = &tilingData;
    SetSysWorkspaceForce(workspace);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    set_ffts_base_addr((uint64_t)ffts_addr);

    int32_t batch = tiling_data->batchSize;
    int32_t headNum = tiling_data->headNum;
    int32_t coreNumPerGroup = tiling_data->coreNumPerGroup;
    int32_t coreGroupNum = tiling_data->coreGroupNum;
    int32_t baseS1 = tiling_data->qSeqLength;
    int32_t baseS2 = tiling_data->kSeqLength;
    int32_t headDim = tiling_data->headDim;
    int32_t headGroupSize = tiling_data->headGroupSize;
    int32_t isTriangle = tiling_data->isTriangle;
    int32_t sparseMode = tiling_data->sparseMode;
    int32_t windowLen = tiling_data->windowLen;
    int32_t maskSeqLength = tiling_data->maskSeqLength;
    int32_t isHighPrecision = 1;
    float scale = tiling_data->scale;
    int32_t inputLayout = tiling_data->inputLayout;

    __gm__ float *__restrict__ c_cube2 = (__gm__ float *__restrict__)attention_out_gm;
    __gm__ float *__restrict__ log_sum_gm = (__gm__ float *__restrict__)softmax_max_gm;
    __gm__ uint8_t *__restrict__ const_ones_gm = (__gm__ uint8_t *__restrict__)user;
    __gm__ float *__restrict__ const_zero_gm = (__gm__ float *__restrict__)(const_ones_gm + 128 * 128 * 2);
    __gm__ float *__restrict__ gm_rowsum = (__gm__ float *__restrict__)(softmax_sum_gm);
    __gm__ float *__restrict__ c_cube2_workspace =
        (__gm__ float *__restrict__)(const_zero_gm + batch * headNum * baseS1);
    __gm__ uint8_t *__restrict__ score_gm =
        (__gm__ uint8_t *__restrict__)(c_cube2_workspace + batch * headNum * baseS1 * 128);

    uint32_t col_size = sparseMode ? windowLen : baseS2;
    uint32_t score_size = coreGroupNum * (col_size + 128) * 128 * 2 * 4;
    uint32_t N2 = (headNum + headGroupSize - 1) / headGroupSize;
    uint32_t q_size = batch * headNum * baseS1 * headDim * 2;
    uint32_t kv_size = batch * N2 * baseS2 * 256 * 2;

    __gm__ uint8_t *__restrict__ gmq_out = (__gm__ uint8_t *__restrict__)(score_gm + score_size);
    __gm__ uint8_t *__restrict__ gmk_out = (__gm__ uint8_t *__restrict__)(gmq_out + q_size);
    __gm__ uint8_t *__restrict__ gmv_out = (__gm__ uint8_t *__restrict__)(gmk_out + kv_size);

#ifdef __DAV_C220_CUBE__
    if (headDim == 128) {
        CubeForward<__bf16, true, float> op;
        op.Init(q_gm, k_gm, v_gm, score_gm, c_cube2_workspace, const_ones_gm, gm_rowsum, coreNumPerGroup, coreGroupNum,
                batch, headNum, baseS1, baseS2, headDim, headGroupSize, isTriangle, sparseMode, windowLen);
        op.SetHighPrecision(isHighPrecision);
        op.Run();
    } else {
        CubeForward_192<__bf16, true, float> op;
        if (inputLayout != 3) {
            wait_flag_dev(AIV2AICFLAGID);
            op.Init(gmq_out, gmk_out, gmv_out, score_gm, c_cube2_workspace, const_ones_gm, gm_rowsum, coreNumPerGroup,
                    coreGroupNum, batch, headNum, baseS1, baseS2, headDim, headGroupSize, isTriangle, sparseMode,
                    windowLen);
        } else {
            op.Init(q_gm, k_gm, v_gm, score_gm, c_cube2_workspace, const_ones_gm, gm_rowsum, coreNumPerGroup,
                    coreGroupNum, batch, headNum, baseS1, baseS2, headDim, headGroupSize, isTriangle, sparseMode,
                    windowLen);
        }
        op.SetHighPrecision(isHighPrecision);
        op.Run();
    }
#elif __DAV_C220_VEC__
    VectorForward<__bf16, true, float> op;
    if (inputLayout != 3) {
        TransposeCustom<__bf16, true> op_in;
        op_in.Init(q_gm, gmq_out, batch, headNum, baseS1, 192, inputLayout);
        op_in.Run();
        op_in.Init(k_gm, gmk_out, batch, N2, baseS2, 256, inputLayout);
        op_in.Run();
        op_in.Init(v_gm, gmv_out, batch, N2, baseS2, 128, inputLayout);
        op_in.Run();

        uint64_t mode = 0;
        uint64_t config = 1 | (mode << 4) | (AIVFLAGID << 8);
        ffts_cross_core_sync(PIPE_MTE3, config);
        wait_flag_dev(AIVFLAGID);
        mode = 2;
        config = 1 | (mode << 4) | (AIV2AICFLAGID << 8);
        ffts_cross_core_sync(PIPE_MTE3, config);
    }
    op.Init(q_gm, k_gm, v_gm, atten_mask_gm, const_ones_gm, const_zero_gm, score_gm, c_cube2_workspace,
            attention_out_gm, log_sum_gm, gm_rowsum, baseS1, baseS2, headNum, batch, coreNumPerGroup, isTriangle,
            windowLen / BASE_BLOCK_SIDE_LEN, maskSeqLength, scale, inputLayout);
    op.SetHighPrecision(isHighPrecision);
    op.Run();
#endif

    if (inputLayout != 3) {
        __gm__ __bf16 *__restrict__ gm_O_out = (__gm__ __bf16 *__restrict__)attention_out_gm;
#ifdef __DAV_C220_VEC__
        TransposeWithDtype<__bf16, true> op_transpose;
        op_transpose.Init(c_cube2_workspace, gm_rowsum, gm_O_out, batch, headNum, baseS1, 128, inputLayout, 1);
        op_transpose.Run();
#endif
    }
}
