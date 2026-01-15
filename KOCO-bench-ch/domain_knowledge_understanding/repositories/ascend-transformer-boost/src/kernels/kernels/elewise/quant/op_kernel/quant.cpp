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
#include "kernels/utils/kernel/kernel_utils.h"
#include "kernels/elewise/quant/quant_tiling/tiling_data.h"
static constexpr int32_t BUFFER_NUM = 1; // tensor num for each queue
static constexpr uint32_t DATA_BYTE = 2;
static constexpr uint32_t BLOCK_NUMEL = 16;

using AscendC::HardEvent;

class KernelQuantization {
public:
    __aicore__ inline KernelQuantization() {}

    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
    {
        if (y == 0) {
            return UINT32_MAX;
        }
        return (x + y - 1) / y;
    }

    __aicore__ inline uint32_t ROUND_UP(uint32_t x) { return (x + BLOCK_NUMEL - 1) / BLOCK_NUMEL * BLOCK_NUMEL; }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y) { return x < y ? x : y; }
    __aicore__ inline uint32_t MAX(uint32_t x, uint32_t y) { return x > y ? x : y; }

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *z, uint32_t num_core_, uint32_t num_Last_dim_,
                                uint32_t num_first_dim_, uint32_t nl_first_dim_per_core_,
                                uint32_t l_first_dim_per_core_, uint32_t first_dim_per_times_, uint32_t scale_,
                                uint32_t offset_, int32_t quant_min_)
    {
        // 一次搬入多行
        num_core = num_core_;
        num_last_dim = num_Last_dim_;
        num_first_dim = num_first_dim_;
        nl_first_dim_per_core = nl_first_dim_per_core_;
        l_first_dim_per_core = l_first_dim_per_core_;
        first_dim_per_times = first_dim_per_times_;
        quant_min = static_cast<half>(quant_min_);

        input_scale = *reinterpret_cast<float *>(&scale_);
        input_offset = *reinterpret_cast<int32_t *>(&offset_);

        if (AscendC::GetBlockIdx() != num_core - 1) {
            row_work = nl_first_dim_per_core;
            row_step = first_dim_per_times;
        } else {
            row_work = l_first_dim_per_core;
            row_step = MIN(first_dim_per_times, row_work);
        }
        row_tail_ = (row_work % first_dim_per_times == 0) ? first_dim_per_times : (row_work % first_dim_per_times);
        gm_offset_ = nl_first_dim_per_core * num_last_dim;
        x_gm.SetGlobalBuffer((__gm__ half *)x + AscendC::GetBlockIdx() * gm_offset_);
        z_gm.SetGlobalBuffer((__gm__ int8_t *)z + AscendC::GetBlockIdx() * gm_offset_);
        pipe.InitBuffer(x_que, BUFFER_NUM, row_step * ROUND_UP(num_last_dim) * DATA_BYTE);
        pipe.InitBuffer(z_que, BUFFER_NUM, row_step * ROUND_UP(num_last_dim) * sizeof(int8_t));
    }

    __aicore__ inline void Process()
    {
        uint32_t move_cnt = CEIL_DIV(row_work, row_step); // 一个核需要做多少次
        for (uint64_t i = 0; i < move_cnt; ++i) {
            if (i < move_cnt - 1) {
                CopyIn(i, row_step * num_last_dim);

                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

                Compute(row_step);

                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

                CopyOut(i, row_step * num_last_dim);
            } else {
                CopyIn(i, row_tail_ * num_last_dim);

                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

                Compute(row_tail_);

                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);

                CopyOut(i, row_tail_ * num_last_dim);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t proc_id, int32_t size)
    {
        // alloc tensor from queue memory
        AscendC::LocalTensor<half> x_local = x_que.AllocTensor<half>();
        uint64_t offset = proc_id * row_step * num_last_dim;
        DataCopy(x_local, x_gm[offset], size);
        x_que.EnQue(x_local);
    }

    __aicore__ inline void Compute(int32_t nums)
    {
        AscendC::LocalTensor<half> x_local = x_que.DeQue<half>();
        AscendC::LocalTensor<int8_t> z_local = z_que.AllocTensor<int8_t>();

        for (int32_t rid = 0; rid < nums; ++rid) {
            AscendC::PipeBarrier<PIPE_V>();
            Muls(x_local[rid * num_last_dim], x_local[rid * num_last_dim], static_cast<half>(input_scale),
                 num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Adds(x_local[rid * num_last_dim], x_local[rid * num_last_dim], static_cast<half>(input_offset),
                 num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            CastFromF16ToI8(z_local[rid * num_last_dim], x_local[rid * num_last_dim], quant_min, num_last_dim);
        }

        z_que.EnQue(z_local);
        x_que.FreeTensor(x_local);
    }

    __aicore__ inline void CopyOut(uint64_t proc_id, int32_t size)
    {
        AscendC::LocalTensor<int8_t> z = z_que.DeQue<int8_t>();
        uint64_t offset = proc_id * row_step * num_last_dim; // 单核一次总共做了多少。
        DataCopy(z_gm[offset], z, size);
        z_que.FreeTensor(z);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> z_que;
    AscendC::GlobalTensor<half> x_gm;
    AscendC::GlobalTensor<int8_t> z_gm;
    float input_scale{0};
    int32_t input_offset{0};
    uint32_t num_core{0};       // 一共激活多少AICORE
    uint32_t num_first_dim{0};  // 输入的列数
    uint32_t num_last_dim{0};   // 输入的列数
    uint32_t row_work{0};       // 每个AICORE需要计算多少行
    uint32_t row_step{0};       // 除最后一次，每次搬入多少行
    uint32_t row_tail_{0};      // 最后一次搬入多少行数据
    uint64_t gm_offset_{0};     // GM数据起始位置偏移量
    uint32_t nl_first_dim_per_core{0};
    uint32_t l_first_dim_per_core{0};
    uint32_t first_dim_per_times{0};
    half quant_min = -128;
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::QuantF16TilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numCore = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->numLastDim = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->numFirstDim = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->nlFirstdimPerCore = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->lFirstdimPerCore = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->firstDimPerTimes = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->inputScale = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
    tilingdata->inputOffset = (*(const __gm__ uint32_t *)(p_tilingdata + 28));
    tilingdata->quantMin = (*(const __gm__ float *)(p_tilingdata + 32));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::QuantF16TilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numCore = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 0));
    tilingdata->numLastDim = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 4));
    tilingdata->numFirstDim = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 8));
    tilingdata->nlFirstdimPerCore = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 12));
    tilingdata->lFirstdimPerCore = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 16));
    tilingdata->firstDimPerTimes = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 20));
    tilingdata->inputScale = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 24));
    tilingdata->inputOffset = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 28));
    tilingdata->quantMin = (*(__ubuf__ float *)((__ubuf__ uint8_t *)tilingdata_in_ub + 32));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                                       \
    AsdOps::QuantF16TilingData tiling_data;                                                                            \
    InitTilingData(tiling_arg, &(tiling_data))

extern "C" __global__ __aicore__ void quant(GM_ADDR x, GM_ADDR z, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelQuantization op;
    op.Init(x, z, tiling_data.numCore, tiling_data.numLastDim, tiling_data.numFirstDim,
            tiling_data.nlFirstdimPerCore, tiling_data.lFirstdimPerCore, tiling_data.firstDimPerTimes,
            tiling_data.inputScale, tiling_data.inputOffset, tiling_data.quantMin);
    op.Process();
}