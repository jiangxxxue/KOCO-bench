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
#include "kernels/norm/common/op_kernel/common_quant.h"
#include "kernels/norm/layernormquant/tiling/tiling_data.h"

static constexpr uint32_t BUFFER_NUM = 1;
static constexpr uint32_t BLOCK_NUMEL = 16;
static constexpr uint32_t TAIL_BUFFER_SIZE = 32;
static constexpr uint32_t MAX_UB_SIZE_NUM = 98304;
static constexpr uint32_t DATA_BYTE = 2;
static constexpr uint32_t INT8_DATA_BYTE = 1;
static constexpr float QUANT_MIN = -128;

using AscendC::HardEvent;

template <typename T, bool FastComputeMode = false>
class KernelLayerNormQuant {
public:
    __aicore__ inline KernelLayerNormQuant() {}
    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
    {
        if (y > 0) {
            return (x + y - 1) / y;
        }
        return 0;
    }
    __aicore__ inline uint32_t ROUND_UP(uint32_t x) { return (x + BLOCK_NUMEL - 1) / BLOCK_NUMEL * BLOCK_NUMEL; }
    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y) { return x < y ? x : y; }
    __aicore__ inline uint32_t MAX(uint32_t x, uint32_t y) { return x > y ? x : y; }

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *gamma, __gm__ uint8_t *beta, __gm__ uint8_t *scale,
                                __gm__ uint8_t *offset, __gm__ uint8_t *z, AsdOps::LayerNormQuantTilingData tilingData)
    {
        half_num = 2;
        num_core = tilingData.numCore;
        num_last_dim = tilingData.numLastDim;
        num_first_dim = tilingData.numFirstDim;
        nl_first_dim_per_core = tilingData.nlFirstdimPerCore;
        l_first_dim_per_core = tilingData.lFirstdimPerCore;
        first_dim_per_times = tilingData.firstDimPerTimes;
        aveNum = *reinterpret_cast<float *>(&tilingData.aveStr);
        eps = *reinterpret_cast<float *>(&tilingData.epsStr);

        if (AscendC::GetBlockIdx() != num_core - 1) {
            row_work = nl_first_dim_per_core;
            row_step = first_dim_per_times;
        } else {
            row_work = l_first_dim_per_core;
            row_step = MIN(first_dim_per_times, row_work);
        }

        slice_size = tilingData.sliceSize;
        num_slice = tilingData.sliceNum;
        tail_slice_size = tilingData.tailSliceSize;
        row_tail_ = (row_work % first_dim_per_times == 0) ? first_dim_per_times : (row_work % first_dim_per_times);
        gm_offset_ = static_cast<uint64_t>(nl_first_dim_per_core) * num_last_dim;

        x_gm.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gm_offset_);
        z_gm.SetGlobalBuffer((__gm__ int8_t *)z + AscendC::GetBlockIdx() * gm_offset_);
        gamma_gm.SetGlobalBuffer((__gm__ T *)gamma);
        beta_gm.SetGlobalBuffer((__gm__ T *)beta);

        pipe.InitBuffer(x_que, BUFFER_NUM, row_step * ROUND_UP(slice_size) * DATA_BYTE);
        pipe.InitBuffer(z_que, BUFFER_NUM, row_step * ROUND_UP(slice_size) * INT8_DATA_BYTE);
        pipe.InitBuffer(beta_que, BUFFER_NUM, 1 * ROUND_UP(slice_size) * DATA_BYTE);
        pipe.InitBuffer(gamma_que, BUFFER_NUM, 1 * ROUND_UP(slice_size) * DATA_BYTE);
        pipe.InitBuffer(ave_buf, BLOCK_NUMEL * DATA_BYTE);
        pipe.InitBuffer(var_buf, BLOCK_NUMEL * DATA_BYTE);
        pipe.InitBuffer(x_buf_fp32, half_num * ROUND_UP(slice_size) * DATA_BYTE);
        pipe.InitBuffer(y_buf_fp32, half_num * ROUND_UP(slice_size) * DATA_BYTE);
        pipe.InitBuffer(z_buf_fp32, half_num * ROUND_UP(slice_size) * DATA_BYTE);

        GetScaleAndOffset(scale, offset);
    }

    __aicore__ inline void Process()
    {
        if constexpr (FastComputeMode) {
            FastCompute();
        } else {
            SliceCompute();
        }
    }

private:
    __aicore__ inline void FastCompute()
    {
        uint32_t move_cnt = CEIL_DIV(row_work, row_step);
        for (uint32_t i = 0; i < move_cnt; ++i) {
            if (i < move_cnt - 1) {
                FastCopyIn(i, row_step * num_last_dim);

                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                FastPrecisionCompute(row_step);
                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

                FastCopyOut(i, row_step * num_last_dim);
            } else {
                FastCopyIn(i, row_tail_ * num_last_dim);

                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                FastPrecisionCompute(row_tail_);
                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

                FastCopyOut(i, row_tail_ * num_last_dim);
            }
        }
    }

    __aicore__ inline void FastCopyIn(uint64_t proc_id, uint64_t size)
    {
        AscendC::LocalTensor<T> x_local = x_que.AllocTensor<T>();
        AscendC::LocalTensor<T> beta_local = beta_que.AllocTensor<T>();
        AscendC::LocalTensor<T> gamma_local = gamma_que.AllocTensor<T>();
        uint64_t offset = proc_id * row_step * num_last_dim;
        DataCopy(x_local, x_gm[offset], size);
        DataCopy(beta_local, beta_gm[0], num_last_dim);
        DataCopy(gamma_local, gamma_gm[0], num_last_dim);
        x_que.EnQue(x_local);
        beta_que.EnQue(beta_local);
        gamma_que.EnQue(gamma_local);
    }

    __aicore__ inline void FastPrecisionCompute(uint64_t nums)
    {
        AscendC::LocalTensor<T> x_local = x_que.DeQue<T>();
        AscendC::LocalTensor<T> beta_local = beta_que.DeQue<T>();
        AscendC::LocalTensor<T> gamma_local = gamma_que.DeQue<T>();
        AscendC::LocalTensor<float> ave_local = ave_buf.Get<float>();
        AscendC::LocalTensor<float> var_local = var_buf.Get<float>();
        AscendC::LocalTensor<int8_t> z_local = z_que.AllocTensor<int8_t>();
        AscendC::LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        for (uint64_t rid = 0; rid < nums; ++rid) {
            Cast(x_local_fp32, x_local[rid * num_last_dim], AscendC::RoundMode::CAST_NONE, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Duplicate(y_local_fp32, aveNum, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(z_local_fp32, x_local_fp32, y_local_fp32, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            ReduceSum(ave_local, z_local_fp32, y_local_fp32, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
            float ave_local_temp = ave_local.GetValue(0);
            AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
            AscendC::PipeBarrier<PIPE_V>();
            Duplicate(y_local_fp32, ave_local_temp, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Sub(z_local_fp32, x_local_fp32, y_local_fp32, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(x_local_fp32, z_local_fp32, z_local_fp32, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Muls(y_local_fp32, x_local_fp32, aveNum, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            ReduceSum(var_local, y_local_fp32, x_local_fp32, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Adds(var_local, var_local, eps, BLOCK_NUMEL / half_num);
            AscendC::PipeBarrier<PIPE_V>();
            Sqrt(var_local, var_local, BLOCK_NUMEL / half_num);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID2);
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID2);
            float var_local_temp = var_local.GetValue(0);
            AscendC::SetFlag<HardEvent::S_V>(EVENT_ID2);
            AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID2);
            AscendC::PipeBarrier<PIPE_V>();
            Duplicate(x_local_fp32, var_local_temp, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Duplicate(y_local_fp32, (float)1, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Div(y_local_fp32, y_local_fp32, x_local_fp32, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(x_local_fp32, z_local_fp32, y_local_fp32, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(z_local_fp32, gamma_local, AscendC::RoundMode::CAST_NONE, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(y_local_fp32, x_local_fp32, z_local_fp32, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(x_local_fp32, beta_local, AscendC::RoundMode::CAST_NONE, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Add(z_local_fp32, y_local_fp32, x_local_fp32, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Muls(z_local_fp32, z_local_fp32, inputScale, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Adds(z_local_fp32, z_local_fp32, inputOffset, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<half> tmpFp16 = x_buf_fp32.Get<half>();
            CastFrom32To16(tmpFp16, z_local_fp32, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            CastFromF16ToI8(z_local[rid * num_last_dim], tmpFp16, QUANT_MIN, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
        }
        x_que.FreeTensor(x_local);
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void FastCopyOut(uint64_t proc_id, uint64_t size)
    {
        AscendC::LocalTensor<int8_t> z = z_que.DeQue<int8_t>();
        uint64_t offset = proc_id * row_step * num_last_dim;
        DataCopy(z_gm[offset], z, size);
        z_que.FreeTensor(z);
    }

    __aicore__ inline void SliceCompute()
    {
        for (uint32_t pid = 0; pid < row_work; pid++) {
            uint32_t row_offset = pid * num_last_dim;

            float mean = ComputeMean(row_offset);
            float variance = ComputeVariance(row_offset, mean);

            for (uint32_t sid = 0; sid < num_slice; sid++) {
                uint32_t slice_offset = 0;
                uint64_t col_offset = 0;
                uint32_t eleNum = 0;
                GetSliceOffsetAndSize(row_offset, sid, slice_offset, col_offset, eleNum);

                SliceCopyIn(slice_offset, col_offset, eleNum);

                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                SlicePrecisionCompute(eleNum, mean, variance);
                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

                SliceCopyOut(col_offset, eleNum);
            }
        }
    }

    __aicore__ inline void GetScaleAndOffset(__gm__ uint8_t *scale, __gm__ uint8_t *offset)
    {
        AscendC::GlobalTensor<T> gm_s;
        gm_s.SetGlobalBuffer((__gm__ T *)scale);
        AscendC::LocalTensor<T> tmpFp16 = x_buf_fp32.Get<T>();
        DataCopy(tmpFp16, gm_s, BLOCK_SIZE / sizeof(T));
        if constexpr (AscendC::IsSameType<T, half>::value) {
            AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            inputScale = 1 / (float)(tmpFp16.GetValue(0));
        } else {
            AscendC::LocalTensor<float> tmpFp32 = y_buf_fp32.Get<float>();
            AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(tmpFp32, tmpFp16, AscendC::RoundMode::CAST_NONE, 1);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
            inputScale = 1 / (float)(tmpFp32.GetValue(0));
        }
        AscendC::GlobalTensor<int8_t> gm_o;
        gm_o.SetGlobalBuffer((__gm__ int8_t *)offset);
        AscendC::LocalTensor<int8_t> tmpInt8 = x_buf_fp32.Get<int8_t>();
        DataCopy(tmpInt8, gm_o, BLOCK_SIZE / sizeof(int8_t));
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        inputOffset = (float)(tmpInt8.GetValue(0));
    }

    __aicore__ inline void GetSliceOffsetAndSize(uint32_t row_offset, uint32_t sid, uint32_t &slice_offset,
                                                 uint64_t &col_offset, uint32_t &eleNum)
    {
        slice_offset = sid * slice_size;
        col_offset = row_offset + slice_offset;
        eleNum = (sid == (num_slice - 1)) ? tail_slice_size : slice_size;
    }

    __aicore__ inline float ComputeMean(uint32_t row_offset)
    {
        //  num_last_dim -> num_slice/tail_slice
        float temp_sum = 0;
        for (uint32_t sid = 0; sid < num_slice; sid++) {
            uint32_t slice_offset = 0;
            uint64_t col_offset = 0;
            uint32_t eleNum = 0;
            GetSliceOffsetAndSize(row_offset, sid, slice_offset, col_offset, eleNum);
            temp_sum += ComputeSliceMean(col_offset, eleNum);
        }
        return temp_sum * aveNum;
    }

    __aicore__ inline float ComputeVariance(uint32_t row_offset, float mean)
    {
        float ssd = 0;
        for (uint32_t sid = 0; sid < num_slice; sid++) {
            uint32_t slice_offset = 0;
            uint64_t col_offset = 0;
            uint32_t eleNum = 0;
            GetSliceOffsetAndSize(row_offset, sid, slice_offset, col_offset, eleNum);
            ssd += ComputeSliceSSD(col_offset, eleNum, mean);
        }
        return ssd * aveNum + eps;
    }

    __aicore__ inline float ComputeSliceMean(uint64_t col_offset, uint32_t size)
    {
        AscendC::LocalTensor<T> x_local = x_que.AllocTensor<T>();
        DataCopy(x_local, x_gm[col_offset], size);
        x_que.EnQue(x_local);
        AscendC::LocalTensor<T> x_local2 = x_que.DeQue<T>();
        AscendC::LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        AscendC::PipeBarrier<PIPE_V>();
        Cast(x_local_fp32, x_local2, AscendC::RoundMode::CAST_NONE, size);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSum(z_local_fp32, x_local_fp32, y_local_fp32, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float x_sum = z_local_fp32.GetValue(0);
        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        x_que.FreeTensor(x_local2);
        return x_sum;
    }

    __aicore__ inline float ComputeSliceSSD(uint64_t col_offset, uint32_t size, float mean)
    {
        AscendC::LocalTensor<T> x_local = x_que.AllocTensor<T>();
        DataCopy(x_local, x_gm[col_offset], size);
        x_que.EnQue(x_local);
        AscendC::LocalTensor<T> x_local2 = x_que.DeQue<T>();
        AscendC::LocalTensor<float> var_local = var_buf.Get<float>();
        AscendC::LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        AscendC::PipeBarrier<PIPE_V>();
        Cast(x_local_fp32, x_local2, AscendC::RoundMode::CAST_NONE, size);
        AscendC::PipeBarrier<PIPE_V>();
        Duplicate(y_local_fp32, mean, size);
        AscendC::PipeBarrier<PIPE_V>();
        Sub(z_local_fp32, x_local_fp32, y_local_fp32, size);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(x_local_fp32, z_local_fp32, z_local_fp32, size);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSum(var_local, x_local_fp32, y_local_fp32, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float var_local_temp = var_local.GetValue(0);
        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        x_que.FreeTensor(x_local2);
        return var_local_temp;
    }

    __aicore__ inline void SliceCopyIn(uint32_t slice_offset, uint64_t col_offset, uint32_t size)
    {
        AscendC::LocalTensor<T> x_local = x_que.AllocTensor<T>();
        AscendC::LocalTensor<T> beta_local = beta_que.AllocTensor<T>();
        AscendC::LocalTensor<T> gamma_local = gamma_que.AllocTensor<T>();
        DataCopy(x_local, x_gm[col_offset], size);
        DataCopy(beta_local, beta_gm[slice_offset], size);
        DataCopy(gamma_local, gamma_gm[slice_offset], size);
        x_que.EnQue(x_local);
        beta_que.EnQue(beta_local);
        gamma_que.EnQue(gamma_local);
    }

    __aicore__ inline void SlicePrecisionCompute(uint32_t nums, float mean, float variance)
    {
        AscendC::LocalTensor<T> x_local = x_que.DeQue<T>();
        AscendC::LocalTensor<T> beta_local = beta_que.DeQue<T>();
        AscendC::LocalTensor<T> gamma_local = gamma_que.DeQue<T>();
        AscendC::LocalTensor<int8_t> z_local = z_que.AllocTensor<int8_t>();
        AscendC::LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        AscendC::PipeBarrier<PIPE_V>();
        Cast(x_local_fp32, x_local, AscendC::RoundMode::CAST_NONE, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Duplicate(y_local_fp32, mean, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Sub(z_local_fp32, x_local_fp32, y_local_fp32, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Duplicate(y_local_fp32, variance, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Sqrt(y_local_fp32, y_local_fp32, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Duplicate(x_local_fp32, (float)1, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Div(y_local_fp32, x_local_fp32, y_local_fp32, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(x_local_fp32, z_local_fp32, y_local_fp32, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(z_local_fp32, gamma_local, AscendC::RoundMode::CAST_NONE, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(y_local_fp32, x_local_fp32, z_local_fp32, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(x_local_fp32, beta_local, AscendC::RoundMode::CAST_NONE, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Add(z_local_fp32, y_local_fp32, x_local_fp32, nums);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID2);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID2);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(z_local_fp32, z_local_fp32, inputScale, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(z_local_fp32, z_local_fp32, inputOffset, nums);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<half> tmpFp16 = x_buf_fp32.Get<half>();
        CastFrom32To16(tmpFp16, z_local_fp32, nums);
        AscendC::PipeBarrier<PIPE_V>();
        CastFromF16ToI8(z_local, tmpFp16, QUANT_MIN, nums);
        AscendC::PipeBarrier<PIPE_V>();
        x_que.FreeTensor(x_local);
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void SliceCopyOut(uint64_t offset, uint32_t size)
    {
        AscendC::LocalTensor<int8_t> z = z_que.DeQue<int8_t>();
        DataCopy(z_gm[offset], z, size);
        z_que.FreeTensor(z);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> z_que;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> gamma_que;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> beta_que;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ave_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> var_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_buf_fp32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> y_buf_fp32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> z_buf_fp32;
    AscendC::GlobalTensor<T> x_gm, gamma_gm, beta_gm;
    AscendC::GlobalTensor<int8_t> z_gm;
    uint32_t half_num{0};
    uint32_t num_core{0};
    uint32_t num_first_dim{0};
    uint32_t num_last_dim{0};
    uint32_t row_step{0};
    uint32_t row_work{0};
    uint64_t gm_offset_{0};
    uint32_t row_tail_{0};
    uint32_t first_dim_per_times{0};
    uint32_t nl_first_dim_per_core{0};
    uint32_t l_first_dim_per_core{0};
    uint32_t slice_size{0};
    uint32_t num_slice{0};
    uint32_t tail_slice_size{0};
    float eps{0};
    float aveNum{0};
    float inputScale{0};
    float inputOffset{0};
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::LayerNormQuantTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numCore = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->numLastDim = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->numFirstDim = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->nlFirstdimPerCore = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->lFirstdimPerCore = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->firstDimPerTimes = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->epsStr = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
    tilingdata->aveStr = (*(const __gm__ uint32_t *)(p_tilingdata + 28));
    tilingdata->normMode = (*(const __gm__ uint32_t *)(p_tilingdata + 32));
    tilingdata->zoomScale = (*(const __gm__ float *)(p_tilingdata + 36));
    tilingdata->sliceNum = (*(const __gm__ uint32_t *)(p_tilingdata + 40));
    tilingdata->sliceSize = (*(const __gm__ uint32_t *)(p_tilingdata + 44));
    tilingdata->tailSliceSize = (*(const __gm__ uint32_t *)(p_tilingdata + 48));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::LayerNormQuantTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->numLastDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->numFirstDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->nlFirstdimPerCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingdata->lFirstdimPerCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingdata->firstDimPerTimes = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 20));
    tilingdata->epsStr = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    tilingdata->aveStr = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 28));
    tilingdata->normMode = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 32));
    tilingdata->zoomScale = (*(__ubuf__ float *)(tilingdata_in_ub + 36));
    tilingdata->sliceNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 40));
    tilingdata->sliceSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 44));
    tilingdata->tailSliceSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 48));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tilingArg, tilingData)                                                                         \
    AsdOps::LayerNormQuantTilingData tilingData;                                                                       \
    InitTilingData(tilingArg, &(tilingData))

extern "C" __global__ __aicore__ void layer_norm_quant(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR scale,
                                                       GM_ADDR offset, GM_ADDR z, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling, tilingData);
    if (TILING_KEY_IS(2000000000)) { // half & SliceCompute
        KernelLayerNormQuant<half, false> op;
        op.Init(x, gamma, beta, scale, offset, z, tilingData);
        op.Process();
    }
    if (TILING_KEY_IS(2010000000)) { // half & FastCompute
        KernelLayerNormQuant<half, true> op;
        op.Init(x, gamma, beta, scale, offset, z, tilingData);
        op.Process();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(2100000000)) { // bf16 & SliceCompute
        KernelLayerNormQuant<bfloat16_t, false> op;
        op.Init(x, gamma, beta, scale, offset, z, tilingData);
        op.Process();
    }
    if (TILING_KEY_IS(2110000000)) { // bf16 & FastCompute
        KernelLayerNormQuant<bfloat16_t, true> op;
        op.Init(x, gamma, beta, scale, offset, z, tilingData);
        op.Process();
    }
#endif
}
