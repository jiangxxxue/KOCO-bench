/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "kernels/utils/kernel/kernel_utils.h"
#include "kernels/norm/layernormstride/tiling/tiling_data.h"

static constexpr uint32_t BUFFER_NUM = 1;
static constexpr uint32_t BLOCK_NUMEL = 16;
static constexpr uint32_t MAX_UB_SIZE_NUM = 98304;
static constexpr uint32_t DATA_BYTE = 2;
static constexpr uint32_t NCT_MAX_SIZE = 8;     // NCT 相关数组的最大长度
static constexpr uint32_t ROM_STRIDE_NUM = 2;
static constexpr uint32_t DIM_VALUE = 2;

using AscendC::HardEvent;

template <typename T, bool FastComputeMode = false>
class KernelLayerNormStride {
public:
    __aicore__ inline KernelLayerNormStride() {}
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

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *gamma, __gm__ uint8_t *beta,
                                __gm__ uint8_t *z, AsdOps::LayerNormStrideTilingData tilingData)
    {
        x_dim_num_ = tilingData.xDimNum;
        for (size_t i = 0; i < x_dim_num_; ++ i) {
            x_strides_[i] = tilingData.xStrides[i];
        }
        x_offset_ = tilingData.xOffset;
        if (x_dim_num_ > 0) {
            isNct_ = true;
        } else {
            x_offset_ = 0;
        }
        half_num = 2;
        num_core = tilingData.numCore;
        num_last_dim = tilingData.numLastDim;
        num_first_dim = tilingData.numFirstDim;
        nl_first_dim_per_core = tilingData.nlFirstdimPerCore;
        l_first_dim_per_core = tilingData.lFirstdimPerCore;
        first_dim_per_times = tilingData.firstDimPerTimes;
        aveNum = tilingData.aveStr;
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
        act_num_col_ = ROUND_UP(num_last_dim);
        gm_offset_ = static_cast<uint64_t>(nl_first_dim_per_core) * num_last_dim;
        if (isNct_) {
            gm_offset_ = static_cast<uint64_t>(nl_first_dim_per_core) * x_strides_[x_dim_num_ - ROM_STRIDE_NUM];
        }

        x_gm.SetGlobalBuffer((__gm__ T *)x + x_offset_ + AscendC::GetBlockIdx() * gm_offset_);
        z_gm.SetGlobalBuffer((__gm__ T *)z + x_offset_ + AscendC::GetBlockIdx() * gm_offset_);
        gamma_gm.SetGlobalBuffer((__gm__ T *)gamma);
        beta_gm.SetGlobalBuffer((__gm__ T *)beta);

        pipe.InitBuffer(x_que, BUFFER_NUM, row_step * ROUND_UP(slice_size) * DATA_BYTE);
        pipe.InitBuffer(z_que, BUFFER_NUM, row_step * ROUND_UP(slice_size) * DATA_BYTE);
        pipe.InitBuffer(beta_que, BUFFER_NUM, 1 * ROUND_UP(slice_size) * DATA_BYTE);
        pipe.InitBuffer(gamma_que, BUFFER_NUM, 1 * ROUND_UP(slice_size) * DATA_BYTE);
        pipe.InitBuffer(ave_buf, BLOCK_NUMEL * DATA_BYTE);
        pipe.InitBuffer(var_buf, BLOCK_NUMEL * DATA_BYTE);
        pipe.InitBuffer(x_buf_fp32, half_num * ROUND_UP(slice_size) * DATA_BYTE);
        pipe.InitBuffer(y_buf_fp32, half_num * ROUND_UP(slice_size) * DATA_BYTE);
        pipe.InitBuffer(z_buf_fp32, half_num * ROUND_UP(slice_size) * DATA_BYTE);
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

    __aicore__ inline void setSliceInfo(AscendC::SliceInfo srcSliceInfo[], AscendC::SliceInfo dstSliceInfo[],
                                        uint32_t strides[], uint32_t dimNum, uint32_t eleNum, uint32_t  rowNum)
    {
        // 从对应的非连续tensor中构造连续的tensor
        srcSliceInfo[0].startIndex = 0;
        srcSliceInfo[0].endIndex = eleNum - 1;
        srcSliceInfo[0].stride  = strides[dimNum - ROM_STRIDE_NUM] - eleNum;
        srcSliceInfo[0].burstLen = CEIL_DIV(eleNum, BLOCK_NUMEL);
        srcSliceInfo[0].shapeValue = strides[dimNum - ROM_STRIDE_NUM];
        srcSliceInfo[1].startIndex = 0;
        srcSliceInfo[1].endIndex = rowNum - 1;
        srcSliceInfo[1].stride = 0;
        srcSliceInfo[1].burstLen = 1;
        srcSliceInfo[1].shapeValue = rowNum;
        dstSliceInfo[0] = {0, eleNum - 1, 0, CEIL_DIV(eleNum, BLOCK_NUMEL), eleNum};
        dstSliceInfo[1] = {0, rowNum - 1, 0, 1, rowNum};
    }

    __aicore__ inline void FastCompute()
    {
        uint32_t move_cnt = CEIL_DIV(row_work, row_step);
        for (uint32_t i = 0; i < move_cnt; ++i) {
            if (i < move_cnt - 1) {
                FastCopyIn(i, row_step * act_num_col_);

                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                FastPrecisionCompute(row_step);
                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

                FastCopyOut(i, row_step * act_num_col_);
            } else {
                FastCopyIn(i, row_tail_ * act_num_col_);

                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                FastPrecisionCompute(row_tail_);
                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

                FastCopyOut(i, row_tail_ * act_num_col_);
            }
        }
    }

    __aicore__ inline void FastCopyIn(uint64_t proc_id, uint64_t size)
    {
        AscendC::LocalTensor<T> x_local = x_que.AllocTensor<T>();
        AscendC::LocalTensor<T> beta_local = beta_que.AllocTensor<T>();
        AscendC::LocalTensor<T> gamma_local = gamma_que.AllocTensor<T>();
        uint64_t offset = proc_id * row_step * act_num_col_;
        if (isNct_) {
            offset = proc_id * row_step * x_strides_[x_dim_num_ - ROM_STRIDE_NUM];
            setSliceInfo(src_sliceInfo_, dst_sliceInfo_, x_strides_, x_dim_num_, act_num_col_, size/act_num_col_);
            DataCopy(x_local, x_gm[offset], dst_sliceInfo_, src_sliceInfo_, DIM_VALUE);
        } else {
            DataCopy(x_local, x_gm[offset], size);
        }
            
        DataCopy(beta_local, beta_gm[0], act_num_col_);
        DataCopy(gamma_local, gamma_gm[0], act_num_col_);
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
        AscendC::LocalTensor<T> z_local = z_que.AllocTensor<T>();
        AscendC::LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        for (uint64_t rid = 0; rid < nums; ++rid) {
            Cast(x_local_fp32, x_local[rid * act_num_col_], AscendC::RoundMode::CAST_NONE, num_last_dim);
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
            ComputeLayerNorm(z_local_fp32, x_local_fp32, y_local_fp32, eps, aveNum,
                             gamma_local, beta_local, num_last_dim);
            CastFrom32To16(z_local[rid * act_num_col_], z_local_fp32, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
        }
        x_que.FreeTensor(x_local);
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void FastCopyOut(uint64_t proc_id, uint64_t size)
    {
        AscendC::LocalTensor<T> z = z_que.DeQue<T>();
        uint64_t offset = proc_id * row_step * act_num_col_;
        if (isNct_) {
            offset = proc_id * row_step * x_strides_[x_dim_num_ - ROM_STRIDE_NUM];
            DataCopy(z_gm[offset], z, src_sliceInfo_, dst_sliceInfo_, DIM_VALUE);
        } else {
            DataCopy(z_gm[offset], z, size);
        }
        z_que.FreeTensor(z);
    }

    __aicore__ inline void SliceCompute()
    {
        for (uint32_t pid = 0; pid < row_work; pid++) {
            uint32_t row_offset = pid * num_last_dim;
            if (isNct_) {
                row_offset = pid * x_strides_[x_dim_num_ - ROM_STRIDE_NUM];
            }

            float mean = ComputeMean(row_offset);
            float variance = ComputeVariance(row_offset, mean);

            for (uint32_t sid = 0; sid < num_slice; sid++) {
                uint32_t slice_offset = 0;
                uint64_t col_offset = 0;
                uint32_t eleNum = 0;
                GetSliceOffsetAndSize(row_offset, sid, slice_offset, col_offset, eleNum);

                SliceCopyIn(slice_offset, col_offset, ROUND_UP(eleNum));

                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                SlicePrecisionCompute(eleNum, mean, variance);
                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

                SliceCopyOut(col_offset, ROUND_UP(eleNum));
            }
        }
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
        DataCopy(x_local, x_gm[col_offset], ROUND_UP(size));
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
        DataCopy(x_local, x_gm[col_offset], ROUND_UP(size));
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
        AscendC::LocalTensor<T> z_local = z_que.AllocTensor<T>();
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
        CastFrom32To16(z_local, z_local_fp32, nums);
        AscendC::PipeBarrier<PIPE_V>();
        x_que.FreeTensor(x_local);
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void SliceCopyOut(uint64_t offset, uint32_t size)
    {
        AscendC::LocalTensor<T> z = z_que.DeQue<T>();
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
    AscendC::GlobalTensor<T> z_gm;
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
    bool isNct_{false};
    uint32_t x_strides_[NCT_MAX_SIZE];      // 非连续tensor x步长
    uint32_t x_dim_num_{0};                 // 非连续tensor x维度数
    uint32_t x_offset_{0};                 // 非连续tensor x偏移量
    uint32_t act_num_col_{0};
    AscendC::SliceInfo src_sliceInfo_[2];
    AscendC::SliceInfo dst_sliceInfo_[2];
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::LayerNormStrideTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numCore = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->numLastDim = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->numFirstDim = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->nlFirstdimPerCore = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->lFirstdimPerCore = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->firstDimPerTimes = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->epsStr = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
    tilingdata->aveStr = (*(const __gm__ float *)(p_tilingdata + 28));
    tilingdata->normMode = (*(const __gm__ uint32_t *)(p_tilingdata + 32));
    tilingdata->zoomScale = (*(const __gm__ float *)(p_tilingdata + 36));
    tilingdata->sliceNum = (*(const __gm__ uint32_t *)(p_tilingdata + 40));
    tilingdata->sliceSize = (*(const __gm__ uint32_t *)(p_tilingdata + 44));
    tilingdata->tailSliceSize = (*(const __gm__ uint32_t *)(p_tilingdata + 48));
    for (int i = 0; i < NCT_MAX_SIZE; ++ i) {
        tilingdata->xStrides[i] = (*(const __gm__ uint32_t *)(4 * i + p_tilingdata + 52));
    }
    tilingdata->xOffset = (*(const __gm__ uint32_t *)(p_tilingdata + 84));
    tilingdata->xDimNum = (*(const __gm__ uint32_t *)(p_tilingdata + 88));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::LayerNormStrideTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->numLastDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->numFirstDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->nlFirstdimPerCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingdata->lFirstdimPerCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingdata->firstDimPerTimes = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 20));
    tilingdata->epsStr = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    tilingdata->aveStr = (*(__ubuf__ float *)(tilingdata_in_ub + 28));
    tilingdata->normMode = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 32));
    tilingdata->zoomScale = (*(__ubuf__ float *)(tilingdata_in_ub + 36));
    tilingdata->sliceNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 40));
    tilingdata->sliceSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 44));
    tilingdata->tailSliceSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 48));
    for (int i = 0; i < NCT_MAX_SIZE; ++ i) {
        tilingdata->xStrides[i] = (*(__ubuf__ uint32_t *)(4 * i + tilingdata_in_ub + 52));
    }
    tilingdata->xOffset = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 84));
    tilingdata->xDimNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 88));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tilingArg, tilingData)                                                                        \
    AsdOps::LayerNormStrideTilingData tilingData;                                                                     \
    InitTilingData(tilingArg, &(tilingData))

extern "C" __global__ __aicore__ void layer_norm_stride(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta,
                                                        GM_ADDR z, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling, tilingData);
    if (TILING_KEY_IS(2000000000)) { // half & SliceCompute
        KernelLayerNormStride<half, false> op;
        op.Init(x, gamma, beta, z, tilingData);
        op.Process();
    }
    if (TILING_KEY_IS(2010000000)) { // half & FastCompute
        KernelLayerNormStride<half, true> op;
        op.Init(x, gamma, beta, z, tilingData);
        op.Process();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(2100000000)) { // bf16 & SliceCompute
        KernelLayerNormStride<bfloat16_t, false> op;
        op.Init(x, gamma, beta, z, tilingData);
        op.Process();
    }
    if (TILING_KEY_IS(2110000000)) { // bf16 & FastCompute
        KernelLayerNormStride<bfloat16_t, true> op;
        op.Init(x, gamma, beta, z, tilingData);
        op.Process();
    }
#endif
}
