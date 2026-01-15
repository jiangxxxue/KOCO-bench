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
#include "kernels/norm/common/common_tiling_data.h"
#include "kernels/norm/common/op_kernel/common_quant.h"
#include "kernels/norm/rmsnormforward/op_kernel/rms_norm_base.h"
static constexpr uint32_t BUF_FACTOR = 3;        // 1(g) + 1(sqx) + 1(sum) = 3
static constexpr uint32_t OFFSET_GAMMA = 0;      // the offset of gamma is 0
static constexpr uint32_t OFFSET_SQX = 1;        // the offset of sqx is 1
static constexpr uint32_t OFFSET_SUM = 2;        // the offset of sum is 2
static constexpr uint32_t OFFSET_WORKSPACE = 3;  // the offset of workspace is 3
static constexpr uint32_t DIM_2 = 2;
static constexpr uint32_t REPEAT_TIME_256 = 256; // 256 default stride
static constexpr uint32_t REPEAT_TIME_16 = 16; // 16 default stride
static constexpr uint32_t REPEAT_TIME_64 = 64;   // 64 default stride

template <typename T, bool WITH_BETA, bool FastComputeMode = false>
class RmsNormQuant {
public:
    __aicore__ inline RmsNormQuant(__gm__ uint8_t *x, __gm__ uint8_t *g, __gm__ uint8_t *b, __gm__ uint8_t *scale,
                                   __gm__ uint8_t *offset, __gm__ uint8_t *y,
                                   AsdOps::RmsNormQuantCommonTilingData &tiling_data)
    {
        num_core_ = tiling_data.numCore;
        num_col_ = tiling_data.numCol;
        avg_factor_ = *reinterpret_cast<float *>(&tiling_data.avgFactor);
        epsilon_ = *reinterpret_cast<float *>(&tiling_data.epsilon);
        slice_size_ = tiling_data.sliceSize;
        uint32_t num_row = tiling_data.numRow;
        uint32_t row_work = (num_row + num_core_ - 1) / num_core_;
        if (AscendC::GetBlockIdx() != num_core_ - 1) {
            row_work_ = row_work;
        } else {
            row_work_ = num_row - (num_core_ - 1) * row_work;
        }
        gm_offset_ = static_cast<uint64_t>(row_work) * num_col_;
        if (num_col_ <= slice_size_) {
            num_col_align_int8 = (num_col_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
            num_col_align_f16 = (num_col_ + REPEAT_TIME_16 - 1) / REPEAT_TIME_16 * REPEAT_TIME_16;
            num_col_align_f32 = (num_col_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
        } else {
            num_col_align_int8 = slice_size_;
            num_col_align_f16 = slice_size_;
            num_col_align_f32 = slice_size_;
            num_col_align_f32_long = (num_col_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
        }
        quantMin_ = tiling_data.quantMin;
        gm_x_.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gm_offset_);
        gm_g_.SetGlobalBuffer((__gm__ T *)g);
        gm_b_.SetGlobalBuffer((__gm__ T *)b);
        gm_y_.SetGlobalBuffer((__gm__ int8_t *)y + AscendC::GetBlockIdx() * gm_offset_);

        pipe.InitBuffer(fp16_x_que_, BUFFER_NUM, num_col_align_f16 * sizeof(T));
        pipe.InitBuffer(int8_y_que_, BUFFER_NUM, num_col_align_int8 * sizeof(int8_t)); // quant output
        pipe.InitBuffer(fp32_xy_buf_, num_col_align_f32 * sizeof(float));
        pipe.InitBuffer(fp16_buf_, num_col_align_f16 * sizeof(T));
        pipe.InitBuffer(calc_buf_, BUF_FACTOR * num_col_align_f32 * sizeof(float) + 32); // 32 for sum
        GetScaleAndOffset<T>(input_scale_, input_offset_, scale, offset, fp16_buf_);
    }

    __aicore__ inline void Launch()
    {
        if constexpr (FastComputeMode) {
            FastCompute();
        } else {
            SliceCompute();
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<T> fp16_x = fp16_x_que_.AllocTensor<T>();
        DataCopy(fp16_x, gm_x_[offset], {1, static_cast<uint16_t>(numel / 16), 0, 0}); // 16 is for 32B / sizeof(half)
        fp16_x_que_.EnQue(fp16_x);
    }

    __aicore__ inline void FastCompute()
    {
        AscendC::LocalTensor<T> fp16_g = fp32_xy_buf_.Get<T>(num_col_align_f32);
        AscendC::LocalTensor<float> fp32_g = calc_buf_.Get<float>(num_col_align_f16);
        DataCopy(fp16_g, gm_g_, num_col_align_f16);
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(fp32_g[OFFSET_GAMMA * num_col_align_f32], fp16_g, AscendC::RoundMode::CAST_NONE, REPEAT_TIME_64,
             num_col_align_f32 / REPEAT_TIME_64,
             {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
        AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        if constexpr (WITH_BETA) {
            AscendC::LocalTensor<T> fp16_buffer = fp16_buf_.Get<T>();
            DataCopy(fp16_buffer, gm_b_, num_col_align_f16);
        }
        uint64_t pid = 0;
        while (pid < row_work_) {
            uint64_t offset = pid * num_col_;
            CopyIn(offset, num_col_align_f16);
            Compute();
            CopyOut(offset, num_col_align_f16);
            ++pid;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void SliceCompute()
    {
        slice_size_long = slice_size_;
        num_slice_ = (num_col_align_f32_long + slice_size_long - 1) / slice_size_long;
        tail_size_ = num_col_align_f32_long - (num_slice_ - 1) * slice_size_long;
        tail_copy_ = num_col_ - (num_slice_ - 1) * slice_size_long;
        for (uint64_t pid = 0; pid < row_work_; pid++) {
            uint64_t row_offset = pid * num_col_;
            float squareSum = 0.0f;
            for (uint64_t sid = 0; sid < num_slice_; sid++) {
                uint64_t col_offset = row_offset + sid * slice_size_long;
                uint32_t eleNum = (sid == (num_slice_ - 1)) ? tail_size_ : slice_size_long;
                CopyIn(col_offset, eleNum);
                AscendC::PipeBarrier<PIPE_ALL>();
                squareSum += ComputeSquareSum(eleNum);
            }
            float avg = squareSum * avg_factor_;
            float rms = sqrt(avg + epsilon_);
            float factor = 1 / rms;
            for (uint64_t sid = 0; sid < num_slice_; sid++) {
                uint64_t sliceOffset = sid * slice_size_long;
                uint32_t eleNum = (sid == (num_slice_ - 1)) ? tail_size_ : slice_size_long;
                uint32_t copyNum = (sid == (num_slice_ - 1)) ? tail_copy_ : slice_size_long;
                uint64_t totalOffset = row_offset + sliceOffset;
                if constexpr (WITH_BETA) {
                    AscendC::LocalTensor<T> fp16_buffer = fp16_buf_.Get<T>();
                    DataCopy(fp16_buffer, gm_b_[sliceOffset], eleNum);
                }
                CopyInGama(sliceOffset, eleNum);
                AscendC::PipeBarrier<PIPE_ALL>();
                CopyIn(totalOffset, eleNum);
                AscendC::PipeBarrier<PIPE_ALL>();
                ComputeNormandQuant(factor, eleNum);
                AscendC::PipeBarrier<PIPE_ALL>();
                CopyOutLong(totalOffset, copyNum);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInGama(int64_t sliceOffset, uint32_t numel)
    {
        AscendC::LocalTensor<T> fp16_g = fp32_xy_buf_.Get<T>(slice_size_long);
        AscendC::LocalTensor<float> fp32_g = calc_buf_.Get<float>(slice_size_long);
        DataCopy(fp16_g, gm_g_[sliceOffset], numel);
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(fp32_g[OFFSET_GAMMA * slice_size_long], fp16_g,
             AscendC::RoundMode::CAST_NONE, REPEAT_TIME_64, numel / REPEAT_TIME_64,
             {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
        AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> fp16_x = fp16_x_que_.DeQue<T>();
        AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
        AscendC::LocalTensor<int8_t> int8_y = int8_y_que_.AllocTensor<int8_t>();
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<float> g = buf[OFFSET_GAMMA * num_col_align_f32];       // 0
        AscendC::LocalTensor<float> sqx = buf[OFFSET_SQX * num_col_align_f32];       // 1
        AscendC::LocalTensor<float> work = buf[OFFSET_SUM * num_col_align_f32];      // 2
        AscendC::LocalTensor<float> sum = buf[OFFSET_WORKSPACE * num_col_align_f32]; // 4
        Cast(fp32_xy, fp16_x, AscendC::RoundMode::CAST_NONE, REPEAT_TIME_64, num_col_align_f32 / REPEAT_TIME_64,
             {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
        AscendC::PipeBarrier<PIPE_V>();
        Mul(sqx, fp32_xy, fp32_xy, REPEAT_TIME_64, num_col_align_f32 / REPEAT_TIME_64,
            {1, 1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
        AscendC::PipeBarrier<PIPE_V>();
        Muls(sqx, sqx, avg_factor_, num_col_align_f32);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSumCustom(sum, sqx, work, num_col_);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(sum, sum, epsilon_, 1);
        AscendC::PipeBarrier<PIPE_V>();
        Sqrt(sum, sum, 1);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float factor = 1 / sum.GetValue(0);
        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Muls(fp32_xy, fp32_xy, factor, REPEAT_TIME_64, num_col_align_f32 / REPEAT_TIME_64,
             {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
        AscendC::PipeBarrier<PIPE_V>();
        Mul(fp32_xy, fp32_xy, g, REPEAT_TIME_64, num_col_align_f32 / REPEAT_TIME_64,
            {1, 1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (WITH_BETA) { // quant的beta是fp16加的
            AscendC::LocalTensor<T> b = fp16_buf_.Get<T>();
            Cast(work, b, AscendC::RoundMode::CAST_NONE, REPEAT_TIME_64, num_col_align_f32 / REPEAT_TIME_64,
                 {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
            AscendC::PipeBarrier<PIPE_V>();
            Add(fp32_xy, fp32_xy, work, REPEAT_TIME_64, num_col_align_f32 / REPEAT_TIME_64,
                {1, 1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE,
                 AscendC::DEFAULT_REPEAT_STRIDE});
            AscendC::PipeBarrier<PIPE_V>();
        }
        Muls(fp32_xy, fp32_xy, input_scale_, REPEAT_TIME_64, num_col_align_f32 / REPEAT_TIME_64,
             {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
        AscendC::PipeBarrier<PIPE_V>();
        Adds(fp32_xy, fp32_xy, input_offset_, REPEAT_TIME_64, num_col_align_f32 / REPEAT_TIME_64,
             {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<half> tmpfp16Buf = calc_buf_.Get<half>();
        AscendC::LocalTensor<half> tmpfp16 = tmpfp16Buf[OFFSET_SUM * num_col_align_f32 * DIM_2]; // 2：work,float偏移到half
        CastFrom32To16(tmpfp16, fp32_xy, num_col_align_f32);
        CastFromF16ToI8(int8_y, tmpfp16, quantMin_, num_col_align_f16);
        int8_y_que_.EnQue(int8_y);
        fp16_x_que_.FreeTensor(fp16_x);
    }

    __aicore__ inline float ComputeSquareSum(uint32_t numel)
    {
        AscendC::LocalTensor<T> fp16_x = fp16_x_que_.DeQue<T>();
        AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<float> sqx = buf[OFFSET_SQX * slice_size_long];       // 1
        AscendC::LocalTensor<float> work = buf[OFFSET_SUM * slice_size_long];      // 2
        AscendC::LocalTensor<float> sum = buf[OFFSET_WORKSPACE * slice_size_long]; // 4
        Cast(fp32_xy, fp16_x, AscendC::RoundMode::CAST_NONE, REPEAT_TIME_64, numel / REPEAT_TIME_64,
             {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
        Mul(sqx, fp32_xy, fp32_xy, REPEAT_TIME_64, numel / REPEAT_TIME_64,
            {1, 1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
#if __CCE_AICORE__ == 100
        ReduceSumCustom(sum, sqx, work, numel);
#else
        ReduceSum(sum, sqx, work, numel);
#endif
        fp16_x_que_.FreeTensor(fp16_x);
        float sumall = sum.GetValue(0);
        return sumall;
    }

    __aicore__ inline void ComputeNormandQuant(float factor, uint32_t num)
    {
        AscendC::LocalTensor<T> fp16_x = fp16_x_que_.DeQue<T>();
        AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
        AscendC::LocalTensor<int8_t> int8_y = int8_y_que_.AllocTensor<int8_t>();
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<float> g = buf[OFFSET_GAMMA * slice_size_long];  // 0
        AscendC::LocalTensor<float> sqx = buf[OFFSET_SQX * slice_size_long];  // 1
        AscendC::LocalTensor<float> work = buf[OFFSET_SUM * slice_size_long]; // 2
        Cast(fp32_xy, fp16_x, AscendC::RoundMode::CAST_NONE, REPEAT_TIME_64, slice_size_long / REPEAT_TIME_64,
             {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
        Muls(fp32_xy, fp32_xy, factor, REPEAT_TIME_64, slice_size_long / REPEAT_TIME_64,
             {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
        Mul(fp32_xy, fp32_xy, g, REPEAT_TIME_64, slice_size_long / REPEAT_TIME_64,
            {1, 1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
        if constexpr (WITH_BETA) { // quant的beta是fp16加的
            AscendC::LocalTensor<T> b = fp16_buf_.Get<T>();
            Cast(work, b, AscendC::RoundMode::CAST_NONE, REPEAT_TIME_64, slice_size_long / REPEAT_TIME_64,
                 {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE / OFFSET_SUM});
            Add(fp32_xy, fp32_xy, work, REPEAT_TIME_64, slice_size_long / REPEAT_TIME_64,
                {1, 1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE,
                 AscendC::DEFAULT_REPEAT_STRIDE});
        }
        Muls(fp32_xy, fp32_xy, input_scale_, REPEAT_TIME_64, slice_size_long / REPEAT_TIME_64,
             {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
        Adds(fp32_xy, fp32_xy, input_offset_, REPEAT_TIME_64, slice_size_long / REPEAT_TIME_64,
             {1, 1, AscendC::DEFAULT_REPEAT_STRIDE, AscendC::DEFAULT_REPEAT_STRIDE});
        AscendC::LocalTensor<half> tmpfp16Buf = calc_buf_.Get<half>();
        AscendC::LocalTensor<half> tmpfp16 = tmpfp16Buf[OFFSET_SUM * slice_size_long * DIM_2]; // 2：work,float偏移到half
        CastFrom32To16(tmpfp16, fp32_xy, num);
        CastFromF16ToI8(int8_y, tmpfp16, quantMin_, num);
        int8_y_que_.EnQue(int8_y);
        fp16_x_que_.FreeTensor(fp16_x);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<int8_t> int8_y = int8_y_que_.DeQue<int8_t>();
        DataCopyCustom<int8_t>(gm_y_[offset], int8_y, num_col_);
        int8_y_que_.FreeTensor(int8_y);
    }

    __aicore__ inline void CopyOutLong(int64_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<int8_t> int8_y = int8_y_que_.DeQue<int8_t>();
        DataCopyCustom<int8_t>(gm_y_[offset], int8_y, numel);
        int8_y_que_.FreeTensor(int8_y);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> fp16_x_que_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> int8_y_que_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_xy_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16_buf_;

    AscendC::GlobalTensor<T> gm_x_;
    AscendC::GlobalTensor<T> gm_g_;
    AscendC::GlobalTensor<T> gm_b_;
    AscendC::GlobalTensor<int8_t> gm_y_;
    uint32_t num_core_{0};   // 一共激活多少AICORE
    uint32_t num_col_{0};    // 输入的列数
    uint32_t row_work_{0};   // 需要计算多少行
    uint32_t row_step_{0};   // 除最后一次，每次搬入多少行
    uint32_t row_tail_{0};   // 最后一次搬入多少行数据
    uint64_t gm_offset_{0};  // GM数据起始位置偏移量
    float avg_factor_{1.0};  // num_col_的倒数
    float input_scale_{1.0}; // 非对称量化系数
    float input_offset_{0};  // 非对称量化偏移适配高精度
    float epsilon_{1e-12f};  // norm平滑参数
    uint32_t num_col_align_int8{0};
    uint32_t num_col_align_f16{0};
    uint32_t num_col_align_f32{0};
    uint32_t num_col_align_f32_long{0};
    uint32_t num_col_temp;
    half quantMin_{-128};
    uint32_t slice_size_{0};
    uint32_t slice_size_long{0}; // 每一行切分的大小
    uint32_t num_slice_{0};
    uint32_t tail_size_{0};
    uint32_t tail_copy_{0};
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata,
                                      AsdOps::RmsNormQuantCommonTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numCore = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->numCol = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->numRow = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->avgFactor = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->epsilon = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->quantMin = (*(const __gm__ float *)(p_tilingdata + 20));
    tilingdata->sliceSize = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::RmsNormQuantCommonTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->numCol = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->numRow = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->avgFactor = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingdata->epsilon = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingdata->quantMin = (*(__ubuf__ float *)(tilingdata_in_ub + 20));
    tilingdata->sliceSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                                       \
    AsdOps::RmsNormQuantCommonTilingData tiling_data;                                                                  \
    InitTilingData((tiling_arg), &(tiling_data))

extern "C" __global__ __aicore__ void rms_norm_quant(GM_ADDR x, GM_ADDR g, GM_ADDR b, GM_ADDR scale, GM_ADDR offset,
                                                     GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(2000000000)) { // fp16, beta, scale, offset, no slice
        RmsNormQuant<half, true, true> kernel(x, g, b, scale, offset, y, tiling_data);
        kernel.Launch();
    }
    if (TILING_KEY_IS(2010000000)) { // fp16, empty beta, scale, offset, no slice
        RmsNormQuant<half, false, true> kernel(x, g, b, scale, offset, y, tiling_data);
        kernel.Launch();
    }
    if (TILING_KEY_IS(2001000000)) { // fp16, beta, scale, offset, use slice
        RmsNormQuant<half, true, false> kernel(x, g, b, scale, offset, y, tiling_data);
        kernel.Launch();
    }
    if (TILING_KEY_IS(2011000000)) { // fp16, empty beta, scale, offset, use slice
        RmsNormQuant<half, false, false> kernel(x, g, b, scale, offset, y, tiling_data);
        kernel.Launch();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(2100000000)) { // bf16, beta, scale, offset, no slice
        RmsNormQuant<bfloat16_t, true, true> kernel(x, g, b, scale, offset, y, tiling_data);
        kernel.Launch();
    }
    if (TILING_KEY_IS(2110000000)) { // bf16, empty beta, scale, offset, no slice
        RmsNormQuant<bfloat16_t, false, true> kernel(x, g, b, scale, offset, y, tiling_data);
        kernel.Launch();
    }
    if (TILING_KEY_IS(2101000000)) { // bf16, beta, scale, offset, use slice
        RmsNormQuant<bfloat16_t, true, false> kernel(x, g, b, scale, offset, y, tiling_data);
        kernel.Launch();
    }
    if (TILING_KEY_IS(2111000000)) { // bf16, empty beta, scale, offset, use slice
        RmsNormQuant<bfloat16_t, false, false> kernel(x, g, b, scale, offset, y, tiling_data);
        kernel.Launch();
    }
#endif
}