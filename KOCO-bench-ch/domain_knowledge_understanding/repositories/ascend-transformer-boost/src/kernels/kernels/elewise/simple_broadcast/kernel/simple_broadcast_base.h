/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_SIMPLE_BROADCAST_BASE_H
#define ASCEND_SIMPLE_BROADCAST_BASE_H
#include "kernel_operator.h"
#include "kernels/utils/kernel/kernel_utils.h"
#include "kernels/elewise/simple_broadcast/tiling/simple_broadcast_tiling_data.h"

using namespace AscendC;
using namespace AsdOps;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t ELEM_ALIGN = BLOCK_SIZE / sizeof(int8_t); // 最小类型是int8，需要保证int8是对齐的

class SimpleBroadcastBase {
public:
    __aicore__ inline SimpleBroadcastBase() {}

    __aicore__ inline void Init(BroadcastTilingData *tilingData)
    {
        // init tiling
        tilingData_ = tilingData;

        // init idx, offset
        nIdx_ = GetBlockIdx() % tilingData_->dimNBlockNum * tilingData_->dimNBlockSize;
        nOffset_ = nIdx_;
        bIdx_ = GetBlockIdx() / tilingData_->dimNBlockNum * tilingData_->dimBBlockSize;
        bOffset_ = bIdx_;

        if (tilingData_->dimBLoop > 1) {
            outerCount_ = tilingData_->dimBLoop * tilingData_->dimN;
            int64_t leftCount = (tilingData_->dimB - bOffset_) * tilingData_->dimN;
            outerCount_ = outerCount_ > leftCount ? leftCount : outerCount_;
            innerCount_ = tilingData_->dimN;

            outerCountAlign_ = RoundUp(tilingData_->dimN, ELEM_ALIGN) * tilingData_->dimBLoop;
            innerCountAlign_ = RoundUp(innerCount_, ELEM_ALIGN);
        } else {
            outerCount_ = tilingData_->dimNLoop;
            int64_t leftCount = (tilingData_->dimN - nOffset_);
            outerCount_ = leftCount < outerCount_ ? leftCount : outerCount_;
            innerCount_ = outerCount_;

            outerCountAlign_ = RoundUp(outerCount_, ELEM_ALIGN);
            innerCountAlign_ = RoundUp(innerCount_, ELEM_ALIGN);
        }
    }

    // 准备下一次搬入，理论数量dimBLoop * dimNLoop
    __aicore__ inline int64_t OuterNext()
    {
        // 复位b方向分次计算计数器
        innerBOffset_ = 0;
        if (tilingData_->dimBLoop > 1) {
            // 一次搬dimBLoop行
            // 判断下一个循环是否还有计算量
            bOffset_ += tilingData_->dimBLoop;
            if (bOffset_ >= tilingData_->dimB || bOffset_ - bIdx_ >= tilingData_->dimBBlockSize) {
                // 完成任务，退出
                return 0;
            }
            // 与全局尾部距离
            int64_t leftDimB = tilingData_->dimB - bOffset_;
            // 与单核尾部距离
            int64_t leftDimBBlock = tilingData_->dimBBlockSize - (bOffset_ - bIdx_);
            leftDimB = leftDimBBlock < leftDimB ? leftDimBBlock : leftDimB;
            int64_t outerDimB = tilingData_->dimBLoop < leftDimB ? tilingData_->dimBLoop : leftDimB;
            // 下个循环的处理数量
            outerCount_ = outerDimB * tilingData_->dimN;
            outerCountAlign_ = outerDimB * RoundUp(tilingData_->dimN, ELEM_ALIGN);
        } else {
            // 一次搬dimNLoop列
            nOffset_ += tilingData_->dimNLoop;
            // 判断N方向是否还有计算量
            if (nOffset_ >= tilingData_->dimN || nOffset_ - nIdx_ >= tilingData_->dimNBlockSize) {
                // 完成任务，切换到下一行
                nOffset_ = nIdx_;
                bOffset_ += 1;
                if (bOffset_ >= tilingData_->dimB || bOffset_ - bIdx_ >= tilingData_->dimBBlockSize) {
                    return 0;
                }
            }
            // 与全局尾部距离
            int64_t leftDimN = tilingData_->dimN - nOffset_;
            // 与单核尾部距离
            int64_t leftDimNBlock = tilingData_->dimNBlockSize - (nOffset_ - nIdx_);
            leftDimN = leftDimNBlock < leftDimN ? leftDimNBlock : leftDimN;
            int64_t outerDimN = tilingData_->dimNLoop < leftDimN ? tilingData_->dimNLoop : leftDimN;
            // 下个循环的处理数量
            outerCount_ = outerDimN;
            innerCount_ = outerCount_;

            outerCountAlign_ = RoundUp(outerDimN, ELEM_ALIGN);
            innerCountAlign_ = outerCountAlign_;
        }
        return outerCount_;
    }

    __aicore__ inline int64_t InnerNext()
    {
        innerBOffset_ += 1;
        if (innerBOffset_ >= tilingData_->dimBLoop) {
            return 0;
        }
        return innerCount_;
    }

    template <typename T>
    __aicore__ inline void InitBroadcastGlobalTensor(GlobalTensor<T> &gm, GM_ADDR addr)
    {
        gm.SetGlobalBuffer((__gm__ T *)addr, tilingData_->dimB * tilingData_->dimN);
    }

    template <typename T>
    __aicore__ inline void InitNormalGlobalTensor(GlobalTensor<T> &gm, GM_ADDR addr)
    {
        gm.SetGlobalBuffer((__gm__ T *)addr, tilingData_->dimN);
    }

    template <typename T, typename Q>
    __aicore__ inline void InitBroadcastQueue(Q &queue)
    {
        tpipe_.InitBuffer(queue, BUFFER_NUM, outerCountAlign_ * sizeof(T));
    }

    template <typename T, typename Q>
    __aicore__ inline void InitNormalQueue(Q &queue)
    {
        tpipe_.InitBuffer(queue, 1, innerCountAlign_ * sizeof(T));
    }

    template <typename T, typename B>
    __aicore__ inline void InitBroadcastTBuf(B &buf)
    {
        tpipe_.InitBuffer(buf, outerCountAlign_ * sizeof(T));
    }

    template <typename T, typename B>
    __aicore__ inline void InitNormalTBuf(B &buf)
    {
        tpipe_.InitBuffer(buf, innerCountAlign_ * sizeof(T));
    }

    template <typename T, typename Q>
    __aicore__ inline void CopyInBroadcast(GlobalTensor<T> &gm, Q &queue)
    {
        LocalTensor<T> local = queue.template AllocTensor<T>();
        if (tilingData_->dimBLoop > 1) {
            if (tilingData_->dimN * sizeof(T) % BLOCK_SIZE == 0) {
                DataCopy(local, gm[bOffset_ * tilingData_->dimN + nOffset_], outerCount_);
            } else {
#if __CCE_AICORE__ == 220
                DataCopyParams copyParams = { static_cast<uint16_t>(outerCount_ / tilingData_->dimN),
                    static_cast<uint16_t>(tilingData_->dimN * sizeof(T)), 0, 0 };
                DataCopyPad(local, gm[bOffset_ * tilingData_->dimN + nOffset_], copyParams, DataCopyPadParams());
#endif
            }
        } else {
            if (outerCount_ * sizeof(T) % BLOCK_SIZE == 0) {
                DataCopy(local, gm[bOffset_ * tilingData_->dimN + nOffset_], outerCount_);
            } else {
#if __CCE_AICORE__ == 220
                DataCopyParams copyParams = { 1, static_cast<uint16_t>(outerCount_ * sizeof(T)), 0, 0 };
                DataCopyPad(local, gm[bOffset_ * tilingData_->dimN + nOffset_], copyParams, DataCopyPadParams());
#endif
            }
        }

        queue.EnQue(local);
    }

    template <typename T, typename Q>
    __aicore__ inline void CopyInNormal(GlobalTensor<T> &gm, Q &queue)
    {
        LocalTensor<T> local = queue.template AllocTensor<T>();
        if (innerCount_ * sizeof(T) % BLOCK_SIZE == 0) {
            DataCopy(local, gm[nOffset_], innerCount_);
        } else {
#if __CCE_AICORE__ == 220
            DataCopyParams copyParams = { 1, static_cast<uint16_t>(innerCount_ * sizeof(T)), 0, 0 };
            DataCopyPad(local, gm[nOffset_], copyParams, DataCopyPadParams());
#endif
        }

        queue.EnQue(local);
    }

    template <typename T>
    __aicore__ inline LocalTensor<T> GetInnerTensor(LocalTensor<T> &local)
    {
        return local[innerBOffset_ * RoundUp(innerCount_, BLOCK_SIZE / sizeof(T))];
    }

    template <typename T, typename Q>
    __aicore__ inline void CopyOut(GlobalTensor<T> &gm, Q &queue)
    {
        LocalTensor<T> local = queue.template DeQue<T>();
        if (tilingData_->dimBLoop > 1) {
            if (tilingData_->dimN * sizeof(T) % BLOCK_SIZE == 0) {
                DataCopy(gm[bOffset_ * tilingData_->dimN + nOffset_], local, outerCount_);
            } else {
#if __CCE_AICORE__ == 220
                DataCopyParams copyParams = { static_cast<uint16_t>(outerCount_ / tilingData_->dimN),
                    static_cast<uint16_t>(tilingData_->dimN * sizeof(T)), 0, 0 };
                DataCopyPad(gm[bOffset_ * tilingData_->dimN + nOffset_], local, copyParams);
#endif
            }
        } else {
            if (outerCount_ * sizeof(T) % BLOCK_SIZE == 0) {
                DataCopy(gm[bOffset_ * tilingData_->dimN + nOffset_], local, outerCount_);
            } else {
#if __CCE_AICORE__ == 220
                DataCopyParams copyParams = { 1, static_cast<uint16_t>(outerCount_ * sizeof(T)), 0, 0 };
                DataCopyPad(gm[bOffset_ * tilingData_->dimN + nOffset_], local, copyParams);
#endif
            }
        }

        queue.FreeTensor(local);
    }

protected:
    TPipe tpipe_;

    BroadcastTilingData *tilingData_ = nullptr;
    int64_t bIdx_ = 0; // 起始计算的b方向下标
    int64_t bOffset_ = 0; // 当前计算的b方向下标
    int64_t nIdx_ = 0; // 起始计算的n方向下标
    int64_t nOffset_ = 0; // 当前要计算的n方向下标
    int64_t outerCount_ = 0; // 当前准备搬入的元素数量
    int64_t innerCount_ = 0; // 当前准备计算的元素数量
    int64_t innerBOffset_ = 0; // 一次搬dimBLoop行时，用于计算的b下标计数

    int64_t outerCountAlign_ = 0; // outerCount 32字节对齐
    int64_t innerCountAlign_ = 0; // innerCount 32字节对齐
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, BroadcastTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->dimB = (*(const __gm__ int64_t *)(p_tilingdata + 0));
    tilingdata->dimN = (*(const __gm__ int64_t *)(p_tilingdata + 8));
    tilingdata->dimBBlockNum = (*(const __gm__ int64_t *)(p_tilingdata + 16));
    tilingdata->dimNBlockNum = (*(const __gm__ int64_t *)(p_tilingdata + 24));
    tilingdata->dimBBlockSize = (*(const __gm__ int64_t *)(p_tilingdata + 32));
    tilingdata->dimNBlockSize = (*(const __gm__ int64_t *)(p_tilingdata + 40));
    tilingdata->dimBLoop = (*(const __gm__ int64_t *)(p_tilingdata + 48));
    tilingdata->dimNLoop = (*(const __gm__ int64_t *)(p_tilingdata + 56));
#else
    TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(BroadcastTilingData), &pipe);
    PipeBarrier<PIPE_ALL>();
    tilingdata->dimB = (*(__ubuf__ int64_t *)(tilingdata_in_ub + 0));
    tilingdata->dimN = (*(__ubuf__ int64_t *)(tilingdata_in_ub + 8));
    tilingdata->dimBBlockNum = (*(__ubuf__ int64_t *)(tilingdata_in_ub + 16));
    tilingdata->dimNBlockNum = (*(__ubuf__ int64_t *)(tilingdata_in_ub + 24));
    tilingdata->dimBBlockSize = (*(__ubuf__ int64_t *)(tilingdata_in_ub + 32));
    tilingdata->dimNBlockSize = (*(__ubuf__ int64_t *)(tilingdata_in_ub + 40));
    tilingdata->dimBLoop = (*(__ubuf__ int64_t *)(tilingdata_in_ub + 48));
    tilingdata->dimNLoop = (*(__ubuf__ int64_t *)(tilingdata_in_ub + 56));
    PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tilingData, tilingArg)                                                                         \
    BroadcastTilingData tilingData;                                                                              \
    InitTilingData((tilingArg), &(tilingData))

#endif