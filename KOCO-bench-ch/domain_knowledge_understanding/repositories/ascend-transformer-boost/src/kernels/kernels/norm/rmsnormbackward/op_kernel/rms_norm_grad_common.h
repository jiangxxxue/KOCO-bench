/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_RMS_NORM_GRAD_BASE_H
#define ASCEND_OPS_RMS_NORM_GRAD_BASE_H
#include "kernel_operator.h"
#include "kernels/norm/rmsnormbackward/tiling/tiling_data.h"

using namespace AsdOps;
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM_DB = 2;
constexpr uint32_t FLOAT_DTYPE = 0;
constexpr uint32_t FLOAT16_DTYPE = 1;
constexpr uint32_t BFLOAT16_DTYPE = 2;
constexpr uint32_t ALIGN_32 = 8;
constexpr uint32_t ALIGN_16 = 16;
constexpr uint32_t CORE_NUM = 50;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t EACH_CORE_HANDLE_NUM = BLOCK_SIZE / sizeof(int32_t);
const int32_t CAL_ONE_BLOCK_FP32 = 8;
const uint32_t REDUCESUMTHRESHOLD = 64;
const uint32_t SMALLD_THRESHOLD = 640;
const uint32_t WORK_SPACE_TWICE_SIZE = 2;

__aicore__ inline void WaitInitOutput(GM_ADDR sync, GlobalTensor<int32_t> syncGm_,
                                      TQue<QuePosition::VECIN, BUFFER_NUM> inQueGamma, uint32_t blockDim)
{
    int32_t BlockNum = AscendC::GetBlockNum();
    syncGm_.SetGlobalBuffer((__gm__ int32_t *)(sync),
                            BlockNum * BLOCK_SIZE * WORK_SPACE_TWICE_SIZE + BLOCK_SIZE);
    uint64_t blockIdx = AscendC::GetBlockIdx();
    if (blockIdx == 0) {
        InitOutput<int32_t>(syncGm_,
                            BlockNum * CAL_ONE_BLOCK_FP32 * WORK_SPACE_TWICE_SIZE +
                            CAL_ONE_BLOCK_FP32, 0);
        auto syncBuf = inQueGamma.AllocTensor<int32_t>();
        for (uint32_t blockDimI = 1; blockDimI <= blockDim - 1; blockDimI++) {
            AscendC::IBSet(syncGm_, syncBuf, blockDimI, 0);
        }
        inQueGamma.FreeTensor(syncBuf);
    } else {
        auto syncBuf = inQueGamma.AllocTensor<int32_t>();
        AscendC::IBWait(syncGm_, syncBuf, blockIdx, 0);
        inQueGamma.FreeTensor(syncBuf);
    }
}

__aicore__ inline void ReduceSumFP32(uint32_t idx, LocalTensor<float>& dstLocal,
                                     const LocalTensor<float>& srcLocal,
                                     const LocalTensor<float>& workLocal,
                                     int32_t count, uint32_t colValAlign)
{
    if (g_coreType == AIV) {
        int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(float);
        uint64_t mask = elementNumPerRep;
        int32_t repeatTimes = count / elementNumPerRep;
        int32_t tailCount = count % elementNumPerRep;
        int32_t bodyCount = repeatTimes * elementNumPerRep;
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = 0;
        repeatParams.dstBlkStride = 1;
        Duplicate(workLocal, 0.0f, elementNumPerRep);
        pipe_barrier(PIPE_V);
        int32_t startAddr = idx * colValAlign;
        if (likely(repeatTimes > 0)) {
            Add(workLocal, srcLocal[startAddr], workLocal, mask, repeatTimes, repeatParams);
            pipe_barrier(PIPE_V);
        }
        if (unlikely(tailCount != 0)) {
            Add(workLocal, srcLocal[startAddr + bodyCount], workLocal, tailCount, 1, repeatParams);
            pipe_barrier(PIPE_V);
        }
        AscendCUtils::SetMask<float>(elementNumPerRep);
        vcadd((__ubuf__ float*)dstLocal[startAddr].GetPhyAddr(),
              (__ubuf__ float*)workLocal.GetPhyAddr(), 1, 0, 1, 0,
              false);
        pipe_barrier(PIPE_V);
    }
}

__aicore__ inline int64_t RmsNormGradGetAccVal()
{
    return get_acc_val();
}

/*
 * only support count <= 255 * 64 = 16320
 */
__aicore__ inline float ReduceSumFP32_V2(const LocalTensor<float>& srcLocal, int32_t count)
{
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(float);
    int32_t repeatTimes = count / elementNumPerRep;
    int32_t tailCount = count % elementNumPerRep;
    int32_t bodyCount = repeatTimes * elementNumPerRep;
    float value = 0.0;
    if (g_coreType == AIV) {
        if (likely(repeatTimes > 0)) {
            AscendCUtils::SetMask<float>(elementNumPerRep);
            vcadd(nullptr, (__ubuf__ float*)srcLocal.GetPhyAddr(), repeatTimes, 1, 1, 8, true);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
#ifdef __CCE_KT_TEST__
            uint64_t accVal = get_acc_val();
#else
            uint64_t accVal = RmsNormGradGetAccVal();
#endif
            value = *reinterpret_cast<float*>(&accVal);
        }
        if (unlikely(tailCount != 0)) {
            AscendCUtils::SetMask<float>(tailCount);
            vcadd(nullptr, (__ubuf__ float*)srcLocal[bodyCount].GetPhyAddr(), 1, 1, 1, 8, true);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
#ifdef __CCE_KT_TEST__
            uint64_t accVal = get_acc_val();
#else
            auto accVal = RmsNormGradGetAccVal();
#endif
            value += *reinterpret_cast<float*>(&accVal);
        }
    }
    return value;
}

#endif // ASCEND_OPS_RMS_NORM_GRAD_BASE_H