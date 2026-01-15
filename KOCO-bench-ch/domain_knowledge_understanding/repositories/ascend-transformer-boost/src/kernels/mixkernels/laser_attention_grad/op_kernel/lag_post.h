/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef __LAG_POST_H__
#define __LAG_POST_H__

#ifdef __DAV_C220_VEC__

#include "ppmatmul_const_grad.h"
#include "kernel_operator.h"
#include "kernels/utils/kernel/common.h"
#include "kernels/utils/kernel/utils.h"
#include "kernels/utils/kernel/iterator.h"
#include "kernels/utils/kernel/simd.h"

namespace LAG_POST {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template<typename INPUT_T, bool IF_BF16>
class LagPost {
public:
    __aicore__ inline LagPost() {}
    __aicore__ inline void Init(__gm__ float * __restrict__ dqWorkspace,
                                __gm__ float * __restrict__ dkWorkspace,
                                __gm__ float * __restrict__ dvWorkspace,
                                __gm__ INPUT_T * __restrict__ dq,
                                __gm__ INPUT_T * __restrict__ dk,
                                __gm__ INPUT_T * __restrict__ dv,
                                int64_t postBaseNum,
                                int64_t postDqFrontCoreNum,
                                int64_t postDqFrontDataNum,
                                int64_t postDqTailDataNum,
                                int64_t postDkFrontCoreNum,
                                int64_t postDkFrontDataNum,
                                int64_t postDkTailDataNum,
                                int64_t postDvFrontCoreNum,
                                int64_t postDvFrontDataNum,
                                int64_t postDvTailDataNum,
                                float scale);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SetAllFlags();
    __aicore__ inline void WaitAllFlags();
    __aicore__ inline void BeforeCompute();
    __aicore__ inline void AfterCompute();
    __aicore__ inline void Compute(GlobalTensor<INPUT_T> gm, GlobalTensor<float> workspace, uint64_t start,
                                   uint64_t actualSize, bool needMuls);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> fp32Buf;
    TBuf<QuePosition::VECCALC> bf16Buf;

    LocalTensor<float> dqUbFp32, dkUbFp32, dvUbFp32;
    LocalTensor<INPUT_T> dqUbBf16, dkUbBf16, dvUbBf16;

    GlobalTensor<float> dqWorkspaceFp32, dkWorkspaceFp32, dvWorkspaceFp32;
    GlobalTensor<INPUT_T> dqGm, dkGm, dvGm;

    DataCopyExtParams copyExtParams = {1, 0, 0, 0, 0};
    DataCopyPadExtParams<float> copyPadExtParams = {false, 0, 0, 0};

    float scale;
    int32_t thisCoreIdx;
    int64_t baseNum;
    int64_t dqDataNum, dqDataStart, dkDataNum, dkDataStart, dvDataNum, dvDataStart;
    event_t eventIdVMte2 = EVENT_ID0;
    event_t eventIdVMte3 = EVENT_ID0;

    int64_t pingPongFlag = 0;
};

template<typename INPUT_T, bool IF_BF16>
__aicore__ inline void LagPost<INPUT_T, IF_BF16>::SetAllFlags()
{
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
}

template<typename INPUT_T, bool IF_BF16>
__aicore__ inline void LagPost<INPUT_T, IF_BF16>::WaitAllFlags()
{
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
}

template<typename INPUT_T, bool IF_BF16>
__aicore__ inline void LagPost<INPUT_T, IF_BF16>::BeforeCompute()
{
    this->eventIdVMte2 = pingPongFlag ? EVENT_ID0 : EVENT_ID1;
    this->eventIdVMte3 = pingPongFlag ? EVENT_ID0 : EVENT_ID1;
    WaitFlag<HardEvent::V_MTE2>(this->eventIdVMte2);
    WaitFlag<HardEvent::MTE3_V>(this->eventIdVMte3);
}

template<typename INPUT_T, bool IF_BF16>
__aicore__ inline void LagPost<INPUT_T, IF_BF16>::AfterCompute()
{
    SetFlag<HardEvent::V_MTE2>(this->eventIdVMte2);
    SetFlag<HardEvent::MTE3_V>(this->eventIdVMte3);
    this->pingPongFlag = 1 - this->pingPongFlag;
}

template<typename INPUT_T, bool IF_BF16>
__aicore__ inline void LagPost<INPUT_T, IF_BF16>::Init(
                                __gm__ float * __restrict__ dqWorkspace,
                                __gm__ float * __restrict__ dkWorkspace,
                                __gm__ float * __restrict__ dvWorkspace,
                                __gm__ INPUT_T * __restrict__ dq,
                                __gm__ INPUT_T * __restrict__ dk,
                                __gm__ INPUT_T * __restrict__ dv,
                                int64_t postBaseNum,
                                int64_t postDqFrontCoreNum,
                                int64_t postDqFrontDataNum,
                                int64_t postDqTailDataNum,
                                int64_t postDkFrontCoreNum,
                                int64_t postDkFrontDataNum,
                                int64_t postDkTailDataNum,
                                int64_t postDvFrontCoreNum,
                                int64_t postDvFrontDataNum,
                                int64_t postDvTailDataNum,
                                float scale)
{
    this->thisCoreIdx = GetBlockIdx();
    this->scale = scale;

    this->baseNum = postBaseNum;
    this->dqDataNum = this->thisCoreIdx < postDqFrontCoreNum ? postDqFrontDataNum : postDqTailDataNum;
    this->dqDataStart = this->thisCoreIdx < postDqFrontCoreNum ?
                        postDqFrontDataNum * this->thisCoreIdx :
                        postDqFrontDataNum * postDqFrontCoreNum +
                        postDqTailDataNum * (this->thisCoreIdx - postDqFrontCoreNum);
    
    this->dkDataNum = this->thisCoreIdx < postDkFrontCoreNum ? postDkFrontDataNum : postDkTailDataNum;
    this->dkDataStart = this->thisCoreIdx < postDkFrontCoreNum ?
                        postDkFrontDataNum * this->thisCoreIdx :
                        postDkFrontDataNum * postDkFrontCoreNum +
                        postDkTailDataNum * (this->thisCoreIdx - postDkFrontCoreNum);

    this->dvDataNum = this->thisCoreIdx < postDvFrontCoreNum ? postDvFrontDataNum : postDvTailDataNum;
    this->dvDataStart = this->thisCoreIdx < postDvFrontCoreNum ?
                        postDvFrontDataNum * this->thisCoreIdx :
                        postDvFrontDataNum * postDvFrontCoreNum +
                        postDvTailDataNum * (this->thisCoreIdx - postDvFrontCoreNum);

    dqGm.SetGlobalBuffer(dq);
    dkGm.SetGlobalBuffer(dk);
    dvGm.SetGlobalBuffer(dv);
    dqWorkspaceFp32.SetGlobalBuffer(dqWorkspace);
    dkWorkspaceFp32.SetGlobalBuffer(dkWorkspace);
    dvWorkspaceFp32.SetGlobalBuffer(dvWorkspace);

    pipe.InitBuffer(fp32Buf, BUFFER_NUM * baseNum * sizeof(float)); // ping: 前baseNum个；pong: 后baseNum个
    pipe.InitBuffer(bf16Buf, BUFFER_NUM * baseNum * sizeof(INPUT_T)); // ping: 前baseNum个；pong: 后baseNum个
}

template<typename INPUT_T, bool IF_BF16>
__aicore__ inline void LagPost<INPUT_T, IF_BF16>::Process()
{
    // compute dq
    SetAllFlags();
    for (uint64_t i = this->dqDataStart; i < this->dqDataStart + this->dqDataNum; i = i + this->baseNum) {
        BeforeCompute();
        uint64_t actualSize = i + this->baseNum < this->dqDataStart + this->dqDataNum ?
                              this->baseNum : this->dqDataStart + this->dqDataNum - i;
        Compute(dqGm, dqWorkspaceFp32, i, actualSize, true);
        AfterCompute();
    }
    WaitAllFlags();

    // compute dk
    SetAllFlags();
    for (uint64_t i = this->dkDataStart; i < this->dkDataStart + this->dkDataNum; i = i + this->baseNum) {
        BeforeCompute();
        uint64_t actualSize = i + this->baseNum < this->dkDataStart + this->dkDataNum ?
                              this->baseNum : this->dkDataStart + this->dkDataNum - i;
        Compute(dkGm, dkWorkspaceFp32, i, actualSize, true);
        AfterCompute();
    }
    WaitAllFlags();

    // compute dv
    SetAllFlags();
    for (uint64_t i = this->dvDataStart; i < this->dvDataStart + this->dvDataNum; i = i + this->baseNum) {
        BeforeCompute();
        uint64_t actualSize = i + this->baseNum < this->dvDataStart + this->dvDataNum ?
                              this->baseNum : this->dvDataStart + this->dvDataNum - i;
        Compute(dvGm, dvWorkspaceFp32, i, actualSize, false);
        AfterCompute();
    }
    WaitAllFlags();
}

template<typename INPUT_T, bool IF_BF16>
__aicore__ inline void LagPost<INPUT_T, IF_BF16>::Compute(
    GlobalTensor<INPUT_T> gm, GlobalTensor<float> workspace, uint64_t start,
    uint64_t actualSize, bool needMuls)
{
    int32_t ubOffset = pingPongFlag ? 0 : this->baseNum;
    LocalTensor<float> fp32Ub = this->fp32Buf.template Get<float>();
    LocalTensor<INPUT_T> bf16Ub = this->bf16Buf.template Get<INPUT_T>();

    // CopyIn
    this->copyExtParams.blockLen = static_cast<uint32_t>(actualSize * sizeof(float));
    DataCopyPad(fp32Ub[ubOffset], workspace[start], this->copyExtParams, this->copyPadExtParams);
    SetFlag<HardEvent::MTE2_V>(this->eventIdVMte2);

    // Compute
    WaitFlag<HardEvent::MTE2_V>(this->eventIdVMte2);
    if (needMuls) {
        Muls(fp32Ub[ubOffset], fp32Ub[ubOffset], this->scale, actualSize);
        pipe_barrier(PIPE_V);
    }
    Cast(bf16Ub[ubOffset], fp32Ub[ubOffset], RoundMode::CAST_RINT, actualSize);
    SetFlag<HardEvent::V_MTE3>(this->eventIdVMte3);

    // CopyOut
    WaitFlag<HardEvent::V_MTE3>(this->eventIdVMte3);
    this->copyExtParams.blockLen = static_cast<uint32_t>(actualSize * sizeof(INPUT_T));
    DataCopyPad(gm[start], bf16Ub[ubOffset], this->copyExtParams);
}
}
#endif

#endif