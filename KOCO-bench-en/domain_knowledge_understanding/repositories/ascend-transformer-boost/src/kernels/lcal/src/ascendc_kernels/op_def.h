/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCCL_OP_DEF_H
#define LCCL_OP_DEF_H
#define GET_COMM_ARGS \
    GlobalTensor<int> commArgsGm; \
    commArgsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(commArgs), 5); \
    int rank = commArgsGm.GetValue(0); \
    int localRank = commArgsGm.GetValue(1); \
    int rankSize = commArgsGm.GetValue(2); \
    int localRankSize = commArgsGm.GetValue(3); \
    uint32_t extraFlag = commArgsGm.GetValue(4); \
    GM_ADDR dumpAddr = (reinterpret_cast<__gm__ CommArgs *>(commArgs))->dumpAddr; \
    int32_t lcalBlockNum = GetBlockNum()

#ifdef ENABLE_LCCL_MIX
#define SET_MAGIC \
do { \
    __gm__ CommArgs * commArgsTmp = reinterpret_cast<__gm__ CommArgs *>(commArgs); \
    PipeBarrier<PIPE_ALL>(); \
    SetAtomicNone(); \
    SetMaskNormImpl(); \
    SetSyncBaseAddr(commArgsTmp->fftsVal); \
    SetVectorMask<int32_t>((uint64_t)-1, (uint64_t)-1); \
    PipeBarrier<PIPE_ALL>(); \
    LocalTensor<int32_t> localSet; \
    localSet.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN); \
    localSet.address_.bufferAddr = reinterpret_cast<uint64_t>((__ubuf__ int32_t *)96); \
    GlobalTensor<int32_t> magicGt; \
    magicGt.SetGlobalBuffer((__gm__ int32_t *)commArgsTmp->magics); \
    if (GetBlockIdx() == 0) { \
        SetAtomicOpType<int32_t>(Op::ADD); \
        localSet.SetValue(0, 1); \
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0); \
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0); \
        DataCopyExtParams dataCopyParams(1, sizeof(int32_t), 0, 0, 0); \
        DataCopyPad(magicGt[rankSize - 1], localSet, dataCopyParams); \
        AscendC::SetAtomicNone(); \
        PipeBarrier<PIPE_ALL>(); \
    } \
    SyncAll<true>(); \
    DataCopyExtParams dataCopyParams(1, sizeof(int32_t), 0, 0, 0); \
    DataCopyPadExtParams<int32_t> padParams; \
    DataCopyPad(localSet, magicGt[rankSize - 1], dataCopyParams, padParams); \
    AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0); \
    AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0); \
    magic = static_cast<int64_t>(localSet.GetValue(0)); \
    PipeBarrier<PIPE_ALL>(); \
    constexpr int32_t aivNumPerAic = 2; \
    lcalBlockNum = GetBlockNum() * aivNumPerAic; \
} while (0)
#else
#define SET_MAGIC \
do {} while (0)
#endif

#define GET_IPC_MEM_ARGS(type) \
do { \
    SET_MAGIC; \
    GlobalTensor<GM_ADDR> peerMemsAddrGm; \
    peerMemsAddrGm.SetGlobalBuffer(&(reinterpret_cast<__gm__ CommArgs *>(commArgs))->peerMems[0], \
                       LCAL_MAX_RANK_SIZE); \
    for (int i = 0; i < rankSize; ++i) { \
        shareAddrs[i] = (__gm__ type *) (peerMemsAddrGm.GetValue(i) + \
                        (magic % PING_PONG_SIZE) * (IPC_BUFF_MAX_SIZE + IPC_DATA_OFFSET)); \
    } \
    AscendC::PipeBarrier<PIPE_ALL>(); \
} while (0) \

#define CLASS_OP_LAUNCH(name, type) \
do { \
    name<type> opKernel(rank, rankSize, extraFlag); \
    opKernel.Init(KERNELS_ARGS_CALL()); \
    opKernel.Process();                 \
} while (0)

#define CLASS_OP_QUANT_910A5_LAUNCH(name, outputType, addType, inputType) \
do { \
    name<outputType, addType, inputType> opKernel(rank, rankSize, extraFlag); \
    opKernel.Init(KERNELS_ARGS_CALL()); \
    opKernel.Process();                 \
} while (0)

#define LCCL_TYPE_FUNC(fun) \
    fun(int,);fun(int8_t,);fun(int16_t,);fun(int64_t,); \
    fun(float,);fun(float16_t,);fun(bfloat16_t,)

#ifdef ENABLE_LCCL_MIX
#define LCCL_TYPE_AIC_FUNC(fun) \
    fun(int, _mix_aic); fun(int8_t, _mix_aic); fun(int16_t, _mix_aic); fun(int64_t, _mix_aic); \
    fun(float, _mix_aic); fun(float16_t, _mix_aic); fun(bfloat16_t, _mix_aic)

#define LCCL_TYPE_AIV_FUNC(fun) \
    fun(int, _mix_aiv); fun(int8_t, _mix_aiv); fun(int16_t, _mix_aiv); fun(int64_t, _mix_aiv); \
    fun(float, _mix_aiv); fun(float16_t, _mix_aiv); fun(bfloat16_t, _mix_aiv)
#else
#define LCCL_TYPE_AIC_FUNC(fun) \
    (void)0

#define LCCL_TYPE_AIV_FUNC(fun) \
    fun(int,); fun(int8_t,); fun(int16_t,); fun(int64_t,); \
    fun(float,); fun(float16_t,); fun(bfloat16_t,)
#endif

#define LCCL_VADD_910B_TYPE_FUNC(fun) \
    fun(int);fun(int16_t); \
    fun(float);fun(float16_t)

#define LCCL_QUANT_LOW_TYPE_FUNC(fun) \
    fun(int8_t)

#endif