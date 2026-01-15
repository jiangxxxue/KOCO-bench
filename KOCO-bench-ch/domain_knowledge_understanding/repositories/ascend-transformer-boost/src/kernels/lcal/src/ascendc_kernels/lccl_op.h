/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCCL_OP_H
#define LCCL_OP_H

#if defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)

#include "op_def.h"
#include "allgather.h"
#include "91093/allgather_hierarchy_double_ring.h"
#include "allreduce_one_shot.h"
#include "allreduce_two_shot.h"
#include "allreduce_big_data.h"
#include "91093/allreduce_big_data_sio.h"
#include "91093/allreduce_hierarchy_double_ring.h"
#include "reduce_scatter.h"
#include "91093/reduce_scatter_big_data_91093_4step.h"
#include "91093/reduce_scatter_hierarchy_double_ring.h"
#include "91093/all2all_hierarchy.h"
#include "91093/all2all_hierarchy_small.h"

#include "../kernels/lcal_allreduce_2npu_read.cce"
#include "../kernels/lcal_allreduce_2npu_write.cce"
#include "../kernels/lcal_allreduce_2npu_big_write.cce"
#include "../kernels/lcal_allreduce_two_shot.cce"
#include "../kernels/lcal_allreduce_big_data.cce"
#include "../kernels/lcal_allreduce_two_shot_910B2C.cce"
#include "../kernels/lcal_allreduce_big_data_910B2C.cce"
#include "../kernels/lcal_allreduce_deterministic.cce"
#include "../kernels/lcal_allreduce_deterministic_big_data.cce"
#include "../kernels/lcal_reduce_scatter_big_data_write.cce"
#include "../kernels/lcal_reduce_scatter_write.cce"
#include "../kernels/lcal_reduce_scatter.cce"
#include "../kernels/lcal_reduce_scatter_big_data.cce"
#include "../kernels/lcal_allgather_910B2C.cce"
#include "../kernels/lcal_allgather_big_data_910B2C.cce"
#include "../kernels/lcal_allgather_2npu.cce"
#include "../kernels/lcal_allgather_2npu_big_data_write.cce"
#include "../kernels/lcal_allgather.cce"
#include "../kernels/lcal_allgather_big_data.cce"
#include "../kernels/lcal_broadcast_write.cce"
#include "../kernels/lcal_broadcast_big_data.cce"
#include "../kernels/lcal_all2all_transpose.cce"

#define CLASS_OP_910B_RDMA_LAUNCH(name, type) \
do { \
name<type> opKernel(localRank, localRankSize, extraFlag); \
opKernel.Init(KERNELS_ARGS_CALL()); \
opKernel.Process();                 \
} while (0)

#define CLASS_OP_QUANT_LAUNCH(name, outputType, inputType) \
do { \
name<outputType, inputType> opKernel(localRank, localRankSize, extraFlag); \
opKernel.Init(KERNELS_ARGS_CALL()); \
opKernel.Process();                 \
} while (0)

extern "C" __global__ __aicore__ __attribute__((section("Attr_Section_Lcal"))) void LcalDescriptor() {}

#define LCCL_BROADCAST_FUNC_AUTO_DEF(suffix) \
extern "C" __global__ __aicore__ void LcalBroadcast##suffix(KERNELS_ARGS_FUN()) \
{ \
    if ASCEND_IS_AIV { \
    GET_COMM_ARGS; \
    __gm__ char * shareAddrs[LCAL_MAX_RANK_SIZE]; \
    GET_IPC_MEM_ARGS(char); \
    if ((extraFlag & ExtraFlag::TOPO_PCIE) != 0) { \
        LcalBroadcast2npuBigDataWrite(ALLREDUCE_ARGS_CALL(char));   \
    } else { \
        LcalBroadcastBigData(ALLREDUCE_ARGS_CALL(char));   \
    } \
    } \
}

#define LCCL_ALLGATHER_FUNC_AUTO_DEF(type, suffix) \
extern "C" __global__ __aicore__ void LcalAllGather_##type##suffix(KERNELS_ARGS_FUN()) { \
    if ASCEND_IS_AIV { \
    GET_COMM_ARGS; \
    constexpr int32_t quickOneshotRankSize = 2; \
    constexpr int32_t cceSmallDataSize = 2 * 1024 * 1024; \
    constexpr int32_t smallRankSize = 8; \
    constexpr int32_t smallDataSize910a3 = 32 * 1024 * 1024; \
    __gm__ type * shareAddrs[LCAL_MAX_RANK_SIZE]; \
    GET_IPC_MEM_ARGS(type); \
    if ((extraFlag & ExtraFlag::TOPO_910B2C) != 0 && rankSize > smallRankSize) { \
        if (len * sizeof(type) < cceSmallDataSize) {      \
            LcalAllGather910B2C<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } else { \
            LcalAllGatherBigData910B2C<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } \
    } else if ((extraFlag & ExtraFlag::TOPO_PCIE) != 0) { \
        LcalAllGather2npuBigDataWrite<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
    } else if ((extraFlag & ExtraFlag::TOPO_910_93) != 0 && lcalBlockNum != rankSize && \
        (len > smallDataSize910a3 / sizeof(type) || rankSize > smallRankSize) && \
        rankSize > quickOneshotRankSize && rankSize % quickOneshotRankSize == 0) { \
        CLASS_OP_LAUNCH(AllGatherHierarchyDoubleRing, type); \
    } else { \
        if (rankSize == quickOneshotRankSize && len * sizeof(type) < SIZE_OF_8M && lcalBlockNum != rankSize) { \
            LcalAllGather2npu<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } else if (rankSize == quickOneshotRankSize && lcalBlockNum != rankSize) { \
            LcalAllGather2npuBigDataWrite<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        }  else if (rankSize > quickOneshotRankSize && len * sizeof(type) < cceSmallDataSize || \
            lcalBlockNum == rankSize) { \
            LcalAllGather<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } else { \
            LcalAllGatherBigData<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } \
    } \
    } \
}

#define LCCL_ALL_REDUCE_FUNC_AUTO_DEF(type, suffix) \
extern "C" __global__ __aicore__ void LcalAllReduce_##type##suffix(KERNELS_ARGS_FUN()) { \
    if ASCEND_IS_AIV { \
    GET_COMM_ARGS; \
    constexpr int32_t quickOneshotRankSize = 2; \
    constexpr int32_t threeStepNum = 3; \
    constexpr int32_t smallRankSize = 8; \
    constexpr int32_t oneshotDataSize = 16 * 1024; \
    constexpr int64_t quantSmallDataSize = 512 * 1024; \
    constexpr int32_t cceSmallDataSize = 2 * 1024 * 1024; \
    constexpr int32_t smallDataSize910a3 = 32 * 1024 * 1024; \
    constexpr int32_t rankSize910a3 = 16; \
    constexpr int32_t scaleCountMax = 12 * 1024 * 1024; \
    __gm__ type * shareAddrs[LCAL_MAX_RANK_SIZE]; \
    GET_IPC_MEM_ARGS(type); \
    if ((extraFlag & ExtraFlag::TOPO_PCIE) != 0) { \
        if (len * sizeof(type) < SIZE_OF_8M) {      \
            LcalAllReduce2npuWrite<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } else { \
            LcalAllReduce2npuBigDataWrite<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } \
    } else if ((extraFlag & ExtraFlag::QUANT_FP16) != 0 && std::is_same_v<type, int8_t>) { \
        if (len * sizeof(type) <= oneshotDataSize) { \
            CLASS_OP_QUANT_LAUNCH(AllReduceOneShot, half, int8_t); \
        } else if (len * sizeof(type) <= quantSmallDataSize) { \
            CLASS_OP_QUANT_LAUNCH(AllReduceTwoShot, half, int8_t); \
        } else if (scaleCount * rankSize <= scaleCountMax) { \
            CLASS_OP_QUANT_LAUNCH(AllReduceBigData, half, int8_t); \
        } else { \
            AllReduceBigData<half, int8_t> opTmp(localRank, localRankSize, extraFlag); \
            opTmp.Init(KERNELS_ARGS_CALL()); \
            opTmp.SupportBigScale(); \
            input = output; \
            CLASS_OP_LAUNCH(AllReduceBigData, half); \
        } \
    } else if ((extraFlag & ExtraFlag::TOPO_910B2C) != 0 && rankSize > smallRankSize) { \
        if (len * sizeof(type) < cceSmallDataSize) {      \
            LcalAllReduceTwoShot910B2C<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } else { \
            LcalAllReduceBigData910B2C<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } \
    } else if ((extraFlag & ExtraFlag::DETERMINISTIC) != 0) { \
        constexpr uint32_t maxAivNum = 40; \
        const bool isAivNumSupport = ((extraFlag & ExtraFlag::IS_GREATER_THAN_40_AIV) != 0 ||  \
            rankSize * threeStepNum <= maxAivNum); \
        if ((extraFlag & ExtraFlag::TOPO_910_93) != 0) { \
            if (rankSize % quickOneshotRankSize == 1 || rankSize == quickOneshotRankSize || \
                (rankSize <= rankSize910a3 && len * sizeof(type) <= smallDataSize910a3 && isAivNumSupport)) { \
                LcalAllReduceDeterministicBigData<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
            } else { \
                CLASS_OP_LAUNCH(AllReduceHierarchyDoubleRing, type); \
            } \
        } else if (len * sizeof(type) < SMALL_DATA_SIZE) { \
            LcalAllReduceDeterministic<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } else { \
            LcalAllReduceDeterministicBigData<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } \
    } else if ((extraFlag & ExtraFlag::TOPO_910_93) != 0 && lcalBlockNum != rankSize && \
        (rankSize == quickOneshotRankSize || len * sizeof(type) > smallDataSize910a3)) { \
        if (rankSize == quickOneshotRankSize) { \
            LcalAllReduce2npuBigDataWrite<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } else if (rankSize % quickOneshotRankSize == 0) { \
            CLASS_OP_LAUNCH(AllReduceHierarchyDoubleRing, type); \
        } else { \
            CLASS_OP_LAUNCH(AllReduceBigDataSio, type); \
        } \
    } else { \
        if (len * sizeof(type) < cceSmallDataSize or lcalBlockNum == rankSize) {      \
            if (rankSize == quickOneshotRankSize && lcalBlockNum != rankSize) { \
                LcalAllReduce2npuRead<type>(ALLREDUCE_ARGS_CALL(type)); \
            } else { \
            LcalAllReduceTwoShot<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
            } \
        } else { \
            LcalAllReduceBigData<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } \
    } \
    } \
}

#define LCCL_ALL2ALL_FUNC_AUTO_DEF(type, suffix) \
extern "C" __global__ __aicore__ void LcalAll2All_##type##suffix(KERNELS_ARGS_FUN()) { \
    if ASCEND_IS_AIV { \
    GET_COMM_ARGS; \
    __gm__ type * shareAddrs[LCAL_MAX_RANK_SIZE];  \
    GET_IPC_MEM_ARGS(type); \
    constexpr int32_t smallRankSize = 8; \
    if (op != 0 && root != 0) {  \
        LcalAll2AllTranspose<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
    } \
    else if ((extraFlag & ExtraFlag::TOPO_910_93) != 0) { \
        if (rankSize <= smallRankSize && len * sizeof(type) > SMALL_DATA_SIZE && \
            (len * sizeof(type)) % (smallRankSize * smallRankSize * rankSize) == 0) { \
            CLASS_OP_LAUNCH(All2AllHierarchySmall, type); \
    } else { \
            CLASS_OP_LAUNCH(All2AllHierarchy, type); \
        } \
    } \
    } \
}

#define LCCL_REDUCE_SCATTER_FUNC_AUTO_DEF(type, suffix) \
extern "C" __global__ __aicore__ void LcalReduceScatter_##type##suffix(KERNELS_ARGS_FUN()) { \
    if ASCEND_IS_AIV { \
    GET_COMM_ARGS; \
    constexpr int32_t quickOneshotRankSize = 2; \
    constexpr int32_t cceSmallDataSize = 2 * 1024 * 1024; \
    constexpr int32_t a3BigDataSize = 32 * 1024 * 1024; \
    constexpr int32_t a3SupportRankSize = 4; \
    constexpr int32_t smallRankSize = 8; \
    const bool isDbRing = (rankSize == a3SupportRankSize || rankSize == smallRankSize) &&  \
        (len * sizeof(type) * smallRankSize > cceSmallDataSize && \
        len * sizeof(type) * smallRankSize <= a3BigDataSize); \
    __gm__ type * shareAddrs[LCAL_MAX_RANK_SIZE]; \
    GET_IPC_MEM_ARGS(type); \
    if ((extraFlag & ExtraFlag::TOPO_PCIE) != 0) { \
        LcalReduceScatterBigDataWrite<type>(ALLREDUCE_ARGS_CALL(type)); \
    } else if ((extraFlag & ExtraFlag::TOPO_910_93) != 0 && (rankSize > smallRankSize || isDbRing)) { \
        if (isDbRing) { \
            CLASS_OP_LAUNCH(ReduceScatterHierarchyDoubleRing, type); \
        } else if (len * sizeof(type) <= SMALL_DATA_SIZE) { \
            CLASS_OP_LAUNCH(ReduceScatter, type); \
        } else { \
            CLASS_OP_LAUNCH(ReduceScatterBigData91093, type); \
        } \
    } else { \
        if (rankSize == quickOneshotRankSize && len * sizeof(type) < SIZE_OF_8M) { \
            LcalReduceScatterWrite<type>(ALLREDUCE_ARGS_CALL(type)); \
        }  else if (rankSize > quickOneshotRankSize && len * sizeof(type) < cceSmallDataSize){ \
            LcalReduceScatter<type>(ALLREDUCE_ARGS_CALL(type)); \
        } else { \
            LcalReduceScatterBigData<type>(ALLREDUCE_ARGS_CALL(type)); \
        } \
    } \
    } \
}
#endif
#endif