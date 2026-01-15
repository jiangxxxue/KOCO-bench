/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_COMM_ARGS_H
#define LCCL_COMM_ARGS_H
#include <cstdint>

#if !defined(__DAV_C220_VEC__) && !defined(__DAV_C310__)
using GM_ADDR = uint8_t*;
#else
#define FORCE_INLINE_AICORE __attribute__((always_inline)) inline __aicore__
#include "kernel_operator.h"
#endif
namespace Lcal {

constexpr int LCAL_MAX_RANK_SIZE = 128; // lcal通信库最大支持的npu卡数
constexpr int RANK_SIZE_TWO = 2;  // 可用SIO的规模，以及是否需要跨卡搬运数据核的分界规模
constexpr int64_t IPC_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr int64_t IPC_DATA_OFFSET = 2 * 1024 * 1024; // 前2MB作为flag标志位，之后100MB作为数据存储
constexpr int64_t SYNC_FLAG_BIT_NUM = 10;  // cce 算子在用
constexpr int64_t MEM_DMA_UNIT_INT_NUM = 4;
constexpr int64_t EVENT_ID_MASK = 0xFFFFFFFF;
constexpr int64_t PING_PONG_SIZE = 2;
constexpr int64_t UB_SINGLE_DMA_SIZE_MAX = 190 * 1024;
constexpr int64_t SMALL_DATA_SIZE = 1 * 1024 * 1024;
constexpr int64_t UB_SINGLE_PING_PONG_ADD_SIZE_MAX = UB_SINGLE_DMA_SIZE_MAX / 2;
constexpr int UB_ALIGN_SIZE = 32;

// 2step算法中，2个aiv真正用作数据预处理
constexpr int64_t PRE_CORE_REAL_NUM = 2;

constexpr int64_t AIV_PER_AICORE = 2;

constexpr int DFX_COUNT = 50;

constexpr int64_t HALF_NUM = 2;

constexpr int64_t THREE_NUM = 3;

constexpr int64_t FOUR_NUM = 4;

constexpr int64_t VADD_MAX_REPEAT = 255;
constexpr int64_t VADD_UNIT_BYTE = 256;

// vadd单位粒度是256B，vadd最大repeat次数为255，两个相乘的结果
constexpr int64_t MAX_VADD_SIZE = VADD_MAX_REPEAT * VADD_UNIT_BYTE;
constexpr int64_t BLOCK_UNIT_BYTE = 32;
constexpr int64_t VADD_UNIT_TO_BLOCK_UNIT_RATIO = VADD_UNIT_BYTE / BLOCK_UNIT_BYTE;    // 8

constexpr bool ATOMIC_ENABLE = false;

enum Op : int {
    COPYONLY = -1,
    ADD = 0,
    MUL = 1,
    MAX = 2,
    MIN = 3
};

struct ExtraFlag {
    static constexpr uint32_t RDMA = 1;
    static constexpr uint32_t TOPO_910B2C = 1 << 1;
    static constexpr uint32_t TOPO_910_93 = 1 << 2;
    static constexpr uint32_t DETERMINISTIC = 1 << 3;
    static constexpr uint32_t QUANT_FP16 = 1 << 4;
    static constexpr uint32_t QUANT_FP32 = 1 << 5;
    static constexpr uint32_t TOPO_910A5 = 1 << 6;
    static constexpr uint32_t QUANT_DELAY = 1 << 7;
    static constexpr uint32_t QUANT_CURRENT = 1 << 8;
    static constexpr uint32_t TOPO_PCIE = 1 << 9;
};

struct CommArgs {
    int rank = 0;           // attr rank_id, global rank
    int localRank = -1;
    int rankSize = 0; // global rank size
    int localRankSize = -1;  // 此参数是指fullmesh互联的卡数
    uint32_t extraFlag = 0; // 32 bit map，具体每一位的含义就在此文件正上方
    GM_ADDR peerMems[LCAL_MAX_RANK_SIZE] = {}; // 传入初始化获得的buff，所有allreduce都是同一个参数
    /**
     * @param sendCountMatrix 大小是rankSize*rankSize的一维数组
     * eg: sendCountMatrix[1] 的数值，对应二维数组的[0][1]，表示 卡0 要给 卡1 发送的数据个数
     */
    int64_t sendCountMatrix[LCAL_MAX_RANK_SIZE * LCAL_MAX_RANK_SIZE] = {}; // for all2allv
    int64_t dfx[DFX_COUNT] = {};
};
}
#endif // LCCL_COMM_ARGS_H
