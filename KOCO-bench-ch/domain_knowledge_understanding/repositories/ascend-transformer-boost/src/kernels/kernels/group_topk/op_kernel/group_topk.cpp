/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <type_traits>
#include "kernel_operator.h"
#include "kernels/group_topk/tiling/tiling_data.h"
#include "kernels/utils/kernel/kernel_utils.h"

using namespace AscendC;

namespace {

constexpr uint32_t BUFFER_NUM = 3;
constexpr uint32_t B16_SIZE = 2;
constexpr uint32_t B32_SIZE = 4;
constexpr uint32_t B64_SIZE = 8;
constexpr uint32_t BIT64_SIZE = 64;
constexpr uint32_t SORT_REPEAT_COUNT = 32; // Sort/Sort32接口要求32个元素对齐
constexpr uint32_t SORT_RES_FP32_NUM = 2;  // Sort/Sort32接口每个元素的结果占用2个int32_t长度
constexpr uint32_t DATABLOCK_SIZE = 32;
constexpr uint32_t BLOCK_PER_REPEAT = 8;
constexpr uint32_t REPEAT_SIZE = DATABLOCK_SIZE * BLOCK_PER_REPEAT;
constexpr uint32_t B16_PER_BLOCK = DATABLOCK_SIZE / B16_SIZE;
constexpr uint32_t B16_PER_REPEAT = B16_PER_BLOCK * BLOCK_PER_REPEAT;
constexpr uint32_t B32_PER_BLOCK = DATABLOCK_SIZE / B32_SIZE;
constexpr uint32_t B32_PER_REPEAT = B32_PER_BLOCK * BLOCK_PER_REPEAT;
constexpr uint32_t HALF_MIN = 0xfbff;
constexpr uint64_t MASK[16] = {0x0, 0xfe,   0xfc,   0xf8,   0xf0,   0xe0,   0xc0,   0x80,
                               0x0, 0xfe00, 0xfc00, 0xf800, 0xf000, 0xe000, 0xc000, 0x8000};
constexpr uint32_t MASK_COEFFICIENT = 2;
constexpr uint64_t FP32_MASK = 0x5555555555555555;
constexpr uint64_t REPEAT_MASK = 0xffffffffffffffff;
constexpr uint32_t MAX_EXPERT = 1024;

constexpr uint32_t BUFFER_INDEX_0 = 0;
constexpr uint32_t BUFFER_INDEX_1 = 1;
constexpr uint32_t BUFFER_INDEX_2 = 2;
constexpr uint32_t UNLOOP_1 = 1;
constexpr uint32_t UNLOOP_2 = 2;
constexpr uint32_t UNLOOP_3 = 3;
constexpr uint32_t UNLOOP_4 = 4;
constexpr uint32_t UNLOOP_5 = 5;
constexpr uint32_t UNLOOP_6 = 6;
constexpr uint32_t UNLOOP_7 = 7;

constexpr uint32_t LARGE_GROUPNUM_THRESHOLD = 256;
constexpr uint32_t MAX_GROUP_PADDED = 512;
constexpr uint32_t MIN_GROUP_PADDED = 32;
constexpr uint32_t MAX_GROUP_BLOCKS = 64;
constexpr uint32_t MAX_SORT_GROUP = 128;
constexpr uint32_t BUF_SIZE = 180 * 1024;
constexpr uint32_t MAX_TOKEN_SIZE = MAX_EXPERT * SORT_REPEAT_COUNT * B16_SIZE;                            // 65536
constexpr uint32_t TOKEN_SIZE = MAX_GROUP_PADDED * DATABLOCK_SIZE;                                        // 16384
constexpr uint32_t INDEX_ADDR = TOKEN_SIZE;                                                               // 16384
constexpr uint32_t SORTED_ADDR0 = INDEX_ADDR + MAX_GROUP_PADDED * B32_SIZE;                               // 18432
constexpr uint32_t SORTED_ADDR1 = SORTED_ADDR0 + MAX_GROUP_PADDED * B64_SIZE;                             // 22528
constexpr uint32_t SORTED_ADDR2 = SORTED_ADDR1 + MAX_GROUP_PADDED * B64_SIZE;                             // 26624
constexpr uint32_t GROUP_SELECT_ADDR0 = SORTED_ADDR2 + MAX_GROUP_PADDED * B64_SIZE;                       // 30720
constexpr uint32_t GROUP_SELECT_ADDR1 = GROUP_SELECT_ADDR0 + MAX_GROUP_PADDED * B16_SIZE;                 // 31744
constexpr uint32_t GROUP_SELECT_ADDR2 = GROUP_SELECT_ADDR1 + MAX_GROUP_PADDED * B16_SIZE;                 // 32768
constexpr uint32_t TOKEN_ADDR0 = GROUP_SELECT_ADDR2 + MAX_GROUP_PADDED * B16_SIZE;                        // 33792
constexpr uint32_t TOKEN_ADDR1 = TOKEN_ADDR0 + TOKEN_SIZE;                                                // 50176
constexpr uint32_t TOKEN_ADDR2 = TOKEN_ADDR1 + TOKEN_SIZE;                                                // 66560
constexpr uint32_t BLOCK_SELECT_ADDR = TOKEN_ADDR0 + MAX_TOKEN_SIZE;                                      // 99328
constexpr uint32_t BRCB_SELECT_ADDR = BLOCK_SELECT_ADDR + MIN_GROUP_PADDED * MAX_GROUP_BLOCKS * B16_SIZE; // 103424
constexpr uint32_t COPY_SELECT_ADDR = BRCB_SELECT_ADDR + MAX_EXPERT * B16_PER_BLOCK * B16_SIZE;           // 136192
constexpr uint32_t CACL_ADDR = BLOCK_SELECT_ADDR;                                                         // 99328
constexpr uint32_t GROUP_SCORE_ADDR = CACL_ADDR + TOKEN_SIZE / B16_SIZE * B32_SIZE;                       // 132096
constexpr uint32_t SORTED_TOKEN_ADDR = GROUP_SCORE_ADDR + MAX_GROUP_PADDED * B32_SIZE * 2;                // 136192
constexpr uint32_t TEMP_ADDR = GROUP_SCORE_ADDR + MAX_SORT_GROUP * REPEAT_SIZE;                           // 164864

enum class GroupMultiFlag {
    UNDEFINED,
    SUM_MULTI_MAX,
};
} // namespace

template <typename T, GroupMultiFlag GROUP_FLAG = GroupMultiFlag::UNDEFINED> class GroupTopk {
public:
    static constexpr bool IS_RISE_PRECISION = !(std::is_same_v<T, half> && GROUP_FLAG == GroupMultiFlag::UNDEFINED);
    static constexpr T T_MIN = std::is_same_v<T, half> ? -65504 : -3.38953e+38;
    static constexpr uint32_t CACLTYPE_PER_BLOCK = IS_RISE_PRECISION ? B32_PER_BLOCK : B16_PER_BLOCK;
    static constexpr uint32_t CACLTYPE_PER_REPEAT = IS_RISE_PRECISION ? B32_PER_REPEAT : B16_PER_REPEAT;
    using CalcType = std::conditional_t<IS_RISE_PRECISION, float, half>; // Type used to compute scores for each group
    using MoveType = T;                                                  // Type used to copy in and copy out
    using SelectType = half;                                             // Type used to select zero or original value

    __aicore__ explicit GroupTopk(GM_ADDR topKInput, GM_ADDR idxArr, AscendC::TPipe &pipe,
                                  const AsdOps::GroupTopkTilingData &tilingData)
        : blockIdx(AscendC::GetBlockIdx()), groupNum(tilingData.groupNum),
          groupPadded(RoundUp(groupNum, SORT_REPEAT_COUNT)), k(tilingData.k), kInner(tilingData.kInner),
          expertNumPerGroup(tilingData.expertNumPerGroup), expertNumPerGroupPadded(tilingData.expertNumPerGroupPadded),
          expertNumPerToken(tilingData.expertNumPerToken),
          tokenNum(tilingData.tokenNumPerCore + static_cast<uint32_t>(tilingData.tailTokenNum > blockIdx)),
          tokenOffSet(tilingData.tokenNumPerCore * blockIdx +
                      (tilingData.tailTokenNum > blockIdx ? blockIdx : tilingData.tailTokenNum)),
          expertNumPerTokenPadded(expertNumPerGroupPadded * RoundUp(groupNum, BLOCK_PER_REPEAT)),
          blocksPerGroup(expertNumPerGroupPadded / B16_PER_BLOCK),
          calcBlocksPerGroup(blocksPerGroup * (1 + IS_RISE_PRECISION)),
          tokenComputeCnt(groupNum / LARGE_GROUPNUM_THRESHOLD),
          iterTokenOffset(LARGE_GROUPNUM_THRESHOLD * expertNumPerGroupPadded),
          iterSelectOffset(iterTokenOffset / B16_PER_BLOCK),
          tokenTailComputeGroupNum(groupNum % LARGE_GROUPNUM_THRESHOLD),
          needLeftPadding(kInner > 1 && expertNumPerGroup % SORT_REPEAT_COUNT <= B16_PER_BLOCK &&
                          expertNumPerGroup % SORT_REPEAT_COUNT != 0),
          leftPadding(B16_PER_BLOCK * static_cast<uint8_t>(needLeftPadding)),
          rightPadding((B16_PER_BLOCK - expertNumPerGroup % B16_PER_BLOCK) * (expertNumPerGroup % B16_PER_BLOCK != 0)),
          copyParams(static_cast<uint16_t>(groupNum), expertNumPerGroup * sizeof(MoveType), 0, 0, 0),
          padParams(true, leftPadding, rightPadding, T_MIN)
    {
        topKInputGm.SetGlobalBuffer((__gm__ MoveType *)topKInput + tokenOffSet * expertNumPerToken,
                                    expertNumPerToken * tokenNum);
        idxArrGm.SetGlobalBuffer((__gm__ uint32_t *)idxArr, MAX_EXPERT);
        pipe.InitBuffer(buf, BUF_SIZE);
        zeroTensor = buf.Get<SelectType>(expertNumPerTokenPadded);
        groupScoreTensor = buf.GetWithOffset<CalcType>(groupPadded, GROUP_SCORE_ADDR);
        indexTensor = buf.GetWithOffset<uint32_t>(groupPadded, INDEX_ADDR);
        AscendC::Duplicate<SelectType>(zeroTensor, 0, expertNumPerTokenPadded);
        AscendC::Duplicate<CalcType>(groupScoreTensor, T_MIN, groupPadded);
        AscendC::DataCopy<uint32_t>(indexTensor, idxArrGm, groupPadded);
        sortedTensor[BUFFER_INDEX_0] =
            buf.GetWithOffset<CalcType>(groupPadded * B64_SIZE / sizeof(CalcType), SORTED_ADDR0);
        sortedTensor[BUFFER_INDEX_1] =
            buf.GetWithOffset<CalcType>(groupPadded * B64_SIZE / sizeof(CalcType), SORTED_ADDR1);
        sortedTensor[BUFFER_INDEX_2] =
            buf.GetWithOffset<CalcType>(groupPadded * B64_SIZE / sizeof(CalcType), SORTED_ADDR2);
        groupSelectTensor[BUFFER_INDEX_0] = buf.GetWithOffset<uint16_t>(groupPadded, GROUP_SELECT_ADDR0);
        groupSelectTensor[BUFFER_INDEX_1] = buf.GetWithOffset<uint16_t>(groupPadded, GROUP_SELECT_ADDR1);
        groupSelectTensor[BUFFER_INDEX_2] = buf.GetWithOffset<uint16_t>(groupPadded, GROUP_SELECT_ADDR2);
        blockSelectTensor = buf.GetWithOffset<uint16_t>(blocksPerGroup * groupPadded, BLOCK_SELECT_ADDR);
        brcbGroupSelectTensor = buf.GetWithOffset<uint16_t>(groupPadded * B16_PER_BLOCK, BRCB_SELECT_ADDR);
        CopyGroupSelectTensor = buf.GetWithOffset<uint16_t>(
            groupPadded * B16_PER_BLOCK * CeilDiv(expertNumPerGroupPadded, B16_PER_REPEAT), COPY_SELECT_ADDR);
        tempTensor = buf.GetWithOffset<CalcType>(TOKEN_SIZE / B32_SIZE, TEMP_ADDR);
        tokenTensor[BUFFER_INDEX_0] = buf.GetWithOffset<SelectType>(expertNumPerTokenPadded, TOKEN_ADDR0);
        tokenTensor[BUFFER_INDEX_1] = buf.GetWithOffset<SelectType>(expertNumPerTokenPadded, TOKEN_ADDR1);
        tokenTensor[BUFFER_INDEX_2] = buf.GetWithOffset<SelectType>(expertNumPerTokenPadded, TOKEN_ADDR2);
        sortedTokenTensor = buf.GetWithOffset<CalcType>(MAX_SORT_GROUP * REPEAT_SIZE / B32_SIZE, SORTED_TOKEN_ADDR);
        calcTensor = buf.GetWithOffset<CalcType>(expertNumPerTokenPadded, CACL_ADDR);
        moveTensor[BUFFER_INDEX_0] = buf.GetWithOffset<MoveType>(expertNumPerTokenPadded, TOKEN_ADDR0);
        moveTensor[BUFFER_INDEX_1] = buf.GetWithOffset<MoveType>(expertNumPerTokenPadded, TOKEN_ADDR1);
        moveTensor[BUFFER_INDEX_2] = buf.GetWithOffset<MoveType>(expertNumPerTokenPadded, TOKEN_ADDR2);
    }

    __aicore__ inline void Process()
    {
        if (groupNum <= LARGE_GROUPNUM_THRESHOLD || GROUP_FLAG == GroupMultiFlag::UNDEFINED) {
            ProcessSmallGroupNum();
        } else {
            ProcessLargeGroupNum();
        }
    }

    __aicore__ inline void ProcessSmallGroupNum()
    {
        event_t event0 = EVENT_ID0;
        event_t event1 = EVENT_ID1;
        event_t event2 = EVENT_ID2;
        event_t eventTemp = EVENT_ID2;
        AscendC::PipeBarrier<PIPE_V>();
        CopyIn(0, event0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event0);
        Topk(event0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event2);
        for (uint32_t i = 0; i < tokenNum - 1; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event1);
            CopyIn(i + 1, event1);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event1);
            SetZeroGroup(event0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event0);
            CopyOut(i, event0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event1);
            Topk(event1);
            eventTemp = event0;
            event0 = event1;
            event1 = event2;
            event2 = eventTemp;
        }
        SetZeroGroup(event0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event0);
        CopyOut(tokenNum - 1, event0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
    }

    __aicore__ inline void ProcessLargeGroupNum()
    {
        // Non-typical case: GroupNum > 256 and kInner >= 2
        const event_t event = EVENT_ID0;
        for (uint32_t i = 0; i < tokenNum; ++i) {
            CopyIn(i, event);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            Topk(event);
            SetZeroGroup(event);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
            CopyOut(i, event);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
    }

    __aicore__ inline void CopyIn(const uint32_t tokenIndex, const event_t event)
    {
        AscendC::DataCopyPad<MoveType>(moveTensor[event], topKInputGm[tokenIndex * expertNumPerToken], copyParams,
                                       padParams);
    }

    __aicore__ inline void ComputeGroupScoreUndefined(const event_t event)
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            AscendC::Cast<CalcType, MoveType>(calcTensor, moveTensor[event], AscendC::RoundMode::CAST_NONE,
                                              expertNumPerTokenPadded);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            calcTensor = moveTensor[event];
        }
        if (expertNumPerGroup <= CACLTYPE_PER_BLOCK) {
            AscendC::BlockReduceMax<CalcType>(groupScoreTensor, calcTensor, CeilDiv(groupNum, BLOCK_PER_REPEAT),
                                              CACLTYPE_PER_REPEAT, 1, 1 + IS_RISE_PRECISION,
                                              BLOCK_PER_REPEAT * (1 + IS_RISE_PRECISION));
            if (groupNum % BLOCK_PER_REPEAT != 0) {
                AscendC::PipeBarrier<PIPE_V>();
                uint64_t mask[2] = {MASK[groupNum % CACLTYPE_PER_BLOCK], 0};
                AscendC::Duplicate<CalcType>(groupScoreTensor[groupNum / CACLTYPE_PER_BLOCK * CACLTYPE_PER_BLOCK],
                                             T_MIN, mask, 1, 1, 1);
            }
        } else if (expertNumPerGroup <= CACLTYPE_PER_REPEAT) {
            AscendC::WholeReduceMax<CalcType>(groupScoreTensor, calcTensor, expertNumPerGroup, groupNum, 1, 1,
                                              calcBlocksPerGroup, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        } else {
            const uint32_t repeats = calcBlocksPerGroup / BLOCK_PER_REPEAT;
            const uint32_t tailBlocks = calcBlocksPerGroup % BLOCK_PER_REPEAT;
            AscendC::DataCopy(tempTensor, calcTensor,
                              {static_cast<uint16_t>(groupNum), BLOCK_PER_REPEAT,
                               static_cast<uint16_t>(calcBlocksPerGroup - BLOCK_PER_REPEAT), 0});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BinaryRepeatParams repeatParams(1, 1, 1, BLOCK_PER_REPEAT,
                                                     static_cast<uint16_t>(calcBlocksPerGroup), BLOCK_PER_REPEAT);
            for (uint32_t i = 1; i < repeats; i++) {
                AscendC::Max<CalcType>(tempTensor, calcTensor[i * CACLTYPE_PER_REPEAT], tempTensor, CACLTYPE_PER_REPEAT,
                                       groupNum, repeatParams);
                AscendC::PipeBarrier<PIPE_V>();
            }
            if (tailBlocks > 0) {
                AscendC::Max<CalcType>(tempTensor, calcTensor[repeats * CACLTYPE_PER_REPEAT], tempTensor,
                                       tailBlocks * CACLTYPE_PER_BLOCK, groupNum, repeatParams);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::WholeReduceMax<CalcType>(groupScoreTensor, tempTensor, CACLTYPE_PER_REPEAT, groupNum, 1, 1,
                                              BLOCK_PER_REPEAT, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        }
    }

    __aicore__ inline void ComputeGroupScoreSum(const event_t event, const uint32_t calcGroupNum,
                                                const uint32_t groupNumOffset = 0)
    {
        AscendC::Cast<CalcType, MoveType>(calcTensor, moveTensor[event][groupNumOffset * iterTokenOffset],
                                          AscendC::RoundMode::CAST_NONE, calcGroupNum * expertNumPerGroupPadded);
        AscendC::PipeBarrier<PIPE_V>();
        const uint64_t mask[2] = {(REPEAT_MASK >> (BIT64_SIZE - kInner * MASK_COEFFICIENT)) & FP32_MASK, 0};
        if (expertNumPerGroupPadded == SORT_REPEAT_COUNT) {
            const uint32_t sortCnt = calcGroupNum / MAX_SORT_GROUP;
            const uint32_t tailGroupNum = calcGroupNum % MAX_SORT_GROUP;
            for (uint32_t i = 0; i < sortCnt; i++) {
                AscendC::Sort32<CalcType>(sortedTokenTensor, calcTensor[SORT_REPEAT_COUNT * MAX_SORT_GROUP * i],
                                          indexTensor, MAX_SORT_GROUP);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum<CalcType>(
                    groupScoreTensor[groupNumOffset * LARGE_GROUPNUM_THRESHOLD][MAX_SORT_GROUP * i], sortedTokenTensor,
                    mask, MAX_SORT_GROUP, 1, 1, BLOCK_PER_REPEAT);
                AscendC::PipeBarrier<PIPE_V>();
            }
            if (tailGroupNum > 0) {
                AscendC::Sort32<CalcType>(sortedTokenTensor, calcTensor[SORT_REPEAT_COUNT * MAX_SORT_GROUP * sortCnt],
                                          indexTensor, tailGroupNum);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum<CalcType>(
                    groupScoreTensor[groupNumOffset * LARGE_GROUPNUM_THRESHOLD][MAX_SORT_GROUP * sortCnt],
                    sortedTokenTensor, mask, tailGroupNum, 1, 1, BLOCK_PER_REPEAT);
            }
        } else {
            const uint32_t sortRepeats = expertNumPerGroupPadded / SORT_REPEAT_COUNT;
            for (uint32_t i = 0; i < groupNum; i++) {
                AscendC::Sort<CalcType, true>(sortedTokenTensor[expertNumPerGroupPadded * i * MASK_COEFFICIENT],
                                              calcTensor[expertNumPerGroupPadded * i], indexTensor, tempTensor,
                                              sortRepeats);
                AscendC::PipeBarrier<PIPE_V>();
            }
            if (kInner <= SORT_REPEAT_COUNT) {
                AscendC::WholeReduceSum<CalcType>(groupScoreTensor, sortedTokenTensor, mask, groupNum, 1, 1,
                                                  sortRepeats * BLOCK_PER_REPEAT);
            } else {
                AscendC::GatherMask<CalcType>(sortedTokenTensor, sortedTokenTensor, 1, false, 0,
                                              {1, static_cast<uint16_t>(sortRepeats * groupNum), BLOCK_PER_REPEAT, 0},
                                              rsvdCnt);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t i = 0; i < groupNum; i++) {
                    AscendC::ReduceSum<CalcType>(groupScoreTensor[i], sortedTokenTensor[expertNumPerGroupPadded * i],
                                                 tempTensor, kInner);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
        }
    }

    __aicore__ inline void Topk(const event_t event)
    {
        AscendC::Duplicate<uint16_t>(groupSelectTensor[event], 0, groupPadded);
        if constexpr (GROUP_FLAG == GroupMultiFlag::UNDEFINED) {
            ComputeGroupScoreUndefined(event);
        } else {
            uint32_t computeCnt = 0;
            for (; computeCnt < tokenComputeCnt; ++computeCnt) {
                ComputeGroupScoreSum(event, LARGE_GROUPNUM_THRESHOLD, computeCnt);
                AscendC::PipeBarrier<PIPE_V>();
            }
            if (tokenTailComputeGroupNum > 0) {
                ComputeGroupScoreSum(event, tokenTailComputeGroupNum, computeCnt);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (groupNum > SORT_REPEAT_COUNT) {
            AscendC::Sort<CalcType, true>(sortedTensor[event], groupScoreTensor, indexTensor, tempTensor,
                                          groupPadded / SORT_REPEAT_COUNT);
        } else {
            AscendC::Sort32<CalcType>(sortedTensor[event], groupScoreTensor, indexTensor, 1);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(event);
    }

    __aicore__ inline void SetZeroGroup(event_t event)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(event);
        Scatter(event);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
        const uint32_t blocksAlignBlockNum = CeilDiv(blocksPerGroup, B16_PER_BLOCK);
        const uint32_t blocksAlign = blocksAlignBlockNum * B16_PER_BLOCK;
        AscendC::Brcb<uint16_t>(brcbGroupSelectTensor, groupSelectTensor[event],
                                static_cast<uint8_t>(groupPadded / BLOCK_PER_REPEAT), {1, BLOCK_PER_REPEAT});
        AscendC::PipeBarrier<PIPE_V>();
        if (blocksPerGroup <= B16_PER_BLOCK) {
            // expertNumPerTokenPadded <= 256
            AscendC::GatherMask<uint16_t>(
                blockSelectTensor, brcbGroupSelectTensor, 7, true, blocksPerGroup, // src1Pattern: 7, 取repeat全部元素
                {1, static_cast<uint16_t>(groupPadded), static_cast<uint16_t>(blocksAlignBlockNum), 0}, rsvdCnt);
        } else {
            AscendC::Copy<uint16_t>(CopyGroupSelectTensor, brcbGroupSelectTensor, blocksAlign, groupPadded,
                                    {1, 0, static_cast<uint16_t>(blocksAlignBlockNum), 1});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::GatherMask<uint16_t>(
                blockSelectTensor, CopyGroupSelectTensor, 7, true, blocksPerGroup, // src1Pattern: 7, 取repeat全部元素
                {1, static_cast<uint16_t>(groupPadded), static_cast<uint16_t>(blocksAlignBlockNum), 0}, rsvdCnt);
        }
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t selectCnt = 0;
        for (; selectCnt < tokenComputeCnt; ++selectCnt) {
            AscendC::Select<SelectType, uint16_t>(tokenTensor[event][selectCnt * iterTokenOffset],
                                                  blockSelectTensor[selectCnt * iterSelectOffset],
                                                  tokenTensor[event][selectCnt * iterTokenOffset], zeroTensor,
                                                  AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, iterTokenOffset);
        }
        if (tokenTailComputeGroupNum > 0) {
            AscendC::Select<SelectType, uint16_t>(
                tokenTensor[event][selectCnt * iterTokenOffset], blockSelectTensor[selectCnt * iterSelectOffset],
                tokenTensor[event][selectCnt * iterTokenOffset], zeroTensor, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                expertNumPerGroupPadded * RoundUp(tokenTailComputeGroupNum, BLOCK_PER_REPEAT));
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Scatter(event_t event)
    {
        const uint32_t loopUnfoldNumber = 8;
        uint32_t i = 0;
        uint32_t loopNum = k - (k % loopUnfoldNumber);
        AscendC::LocalTensor<uint32_t> sortTensor = sortedTensor[event].template ReinterpretCast<uint32_t>();
        // loop unfold
        for (; i < loopNum; i += loopUnfoldNumber) {
            groupSelectTensor[event].SetValue(sortTensor.GetValue(SORT_RES_FP32_NUM * i + 1), UINT16_MAX);
            groupSelectTensor[event].SetValue(sortTensor.GetValue(SORT_RES_FP32_NUM * (i + UNLOOP_1) + 1), UINT16_MAX);
            groupSelectTensor[event].SetValue(sortTensor.GetValue(SORT_RES_FP32_NUM * (i + UNLOOP_2) + 1), UINT16_MAX);
            groupSelectTensor[event].SetValue(sortTensor.GetValue(SORT_RES_FP32_NUM * (i + UNLOOP_3) + 1), UINT16_MAX);
            groupSelectTensor[event].SetValue(sortTensor.GetValue(SORT_RES_FP32_NUM * (i + UNLOOP_4) + 1), UINT16_MAX);
            groupSelectTensor[event].SetValue(sortTensor.GetValue(SORT_RES_FP32_NUM * (i + UNLOOP_5) + 1), UINT16_MAX);
            groupSelectTensor[event].SetValue(sortTensor.GetValue(SORT_RES_FP32_NUM * (i + UNLOOP_6) + 1), UINT16_MAX);
            groupSelectTensor[event].SetValue(sortTensor.GetValue(SORT_RES_FP32_NUM * (i + UNLOOP_7) + 1), UINT16_MAX);
        }
        // tail loop
        for (; i < k; i++) {
            groupSelectTensor[event].SetValue(sortTensor.GetValue(SORT_RES_FP32_NUM * i + 1), UINT16_MAX);
        }
    }

    __aicore__ inline void CopyOut(uint32_t tokenIndex, event_t event)
    {
        AscendC::DataCopyPad<MoveType>(topKInputGm[tokenIndex * expertNumPerToken], moveTensor[event][leftPadding],
                                       {static_cast<uint16_t>(groupNum),
                                        static_cast<uint32_t>(expertNumPerGroup * sizeof(MoveType)),
                                        static_cast<uint16_t>(needLeftPadding), 0, 0});
    }

private:
    const uint32_t blockIdx{0};
    const uint32_t groupNum{0};                 // 分组数量
    const uint32_t groupPadded{0};              // 分组数量填充对齐后的数量
    const uint32_t k{0};                        // 选k个组
    const uint32_t kInner{0};                   // 组内选取kInner个最大值求和
    const uint32_t expertNumPerGroup{0};        // 每个组的专家数量
    const uint32_t expertNumPerGroupPadded{0};  // 单个group填充至对齐后的专家数量
    const uint32_t expertNumPerToken{0};        // 每个token的专家总数量
    const uint32_t tokenNum{0};                 // 当前核需要处理的token数量
    const uint32_t tokenOffSet{0};              // 当前核处理的token偏移量
    const uint32_t expertNumPerTokenPadded{0};  // group填充至对齐后的整个token的专家数量
    const uint32_t blocksPerGroup{0};           // group填充至对齐后在moveTensor占用的block数
    const uint32_t calcBlocksPerGroup{0};       // group填充至对齐后在calcTensor占用的block数
    const uint32_t tokenComputeCnt{0};          // token计算次数
    const uint32_t tokenTailComputeGroupNum{0}; // token尾块分组数量
    const uint32_t iterTokenOffset{0};          // 主块内每轮计算在tokenTensor的偏移量
    const uint32_t iterSelectOffset{0};         // 主块内每轮计算在blockSelectTensor的偏移量
    const bool needLeftPadding{false};          // CopyIn时是否需要在每组的左侧填充
    const uint8_t leftPadding{0};               // CopyIn时在每组的左侧填充的元素个数
    const uint8_t rightPadding{0};              // CopyIn时在每组的右侧填充的元素个数
    uint64_t rsvdCnt{0};

    const AscendC::DataCopyExtParams copyParams;
    const AscendC::DataCopyPadExtParams<MoveType> padParams;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> buf;
    AscendC::GlobalTensor<MoveType> topKInputGm;
    AscendC::GlobalTensor<uint32_t> idxArrGm;
    AscendC::LocalTensor<SelectType> tokenTensor[BUFFER_NUM];
    AscendC::LocalTensor<MoveType> moveTensor[BUFFER_NUM];
    AscendC::LocalTensor<SelectType> zeroTensor;
    AscendC::LocalTensor<CalcType> groupScoreTensor;
    AscendC::LocalTensor<CalcType> sortedTokenTensor;
    AscendC::LocalTensor<CalcType> sortedTensor[BUFFER_NUM];
    AscendC::LocalTensor<uint32_t> indexTensor;
    AscendC::LocalTensor<uint16_t> groupSelectTensor[BUFFER_NUM];
    AscendC::LocalTensor<uint16_t> blockSelectTensor;
    AscendC::LocalTensor<uint16_t> brcbGroupSelectTensor;
    AscendC::LocalTensor<uint16_t> CopyGroupSelectTensor;
    AscendC::LocalTensor<CalcType> tempTensor;
    AscendC::LocalTensor<CalcType> calcTensor;
};

/*
 * When each group contains only one element, Topk is recommended because the GroupTopk operator zeros non-topk
 * elements slowly.
 */
template <typename T> class SingleValueGroupTopk {
public:
    static constexpr T T_MIN = std::is_same_v<T, half> ? -65504 : -3.38953e+38;
    using CalcType = std::conditional_t<std::is_same_v<T, bfloat16_t>, float, half>;

    __aicore__ explicit SingleValueGroupTopk(GM_ADDR topKInput, GM_ADDR idxArr, AscendC::TPipe &pipe,
                                             const AsdOps::GroupTopkTilingData &tilingData)
        : blockIdx(AscendC::GetBlockIdx()), k(tilingData.k), expertNumPerToken(tilingData.expertNumPerToken),
          tokenNum(tilingData.tokenNumPerCore + static_cast<uint32_t>(tilingData.tailTokenNum > blockIdx)),
          tokenOffSet(tilingData.tokenNumPerCore * blockIdx +
                      (tilingData.tailTokenNum > blockIdx ? blockIdx : tilingData.tailTokenNum)),
          expertNumPerTokenPadded(RoundUp(expertNumPerToken, SORT_REPEAT_COUNT))
    {
        topKInputGm.SetGlobalBuffer((__gm__ T *)topKInput + tokenOffSet * expertNumPerToken,
                                    expertNumPerToken * tokenNum);
        idxArrGm.SetGlobalBuffer((__gm__ uint32_t *)idxArr, MAX_EXPERT);
        pipe.InitBuffer(buf, BUF_SIZE);
        indexTensor = buf.Get<uint32_t>(expertNumPerTokenPadded);
        sortedTensor = buf.GetWithOffset<CalcType>(expertNumPerTokenPadded * B64_SIZE / sizeof(CalcType), SORTED_ADDR0);
        tokenTensor = buf.GetWithOffset<half>(expertNumPerTokenPadded, TOKEN_ADDR0);
        moveTensor = buf.GetWithOffset<T>(expertNumPerTokenPadded, TOKEN_ADDR0);
        calcTensor = buf.GetWithOffset<CalcType>(expertNumPerTokenPadded, CACL_ADDR);
        tempTensor = buf.GetWithOffset<CalcType>(expertNumPerTokenPadded * B64_SIZE / sizeof(CalcType), TEMP_ADDR);
    }

    __aicore__ inline void Process()
    {
        AscendC::Duplicate<T>(moveTensor, T_MIN, expertNumPerTokenPadded);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::DataCopy<uint32_t>(indexTensor, idxArrGm, expertNumPerTokenPadded);
        AscendC::LocalTensor<uint32_t> sortTensor = sortedTensor.template ReinterpretCast<uint32_t>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(expertNumPerToken * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{
            true, 0, static_cast<uint8_t>(RoundUp(expertNumPerToken, B16_PER_BLOCK) - expertNumPerToken), T_MIN};
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        for (uint32_t tokenIndex = 0; tokenIndex < tokenNum; tokenIndex++) {
            AscendC::DataCopyPad(moveTensor, topKInputGm[tokenIndex * expertNumPerToken], copyParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                AscendC::Cast<CalcType, T>(calcTensor, moveTensor, AscendC::RoundMode::CAST_NONE,
                                           expertNumPerTokenPadded);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                calcTensor = moveTensor;
            }
            AscendC::Sort<CalcType, true>(sortedTensor, calcTensor, indexTensor, tempTensor,
                                          expertNumPerTokenPadded / SORT_REPEAT_COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            for (uint32_t i = k; i < expertNumPerToken; i++) {
                tokenTensor.SetValue(sortTensor.GetValue(SORT_RES_FP32_NUM * i + 1), 0);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(topKInputGm[tokenIndex * expertNumPerToken], moveTensor,
                                 {1, static_cast<uint32_t>(expertNumPerToken * sizeof(T)), 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

private:
    const uint32_t blockIdx{0};
    const uint32_t k{0};                 // 选k个组
    const uint32_t expertNumPerToken{0}; // 每个token的专家总数量
    const uint32_t tokenNum{0};          // 当前核需要处理的token数量
    const uint32_t tokenOffSet{0};       // 当前核处理的token偏移量
    const uint32_t expertNumPerTokenPadded{0};

    AscendC::TBuf<AscendC::QuePosition::VECCALC> buf;
    AscendC::GlobalTensor<T> topKInputGm;
    AscendC::GlobalTensor<uint32_t> idxArrGm;
    AscendC::LocalTensor<uint32_t> indexTensor;
    AscendC::LocalTensor<CalcType> sortedTensor;
    AscendC::LocalTensor<half> tokenTensor;
    AscendC::LocalTensor<T> moveTensor;
    AscendC::LocalTensor<CalcType> calcTensor;
    AscendC::LocalTensor<CalcType> tempTensor;
};

__aicore__ inline void InitTilingData(const __gm__ uint8_t *tiling, AsdOps::GroupTopkTilingData *tilingData)
{
    tilingData->groupNum = (*(const __gm__ uint32_t *)(tiling + 0));
    tilingData->k = (*(const __gm__ uint32_t *)(tiling + 4));
    tilingData->kInner = (*(const __gm__ uint32_t *)(tiling + 8));
    tilingData->expertNumPerGroup = (*(const __gm__ uint32_t *)(tiling + 12));
    tilingData->expertNumPerGroupPadded = (*(const __gm__ uint32_t *)(tiling + 16));
    tilingData->expertNumPerToken = (*(const __gm__ uint32_t *)(tiling + 20));
    tilingData->tokenNumPerCore = (*(const __gm__ uint32_t *)(tiling + 24));
    tilingData->tailTokenNum = (*(const __gm__ uint32_t *)(tiling + 28));
}

#define GET_TILING_DATA(tiling, tilingData)                                                                            \
    AsdOps::GroupTopkTilingData tilingData;                                                                            \
    InitTilingData(tiling, &(tilingData))

template <typename T, GroupMultiFlag GROUP_FLAG>
__aicore__ inline void group_topk_impl(GM_ADDR topKInput, GM_ADDR idxArr, AsdOps::GroupTopkTilingData &tilingData)
{
    AscendC::TPipe pipe;
    GroupTopk<T, GROUP_FLAG> op(topKInput, idxArr, pipe, tilingData);
    op.Process();
    return;
}

template <typename T>
__aicore__ inline void single_value_group_topk_impl(GM_ADDR topKInput, GM_ADDR idxArr,
                                                    AsdOps::GroupTopkTilingData &tilingData)
{
    AscendC::TPipe pipe;
    SingleValueGroupTopk<T> op(topKInput, idxArr, pipe, tilingData);
    op.Process();
    return;
}

extern "C" __global__ __aicore__ void group_topk(GM_ADDR topKInput, GM_ADDR idxArr, GM_ADDR topKOutput, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling, tilingData);
    if (TILING_KEY_IS(20000000000)) {
        group_topk_impl<half, GroupMultiFlag::UNDEFINED>(topKInput, idxArr, tilingData);
    } else if (TILING_KEY_IS(20000000010)) {
        group_topk_impl<half, GroupMultiFlag::SUM_MULTI_MAX>(topKInput, idxArr, tilingData);
    } else if (TILING_KEY_IS(900)) {
        single_value_group_topk_impl<half>(topKInput, idxArr, tilingData);
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    else if (TILING_KEY_IS(20000000001)) {
        group_topk_impl<bfloat16_t, GroupMultiFlag::UNDEFINED>(topKInput, idxArr, tilingData);
    } else if (TILING_KEY_IS(20000000011)) {
        group_topk_impl<bfloat16_t, GroupMultiFlag::SUM_MULTI_MAX>(topKInput, idxArr, tilingData);
    } else if (TILING_KEY_IS(901)) {
        single_value_group_topk_impl<bfloat16_t>(topKInput, idxArr, tilingData);
    }
#endif
}