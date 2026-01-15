/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef __VECTORBACKWARD_FP32_OP_H__
#define __VECTORBACKWARD_FP32_OP_H__
#define USE_ASCENDC 1

#ifdef __DAV_C220_VEC__

#include "ppmatmul_const_grad.h"
#include "kernel_operator.h"
#include "kernels/utils/kernel/common.h"
#include "kernels/utils/kernel/utils.h"
#include "kernels/utils/kernel/iterator.h"
#include "kernels/utils/kernel/simd.h"
namespace VEC_BACKWARD_FP32_OP {
using namespace AscendC;
using WORKSPACE_DTYPE = float;
using WORKSPACE_DTYPE_DP = float;
constexpr int32_t REPEAT_TIME_FP32 = 4;

/*
* 存放分段信息
*/
struct SectionInfo {
    // 当前处理的block条，包含block数量
    int32_t sectionNum{0};

    // 在当前workspace里的偏移，与对应的cube_id有关
    int32_t curBlockOffset{0};
    // 当前处理的block的总数量
    int32_t curBlockNum{0};

    // 起始block编号
    int32_t sectionStartBlock[MAX_SWITCH_TIME] = {0};
    // 结束block编号
    int32_t sectionEndBlock[MAX_SWITCH_TIME] = {0};

    // 在所有heads中的起始行
    int32_t globalLinesInHead[MAX_SWITCH_TIME] = {0};

    // 是否需要mask（如果需要，总在section end）
    bool apply_mask[MAX_SWITCH_TIME] = {false};

    // 每一段的长度
    int32_t len[MAX_SWITCH_TIME] = {0};

    // 头尾block是否需要mask
    bool headApplyMask[MAX_SWITCH_TIME] = {false};
    bool tailApplyMask[MAX_SWITCH_TIME] = {false};
};

/**
* 全局相关的信息
*/
struct GlobalInfo {
    // cube数量
    int32_t cubeNum{0};
    // 每个cube处理block的数量
    int32_t blockNumPerCube{0};
    int32_t headNum{0};
    int32_t batchNum{0};
    int32_t seq_len{0};
    int32_t qSeqLen{0};
    int32_t kSeqLen{0};

    // 是否是三角阵
    bool isTriMatrix{false};

    // 一行有几个block（三角阵和非三角阵不同）
    int32_t blockNumPerRow{0};
    // 一列有几个block（也就是一个head有几个行）
    int32_t blockNumPerCol{0};

    // 一个loop处理block的数量
    int32_t blockNumPerLoop{0};
    // 一个head包含的block数量
    int32_t blockNumPerHead{0};

    // 大循环次数
    int32_t loopTimes{0};
    // 最后一次循环处理的block数量
    int32_t tailBlockNum{0};

    int32_t isSparse{0};
    // 滑窗大小, sparse使用
    int32_t windowsBlockNum{0};
};

/**
* 当前vector的信息
*/
struct LocalInfo {
    // 所属的cube分组（第几个cube很重要）
    int32_t cubeIndex{0};
    // 当前vector在cube内的编号（0或1）
    int32_t vectorIndex{0};

    // 处理基本块的起始行（都是2个vector处理一个基本块）
    int32_t startLineInBaseBlock{0};

    // 是否参与尾块处理
    bool processTail{false};
    // 处理尾块中的block数量
    int32_t tailProcessedBlockNum{0};
};

template<typename INPUT_T, bool IF_BF16>
class VectorBackward {
public:
    __aicore__ inline VectorBackward() {}
    __aicore__ inline void Init(__gm__ INPUT_T * __restrict__ doGm,
                                __gm__ INPUT_T* __restrict__ oGm,
                                __gm__ WORKSPACE_DTYPE * __restrict__ sGm,
                                __gm__ float * __restrict__ rowmaxGm,
                                __gm__ float * __restrict__ rowsumGm,
                                __gm__ WORKSPACE_DTYPE_DP * __restrict__ dpGm,
                                __gm__ INPUT_T * __restrict__ maskGm,
                                bool isTriangle,
                                int32_t qSize,
                                int32_t kSize,
                                int32_t batch,
                                int32_t head,
                                int32_t baseLength,
                                int32_t blockNumPerCore,
                                int32_t isSparse,
                                int32_t windowLength,
                                int32_t maskSeqLength,
                                float scale);
    __aicore__ inline void Run();

private:
    __aicore__ inline void GetGlobalInfo(GlobalInfo *globalInfo);
    __aicore__ inline void GetLocalInfo(GlobalInfo *globalInfo,
                                        LocalInfo *localInfo,
                                        SectionInfo *sectionInfo);
    __aicore__ inline void AllocateUbufForNorm ();
    __aicore__ inline void UpdateLoopInfo(int32_t curLoop,
                                            GlobalInfo *globalInfo,
                                            LocalInfo *localInfo,
                                            SectionInfo *sectionInfo);
    // curProcessLines：总处理64行（2个vector处理一个基本块）BASE_BLOCK_SIZE_LEN_BACKWARD /2
    // oTotalOffset：O,dO的总偏移量，global_lines_in_heads*BASE_BLOCK_SIZE_LEN_BACKWARD，L是global_lines_in_heads
    // blockNumber：S,dP一次处理几块  endBlock - startBlock
    // sTotalOffset：S,dP的总偏移量start_block * BASE_BLOCK_DATA_NUM_BACKWARD 128*128
    // gradAttnOutputGm：dO
    // attnOutputGm：O
    // attnScoreGm：S
    // attnRowmaxGm：rowmax
    // attnRowsumGm：rowsum
    __aicore__ inline void ProcessBackwardFp32(int32_t curProcessLines,
                                                int32_t oTotalOffset,
                                                int32_t blockNumber,
                                                int32_t sTotalOffset,
                                                bool headApplyMask,
                                                bool tailApplyMask,
                                                __gm__ INPUT_T * __restrict__ gradAttnOutputGm,
                                                __gm__ INPUT_T * __restrict__ attnOutputGm,
                                                __gm__ WORKSPACE_DTYPE * __restrict__ attnScoreGm,
                                                __gm__ float *__restrict__ attnRowmaxGm,
                                                __gm__ float *__restrict__ attnRowsumGm,
                                                __gm__ WORKSPACE_DTYPE_DP * __restrict__ gradPGm,
                                                __gm__ INPUT_T * __restrict__ attnMaskGm);

private:
    // dO
    __gm__ INPUT_T *__restrict__ doGm{nullptr};
    // O
    __gm__ INPUT_T *__restrict__ oGm{nullptr};
    // S
    __gm__ WORKSPACE_DTYPE *__restrict__ sGm{nullptr};
    __gm__ float *__restrict__ rowmaxGm;
    __gm__ float *__restrict__ rowsumGm;
    __gm__ WORKSPACE_DTYPE_DP *__restrict__ dpGm{nullptr};
    __gm__ INPUT_T *__restrict__ maskGm{nullptr};
    __gm__ uint8_t *__restrict__ tilingParaGm{nullptr};
    __gm__ uint8_t *__restrict__ fftsAddr{nullptr};

    GlobalInfo globalInfo;
    LocalInfo localInfo;
    SectionInfo sectionInfo;
    bool isVector;
    bool isSync;
    int32_t H;
    int32_t maskSeqLength;
    float SCALE;
    AscendC::LocalTensor<float> temp0BufFp32;
    AscendC::LocalTensor<float> temp1BufFp32;
    AscendC::LocalTensor<INPUT_T> temp0BufFp16;
    AscendC::LocalTensor<INPUT_T> temp1BufFp16;
    AscendC::LocalTensor<float> bufForCalcRowsumFp32Buf;
    AscendC::LocalTensor<float> bufForBrcbRowsumFp32Buf;
    AscendC::LocalTensor<float> bufForLoadLFp32Buf;
    AscendC::LocalTensor<float> bufForBrcbLFp32Buf;
    AscendC::LocalTensor<float> bufForRowmaxFp32Buf;
    AscendC::LocalTensor<float> bufForRowsumFp32Buf;

    AscendC::LocalTensor<INPUT_T> fp16_temp_2_ub_buf;
    AscendC::LocalTensor<INPUT_T> bufForMaskFp16Buf;
    AscendC::LocalTensor<float> bufForMaskFp32Buf;

    int32_t doubleBufferAddr = MAX_LENG_PER_UB_PROC_BACKWARD / sizeof(float);
    int32_t doubleBufferAddrInputT = MAX_LENG_PER_UB_PROC_BACKWARD / sizeof(INPUT_T);
    AscendC::LocalTensor<float> temp0BufFp32Ping;
    AscendC::LocalTensor<float> temp1BufFp32Ping;
    AscendC::LocalTensor<INPUT_T> temp0BufFp16Ping;
    AscendC::LocalTensor<INPUT_T> temp1BufFp16Ping;
    AscendC::LocalTensor<float> bufForLoadLFp32BufPing;
    AscendC::LocalTensor<float> bufForRowmaxFp32BufPing;
    AscendC::LocalTensor<float> bufForRowsumFp32BufPing;
    AscendC::LocalTensor<INPUT_T> bufForMaskFp16BufPing;
    AscendC::LocalTensor<float> bufForMaskFp32BufPing;
    AscendC::GlobalTensor<INPUT_T> doGmGlobalTensor;
    AscendC::GlobalTensor<INPUT_T> oGmGlobalTensor;
    AscendC::GlobalTensor<WORKSPACE_DTYPE> sGmGlobalTensor;
    AscendC::GlobalTensor<float> rowmaxGmGlobalTensor;
    AscendC::GlobalTensor<float> rowsumGmGlobalTensor;
    AscendC::GlobalTensor<WORKSPACE_DTYPE_DP> dpGmGlobalTensor;
    AscendC::GlobalTensor<INPUT_T> maskGmGlobalTensor;
};

template<typename INPUT_T, bool IF_BF16>
__aicore__ inline void VectorBackward<INPUT_T, IF_BF16>::Init(
                                __gm__ INPUT_T * __restrict__ doGm,
                                __gm__ INPUT_T * __restrict__ oGm,
                                __gm__ WORKSPACE_DTYPE * __restrict__ sGm,
                                __gm__ float * __restrict__ rowmaxGm,
                                __gm__ float * __restrict__ rowsumGm,
                                __gm__ WORKSPACE_DTYPE_DP * __restrict__ dpGm,
                                __gm__ INPUT_T * __restrict__ maskGm,
                                bool isTriangle,
                                int32_t qSize,
                                int32_t kSize,
                                int32_t batch,
                                int32_t head,
                                int32_t baseLength,
                                int32_t blockNumPerCore,
                                int32_t isSparse,
                                int32_t windowLength,
                                int32_t maskSeqLength,
                                float scale)
{
    this->maskSeqLength = maskSeqLength;
    this->doGm = doGm;
    this->oGm = oGm;
    this->sGm = sGm;
    this->rowmaxGm = rowmaxGm;
    this->rowsumGm = rowsumGm;
    this->dpGm = dpGm;
    this->maskGm = maskGm;
    this->SCALE = scale;

    isVector = true;
    isSync = true;

    // tiling_para[10]
    this->globalInfo.qSeqLen = qSize;
    this->globalInfo.kSeqLen = kSize;
    // tiling_para[12]
    this->globalInfo.headNum = head;
    // tiling_para[0]
    this->globalInfo.batchNum = batch;
    // tiling_para[11]
    this->globalInfo.blockNumPerCube = blockNumPerCore;
    this->globalInfo.isTriMatrix = isTriangle;

    this->globalInfo.isSparse = isSparse;
    this->globalInfo.windowsBlockNum = windowLength / BASE_BLOCK_SIZE_LEN_BACKWARD;

    H = qSize / 128;
    // 获取全局信息
    GetGlobalInfo(&this->globalInfo);
    GetLocalInfo(&this->globalInfo, &this->localInfo, &this->sectionInfo);
    AllocateUbufForNorm();
    doGmGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(doGm));
    oGmGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(oGm));
    sGmGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ WORKSPACE_DTYPE *>(sGm));
    rowmaxGmGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rowmaxGm));
    rowsumGmGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rowsumGm));
    dpGmGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ WORKSPACE_DTYPE_DP *>(dpGm));
    maskGmGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(maskGm));
}

template<typename INPUT_T, bool IF_BF16>
__aicore__ inline void VectorBackward<INPUT_T, IF_BF16>::Run() {
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

    // attention score共有的block数量
    int32_t totalBlockNum = globalInfo.blockNumPerRow * globalInfo.blockNumPerCol *
                            globalInfo.headNum * globalInfo.batchNum;

    for (int32_t loopTimes = 0; loopTimes < globalInfo.loopTimes; loopTimes++) {
        // cube核ID*每个cube核16个block*（128*128）+ 24个cube核*每个cube核16个block*（128*128）*（0或1）
        // 每个核上的元素偏移，每两个loop是一对，因为两个vector核对应一个cube
        int32_t totalOffsetEachCore = get_block_idx() * globalInfo.blockNumPerCube *
                                        BASE_BLOCK_DATA_NUM_BACKWARD +
                                        get_block_num() * globalInfo.blockNumPerCube *
                                        BASE_BLOCK_DATA_NUM_BACKWARD * (loopTimes % 2);
        int32_t curBlockInGlobal = globalInfo.blockNumPerLoop * loopTimes + sectionInfo.curBlockOffset;
        UpdateLoopInfo(loopTimes, &this->globalInfo, &this->localInfo,  &this->sectionInfo);

        if (isSync) {
            WaitFlagDev(AIC2AIVFLAGID);
        }
        // 如果当前block index 超过block 总数，那么不计算
        if (curBlockInGlobal < totalBlockNum && isVector) {
            auto accProcessBlockNum = sectionInfo.sectionNum;
            for (int32_t section = 0; section < accProcessBlockNum; section ++) {

                auto startBlock = sectionInfo.sectionStartBlock[section];
                auto endBlock = sectionInfo.sectionEndBlock[section];
                auto headApplyMask = sectionInfo.headApplyMask[section];
                auto tailApplyMask = sectionInfo.tailApplyMask[section];
                auto curProcessBlockNum = sectionInfo.len[section];
                auto globalLinesInHead = sectionInfo.globalLinesInHead[section];
                ProcessBackwardFp32(BASE_BLOCK_SIZE_LEN_BACKWARD / 2,
                            globalLinesInHead * BASE_BLOCK_SIZE_LEN_BACKWARD,
                            curProcessBlockNum,
                            startBlock * BASE_BLOCK_DATA_NUM_BACKWARD + totalOffsetEachCore,
                            headApplyMask,
                            tailApplyMask,
                            doGm,
                            oGm,
                            sGm,
                            rowmaxGm,
                            rowsumGm,
                            dpGm,
                            maskGm);
            }
        }
        if (isSync) {
            FftsCrossCoreSync<PIPE_MTE3, 2>(AIV2AICFLAGID);
        }
    }
}

/**
* 全局信息
*/
template<typename INPUT_T, bool IF_BF16>
__aicore__ inline void VectorBackward<INPUT_T, IF_BF16>::GetGlobalInfo(GlobalInfo *globalInfo)
{
    globalInfo->cubeNum = GetBlockNum();

    int32_t blockNumPerCol = globalInfo->qSeqLen / BASE_BLOCK_SIZE_LEN_BACKWARD;
    globalInfo->blockNumPerRow = globalInfo->kSeqLen / BASE_BLOCK_SIZE_LEN_BACKWARD;
    globalInfo->blockNumPerCol = blockNumPerCol;

    if (globalInfo->isTriMatrix) {
        // 上下拼接后，每行需要处理的block数多一个
        globalInfo->blockNumPerRow += 1;
        // 上下拼接后，每列需要处理的block数减半
        globalInfo->blockNumPerCol /= 2;
    }

    if (globalInfo->isSparse == 1) {
        // 滑动窗口的大小再多一个block
        globalInfo->blockNumPerRow = globalInfo->windowsBlockNum + 1;
        // 三角阵部分上下拼接后折半
        globalInfo->blockNumPerCol = blockNumPerCol - globalInfo->windowsBlockNum / 2;
    }

    // 一次循环处理的block数量
    globalInfo->blockNumPerLoop = globalInfo->cubeNum * globalInfo->blockNumPerCube;
    globalInfo->blockNumPerHead = globalInfo->blockNumPerRow * globalInfo->blockNumPerCol;

    // 一个[b,n,s1,s2]有多少个block
    int32_t totalBlockNum = globalInfo->blockNumPerHead *
                            globalInfo->headNum * globalInfo->batchNum;
    // loop_times展开为：seqLenK/128 * qSeqLen/128 * headNum * batchNum / (24 * 16)
    globalInfo->loopTimes = totalBlockNum / globalInfo->blockNumPerLoop;
    // 最后的尾块需要处理的block数量
    globalInfo->tailBlockNum = totalBlockNum % globalInfo->blockNumPerLoop;

    // 如果有tail_block，loop_times需要多加一次
    if (globalInfo->tailBlockNum > 0) {
        globalInfo->loopTimes += 1;
    }
}

template<typename INPUT_T, bool IF_BF16>
__aicore__ inline void VectorBackward<INPUT_T, IF_BF16>::GetLocalInfo(GlobalInfo *globalInfo,
                                                                LocalInfo *localInfo,
                                                                SectionInfo * sectionInfo)
{
    localInfo->cubeIndex = get_block_idx();
    // 0或1
    localInfo->vectorIndex = get_subblockid();

    // 128 / 2 * 0或1，代表每个vcore从base_block的第几行开始处理，编号0从第0行，编号1从第64行开始
    localInfo->startLineInBaseBlock = BASE_BLOCK_SIZE_LEN_BACKWARD / 2 * localInfo->vectorIndex;

    // 初始化secion_info
    sectionInfo->curBlockNum = globalInfo->blockNumPerCube;

    // 当前一组vector处理的偏移量总是相同的
    sectionInfo->curBlockOffset = globalInfo->blockNumPerCube * localInfo->cubeIndex;

    localInfo->processTail = false;

    // 判断尾块处理信息
    if (globalInfo->tailBlockNum > 0)
    {
        int32_t cubeUsed = globalInfo->tailBlockNum / globalInfo->blockNumPerCube;
        int32_t tailBlock = globalInfo->tailBlockNum % globalInfo->blockNumPerCube;


        if (localInfo->cubeIndex < cubeUsed) {
            localInfo->processTail = true;
            localInfo->tailProcessedBlockNum = globalInfo->blockNumPerCube;
        }

        if (tailBlock > 0) {
            if (localInfo->cubeIndex == cubeUsed) {
                localInfo->processTail = true;
                // 会有两个vector处理少于blockNumPerCube个block
                localInfo->tailProcessedBlockNum = tailBlock;
            }
        }
    }
}

/**
* 获得循环信息
*/
template<typename INPUT_T, bool IF_BF16>
__aicore__ inline void VectorBackward<INPUT_T, IF_BF16>::UpdateLoopInfo(int32_t curLoop,
                                                                GlobalInfo *globalInfo,
                                                                LocalInfo *localInfo,
                                                                SectionInfo * sectionInfo)
{
    // 主要要求解的：1. 每段的长度 2. 每段最后一个是否要加mask 3.每一段 的全局行index
    int64_t blocksPerBatch = globalInfo->headNum *
                    globalInfo->blockNumPerRow * globalInfo->blockNumPerCol;
    int64_t allBlocks = globalInfo->batchNum * blocksPerBatch;
    // 16*24*当前第几个loop + 当前cube的index*16，每次处理16个blcok，此为第一个block的id
    int64_t curBlockId = globalInfo->blockNumPerCube * globalInfo->cubeNum * curLoop +
                localInfo->cubeIndex * globalInfo->blockNumPerCube;

    int32_t blocksPerRow = globalInfo->blockNumPerRow;
    int32_t blocksPerCol = globalInfo->blockNumPerCol;

    // 倒三角或者sparse场景负载均衡后折叠的行数
    int32_t balancedRowNum = globalInfo->windowsBlockNum / 2;

    // 第一个block所在的第b个batch
    int64_t b = curBlockId / blocksPerBatch;
    // 第一个block所在的第h个head
    int64_t n = (curBlockId % blocksPerBatch) / (blocksPerRow * blocksPerCol);
    // 第一个block所在的第ir行
    int64_t ir = (curBlockId % blocksPerBatch) % (blocksPerRow * blocksPerCol) / (blocksPerRow);
    // 第一个block所在的第ic行
    int64_t ic = (curBlockId % blocksPerBatch) % (blocksPerRow * blocksPerCol) % (blocksPerRow);

    // 当前轮次cur_core_index处理的行

    // 默认remain为每个cube处理16个block
    int64_t remain = globalInfo->blockNumPerCube;
    // 如果是最后一个cube，则remain的数量更新为剩余的block数量
    if (curBlockId + globalInfo->blockNumPerCube > allBlocks) {
        remain = allBlocks - curBlockId;
    }
    // 当前处理的block条，包含几个block，如果已经没有remain了，自然就没有block了，结束
    if (remain <= 0) {
        sectionInfo->sectionNum = 0;
        return;
    }
    // 当前16个block的block条占的行数
    int64_t rows = (ic + remain + blocksPerRow - 1) / blocksPerRow;

    Addr addr[32];
    addr[0].b = b;
    addr[0].n = n;
    addr[0].iR = ir;
    addr[0].iC = ic;
    addr[0].k = remain;
    int64_t curAddrLen = 0;
    // 给addr赋值
    for (int i = 0; i < rows && remain > 0; ++i) {
        // 如果剩余需要处理的数量超过一行的长度
        if (addr[curAddrLen].iC + remain > blocksPerRow) {
            // 当前行需要处理的剩余block数量为每行的block的数量减去当前block的column index
            addr[curAddrLen].k = blocksPerRow - addr[curAddrLen].iC;
            // 下一行的剩余block数量为剩余block数量减去上一行需要处理的block数量
            addr[curAddrLen + 1].k = remain - addr[curAddrLen].k;

            // 下一行的b，n不变，iR切换到下一行，所以+1，iC从头开始，所以为0
            addr[curAddrLen + 1].b = addr[curAddrLen].b;
            addr[curAddrLen + 1].n = addr[curAddrLen].n;
            addr[curAddrLen + 1].iR = addr[curAddrLen].iR + 1;
            addr[curAddrLen + 1].iC = 0;
            // 如果当前的row index大于等于每列的block个数，也就是说row index超出边界
            if (addr[curAddrLen + 1].iR >= blocksPerCol) {
                // 则切换到下一个n，且row_index归零从头开始
                addr[curAddrLen + 1].n = addr[curAddrLen].n + 1;
                addr[curAddrLen + 1].iR = 0;
                // 如果n超过n的上线，切换到下一个batch，且n归零从头开始
                if (addr[curAddrLen + 1].n >= globalInfo->headNum) {
                    addr[curAddrLen + 1].b = addr[curAddrLen].b + 1;
                    addr[curAddrLen + 1].n = 0;
                }
            }
        }
        // 每一个循环处理完之后，把剩余的block数量减去已经处理过的block数量
        remain -= addr[curAddrLen].k;
        ++curAddrLen;
    }

    // 非三角阵（也要知道原始head的行号，head收尾拼接）
    if (globalInfo->isTriMatrix == false) {
        int64_t index = 0;
        // 通过addr给sectionInfo赋值，从addr[0]开始，一直到addr[curAddrLen - 1]
        for (int i = 0; i < curAddrLen; ++i) {
            int32_t iR = addr[i].iR;
            int32_t iC = addr[i].iC;
            int32_t k = addr[i].k;
            int32_t rowLeftIndex = iR;
            int32_t rowRightIndex = blocksPerCol - 1 - iR;
            int64_t col_left_index = iC;
            int32_t switchIndex = blocksPerRow;
            // 用在[b,n,r,c]中的block的index
            // 减去当前loop已经处理过的block数量，减去当前cube之前的cube已经处理过的block数量
            // 得到
            int32_t ColIndexForDb = (addr[i].b*globalInfo->headNum + addr[i].n) *
                                (blocksPerRow) * (blocksPerCol) +
                                iR * (blocksPerRow) + iC -
                                curLoop * globalInfo->blockNumPerCube * globalInfo->cubeNum -
                                localInfo->cubeIndex * globalInfo->blockNumPerCube;

            // 起始块
            sectionInfo->sectionStartBlock[index] = ColIndexForDb;
            // 结束块（不含）
            sectionInfo->len[index] = k;
            sectionInfo->apply_mask[index] = false;
            sectionInfo->headApplyMask[index] = false;
            sectionInfo->tailApplyMask[index] = false;
            sectionInfo->globalLinesInHead[index] = (addr[i].b *globalInfo->headNum + addr[i].n) *
                                    128 * blocksPerCol +
                                    rowLeftIndex * 128 +
                                    localInfo->startLineInBaseBlock;
            index++;
        }
        sectionInfo->sectionNum = index;
    }

    if (globalInfo->isTriMatrix == true || globalInfo->isSparse == 1) {
        int64_t index = 0;
        for (int i = 0; i < curAddrLen; ++i) {
            int32_t iR = addr[i].iR;
            int32_t iC = addr[i].iC;
            int32_t k = addr[i].k;
            int32_t rowLeftIndex = balancedRowNum + iR;
            int32_t rowRightIndex = balancedRowNum - 1 - iR;
            int64_t col_left_index = iC;
            int32_t switchIndex = rowLeftIndex;
            int64_t col_right_index = iC - (iR + blocksPerCol) - 1;
            int32_t ColIndexForDb = (addr[i].b * globalInfo->headNum  + addr[i].n) *
                                (blocksPerRow) * (blocksPerCol) +
                                iR * (blocksPerRow) + iC -
                                curLoop * globalInfo->blockNumPerCube * globalInfo->cubeNum -
                                localInfo->cubeIndex * globalInfo->blockNumPerCube;

            if (iR >= balancedRowNum) {
                // 起始iC
                sectionInfo->sectionStartBlock[index] = ColIndexForDb;
                sectionInfo->len[index] = k;
                // 最后iC
                sectionInfo->sectionEndBlock[index] = ColIndexForDb + sectionInfo->len[index];
                // 起始iR(全局)
                sectionInfo->globalLinesInHead[index] = (addr[i].b * globalInfo->headNum + addr[i].n) *
                                        globalInfo->qSeqLen +
                                        rowLeftIndex * 128 +
                                        localInfo->startLineInBaseBlock;
                sectionInfo->headApplyMask[index] = (iC == 0);
                sectionInfo->tailApplyMask[index] = (iC + sectionInfo->len[index] == blocksPerRow);
                ++index;
            } else if (switchIndex < iC) {
                // 起始iC
                sectionInfo->sectionStartBlock[index] = ColIndexForDb;
                sectionInfo->len[index] = k;
                // 最后iC
                sectionInfo->sectionEndBlock[index] = ColIndexForDb + sectionInfo->len[index];
                // 起始iR(全局)
                sectionInfo->globalLinesInHead[index] = (addr[i].b * globalInfo->headNum + addr[i].n) *
                                        globalInfo->qSeqLen +
                                        rowRightIndex * 128 +
                                        localInfo->startLineInBaseBlock;
                sectionInfo->headApplyMask[index] = false;
                sectionInfo->tailApplyMask[index] = (iC + sectionInfo->len[index] == blocksPerRow);
                ++index;
            } else if (iC <= switchIndex && iC + k - 1 > switchIndex) {
                // 起始iC
                sectionInfo->sectionStartBlock[index] = ColIndexForDb;
                sectionInfo->len[index] = switchIndex - iC + 1 ;
                // 最后iC
                sectionInfo->sectionEndBlock[index] = ColIndexForDb + sectionInfo->len[index];
                // 起始iR(全局)
                sectionInfo->globalLinesInHead[index] = (addr[i].b * globalInfo->headNum + addr[i].n) *
                                        globalInfo->qSeqLen +
                                        rowLeftIndex * 128 +
                                        localInfo->startLineInBaseBlock;
                sectionInfo->headApplyMask[index] = false;
                sectionInfo->tailApplyMask[index] = true;

                // 起始iC (switchIndex + 1)
                sectionInfo->sectionStartBlock[index+1] = sectionInfo->sectionStartBlock[index] +
                                    sectionInfo->len[index];
                sectionInfo->len[index+1] = k - sectionInfo->len[index];
                // 最后iC
                sectionInfo->sectionEndBlock[index+1] = sectionInfo->sectionStartBlock[index+1] +
                                    sectionInfo->len[index+1];
                // 起始iR(全局)
                sectionInfo->globalLinesInHead[index+1] = (addr[i].b *globalInfo->headNum + addr[i].n) *
                                        globalInfo->qSeqLen +
                                        rowRightIndex * 128 +
                                        localInfo->startLineInBaseBlock;
                sectionInfo->headApplyMask[index+1] = false;
                // 最后一块是否需要 mask, 是否是跳变点
                sectionInfo->tailApplyMask[index+1] =
                                    (switchIndex+1 + sectionInfo->len[index+1] == blocksPerRow);
                index += 2;
            } else {
                // 起始iC
                sectionInfo->sectionStartBlock[index] = ColIndexForDb;
                sectionInfo->len[index] = k;
                // 最后iC
                sectionInfo->sectionEndBlock[index] = ColIndexForDb + sectionInfo->len[index];
                sectionInfo->globalLinesInHead[index] = (addr[i].b *globalInfo->headNum  + addr[i].n) *
                                        globalInfo->qSeqLen +
                                        rowLeftIndex*128 +
                                        localInfo->startLineInBaseBlock;
                sectionInfo->headApplyMask[index] = false;
                sectionInfo->tailApplyMask[index] = ((iC + k - 1) == switchIndex);
                index++;
            }
        }
        sectionInfo->sectionNum = index;
    }
}

/**
* 为反向计算分配UB空间
*/
template<typename INPUT_T, bool IF_BF16>
__aicore__ inline void  VectorBackward<INPUT_T, IF_BF16>::AllocateUbufForNorm () {
    AsdopsBuffer<ArchType::ASCEND_V220> UBuf; // 模板参数为TPosition中的VECCALC类型
    // O 32kb
    int32_t temp0BufAddrFp32Size = MAX_LENG_PER_UB_PROC_BACKWARD * 4 * 2;
    // do 32kb
    int32_t temp1BufAddrFp32Size = MAX_LENG_PER_UB_PROC_BACKWARD * 4 * 2;

    int32_t temp0BufAddrFp16Size = MAX_LENG_PER_UB_PROC_BACKWARD * 2 * 2;
    int32_t temp1BufAddrFp16Size = MAX_LENG_PER_UB_PROC_BACKWARD * 2 * 2;

    // 最多处理的FP32数字个数  //d=rowsum(odo)
    int32_t bufForCalcRowsumFp32Size =  128 * 4 * 2;
    // 展开做32字节对齐 8倍rowsum
    int32_t bufForBrcbRowsumFp32Size = bufForCalcRowsumFp32Size * (32 / 4);

    // L
    int32_t bufForLoadLFp32Size = 128 * 4 * 2;
    // 8倍L
    int32_t bufForBrcbLFp32Size = bufForLoadLFp32Size * (32 / 4);

    int32_t bufForRowmaxFp32Size = BASE_BLOCK_SIZE_LEN_BACKWARD * 4 * 2;
    int32_t bufForRowsumFp32Size = BASE_BLOCK_SIZE_LEN_BACKWARD * 4 * 2;

    // exp(S-L) //32kb
    int32_t temp2BufAddrFp16Size = MAX_LENG_PER_UB_PROC_BACKWARD * 2 * 2;

    int32_t bufForMaskFp16Size = 64 * 128 * 2;
    int32_t bufForMaskFp32Size = 64 * 128 * 4;

    int32_t lastAddr = 0;

    // load O
    // D计算完毕后,s复用fp32_temp_0_ub_buf_addr，s-l也可以用该空间，exp(s-l)
    temp0BufFp32 = UBuf.GetBuffer<BufferType::ASCEND_UB, float>(lastAddr);
    temp0BufFp32Ping = temp0BufFp32[MAX_LENG_PER_UB_PROC_BACKWARD];

    lastAddr += temp0BufAddrFp32Size;

    temp1BufFp32 = UBuf.GetBuffer<BufferType::ASCEND_UB, float>(lastAddr);
    temp1BufFp32Ping = temp1BufFp32[MAX_LENG_PER_UB_PROC_BACKWARD];

    // load dO

    lastAddr += temp1BufAddrFp32Size;

    temp0BufFp16 = UBuf.GetBuffer<BufferType::ASCEND_UB, INPUT_T>(lastAddr);
    temp0BufFp16Ping = temp0BufFp16[MAX_LENG_PER_UB_PROC_BACKWARD];

    lastAddr += temp0BufAddrFp16Size;

    temp1BufFp16 = UBuf.GetBuffer<BufferType::ASCEND_UB, INPUT_T>(lastAddr);
    temp1BufFp16Ping = temp1BufFp16[MAX_LENG_PER_UB_PROC_BACKWARD];

    lastAddr += temp1BufAddrFp16Size;

    bufForCalcRowsumFp32Buf = UBuf.GetBuffer<BufferType::ASCEND_UB, float>(lastAddr);

    // cacl rowsum  32字节对齐后，输出到 bufForBrcbRowsumFp32
    lastAddr += bufForCalcRowsumFp32Size;

    bufForBrcbRowsumFp32Buf = UBuf.GetBuffer<BufferType::ASCEND_UB, float>(lastAddr);

    lastAddr += bufForBrcbRowsumFp32Size;

    bufForLoadLFp32Buf = UBuf.GetBuffer<BufferType::ASCEND_UB, float>(lastAddr);
    bufForLoadLFp32BufPing = bufForLoadLFp32Buf[MAX_BLOCK_PER_ONE_PROC_BACKWARD];

    // load l
    lastAddr += bufForLoadLFp32Size;

    bufForBrcbLFp32Buf = UBuf.GetBuffer<BufferType::ASCEND_UB, float>(lastAddr);

    // load 8倍l
    lastAddr += bufForBrcbLFp32Size;

    bufForRowmaxFp32Buf = UBuf.GetBuffer<BufferType::ASCEND_UB, float>(lastAddr);
    bufForRowmaxFp32BufPing = bufForRowmaxFp32Buf[MAX_BLOCK_PER_ONE_PROC_BACKWARD];

    // load rowmax
    lastAddr += bufForRowmaxFp32Size;

    bufForRowsumFp32Buf = UBuf.GetBuffer<BufferType::ASCEND_UB, float>(lastAddr);
    bufForRowsumFp32BufPing = bufForRowsumFp32Buf[MAX_BLOCK_PER_ONE_PROC_BACKWARD];

    // load rowsum
    lastAddr += bufForRowsumFp32Size;

    fp16_temp_2_ub_buf = UBuf.GetBuffer<BufferType::ASCEND_UB, INPUT_T>(lastAddr);

    // exp(s-L)
    lastAddr += temp2BufAddrFp16Size;

    bufForMaskFp16Buf = UBuf.GetBuffer<BufferType::ASCEND_UB, INPUT_T>(lastAddr);
    bufForMaskFp16BufPing = bufForMaskFp16Buf[MAX_LENG_PER_UB_PROC_BACKWARD];

    lastAddr += bufForMaskFp16Size;

    bufForMaskFp32Buf = UBuf.GetBuffer<BufferType::ASCEND_UB, float>(lastAddr);
    bufForMaskFp32BufPing = bufForMaskFp32Buf[MAX_LENG_PER_UB_PROC_BACKWARD];

    doGmGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(this->doGm));
    oGmGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(this->oGm));
    sGmGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ WORKSPACE_DTYPE *>(this->sGm));
    rowmaxGmGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(this->rowmaxGm));
    rowsumGmGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(this->rowsumGm));
    dpGmGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ WORKSPACE_DTYPE *>(this->dpGm));
    maskGmGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(this->maskGm));
}

template<typename INPUT_T, bool IF_BF16>
__aicore__ inline void VectorBackward<INPUT_T, IF_BF16>::ProcessBackwardFp32(int32_t curProcessLines,
                                        int32_t oTotalOffset,
                                        int32_t blockNumber,
                                        int32_t sTotalOffset,
                                        bool headApplyMask,
                                        bool tailApplyMask,
                                        __gm__ INPUT_T * __restrict__ gradAttnOutputGm,
                                        __gm__ INPUT_T * __restrict__ attnOutputGm,
                                        __gm__ WORKSPACE_DTYPE * __restrict__ attnScoreGm,
                                        __gm__ float *__restrict__ attnRowmaxGm,
                                        __gm__ float *__restrict__ attnRowsumGm,
                                        __gm__ WORKSPACE_DTYPE_DP * __restrict__ gradPGm,
                                        __gm__ INPUT_T * __restrict__ attnMaskGm)
{
    int32_t pingFlag = 0;

    // dO * O
    SET_FLAG(V, MTE2, EVENT_ID0);
    SET_FLAG(V, MTE2, EVENT_ID1);

    for (int i = 0; i < 2;i++) {
        int32_t oOffset = curProcessLines / 2 * BASE_BLOCK_SIZE_LEN_BACKWARD * i;
        int32_t lOffset = oOffset / BASE_BLOCK_SIZE_LEN_BACKWARD;
        auto eventId = pingFlag ? EVENT_ID0:EVENT_ID1;

        WAIT_FLAG(V, MTE2, eventId);
        AscendC::LocalTensor<INPUT_T> temp0BufPingpongFP16 = pingFlag ? temp0BufFp16Ping : temp0BufFp16;
        AscendC::LocalTensor<INPUT_T> temp1BufPingpongFP16 = pingFlag ? temp1BufFp16Ping : temp1BufFp16;
        AscendC::LocalTensor<float> bufForLoadLFp32BufPingpongFp32 = pingFlag ?
                                                            bufForLoadLFp32BufPing : bufForLoadLFp32Buf;
        AscendC::LocalTensor<float> bufForRowmaxFp32BufPingpongFp32 = pingFlag ?
                                                            bufForRowmaxFp32BufPing : bufForRowmaxFp32Buf;
        AscendC::LocalTensor<float> bufForRowsumFp32BufPingpongFp32 = pingFlag ?
                                                            bufForRowsumFp32BufPing : bufForRowsumFp32Buf;

        // copy O to ub
        AscendC::DataCopy(temp0BufPingpongFP16, oGmGlobalTensor[oTotalOffset + oOffset],
                          AscendC::DataCopyParams(1,
                                                  MAX_LENG_PER_UB_PROC_BACKWARD * 2 / 32, // 一个block=32B N*2/32
                                                  0,
                                                  0));
        // copy dO to ub
        AscendC::DataCopy(temp1BufPingpongFP16, doGmGlobalTensor[oTotalOffset + oOffset],
                          AscendC::DataCopyParams(1,
                                                  MAX_LENG_PER_UB_PROC_BACKWARD * 2 / 32, // 一个block=32B N*2/32
                                                  0,
                                                  0));
        // copy rowmax to ub
        AscendC::DataCopy(bufForRowmaxFp32BufPingpongFp32, rowmaxGmGlobalTensor[oTotalOffset / 128 + lOffset],
                          AscendC::DataCopyParams(1,
                                                  MAX_BLOCK_PER_ONE_PROC_BACKWARD * 4 / 32,
                                                  0,
                                                  0));
        // copy rowsum to ub
        AscendC::DataCopy(bufForRowsumFp32BufPingpongFp32, rowsumGmGlobalTensor[oTotalOffset / 128 + lOffset],
                          AscendC::DataCopyParams(1,
                                                  MAX_BLOCK_PER_ONE_PROC_BACKWARD * 4 / 32,
                                                  0,
                                                  0));
        SET_FLAG(MTE2, V, eventId);
        WAIT_FLAG(MTE2, V, eventId);

        // O,dO fp16--fp32
        AscendC::SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        if constexpr(IF_BF16) {
            conv_v<ArchType::ASCEND_V220>(temp0BufFp32, temp0BufPingpongFP16,
                                          MAX_LENG_PER_UB_PROC_BACKWARD/64, 1, 1, 8, 4);
            PIPE_BARRIER(V);
            conv_v<ArchType::ASCEND_V220>(temp1BufFp32, temp1BufPingpongFP16,
                                          MAX_LENG_PER_UB_PROC_BACKWARD/64, 1, 1, 8, 4);
            PIPE_BARRIER(V);
        }
        else {
            conv_v<ArchType::ASCEND_V220>(temp0BufFp32, temp0BufPingpongFP16,
                                          MAX_LENG_PER_UB_PROC_BACKWARD/64, 1, 1, 8, 4);
            PIPE_BARRIER(V);

            conv_v<ArchType::ASCEND_V220>(temp1BufFp32, temp1BufPingpongFP16,
                                          MAX_LENG_PER_UB_PROC_BACKWARD/64, 1, 1, 8, 4);
            PIPE_BARRIER(V);
        }

        // O*dO
        AscendC::SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);

        mul_v<ArchType::ASCEND_V220>(temp0BufFp32,
                                    temp0BufFp32,
                                    temp1BufFp32,
                                    MAX_LENG_PER_UB_PROC_BACKWARD/64,
                                    1, 1, 1, 8, 8, 8);

        PIPE_BARRIER(V);

        // 折半
        AscendC::SetVectorMask<int8_t>(0x0, 0xffffffffffffffff); // 只对每个repeat的前64个元素处理

        add_v<ArchType::ASCEND_V220>(
            temp0BufFp32,
            temp0BufFp32,
            temp0BufFp32[64],
            MAX_BLOCK_PER_ONE_PROC_BACKWARD, 1, 1, 1,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32);
        PIPE_BARRIER(V);

        AscendC::SetVectorMask<int8_t>(0x0, 0xffffffff); // 只对每个repeat的前32个元素处理
        add_v<ArchType::ASCEND_V220>(
            temp0BufFp32,
            temp0BufFp32,
            temp0BufFp32[32],
            MAX_BLOCK_PER_ONE_PROC_BACKWARD, 1, 1, 1,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32);
        PIPE_BARRIER(V);

        AscendC::SetVectorMask<int8_t>(0x0, 0xffff); // 只对每个repeat的前16个元素处理
        add_v<ArchType::ASCEND_V220>(
            temp0BufFp32,
            temp0BufFp32,
            temp0BufFp32[16],
            MAX_BLOCK_PER_ONE_PROC_BACKWARD, 1, 1, 1,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32);
        PIPE_BARRIER(V);

        AscendC::SetVectorMask<int8_t>(0x0, 0xff); // 只对每个repeat的前8个元素处理
        add_v<ArchType::ASCEND_V220>(
            temp0BufFp32,
            temp0BufFp32,
            temp0BufFp32[8],
            MAX_BLOCK_PER_ONE_PROC_BACKWARD, 1, 1, 1, 1,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32);
        PIPE_BARRIER(V);

        // vcgadd将每32B的元素reduce add成一个数
        AscendC::SetVectorMask<int8_t>(0x0, 0xffffffffffffffff);
        auto rowsumOffset = pingFlag ? MAX_BLOCK_PER_ONE_PROC_BACKWARD : 0;
        BlockReduceSum<float, false>(bufForCalcRowsumFp32Buf[rowsumOffset], temp0BufFp32, 8, 0, 1, 1, 8);
        PIPE_BARRIER(V);

        SET_FLAG(V, MTE2, eventId);
        pingFlag = 1 - pingFlag;
    }

    WAIT_FLAG(V, MTE2, EVENT_ID0);
    WAIT_FLAG(V, MTE2, EVENT_ID1);

    AscendC::SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);

    // rowsum广播（怎么控制一个数广播后和下面的一行去做）
    brcb_v<ArchType::ASCEND_V220>(bufForBrcbRowsumFp32Buf, bufForCalcRowsumFp32Buf, 1, 8, 8);

    PIPE_BARRIER(V);

    ln_v<ArchType::ASCEND_V220>(bufForRowsumFp32Buf, bufForRowsumFp32Buf, 1, 1, 1, 8, 8);

    PIPE_BARRIER(V);

    add_v<ArchType::ASCEND_V220>(bufForLoadLFp32Buf, bufForRowmaxFp32Buf, bufForRowsumFp32Buf,
        1, 1, 1, 1, BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
        BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
        BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32);
    PIPE_BARRIER(V);

    // 8倍l
    brcb_v<ArchType::ASCEND_V220>(bufForBrcbLFp32Buf, bufForLoadLFp32Buf, 1, 8, 8);
    PIPE_BARRIER(V);

    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);

    // 逐行计算
    for (int k = 0; k < blockNumber; k++) {
        // 每次copy 128 * block_num 个数据  ub = 128 * block_num * sizeof() * 2
        for (int n = 0; n < 2; n++) {
            // vector之间的偏移
            int32_t sOffset = 64 * 128 * get_subblockid() + k * 128 * 128 + n * MAX_LENG_PER_UB_PROC_BACKWARD;
            int32_t ssOffset = 64 * 128 * get_subblockid() + k * 128 * 128 + n * MAX_LENG_PER_UB_PROC_BACKWARD/2;
            auto eventId = pingFlag ? EVENT_ID0:EVENT_ID1;
            WAIT_FLAG(MTE3, MTE2, eventId);
            AscendC::LocalTensor<float> temp0BufPingpongFP32 = pingFlag ? temp0BufFp32Ping : temp0BufFp32;
            AscendC::LocalTensor<INPUT_T> temp0BufPingpongFP16 = pingFlag ? temp0BufFp16Ping : temp0BufFp16;
            AscendC::LocalTensor<INPUT_T> bufForMaskFp16BufPingpong = pingFlag ?
                                                            bufForMaskFp16BufPing : bufForMaskFp16Buf;
            AscendC::LocalTensor<float> bufForMaskFp32BufPingpong = pingFlag ?
                                                            bufForMaskFp32BufPing : bufForMaskFp32Buf;

            // copy S to ub

            AscendC::DataCopy(temp0BufPingpongFP32, sGmGlobalTensor[sTotalOffset + sOffset],
                              AscendC::DataCopyParams(1,
                                                      MAX_LENG_PER_UB_PROC_BACKWARD * REPEAT_TIME_FP32 / 32,
                                                      0,
                                                      0));
            SET_FLAG(MTE2, V, eventId);
            WAIT_FLAG(MTE2, V, eventId);
            // S*scale

            muls_v<ArchType::ASCEND_V220>(temp0BufPingpongFP32,
                                          temp0BufPingpongFP32,
                                          (float)SCALE,
                                          MAX_BLOCK_PER_ONE_PROC_BACKWARD*2, 1, 1, 8, 8);
            PIPE_BARRIER(V);

            if (headApplyMask && k == 0) {
                SET_FLAG(V, MTE2, eventId);
                WAIT_FLAG(V, MTE2, eventId);

                AscendC::DataCopy(bufForMaskFp16BufPingpong,
                                maskGmGlobalTensor[64 * maskSeqLength  * get_subblockid() + 1 + n * 32 * maskSeqLength],
                                AscendC::DataCopyParams(32,
                                                        128 * 2 / 32,
                                                        (maskSeqLength  -128)/16,
                                                        0));
                SET_FLAG(MTE2, V, eventId);
                WAIT_FLAG(MTE2, V, eventId);

                Duplicate<float, false>(temp1BufFp32, float(1.0), uint64_t(0),
                                        MAX_BLOCK_PER_ONE_PROC_BACKWARD*2, 1, 8);

                PIPE_BARRIER(V);

                if constexpr(IF_BF16) {
                    conv_v<ArchType::ASCEND_V220>(bufForMaskFp32BufPingpong, bufForMaskFp16BufPingpong,
                                                  MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2, 1, 1, 8, 4);
                    PIPE_BARRIER(V);
                }
                else {
                    conv_v<ArchType::ASCEND_V220>(bufForMaskFp32BufPingpong, bufForMaskFp16BufPingpong,
                                                  MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2, 1, 1, 8, 4);
                    PIPE_BARRIER(V);
                }

                sub_v<ArchType::ASCEND_V220>(bufForMaskFp32BufPingpong,
                                             temp1BufFp32,
                                             bufForMaskFp32BufPingpong,
                                             MAX_BLOCK_PER_ONE_PROC_BACKWARD*2, 1, 1, 1, 8, 8, 8);
                PIPE_BARRIER(V);

                muls_v<ArchType::ASCEND_V220>(bufForMaskFp32BufPingpong,
                                              bufForMaskFp32BufPingpong,
                                              (float)(PADDING_FOR_MAX), 32 *2, 1, 1, 8, 8);
                PIPE_BARRIER(V);

                add_v<ArchType::ASCEND_V220>(
                    temp0BufPingpongFP32,
                    temp0BufPingpongFP32,
                    bufForMaskFp32BufPingpong, 32*2, 1, 1, 1, 8, 8, 8);
                PIPE_BARRIER(V);
            }

            if (tailApplyMask && k == blockNumber - 1) {
                SET_FLAG(V, MTE2, eventId);
                WAIT_FLAG(V, MTE2, eventId);

                AscendC::DataCopy(bufForMaskFp16BufPingpong,
                                  maskGmGlobalTensor[64 * maskSeqLength * get_subblockid() + n * 32 * maskSeqLength],
                                  AscendC::DataCopyParams(32,
                                                          128 * 2 / 32,
                                                          (maskSeqLength - 128)/16,
                                                          0));
                SET_FLAG(MTE2, V, eventId);
                WAIT_FLAG(MTE2, V, eventId);

                if constexpr(IF_BF16) {
                    conv_v<ArchType::ASCEND_V220>(bufForMaskFp32BufPingpong, bufForMaskFp16BufPingpong,
                                                  MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2, 1, 1, 8, 4);
                    PIPE_BARRIER(V);
                }
                else {
                    conv_v<ArchType::ASCEND_V220>(bufForMaskFp32BufPingpong, bufForMaskFp16BufPingpong,
                                                  MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2, 1, 1, 8, 4);
                    PIPE_BARRIER(V);
                }

                muls_v<ArchType::ASCEND_V220>(bufForMaskFp32BufPingpong,
                                              bufForMaskFp32BufPingpong,
                                              (float)(PADDING_FOR_MAX), 32 *2, 1, 1, 8, 8);
                PIPE_BARRIER(V);

                add_v<ArchType::ASCEND_V220>(
                    temp0BufPingpongFP32,
                    temp0BufPingpongFP32,
                    bufForMaskFp32BufPingpong, 32*2, 1, 1, 1, 8, 8, 8);
                PIPE_BARRIER(V);
            }

            auto rowsumOffset = pingFlag ? 32*8  : 0 ;
            // vsub一次最多64个数 (s-L)
            sub_v<ArchType::ASCEND_V220>(temp0BufPingpongFP32,
                                         temp0BufPingpongFP32,
                                         bufForBrcbLFp32Buf[rowsumOffset],
                                         MAX_BLOCK_PER_ONE_PROC_BACKWARD, 1, 1, 0,
                                         BASE_BLOCK_SIZE_LEN_BACKWARD / 8,
                                         BASE_BLOCK_SIZE_LEN_BACKWARD / 8, 1);
            PIPE_BARRIER(V);
            sub_v<ArchType::ASCEND_V220>(temp0BufPingpongFP32[64],
                                         temp0BufPingpongFP32[64],
                                         bufForBrcbLFp32Buf[rowsumOffset],
                                         MAX_BLOCK_PER_ONE_PROC_BACKWARD, 1, 1, 0,
                                         BASE_BLOCK_SIZE_LEN_BACKWARD / 8,
                                         BASE_BLOCK_SIZE_LEN_BACKWARD / 8, 1);
            PIPE_BARRIER(V);

            // exp(s-L)
            exp_v<ArchType::ASCEND_V220>(
                temp0BufPingpongFP32,
                temp0BufPingpongFP32,
                MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2, 1, 1, 8, 8);
            PIPE_BARRIER(V);

            if constexpr(IF_BF16) {
                // fp32 exp(s-L)--fp16 exp(s-L)
                convr_v<ArchType::ASCEND_V220>(temp0BufPingpongFP16, temp0BufPingpongFP32,
                                              MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2, 1, 1, 4, 8);
            }
            else {
                // fp32 exp(s-L)--fp16 exp(s-L)
                conv_v<ArchType::ASCEND_V220>(temp0BufPingpongFP16, temp0BufPingpongFP32,
                                              MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2, 1, 1, 4, 8);
            }

            SET_FLAG(V, MTE3, eventId);
            WAIT_FLAG(V, MTE3, eventId);

            // exp(s-L)结果拷回s的原位gm上
            AscendC::DataCopy(sGmGlobalTensor[sTotalOffset + ssOffset],
                              temp0BufPingpongFP16.template ReinterpretCast<WORKSPACE_DTYPE>(),
                              AscendC::DataCopyParams(1,
                                                      MAX_LENG_PER_UB_PROC_BACKWARD * 2 / 32,
                                                      0,
                                                      0));
            SET_FLAG(MTE3, MTE2, eventId);
            pingFlag = 1 - pingFlag;
        }

        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        SET_FLAG(MTE3, MTE2, EVENT_ID0);

        for (int m = 0; m < 2; m++) {
            // vector之间的偏移
            int32_t dp_offset = 64 * 128 * get_subblockid() + k * 128 * 128 + m * MAX_LENG_PER_UB_PROC_BACKWARD;
            int32_t p_offset = 64 * 128 * get_subblockid() + k * 128 * 128 +
                        m * MAX_LENG_PER_UB_PROC_BACKWARD/2;
            auto eventId = pingFlag ? EVENT_ID0:EVENT_ID1;
            WAIT_FLAG(MTE3, MTE2, eventId);
            AscendC::LocalTensor<INPUT_T> temp0BufPingpongFP16 = pingFlag ? temp0BufFp16Ping : temp0BufFp16;
            AscendC::LocalTensor<INPUT_T> temp1BufPingpongFP16 = pingFlag ? temp1BufFp16Ping : temp1BufFp16;
            AscendC::LocalTensor<float> temp0BufPingpongFP32 = pingFlag ? temp0BufFp32Ping : temp0BufFp32;
            AscendC::LocalTensor<float> temp1BufPingpongFP32 = pingFlag ? temp1BufFp32Ping : temp1BufFp32;

            // copy dp to ub
            AscendC::DataCopy(temp1BufPingpongFP32,
                              dpGmGlobalTensor[sTotalOffset + dp_offset],
                              AscendC::DataCopyParams(1,
                                                      MAX_LENG_PER_UB_PROC_BACKWARD * 4 / 32,
                                                      0,
                                                      0));
            SET_FLAG(MTE2, V, eventId);
            WAIT_FLAG(MTE2, V, eventId);

            // (dP - D)manm
            auto rowsumOffset = pingFlag ? 32*8 : 0;
            sub_v<ArchType::ASCEND_V220>(temp1BufPingpongFP32,
                                         temp1BufPingpongFP32,
                                         bufForBrcbRowsumFp32Buf[256*m],
                                         MAX_BLOCK_PER_ONE_PROC_BACKWARD, 1, 1, 0, 16, 16, 1);

            PIPE_BARRIER(V);
            sub_v<ArchType::ASCEND_V220>(temp1BufPingpongFP32[64],
                                         temp1BufPingpongFP32[64],
                                         bufForBrcbRowsumFp32Buf[rowsumOffset],
                                         MAX_BLOCK_PER_ONE_PROC_BACKWARD, 1, 1, 0, 16, 16, 1);

            PIPE_BARRIER(V);

            // P*(dP - D)
            mul_v<ArchType::ASCEND_V220>(
                temp1BufPingpongFP32,
                temp0BufPingpongFP32,
                temp1BufPingpongFP32,
                MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2, 1, 1, 1, 8, 8, 8);

            PIPE_BARRIER(V);

            if constexpr(IF_BF16) {
                // fp32 dS--fp16 dS  //pingpong拷出和addr拷出的区别!
                convr_v<ArchType::ASCEND_V220>(temp1BufPingpongFP16, temp1BufPingpongFP32,
                                              MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2, 1, 1, 4, 8);
                PIPE_BARRIER(V);
            }
            else {
                // fp32 dS--fp16 dS  //pingpong拷出和addr拷出的区别！
                PIPE_BARRIER(V);
                conv_v<ArchType::ASCEND_V220>(temp1BufPingpongFP16, temp1BufPingpongFP32,
                                              MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2, 1, 1, 4, 8);
            }

            SET_FLAG(V, MTE3, eventId);
            WAIT_FLAG(V, MTE3, eventId);

            // ds结果拷回gm dp
            AscendC::DataCopy(dpGmGlobalTensor[sTotalOffset + p_offset],
                              temp1BufPingpongFP16.template ReinterpretCast<WORKSPACE_DTYPE_DP>(),
                              AscendC::DataCopyParams(1,
                                                      MAX_LENG_PER_UB_PROC_BACKWARD * 2 / 32,
                                                      0,
                                                      0));
            SET_FLAG(MTE3, MTE2, eventId);
            pingFlag = 1 - pingFlag;
        }
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}
}
#endif

#endif