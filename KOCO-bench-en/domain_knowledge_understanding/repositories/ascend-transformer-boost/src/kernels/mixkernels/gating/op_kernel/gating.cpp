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
#include "mixkernels/utils/common/kernel/kernel_utils.h"
#include "mixkernels/gating/tiling/tiling_data.h"

using namespace AscendC;

constexpr int32_t INT32_SIZE = sizeof(int32_t);
constexpr int32_t INT64_SIZE = sizeof(int64_t);
constexpr int32_t BUFFER_NUM = 1;
constexpr int64_t DOUBLE = 2;
constexpr int64_t TRIPLE = 3; // Index of sorting queue
constexpr int64_t MAX_SORT_QUEUE_NUM = 4;
constexpr int64_t SORT_STRUCT_MULTIPLE = 8; // Sorting structure size is calculated as 8 times of tile size
constexpr int32_t SYNC_UB_BYTES = 32 * 32;
constexpr int64_t TILE_NUM = 512;
constexpr int64_t STRUCT_TILE_NUM = 8 * TILE_NUM;
constexpr int32_t TOPK_PROPOSAL_IDX = 4;
constexpr int32_t IDX_PROPOSAL_IDX = 5;

template <typename CumSumNumType>
class Gating {
public:
    __aicore__ inline Gating() {}

    __aicore__ inline void Init(GM_ADDR topk, GM_ADDR idxArr,
                                GM_ADDR tokenIndex, GM_ADDR CumSum,
                                GM_ADDR originalIndex, GM_ADDR globalSortWorkspace,
                                GM_ADDR cumSumWorkspace, GM_ADDR syncWorkspace,
                                AtbOps::GatingTilingData *tiling_data)
    {
        InitParams(tiling_data);
        // 搬运的数据对齐block时候补齐的数字(一个block 32bytes 对应8个float)
        topkGm.SetGlobalBuffer((__gm__ int32_t *)topk, topKNum);
        idxArrGm.SetGlobalBuffer((__gm__ int32_t *)idxArr, topKNum);
        tokenIndexGm.SetGlobalBuffer((__gm__ int32_t *)tokenIndex, topKNum);
        cumSumGm.SetGlobalBuffer((__gm__ CumSumNumType *)CumSum, cumSumNum);
        globalSortBlock.SetGlobalBuffer((__gm__ float *)globalSortWorkspace, SORT_STRUCT_MULTIPLE * topKNumPadded);
        globalSortBlock2.SetGlobalBuffer((__gm__ float *)globalSortWorkspace + SORT_STRUCT_MULTIPLE * topKNumPadded,
                                          SORT_STRUCT_MULTIPLE * topKNumPadded);
        cumSumBlock.SetGlobalBuffer((__gm__ int32_t *)cumSumWorkspace, actualCoreNum * cumSumNum32BytesPadded);
        syncGm.SetGlobalBuffer((__gm__ int32_t *)syncWorkspace, syncSize);

        originalGm.SetGlobalBuffer((__gm__ int32_t *)originalIndex, topKNum);

        InitPipe();
    }

    __aicore__ inline void Process()
    {
        if (blockIdx == actualCoreNum) {  // 尾核做cumSum聚合计算，需要等待其他核对cumSum局部计算完成
            if (cumSumNum > 0) {
                // 等待其他核局部计算cumSum完成
                for (int32_t i = 0; i < actualCoreNum; ++i) {
                    auto sync_buf = syncTQue.AllocTensor<int32_t>();
                    IBWait(syncGm, sync_buf, i, actualCoreNum);
                    syncTQue.FreeTensor(sync_buf);
                }
                // 全局计算cumsum
                calculateAndCopy2CumSumGm();
            }
        } else if (blockIdx > 0) {  // 非0核做局部计算：局部排序 + cumSum局部计算
            // 单核局部排序
            PartSort();
            // 通知0核，当前核局部排序完成
            auto sync_buf = syncTQue.AllocTensor<int32_t>();
            IBSet(syncGm, sync_buf, blockIdx, 0);
            syncTQue.FreeTensor(sync_buf);
            if (cumSumNum > 0) {
                // 单核计算cumSum
                PartCumSum();
                // 通知尾核，当前核计算cumSum完成
                auto sync_buf2 = syncTQue.AllocTensor<int32_t>();
                IBSet(syncGm, sync_buf2, blockIdx, actualCoreNum);
                syncTQue.FreeTensor(sync_buf2);
            }
        } else {  // 0核做全局排序
            // 单核局部排序
            PartSort();
            if (cumSumNum > 0) {
                // 单核计算cumSum
                PartCumSum();
                // 通知尾核，当前核计算cumSum完成
                auto sync_buf2 = syncTQue.AllocTensor<int32_t>();
                IBSet(syncGm, sync_buf2, blockIdx, actualCoreNum);
                syncTQue.FreeTensor(sync_buf2);
            }
            // 等待其他核（除尾核）局部排序完成，如果实际核数为1，则无需等待其他核
            if (actualCoreNum > 1) {
                for (int32_t i = 1; i < actualCoreNum; ++i) {
                    auto sync_buf = syncTQue.AllocTensor<int32_t>();
                    IBWait(syncGm, sync_buf, i, 0);
                    syncTQue.FreeTensor(sync_buf);
                }
            }
            GlobalTensor<float> resultGlobalSortBlock = GlobalSort();
            // 输出original_index，把数据从globalSortBlock拷贝到tokenIndexGm
            CopyGm2Gm(resultGlobalSortBlock, originalGm);
        }
    }

private:

    // 全局排序的参数结构体
    struct GmsParams {
        // 每一个队列需要排序的长度
        int (&gmsLengths)[MAX_SORT_QUEUE_NUM];
        // 每一个队列的排序初始元素在workspace上的起始位置
        int (&gmsCurrentHead)[MAX_SORT_QUEUE_NUM];
        // 队列的总长度(<=4)
        int &queueNum;
        // ub上数据加载的tensor（input，会切成四部分使用）
        LocalTensor<float> &srcLocalTensor;
        // ub上排序的结果（output）
        LocalTensor<float> &dstLocalTensor;
        // 数据在gm上存放的位置
        bool &gmTensorIndex;
        // Two possible position for data in GM
        GlobalTensor<float> (&buffLocal)[DOUBLE];
    };

    __aicore__ inline void InitPipe()
    {
        pipe.InitBuffer(syncTQue, BUFFER_NUM, SYNC_UB_BYTES);
        pipe.InitBuffer(inQueueTopK, BUFFER_NUM, TILE_NUM * INT32_SIZE);
        pipe.InitBuffer(inQueueIdxArr, BUFFER_NUM, TILE_NUM * INT32_SIZE);
        pipe.InitBuffer(inQueueWorkspace, BUFFER_NUM, MAX_SORT_QUEUE_NUM * STRUCT_TILE_NUM * INT32_SIZE);
        pipe.InitBuffer(outQueueWorkspace, BUFFER_NUM, DOUBLE * STRUCT_TILE_NUM * INT32_SIZE);
        pipe.InitBuffer(outQueueTopK, BUFFER_NUM, STRUCT_TILE_NUM * INT32_SIZE);

        // ub需要32字节对齐，所以分配cumSumNum32BytesPadded大小空间，实际有效数据个数为cumSumNum
        pipe.InitBuffer(inQueueCumsumPart, BUFFER_NUM, cumSumNum32BytesPadded * INT32_SIZE);
        pipe.InitBuffer(outQueueCumsumPartAddSrc0, BUFFER_NUM, cumSumNum32BytesPadded * INT32_SIZE);
        pipe.InitBuffer(outQueueCumsumPartAddSrc1, BUFFER_NUM, cumSumNum32BytesPadded * INT32_SIZE);
        // TBuf
        pipe.InitBuffer(tileNumTempBuf, TILE_NUM * INT32_SIZE);
        pipe.InitBuffer(structTileNumTempBuf, STRUCT_TILE_NUM * INT32_SIZE);
        pipe.InitBuffer(expertNumTempBuf, cumSumNum32BytesPadded * sizeof(CumSumNumType));
        pipe.InitBuffer(CopyUb2GmPadtemp, SORT_STRUCT_MULTIPLE * INT32_SIZE);
    }

    __aicore__ inline void InitParams(AtbOps::GatingTilingData *tiling_data)
    {
        blockIdx = GetBlockIdx();
        topkExpertNum = tiling_data->topkExpertNum;
        topKNum = tiling_data->topKNum;
        cumSumNum = tiling_data->cumSumNum;
        cumSumNum32BytesPadded = tiling_data->cumSumNum32BytesPadded;
        actualCoreNum = tiling_data->actualCoreNum;
        tailBlockDataSize = tiling_data->tailBlockDataSize;
        syncSize = tiling_data->syncSize;
        blockNumPerCore = tiling_data->blockNumPerCore[blockIdx];
        offSet = tiling_data->offSetPerCore[blockIdx];
        topKNumPadded = tiling_data->topKNumPadded;
    }

    // 单核局部排序
    __aicore__ inline void PartSort()
    {
        // 当前核处理数据块数量
        int32_t executeTimes = blockNumPerCore;
        // 尾块有效数据长度
        int32_t tailNum = blockIdx == (actualCoreNum - 1) ? tailBlockDataSize : TILE_NUM;

        // 单核排序，处理当前核下的每个tile
        for (uint32_t i = 0; i < executeTimes; i++) {
            // processNum 该次处理块的有效数据长度，如果是最后一个tiling，processNum可能小于TILE_NUM，需要填充
            uint32_t processNum = i == executeTimes - 1 ? tailNum : TILE_NUM;
            CopyIn(i, processNum);
            Compute(i, processNum);
            CopyOut(i, processNum);
        }
    }

    // 单核cumSum局部计算
    __aicore__ inline void PartCumSum()
    {
        // 单核计算局部cum_sum
        if (cumSumNum > 0) {
            ComputeCumSumPart();
        }
    }

    __aicore__ inline GlobalTensor<float> GlobalSort()
    {
        // Mark the globalSortBlock in use
        bool switchFlag = false;

        // 申请中间用于排序的tensor
        LocalTensor<float> srcLocalTensor = inQueueWorkspace.AllocTensor<float>();
        LocalTensor<float> dstLocalTensor = outQueueWorkspace.AllocTensor<float>();
        GlobalTensor<float> sortedGlobal[2] = {globalSortBlock, globalSortBlock2};

        // 目前已经有多少块tile已经有序
        int32_t orderBlock = topKNumPadded / TILE_NUM;
        // 每次把4个有序的队列合成一块
        int32_t globalSortBlockCount = MAX_SORT_QUEUE_NUM;

        int32_t length[MAX_SORT_QUEUE_NUM];
        int32_t currentHead[MAX_SORT_QUEUE_NUM];

        // blockSize代表每次排序多少块Tile
        for (int32_t blockSize = 1; blockSize < orderBlock; blockSize *= globalSortBlockCount) {
            for (int32_t tileIndex = 0; tileIndex < orderBlock; tileIndex += blockSize * globalSortBlockCount) {
                int32_t mrgTileNum = orderBlock - tileIndex < blockSize * globalSortBlockCount ?
                                    (orderBlock - tileIndex) : (blockSize * globalSortBlockCount);
                int32_t queueNum = (mrgTileNum + blockSize - 1) / blockSize;
                uint16_t lastQueTileNum = mrgTileNum % blockSize == 0 ? blockSize : mrgTileNum % blockSize;
                for (int i = 0; i < queueNum; i++) {
                    currentHead[i] = TILE_NUM * (tileIndex + i * blockSize);
                }
                for (int i = 0; i < queueNum - 1; i++) {
                    length[i] = TILE_NUM * blockSize;
                }
                length[queueNum-1] = TILE_NUM * lastQueTileNum;

                // 构建GMSParams
                GmsParams params{length, currentHead, queueNum,
                                 srcLocalTensor, dstLocalTensor, switchFlag, sortedGlobal};
                GlobalMrgSort(params);
                PipeBarrier<PIPE_V>();
            }
            switchFlag = !switchFlag;
        }

        inQueueWorkspace.FreeTensor(srcLocalTensor);
        outQueueWorkspace.FreeTensor(dstLocalTensor);

        return switchFlag ? globalSortBlock2 : globalSortBlock;
    }

    struct MrgQueue {
        int32_t queueNum[MAX_SORT_QUEUE_NUM]; // 每个队列还需要排序的数据有多长
        int32_t queueLength; // 有效队列的数量
        int32_t currentHead[MAX_SORT_QUEUE_NUM]; // 每个队列里面要排序数的起始位置
        int32_t totalMrgLen; // 已经有多少个数完成排序了
        int32_t originalPosition; // 这次排序的数据在gm上的起始位置
    };

    __aicore__ inline void GlobalMrgSort(GmsParams &params)
    {
        // 得到tensor的指针
        LocalTensor<float> srcLocalTensor {params.srcLocalTensor};
        LocalTensor<float> dstLocalTensor {params.dstLocalTensor};
        GlobalTensor<float> srcGmTensor = params.buffLocal[params.gmTensorIndex];
        GlobalTensor<float> dstGmTensor = params.buffLocal[!params.gmTensorIndex];

        MrgQueue queueToMrg{
            {params.gmsLengths[0], params.gmsLengths[1], params.gmsLengths[2], params.gmsLengths[3]},
            params.queueNum,
            {params.gmsCurrentHead[0], params.gmsCurrentHead[1], params.gmsCurrentHead[2], params.gmsCurrentHead[3]},
            0,
            params.gmsCurrentHead[0]
        };

        if (queueToMrg.queueLength == 1) {
            HandleSingleQueue(queueToMrg, dstLocalTensor, srcGmTensor, dstGmTensor);
            return;
        }

        while (queueToMrg.queueLength > 1) {
            PerformMergeSortStep(queueToMrg, srcLocalTensor, dstLocalTensor, srcGmTensor, dstGmTensor);
            PipeBarrier<PIPE_ALL>();
        }

        // 处理尾部数据
        // 最后只会有一个队列还剩余值，因而直接拷贝queueNum[0]即可
        if (queueToMrg.queueNum[0] > 0) {
            HandleSingleQueue(queueToMrg, dstLocalTensor, srcGmTensor, dstGmTensor);
        }
    }

    __aicore__ inline void HandleSingleQueue(MrgQueue &queueToMrg, LocalTensor<float> &dstLocalTensor,
                                             GlobalTensor<float> &srcGmTensor, GlobalTensor<float> &dstGmTensor)
    {
        int32_t repeatTimes = (queueToMrg.queueNum[0] + TILE_NUM - 1) / TILE_NUM;
        int32_t tailNum = queueToMrg.queueNum[0] % TILE_NUM == 0 ? TILE_NUM : queueToMrg.queueNum[0] % TILE_NUM;
        for (int i = 0; i < repeatTimes; i++) {
            int32_t executeNum = i == repeatTimes - 1 ? tailNum : TILE_NUM;

            DataCopy(dstLocalTensor,
                     srcGmTensor[SORT_STRUCT_MULTIPLE * queueToMrg.currentHead[0] +
                                 SORT_STRUCT_MULTIPLE * i * TILE_NUM],
                     static_cast<uint32_t>(SORT_STRUCT_MULTIPLE * executeNum));
            PipeBarrier<PIPE_ALL>();
            DataCopy(dstGmTensor[SORT_STRUCT_MULTIPLE * queueToMrg.originalPosition +
                                 SORT_STRUCT_MULTIPLE * queueToMrg.totalMrgLen +
                                 SORT_STRUCT_MULTIPLE * i * TILE_NUM],
                     dstLocalTensor,
                     static_cast<uint32_t>(SORT_STRUCT_MULTIPLE * executeNum));
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void PerformMergeSortStep(MrgQueue &queueToMrg, LocalTensor<float> &srcLocalTensor,
                                                LocalTensor<float> &dstLocalTensor, GlobalTensor<float> &srcGmTensor,
                                                GlobalTensor<float> &dstGmTensor)
    {
        // 本轮排序实际排序的数量
        uint16_t tmpSortLen[MAX_SORT_QUEUE_NUM];

        // 把srcLocalTensor切成4份，每份的起始地址都是i *  TILE_NUM * SORT_STRUCT_MULTIPLE
        for (int i = 0; i < queueToMrg.queueLength; i++) {
            uint16_t sortLength = queueToMrg.queueNum[i] < maxSortLengthArr[i] ? queueToMrg.queueNum[i] :
                                                                                 maxSortLengthArr[i];
            tmpSortLen[i] = sortLength;
            int32_t gmStartPosition = queueToMrg.currentHead[i];
            DataCopy(srcLocalTensor[i * STRUCT_TILE_NUM], srcGmTensor[SORT_STRUCT_MULTIPLE * gmStartPosition],
                     static_cast<uint32_t>(SORT_STRUCT_MULTIPLE * sortLength));
        }

        // 排序
        MrgSort4Info mrgParams(tmpSortLen, true, validQueueArr[queueToMrg.queueLength], 1);

        MrgSortSrcList<float> srcList(srcLocalTensor[0], srcLocalTensor[STRUCT_TILE_NUM],
                                      srcLocalTensor[DOUBLE * STRUCT_TILE_NUM],
                                      srcLocalTensor[TRIPLE * STRUCT_TILE_NUM]);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        MrgSort4<float>(dstLocalTensor, srcList, mrgParams);
        PipeBarrier<PIPE_ALL>();

        // 排序排了多少数
        uint16_t sortedLen[MAX_SORT_QUEUE_NUM];
        GetMrgSortResult(sortedLen[0], sortedLen[1], sortedLen[2], sortedLen[3]);  // 2 3 为队列下标
        const uint16_t localMrgLen = sortedLen[0] + sortedLen[1] + sortedLen[2] + sortedLen[3];  // 2 3 为队列下标

        // 将排序的结果拷贝到dstGm上
        DataCopy(dstGmTensor[SORT_STRUCT_MULTIPLE * queueToMrg.originalPosition +
                             SORT_STRUCT_MULTIPLE * queueToMrg.totalMrgLen],
                 dstLocalTensor, static_cast<uint32_t>(SORT_STRUCT_MULTIPLE * localMrgLen));

        queueToMrg.totalMrgLen += localMrgLen;

        // 根据此次排序的结果更新排序的数据
        for (int i = 0; i < queueToMrg.queueLength; i++) {
            queueToMrg.queueNum[i] -= sortedLen[i];
            queueToMrg.currentHead[i] += sortedLen[i];
        }

        for (int i = 0; i < queueToMrg.queueLength; i++) {
            if (queueToMrg.queueNum[i] == 0) {
                for (int j = i; j < 3; j++) {  // 3 Sort排序队列为4个
                    queueToMrg.queueNum[j] = queueToMrg.queueNum[j + 1];
                    queueToMrg.currentHead[j] = queueToMrg.currentHead[j + 1];
                }
                queueToMrg.queueNum[queueToMrg.queueLength - 1] = 0;
                queueToMrg.queueLength -= 1;
                break;
            }
        }
    }

    __aicore__ inline void CopyIn(uint32_t processIndex, uint32_t processNum)
    {
        // paddingNum  当前tiling，为了可以切分为完整block，需要补充的int数
        uint32_t paddingNum = (processNum * INT32_SIZE) % 32 == 0 ?
                               0 : (32 - (processNum * INT32_SIZE) % 32) / INT32_SIZE;
        LocalTensor<int32_t> topkLocal = inQueueTopK.AllocTensor<int32_t>();
        LocalTensor<int32_t> idxArrLocal = inQueueIdxArr.AllocTensor<int32_t>();
        DataCopy(topkLocal, topkGm[offSet + processIndex * TILE_NUM], processNum + paddingNum);
        DataCopy(idxArrLocal, idxArrGm[offSet + processIndex * TILE_NUM], processNum + paddingNum);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        if (paddingNum != 0) {
            uint64_t bit_mask = 1;
            const uint32_t countPreBlock = 8;
            const uint32_t dstRepeatStride = (processNum + paddingNum) / countPreBlock - 1;
            bit_mask <<= paddingNum;
            bit_mask -= 1;
            bit_mask <<= countPreBlock - paddingNum;
            uint64_t mask[2]{ bit_mask, 0 };
            Duplicate(topkLocal[dstRepeatStride * countPreBlock], paddingValueInt, mask, 1, 1, 1);
            Duplicate(idxArrLocal[dstRepeatStride * countPreBlock], paddingValueInt, mask, 1, 1, 1);
        }
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        inQueueTopK.EnQue(topkLocal);
        inQueueIdxArr.EnQue(idxArrLocal);
    }

    __aicore__ inline void Compute(uint32_t processIndex, uint32_t processNum)
    {
        const uint32_t paddingNum = (processNum * INT32_SIZE) % 32 == 0 ?
            0 : (32 - (processNum * INT32_SIZE) % 32) / INT32_SIZE;

        // 数据出队，topkLocalInt、idxArrLocalInt中均一共有processNum+paddingNum个数，已满足32bytes对齐
        LocalTensor<int32_t> topkLocalInt = inQueueTopK.DeQue<int32_t>();
        LocalTensor<int32_t> idxArrLocalInt = inQueueIdxArr.DeQue<int32_t>();
        // 因为idxArrLocal不参与计算,且ProposalConcat要求类型一致
        // 所以直接按照二进制解释为浮点数，最后直接重新解释为整数即可
        LocalTensor<float> idxArrLocal = idxArrLocalInt.ReinterpretCast<float>();

        // 计算bincount
        if (cumSumNum > 0) {
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            for (uint32_t i = 0; i < processNum; i++) {
                uint32_t expertIndex = topkLocalInt.GetValue(i);
                selectedExpertCount[expertIndex]++;
            }
        }
        LocalTensor<float> groupSortLocal = structTileNumTempBuf.Get<float>();
        LocalTensor<float> topkLocal_float = tileNumTempBuf.Get<float>();
        Cast(topkLocal_float, topkLocalInt, RoundMode::CAST_NONE, processNum + paddingNum);
        
        // 把 topkLocal_float 进行填充，长度达到 TILE_NUM
        Duplicate<float>(topkLocal_float[processNum + paddingNum], paddingValueFloat,
            TILE_NUM - (processNum + paddingNum));
        Duplicate<float>(idxArrLocal[processNum + paddingNum], paddingValueFloat,
            TILE_NUM - (processNum + paddingNum));
        // 乘以-1实现降序排列
        float factor = -1.0;
        Muls(topkLocal_float, topkLocal_float, factor, TILE_NUM);
        PipeBarrier<PIPE_V>();

        // 构建 Proposal
        LocalTensor<float> sortLocal = outQueueWorkspace.AllocTensor<float>();
        uint32_t repeatTimes = (TILE_NUM) / 16;
        ProposalConcat<float>(sortLocal, topkLocal_float, repeatTimes, TOPK_PROPOSAL_IDX);
        ProposalConcat<float>(sortLocal, idxArrLocal, repeatTimes, IDX_PROPOSAL_IDX);
        PipeBarrier<PIPE_V>();

        // 单路排序
        RpSort16(groupSortLocal, sortLocal, repeatTimes);
        PipeBarrier<PIPE_V>();

        // 多路归并
        MergeSort4Queue(sortLocal, groupSortLocal);

        // 结果入队
        outQueueWorkspace.EnQue<float>(sortLocal);
        // 释放本地local
        inQueueTopK.FreeTensor(topkLocalInt);
        inQueueIdxArr.FreeTensor(idxArrLocal);
    }

    __aicore__ inline void CopyOut(uint32_t processIndex, uint32_t processNum)
    {
        // 结果出队
        LocalTensor<float> sortLocal = outQueueWorkspace.DeQue<float>();

        // 拷贝到gm
        DataCopy(globalSortBlock[offSet * SORT_STRUCT_MULTIPLE + processIndex * STRUCT_TILE_NUM], sortLocal,
                 STRUCT_TILE_NUM);

        // 释放本地local
        outQueueWorkspace.FreeTensor(sortLocal);
    }

    __aicore__ inline void CopyGm2Gm(const GlobalTensor<float> &srcGlobal, const GlobalTensor<int32_t> &dstGlobal)
    {
        const uint32_t copyTimes = (topKNum / TILE_NUM) + (topKNum % TILE_NUM == 0 ? 0 : 1);
        LocalTensor<float> tmpLocal = structTileNumTempBuf.Get<float>();
        LocalTensor<float> originIndex = outQueueTopK.AllocTensor<float>();
        LocalTensor<float> divFloatLocal = tileNumTempBuf.Get<float>();
        float topkExpertNumFloat = topkExpertNum;
        Duplicate(divFloatLocal, topkExpertNumFloat, TILE_NUM);

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        for (int i = 0; i < copyTimes; i++) {
            // 利用tmpLocal做中转实现gm2gm的拷贝
            DataCopy(tmpLocal, srcGlobal[i * STRUCT_TILE_NUM], STRUCT_TILE_NUM);
            PipeBarrier<PIPE_ALL>();
            ProposalExtract(originIndex, tmpLocal, TILE_NUM / (DOUBLE * SORT_STRUCT_MULTIPLE), IDX_PROPOSAL_IDX);

            LocalTensor<int32_t> originIndexLocalInt = originIndex.ReinterpretCast<int32_t>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

            if (i == copyTimes - 1) {
                uint32_t tailNum = topKNum % TILE_NUM == 0 ? TILE_NUM : topKNum % TILE_NUM;
                CopyUb2GmPad(dstGlobal[i * TILE_NUM], originIndexLocalInt, tailNum);
            } else {
                DataCopy(dstGlobal[i * TILE_NUM], originIndexLocalInt, TILE_NUM);
            }
            PipeBarrier<PIPE_ALL>();

            if (cumSumNum > 0) {
                LocalTensor<float> originIndexLocalFloat = originIndexLocalInt.ReinterpretCast<float>();
                Cast(originIndexLocalFloat, originIndexLocalInt, RoundMode::CAST_NONE, TILE_NUM);
                PipeBarrier<PIPE_V>();
                Div(originIndexLocalFloat, originIndexLocalFloat, divFloatLocal, TILE_NUM);
                PipeBarrier<PIPE_V>();
                Cast(originIndexLocalInt, originIndexLocalFloat, RoundMode::CAST_FLOOR, TILE_NUM);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

                if (i == copyTimes - 1) {
                    uint32_t tailNum = topKNum % TILE_NUM == 0 ? TILE_NUM : topKNum % TILE_NUM;
                    CopyUb2GmPad(tokenIndexGm[i * TILE_NUM], originIndexLocalInt, tailNum);
                } else {
                    DataCopy(tokenIndexGm[i * TILE_NUM], originIndexLocalInt, TILE_NUM);
                }
            }
        }

        outQueueTopK.FreeTensor(originIndex);
    }

    __aicore__ inline void MergeSort4Queue(LocalTensor<float> &sortBuf, LocalTensor<float> &tmpBuf)
    {
        const uint16_t mergeCount = MAX_SORT_QUEUE_NUM;
        LocalTensor<float> sortedQue[2] = {tmpBuf, sortBuf};
        int switchFlag = 0;
        const uint16_t proposalSize = SORT_STRUCT_MULTIPLE; // 单个 proposal 的32位元素个数 8
        uint16_t singleQueSize = 16; // 每条队列的 proposal 数量 每次乘 4
        while (singleQueSize < TILE_NUM) {
            uint16_t QueNum = (TILE_NUM / singleQueSize) % MAX_SORT_QUEUE_NUM;
            uint16_t repeatTimes = TILE_NUM / singleQueSize / MAX_SORT_QUEUE_NUM;
            if (QueNum != 0) {
                repeatTimes++;
            }
            uint16_t validBit = (1 << (MAX_SORT_QUEUE_NUM - QueNum)) - 1;
            struct MrgSortSrcList<float> srcList{
                sortedQue[switchFlag][0],
                sortedQue[switchFlag][singleQueSize * proposalSize],
                sortedQue[switchFlag][singleQueSize * proposalSize * DOUBLE],
                sortedQue[switchFlag][singleQueSize * proposalSize * TRIPLE]
            };
            uint16_t elementLengths[MAX_SORT_QUEUE_NUM]{
                singleQueSize,
                singleQueSize,
                singleQueSize,
                singleQueSize
            };
            struct MrgSort4Info srcInfo(elementLengths, false, validBit, repeatTimes);
            MrgSort4(sortedQue[!switchFlag], srcList, srcInfo);
            switchFlag = !switchFlag;
            singleQueSize *= mergeCount;
        }

        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if (!switchFlag) {
            DataCopy(sortBuf, tmpBuf, STRUCT_TILE_NUM);
        }
    }

    __aicore__ inline void ComputeCumSumPart()
    {
        LocalTensor<int32_t> cumSumPartLocalTensor = inQueueCumsumPart.AllocTensor<int32_t>();
        for (int i = 0; i < cumSumNum; i++) {
            cumSumPartLocalTensor.SetValue(i, selectedExpertCount[i]);
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        CopyUb2GmPad(cumSumBlock[GetBlockIdx() * cumSumNum32BytesPadded], cumSumPartLocalTensor, cumSumNum);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        inQueueCumsumPart.FreeTensor(cumSumPartLocalTensor);
    }

    __aicore__ inline void calculateAndCopy2CumSumGm()
    {
        LocalTensor<int32_t> accumulator = inQueueCumsumPart.AllocTensor<int32_t>();
        LocalTensor<int32_t> src0 = outQueueCumsumPartAddSrc0.AllocTensor<int32_t>();
        LocalTensor<int32_t> src1 = outQueueCumsumPartAddSrc1.AllocTensor<int32_t>();

        DataCopy(accumulator, cumSumBlock, cumSumNum32BytesPadded);

        for (int i = 1; i < actualCoreNum; i++) {
            // src0
            DataCopy(src0, cumSumBlock[i * cumSumNum32BytesPadded], cumSumNum32BytesPadded);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            // src1
            // DataCopyPad不支持localTensor->localTensor，已32字节对齐，因此可用DataCopy
            DataCopy(src1, accumulator, cumSumNum32BytesPadded);
            PipeBarrier<PIPE_ALL>();
            // add
            Add(accumulator, src0, src1, cumSumNum);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }

        // 将聚合计算binCount结果累加并搬运到cumSumGm输出
        PipeBarrier<PIPE_ALL>();
        LocalTensor<CumSumNumType> cumSumLocalTensor = expertNumTempBuf.Get<CumSumNumType>();
        CumSumNumType acc = 0;
        for (int i = 0; i < cumSumNum; i++) {
            acc = acc + static_cast<CumSumNumType>(accumulator.GetValue(i));
            cumSumLocalTensor.SetValue(i, acc);
        }
        PipeBarrier<PIPE_ALL>();
        CopyUb2GmPad(cumSumGm, cumSumLocalTensor, cumSumNum);

        inQueueCumsumPart.FreeTensor(accumulator);

        outQueueCumsumPartAddSrc0.FreeTensor(src0);
        outQueueCumsumPartAddSrc1.FreeTensor(src1);
    }

    template <typename T>
    __aicore__ inline void CopyUb2GmPad(const GlobalTensor<T> &dstGlobal, const LocalTensor<T> &srcLocal,
                                        uint32_t length)
    {
        LocalTensor<T> tempLocal = CopyUb2GmPadtemp.Get<T>();
        uint32_t countPreBlock = 8;
        if (length <= countPreBlock) {
            DataCopy(dstGlobal, srcLocal, countPreBlock);
        } else {
            DataCopy(dstGlobal, srcLocal, length);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            if (length % countPreBlock != 0) {
                for (T i = 0; i < countPreBlock; ++i) {
                    T t = srcLocal.GetValue(length - countPreBlock + i);
                    tempLocal.SetValue(i, t);
                }
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(dstGlobal[length - countPreBlock], tempLocal, countPreBlock);
            }
        }
    }

private:
    TPipe pipe;
    GlobalTensor<int32_t> topkGm;
    GlobalTensor<int32_t> idxArrGm;
    GlobalTensor<int32_t> tokenIndexGm;
    GlobalTensor<int32_t> originalGm;
    GlobalTensor<CumSumNumType> cumSumGm;
    GlobalTensor<float> globalSortBlock; // workspace
    GlobalTensor<float> globalSortBlock2; // workspace2
    GlobalTensor<int32_t> cumSumBlock; // workspace
    GlobalTensor<int32_t> syncGm; // workspace

    TQue<QuePosition::VECIN, BUFFER_NUM> syncTQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueTopK, inQueueIdxArr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueWorkspace;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCumsumPart;

    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueWorkspace;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueTopK;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueCumsumPartAddSrc0;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueCumsumPartAddSrc1;

    TBuf<AscendC::TPosition::VECCALC> tileNumTempBuf;
    TBuf<AscendC::TPosition::VECCALC> structTileNumTempBuf;
    TBuf<AscendC::TPosition::VECCALC> expertNumTempBuf;
    TBuf<AscendC::TPosition::VECCALC> CopyUb2GmPadtemp;

    float paddingValueFloat = static_cast<float>(0x0FFFFFFF);
    int32_t paddingValueInt = 0x0FFFFFFF;
    uint32_t paddingValueUint = 0x0FFFFFFF;
    int32_t topkExpertNum = 0;
    int64_t topKNum = -1;
    int32_t cumSumNum = -1;
    int32_t cumSumNum32BytesPadded = -1;
    int32_t actualCoreNum = 1;
    int32_t tailBlockDataSize = 0;
    int32_t syncSize = 0;
    int32_t blockNumPerCore = 0;
    uint32_t offSet = 0;
    int64_t topKNumPadded = 0;
    int32_t blockIdx = 0;
    // 每个专家被多少个token选中
    int32_t selectedExpertCount[1025] = {0};
    int maxSortLengthArr[MAX_SORT_QUEUE_NUM] = {512, 512, 512, 512};
    uint16_t validQueueArr[5] = {0, 0, 0b11, 0b111, 0b1111};
};

__aicore__ inline void InitGatingTilingData(const __gm__ uint8_t *tiling,
                                            AtbOps::GatingTilingData *tilingData)
{
    TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, tiling, sizeof(AtbOps::GatingTilingData), &pipe);
    __ubuf__ AtbOps::GatingTilingData *tilingDataPointer =
        reinterpret_cast<__ubuf__ AtbOps::GatingTilingData *>(tilingdata_in_ub);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    tilingData->topkExpertNum = tilingDataPointer->topkExpertNum;
    tilingData->topKNum = tilingDataPointer->topKNum;
    tilingData->topKNumPadded = tilingDataPointer->topKNumPadded;
    tilingData->cumSumNum = tilingDataPointer->cumSumNum;
    tilingData->cumSumNum32BytesPadded = tilingDataPointer->cumSumNum32BytesPadded;
    tilingData->actualCoreNum = tilingDataPointer->actualCoreNum;
    tilingData->blockNum = tilingDataPointer->blockNum;
    tilingData->tailBlockDataSize = tilingDataPointer->tailBlockDataSize;
    tilingData->syncSize = tilingDataPointer->syncSize;
    for (int i = 0; i < MAX_CORE_NUM; ++i) {
        tilingData->blockNumPerCore[i] = tilingDataPointer->blockNumPerCore[i];
        tilingData->beginBlockIndexPerCore[i] = tilingDataPointer->beginBlockIndexPerCore[i];
        tilingData->offSetPerCore[i] = tilingDataPointer->offSetPerCore[i];
    }
    tilingData->cumSumInt64 = tilingDataPointer->cumSumInt64;
}

// implementation of kernel function
extern "C" __global__ __aicore__ void gating(GM_ADDR topk, GM_ADDR idxArr,
                                             GM_ADDR tokenIndex, GM_ADDR cumSum,
                                             GM_ADDR originalIndex, GM_ADDR validIndex,
                                             GM_ADDR globalSortWorkspace, GM_ADDR cumSumWorkspace,
                                             GM_ADDR syncWorkspace, GM_ADDR tiling)
{
    AtbOps::GatingTilingData tilingData;
    InitGatingTilingData(tiling, &tilingData);
    if (TILING_KEY_IS(2000000000)) {
        Gating<int32_t> op;
        op.Init(topk, idxArr, tokenIndex, cumSum, originalIndex, globalSortWorkspace,
                cumSumWorkspace, syncWorkspace, &tilingData);
        op.Process();
    }
    if (TILING_KEY_IS(2000000001)) {
        Gating<int64_t> op;
        op.Init(topk, idxArr, tokenIndex, cumSum, originalIndex, globalSortWorkspace,
                cumSumWorkspace, syncWorkspace, &tilingData);
        op.Process();
    }
}