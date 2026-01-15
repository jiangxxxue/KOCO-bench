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
#include "mixkernels/gating/tiling/tiling_data.h"

using namespace AscendC;

constexpr int32_t TOKEN_NUM = 1024;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t FLOAT_DATA_SIZE = 4;
constexpr int32_t TAIL_COPY_FLAG = 2;
constexpr int64_t TILE_NUM = 2048;
constexpr int64_t DOUBLE_TILE_NUM = 2 * TILE_NUM;
constexpr int64_t DOUBLE = 2;
constexpr int64_t THREE = 3;
constexpr int64_t GATHER_MASK_BEFORE_TILE_COUNT = 4;
constexpr int64_t GATHER_MASK_AFTER_TILE_COUNT = 2;
constexpr int32_t INT32_SIZE = sizeof(int32_t);
constexpr int32_t BYTES32 = 32;
constexpr int32_t SYNC_UB_BYTES = 1 * 1024;
constexpr uint16_t GLOBAL_SORT_NUM = 2560;  // GLOBAL_SORT_UB_HALF_LENGTH / 8
constexpr uint16_t GLOBAL_SORT_UB_HALF_LENGTH = 20 * 1024;
// 每一次都是4路排序，所以需要乘以4
constexpr int32_t GLOBAL_SORT_SIZE = TILE_NUM * 4 * 2;
constexpr uint32_t DEVIC_EEXPERT_NUM_OFFSET = 9;
constexpr uint32_t CUM_SUM_NUM_OFFSET = 3;

struct CopyInfo {
    uint32_t copyTimes;
    uint32_t tailNum;
    int32_t startCopyLocation;
    int32_t completedCopy;
};

__aicore__ inline void InitGatingTilingData(const __gm__ uint8_t *tiling, AtbOps::GatingTilingData *tilingData)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)tilingData;
    // 提前获取deviceExpertNum 与 cumSumGm 决定是否为EP
    // 当不为EP时，EP相关参数不用加载
    int32_t deviceExpertNum = *(src + DEVIC_EEXPERT_NUM_OFFSET);
    int32_t cumSumGm = *(src + CUM_SUM_NUM_OFFSET);
    if (deviceExpertNum == cumSumGm) {
        deviceExpertNum = 0;
    }
    for (auto i = 0; i < sizeof(AtbOps::GatingTilingData) / sizeof(int32_t) - 1025 + deviceExpertNum; i++) {
        *(dst + i) = *(src + i);
    }
}


template <typename CumSumNumType>
class GatingEP {
public:
    __aicore__ inline GatingEP() {}

    __aicore__ inline void Init(GM_ADDR topk, GM_ADDR idxArr,
                                GM_ADDR tokenIndex, GM_ADDR CumSum,
                                GM_ADDR originalIndex, GM_ADDR validIndex,
                                GM_ADDR globalSortWorkspace, GM_ADDR cumSumWorkspace,
                                GM_ADDR syncWorkspace, AtbOps::GatingTilingData *tiling_data)
    {
        InitParams(tiling_data);
        int32_t selectNum = deviceExpertNum > 0 ? deviceExpertNum : expertNum;
        // 搬运的数据对齐block时候补齐的数字(一个block 32bytes 对应8个float)
        topkGm.SetGlobalBuffer((__gm__ int32_t *)topk, topKNum);
        idxArrGm.SetGlobalBuffer((__gm__ int32_t *)idxArr, topKNum);
        tokenIndexGm.SetGlobalBuffer((__gm__ int32_t *)tokenIndex, topKNum);
        cumSumGm.SetGlobalBuffer((__gm__ CumSumNumType *)CumSum, selectNum);
        originalGm.SetGlobalBuffer((__gm__ int32_t *)originalIndex, topKNum);
        if (deviceExpertNum > 0) {
            validIndexGm.SetGlobalBuffer((__gm__ int32_t *)validIndex, 1);
        } else {
            validIndexGm.SetGlobalBuffer((__gm__ int32_t *)validIndex, 0);
        }
        // 把一块workspace切成两块使用
        globalSortBlock.SetGlobalBuffer((__gm__ float *)globalSortWorkspace, 2 * topKNumPadded);
        globalSortBlock2.SetGlobalBuffer((__gm__ float *)globalSortWorkspace + 2 * topKNumPadded, 2 * topKNumPadded);
        cumSumBlock.SetGlobalBuffer((__gm__ int32_t *)cumSumWorkspace, actualCoreNum * expertNum);
        syncGm.SetGlobalBuffer((__gm__ int32_t *)syncWorkspace, syncSize);

        // ub总大小192KB
        pipe.InitBuffer(inQueueTopK, BUFFER_NUM, TILE_NUM * INT32_SIZE);  // 2 * 4 KB
        pipe.InitBuffer(inQueueIdxArr, BUFFER_NUM, TILE_NUM * INT32_SIZE);  // 2 * 4 KB
        pipe.InitBuffer(outQueueGroupSortToken, BUFFER_NUM, GLOBAL_SORT_UB_HALF_LENGTH * INT32_SIZE);  // 21.5 * 4 KB
        pipe.InitBuffer(outQueueSortToken, BUFFER_NUM, GLOBAL_SORT_UB_HALF_LENGTH * INT32_SIZE);  // 21.5 * 4 KB
        // 同步用
        pipe.InitBuffer(syncTQue, BUFFER_NUM, SYNC_UB_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx == actualCoreNum) {  // 尾核做cumSum聚合计算，需要等待其他核对cumSum局部计算完成
            if (expertNum > 0) {
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
            if (expertNum > 0) {
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
            if (expertNum > 0) {
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
            // 所有核均已完成局部排序，进行全局排序
            PipeBarrier<PIPE_ALL>();
            // 标明目前的数据在sortedGlobal的位置
            bool switchFlag = false;
            GlobalSort(switchFlag);
            PipeBarrier<PIPE_ALL>();
            if (expertNum > 0) {
                auto sync_buf_read = syncTQue.AllocTensor<int32_t>();
                IBWait(syncGm, sync_buf_read, actualCoreNum, 1);
                syncTQue.FreeTensor(sync_buf_read);
            }

            // 输出original_index，把数据从workspace拷贝到tokenIndexGm
            // switchFlag标明在globalSortBlock还是globalSortBlock2
            CopyGm2Gm(switchFlag ? globalSortBlock2 : globalSortBlock, originalGm);
        }
    }

private:

    // 全局排序的参数结构体
    struct GmsParams {
        // 每一个队列需要排序的长度
        int (&gmsLengths)[4];
        // 每一个队列的排序初始元素在workspace上的起始位置
        int (&gmsCurrentHead)[4];
        // 队列的总长度(<=4)
        int &queueNum;
        // ub上数据加载的tensor（input，会切成四部分使用）
        LocalTensor<float> &srcLocalTensor;
        // ub上排序的结果（output）
        LocalTensor<float> &dstLocalTensor;
        // 数据在gm上存放的位置
        bool &gmTensorIndex;
        GlobalTensor<float> (&buffLocal)[2];
    };

    __aicore__ inline void InitParams(AtbOps::GatingTilingData *tiling_data)
    {
        blockIdx = GetBlockIdx();
        topkExpertNum = tiling_data->topkExpertNum;
        topKNum = tiling_data->topKNum;
        expertNum = tiling_data->cumSumNum;
        cumSumNum32BytesPadded = tiling_data->cumSumNum32BytesPadded;
        actualCoreNum = tiling_data->actualCoreNum;
        tailBlockDataSize = tiling_data->tailBlockDataSize;
        syncSize = tiling_data->syncSize;
        blockNumPerCore = tiling_data->blockNumPerCore[blockIdx];
        offSet = tiling_data->offSetPerCore[blockIdx];
        topKNumPadded = tiling_data->topKNumPadded;
        deviceExpertNum = tiling_data->deviceExpertNum;
        for (int i = 0; i < deviceExpertNum; i++) {
            deviceExpert[i] = tiling_data->deviceExpert[i];
        }
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
            PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_ALL>();
    }

    // 单核cumSum局部计算
    __aicore__ inline void PartCumSum()
    {
        // 单核计算局部cum_sum
        if (expertNum > 0) {
            ComputeCumSumPart();
            PipeBarrier<PIPE_ALL>();
        }
    }

    // 单核计算局部cumSum，并把局部结果搬到cumSumWorkspace
    __aicore__ inline void ComputeCumSumPart()
    {
        LocalTensor<int32_t> cumSumPartLocalTensor = outQueueGroupSortToken.AllocTensor<int32_t>();
        for (int i = 0; i < expertNum; i++) {
            cumSumPartLocalTensor.SetValue(i, selectedExpertCount[i]);
        }
        DataCopyExtParams copyParams1{1, static_cast<uint32_t>(expertNum * INT32_SIZE), 0, 0, 0};
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(cumSumBlock[GetBlockIdx() * expertNum], cumSumPartLocalTensor, copyParams1);
        PipeBarrier<PIPE_ALL>();
        outQueueGroupSortToken.FreeTensor(cumSumPartLocalTensor);
    }

    // 将分核计算的局部结果Add聚合，放到一个localTensor中
    __aicore__ inline void calculateAndCopy2CumSumGm()
    {
        LocalTensor<int32_t> all = outQueueGroupSortToken.AllocTensor<int32_t>();
        LocalTensor<int32_t> accumulator = all[0];
        LocalTensor<int32_t> src0 = all[cumSumNum32BytesPadded];
        LocalTensor<int32_t> src1 = all[cumSumNum32BytesPadded + cumSumNum32BytesPadded];

        DataCopyPadExtParams<int32_t> padParams{false, 0, static_cast<uint8_t>(0), 0};  // 不填充

        // 聚合计算cumSum
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(expertNum * INT32_SIZE), 0, 0, 0};
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(accumulator, cumSumBlock, copyParams, padParams);
        PipeBarrier<PIPE_ALL>();

        for (int i = 1; i < actualCoreNum; i++) {
            // src0
            DataCopyExtParams copyParams0{1, static_cast<uint32_t>(expertNum * INT32_SIZE), 0, 0, 0};
            DataCopyPad(src0, cumSumBlock[i * expertNum], copyParams0, padParams);
            PipeBarrier<PIPE_ALL>();
            // src1
            // DataCopyPad不支持localTensor->localTensor，已32字节对齐，因此可用DataCopy
            DataCopy(src1, accumulator, cumSumNum32BytesPadded);
            PipeBarrier<PIPE_ALL>();
            // add
            Add(accumulator, src0, src1, expertNum);
        }

        // 将聚合计算binCount结果累加并搬运到cumSumGm输出
        PipeBarrier<PIPE_ALL>();
        LocalTensor<CumSumNumType> cumSumLocalTensor = outQueueSortToken.AllocTensor<CumSumNumType>();
        int32_t acc = 0;

        int32_t selectNum = deviceExpertNum > 0 ? deviceExpertNum : expertNum;
        for (int i = 0; i < selectNum; i++) {
            int chosenIndex = deviceExpertNum > 0 && deviceExpertNum < expertNum
                              ? deviceExpert[i] : i;
            acc = acc + accumulator.GetValue(chosenIndex);
            cumSumLocalTensor.SetValue(i, static_cast<CumSumNumType>(acc));
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copyParams1{1, static_cast<uint32_t>(expertNum * INT32_SIZE), 0, 0, 0};
        DataCopyPad(cumSumBlock, accumulator, copyParams1);
        if (sizeof(CumSumNumType) == INT32_SIZE) {
            DataCopyExtParams copyParams2{1, static_cast<uint32_t>(selectNum * INT32_SIZE), 0, 0, 0};
            DataCopyPad(cumSumGm, cumSumLocalTensor, copyParams2);
        } else {
            DataCopyExtParams copyParams2{1, static_cast<uint32_t>(selectNum * sizeof(CumSumNumType)), 0, 0, 0};
            DataCopyPadUB2GMImpl((__gm__ uint32_t*)cumSumGm.GetPhyAddr(),
                            (__ubuf__ uint32_t*)cumSumLocalTensor.GetPhyAddr(),
                            copyParams2); // datacopypad int64
        }
        PipeBarrier<PIPE_ALL>();
        outQueueGroupSortToken.FreeTensor(all);
        outQueueSortToken.FreeTensor(cumSumLocalTensor);

        auto sync_buf_write = syncTQue.AllocTensor<int32_t>();
        IBSet(syncGm, sync_buf_write, actualCoreNum, 1);
        syncTQue.FreeTensor(sync_buf_write);
    }

    __aicore__ inline void GlobalSort(bool &switchFlag)
    {
        // 申请中间用于排序的tensor
        LocalTensor<float> srcLocalTensor = outQueueGroupSortToken.AllocTensor<float>();
        LocalTensor<float> dstLocalTensor = outQueueSortToken.AllocTensor<float>();
        GlobalTensor<float> sortedGlobal[2] = {globalSortBlock, globalSortBlock2};
        PipeBarrier<PIPE_ALL>();

        // 目前已经有多少块tile已经有序
        int orderBlock = topKNumPadded / TILE_NUM;
        // 每次把4个有序的队列合成一块
        int globalSortBlockCount = 4;

        int length[4];
        int currentHead[4];

        // blockSize代表每次排序多少块Tile
        int blockSize = 1;
        for (blockSize = 1; blockSize < orderBlock; blockSize *= globalSortBlockCount) {
            for (int tileIndex = 0; tileIndex < orderBlock; tileIndex += blockSize * globalSortBlockCount) {
                int mrgTileNum = orderBlock - tileIndex < blockSize * globalSortBlockCount
                                 ? (orderBlock - tileIndex) : (blockSize * globalSortBlockCount);
                int queueNum = (mrgTileNum + blockSize - 1) / blockSize;
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
                PipeBarrier<PIPE_ALL>();
                GlobalMrgSort(params);
                PipeBarrier<PIPE_ALL>();
            }
            PipeBarrier<PIPE_ALL>();
            switchFlag = !switchFlag;
        }
        PipeBarrier<PIPE_ALL>();

        outQueueGroupSortToken.FreeTensor(srcLocalTensor);
        outQueueSortToken.FreeTensor(dstLocalTensor);
    }

    __aicore__ inline void GlobalMrgSort(GmsParams & params)
    {
        // 每个队列还需要排序的数据有多长
        int queueNum[4] {params.gmsLengths[0], params.gmsLengths[1],
                         params.gmsLengths[2], params.gmsLengths[3]};
        // 有效队列的数量
        int queueLength = params.queueNum;
        // 每个队列里面要排序数的起始位置
        int currentHead[4] {params.gmsCurrentHead[0], params.gmsCurrentHead[1],
                            params.gmsCurrentHead[2], params.gmsCurrentHead[3]};
        // 已经有多少个数完成排序了
        int totalMrgLen = 0;
        // 这次排序的数据在gm上的起始位置
        int originalPosition = currentHead[0];
        // 得到tensor的指针
        LocalTensor<float> srcLocalTensor {params.srcLocalTensor};
        LocalTensor<float> dstLocalTensor {params.dstLocalTensor};
        GlobalTensor<float> srcGmTensor = params.buffLocal[params.gmTensorIndex];
        GlobalTensor<float> dstGmTensor = params.buffLocal[!params.gmTensorIndex];
        PipeBarrier<PIPE_ALL>();

        if (queueLength == 1) {
            int repeatTimes = (queueNum[0] + TILE_NUM - 1) / TILE_NUM;
            int tailNum = queueNum[0] % TILE_NUM == 0 ? TILE_NUM : queueNum[0] % TILE_NUM;
            for (int i = 0; i < repeatTimes; i++) {
                int executeNum = i == repeatTimes - 1
                                 ? tailNum
                                 : TILE_NUM;
                DataCopyPad(dstLocalTensor, srcGmTensor[DOUBLE * currentHead[0] + DOUBLE * i * TILE_NUM],
                            {1, static_cast<uint32_t>(DOUBLE * executeNum * INT32_SIZE), 0, 0, 0},
                            {false, 0, 0, 0});
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(dstGmTensor[DOUBLE * currentHead[0] + DOUBLE * i * TILE_NUM], dstLocalTensor,
                            {1, static_cast<uint32_t>(DOUBLE * executeNum * INT32_SIZE), 0, 0, 0});
                PipeBarrier<PIPE_ALL>();
            }
            PipeBarrier<PIPE_ALL>();
            return;
        }

        while (queueLength > 1) {
            // 本轮排序实际排序的数量
            uint16_t tmpSortLen[4];

            // 把srcLocalTensor切成4份，每份的起始地址都是i * TILE_NUM * 2
            for (int i = 0; i < queueLength; i++) {
                int sortLength = queueNum[i] < maxSortLengthArr[i] ? queueNum[i] : maxSortLengthArr[i];
                tmpSortLen[i] = sortLength;
                int gmStartPosition = currentHead[i];
                DataCopyPad(srcLocalTensor[i * GLOBAL_SORT_NUM * DOUBLE], srcGmTensor[DOUBLE * gmStartPosition],
                            {1, static_cast<uint32_t>(DOUBLE * sortLength * INT32_SIZE), 0, 0, 0},
                            {false, 0, 0, 0});
                PipeBarrier<PIPE_ALL>();
            }
            PipeBarrier<PIPE_ALL>();

            // 排序
            MrgSort4Info params{tmpSortLen, true, validQueueArr[queueLength], 1};
            PipeBarrier<PIPE_ALL>();

            MrgSort<float>(dstLocalTensor,
                           {srcLocalTensor[0], srcLocalTensor[GLOBAL_SORT_NUM * DOUBLE],
                            srcLocalTensor[GLOBAL_SORT_NUM * DOUBLE * DOUBLE],
                            srcLocalTensor[GLOBAL_SORT_NUM * DOUBLE * 3]},
                           params);
            PipeBarrier<PIPE_ALL>();

            // 排序排了多少数
            uint16_t sortedLen[4];
            GetMrgSortResult(sortedLen[0], sortedLen[1], sortedLen[2], sortedLen[3]);  // 2 3 为队列下标
            const uint16_t localMrgLen = sortedLen[0] + sortedLen[1] + sortedLen[2] + sortedLen[3];  // 2 3 为队列下标
            PipeBarrier<PIPE_ALL>();

            // 将排序的结果拷贝到dstGm上
            DataCopyPad(dstGmTensor[DOUBLE * originalPosition + DOUBLE * totalMrgLen], dstLocalTensor,
                        {1, static_cast<uint32_t>(DOUBLE * localMrgLen * INT32_SIZE), 0, 0, 0});
            PipeBarrier<PIPE_ALL>();

            totalMrgLen += localMrgLen;
            PipeBarrier<PIPE_ALL>();

            // 根据此次排序的结果更新排序的数据
            for (int i = 0; i < queueLength; i++) {
                queueNum[i] -= sortedLen[i];
                currentHead[i] += sortedLen[i];
            }
            PipeBarrier<PIPE_ALL>();

            for (int i = 0; i < queueLength; i++) {
                if (queueNum[i] == 0) {
                    for (int j = i; j < 3; j++) {  // 3 Sort排序队列为4个
                        queueNum[j] = queueNum[j + 1];
                        currentHead[j] = currentHead[j + 1];
                    }
                    queueNum[queueLength - 1] = 0;
                    queueLength -= 1;
                    break;
                }
            }

            PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_ALL>();

        // 处理尾部数据
        // 最后只会有一个队列还剩余值，因而直接拷贝queueNum[0]即可
        if (queueNum[0] > 0) {
            int repeatTimes = (queueNum[0] + TILE_NUM - 1) / TILE_NUM;
            int tailNum = queueNum[0] % TILE_NUM == 0 ? TILE_NUM : queueNum[0] % TILE_NUM;
            for (int i = 0; i < repeatTimes; i++) {
                int executeNum = i == repeatTimes - 1
                                 ? tailNum
                                 : TILE_NUM;

                DataCopyPad(dstLocalTensor, srcGmTensor[DOUBLE * currentHead[0] + DOUBLE * i * TILE_NUM],
                            {1, static_cast<uint32_t>(DOUBLE * executeNum * INT32_SIZE), 0, 0, 0},
                            {false, 0, 0, 0});
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(dstGmTensor[DOUBLE * originalPosition + DOUBLE * totalMrgLen + DOUBLE * i * TILE_NUM],
                            dstLocalTensor, {1, static_cast<uint32_t>(DOUBLE * executeNum * INT32_SIZE), 0, 0, 0});
                PipeBarrier<PIPE_ALL>();
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void CopyTile2Gm(const CopyInfo& info, LocalTensor<float> &tmp, GlobalTensor<float> &src)
    {
        uint32_t copyTimes = info.copyTimes;
        uint32_t tailNum = info.tailNum;
        int32_t startCopyLocation = info.startCopyLocation;
        int32_t completedCopy = info.completedCopy;
        uint64_t rsvdCnt = 0;
        LocalTensor<float> gatherMaskTmpLocal = tmp[DOUBLE_TILE_NUM];
        LocalTensor<float> divFloatLocal = tmp[DOUBLE * DOUBLE_TILE_NUM];
        float topkExpertNumFloat = topkExpertNum;
        Duplicate(divFloatLocal, topkExpertNumFloat, TILE_NUM);
        for (int i = 0; i < copyTimes; i++) {
            PipeBarrier<PIPE_ALL>();
            // 利用tmpLocal做中转实现gm2gm的拷贝
            DataCopy(tmp, src[startCopyLocation * 2 + i * DOUBLE_TILE_NUM], DOUBLE_TILE_NUM);
            PipeBarrier<PIPE_ALL>();
            const uint8_t PATTERN_ODD = 2;
            const bool REDUCE_MODE_NORMAL = false;
            const uint32_t MASK_NORMAL = 0;
            const uint8_t src0BlockStride = 1;
            const uint32_t repeatTimes = static_cast<uint32_t>((DOUBLE_TILE_NUM + 63) / 64);
            const uint32_t src0RepeatStride = 8;
            const uint8_t src1RepeatStride = 0;
            GatherMask(gatherMaskTmpLocal, tmp, PATTERN_ODD, REDUCE_MODE_NORMAL, MASK_NORMAL,
                {src0BlockStride, repeatTimes, src0RepeatStride, src1RepeatStride}, rsvdCnt);
            LocalTensor<int32_t> gatherMaskTmpLocalInt = gatherMaskTmpLocal.ReinterpretCast<int32_t>();

            // 尾块可能存在不对齐的情况，需要额外处理
            PipeBarrier<PIPE_ALL>();
            if (i == copyTimes - 1) {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailNum * INT32_SIZE), 0, 0, 0};
                DataCopyPad(originalGm[completedCopy + i * TILE_NUM], gatherMaskTmpLocalInt, copyParams);
            } else {
                DataCopy(originalGm[completedCopy + i * TILE_NUM], gatherMaskTmpLocalInt, TILE_NUM);
            }
            PipeBarrier<PIPE_ALL>();

            if (expertNum > 0) {
                LocalTensor<float> gatherMaskTmpLocalFloat = gatherMaskTmpLocalInt.ReinterpretCast<float>();
                Cast(gatherMaskTmpLocalFloat, gatherMaskTmpLocalInt, RoundMode::CAST_FLOOR, TILE_NUM);
                PipeBarrier<PIPE_V>();
                Div(gatherMaskTmpLocalFloat, gatherMaskTmpLocalFloat, divFloatLocal, TILE_NUM);
                PipeBarrier<PIPE_V>();
                Cast(gatherMaskTmpLocalInt, gatherMaskTmpLocalFloat, RoundMode::CAST_FLOOR, TILE_NUM);
                PipeBarrier<PIPE_ALL>();
                if (i == copyTimes - 1) {
                    DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailNum * INT32_SIZE), 0, 0, 0};
                    DataCopyPad(tokenIndexGm[completedCopy + i * TILE_NUM], gatherMaskTmpLocalInt, copyParams);
                    PipeBarrier<PIPE_ALL>();
                } else {
                    DataCopyExtParams copyParams{1, static_cast<uint32_t>(TILE_NUM * INT32_SIZE), 0, 0, 0};
                    DataCopyPad(tokenIndexGm[completedCopy + i * TILE_NUM], gatherMaskTmpLocalInt, copyParams);
                    PipeBarrier<PIPE_ALL>();
                }
            }
        }
    }

__aicore__ inline void CopyGm2Gm(GlobalTensor<float> &srcLocal, GlobalTensor<int32_t> &dstLocal)
    {
        uint32_t copyTimes = (topKNum / TILE_NUM) + (topKNum % TILE_NUM == 0 ? 0 : 1);
        LocalTensor<float> tmpLocal = outQueueSortToken.AllocTensor<float>();

        int32_t validIndexNum = topKNum;
        if (deviceExpertNum > 0 && deviceExpertNum < expertNum) {
            validIndexNum = cumSumBlock.GetValue(deviceExpert[0]);
            int32_t previousExpert = deviceExpert[0];
            int32_t currentExpert = previousExpert;
            int32_t cumSumFull = 0;
            for (int i = 0; i <= previousExpert; i++) {
                cumSumFull += cumSumBlock.GetValue(i);
            }
            int32_t startCopyLocation = cumSumFull - cumSumBlock.GetValue(previousExpert);
            int32_t endCopyLocation = cumSumFull;
            int32_t completedCopy = 0;
            for (int i = 1; i < deviceExpertNum + 1; i++) {
                if (i < deviceExpertNum) {
                    validIndexNum += cumSumBlock.GetValue(deviceExpert[i]);
                    currentExpert = deviceExpert[i];
                } else {
                    currentExpert = previousExpert + TAIL_COPY_FLAG;
                }
                for (int j = previousExpert + 1; j <= currentExpert; j++) {
                    cumSumFull += cumSumBlock.GetValue(j);
                }
                if (currentExpert == previousExpert + 1) {
                    endCopyLocation = cumSumFull;
                } else {
                    uint32_t copyTimes = (endCopyLocation - startCopyLocation) / TILE_NUM +
                                         ((endCopyLocation - startCopyLocation) % TILE_NUM == 0 ? 0 : 1);
                    uint32_t tailNum = (endCopyLocation - startCopyLocation) % TILE_NUM == 0 ? TILE_NUM :
                                       (endCopyLocation - startCopyLocation) % TILE_NUM;
                    struct CopyInfo info = {copyTimes, tailNum, startCopyLocation, completedCopy};
                    CopyTile2Gm(info, tmpLocal, srcLocal);
                    completedCopy += (endCopyLocation - startCopyLocation);
                    endCopyLocation = cumSumFull;
                    startCopyLocation = endCopyLocation - cumSumBlock.GetValue(currentExpert);
                }
                previousExpert = currentExpert;
            }
        } else {
            uint32_t copyTimes = (topKNum / TILE_NUM) + (topKNum % TILE_NUM == 0 ? 0 : 1);
            uint32_t tailNum = topKNum % TILE_NUM == 0 ? TILE_NUM : topKNum % TILE_NUM;
            CopyTile2Gm({copyTimes, tailNum, 0, 0}, tmpLocal, srcLocal);
        }
        if (deviceExpertNum > 0) {
            validIndexGm.SetValue(0, validIndexNum);
        }

        outQueueSortToken.FreeTensor(tmpLocal);
    }

    __aicore__ inline void CopyIn(uint32_t processIndex, uint32_t processNum)
    {
        // paddingNum  当前tiling，为了可以切分为完整block，需要补充的int数
        uint32_t paddingNum = (processNum * INT32_SIZE) % 32 == 0 ?
                              0 : (32 - (processNum * INT32_SIZE) % 32) / INT32_SIZE;
        LocalTensor<int32_t> topkLocal = inQueueTopK.AllocTensor<int32_t>();
        DataCopyExtParams copyParams1{1, static_cast<uint32_t>(processNum * INT32_SIZE), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams1{true, 0, static_cast<uint8_t>(paddingNum), paddingValueInt};
        DataCopyPad(topkLocal, topkGm[offSet + processIndex * TILE_NUM], copyParams1, padParams1);
        PipeBarrier<PIPE_ALL>();

        LocalTensor<int32_t> idxArrLocal = inQueueIdxArr.AllocTensor<int32_t>();
        DataCopyExtParams copyParams2{1, static_cast<uint32_t>(processNum * sizeof(uint32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams2{true, 0, static_cast<uint8_t>(paddingNum), paddingValueInt};
        DataCopyPad(idxArrLocal, idxArrGm[offSet + processIndex * TILE_NUM], copyParams2, padParams2);
        PipeBarrier<PIPE_ALL>();

        inQueueTopK.EnQue(topkLocal);
        inQueueIdxArr.EnQue(idxArrLocal);
    }

    __aicore__ inline void Compute(uint32_t processIndex, uint32_t processNum)
    {
        // Sort32需要32bytes对齐，对尾部数据填充，填充paddingNum个int数据，此次数据长度为block的整数倍
        uint32_t paddingNum = (processNum * INT32_SIZE) % 32 == 0 ?
                              0 : (32 - (processNum * INT32_SIZE) % 32) / INT32_SIZE;

        // 数据出队，topkLocalInt、idxArrLocalInt中均一共有processNum+paddingNum个数，已满足32bytes对齐
        LocalTensor<int32_t> topkLocalInt = inQueueTopK.DeQue<int32_t>();
        LocalTensor<int32_t> idxArrLocalInt = inQueueIdxArr.DeQue<int32_t>();
        LocalTensor<uint32_t> idxArrLocal = idxArrLocalInt.ReinterpretCast<uint32_t>();

        // 计算bincount
        if (expertNum > 0) {
            for (uint32_t i = 0; i < processNum; i++) {
                uint32_t expertIndex = topkLocalInt.GetValue(i);
                selectedExpertCount[expertIndex]++;
            }
        }
        PipeBarrier<PIPE_ALL>();

        LocalTensor<float> groupSortLocal = outQueueGroupSortToken.AllocTensor<float>();  // 4 * TILE_NUM
        LocalTensor<float> sortLocal = outQueueSortToken.AllocTensor<float>();
        Cast(sortLocal, topkLocalInt, RoundMode::CAST_FLOOR, processNum + paddingNum);
        PipeBarrier<PIPE_ALL>();

        // 把sortLocal进行填充，长度达到TILE_NUM
        Duplicate<float>(sortLocal[processNum + paddingNum], paddingValueFloat, TILE_NUM - (processNum + paddingNum));
        Duplicate<uint32_t>(idxArrLocal[processNum + paddingNum], paddingValueUint,
                            TILE_NUM - (processNum + paddingNum));

        // 乘以-1实现降序排列
        float factor = -1;
        Muls(sortLocal, sortLocal, factor, TILE_NUM);
        PipeBarrier<PIPE_V>();

        // 局部排序
        const int elementsPerSort = 32; // 每次排序的元素数
        Sort32<float>(groupSortLocal, sortLocal, idxArrLocal, TILE_NUM / elementsPerSort);
        PipeBarrier<PIPE_V>();

        // 归并排序
        MergeSort4Queue(groupSortLocal, sortLocal);
        PipeBarrier<PIPE_ALL>();

        // 结果入队
        outQueueSortToken.EnQue<float>(sortLocal);
        // 释放本地local
        inQueueTopK.FreeTensor(topkLocalInt);
        inQueueIdxArr.FreeTensor(idxArrLocal);
        outQueueGroupSortToken.FreeTensor(groupSortLocal);
    }

    __aicore__ inline void MergeSort4Queue(LocalTensor<float> &groupSortLocal, LocalTensor<float> &sortLocal)
    {
        const uint16_t mergeCount = 4;
        LocalTensor<float> sortedQue[2] = {groupSortLocal, sortLocal};
        bool switchFlag = false;
        uint16_t currentQueLength = 32;
        uint16_t currentQueNum = TILE_NUM / currentQueLength;
        while (currentQueLength < TILE_NUM) {
            const uint16_t elementLengths[4] = {currentQueLength,
                                                currentQueLength,
                                                currentQueLength,
                                                currentQueLength};
            const uint16_t fullMrgSortTime = currentQueNum / mergeCount;
            if (fullMrgSortTime > 0) {
                MrgSort4Info params = {elementLengths, false, 0b1111, fullMrgSortTime};
                MrgSort<float>(sortedQue[!switchFlag],
                               {sortedQue[switchFlag][0], sortedQue[switchFlag][currentQueLength * 1 * DOUBLE],
                                sortedQue[switchFlag][currentQueLength * 2 * DOUBLE],
                                sortedQue[switchFlag][currentQueLength * 3 * DOUBLE]},
                               params);
                switchFlag = !switchFlag;
            }
            currentQueNum = fullMrgSortTime;
            currentQueLength *= mergeCount;
        }

        PipeBarrier<PIPE_ALL>();
        if (!switchFlag) {
            DataCopy(sortLocal, groupSortLocal, DOUBLE_TILE_NUM);
        }
    }

    __aicore__ inline void CopyOut(uint32_t processIndex, uint32_t processNum)
    {
        // 结果出队
        LocalTensor<float> sortLocal = outQueueSortToken.DeQue<float>();
        PipeBarrier<PIPE_ALL>();

        // 拷贝到gm
        DataCopy(globalSortBlock[offSet * DOUBLE + processIndex * DOUBLE_TILE_NUM], sortLocal, DOUBLE_TILE_NUM);
        PipeBarrier<PIPE_ALL>();

        // 释放本地local
        outQueueSortToken.FreeTensor(sortLocal);
    }

private:
    TPipe pipe;
    GlobalTensor<int32_t> topkGm;
    GlobalTensor<int32_t> idxArrGm;
    GlobalTensor<int32_t> tokenIndexGm;
    GlobalTensor<int32_t> originalGm;
    GlobalTensor<CumSumNumType> cumSumGm;
    GlobalTensor<int32_t> validIndexGm;
    GlobalTensor<float> globalSortBlock; // workspace
    GlobalTensor<float> globalSortBlock2; // workspace2
    GlobalTensor<int32_t> cumSumBlock; // workspace
    GlobalTensor<int32_t> syncGm; // workspace
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueTopK, inQueueIdxArr;
    TQue<QuePosition::VECIN, BUFFER_NUM> syncTQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueGroupSortToken;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueSortToken;

    float paddingValueFloat = 11300000.0;
    int32_t paddingValueInt = 11300000;
    uint32_t paddingValueUint = 11300000;
    int32_t topkExpertNum = 0;
    int64_t topKNum = -1;
    int32_t expertNum = -1;
    int32_t cumSumNum32BytesPadded = -1;
    int32_t actualCoreNum = 1;
    int32_t tailBlockDataSize = 0;
    int32_t syncSize = 0;
    int32_t blockNumPerCore = 0;
    uint32_t offSet = 0;
    int64_t topKNumPadded = 0;
    int32_t blockIdx = 0;
    int32_t deviceExpertNum = 0;
    // 每个专家被多少个token选中
    int32_t selectedExpertCount[1025] = {0};
    int32_t deviceExpert[1025] = {0};
    int maxSortLengthArr[4] = {2560, 2560, 2560, 2560};
    uint16_t validQueueArr[5] = {0, 0, 0b11, 0b111, 0b1111};
};

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
        GatingEP<int32_t> op;
        op.Init(topk, idxArr, tokenIndex,
                cumSum, originalIndex, validIndex,
                globalSortWorkspace, cumSumWorkspace, syncWorkspace, &tilingData);
        op.Process();
    }
    if (TILING_KEY_IS(2000000001)) {
        GatingEP<int64_t> op;
        op.Init(topk, idxArr, tokenIndex,
                cumSum, originalIndex, validIndex,
                globalSortWorkspace, cumSumWorkspace, syncWorkspace, &tilingData);
        op.Process();
    }
}
