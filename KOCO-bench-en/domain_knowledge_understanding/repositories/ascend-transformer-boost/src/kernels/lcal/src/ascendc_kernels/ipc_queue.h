/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCCL_IPC_QUEUE_H
#define LCCL_IPC_QUEUE_H
#include "sync_collectives.h"
using namespace AscendC;

template <typename T>
class IpcQueue {
public:
    FORCE_INLINE_AICORE IpcQueue() {}

    FORCE_INLINE_AICORE void Init(SyncCollectives *sync, int64_t magic, GM_ADDR workSpace, uint64_t bufferNum,
                                   uint64_t blockNum)
    {
        this->sync = sync;
        this->magic = magic;
        depth = bufferNum / blockNum;
        front = 0;
        rear = 0;
        count = 0;
        this->blockNum = blockNum;
        buff.SetGlobalBuffer((__gm__ T*)workSpace, bufferNum);
        blockIdx = GetBlockIdx();
    }

    FORCE_INLINE_AICORE bool Full()
    {
        if ((rear + 1) % depth == front) {
            return true;
        }
        return false;
    }

    FORCE_INLINE_AICORE GlobalTensor<T> EnQue()
    {
        uint64_t rearOld = rear;
        rear = (rear + 1) % depth;
        return buff[rearOld * blockNum];
    }

    FORCE_INLINE_AICORE void DeQue(int checkRank, int checkBlock = -1)
    {
        if (!Full()) {
            return;
        }
        if (checkBlock == -1) {
            checkBlock = blockIdx;
        }
        sync->WaitInnerFlag(magic, count, checkRank, checkBlock);
        PipeBarrier<PIPE_ALL>();
        int64_t val = sync->GetInnerFlag(checkRank, checkBlock) & EVENT_ID_MASK;
        count = val + 1;
        front = (val + 1) % depth;
    }

    FORCE_INLINE_AICORE void DeQue(int *rankList, int checkCount, int checkBlock = -1)
    {
        if (!Full()) {
            return;
        }
        if (checkBlock == -1) {
            checkBlock = blockIdx;
        }
        int64_t minIndex = LLONG_MAX;
        for (int i = 0; i < checkCount; i++) {
            sync->WaitInnerFlag(magic, count, rankList[i], checkBlock);
            PipeBarrier<PIPE_ALL>();

            int64_t val = sync->GetInnerFlag(rankList[i], checkBlock) & EVENT_ID_MASK;
            if (minIndex > val) {
                minIndex = val;
            }
        }
        count = minIndex + 1;
        front = (minIndex + 1) % depth;
    }
    FORCE_INLINE_AICORE void DeQue(int *rankList, int *blockIdxList, int checkCount)
    {
        if (!Full()) {
            return;
        }

        int64_t minIndex = LLONG_MAX;
        for (int i = 0; i < checkCount; i++) {
            sync->WaitInnerFlag(magic, count, rankList[i], blockIdxList[i]);
            PipeBarrier<PIPE_ALL>();

            int64_t val = sync->GetInnerFlag(rankList[i], blockIdxList[i]) & EVENT_ID_MASK;
            if (minIndex > val) {
                minIndex = val;
            }
        }
        count = minIndex + 1;
        front = (minIndex + 1) % depth;
    }

    FORCE_INLINE_AICORE GlobalTensor<T> ReadFront()
    {
        uint64_t frontOld = front;
        front = (front + 1) % depth;
        return buff[frontOld * blockNum];
    }

private:
    int64_t magic;
    uint64_t depth;
    uint64_t front;
    uint64_t rear;
    uint64_t count;
    uint64_t blockNum;
    GlobalTensor<T> buff;
    SyncCollectives *sync;
    int blockIdx;
};
#endif // LCCL_IPC_QUEUE_H
