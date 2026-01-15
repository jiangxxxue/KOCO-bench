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
#include "kernels/index/add_valid/tiling/tiling_data.h"
#include "kernels/utils/kernel/kernel_utils.h"
using namespace AscendC;
static constexpr int32_t BUFFER_NUM = 2;
static constexpr int32_t INDEX_TILE_LENGTH = 512;
static constexpr int32_t INT32_SIZE = sizeof(int32_t);
static constexpr int32_t FP16_SIZE = sizeof(half);
static constexpr int32_t FP32_SIZE = sizeof(float);
static constexpr int32_t VALID_LENGTH_UB_SIZE = 32;  // validIndices所占ub大小，单位byte，需要32byte对齐。实际使用4byte
static constexpr int32_t UPDATES_UB_SIZE = 16;  // updates如果不是32byte整数倍的情况，需要32byte对齐，对应16个half


template <typename T>
class IndexAddValid {
public:
    __aicore__ inline IndexAddValid() {}
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                 GM_ADDR validIndicesNum, GM_ADDR workspace,
                                 uint32_t indicesValid_, uint32_t indicesTotal_, uint32_t valueSize_)
    {
        // 超参数初始化
        this->indicesValid = indicesValid_;
        this->indicesTotal = indicesTotal_;
        this->valueSize = valueSize_;
        this->numCore = static_cast<int32_t>(GetBlockNum());

        // gm空间大小设置
        indicesGm.SetGlobalBuffer((__gm__ int32_t *)indices, indicesTotal_);
        validIndicesGm.SetGlobalBuffer((__gm__ int32_t *)validIndicesNum, 1);
        updatesGm.SetGlobalBuffer((__gm__ T *)updates, static_cast<uint64_t>(indicesTotal_) * valueSize);
        varGm.SetGlobalBuffer((__gm__ T *)var + GetBlockIdx() * valueSize,
             static_cast<uint64_t>(indicesTotal_) * valueSize);
        resultGm.SetGlobalBuffer((__gm__ float *)workspace + GetBlockIdx() * valueSize,
             static_cast<uint64_t>(indicesTotal_) * valueSize);

        // valueSize 32B对齐 不做判断直接对齐 每次搬运的时候就要用DataCopyPad做填充 会有更多的开销
        int32_t updatesTileLength;
        if (valueSize % UPDATES_UB_SIZE == 0) {
            updatesTileLength = valueSize;
        } else {
            updatesTileLength = valueSize + UPDATES_UB_SIZE - valueSize % UPDATES_UB_SIZE;
        }

        pipe.InitBuffer(inQueueIndices, BUFFER_NUM, INDEX_TILE_LENGTH * INT32_SIZE);     // index按照512切分
        pipe.InitBuffer(inQueueValidIndices, 1, VALID_LENGTH_UB_SIZE);
        pipe.InitBuffer(inQueueUpdates, BUFFER_NUM, updatesTileLength * FP16_SIZE);
        pipe.InitBuffer(inQueueResult, 1, updatesTileLength * FP32_SIZE);
        pipe.InitBuffer(outQueueVar, 1, updatesTileLength * FP16_SIZE);
        pipe.InitBuffer(outQueueResult, BUFFER_NUM, updatesTileLength * FP32_SIZE);
        pipe.InitBuffer(fp32Buf, updatesTileLength * FP32_SIZE);
    }

    __aicore__ inline void Process()
    {
        // 初始化validIndicesLength
        InitValidIndicesLength();

        int32_t indexBlockNum = static_cast<int32_t>((static_cast<int64_t>(validIndicesLength)
                                 + INDEX_TILE_LENGTH - 1) / INDEX_TILE_LENGTH);   // indices切分块数
        int32_t indexBlockLength = INDEX_TILE_LENGTH;                             // 每个indices块长度
        int32_t indexTailBlockLength;                                             // indices尾块长度
        if (validIndicesLength % INDEX_TILE_LENGTH == 0) {
            indexTailBlockLength = INDEX_TILE_LENGTH;
        } else {
            indexTailBlockLength = validIndicesLength % INDEX_TILE_LENGTH;
        }

        // 将var搬出到fp32的workspace中
        for (int64_t k = 0; k < static_cast<int64_t>(indicesValid); k++) {
            if (k % numCore == GetBlockIdx()) {
                int64_t varOffset = (k / numCore) * numCore * valueSize;
                VarCopyInWork(varOffset);
                VarToWork();
                WorkCopyOut(varOffset);
            }
        }
        // 处理第i个indices块
        for (int32_t i = 0; i < indexBlockNum; i++) {
            if (i == indexBlockNum -1) {
                indexBlockLength = indexTailBlockLength;
            }
            IndexCopyIn(i, indexBlockLength);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AddCompute(i, indexBlockLength);
        }
        // 将fp32中间结果result搬出到fp16的var中
        for (int64_t k = 0; k < static_cast<int64_t>(indicesValid); k++) {
            if (k % numCore == GetBlockIdx()) {
                int64_t varOffset = (k / numCore) * numCore * valueSize;
                VarCopyIn(varOffset);
                ResultToVar();
                VarCopyOut(varOffset);
            }
        }
    }

private:
    __aicore__ inline void InitValidIndicesLength()
    {
        LocalTensor<int32_t> validIndicesLocal = inQueueValidIndices.AllocTensor<int32_t>();
        DataCopyPad(validIndicesLocal, validIndicesGm,
                    {1, static_cast<uint32_t>(INT32_SIZE), 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        validIndicesLength = validIndicesLocal.GetValue(0);
        inQueueValidIndices.FreeTensor(validIndicesLocal);
    }
    __aicore__ inline void VarCopyInWork(int64_t varOffset)
    {
        LocalTensor<T> varLocal = inQueueUpdates.AllocTensor<T>();
        DataCopyPad(varLocal, varGm[varOffset],
                    {1, static_cast<uint32_t>(valueSize * FP16_SIZE), 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        inQueueUpdates.EnQue(varLocal);
    }
    __aicore__ inline void VarToWork()
    {
        LocalTensor<T> varLocal = inQueueUpdates.DeQue<T>();
        LocalTensor<float> var2Local = outQueueResult.AllocTensor<float>();
        Cast(var2Local, varLocal, RoundMode::CAST_NONE, valueSize);
        PipeBarrier<PIPE_V>();
        outQueueResult.EnQue(var2Local);
        inQueueUpdates.FreeTensor(varLocal);
    }
    __aicore__ inline void WorkCopyOut(int64_t varOffset)
    {
        LocalTensor<float> var2Local = outQueueResult.DeQue<float>();
        DataCopyPad(resultGm[varOffset], var2Local,
                    {1, static_cast<uint32_t>(valueSize * FP32_SIZE), 0, 0, 0});
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        outQueueResult.FreeTensor(var2Local);
    }
    __aicore__ inline void IndexCopyIn(int32_t i, int32_t indexBlockLength)
    {
        LocalTensor<int32_t> indexLocal = inQueueIndices.AllocTensor<int32_t>();
        // DataCopy最小搬运单元为32B 尾块的indexBlockLength未必是32B整数倍 需要用DataCopyPad对齐
        // 虽然搬入时会为了对齐会填充0（大于indexBlockLength） 但在Cpomute时会计算真实的个数（indexBlockLength)
        DataCopyPad(indexLocal, indicesGm[i * INDEX_TILE_LENGTH],
                    {1, static_cast<uint32_t>(indexBlockLength * INT32_SIZE), 0, 0, 0}, {false, 0, 0, 0});
        inQueueIndices.EnQue(indexLocal);
    }
    __aicore__ inline void AddendsCopyIn(uint64_t updatesOffset, uint32_t addendOffset)
    {
        LocalTensor<float> addendLocal = inQueueResult.AllocTensor<float>();
        LocalTensor<T> updatesLocal = inQueueUpdates.AllocTensor<T>();
        DataCopyPad(updatesLocal, updatesGm[updatesOffset],
                    {1, static_cast<uint32_t>(valueSize * FP16_SIZE), 0, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(addendLocal, resultGm[addendOffset],
                    {1, static_cast<uint32_t>(valueSize * FP32_SIZE), 0, 0, 0}, {false, 0, 0, 0});
        inQueueResult.EnQue(addendLocal);
        inQueueUpdates.EnQue(updatesLocal);
    }
    __aicore__ inline void AddCompute(int32_t i, int32_t indexBlockLength)
    {
        LocalTensor<int32_t> indexLocal = inQueueIndices.DeQue<int32_t>();
        int32_t index = 0;
        uint32_t blockNum = 0;
        // 处理第i个indices块的第j个索引
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (int32_t j = 0; j < indexBlockLength; j++) {
            // 将[i*blocklength+j]个updates和第index个var搬入
            index = indexLocal.GetValue(j);
            if (index % numCore == GetBlockIdx()) {
                uint64_t updatesOffset = (static_cast<uint64_t>(i) * INDEX_TILE_LENGTH + j) * valueSize;
                uint32_t addendOffset = (index / numCore) * numCore * valueSize;
                WaitFlag<HardEvent::MTE3_MTE2>(index % BUFFER_NUM);
                AddendsCopyIn(updatesOffset, addendOffset);

                LocalTensor<T> updatesLocal = inQueueUpdates.DeQue<T>();
                LocalTensor<float> addendLocal = inQueueResult.DeQue<float>();
                LocalTensor<float> addend2Local = fp32Buf.Get<float>();
                LocalTensor<float> resultLocal = outQueueResult.AllocTensor<float>();
                Cast(addend2Local, updatesLocal, RoundMode::CAST_NONE, valueSize);
                PipeBarrier<PIPE_V>();

                // result = result[indices[i]] + updates[i] 计算结果暂存在result(float)中 最后一次性搬出到var(half)里
                Add(resultLocal, addendLocal, addend2Local, valueSize);
                outQueueResult.EnQue<float>(resultLocal);
                inQueueResult.FreeTensor(addendLocal);
                inQueueUpdates.FreeTensor(updatesLocal);
                AddendsCopyOut(addendOffset);
                SetFlag<HardEvent::MTE3_MTE2>(index % BUFFER_NUM);
            }
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        inQueueIndices.FreeTensor(indexLocal);
    }
    __aicore__ inline void AddendsCopyOut(int32_t addendOffset)
    {
        LocalTensor<float> resultLocal = outQueueResult.DeQue<float>();
        DataCopyPad(resultGm[addendOffset], resultLocal,
                    {1, static_cast<uint32_t>(valueSize * FP32_SIZE), 0, 0, 0});
        outQueueResult.FreeTensor(resultLocal);
    }
    __aicore__ inline void VarCopyIn(int64_t varOffset)
    {
        LocalTensor<float> resultLocal = inQueueResult.AllocTensor<float>();
        DataCopyPad(resultLocal, resultGm[varOffset],
                    {1, static_cast<uint32_t>(valueSize * FP32_SIZE), 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        inQueueResult.EnQue(resultLocal);
    }
    __aicore__ inline void ResultToVar()
    {
        LocalTensor<float> resultLocal = inQueueResult.DeQue<float>();
        LocalTensor<T> varLocal = outQueueVar.AllocTensor<T>();
        Cast(varLocal, resultLocal, RoundMode::CAST_RINT, valueSize);
        PipeBarrier<PIPE_V>();
        outQueueVar.EnQue(varLocal);
        inQueueResult.FreeTensor(resultLocal);
    }
    __aicore__ inline void VarCopyOut(int64_t varOffset)
    {
        LocalTensor<T> varLocal = outQueueVar.DeQue<T>();
        DataCopyPad(varGm[varOffset], varLocal,
                    {1, static_cast<uint32_t>(valueSize * FP16_SIZE), 0, 0, 0});
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        outQueueVar.FreeTensor(varLocal);
    }

private:
    TPipe pipe;
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<int32_t> validIndicesGm;
    GlobalTensor<T> updatesGm;
    GlobalTensor<T> varGm;        // 输出原地写到该输入
    GlobalTensor<float> resultGm;    // workspace 用fp32暂存中间结果
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueIndices, inQueueValidIndices, inQueueUpdates, inQueueResult;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueResult, outQueueVar;
    TBuf<TPosition::VECCALC> fp32Buf;
    uint32_t indicesValid = 0;       // var第一维度的长度，通过外部传入，大于等于indices中的最大值
    uint32_t indicesTotal = 0;       // indices元素总个数，用于分配gm
    uint32_t valueSize = 0;          // var和updates第二维度的长度
    int32_t validIndicesLength = 0;  // 有效indices元素个数
    int32_t numCore = 0;
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::IndexAddValidTilingData *tilingdata)
{
    tilingdata->indicesValid = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->indicesTotal = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->valueSize = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
}

#define GET_TILING_DATA(tiling_arg, tiling_data)                                                            \
    AsdOps::IndexAddValidTilingData tiling_data;                                                            \
    InitTilingData(tiling_arg, &(tiling_data))

// 核函数
extern "C" __global__ __aicore__ void index_add_valid(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                                 GM_ADDR validIndicesNum, GM_ADDR var_out, GM_ADDR workspace,
                                                 GM_ADDR tiling) {
    GET_TILING_DATA(tiling, tiling_data);
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(2000000000)) { // half
                IndexAddValid<half> op;
                op.Init(var, indices, updates, validIndicesNum, workspace, tiling_data.indicesValid,
                    tiling_data.indicesTotal, tiling_data.valueSize);
                op.Process();
            }
    if (TILING_KEY_IS(2100000000)) { // bf16
        IndexAddValid<bfloat16_t> op;
        op.Init(var, indices, updates, validIndicesNum, workspace, tiling_data.indicesValid, tiling_data.indicesTotal,
             tiling_data.valueSize);
        op.Process();
    }
#endif
}