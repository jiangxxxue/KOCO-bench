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

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BLOCK_SIZE = 32;
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
constexpr int32_t UB_SIZE = 192 * 1024;
#elif (__CCE_AICORE__ == 300)
constexpr int32_t UB_SIZE = 120 * 1024;
constexpr int32_t UB_SLOT_MAPPING_SIZE = 128 * 1024;
#else
constexpr int32_t UB_SIZE = 128 * 1024;
constexpr int32_t UB_SLOT_MAPPING_SIZE = 128 * 1024;
#endif

template <typename T1, typename T2 = half, bool ISFULL = false>
class ReshapeAndCacheNz {
public:
    __aicore__ inline ReshapeAndCacheNz(AscendC::TPipe *p) : pipe(p) {}
    
    __aicore__ inline void Init(GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn, GM_ADDR valueCacheIn,
        GM_ADDR slotMapping, GM_ADDR keyCacheOut, GM_ADDR valueCacheOut, GM_ADDR tiling)
    {
        InitTilingData(tiling);
        lastDimK = BLOCK_SIZE / sizeof(T1);
        lastDimV = BLOCK_SIZE / sizeof(T2);
        tokenSizeK = numHeads * headSizeK;
        tokenSizeV = numHeads * headSizeV;
        uint32_t coreNum = AscendC::GetBlockNum(); // 核数block_dim
        baseTaskNum = numTokens / coreNum; // 用token_nums去计算基础任务量，一个基础任务量代表一次k和一次v的搬运
        uint32_t tailTaskNum = numTokens % coreNum;

        uint32_t coreIdx = AscendC::GetBlockIdx(); // 当前在第几个核
        startTaskId = coreIdx * baseTaskNum; // 当前kv table存放起始点

        // 任务分配不均的时候，前面的核需要多处理一个任务
        if (coreIdx < tailTaskNum) {
            baseTaskNum++;
            startTaskId += coreIdx;
        } else {
            startTaskId += tailTaskNum;
        }

        slotMappingGM.SetGlobalBuffer((__gm__ int32_t *)slotMapping);
        keyInGM.SetGlobalBuffer((__gm__ T1 *)keyIn);
        keyCacheGM.SetGlobalBuffer((__gm__ T1 *)keyCacheIn);
        valueGM.SetGlobalBuffer((__gm__ T2 *)valueIn);
        valueCacheGM.SetGlobalBuffer((__gm__ T2 *)valueCacheIn);
        if constexpr (ISFULL) {
            maxUbUsed = UB_SIZE / BUFFER_NUM / BUFFER_NUM;
            loopK = (tokenSizeK * sizeof(T1)) / maxUbUsed;
            tailK = (tokenSizeK * sizeof(T1)) % maxUbUsed;
            loopV = (tokenSizeV * sizeof(T2)) / maxUbUsed;
            tailV = (tokenSizeV * sizeof(T2)) % maxUbUsed;
            ubSizeK = loopK == 0 ? tokenSizeK * sizeof(T1) : maxUbUsed;
            ubSizeV = loopV == 0 ? tokenSizeV * sizeof(T2) : maxUbUsed;
            pipe->InitBuffer(queBindK, BUFFER_NUM, ubSizeK);
            pipe->InitBuffer(queBindV, BUFFER_NUM, ubSizeV);
        } else {
            ubSizeK = tokenSizeK * sizeof(T1);
            ubSizeV = tokenSizeV * sizeof(T2);
            pipe->InitBuffer(uBufK, ubSizeK);
            pipe->InitBuffer(uBufV, ubSizeV);
            tempBufK = uBufK.Get<T1>(); // 临时存放token
            tempBufV = uBufV.Get<T2>(); // 临时存放token
        }
    }

    template <typename T>
    __aicore__ inline void CopyToCacheNz(
        uint64_t ubSize, uint32_t loop, uint32_t tail,
        uint32_t start, uint32_t cacheStart,
        AscendC::GlobalTensor<T>& src, AscendC::GlobalTensor<T>& dst,
        AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 1>& queBind)
    {
        AscendC::DataCopyParams copyParamsIn{1, static_cast<uint16_t>(ubSize / BLOCK_SIZE), 0, 0};
        AscendC::DataCopyParams copyParamsOut{static_cast<uint16_t>(ubSize / BLOCK_SIZE), 1, 0,
                                                static_cast<uint16_t>(blockSize - 1)};
        for (uint32_t j = 0; j < loop; j++) {
            auto local = queBind.AllocTensor<T>();
            DataCopy(local, src[start], copyParamsIn);
            queBind.EnQue(local);
            local = queBind.DeQue<T>();
            DataCopy(dst[cacheStart], local, copyParamsOut);
            queBind.FreeTensor(local);
            start += (ubSize / sizeof(T));
            cacheStart += static_cast<uint64_t>((ubSize / sizeof(T)) * blockSize);
        }
        if (tail > 0) {
            copyParamsIn.blockLen = tail / BLOCK_SIZE;
            copyParamsOut.blockCount = tail / BLOCK_SIZE;
            auto local = queBind.AllocTensor<T>();
            DataCopy(local, src[start], copyParamsIn);
            queBind.EnQue(local);
            local = queBind.DeQue<T>();
            DataCopy(dst[cacheStart], local, copyParamsOut);
            queBind.FreeTensor(local);
        }
    }

    __aicore__ inline void Process()
    {
        AscendC::DataCopyParams copyParamsInK{1, static_cast<uint16_t>(ubSizeK / BLOCK_SIZE), 0, 0};
        AscendC::DataCopyParams copyParamsOutK{static_cast<uint16_t>(ubSizeK / BLOCK_SIZE), 1, 0,
                                                static_cast<uint16_t>(blockSize - 1)};
        AscendC::DataCopyParams copyParamsInV{1, static_cast<uint16_t>(ubSizeV / BLOCK_SIZE), 0, 0};
        AscendC::DataCopyParams copyParamsOutV{static_cast<uint16_t>(ubSizeV / BLOCK_SIZE), 1, 0,
                                                static_cast<uint16_t>(blockSize - 1)};
        for (uint32_t i = 0; i < baseTaskNum; i++) {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
            int32_t slotValue = slotMappingGM.GetValue(i + startTaskId);
#else
            uint32_t maxNumTokens = UB_SLOT_MAPPING_SIZE / sizeof(int32_t); // 一次最多搬入max_num_tokens个slot_id
            slotMappingUB = slotMappingBuf.Get<int32_t>();
            uint32_t slotUbOffset = i % maxNumTokens;
            if (slotUbOffset == 0) {
                // slot_mapping从gm搬运到ub
                uint64_t slotGmOffset = static_cast<uint64_t>(startTaskId + i);
                int32_t slotCopyNum = baseTaskNum - i < maxNumTokens ? baseTaskNum - i : maxNumTokens;
                DataCopy(slotMappingUB, slotMappingGM[slotGmOffset],
                    {1, static_cast<uint16_t>((slotCopyNum * sizeof(uint32_t) + BLOCK_SIZE - 1) / BLOCK_SIZE), 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            }
            int32_t slotValue = slotMappingUB.GetValue(slotUbOffset);
#endif
            if (slotValue < 0) {
                continue;
            }
            uint64_t startK = (i + startTaskId) * tokenSizeK;
            uint64_t startV = (i + startTaskId) * tokenSizeV;
            uint64_t blocksIdx = static_cast<uint64_t>(slotValue) / blockSize;
            uint64_t blocksOffset = static_cast<uint64_t>(slotValue) % blockSize;
            uint64_t cacheStartK = blocksIdx * blockSize * tokenSizeK + blocksOffset * lastDimK;
            uint64_t cacheStartV = blocksIdx * blockSize * tokenSizeV + blocksOffset * lastDimV;

            if constexpr (ISFULL) {
                CopyToCacheNz<T1>(ubSizeK, loopK, tailK, startK, cacheStartK, keyInGM, keyCacheGM, queBindK);
                CopyToCacheNz<T2>(ubSizeV, loopV, tailV, startV, cacheStartV, valueGM, valueCacheGM, queBindV);
            } else {
                DataCopy(tempBufK, keyInGM[startK], copyParamsInK);
                DataCopy(tempBufV, valueGM[startV], copyParamsInV);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                DataCopy(keyCacheGM[cacheStartK], tempBufK, copyParamsOutK);
                DataCopy(valueCacheGM[cacheStartV], tempBufV, copyParamsOutV);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
    }

private:
    inline __aicore__ void InitTilingData(GM_ADDR tiling)
    {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        // 取tiling参数
        numTokens = (uint32_t)(*((__gm__ uint32_t *)tiling)); // 取地址解引用，结构体地址偏移再取值
        numHeads = (uint32_t)(*((__gm__ uint32_t *)tiling + 1)); // numHeads index
        headSizeK = (uint32_t)(*((__gm__ uint32_t *)tiling + 2)); // headSizeK index
        headSizeV = (uint32_t)(*((__gm__ uint32_t *)tiling + 3)); // headSizeV index
        blockSize = (uint32_t)(*((__gm__ uint32_t *)tiling + 4)); // blockSize index
#else
        pipe->InitBuffer(slotMappingBuf, UB_SLOT_MAPPING_SIZE);
        // tiling_param从gm搬运到ub
        AscendC::GlobalTensor<uint32_t> tilingParamGM;
        AscendC::LocalTensor<uint32_t> tilingParamLT = slotMappingBuf.Get<uint32_t>();
        tilingParamGM.SetGlobalBuffer((__gm__ uint32_t *)tiling);
        DataCopy(tilingParamLT, tilingParamGM, 32); // tiling num
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        __ubuf__ uint32_t *tilingParamsUB = (__ubuf__ uint32_t *)tilingParamLT.GetPhyAddr();
        // 取tiling参数
        numTokens = (uint32_t)(*((__ubuf__ uint32_t *)tilingParamsUB)); // 取地址解引用，结构体地址偏移再取值
        numHeads = (uint32_t)(*((__ubuf__ uint32_t *)tilingParamsUB + 1)); // numHeads index
        headSizeK = (uint32_t)(*((__ubuf__ uint32_t *)tilingParamsUB + 2)); // headSizeK index
        headSizeV = (uint32_t)(*((__ubuf__ uint32_t *)tilingParamsUB + 3)); // headSizeV index
        blockSize = (uint32_t)(*((__ubuf__ uint32_t *)tilingParamsUB + 4)); // blockSize index
#endif
    }

private:
    uint32_t maxUbUsed{0};
    int32_t lastDimK{0};
    int32_t lastDimV{0};
    uint32_t numHeads{0};
    uint32_t numTokens{0};
    uint32_t blockSize{0};
    uint32_t headSizeK{0};
    uint32_t headSizeV{0};
    uint64_t tokenSizeK{0};
    uint64_t tokenSizeV{0};
    uint32_t startTaskId{0};
    uint32_t baseTaskNum{0};
    uint32_t loopK{0};
    uint32_t tailK{0};
    uint32_t loopV{0};
    uint32_t tailV{0};
    uint64_t ubSizeK{0};
    uint64_t ubSizeV{0};

    AscendC::TPipe* pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> slotMappingBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> uBufK;
    AscendC::TBuf<AscendC::TPosition::VECCALC> uBufV;
    AscendC::LocalTensor<int32_t> slotMappingUB;
    AscendC::LocalTensor<T1> tempBufK;
    AscendC::LocalTensor<T2> tempBufV;
    AscendC::GlobalTensor<int32_t> slotMappingGM;
    AscendC::GlobalTensor<T1> keyInGM;
    AscendC::GlobalTensor<T1> keyCacheGM;
    AscendC::GlobalTensor<T2> valueGM;
    AscendC::GlobalTensor<T2> valueCacheGM;

    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 1> queBindK;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 1> queBindV;
};


extern "C" __global__ __aicore__ void reshape_and_cache_nz(
    GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn, GM_ADDR valueCacheIn,
    GM_ADDR slotMapping, GM_ADDR keyCacheOut, GM_ADDR valueCacheOut, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(110000000)) {
        ReshapeAndCacheNz<int8_t> op(&pipe);
        op.Init(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut, tiling);
        op.Process();
    } else if (TILING_KEY_IS(210000000)) {
        ReshapeAndCacheNz<half> op(&pipe);
        op.Init(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut, tiling);
        op.Process();
    } else if (TILING_KEY_IS(112000000)) {
        ReshapeAndCacheNz<int8_t, half, true> op(&pipe);
        op.Init(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut, tiling);
        op.Process();
    } else if (TILING_KEY_IS(212000000)) {
        ReshapeAndCacheNz<half, half, true> op(&pipe);
        op.Init(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut, tiling);
        op.Process();
    }
}
