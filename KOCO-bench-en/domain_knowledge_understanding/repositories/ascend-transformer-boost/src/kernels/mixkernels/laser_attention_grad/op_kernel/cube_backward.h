/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef __CUBEBACKWARD_FP32_OP_H__
#define __CUBEBACKWARD_FP32_OP_H__

#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))

#include "ppmatmul_const_grad.h"
#include "kernels/utils/kernel/utils.h"
#include "kernels/utils/kernel/iterator.h"

using namespace AscendC;

#ifdef __DAV_C220_CUBE__
namespace CUBE_BACKWARD_FP32_OP {

using WORKSPACE_DTYPE = float;
using WORKSPACE_DTYPE_DP = float;
const int CONTINUED_BLOCKS_NUM_S = 2;
const int CONTINUED_BLOCKS_NUM_DP = 2;
constexpr int32_t SIZE_128 = 128;
constexpr int32_t SIZE_256 = 256;
constexpr int32_t SIZE_384 = 384;
constexpr int32_t SIZE_ONE_K = 1024;
constexpr int64_t BASE_LEN = 128;
constexpr int64_t BASE_BLOCK = 128 * 128;
constexpr int64_t REPEAT_TIME_8 = 8;
constexpr int64_t REPEAT_TIME_64 = 64;
constexpr int64_t SRC_STRIDE_1 = 1;
constexpr int64_t SRC_STRIDE_8 = 8;

template<typename TYPE = half, typename WORKSPACE_TYPE = float>
struct PhyAddrCube1 {
    __gm__ TYPE* left;
    __gm__ TYPE* right;
    __gm__ WORKSPACE_TYPE* out;
    int64_t k = 0;
};

template<typename TYPE = half, typename WORKSPACE_TYPE = float>
struct PhyAddrCube2 {
    __gm__ WORKSPACE_TYPE* left;
    __gm__ TYPE* right;
    __gm__ float* out;
    int64_t k = 0;
};

template<typename TYPE = half, typename WORKSPACE_TYPE = float>
struct PhyAddrCube3 {
    __gm__ WORKSPACE_TYPE* left;
    __gm__ TYPE* right;
    __gm__ T_OUTPUT* out;
    int64_t k = 0;
};

template <typename TYPE, bool IF_BF16>
class CubeBackward {
public:
    __aicore__ inline CubeBackward() {}

    __aicore__ inline void Init(
        __gm__ TYPE * __restrict__ gmQ, __gm__ TYPE * __restrict__ gmK,
        __gm__ TYPE * __restrict__ gmV, __gm__ TYPE * __restrict__ gmDO,
        __gm__ WORKSPACE_DTYPE_DP * __restrict__ gm_dP, __gm__ WORKSPACE_DTYPE * __restrict__ gmS,
        __gm__ float * __restrict__ gmDQ, __gm__ float * __restrict__ gmDK, __gm__ float * __restrict__ gmDV,
        bool isTri, int64_t qSize, int64_t kSize, int64_t batch, int64_t head, int64_t groupSize,
        int64_t baseLength, int64_t sparseMode, int64_t windowLength, int64_t blockNumPerCore);

    __aicore__ inline void Run() ;
    __aicore__ inline void Run_mix();

private:
    __aicore__ __inline__ void addrMappingPre(Addr *addr, int64_t& curAddrLen, int64_t roundId);
    __aicore__ __inline__ void addrMapping_cube1(int64_t roundId, PhyAddrCube1<TYPE, WORKSPACE_DTYPE> *src1,
                                                 PhyAddrCube1<TYPE, WORKSPACE_DTYPE_DP> *src2, int *srcLength);
    __aicore__ __inline__ void cube1(int64_t roundId);
    __aicore__ __inline__ void base_block_mad(PhyAddrCube1<TYPE, WORKSPACE_DTYPE> addr_1,
                                              PhyAddrCube1<TYPE, WORKSPACE_DTYPE_DP> addr_2,
                                              int64_t l0a_offset_remain = -1);
    __aicore__ inline void Mat_mix_cube2_cube3(int64_t roundId);
    template <typename T> __aicore__ __inline__ void AddrToPhyAddr(Addr *addr, int length, __gm__ TYPE * gm_a,
                                                                   __gm__ TYPE * gm_b, __gm__ T * gm_c,
                                                                   PhyAddrCube1<TYPE, T> *src, int *srcLength,
                                                                   int64_t roundId);
    template <typename T> __aicore__ __inline__ void addrMapping_cube2(__gm__ T *__restrict__ left,
                                                                        __gm__ TYPE *__restrict__ right,
                                                                        __gm__ T_OUTPUT *__restrict__ out,
                                                                        int64_t roundId, PhyAddrCube2<TYPE, T> *src,
                                                                        int64_t &len);
    template <typename T> __aicore__ __inline__ void addrMapping_cube3(__gm__ T *__restrict__ left,
                                                                        __gm__ TYPE *__restrict__ right,
                                                                        __gm__ T_OUTPUT *__restrict__ out,
                                                                        int64_t roundId, PhyAddrCube3<TYPE, T> *src,
                                                                        int64_t& switchIndex);
    template <typename T> __aicore__ __inline__ void cube3_matmul(const PhyAddrCube3<TYPE, T> *src, int64_t srcLen,
                                                                    int64_t enventIdA, int64_t enventIdB,
                                                                    int64_t vcoreNumPerHead);
    template <typename T> __aicore__ __inline__ void cube2_cube3_mix(const PhyAddrCube2<TYPE, T> *cube2_addr,
                                                                        int64_t cube2_length,
                                                                        const PhyAddrCube3<TYPE, T> *cube3_addr,
                                                                        int64_t cube3_length,
                                                                        int64_t vcoreNumPerHead);

    __gm__ TYPE* __restrict__ a_cube1{nullptr};
    __gm__ TYPE* __restrict__ b_cube1{nullptr};
    __gm__ TYPE* __restrict__ b_cube2{nullptr};
    __gm__ TYPE* __restrict__ c_cube1{nullptr};
    __gm__ float* __restrict__ c_cube2{nullptr};

    __gm__ TYPE* __restrict__ gmQ{nullptr};
    __gm__ TYPE* __restrict__ gmK{nullptr};
    __gm__ TYPE* __restrict__ gmV{nullptr};
    __gm__ TYPE* __restrict__ gmDO{nullptr};

    __gm__ WORKSPACE_DTYPE_DP* __restrict__ gm_dP{nullptr};
    __gm__ WORKSPACE_DTYPE* __restrict__ gmS{nullptr};
    __gm__ float* __restrict__ gmDQ{nullptr};
    __gm__ float* __restrict__ gmDK{nullptr};
    __gm__ float* __restrict__ gmDV{nullptr};
    __gm__ uint8_t* __restrict__ ffts_addr{nullptr};
    __gm__ uint8_t* __restrict__ tiling_para_gm{nullptr};

    AscendC::Nd2NzParams commonNd2NzParams {
        1,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        0,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        1,
        0
    };

    AscendC::LoadData2dParams commonLoadData2dParamsTranspose {
        0,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        0,
        0,
        true,
        0
    };

    AscendC::LoadData2dParams commonLoadData2dParamsNoTranspose {
        0,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        0,
        0,
        false,
        0
    };

    AscendC::MmadParams commonMadParams {
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        3,
        false,
        true
    };

    AscendC::FixpipeParamsV220 commonFixpipeParamsV220 {
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        false
    };

    AsdopsBuffer<ArchType::ASCEND_V220> asdopsBuf;

    LocalTensor<TYPE> l1_base_a_cube1_tensor;
    LocalTensor<TYPE> l1_base_b_cube1_tensor;
    LocalTensor<TYPE> l1_base_a_cube2_tensor;
    LocalTensor<TYPE> l1_base_b_cube2_tensor;

    LocalTensor<TYPE> l0_a_base_tensor;
    LocalTensor<TYPE> l0_b_base_tensor;
    LocalTensor<float> l0_c_base_tensor;
    LocalTensor<float> l0_c_buf_tensor;

    GlobalTensor<TYPE> temp_tensor_bf16;
    GlobalTensor<float> temp_tensor_fp32;

    int64_t coreNum;
    int64_t curCoreIndex;
    int64_t blockNumPerCore;
    int64_t batchSize;
    int64_t headNum;
    int64_t gqaGroupNum;
    int64_t headDim;
    int64_t seqLenQ;
    int64_t seqLenK;
    int64_t blocksPerColumn;
    int64_t blocksPerRow;
    bool isTri;
    int64_t sparseMode;
    int64_t windowLength;

    bool isCube1;
    bool isCube2;
    bool isCube3;
    bool isSyn;
};

template <typename TYPE, bool IF_BF16>
__aicore__ inline void CubeBackward<TYPE, IF_BF16>::Init(
    __gm__ TYPE * __restrict__ gmQ, __gm__ TYPE * __restrict__ gmK,
    __gm__ TYPE * __restrict__ gmV, __gm__ TYPE * __restrict__ gmDO,
    __gm__ WORKSPACE_DTYPE_DP * __restrict__ gm_dP, __gm__ WORKSPACE_DTYPE * __restrict__ gmS,
    __gm__ float * __restrict__ gmDQ, __gm__ float * __restrict__ gmDK, __gm__ float * __restrict__ gmDV,
    bool isTri, int64_t qSize, int64_t kSize, int64_t batch, int64_t head, int64_t groupSize, int64_t baseLength,
    int64_t sparseMode, int64_t windowLength, int64_t blockNumPerCore) {
    this->gmQ = gmQ;
    this->gmK = gmK;
    this->gmV = gmV;
    this->gmDO = gmDO;

    this->gm_dP = gm_dP;
    this->gmS = gmS;
    this->gmDQ = gmDQ;
    this->gmDK = gmDK;
    this->gmDV = gmDV;

    this->seqLenQ = qSize;
    this->seqLenK = kSize;
    this->batchSize = batch;
    this->headNum = head;
    this->headDim = baseLength;
    this->gqaGroupNum = (head + groupSize - 1) / groupSize;
    this->blockNumPerCore = blockNumPerCore;
    this->isTri = isTri;
    this->sparseMode = sparseMode;
    this->windowLength = windowLength;

    this->coreNum =  get_block_num();
    this->curCoreIndex = get_block_idx();

    this->blocksPerColumn = this->isTri ? ((this->seqLenQ) / BASE_LEN / 2)  : ((this->seqLenQ) / BASE_LEN);
    this->blocksPerRow =  this->isTri ? ((this->seqLenK) / BASE_LEN + 1) : ((this->seqLenK) / BASE_LEN);

    if (this->isTri && this->sparseMode == 1) {
        this->blocksPerColumn = (this->seqLenQ / BASE_LEN) - (this->windowLength / BASE_LEN / 2);
        this->blocksPerRow = (this->windowLength / BASE_LEN) + 1;
    }

    isCube1 = true;
    isCube2 = true;
    isCube3 = true;
    isSyn = true;

    temp_tensor_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(0));
    temp_tensor_fp32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(0));

    // init L1 tensor
    l1_base_a_cube1_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(0);
    l1_base_b_cube1_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_128 * SIZE_ONE_K);
    l1_base_a_cube2_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_256 * SIZE_ONE_K);
    l1_base_b_cube2_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_384 * SIZE_ONE_K);

    l0_a_base_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, TYPE>(0);
    l0_b_base_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, TYPE>(0);
    l0_c_base_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
    l0_c_buf_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    commonFixpipeParamsV220.quantPre = QuantMode_t::NoQuant;
    commonFixpipeParamsV220.unitFlag = 3;

    commonMadParams.m = BASE_LEN;
    commonMadParams.n = BASE_LEN;
    commonMadParams.k = BASE_LEN;
}


template <typename TYPE, bool IF_BF16>
__aicore__ inline void CubeBackward<TYPE, IF_BF16>::cube1(int64_t roundId) {
    PhyAddrCube1<TYPE, WORKSPACE_DTYPE> src1[MAX_SWITCH_TIME];
    PhyAddrCube1<TYPE, WORKSPACE_DTYPE_DP> src2[MAX_SWITCH_TIME];
    int srcLength = 0;
    addrMapping_cube1(roundId, src1, src2, &srcLength);

    SET_FLAG(M, MTE1, EVENT_ID6);
    SET_FLAG(M, MTE1, EVENT_ID7);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);
    SET_FLAG(MTE1, MTE2, EVENT_ID4);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(M, MTE1, EVENT_ID2);

    for (int i = 0; i < srcLength; i++) {
        WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        commonNd2NzParams.nValue = BASE_LEN;
        commonNd2NzParams.dValue = BASE_LEN;
        commonNd2NzParams.srcDValue = BASE_LEN;
        commonNd2NzParams.srcNdMatrixStride = 0;
        commonNd2NzParams.dstNzC0Stride = BASE_LEN;
        commonNd2NzParams.dstNzMatrixStride = 0;
        temp_tensor_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(src1[i].left));
        AscendC::DataCopy(
            l1_base_a_cube1_tensor,
            temp_tensor_bf16,
            commonNd2NzParams
        );

        temp_tensor_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(src2[i].left));
        AscendC::DataCopy(
            l1_base_a_cube1_tensor[BASE_BLOCK_SIZE],
            temp_tensor_bf16,
            commonNd2NzParams
        );

        SET_FLAG(MTE2, MTE1, EVENT_ID3);
        WAIT_FLAG(MTE2, MTE1, EVENT_ID3);
        WAIT_FLAG(M, MTE1, EVENT_ID2);

        for (int64_t i = 0; i < BASE_LEN / BLOCK_SIZE; i++) {
            commonLoadData2dParamsNoTranspose.repeatTimes = REPEAT_TIME_8;
            commonLoadData2dParamsNoTranspose.srcStride = SRC_STRIDE_8;
            AscendC::LoadData(
                l0_a_base_tensor[i * BASE_LEN * BLOCK_SIZE],
                l1_base_a_cube1_tensor[i * CUBE_MATRIX_SIZE],
                commonLoadData2dParamsNoTranspose
            );
        }

        for (int64_t i = 0; i < BASE_LEN / BLOCK_SIZE; i++) {
            AscendC::LoadData(
                l0_a_base_tensor[BASE_LEN * BASE_LEN + i * BASE_LEN * BLOCK_SIZE],
                l1_base_a_cube1_tensor[BASE_LEN * BASE_LEN + i * CUBE_MATRIX_SIZE],
                AscendC::LoadData2dParams(
                    0,                // startIndex
                    REPEAT_TIME_8,    // repeatTimes
                    SRC_STRIDE_8,     // srcStride
                    0,                // sid
                    0,                // dstGap
                    false,            // ifTranspose
                    0                 // addrMode
                )
            );
        }

        base_block_mad(src1[i], src2[i]);

        SET_FLAG(MTE1, MTE2, EVENT_ID1);
        SET_FLAG(M, MTE1, EVENT_ID2);
    }

    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(M, MTE1, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID4);
    WAIT_FLAG(M, MTE1, EVENT_ID6);
    WAIT_FLAG(M, MTE1, EVENT_ID7);
}

template <typename TYPE, bool IF_BF16>
__aicore__ __inline__ void CubeBackward<TYPE, IF_BF16>::base_block_mad(
    PhyAddrCube1<TYPE, WORKSPACE_DTYPE> addr_1,
    PhyAddrCube1<TYPE, WORKSPACE_DTYPE_DP> addr_2, int64_t l0a_offset_remain)
{
    int64_t ping_flag_l1_cube1 = 1;
    int64_t ping_flag_l0b_cube1 = 1;
    int64_t ping_flag_l0c_cube1 = 1;

    auto gm_k = addr_1.right;
    auto gm_c = addr_1.out;
    auto gm_c2 = addr_2.out;
    bool is_cal_s = true;


    for (int64_t j = 0; j < addr_1.k + addr_2.k; j++) {
        auto event_id = ping_flag_l1_cube1 ? EVENT_ID3 : EVENT_ID4;
        if (j == addr_1.k) {
            gm_k = addr_2.right;
            gm_c = addr_2.out;
        }

        WAIT_FLAG(MTE1, MTE2, event_id);

        auto l1_buf_b_tensor = ping_flag_l1_cube1 ? l1_base_b_cube1_tensor : l1_base_b_cube1_tensor[BASE_BLOCK_SIZE];
        commonNd2NzParams.nValue = BASE_LEN;
        commonNd2NzParams.dValue = BASE_LEN;
        commonNd2NzParams.srcDValue = BASE_LEN;
        commonNd2NzParams.srcNdMatrixStride = 0;
        commonNd2NzParams.dstNzC0Stride = BASE_LEN;
        commonNd2NzParams.dstNzMatrixStride = 0;
        temp_tensor_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(gm_k));
            AscendC::DataCopy(
                l1_buf_b_tensor,
                temp_tensor_bf16,
                commonNd2NzParams
            );

        SET_FLAG(MTE2, MTE1, event_id);
        ping_flag_l1_cube1 = 1 - ping_flag_l1_cube1;

        auto event_id_2 = ping_flag_l0b_cube1 ? EVENT_ID6 : EVENT_ID7;
        int64_t offset_l0b_cube1 = ping_flag_l0b_cube1 ? 0 : BASE_BLOCK_SIZE;

        WAIT_FLAG(MTE2, MTE1, event_id);
        WAIT_FLAG(M, MTE1, event_id_2);
        commonLoadData2dParamsNoTranspose.repeatTimes = REPEAT_TIME_64;
        commonLoadData2dParamsNoTranspose.srcStride = SRC_STRIDE_1;
        AscendC::LoadData(
                l0_b_base_tensor[offset_l0b_cube1],
                l1_buf_b_tensor,
                commonLoadData2dParamsNoTranspose
            );

        SET_FLAG(MTE1, MTE2, event_id);
        SET_FLAG(MTE1, M, event_id_2);
        WAIT_FLAG(MTE1, M, event_id_2);

        int64_t offset_l0c_offset_cube1 = ping_flag_l0c_cube1 ? 0 : BASE_BLOCK_SIZE;
        int64_t l0a_offset = 0;

        if (j >= addr_1.k) {
            l0a_offset = BASE_BLOCK_SIZE;
        }

        int unit_flag = 0b11;
        commonMadParams.unitFlag = unit_flag;
        commonMadParams.cmatrixInitVal = true;
        AscendC::Mmad(
            l0_c_buf_tensor[offset_l0c_offset_cube1],
            l0_a_base_tensor[l0a_offset],
            l0_b_base_tensor[offset_l0b_cube1],
            commonMadParams
        );

        SET_FLAG(M, MTE1, event_id_2);
        ping_flag_l0b_cube1 = 1 - ping_flag_l0b_cube1;

        auto intriParams = AscendC::FixpipeParamsV220(
            BASE_LEN,
            BASE_LEN,
            BASE_LEN,
            BASE_LEN,
            false
        );
        intriParams.quantPre = QuantMode_t::NoQuant;
        intriParams.unitFlag = unit_flag;

        temp_tensor_fp32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_c));
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
            temp_tensor_fp32,
            l0_c_buf_tensor[offset_l0c_offset_cube1],
            intriParams
        );

        gm_c += BASE_BLOCK_SIZE;
        ping_flag_l0c_cube1 = 1 - ping_flag_l0c_cube1;
        gm_k += BASE_BLOCK_SIZE;
    }
}

template <typename TYPE, bool IF_BF16>
template <typename T>
__aicore__ __inline__ void CubeBackward<TYPE, IF_BF16>::cube3_matmul(const PhyAddrCube3<TYPE, T> *src,
                            int64_t  srcLen, int64_t enventIdA, int64_t enventIdB, int64_t vcoreNumPerHead) {
    if (srcLen == 0) return;

    int64_t switchIndexList[MAX_SWITCH_TIME] = {0};
    int64_t totalLen = 0;
    for (int64_t srcIndex = 0; srcIndex < srcLen; srcIndex ++) {
        switchIndexList[srcIndex] = totalLen;
        totalLen += src[srcIndex].k;
    }

    SET_FLAG(MTE1, MTE2, enventIdA);
    SET_FLAG(MTE1, MTE2, enventIdB);

    commonNd2NzParams.nValue = BASE_LEN;
    commonNd2NzParams.dValue = BASE_LEN;
    commonNd2NzParams.srcDValue = BASE_LEN;
    commonNd2NzParams.srcNdMatrixStride = 0;
    commonNd2NzParams.dstNzC0Stride = BASE_LEN;
    commonNd2NzParams.dstNzMatrixStride = 0;

    temp_tensor_bf16.SetGlobalBuffer(
        reinterpret_cast<__gm__ TYPE *>(src[0].right));
    AscendC::DataCopy(
        l1_base_b_cube2_tensor,
        temp_tensor_bf16,
        commonNd2NzParams
    );

    SET_FLAG(MTE2, MTE1, enventIdA);
    WAIT_FLAG(MTE2, MTE1, enventIdA);

    SET_FLAG(MTE2, MTE1, enventIdB);
    WAIT_FLAG(MTE2, MTE1, enventIdB);


    for (int64_t i = 0; i < BASE_LEN / BLOCK_SIZE; i++) {
        commonLoadData2dParamsTranspose.repeatTimes = BASE_LEN / BLOCK_SIZE;
        commonLoadData2dParamsTranspose.srcStride = BASE_LEN / BLOCK_SIZE;
        AscendC::LoadData(
            l0_b_base_tensor[i * BASE_LEN * BLOCK_SIZE],
            l1_base_b_cube2_tensor[i * CUBE_MATRIX_SIZE],
            commonLoadData2dParamsTranspose
        );
    }

    int64_t FlagL1BPingPong  = 1;
    int64_t FlagL1APingPong  = 1;
    int64_t FlagL0APingPong = 1;
    int64_t FlagL0BPingPong = 1;
    int64_t FlagL0CPingPong = 1;
    int64_t mte1MadPingFlag = 1;

    SET_FLAG(M, MTE1, enventIdA);
    SET_FLAG(M, MTE1, enventIdB);

    int64_t curSwitchIndex = 0;
    int64_t curSwitch  = switchIndexList[curSwitchIndex];
    auto last_event_id = EVENT_ID3;
    for (int64_t kIdx = 0; kIdx < totalLen; kIdx++) {
        if ((srcLen >= 2 && kIdx == switchIndexList[curSwitchIndex+1])) {
            curSwitchIndex ++;
            curSwitch = switchIndexList[curSwitchIndex];
        }
        int64_t actualKIdx =  kIdx - curSwitch;
        int64_t iC = actualKIdx;

        auto mte1MadEventId = FlagL0APingPong ? enventIdA : enventIdB;
        auto eventIdCube3 = FlagL1APingPong ? enventIdA : enventIdB;

        auto l0a_buf_tensor = FlagL0APingPong ? l0_a_base_tensor : l0_a_base_tensor[L0AB_PINGPONG_BUFFER_LEN];
        auto l1b_buf_tensor = FlagL0APingPong ? l1_base_a_cube2_tensor : l1_base_a_cube2_tensor[L1_PINGPONG_BUFFER_LEN];

        if (srcLen >= 2 && curSwitchIndex >= 1 && kIdx == curSwitch) {
            commonNd2NzParams.nValue = BASE_LEN;
            commonNd2NzParams.dValue = BASE_LEN;
            commonNd2NzParams.srcDValue = BASE_LEN;
            commonNd2NzParams.srcNdMatrixStride = 0;
            commonNd2NzParams.dstNzC0Stride = BASE_LEN;
            commonNd2NzParams.dstNzMatrixStride = 0;
            temp_tensor_bf16.SetGlobalBuffer(
                reinterpret_cast<__gm__ TYPE *>(src[curSwitchIndex].right));
            AscendC::DataCopy(
                l1_base_b_cube2_tensor[(curSwitchIndex % 2) * BASE_LEN * BASE_LEN],
                temp_tensor_bf16,
                commonNd2NzParams
            );

            SET_FLAG(MTE2, MTE1, EVENT_ID6);
            WAIT_FLAG(MTE2, MTE1, EVENT_ID6);

            if (src[curSwitchIndex-1].k == 1) {
                SET_FLAG(M, MTE1, EVENT_ID6);
                WAIT_FLAG(M, MTE1, EVENT_ID6);
            }

            for (int64_t i = 0; i < BASE_LEN / BLOCK_SIZE; i++) {
                commonLoadData2dParamsTranspose.repeatTimes = BASE_LEN / BLOCK_SIZE;
                commonLoadData2dParamsTranspose.srcStride = BASE_LEN / BLOCK_SIZE;
                AscendC::LoadData(
                    l0_b_base_tensor[(curSwitchIndex % 2) * BASE_LEN * BASE_LEN + i * BASE_LEN * BLOCK_SIZE],
                    l1_base_b_cube2_tensor[(curSwitchIndex % 2) * BASE_LEN * BASE_LEN +  i * CUBE_MATRIX_SIZE],
                    commonLoadData2dParamsTranspose
                );
            }
        }

        WAIT_FLAG(MTE1, MTE2, eventIdCube3);

        int64_t localOffsetA =  actualKIdx * BASE_LEN * BASE_LEN;
        temp_tensor_bf16.SetGlobalBuffer(
                reinterpret_cast<__gm__ TYPE *>(src[curSwitchIndex].left + localOffsetA));
        AscendC::DataCopy(
            l1b_buf_tensor,
            temp_tensor_bf16,
            AscendC::Nd2NzParams(
                vcoreNumPerHead,
                BASE_LEN / vcoreNumPerHead,
                BASE_LEN,
                BASE_LEN / vcoreNumPerHead * BASE_LEN * 2,
                BASE_LEN,
                BASE_LEN,
                1,
                BASE_LEN * BLOCK_SIZE / vcoreNumPerHead
            )
        );

        SET_FLAG(MTE2, MTE1, eventIdCube3);
        WAIT_FLAG(MTE2, MTE1, eventIdCube3);
        WAIT_FLAG(M, MTE1, mte1MadEventId);

        for (int64_t i = 0; i < BASE_LEN / BLOCK_SIZE; i++) {
            commonLoadData2dParamsTranspose.repeatTimes = BASE_LEN / BLOCK_SIZE;
            commonLoadData2dParamsTranspose.srcStride = SRC_STRIDE_1;
            AscendC::LoadData(
                l0a_buf_tensor[i * BASE_LEN * BLOCK_SIZE],
                l1b_buf_tensor[i * BASE_LEN * BLOCK_SIZE],
                commonLoadData2dParamsTranspose
            );
        }

        SET_FLAG(MTE1, MTE2, eventIdCube3);
        SET_FLAG(MTE1, M, mte1MadEventId);   // mte1MadEventId
        WAIT_FLAG(MTE1, M, mte1MadEventId);  // mte1MadEventId

        int unit_flag = 0b11;                     // allow reading from L0C when finish mad
        auto l0b_buf_tensor = l0_b_base_tensor[(curSwitchIndex % 2) * BASE_LEN * BASE_LEN];
        auto l0c_buf_tensor = FlagL0CPingPong ? l0_c_base_tensor : l0_c_base_tensor[L0C_PINGPONG_BUFFER_LEN];
        commonMadParams.unitFlag = unit_flag;
        commonMadParams.cmatrixInitVal = true;
        AscendC::Mmad(
            l0c_buf_tensor,
            l0a_buf_tensor,
            l0b_buf_tensor,
            commonMadParams
        );

        SET_FLAG(M, MTE1, mte1MadEventId); // mte1MadEventId

        AscendC::SetAtomicAdd<float>();
        AscendC::SetAtomicType<float>();
        int64_t localOffsetGmC = actualKIdx  * BASE_LEN * BASE_LEN;
        commonFixpipeParamsV220.unitFlag = unit_flag;
        temp_tensor_fp32.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(src[curSwitchIndex].out  + localOffsetGmC));
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
            temp_tensor_fp32,
            l0c_buf_tensor,
            commonFixpipeParamsV220
        );
        AscendC::SetAtomicNone();

        FlagL0APingPong = 1 - FlagL0APingPong;
        FlagL1APingPong = 1 - FlagL1APingPong;
        FlagL0CPingPong = 1 - FlagL0CPingPong ;
    }

    WAIT_FLAG(M, MTE1, enventIdA);
    WAIT_FLAG(M, MTE1, enventIdB);
    WAIT_FLAG(MTE1, MTE2, enventIdA);
    WAIT_FLAG(MTE1, MTE2, enventIdB);
}

template <typename TYPE, bool IF_BF16>
__aicore__ __inline__ void CubeBackward<TYPE, IF_BF16>::addrMappingPre(
                                Addr *addr, int64_t& curAddrLen, int64_t roundId) {
    int64_t blocksPerBatch = this->headNum * blocksPerRow * blocksPerColumn;
    int64_t allBlocks = this->batchSize * blocksPerBatch;
    int64_t curBlockId = this->blockNumPerCore * this->coreNum * roundId + curCoreIndex * this->blockNumPerCore;

    int64_t remain = this->blockNumPerCore;
    if (curBlockId + this->blockNumPerCore  > allBlocks) {
        remain = allBlocks - curBlockId;
    }

    int64_t b = curBlockId / blocksPerBatch;

    int64_t n = (curBlockId % blocksPerBatch) / (blocksPerRow * blocksPerColumn);

    int64_t ir = (curBlockId % blocksPerBatch) % (blocksPerRow * blocksPerColumn) / (blocksPerRow);

    int64_t ic = (curBlockId % blocksPerBatch) % (blocksPerRow * blocksPerColumn) % (blocksPerRow);
    int64_t rows = (ic + remain + blocksPerRow - 1) / blocksPerRow;

    addr[0].b = b;
    addr[0].n = n;
    addr[0].iR = ir;
    addr[0].iC = ic;
    addr[0].k = remain;

    for (int i = 0; i < rows && remain > 0; i++) {
        if (addr[curAddrLen].iC + remain > blocksPerRow) {
            addr[curAddrLen].k = blocksPerRow - addr[curAddrLen].iC;
            addr[curAddrLen + 1].k = remain - addr[curAddrLen].k;
            addr[curAddrLen + 1].b = addr[curAddrLen].b;
            addr[curAddrLen + 1].n = addr[curAddrLen].n;
            addr[curAddrLen + 1].iR = addr[curAddrLen].iR + 1;
            addr[curAddrLen + 1].iC = 0;
            if (addr[curAddrLen + 1].iR >= blocksPerColumn) {
                addr[curAddrLen + 1].n = addr[curAddrLen].n + 1;
                addr[curAddrLen + 1].iR = 0;
                if (addr[curAddrLen + 1].n >= this->headNum) {
                    addr[curAddrLen + 1].b = addr[curAddrLen].b + 1;
                    addr[curAddrLen + 1].n = 0;
                }
            }
        }
        remain -= addr[curAddrLen].k;
        ++curAddrLen;
    }
}

template <typename TYPE, bool IF_BF16>
__aicore__ __inline__ void CubeBackward<TYPE, IF_BF16>::addrMapping_cube1(
    int64_t roundId, PhyAddrCube1<TYPE, WORKSPACE_DTYPE> *src1,
    PhyAddrCube1<TYPE, WORKSPACE_DTYPE_DP> *src2, int *srcLength) {

    int64_t curAddrLen = 0;
    Addr addr[MAX_SWITCH_TIME];
    addrMappingPre(addr, curAddrLen, roundId);

    AddrToPhyAddr(addr, curAddrLen, gmQ, gmK, gmS, src1, srcLength, roundId);
    AddrToPhyAddr(addr, curAddrLen, gmDO, gmV, gm_dP, src2, srcLength, roundId);
}

template <typename TYPE, bool IF_BF16>
template <typename T>
__aicore__ __inline__ void CubeBackward<TYPE, IF_BF16>::AddrToPhyAddr(
    Addr *addr, int length, __gm__ TYPE * gm_a, __gm__ TYPE * gm_b, __gm__ T * gm_c,
    PhyAddrCube1<TYPE, T> *src, int *srcLength, int64_t roundId)
{
    int index = 0;
    for (int i = 0 ; i < length; i++) {
        int64_t iR = addr[i].iR;
        int64_t iC = addr[i].iC;
        int64_t k = addr[i].k;
        int64_t switchIndex = this->isTri ? blocksPerColumn + iR : blocksPerRow;
        int64_t rowIndexLeftSection = this->isTri ? blocksPerColumn + iR :iR;
        int64_t rowIndexRightSection = this->isTri ? blocksPerColumn - 1 - iR : iR;

        int64_t colIndexLeftSection = iC;
        int64_t colIndexRightSection =  this->isTri ?  (iC - switchIndex - 1) : iC;

        int64_t gIndex = addr[i].n / (this->headNum / this->gqaGroupNum);
        int64_t bnOffsetGqaRightMatrix = (addr[i].b * this->gqaGroupNum + gIndex) * this->seqLenK *
                                         this->headDim; // for gqa mode
        int64_t bnOffsetRightMatrix = (addr[i].b * headNum  + addr[i].n  ) * this->seqLenK * this->headDim;
        int64_t bnOffsetLeftMatrix =  (addr[i].b * headNum  + addr[i].n  ) * this->seqLenQ * this->headDim;

        int64_t qLeftOffsetSection =   rowIndexLeftSection * BASE_BLOCK_SIZE;
        int64_t qRightOffsetSection =  rowIndexRightSection * BASE_BLOCK_SIZE;
        int64_t kLeftOffsetSection =  colIndexLeftSection * BASE_BLOCK_SIZE;
        int64_t kRightOffsetSection =  colIndexRightSection * BASE_BLOCK_SIZE;

        // sparse场景：isTri == true 以及 sparseMode == 1
        bool sparseFlag = false;
        int64_t windowBlockSize = this->windowLength / BASE_LEN;
        if (this->isTri && this->sparseMode == 1) {
            sparseFlag = iR > ((windowBlockSize -1) / 2) ? true: false;
            switchIndex = (windowBlockSize / 2)+ iR;
            rowIndexLeftSection = (windowBlockSize / 2) + iR;
            rowIndexRightSection = (windowBlockSize / 2) - 1 - iR;
            colIndexLeftSection = iC;
            colIndexRightSection = iC - switchIndex - 1;
            qLeftOffsetSection =   rowIndexLeftSection * BASE_BLOCK_SIZE;
            qRightOffsetSection =  rowIndexRightSection * BASE_BLOCK_SIZE;
            kLeftOffsetSection =  colIndexLeftSection * BASE_BLOCK_SIZE;
            kRightOffsetSection =  colIndexRightSection * BASE_BLOCK_SIZE;
        }
        int64_t rowIndexSparseSection = iR + (windowBlockSize / 2);
        int64_t colIndexSparseSection = iR + iC - (windowBlockSize / 2);
        int64_t qSparseOffsetSection = rowIndexSparseSection * BASE_BLOCK_SIZE;
        int64_t kSparseOffsetSection = colIndexSparseSection * BASE_BLOCK_SIZE;

        int64_t outOffset = ((addr[i].b * headNum + addr[i].n) * blocksPerRow * blocksPerColumn +
                            (iR * blocksPerRow)) * BASE_BLOCK_SIZE;

        int64_t dbOffset = (roundId % 2) * (coreNum * blockNumPerCore * BASE_BLOCK_SIZE);
        if (index == 0) {
            src[index].out = gm_c + dbOffset + curCoreIndex * blockNumPerCore * BASE_BLOCK_SIZE;
        } else {
            src[index].out = src[index - 1].out + src[index - 1].k * BASE_BLOCK_SIZE;
        }

        if (!sparseFlag && switchIndex < iC) {
            src[index].left = gm_a + bnOffsetLeftMatrix  + qRightOffsetSection;
            src[index].right = gm_b + bnOffsetGqaRightMatrix + kRightOffsetSection;
            src[index].k = k;
            index++;
        } else if (!sparseFlag && iC <= switchIndex && iC + k - 1 > switchIndex) {
            src[index].left = gm_a + bnOffsetLeftMatrix + qLeftOffsetSection;
            src[index].right = gm_b + bnOffsetGqaRightMatrix + kLeftOffsetSection;
            src[index].k = switchIndex - iC + 1;

            src[index + 1].left = gm_a + bnOffsetLeftMatrix + qRightOffsetSection;
            src[index + 1].right = gm_b + bnOffsetGqaRightMatrix;
            src[index + 1].out = src[index].out + src[index].k * BASE_BLOCK_SIZE;
            src[index + 1].k = k - src[index].k;
            index += 2;
        } else if (!sparseFlag && iC <= switchIndex && iC + k - 1 <= switchIndex) {
            src[index].left = gm_a + bnOffsetLeftMatrix + qLeftOffsetSection;
            src[index].right = gm_b + bnOffsetGqaRightMatrix + kLeftOffsetSection;
            src[index].k = k;
            index++;
        } else {
            src[index].left = gm_a + bnOffsetLeftMatrix + qSparseOffsetSection;
            src[index].right = gm_b + bnOffsetGqaRightMatrix + kSparseOffsetSection;
            src[index].k = k;
            index++;
        }
    }

    *srcLength = index;
}

template <typename TYPE, bool IF_BF16>
template <typename T> __aicore__ __inline__ void  CubeBackward<TYPE, IF_BF16>::addrMapping_cube2(
        __gm__ T *__restrict__ left,
        __gm__ TYPE *__restrict__ right,
        __gm__ float *__restrict__ out,
        int64_t roundId,
        PhyAddrCube2<TYPE, T> *src,
        int64_t &len) {
    int64_t curAddrLen = 0;
    Addr addr[MAX_SWITCH_TIME];
    addrMappingPre(addr, curAddrLen, roundId);
    auto BASE_BLOCK_SIZE = this->headDim * this->headDim;

    int64_t index = 0;
    for (int i = 0; i < curAddrLen; ++i) {
        int64_t iR = addr[i].iR;
        int64_t iC = addr[i].iC;
        int64_t k = addr[i].k;
        int64_t switchIndex = this->isTri ? blocksPerColumn + iR : blocksPerRow;
        int64_t rowIndexLeftSection = this->isTri ? blocksPerColumn + iR :iR;
        int64_t rowIndexRightSection = this->isTri ? blocksPerColumn - 1 - iR : iR;
        int64_t colIndexLeftSection = iC;
        int64_t colIndexRightSection =  this->isTri ?  (iC - switchIndex - 1) : iC;
        int64_t gIndex = addr[i].n / (this->headNum / this->gqaGroupNum);
        int64_t bnOffsetGqaRightMatrix = (addr[i].b * this->gqaGroupNum + gIndex) * this->seqLenK * this->headDim;
        int64_t bnOffsetLeftMatrix = ((addr[i].b * this->headNum + addr[i].n) * blocksPerRow * blocksPerColumn) *
                                     BASE_BLOCK_SIZE;
        int64_t bnOffsetRightMatrix = (addr[i].b * this->headNum + addr[i].n) * this->seqLenK * this->headDim;
        int64_t bn_offset_out = (addr[i].b * this->headNum + addr[i].n) * this->seqLenQ * this->headDim;

        bool sparseFlag = false;
        int64_t windowBlockSize = this->windowLength / BASE_LEN;
        if (this->isTri && this->sparseMode == 1) {
            sparseFlag = iR > ((windowBlockSize -1) / 2) ? true: false;
            switchIndex = (windowBlockSize / 2)+ iR;
            rowIndexLeftSection = (windowBlockSize / 2) + iR;
            rowIndexRightSection = (windowBlockSize / 2) - 1 - iR;
            colIndexLeftSection = iC;
            colIndexRightSection = iC - switchIndex - 1;
        }
        int64_t rowIndexSparseSection = iR + (windowBlockSize / 2);
        int64_t colIndexSparseSection = iR + iC - (windowBlockSize / 2);
        int64_t dbOffset = (roundId % 2) * (coreNum * blockNumPerCore * BASE_BLOCK_SIZE);
        if (index == 0) {
            src[index].left = left + dbOffset + curCoreIndex * blockNumPerCore * BASE_BLOCK_SIZE;
        } else {
            src[index].left = src[index - 1].left + src[index - 1].k * BASE_BLOCK_SIZE;
        }

        if (!sparseFlag && switchIndex < iC) {
            src[index].right = right + bnOffsetGqaRightMatrix + colIndexRightSection * BASE_BLOCK_SIZE;
            src[index].out = out + bn_offset_out + rowIndexRightSection * BASE_BLOCK_SIZE;
            src[index].k = k;
            ++index;
        } else if (!sparseFlag && iC <= switchIndex && iC + k -1 > switchIndex) {
            src[index].right = right + bnOffsetGqaRightMatrix + colIndexLeftSection * BASE_BLOCK_SIZE;
            src[index].out = out + bn_offset_out + rowIndexLeftSection * BASE_BLOCK_SIZE;
            src[index].k = switchIndex - iC + 1;

            src[index + 1].left = src[index].left + src[index].k * BASE_BLOCK_SIZE;
            src[index + 1].right = right + bnOffsetGqaRightMatrix;
            src[index + 1].out = out + bn_offset_out + rowIndexRightSection * BASE_BLOCK_SIZE;
            src[index + 1].k = k - src[index].k;
            index += 2;
        } else if (!sparseFlag && iC <= switchIndex && iC + k -1 <= switchIndex) {
            src[index].right = right + bnOffsetGqaRightMatrix + colIndexLeftSection * BASE_BLOCK_SIZE;
            src[index].out = out + bn_offset_out + rowIndexLeftSection * BASE_BLOCK_SIZE;
            src[index].k = k;
            index++;
        } else {
            src[index].right = right + bnOffsetGqaRightMatrix + colIndexSparseSection * BASE_BLOCK_SIZE;
            src[index].out = out + bn_offset_out + rowIndexSparseSection * BASE_BLOCK_SIZE;
            src[index].k = k;
            ++index;
        }
    }
    len = index;
}

template <typename TYPE, bool IF_BF16>
template <typename T>
__aicore__ __inline__ void CubeBackward<TYPE, IF_BF16>::addrMapping_cube3(
    __gm__ T * __restrict__ left, __gm__ TYPE * __restrict__ right, __gm__ T_OUTPUT * __restrict__ out, int64_t roundId,
    PhyAddrCube3<TYPE, T>* src, int64_t& len) {

    int64_t curAddrLen = 0;
    Addr addr[MAX_SWITCH_TIME];
    addrMappingPre(addr, curAddrLen, roundId);
    auto BASE_BLOCK_SIZE =  this->headDim * this->headDim;

    int64_t index = 0;
    for (int i = 0; i < curAddrLen; ++i) {
        int64_t iR = addr[i].iR;
        int64_t iC = addr[i].iC;
        int64_t k = addr[i].k;

        int64_t switchIndex = this->isTri ? blocksPerColumn + iR : blocksPerRow;
        int64_t rowIndexLeftSection = this->isTri ? blocksPerColumn + iR :iR;
        int64_t rowIndexRightSection = this->isTri ? blocksPerColumn - 1 - iR : iR;
        int64_t colIndexLeftSection = iC;
        int64_t colIndexRightSection =  this->isTri ?  (iC - switchIndex - 1) : iC;

        int64_t bnOffsetLeftMatrix = ((addr[i].b * this->headNum + addr[i].n) * blocksPerRow * blocksPerColumn) *
                                     BASE_BLOCK_SIZE;  // 当前b,n下的偏移量
        int64_t bnOffsetRightMatrix = (addr[i].b * this->headNum + addr[i].n) * this->seqLenQ *
                                      this->headDim;   // 当前b,n下的偏移量
        int64_t bn_offset_out = (addr[i].b * this->headNum + addr[i].n) * this->seqLenK *
                                this->headDim;         // 当前b,n下的偏移量
        int64_t gIndex = addr[i].n / (this->headNum / this->gqaGroupNum);
        int64_t bn_offset_gqa_out_matrix = (addr[i].b * this->gqaGroupNum + gIndex) * this->seqLenK * this->headDim;
        bool sparseFlag = false;
        int64_t windowBlockSize = this->windowLength / BASE_LEN;
        if (this->isTri && this->sparseMode == 1) {
            sparseFlag = iR > ((windowBlockSize -1) / 2) ? true: false;
            switchIndex = (windowBlockSize / 2)+ iR;
            rowIndexLeftSection = (windowBlockSize / 2) + iR;
            rowIndexRightSection = (windowBlockSize / 2) - 1 - iR;
            colIndexLeftSection = iC;
            colIndexRightSection = iC - switchIndex - 1;
        }
        int64_t rowIndexSparseSection = iR + (windowBlockSize / 2);
        int64_t colIndexSparseSection = iR + iC - (windowBlockSize / 2);
        int64_t dbOffset = (roundId % 2) * (coreNum * blockNumPerCore * BASE_BLOCK_SIZE);

        if (index == 0) {
            src[index].left = left + dbOffset + curCoreIndex * blockNumPerCore * BASE_BLOCK_SIZE;
        } else {
            src[index].left = src[index - 1].left + src[index - 1].k * BASE_BLOCK_SIZE;
        }

        if (!sparseFlag && switchIndex < iC) {
            src[index].right = right + bnOffsetRightMatrix + rowIndexRightSection * BASE_BLOCK_SIZE;
            src[index].out = out + bn_offset_gqa_out_matrix + colIndexRightSection * BASE_BLOCK_SIZE;
            src[index].k = k;
            ++index;
        } else if (!sparseFlag && iC <= switchIndex && iC + k -1 > switchIndex) {
            src[index].right = right + bnOffsetRightMatrix + rowIndexLeftSection * BASE_BLOCK_SIZE;
            src[index].out = out + bn_offset_gqa_out_matrix + colIndexLeftSection * BASE_BLOCK_SIZE;
            src[index].k = switchIndex - iC + 1;

            src[index + 1].left = src[index].left + src[index].k * BASE_BLOCK_SIZE;
            src[index + 1].right = right + bnOffsetRightMatrix + rowIndexRightSection * BASE_BLOCK_SIZE;
            src[index + 1].out = out + bn_offset_gqa_out_matrix;
            src[index + 1].k = k - src[index].k;
            index += 2;
        } else if (!sparseFlag && iC <= switchIndex && iC + k -1 <= switchIndex) {
            src[index].right = right + bnOffsetRightMatrix + rowIndexLeftSection * BASE_BLOCK_SIZE;
            src[index].out = out + bn_offset_gqa_out_matrix + colIndexLeftSection * BASE_BLOCK_SIZE;
            src[index].k = k;
            index++;
        } else {
            src[index].right = right + bnOffsetRightMatrix + rowIndexSparseSection * BASE_BLOCK_SIZE;
            src[index].out = out + bn_offset_gqa_out_matrix + colIndexSparseSection * BASE_BLOCK_SIZE;
            src[index].k = k;
            ++index;
        }
    }
    len = index;
}

template <typename TYPE, bool IF_BF16>
__aicore__ inline void CubeBackward<TYPE, IF_BF16>::Mat_mix_cube2_cube3(int64_t roundId) {

    PhyAddrCube3<TYPE, WORKSPACE_DTYPE_DP> src_dK[MAX_SWITCH_TIME];
    int64_t switchIndex = 0;
    addrMapping_cube3(this->gm_dP, this->gmQ, this->gmDK, roundId, src_dK, switchIndex);

    PhyAddrCube2<TYPE, WORKSPACE_DTYPE_DP> src_cube2[MAX_SWITCH_TIME];
    int64_t src_length_cube2 = 0;
    addrMapping_cube2(this->gm_dP, this->gmK, this->gmDQ, roundId, src_cube2, src_length_cube2);

    cube2_cube3_mix(src_cube2, src_length_cube2, src_dK, switchIndex, 2);
}

template <typename TYPE, bool IF_BF16>
__aicore__ inline void CubeBackward<TYPE, IF_BF16>::Run_mix() {

    SetPadding(0);
    uint64_t config = 0x1;
    AscendC::SetNdParaImpl(config);
    int64_t continuedBlocksNum = 2;

    int64_t blocksPerBatch = headNum * blocksPerRow * blocksPerColumn;
    int64_t allBlocks = batchSize * blocksPerBatch ;
    int64_t Z = (allBlocks + blockNumPerCore * coreNum - 1) / (blockNumPerCore * coreNum);

    int64_t switchIndex = 0;
    int64_t src_length_cube2 = 0;
    int64_t remain_length_cube2 = 0;
    uint64_t mode;
    uint64_t sync_config;

    if (Z == 1) {
        if (curCoreIndex * blockNumPerCore < allBlocks) {
            cube1(0);
        }
        FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);

        WaitFlagDev(AIV2AICFLAGID);

        if (curCoreIndex * blockNumPerCore < allBlocks) {
            PhyAddrCube3<TYPE, WORKSPACE_DTYPE> srcdV[MAX_SWITCH_TIME];
            switchIndex = 0;
            addrMapping_cube3(this->gmS, this->gmDO, this->gmDV, 0, srcdV, switchIndex);
            cube3_matmul(srcdV, switchIndex, EVENT_ID3, EVENT_ID4, continuedBlocksNum);
            Mat_mix_cube2_cube3(0);
        }
    } else {
        for (int64_t roundId = 0; roundId < 2; roundId++) {
            if (roundId * blockNumPerCore * coreNum + curCoreIndex * blockNumPerCore < allBlocks && isCube1) {
                cube1(roundId);
            }
            if (isSyn) {
                if (roundId == 0) {
                    FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
                }
            }
        }

        for (int64_t roundId =2; roundId < Z; roundId++) {
            if (isSyn) {
                WaitFlagDev(AIV2AICFLAGID);
                FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
            }
            if ((roundId - 2) * blockNumPerCore * coreNum + curCoreIndex * blockNumPerCore < allBlocks) {
                if (isCube3) {
                    PhyAddrCube3<TYPE, WORKSPACE_DTYPE> srcdV[MAX_SWITCH_TIME];
                    switchIndex = 0;
                    addrMapping_cube3(this->gmS, this->gmDO, this->gmDV, roundId - 2, srcdV, switchIndex);
                    cube3_matmul(srcdV, switchIndex, EVENT_ID3, EVENT_ID4, continuedBlocksNum);
                    Mat_mix_cube2_cube3(roundId - 2);
                }
            }
            if (roundId * blockNumPerCore * coreNum + curCoreIndex * blockNumPerCore < allBlocks && isCube1) {
                cube1(roundId);
            }
        }

        for (int64_t roundId = 0; roundId < 2; roundId++) {
            if (isSyn) {
                WaitFlagDev(AIV2AICFLAGID);
                if (roundId == 0) {
                    FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
                }
            }
            if ((roundId + Z - 2) * blockNumPerCore * coreNum + curCoreIndex * blockNumPerCore < allBlocks) {
                if (isCube3) {
                    PhyAddrCube3<TYPE, WORKSPACE_DTYPE> srcdV[MAX_SWITCH_TIME];
                    switchIndex = 0;
                    addrMapping_cube3(this->gmS, this->gmDO, this->gmDV, roundId+Z-2, srcdV, switchIndex);
                    cube3_matmul(srcdV, switchIndex, EVENT_ID3, EVENT_ID4, continuedBlocksNum);
                    Mat_mix_cube2_cube3(roundId + Z - 2);
                }
            }
        }
    }
}

template <typename TYPE, bool IF_BF16>
template <typename T>
__aicore__ __inline__ void CubeBackward<TYPE, IF_BF16>::cube2_cube3_mix(
    const PhyAddrCube2<TYPE, T> *cube2_addr, int64_t cube2_length, const PhyAddrCube3<TYPE, T> *cube3_addr,
    int64_t cube3_length, int64_t vcoreNumPerHead) {

    if (cube2_length == 0 || cube3_length == 0) return;

    auto l0a_buf_cube3 = l0_a_base_tensor;
    auto l0b_buf_cube3 = l0_b_base_tensor;
    auto l0c_buf_cube3 = l0_c_base_tensor;

    auto l0a_buf_cube2 = l0_a_base_tensor[BASE_BLOCK];
    auto l0b_buf_cube2 = l0_b_base_tensor[BASE_BLOCK];
    auto l0c_buf_cube2 = l0_c_base_tensor[BASE_BLOCK];

    int64_t switchIndexList[MAX_SWITCH_TIME] = {0};
    int64_t totalLen = 0;

    for (int64_t srcIndex = 0; srcIndex < cube2_length; srcIndex ++) {
        switchIndexList[srcIndex] = totalLen;
        totalLen += cube2_addr[srcIndex].k;
    }

    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);
    SET_FLAG(MTE1, MTE2, EVENT_ID4);
    SET_FLAG(MTE1, MTE2, EVENT_ID5);
    SET_FLAG(MTE1, MTE2, EVENT_ID6);
    SET_FLAG(MTE1, MTE2, EVENT_ID7);

    commonNd2NzParams.nValue = BASE_LEN;
    commonNd2NzParams.dValue = BASE_LEN;
    commonNd2NzParams.srcDValue = BASE_LEN;
    commonNd2NzParams.srcNdMatrixStride = 0;
    commonNd2NzParams.dstNzC0Stride = BASE_LEN;
    commonNd2NzParams.dstNzMatrixStride = 0;
    temp_tensor_bf16.SetGlobalBuffer(
        reinterpret_cast<__gm__ TYPE *>(cube3_addr[0].right));
    AscendC::DataCopy(
        l1_base_b_cube1_tensor,
        temp_tensor_bf16,
        commonNd2NzParams
    );

    SET_FLAG(MTE2, MTE1, EVENT_ID0);
    WAIT_FLAG(MTE2, MTE1, EVENT_ID0);

    for (int64_t i = 0; i < BASE_LEN / BLOCK_SIZE; i++) {
        commonLoadData2dParamsTranspose.repeatTimes = BASE_LEN / BLOCK_SIZE;
        commonLoadData2dParamsTranspose.srcStride = BASE_LEN / BLOCK_SIZE;
        AscendC::LoadData(
            l0b_buf_cube3[i * BASE_LEN * BLOCK_SIZE],
            l1_base_b_cube1_tensor[i * CUBE_MATRIX_SIZE],
            commonLoadData2dParamsTranspose
        );
    }

    int64_t pingFlagCube2 = 1;
    int64_t pingFlagCube3 = 1;
    int64_t mte1MadPingFlag = 1;

    SET_FLAG(M, MTE1, EVENT_ID0);
    SET_FLAG(M, MTE1, EVENT_ID1);
    SET_FLAG(M, MTE1, EVENT_ID2);
    SET_FLAG(M, MTE1, EVENT_ID3);
    SET_FLAG(M, MTE1, EVENT_ID4);
    SET_FLAG(M, MTE1, EVENT_ID5);
    SET_FLAG(M, MTE1, EVENT_ID6);
    SET_FLAG(M, MTE1, EVENT_ID7);

    int64_t curSwitchIndex = 0;
    int64_t curSwitch  = switchIndexList[curSwitchIndex];
    int unit_flag;
    for (int64_t kIdx = 0; kIdx < totalLen; kIdx ++) {

        if ((cube2_length >= 2 && kIdx == switchIndexList[curSwitchIndex+1])) {
            curSwitchIndex ++;
            curSwitch = switchIndexList[curSwitchIndex];
        }

        int64_t actualKIdx = kIdx - curSwitch;
        int64_t iC = actualKIdx;

        pingFlagCube2 = 1 - pingFlagCube2;
        pingFlagCube3 = 1 - pingFlagCube3;
        mte1MadPingFlag = 1 - mte1MadPingFlag;
        auto l1BufACube2 = pingFlagCube2 ? l1_base_a_cube2_tensor : l1_base_a_cube2_tensor[L1_PINGPONG_BUFFER_LEN];
        auto l1BufBCube2 = pingFlagCube2 ? l1_base_b_cube2_tensor : l1_base_b_cube2_tensor[L1_PINGPONG_BUFFER_LEN];

        auto eventIdCube2 = pingFlagCube2 ? EVENT_ID4 : EVENT_ID7;
        auto eventIdCube3 = pingFlagCube3 ? EVENT_ID3 : EVENT_ID6;

        auto mte1Cube2MadEventId = mte1MadPingFlag ? EVENT_ID4 : EVENT_ID3;
        auto mte1Cube3MadEventId = mte1MadPingFlag ? EVENT_ID3 : EVENT_ID4;

        WAIT_FLAG(MTE1, MTE2, eventIdCube3);
        int64_t localOffsetA =  actualKIdx * BASE_LEN * BASE_LEN;
        temp_tensor_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(
            cube3_addr[curSwitchIndex].left + localOffsetA));
        AscendC::DataCopy(
            l1BufACube2,
            temp_tensor_bf16,
            AscendC::Nd2NzParams(
                vcoreNumPerHead,
                BASE_LEN / vcoreNumPerHead,
                BASE_LEN,
                BASE_LEN / vcoreNumPerHead * SIZE_128 * 2,
                BASE_LEN,
                BASE_LEN,
                1,
                BASE_LEN * BLOCK_SIZE / vcoreNumPerHead
            )
        );

        SET_FLAG(MTE2, MTE1, eventIdCube3);
        WAIT_FLAG(MTE2, MTE1, eventIdCube3);
        WAIT_FLAG(M, MTE1, mte1Cube3MadEventId);

        for (int64_t i = 0; i < BASE_LEN / BLOCK_SIZE; i++) {
            commonLoadData2dParamsTranspose.repeatTimes = BASE_LEN / BLOCK_SIZE;
            commonLoadData2dParamsTranspose.srcStride = SRC_STRIDE_1;
            AscendC::LoadData(
                l0a_buf_cube3[i * BASE_LEN * BLOCK_SIZE],
                l1BufACube2[i * BASE_LEN * BLOCK_SIZE],
                commonLoadData2dParamsTranspose
            );
        }

        commonNd2NzParams.nValue = BASE_LEN;
        commonNd2NzParams.dValue = BASE_LEN;
        commonNd2NzParams.srcDValue = BASE_LEN;
        commonNd2NzParams.srcNdMatrixStride = 0;
        commonNd2NzParams.dstNzC0Stride = BASE_LEN;
        commonNd2NzParams.dstNzMatrixStride = 0;
        if (cube2_length >= 2 && curSwitchIndex >= 1 && kIdx == curSwitch) {
            temp_tensor_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(cube3_addr[curSwitchIndex].right));
            AscendC::DataCopy(
                l1_base_b_cube1_tensor[(curSwitchIndex % 2) * BASE_LEN * BASE_LEN],
                temp_tensor_bf16,
                commonNd2NzParams
            );

            SET_FLAG(MTE2, MTE1, eventIdCube3);
            WAIT_FLAG(MTE2, MTE1, eventIdCube3);

            for (int64_t i = 0; i < BASE_LEN / BLOCK_SIZE; i++) {
                commonLoadData2dParamsTranspose.repeatTimes = BASE_LEN / BLOCK_SIZE;
                commonLoadData2dParamsTranspose.srcStride = BASE_LEN / BLOCK_SIZE;
                AscendC::LoadData(
                    l0b_buf_cube3[i * BASE_LEN * BLOCK_SIZE],
                    l1_base_b_cube1_tensor[(curSwitchIndex % 2) * BASE_BLOCK +  i * CUBE_MATRIX_SIZE],
                    commonLoadData2dParamsTranspose
                );
            }
        }

        SET_FLAG(MTE1, M, mte1Cube3MadEventId);  // mte1MadEventId
        WAIT_FLAG(MTE1, M, mte1Cube3MadEventId); // mte1MadEventId

        unit_flag = 0b11;
        commonMadParams.unitFlag = unit_flag;
        commonMadParams.cmatrixInitVal = true;
        AscendC::Mmad(
            l0c_buf_cube3,
            l0a_buf_cube3,
            l0b_buf_cube3,
            commonMadParams
        );

        SET_FLAG(M, MTE1, mte1Cube3MadEventId); // mte1MadEventId

        AscendC::SetAtomicAdd<float>();
        AscendC::SetAtomicType<float>();
        int64_t localOffsetGmC = actualKIdx * BASE_LEN * BASE_LEN;
        commonFixpipeParamsV220.unitFlag = unit_flag;
        temp_tensor_fp32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            cube3_addr[curSwitchIndex].out  + localOffsetGmC));
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
            temp_tensor_fp32,
            l0c_buf_cube3,
            commonFixpipeParamsV220
        );
        AscendC::SetAtomicNone();

        // 计算cube2
        if (true) {
            for (int64_t i = 0; i < BASE_LEN / BLOCK_SIZE; i++) {
                commonLoadData2dParamsNoTranspose.repeatTimes = BASE_LEN / BLOCK_SIZE;
                commonLoadData2dParamsNoTranspose.srcStride = BASE_LEN / BLOCK_SIZE;
                AscendC::LoadData(
                    l0a_buf_cube2[i * BASE_LEN * BLOCK_SIZE],
                    l1BufACube2[i * CUBE_MATRIX_SIZE],
                    commonLoadData2dParamsNoTranspose
                );
            }

            SET_FLAG(MTE1, MTE2, eventIdCube3);
            WAIT_FLAG(MTE1, MTE2, eventIdCube2);

            commonNd2NzParams.nValue = BASE_LEN;
            commonNd2NzParams.dValue = BASE_LEN;
            commonNd2NzParams.srcDValue = BASE_LEN;
            commonNd2NzParams.srcNdMatrixStride = BASE_LEN * BASE_LEN;
            commonNd2NzParams.dstNzMatrixStride = BASE_LEN * BASE_LEN;
            commonNd2NzParams.dstNzC0Stride = BASE_LEN;
            temp_tensor_bf16.SetGlobalBuffer(
            reinterpret_cast<__gm__ TYPE *>(cube2_addr[curSwitchIndex].right + actualKIdx * BASE_LEN * BASE_LEN));
            AscendC::DataCopy(
                l1BufBCube2,
                temp_tensor_bf16,
                commonNd2NzParams
            );

            SET_FLAG(MTE2, MTE1, eventIdCube2);
            WAIT_FLAG(M, MTE1, mte1Cube2MadEventId);
            WAIT_FLAG(MTE2, MTE1, eventIdCube2);

            for (int64_t i = 0; i < BASE_LEN / BLOCK_SIZE; i++) {
                commonLoadData2dParamsTranspose.repeatTimes = BASE_LEN / BLOCK_SIZE;
                commonLoadData2dParamsTranspose.srcStride = BASE_LEN / BLOCK_SIZE;
                AscendC::LoadData(
                    l0b_buf_cube2[i * BASE_LEN * BLOCK_SIZE],
                    l1BufBCube2[i * CUBE_MATRIX_SIZE],
                    commonLoadData2dParamsTranspose
                );
            }

            SET_FLAG(MTE1, MTE2, eventIdCube2);
            SET_FLAG(MTE1, M, mte1Cube2MadEventId);
            WAIT_FLAG(MTE1, M, mte1Cube2MadEventId);

            bool init_c = (actualKIdx == 0);
            bool out_c = (actualKIdx == cube2_addr[curSwitchIndex].k - 1);
            unit_flag = out_c ? 0b11 : 0b10;
            commonMadParams.unitFlag = unit_flag;
            commonMadParams.cmatrixInitVal = init_c;
            AscendC::Mmad(
                l0c_buf_cube2,
                l0a_buf_cube2,
                l0b_buf_cube2,
                commonMadParams
            );

            SET_FLAG(M, MTE1, mte1Cube2MadEventId);

            if (out_c) {
                AscendC::SetAtomicAdd<float>();
                AscendC::SetAtomicType<float>();
                commonFixpipeParamsV220.unitFlag = unit_flag;
                temp_tensor_fp32.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float *>(cube2_addr[curSwitchIndex].out));
                AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                    temp_tensor_fp32,
                    l0c_buf_cube2,
                    commonFixpipeParamsV220
                );
                AscendC::SetAtomicNone();
            }
        }
    }

    WAIT_FLAG(M, MTE1, EVENT_ID0);
    WAIT_FLAG(M, MTE1, EVENT_ID1);
    WAIT_FLAG(M, MTE1, EVENT_ID2);
    WAIT_FLAG(M, MTE1, EVENT_ID3);
    WAIT_FLAG(M, MTE1, EVENT_ID4);
    WAIT_FLAG(M, MTE1, EVENT_ID5);
    WAIT_FLAG(M, MTE1, EVENT_ID6);
    WAIT_FLAG(M, MTE1, EVENT_ID7);

    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID4);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID5);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID6);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID7);
}

}

#endif // __DAV_C220_CUBE__

#endif // __CUBEBACKWARD_H__
