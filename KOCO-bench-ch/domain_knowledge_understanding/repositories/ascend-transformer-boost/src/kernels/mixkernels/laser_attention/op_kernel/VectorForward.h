/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __VECTORFORWARD_H__
#define __VECTORFORWARD_H__

#define USE_ASCENDC
#include "ppmatmul_const.h"
#include "kernels/utils/kernel/common.h"
#include "kernels/utils/kernel/common_func.h"
#include "kernels/utils/kernel/hardware.h"
#include "kernels/utils/kernel/simd.h"
#include "kernels/utils/kernel/iterator.h"
#include "kernels/utils/kernel/mma.h"
#include "kernels/utils/kernel/utils.h"

using namespace AscendC;
#ifdef __DAV_C220_VEC__

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T> class VectorForward {
public:
    __aicore__ inline VectorForward(){};
    __aicore__ inline void Init(__gm__ uint8_t *__restrict__ aCube1, __gm__ uint8_t *__restrict__ bCube1,
                                __gm__ uint8_t *__restrict__ bCube2, __gm__ uint8_t *__restrict__ maskGm,
                                __gm__ uint8_t *__restrict__ onesGm, __gm__ float *__restrict__ zerosGm,
                                __gm__ uint8_t *__restrict__ scoreGm, __gm__ float *__restrict__ cCube2,
                                __gm__ uint8_t *__restrict__ cube2_out, __gm__ float *__restrict__ logSumGm,
                                __gm__ float *__restrict__ gmRowsum, int32_t qSeqLength, int32_t kSeqLength, int32_t H,
                                int32_t B, int32_t Y, bool qk, int32_t windowsBlockNum, int32_t maskSeqLength,
                                float scale, int32_t transposeType);
    __aicore__ inline void Run();
    __aicore__ inline void SetHighPrecision(bool isHighPrecision)
    {
        this->isHighPrecision = isHighPrecision;
    };

    struct UB_FOR_SHORT_LEN_ATTN_SCORE {
        LocalTensor<float> tensorForLoadAttnScoreFp32{};
        LocalTensor<float> tensorForStoreSecondAttnScoreFp32{};
        LocalTensor<float> tensorForLoadOneBlockTriMaskFp32{};
        LocalTensor<float> tensorForCaclRowmaxFp32{};
        LocalTensor<float> tensorForCaclFinalRowmaxFp32{};
        LocalTensor<float> tensorForCaclSecondFinalRowmaxFp32{};
        LocalTensor<float> tensorForCaclFinalRowsumFp32{};
        LocalTensor<float> tensorForCaclSecondFinalRowsumFp32{};
        LocalTensor<float> tensorForVbrcbRowmaxFp32{};
        LocalTensor<float> tensorForVbrcbSecondRowmaxFp32{};
        LocalTensor<float> tensorForRecordRowmaxFp32{};
        LocalTensor<float> tensorForRecordRowsumFp32{};
    };

    struct UB_FOR_LONG_SEQ_ATTN_SCORE {
        LocalTensor<float> tensorForLoadAttnScoreFp32{};
        LocalTensor<float> tensorForLoadExtraFirstAttnScoreFp32{};
        LocalTensor<float> tensorForLoadOneBlockTriMaskFp32{};
        LocalTensor<float> tensorForCaclRowmaxFp32{};
        LocalTensor<float> tensorForCaclFinalRowmaxFp32{};
        LocalTensor<float> tensorForCaclFinalRowsumFp32{};
        LocalTensor<float> tensorForVbrcbRowmaxFp32{};
        LocalTensor<float> tensorForRecordRowmaxFp32{};
        LocalTensor<float> tensorForRecordRowsumFp32{};
    };

    struct UB_FOR_MDDIUM_SEQ_ATTN_SCORE {
        LocalTensor<float> tensorForStoreOneLineAttnScoreFp32{};
        LocalTensor<float> tensorForLoadAttnScoreFp32{};
        LocalTensor<float> tensorForCaclRowmaxFp32{};
        LocalTensor<float> tensorForCaclFinalRowmaxFp32{};
        LocalTensor<float> tensorForCaclFinalRowsumFp32{};
        LocalTensor<float> tensorForVbrcbRowmaxFp32{};
        LocalTensor<float> tensorForRecordRowmaxFp32{};
        LocalTensor<float> tensorForRecordRowsumFp32{};
        LocalTensor<float> tensorForLoadOneBlockTriMaskFp32{};
    };

    // UB空间划分，求归一化
    // bufForLoadOFp32：装载计算好的O
    // bufForLoadRowsumFp32：装载计算好的rowsum
    // bufForBrcbRowsumFp32：32字节对齐的rowsum
    struct UB_FOR_NORMALIZE {
        LocalTensor<float> tensorForLoadOFp32{};
        LocalTensor<float> tensorForLoadRowsumFp32{};
        LocalTensor<float> tensorForBrcbRowsumFp32{};

        int32_t oPingPongInterval;
        int32_t rowsumPingPongInterval;
        int32_t rowsumBrcbPingPongInterval;
    };

    // blockNumPerStep：当前处理的块数量（对齐MAX_BLOCK_PER_ONE_PROC）
    // blockNumForLast：最后一次ping-pong（不满MAX_BLOCK_PER_ONE_PROC时）
    // lastPaddingBlockNum：当前需要padding的块数量
    // sectionStartLineOffset：记录当前行的起始地址
    // totalFragNum：总分段数量
    // curFrag：当前分段id
    // triMatrixNum：三角阵的数量  0-非三角阵；1-三角阵，非unirow；2-三角阵，unirow
    // bufOffset：存score * scal的偏移
    struct PARAM_MEDIUM_SEQ_EXP {
        int32_t blockNumPerStep;
        int32_t blockNumForLast;

        int32_t lastPaddingBlockNum;

        int32_t sectionStartLineOffset;
        int32_t sectionMaskOffset;

        int32_t totalFragNum;
        int32_t curFrag;

        bool tailBlock;

        int32_t triMatrixNum;
        bool applyTriMask;

        int32_t bufOffset;

        int32_t recordRowmaxOffset;
    };

    // sectionStartLineOffset：attn score 起始地址
    // sectionMaskOffset：mask的地址 （两个secion的相同）
    // recordRowmaxOffset：记录rowmax的偏移
    struct PARAM_SHORT_SEQ_MAX {
        int32_t sectionOneBlockNum;
        int32_t sectionTwoBlockNum;
        int32_t sectionOnePaddingBlockNum;
        int32_t sectionTwoPaddingBlockNum;

        int32_t sectionStartLineOffset;
        int32_t sectionMaskOffset;

        int32_t recordRowmaxOffset;
    };

    // sectionBlockNum：当前处理的块数量
    // sectionPaddingBlockNum：当前需要padding的块数量
    // sectionStartLineOffset：attn score 起始地址
    // sectionMaskOffset：mask的地址 （两个secion的相同）
    // totalFragNum：总分段数量
    // curFrag：当前分段id
    // triMatrixNum：三角阵的数量  0-非三角阵；1-三角阵，非unirow；2-三角阵，unirow
    struct PARAM_LONG_SEQ_EXP {
        int32_t sectionBlockNum;
        int32_t sectionPaddingBlockNum;

        int32_t sectionStartLineOffset;
        int32_t sectionMaskOffset;

        int32_t totalFragNum;
        int32_t curFrag;

        int32_t triMatrixNum;
        int32_t applyTriMask;
    };

    struct UB_FOR_LN_ROWSUM {
        __ubuf__ float *ubBufForRowsum;
        __ubuf__ float *ubBufForRowsumRes;
    };

private:
    __aicore__ __inline__ void initCube2WorkSpace();
    __aicore__ __inline__ void InitWorkSpace();
    __aicore__ __inline__ void AllocateUbufForNorm(UB_FOR_NORMALIZE *UbNorm);
    __aicore__ __inline__ void GetGlobalInfo(int32_t batchNum, int32_t headNum, int32_t yNumCubePerLine,
                                             int32_t qSeqLen, int32_t kSeqLen, bool qkTriangle, int32_t VNum,
                                             int32_t *cubeNum, int32_t *fCubeGroupNum, int32_t *blockNumPerFullLine,
                                             int32_t *blockNumPerHead, int32_t *loopTimesForAllFullLine,
                                             int32_t *loopTailsForAllFullLine);
    __aicore__ __inline__ void GetLocalInfo(int32_t yNumCubePerLine, int32_t *cubeGroupBelong,
                                            int32_t *cubeGroupInnderIdx, int32_t *eachVectorProcLineNum);
    __aicore__ __inline__ void GetInitOffsetForCurrentCore(int32_t cubeGroupBelong, int32_t cubeGroupInnderIdx,
                                                           int32_t blockNumPerFullLine, int32_t eachVectorProcLineNum,
                                                           int32_t *eachVectorStartLineOffset,
                                                           int32_t *triMatrixMaskOffset);
    __aicore__ __inline__ void GetSubSeqLengthPerProc(int32_t kSeqLen, int32_t blockNumPerFullLine,
                                                      int32_t *subSeqLengthPerProc);
    __aicore__ __inline__ void GetPaddingInfoForRowMax(int32_t totalBlockNum, int32_t *paddingBlockNum);
    __aicore__ __inline__ void PaddingForRowMaxTensor(int32_t totalBlockNum, int32_t paddingBlockNum,
                                                      const LocalTensor<float> &paddingTensor, float value);
    __aicore__ __inline__ void GetFullLineOffsetInOneHead(int32_t curLoopIndex, int32_t cubeGroupBelong,
                                                          int32_t fCubeGroupNum, int32_t blockNumPerHead,
                                                          int32_t *blockLineOffsetInOneHead);
    __aicore__ __inline__ void
    GetTriMatrixSectionBlockOffset(int32_t blockLineOffsetInOneHead, int32_t blockNumPerFullLine,
                                   int32_t *sectionOneStartBlockOffset, int32_t *sectionTwoStartBlockOffset,
                                   int32_t *sectionOneEndBlockOffset, int32_t *sectionTwoEndBlockOffset);
    __aicore__ __inline__ void CaclMaxTensor(const LocalTensor<float> &tensorForCacl, int32_t blockNum);
    __aicore__ __inline__ __attribute__((noinline)) void
    ProcessCaclMaxTensor(int32_t basicBlockNum, int32_t paddingBlockNum, bool ppFirstSection,
                         const LocalTensor<float> &curTensorForAttnScore, const LocalTensor<float> &curTensorForRowmax,
                         const LocalTensor<float> &tensorForCaclFinalRowmaxFp32);
    __aicore__ __inline__ void CaclSumTensor(const LocalTensor<float> &tensorForCacl, int32_t blockNum);
    __aicore__ __inline__ __attribute__((noinline)) void
    ProcessCaclSumTensor(int32_t basicBlockNum, int32_t paddingBlockNum, bool ppFirstSection,
                         const LocalTensor<float> &curTensorForAttnScore, const LocalTensor<float> &curTensorForRowsum,
                         const LocalTensor<float> &tensorForCaclFinalRowsumFp32);
    __aicore__ __inline__ void ProcessLinePhaseOneForShortSeqMax(bool qkTriangle, PARAM_SHORT_SEQ_MAX Param,
                                                                 int32_t pingPongFlag, bool firstLine,
                                                                 __gm__ WORKSPACE_T *attnScoreGm,
                                                                 __gm__ INPUT_T *attnMaskGm,
                                                                 UB_FOR_SHORT_LEN_ATTN_SCORE &UbAttn);
    __aicore__ __inline__ void ProcessLinePhaseOneForShortSeqExp(bool qkTriangle, PARAM_SHORT_SEQ_MAX Param,
                                                                 int32_t pingPongFlag, __gm__ WORKSPACE_T *attnScoreGm,
                                                                 __gm__ INPUT_T *attnMaskGm,
                                                                 UB_FOR_SHORT_LEN_ATTN_SCORE &UbAttn, int32_t offset);
    __aicore__ __inline__ void ProcessLinePhaseOneForMax(PARAM_LONG_SEQ_EXP Param, int32_t pingPongFlag,
                                                         __gm__ WORKSPACE_T *attnScoreGm, __gm__ INPUT_T *attnMaskGm,
                                                         UB_FOR_LONG_SEQ_ATTN_SCORE &UbAttn, bool sparseFlag);
    __aicore__ __inline__ void ProcessLinePhaseOneForExp(PARAM_LONG_SEQ_EXP Param, int32_t pingPongFlag,
                                                         __gm__ WORKSPACE_T *attnScoreGm, __gm__ INPUT_T *attnMaskGm,
                                                         UB_FOR_LONG_SEQ_ATTN_SCORE &UbAttn, int32_t tmp,
                                                         bool sparseFlag);
    __aicore__ __inline__ void GetNormLocalInfo(int32_t qSeqLen, int32_t headNum, int32_t batchNum,
                                                int32_t *eachCoreProcessLines, int32_t *eachCoreOffsetLines);
    __aicore__ __inline__ void ProcessForNormalize(int32_t linesPerLoop, int32_t pvResultOffset, int32_t rowsumOffset,
                                                   int32_t pingPongFlag, UB_FOR_NORMALIZE &UbNorm,
                                                   __gm__ float *__restrict__ gmCCube2,
                                                   __gm__ INPUT_T *__restrict__ gm_cube2_out,
                                                   __gm__ float *__restrict__ rowsumGm);
    __aicore__ __inline__ void AttentionScoreNormalize(int32_t maxProcLen, int32_t curCoreProcessLines,
                                                       int32_t curCoreOffsetLines, UB_FOR_NORMALIZE &UbNorm,
                                                       __gm__ float *__restrict__ gmCCube2,
                                                       __gm__ INPUT_T *__restrict__ gm_cube2_out,
                                                       __gm__ float *__restrict__ rowsumGm);
    __aicore__ __inline__ void
    AttentionScoreShortDoubleLineOne(bool qkTriangle, int32_t sectionOneBlockNums, int32_t sectionTwoBlockNums,
                                     int32_t triMatrixMaskOffset, int32_t eachVectorProcLineNum,
                                     int32_t localSectionOneStartLineOffset, int32_t localSectionTwoStartLineOffset,
                                     __gm__ WORKSPACE_T *__restrict__ attnScoreGm,
                                     __gm__ INPUT_T *__restrict__ attnMaskGm, UB_FOR_SHORT_LEN_ATTN_SCORE &UbAttn);
    __aicore__ __inline__ void
    AttentionScoreDoubleLine(bool qkTriangle, int32_t sectionLoopTimes, int32_t sectionOneBlockNums,
                             int32_t sectionTwoBlockNums, int32_t triMatrixMaskOffset, int32_t eachVectorProcLineNum,
                             int32_t localSectionOneStartLineOffset, int32_t localSectionTwoStartLineOffset,
                             int32_t triMatrixNum, int32_t triMatrixOffset[],
                             __gm__ WORKSPACE_T *__restrict__ attnScoreGm, __gm__ INPUT_T *__restrict__ attnMaskGm,
                             UB_FOR_LONG_SEQ_ATTN_SCORE &UbAttn, bool sparseFlag);
    __aicore__ __inline__ void GetUniRowmaxSeqInfoPerProc(int32_t blockNumPerFullLine, int32_t subSeqLengthPerProc,
                                                          int32_t *pingBlockOffsetNum, int32_t *pongBlockOffsetNum,
                                                          int32_t *tailBlockOffsetNum, int32_t *tailBlockNum,
                                                          int32_t *pingPongTimes);
    __aicore__ __inline__ void ProcessLinePhaseOneForMax(PARAM_MEDIUM_SEQ_EXP Param, int32_t pingPongFlag,
                                                         bool firstProc, __gm__ WORKSPACE_T *__restrict__ attnScoreGm,
                                                         __gm__ INPUT_T *__restrict__ attnMaskGm,
                                                         UB_FOR_MDDIUM_SEQ_ATTN_SCORE &UbAttn, int32_t lines);
    __aicore__ __inline__ void ProcessLinePhaseOneForExp(PARAM_MEDIUM_SEQ_EXP Param, int32_t pingPongFlag,
                                                         bool firstProc, __gm__ WORKSPACE_T *__restrict__ attnScoreGm,
                                                         __gm__ INPUT_T *__restrict__ attnMaskGm,
                                                         UB_FOR_MDDIUM_SEQ_ATTN_SCORE &UbAttn, int32_t fp32Offset);
    __aicore__ __inline__ void
    AttentionScoreSingleLine(bool qkTriangle, int32_t sectionLoopTimes, int32_t sectionOneBlockNums,
                             int32_t sectionTwoBlockNums, int32_t triMatrixMaskOffset, int32_t eachVectorProcLineNum,
                             int32_t localSectionOneStartLineOffset, int32_t localSectionTwoStartLineOffset,
                             int32_t triMatrixNum, __gm__ WORKSPACE_T *__restrict__ attnScoreGm,
                             __gm__ INPUT_T *__restrict__ attnMaskGm, UB_FOR_MDDIUM_SEQ_ATTN_SCORE &UbAttn);
    __aicore__ __inline__ void AllocateUbufForLongSeqAttnScore(UB_FOR_LONG_SEQ_ATTN_SCORE *UbAttnScore);
    __aicore__ __inline__ void AllocateUbufForMediumSeqAttnScore(UB_FOR_MDDIUM_SEQ_ATTN_SCORE *UbAttnScore);
    __aicore__ __inline__ void AllocateUbufForShortSeqAttnScore(UB_FOR_SHORT_LEN_ATTN_SCORE *UbAttnScore);
    __aicore__ __inline__ void GetSoftmaxOffset(int32_t sectionNum, int32_t yCubeNumPerLine, int32_t timesSyncCube,
                                                int32_t qSeqLen, int32_t headNum, bool qkTriangle,
                                                int32_t *totalOffsetForRowmax);
    __aicore__ __inline__ void RecordRowmaxAndRowsum(bool sparseFlag, int32_t timesSyncCube,
                                                     LocalTensor<float> tensorForLong, LocalTensor<float> tensorForMid,
                                                     LocalTensor<float> tensorForShort,
                                                     GlobalTensor<float> tensorOutput);
    __aicore__ __inline__ void InitUbForSoftSync();
    __aicore__ __inline__ void SoftSyncByGroup(int32_t cubeGroupBelong);

private:
    __gm__ INPUT_T *__restrict__ gmACube1;
    __gm__ INPUT_T *__restrict__ gmBCube1;
    __gm__ INPUT_T *__restrict__ gmBCube2;
    __gm__ INPUT_T *__restrict__ attnMaskGm;
    __gm__ INPUT_T *__restrict__ gmOnes;
    __gm__ INPUT_T *__restrict__ gmZeros;
    __gm__ WORKSPACE_T *__restrict__ attnScoreGm;
    __gm__ float *__restrict__ gmCCube2;
    __gm__ INPUT_T *__restrict__ gm_cube2_out;
    __gm__ float *__restrict__ logSumMaxGm;
    __gm__ float *__restrict__ rowsumGm;

    __ubuf__ float *__restrict__ ubForSoftsyncFlags;
    __ubuf__ float *__restrict__ ubForSoftsyncCheck;

    AsdopsBuffer<ArchType::ASCEND_V220> calBuf;
    GlobalTensor<INPUT_T> gmACube1Tensor;
    GlobalTensor<INPUT_T> gmBCube1Tensor;
    GlobalTensor<INPUT_T> gmBCube2Tensor;
    GlobalTensor<INPUT_T> attnMaskGmTensor;
    GlobalTensor<INPUT_T> gmOnesTensor;
    GlobalTensor<INPUT_T> gmZerosTensor;
    GlobalTensor<float> attnScoreGmTensor;
    GlobalTensor<float> gmCCube2Tensor;
    GlobalTensor<INPUT_T> gmCCube2OutTensor;
    GlobalTensor<float> logSumMaxGmTensor;
    GlobalTensor<float> rowsumGmTensor;
    LocalTensor<float> ubForSoftsyncFlagsTensor;
    LocalTensor<float> ubForSoftsyncCheckTensor;
    LocalTensor<float> fp32TestTensor;
    LocalTensor<float> ubScoreZerotensor;
    // int32_t seq_len;
    int32_t qSeqLen;
    int32_t kSeqLen;
    int32_t headNum;
    int32_t batchNum;
    int32_t yCubeNumPerLine;
    bool qkTriangle;
    int32_t VNum;
    int32_t softSyncTimesCount = 0;
    bool useSoftSync = false;
    bool isHighPrecision = true;
    int32_t maskSeqLength;
    int32_t rowPerTime = 1;
    float scale;
    int32_t transposeType;
};

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ inline void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::Init(
    __gm__ uint8_t *__restrict__ aCube1, __gm__ uint8_t *__restrict__ bCube1, __gm__ uint8_t *__restrict__ bCube2,
    __gm__ uint8_t *__restrict__ maskGm, __gm__ uint8_t *__restrict__ onesGm, __gm__ float *__restrict__ zerosGm,
    __gm__ uint8_t *__restrict__ scoreGm, __gm__ float *__restrict__ cCube2, __gm__ uint8_t *__restrict__ cube2_out,
    __gm__ float *__restrict__ logSumGm, __gm__ float *__restrict__ gmRowsum,
    // int32_t S
    int32_t qSeqLength, int32_t kSeqLength, int32_t H, int32_t B, int32_t Y, bool qk, int32_t windowsBlockNum,
    int32_t maskSeqLength, float scale, int32_t transposeType)
{
    gmACube1 = (__gm__ INPUT_T *__restrict__)aCube1;
    gmBCube1 = (__gm__ INPUT_T *__restrict__)bCube1;
    gmBCube2 = (__gm__ INPUT_T *__restrict__)bCube2;
    attnMaskGm = (__gm__ INPUT_T *__restrict__)maskGm;
    gmOnes = (__gm__ INPUT_T *__restrict__)onesGm;
    gmZeros = (__gm__ INPUT_T *__restrict__)zerosGm;
    attnScoreGm = (__gm__ WORKSPACE_T *__restrict__)scoreGm;
    gmCCube2 = cCube2;
    gm_cube2_out = (__gm__ INPUT_T *__restrict__)cube2_out;
    logSumMaxGm = logSumGm;
    rowsumGm = gmRowsum;

    gmACube1Tensor.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(aCube1));
    gmBCube1Tensor.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(bCube1));
    gmBCube2Tensor.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(bCube2));
    attnMaskGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(maskGm));
    gmOnesTensor.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(onesGm));
    gmZerosTensor.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(zerosGm));
    attnScoreGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>((__gm__ float *__restrict__)scoreGm));
    gmCCube2Tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(cCube2));
    gmCCube2OutTensor.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(cube2_out));
    logSumMaxGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(logSumGm));
    rowsumGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gmRowsum));
    fp32TestTensor = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(192 * 1024 - 128 * 4);
    qSeqLen = qSeqLength;
    kSeqLen = kSeqLength;
    headNum = H;
    batchNum = B;
    yCubeNumPerLine = Y;
    qkTriangle = qk;
    VNum = windowsBlockNum;
    this->maskSeqLength = maskSeqLength;
    this->scale = scale;
    this->transposeType = transposeType;
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::InitWorkSpace()
{
    int32_t matrixNumbers = BASE_BLOCK_SIZE;
    int32_t vectorNum = AscendC::GetBlockNum() * VEC_NUM_PER_CUBE;
    int32_t vectorId = AscendC::GetBlockIdx();

    if (yCubeNumPerLine >= 1) {
        vectorNum = 1;
        vectorId = 0;
    }

    LocalTensor<INPUT_T> tensorScoreOne = calBuf.GetBuffer<BufferType::ASCEND_UB, INPUT_T>(0);

    AscendC::Duplicate(tensorScoreOne, (INPUT_T)1, 128, 128, 1, 8);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    AscendC::DataCopy(gmOnesTensor, tensorScoreOne, AscendC::DataCopyParams(1, matrixNumbers / 16, 0, 0));
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> tensorScoreZero = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    int32_t perLength = 256;
    int32_t numbersPerVector = (batchNum * headNum * qSeqLen + vectorNum - 1) / vectorNum;
    int32_t repeat = (numbersPerVector + perLength - 1) / perLength;
    AscendC::Duplicate(tensorScoreZero, (float)0, 64, 4, 1, 8);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    // Y >=1 时，得到大小为repeat*256*4B的gm空间
    for (int i = 0; i < repeat; i++) {
        AscendC::DataCopy(rowsumGmTensor[vectorId * perLength * repeat + i * perLength], tensorScoreZero,
                          AscendC::DataCopyParams(1, perLength / 8, 0, 0));
    }

    if (useSoftSync) {
        // 拷贝0到zerosGm，用于软同步做累加
        for (int i = 0; i < 16; i++) {
            GlobalTensor<float> tmpZeroGmTensor;
            tmpZeroGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gmZeros + i * perLength));
            AscendC::DataCopy(tmpZeroGmTensor, tensorScoreZero, AscendC::DataCopyParams(1, perLength / 8, 0, 0));
        }
    }

    PipeBarrier<PIPE_ALL>();
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ inline void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::Run()
{
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

    // ture: 三角阵，一行统一计算 (还未支持末尾tri mask)；false：一行分两个section计算
    bool unirowMode = false;

    int32_t cubeNum = 0;
    int32_t fCubeGroupNum = 0;
    int32_t blockNumPerFullLine = 0;
    int32_t blockNumPerHead = 0;
    int32_t loopTimesForAllFullLine = 0;
    int32_t loopTailsForAllFullLine = 0;

    GetGlobalInfo(batchNum, headNum, yCubeNumPerLine, qSeqLen, kSeqLen, qkTriangle, VNum, &cubeNum, &fCubeGroupNum,
                  &blockNumPerFullLine, &blockNumPerHead, &loopTimesForAllFullLine, &loopTailsForAllFullLine);

    int32_t cubeGroupBelong = 0;
    int32_t cubeGroupInnderIdx = 0;
    int32_t eachVectorProcLineNum = 0;

    GetLocalInfo(yCubeNumPerLine, &cubeGroupBelong, &cubeGroupInnderIdx, &eachVectorProcLineNum);

    int32_t eachVectorStartLineOffset = 0;
    int32_t triMatrixMaskOffset = 0;

    GetInitOffsetForCurrentCore(cubeGroupBelong, cubeGroupInnderIdx, blockNumPerFullLine, eachVectorProcLineNum,
                                &eachVectorStartLineOffset, &triMatrixMaskOffset);

    UB_FOR_NORMALIZE UbNorm;
    // 归一化在EXP计算完成之后，这里重新分配UB地址，不会影响rowmax计算
    AllocateUbufForNorm(&UbNorm);

    // attention score （取代UB_FOR_EXP）
    UB_FOR_SHORT_LEN_ATTN_SCORE UbShortSeqAttn;
    UB_FOR_LONG_SEQ_ATTN_SCORE UbLongSeqAttn;
    UB_FOR_MDDIUM_SEQ_ATTN_SCORE UbMediumSeqAttn;

    // 三角阵拼接，左侧部分的起始block
    int32_t sectionOneStartBlockOffset = 0;
    // 三角阵拼接，右侧部分的起始block
    int32_t sectionTwoStartBlockOffset = 0;

    // 三角阵拼接, 左侧最后一个block的偏移
    int32_t sectionOneEndBlockOffset = 0;
    // 三角阵拼接, 右侧最后一个block的偏移
    int32_t sectionTwoEndBlockOffset = 0;

    int32_t sectionOneBlockNums = 0;
    int32_t sectionTwoBlockNums = 0;

    // 当前处理的完整行，在当前Head中位于第几行
    int32_t blockLineOffsetInOneHead = 0;

    // 循环内部使用
    int32_t localSectionOneStartLineOffset = 0;
    int32_t localSectionTwoStartLineOffset = 0;

    // 三角块的offset
    int32_t triMatrixOffset[2] = {0};
    int32_t triMatrixNum = (qkTriangle == false ? 0 : 2);

    initCube2WorkSpace();

    FftsCrossCoreSync<PIPE_MTE3, 0>(AIVFLAGID);
    WaitFlagDev(AIVFLAGID);

    FftsCrossCoreSync<PIPE_MTE3, 2>(AIV2AICFLAGID);

    if (useSoftSync) {
        InitUbForSoftSync();
    }

    // 大循环
    for (int timesSyncCube = 0; timesSyncCube < loopTimesForAllFullLine; timesSyncCube++) {
        // workspace double buffer
        if (timesSyncCube % 2 == 0 && timesSyncCube != 0) {
            attnScoreGm -= blockNumPerFullLine * fCubeGroupNum * BASE_BLOCK_DATA_NUM;
        }
        if (timesSyncCube % 2 == 1) {
            attnScoreGm += blockNumPerFullLine * fCubeGroupNum * BASE_BLOCK_DATA_NUM;
        }

        bool sparseFlag = false;
        if (VNum > 0) {
            GetFullLineOffsetInOneHead(timesSyncCube, cubeGroupBelong, fCubeGroupNum, blockNumPerHead,
                                       &blockLineOffsetInOneHead);
            if (blockLineOffsetInOneHead >= VNum / 2) {
                sparseFlag = true;
                qkTriangle = false;
            } else {
                sparseFlag = false;
                qkTriangle = true;
            }
        }
        // Section loop times
        int32_t sectionLoopTimes = qkTriangle == false ? 1 : 2;
        if (qkTriangle == true) {
            GetFullLineOffsetInOneHead(timesSyncCube, cubeGroupBelong, fCubeGroupNum, blockNumPerHead,
                                       &blockLineOffsetInOneHead);
        }

        if (loopTailsForAllFullLine > 0 && timesSyncCube == loopTimesForAllFullLine - 1) {
            // 处理尾块行（如果有）；部分vector会被闲置 -- 当前128一定能整除num(vector)
            if (cubeGroupBelong >= loopTailsForAllFullLine) {
                if (yCubeNumPerLine > 1) {
                    if (useSoftSync) {
                        SoftSyncByGroup(cubeGroupBelong);
                    } else {
                        FftsCrossCoreSync<PIPE_MTE3, 0>(AIVFLAGID);
                        WaitFlagDev(AIVFLAGID);
                    }
                }
                continue;
            }
        }

        if (qkTriangle == true) {
            // 只要是三角阵，都需要计算两段地址；非三角阵没有必要计算
            GetTriMatrixSectionBlockOffset(blockLineOffsetInOneHead, blockNumPerFullLine, &sectionOneStartBlockOffset,
                                           &sectionTwoStartBlockOffset, &sectionOneEndBlockOffset,
                                           &sectionTwoEndBlockOffset);

            triMatrixOffset[0] = sectionOneEndBlockOffset;
            triMatrixOffset[1] = sectionTwoEndBlockOffset;

            // 每个Section， 如果有tail，则一定是tri； 否则，ping不会有， pong最后一次的最后一个
            sectionTwoBlockNums = sectionTwoEndBlockOffset - sectionTwoStartBlockOffset + 1;
            // 非三角阵， 该参数没用
            localSectionTwoStartLineOffset =
                eachVectorStartLineOffset + BASE_BLOCK_DATA_NUM * sectionTwoStartBlockOffset;
        } else {
            sectionOneStartBlockOffset = 0;
            // 每行最后一个block
            sectionOneEndBlockOffset = blockNumPerFullLine - 1;
        }

        // section one其实偏移一定是block，不用再累加
        localSectionOneStartLineOffset = eachVectorStartLineOffset;

        sectionOneBlockNums = sectionOneEndBlockOffset - sectionOneStartBlockOffset + 1;

        WaitFlagDev(AIC2AIVFLAGID);
        if (kSeqLen > MDDIUM_SEQ_THRESHOLD || (kSeqLen > SHORT_SEQ_THRESHOLD && qkTriangle == true) ||
            sparseFlag == true) {
            rowPerTime = 1;
            AllocateUbufForLongSeqAttnScore(&UbLongSeqAttn);
            AttentionScoreDoubleLine(qkTriangle, sectionLoopTimes, sectionOneBlockNums, sectionTwoBlockNums,
                                     triMatrixMaskOffset, eachVectorProcLineNum, localSectionOneStartLineOffset,
                                     localSectionTwoStartLineOffset, triMatrixNum, triMatrixOffset, attnScoreGm,
                                     attnMaskGm, UbLongSeqAttn, sparseFlag);
        } else if (kSeqLen > SHORT_SEQ_THRESHOLD) {
            rowPerTime = 1;
            AllocateUbufForMediumSeqAttnScore(&UbMediumSeqAttn);
            AttentionScoreSingleLine(qkTriangle, sectionLoopTimes, sectionOneBlockNums, sectionTwoBlockNums,
                                     triMatrixMaskOffset, eachVectorProcLineNum, localSectionOneStartLineOffset,
                                     localSectionTwoStartLineOffset, triMatrixNum, attnScoreGm, attnMaskGm,
                                     UbMediumSeqAttn);
        } else {
            if (kSeqLen == 4096 || kSeqLen == 2048 || kSeqLen == 1024 || kSeqLen == 512 || kSeqLen == 256) {
                rowPerTime = 2;
            }
            AllocateUbufForShortSeqAttnScore(&UbShortSeqAttn);
            AttentionScoreShortDoubleLineOne(qkTriangle, sectionOneBlockNums, sectionTwoBlockNums, triMatrixMaskOffset,
                                             eachVectorProcLineNum, localSectionOneStartLineOffset,
                                             localSectionTwoStartLineOffset, attnScoreGm, attnMaskGm, UbShortSeqAttn);
        }

        // 同步
        // 多核处理一个完整行，需要全核同步
        if (yCubeNumPerLine > 1) {
            if (useSoftSync) {
                SoftSyncByGroup(cubeGroupBelong);
            } else {
                FftsCrossCoreSync<PIPE_MTE3, 0>(AIVFLAGID);
                WaitFlagDev(AIVFLAGID);
            }
        }

        FftsCrossCoreSync<PIPE_MTE3, 2>(AIV2AICFLAGID);

        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);

        // scalar 捞 rowmax
        RecordRowmaxAndRowsum(sparseFlag, timesSyncCube, UbLongSeqAttn.tensorForRecordRowmaxFp32,
                              UbMediumSeqAttn.tensorForRecordRowmaxFp32, UbShortSeqAttn.tensorForRecordRowmaxFp32,
                              logSumMaxGmTensor);
        // scalar 捞 rowsum
        RecordRowmaxAndRowsum(sparseFlag, timesSyncCube, UbLongSeqAttn.tensorForRecordRowsumFp32,
                              UbMediumSeqAttn.tensorForRecordRowsumFp32, UbShortSeqAttn.tensorForRecordRowsumFp32,
                              rowsumGmTensor);

    } // 大循环 -- 第一次，将attention exp 计算完毕
    WaitFlagDev(AIC2AIVFLAGID);
    // 由cube计算rowsum ，这里需要同步一次，1. rowsum从rowsum_gm读取；2.前期计算得到的exp值，放在gm_c_cube2
    // 需要逐行将数据读入，除以sum，执行归一化 -- 拷贝多少数据，每行不一样

    // ~~~~~~~~~~~~~~~~ 归一化处理 ~~~~~~~~~~~~~~
    FftsCrossCoreSync<PIPE_MTE3, 0>(AIVFLAGID);
    WaitFlagDev(AIVFLAGID);

    if (transposeType == 3) {
        // 归一化时， 每个vector处理的行数
        int32_t normEachVecProcessLines = 0;
        // 归一化时， 每个vector处理的起始行
        int32_t normEachVecOffsetLines = 0;

        // normEachVecOffsetLines：会有尾行，不能被48个核整除，尾行会从第0个核开始补充
        GetNormLocalInfo(qSeqLen, headNum, batchNum, &normEachVecProcessLines, &normEachVecOffsetLines);

        // 一次处理的最大长度（ping+pong）
        // 处理的数量
        int32_t maxProcLen = normEachVecProcessLines * HEAD_DIM;
        // 保证偶数，可以被pingpong平分
        if (normEachVecProcessLines % 2 > 0) {
            maxProcLen += HEAD_DIM;
        }
        maxProcLen = maxProcLen > MAX_LENG_PER_UB_PROC * 2 ? MAX_LENG_PER_UB_PROC * 2 : maxProcLen;

        AttentionScoreNormalize(maxProcLen, normEachVecProcessLines, normEachVecOffsetLines, UbNorm, gmCCube2,
                                gm_cube2_out, rowsumGm);
    }
    // ~~~~~~~~~~~~~~~~ 归一化处理 ~~~~~~~~~~~~~~
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::initCube2WorkSpace()
{
    int32_t total_block_number = batchNum * headNum * qSeqLen / BASE_BLOCK_LENGTH;
    int32_t core_num = get_block_num();
    int32_t vector_num = core_num * 2;
    int32_t vector_id = get_block_idx() * 2 + get_subblockid(); // [0,47]

    int64_t global_offset = 0;
    int64_t local_offset = 0;
    int32_t max_block_per_vector = 3;
    int32_t total_block_number_per_core = total_block_number / vector_num; // 计算总轮次
    int32_t tail_block_num = total_block_number % vector_num;
    int32_t init_offset = vector_id * total_block_number_per_core * BASE_BLOCK_SIZE;
    if (tail_block_num > 0) {
        if (vector_id < tail_block_num) {
            total_block_number_per_core++;
            init_offset = vector_id * total_block_number_per_core * BASE_BLOCK_SIZE;
        } else {
            init_offset = (tail_block_num * (total_block_number_per_core + 1) +
                           (vector_id - tail_block_num) * total_block_number_per_core) *
                          BASE_BLOCK_SIZE;
        }
    }

    int32_t Z = total_block_number_per_core / max_block_per_vector;
    int32_t tail = total_block_number_per_core % max_block_per_vector;

    if (tail > 0) {
        Z++;
    }

    ubScoreZerotensor = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    // __ubuf__ float *ubScoreZero =  reinterpret_cast<__ubuf__ float *>((uintptr_t)0);

    int32_t dup_times = 0;
    if (total_block_number_per_core >= max_block_per_vector) {
        dup_times = max_block_per_vector * 2; // 一次最多dup半块
    } else {
        dup_times = total_block_number_per_core * 2;
    }
    for (int32_t i = 0; i < dup_times; i++) {
        AscendC::Duplicate(ubScoreZerotensor[i * BASE_BLOCK_SIZE / 2], (float)0, 64, BASE_BLOCK_SIZE * 2 / 256, 1, 8);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    for (int32_t loop_time = 0; loop_time < Z; loop_time++) {
        int32_t cur_block_num = max_block_per_vector;
        if (tail > 0 && loop_time == Z - 1) {
            cur_block_num = tail;
        }
        int32_t local_offset = loop_time * max_block_per_vector * BASE_BLOCK_SIZE;

        GlobalTensor<float> tmpCCube2GmTensorOut;
        tmpCCube2GmTensorOut.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>((__gm__ float *)(gmCCube2 + init_offset + local_offset)));

        AscendC::DataCopy(tmpCCube2GmTensorOut, ubScoreZerotensor,
                          AscendC::DataCopyParams(1, (cur_block_num * BASE_BLOCK_SIZE) * 4 / 32, 0, 0));
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::SoftSyncByGroup(int32_t cubeGroupBelong)
{
    uint32_t flagSize = 128;
    PipeBarrier<PIPE_MTE3>();
    AscendC::SetAtomicType<float>();
    AscendC::SetAtomicAdd<float>();
    GlobalTensor<float> tmpZeroGmTensor;
    tmpZeroGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gmZeros));
    AscendC::DataCopy(tmpZeroGmTensor[cubeGroupBelong * flagSize], ubForSoftsyncFlagsTensor,
                      AscendC::DataCopyParams(1, 1, 0, 0));
    AscendC::SetAtomicNone();

    // 2*Y个vector往同一个地址累加1
    softSyncTimesCount += yCubeNumPerLine * VEC_NUM_PER_CUBE;

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
    PipeBarrier<PIPE_MTE2>();

    while (true) {
        AscendC::DataCopy(ubForSoftsyncCheckTensor, tmpZeroGmTensor[cubeGroupBelong * flagSize],
                          AscendC::DataCopyParams(1, 1, 0, 0));
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        auto diff = softSyncTimesCount - (float)*(ubForSoftsyncCheck);
        if (diff < 0.1f) {
            break;
        }
    }
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::GetSoftmaxOffset(int32_t sectionNum, int32_t yCubeNumPerLine,
                                                               int32_t timesSyncCube, int32_t qSeqLen, int32_t headNum,
                                                               bool qkTriangle, int32_t *totalOffsetForRowmax)
{
    int32_t cubeNumber = AscendC::GetBlockNum();
    int32_t cubeIdx = get_block_idx();
    int32_t vecIdx = get_subblockid();

    int32_t groupNumber = cubeNumber / yCubeNumPerLine;
    int32_t F = cubeNumber / yCubeNumPerLine;
    int32_t cubeIdInOneGroup = cubeIdx % yCubeNumPerLine;
    int32_t coreGroupIndex = cubeIdx / yCubeNumPerLine;
    // 倒三角:VNum=qSeqLen/128;  非倒三角:VNum=0; sparse:VNum=windowsBlockNum
    int32_t H = this->VNum;
    int32_t L = H / 2;
    int32_t rowPerBatch = headNum * (qSeqLen / BASE_BLOCK_LENGTH - L);

    // 当前在处理哪一行(总行数)(负载均衡后)
    int32_t curCalcRowBlock = timesSyncCube * F + coreGroupIndex;
    // Current row's batch number  0
    int32_t b = curCalcRowBlock / rowPerBatch;
    // Current batch's head number   0
    int32_t n = (curCalcRowBlock % rowPerBatch) / (qSeqLen / BASE_BLOCK_LENGTH - L);
    // 当前head内处理的是哪一行(负载均衡后)
    int32_t iRow = (curCalcRowBlock % rowPerBatch) % (qSeqLen / BASE_BLOCK_LENGTH - L);

    int32_t aivIRow = iRow * BASE_BLOCK_LENGTH + BASE_BLOCK_LENGTH / yCubeNumPerLine * cubeIdInOneGroup +
                      BASE_BLOCK_LENGTH / (VEC_NUM_PER_CUBE * yCubeNumPerLine) * vecIdx;
    int32_t seg0Index = L * BASE_BLOCK_LENGTH + aivIRow;

    if (sectionNum == 0) {
        *totalOffsetForRowmax = (b * headNum + n) * qSeqLen + seg0Index;
    } else {
        // head内 section2 负载均衡前的行数
        int32_t seg1Index = ((L - 1) - iRow) * BASE_BLOCK_LENGTH +
                            BASE_BLOCK_LENGTH / yCubeNumPerLine * cubeIdInOneGroup +
                            BASE_BLOCK_LENGTH / (VEC_NUM_PER_CUBE * yCubeNumPerLine) * vecIdx;
        *totalOffsetForRowmax = (b * headNum + n) * qSeqLen + seg1Index;
    }
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::RecordRowmaxAndRowsum(
    bool sparseFlag, int32_t timesSyncCube, LocalTensor<float> tensorForLong, LocalTensor<float> tensorForMid,
    LocalTensor<float> tensorForShort, GlobalTensor<float> tensorOutput)
{
    int32_t sectionNum = (qkTriangle == true) ? 2 : 1;
    int32_t totalOffsetForRowmax = 0;
    int32_t rowmaxUbOffset = 0;
    for (int section = 0; section < sectionNum; section++) {
        if (kSeqLen > MDDIUM_SEQ_THRESHOLD || (kSeqLen > SHORT_SEQ_THRESHOLD && qkTriangle == true) ||
            sparseFlag == true) {
            LocalTensor<float> tmpTensorForRecordRowmaxFp32 = section == 0 ? tensorForLong : tensorForLong[256];
            for (int i = 0; i < 32 / yCubeNumPerLine; i++) {
                fp32TestTensor.SetValue(rowmaxUbOffset, (float)tmpTensorForRecordRowmaxFp32.GetValue(0 + 8 * i));
                rowmaxUbOffset++;
                fp32TestTensor.SetValue(rowmaxUbOffset, (float)tmpTensorForRecordRowmaxFp32.GetValue(1 + 8 * i));
                rowmaxUbOffset++;
            }
        } else if (kSeqLen > SHORT_SEQ_THRESHOLD) {
            for (int i = 0; i < 64 / yCubeNumPerLine; i++) {
                fp32TestTensor.SetValue(rowmaxUbOffset,
                                        (float)tensorForMid.GetValue(i * 8 + section * 64 / yCubeNumPerLine * 8));
                rowmaxUbOffset++;
            }
        } else {
            if (rowPerTime == 1) {
                for (int i = 0; i < 64 / (yCubeNumPerLine * VEC_NUM_PER_CUBE); i++) {
                    fp32TestTensor.SetValue(rowmaxUbOffset, (float)tensorForShort.GetValue(section + 16 * i));
                    rowmaxUbOffset++;
                    fp32TestTensor.SetValue(rowmaxUbOffset, (float)tensorForShort.GetValue(512 + section + 16 * i));
                    rowmaxUbOffset++;
                }
            } else {
                for (int i = 0; i < 64 / (yCubeNumPerLine * VEC_NUM_PER_CUBE) / 2; i++) {
                    fp32TestTensor.SetValue(rowmaxUbOffset, (float)tensorForShort.GetValue(section * 2 + 16 * i));
                    rowmaxUbOffset++;
                    fp32TestTensor.SetValue(rowmaxUbOffset, (float)tensorForShort.GetValue(section * 2 + 1 + 16 * i));
                    rowmaxUbOffset++;
                    fp32TestTensor.SetValue(rowmaxUbOffset, (float)tensorForShort.GetValue(512 + section * 2 + 16 * i));
                    rowmaxUbOffset++;
                    fp32TestTensor.SetValue(rowmaxUbOffset,
                                            (float)tensorForShort.GetValue(512 + section * 2 + 1 + 16 * i));
                    rowmaxUbOffset++;
                }
            }
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        GetSoftmaxOffset(section, yCubeNumPerLine, timesSyncCube, qSeqLen, headNum, qkTriangle, &totalOffsetForRowmax);
        DataCopy(tensorOutput[totalOffsetForRowmax],
                 fp32TestTensor[section * BASE_BLOCK_SIDE_LEN / (yCubeNumPerLine * VEC_NUM_PER_CUBE)],
                 DataCopyParams(1, 128 / (yCubeNumPerLine * VEC_NUM_PER_CUBE) * B32_SIZE / 32, 0, 0));
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
}
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::InitUbForSoftSync()
{
    // 等InitWorkSpace完成
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);

    ubForSoftsyncFlagsTensor = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    ubForSoftsyncCheckTensor = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(256);

    AscendC::Duplicate(ubForSoftsyncFlagsTensor, (float)1, 64, 1, 1, 8);
    PipeBarrier<PIPE_V>();
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::AllocateUbufForShortSeqAttnScore(UB_FOR_SHORT_LEN_ATTN_SCORE *UbAttnScore)
{
    int32_t offset = 512;

    UbAttnScore->tensorForLoadAttnScoreFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += (MAX_LENG_PER_UB_PROC + BASE_BLOCK_SIDE_LEN * rowPerTime * 2) * B32_SIZE * PING_PONG_NUM;

    UbAttnScore->tensorForStoreSecondAttnScoreFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += (MAX_LENG_PER_UB_PROC / 2 + BASE_BLOCK_SIDE_LEN) * B32_SIZE * PING_PONG_NUM;

    UbAttnScore->tensorForLoadOneBlockTriMaskFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += BASE_BLOCK_SIDE_LEN * B32_SIZE * PING_PONG_NUM * rowPerTime;

    UbAttnScore->tensorForCaclRowmaxFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += MAX_LENG_PER_UB_PROC * B32_SIZE * PING_PONG_NUM;

    UbAttnScore->tensorForCaclFinalRowmaxFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += BASE_BLOCK_SIDE_LEN * B32_SIZE * PING_PONG_NUM * rowPerTime;

    UbAttnScore->tensorForCaclSecondFinalRowmaxFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += BASE_BLOCK_SIDE_LEN * B32_SIZE * PING_PONG_NUM;

    UbAttnScore->tensorForCaclFinalRowsumFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += BASE_BLOCK_SIDE_LEN * B32_SIZE * PING_PONG_NUM * rowPerTime;

    UbAttnScore->tensorForCaclSecondFinalRowsumFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += BASE_BLOCK_SIDE_LEN * B32_SIZE * PING_PONG_NUM;

    UbAttnScore->tensorForVbrcbRowmaxFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += BASE_BLOCK_SIDE_LEN * B32_SIZE * PING_PONG_NUM;

    UbAttnScore->tensorForVbrcbSecondRowmaxFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += BASE_BLOCK_SIDE_LEN * B32_SIZE * PING_PONG_NUM;

    UbAttnScore->tensorForRecordRowmaxFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += 4096;

    UbAttnScore->tensorForRecordRowsumFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += 4096;
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::AllocateUbufForMediumSeqAttnScore(
    UB_FOR_MDDIUM_SEQ_ATTN_SCORE *UbAttnScore)
{
    int32_t offset = 512;

    UbAttnScore->tensorForStoreOneLineAttnScoreFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += MDDIUM_SEQ_THRESHOLD * B32_SIZE;

    auto maxLength = MAX_LENG_PER_UB_PROC > 4096 ? 4096 : MAX_LENG_PER_UB_PROC;

    // 32K
    UbAttnScore->tensorForLoadAttnScoreFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += maxLength * B32_SIZE * PING_PONG_NUM;

    // 16k
    UbAttnScore->tensorForCaclRowmaxFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += maxLength / 2 * B32_SIZE * PING_PONG_NUM;

    // 1K  vmax到最后，只需要64个FP32 （一行计算，不需要ping-pong)
    UbAttnScore->tensorForCaclFinalRowmaxFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += BASE_BLOCK_SIDE_LEN * B32_SIZE * PING_PONG_NUM;

    UbAttnScore->tensorForCaclFinalRowsumFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += BASE_BLOCK_SIDE_LEN * B32_SIZE * PING_PONG_NUM;
    // 1K
    UbAttnScore->tensorForVbrcbRowmaxFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += BASE_BLOCK_SIDE_LEN * B32_SIZE * PING_PONG_NUM;

    // 单行计算，一个FP32占32字节，最多64 * 32 = 2048， ping-pong翻倍
    UbAttnScore->tensorForRecordRowmaxFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += 4096;

    UbAttnScore->tensorForRecordRowsumFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += 4096;
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::AllocateUbufForLongSeqAttnScore(UB_FOR_LONG_SEQ_ATTN_SCORE *UbAttnScore)
{
    int32_t offset = 512;

    // 64K
    UbAttnScore->tensorForLoadAttnScoreFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += MAX_LENG_PER_UB_PROC * B32_SIZE * PING_PONG_NUM;

    // 64K
    UbAttnScore->tensorForLoadExtraFirstAttnScoreFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += MAX_LENG_PER_UB_PROC * B32_SIZE * PING_PONG_NUM;

    // sparse双mask
    UbAttnScore->tensorForLoadOneBlockTriMaskFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += BASE_BLOCK_SIDE_LEN * B32_SIZE * 4;

    // 32k
    UbAttnScore->tensorForCaclRowmaxFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += MAX_LENG_PER_UB_PROC / 2 * B32_SIZE * PING_PONG_NUM;

    // 1K  vmax到最后，只需要64个FP32
    UbAttnScore->tensorForCaclFinalRowmaxFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += BASE_BLOCK_SIDE_LEN * B16_SIZE * PING_PONG_NUM;

    UbAttnScore->tensorForCaclFinalRowsumFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += BASE_BLOCK_SIDE_LEN * B16_SIZE * PING_PONG_NUM;

    // 1K
    UbAttnScore->tensorForVbrcbRowmaxFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += BASE_BLOCK_SIDE_LEN * B32_SIZE * PING_PONG_NUM;

    // 2K BASE_BLOCK_SIDE_LEN(128)最少被2个vec处理，有64*2个最大值，每2个最大值占32B，需要2K空间
    UbAttnScore->tensorForRecordRowmaxFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += 2048;

    UbAttnScore->tensorForRecordRowsumFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += 2048;
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::AllocateUbufForNorm(UB_FOR_NORMALIZE *UbNorm)
{
    int32_t offset = 512;

    // 64k 8192个FP32 + ping-pong
    int32_t sizeofBufForLoadOFp32 = MAX_LENG_PER_UB_PROC * B32_SIZE * PING_PONG_NUM;
    // 0.5K 128个O对应1个rowsum
    int32_t sizeofBufForLoadRowsumFp32 = MAX_LENG_PER_UB_PROC / HEAD_DIM * B32_SIZE * PING_PONG_NUM;
    // 展开成32字节对齐
    int32_t sizeofBufForBrcbRowsumFp32 = MAX_LENG_PER_UB_PROC / HEAD_DIM * PING_PONG_NUM * B32_SIZE * (32 / B32_SIZE);

    UbNorm->tensorForLoadOFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += sizeofBufForLoadOFp32;

    UbNorm->tensorForLoadRowsumFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += sizeofBufForLoadRowsumFp32;

    UbNorm->tensorForBrcbRowsumFp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(offset);
    offset += sizeofBufForBrcbRowsumFp32;

    UbNorm->oPingPongInterval = MAX_LENG_PER_UB_PROC;
    UbNorm->rowsumPingPongInterval = MAX_LENG_PER_UB_PROC / HEAD_DIM;
    UbNorm->rowsumBrcbPingPongInterval = UbNorm->rowsumPingPongInterval * (32 / B32_SIZE);
}

/**
 * 获取全局信息
 * batchNum
 * headNum：全局处理Head的数量
 * yNumCubePerLine：Y值，每Y个CUBE CORE处理一个完整行
 * qkTriangle：QK_T是否是三角阵
 * VNum：sparse窗口长度
 * cubeNum：NPU上CUBE Core的数量（20\24\40\48...）
 * fCubeGroupNum：CUBE Core所有的组数量
 * blockNumPerFullLine：每个完整行包含基本块数量
 * blockNumPerHead：每个Head包含的基本块数量，当Q、K不等长时，与block_num_per_full_line值不同
 * loopTimesForAllFullLine：最外层循环的尾块，剩余的完整行数量（小于cube_group_num）
 * loopTailsForAllFullLine：最外层循环
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::GetGlobalInfo(
    int32_t batchNum, int32_t headNum, int32_t yNumCubePerLine, int32_t qSeqLen, int32_t kSeqLen, bool qkTriangle,
    int32_t VNum, int32_t *cubeNum, int32_t *fCubeGroupNum, int32_t *blockNumPerFullLine, int32_t *blockNumPerHead,
    int32_t *loopTimesForAllFullLine, int32_t *loopTailsForAllFullLine)
{
    *cubeNum = AscendC::GetBlockNum();
    *fCubeGroupNum = *cubeNum / yNumCubePerLine;

    int32_t fullLineNum = qSeqLen * headNum * batchNum / BASE_BLOCK_SIDE_LEN;
    if (VNum == 0) {
        *blockNumPerFullLine = kSeqLen / BASE_BLOCK_SIDE_LEN;

        if (qkTriangle == true) {
            fullLineNum = fullLineNum / 2;
            (*blockNumPerFullLine) += 1;
        }

        // 尾块计算 要保证是偶数，否则pingpong会把一个基本块拆成几个了
        *blockNumPerHead = fullLineNum / (headNum * batchNum);

        *loopTimesForAllFullLine = fullLineNum / (*fCubeGroupNum);
        *loopTailsForAllFullLine = fullLineNum % (*fCubeGroupNum);
        if (*loopTailsForAllFullLine > 0) {
            (*loopTimesForAllFullLine) += 1;
        }
    } else {
        *blockNumPerFullLine = VNum + 1;
        fullLineNum -= headNum * batchNum * VNum / 2;
        *blockNumPerHead = fullLineNum / (headNum * batchNum);
        *loopTimesForAllFullLine = fullLineNum / (*fCubeGroupNum);
        *loopTailsForAllFullLine = fullLineNum % (*fCubeGroupNum);
        if (*loopTailsForAllFullLine > 0) {
            (*loopTimesForAllFullLine) += 1;
        }
    }
}

/**
 * 获取每个vector的局部信息
 * yNumCubePerLine: Y值，每Y个CUBE CORE处理一个完整行
 * cubeGroupBelong：所属的CUBE组  -->  决定偏移N个完整行
 * cubeGroupInnderIdx：组内vector code的顺序  --> 决定完整行内偏移
 * eachVectorProcLineNum：每个vector处理的数量 （暂确定一定能整除）
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::GetLocalInfo(int32_t yNumCubePerLine, int32_t *cubeGroupBelong,
                                                           int32_t *cubeGroupInnderIdx, int32_t *eachVectorProcLineNum)
{
    int32_t cubeIdx = get_block_idx();
    *cubeGroupBelong = cubeIdx / yNumCubePerLine;
    *cubeGroupInnderIdx = (cubeIdx % yNumCubePerLine) * 2 + get_subblockid();

    *eachVectorProcLineNum = BASE_BLOCK_SIDE_LEN / (yNumCubePerLine * 2);
    // 若不能整除，最后几个vector core会多处理一些行 - 当前不考虑
}

/**
 * 获取当前vector core的地址偏移信息，需要区分左右section
 * eachVectorStartLineOffset：整体的起始偏移(N块完整行的偏移 + 块内完整行的偏移)
 * triMatrixMaskOffset：128*128的mask矩阵的偏移
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::GetInitOffsetForCurrentCore(
    int32_t cubeGroupBelong, int32_t cubeGroupInnderIdx, int32_t blockNumPerFullLine, int32_t eachVectorProcLineNum,
    int32_t *eachVectorStartLineOffset, int32_t *triMatrixMaskOffset)
{
    // 一块完整行（128行）需要跳过的数据量  （128^2 * 5）
    int32_t fullLineOffset = BASE_BLOCK_SIDE_LEN * (BASE_BLOCK_SIDE_LEN * blockNumPerFullLine);
    // 当前vector core属于第cube_group_belong个分组，需要跳过前面所有块的完整行 (0)
    int32_t interGroupOffset = cubeGroupBelong * fullLineOffset;

    // 当前vector core处理的起始行 (0/64)
    int32_t innerGroupStartLine = eachVectorProcLineNum * cubeGroupInnderIdx;
    // 每行第一个基本块的宽度为BASE_BLOCK_SIZE (0*128 / 64*128)
    int32_t innerGroupStartLineOffset = BASE_BLOCK_SIDE_LEN * innerGroupStartLine;

    *eachVectorStartLineOffset = (interGroupOffset + innerGroupStartLineOffset);
    *triMatrixMaskOffset = innerGroupStartLine * maskSeqLength;
}

/**
 * UB每次处理信号的长度，三角阵拼接后，即使按左右两个section计算，也按最长的预留空间
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ inline void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::GetSubSeqLengthPerProc(int32_t kSeqLen, int32_t blockNumPerFullLine,
                                                                     int32_t *subSeqLengthPerProc)
{
    // 分两个section处理，不会总是2的幂；求max时用折半的方法，就不行
    *subSeqLengthPerProc = kSeqLen > MAX_LENG_PER_UB_PROC ? MAX_LENG_PER_UB_PROC : kSeqLen;

    // 序列小于MAX_LENG_PER_UB_PROC, 需要减半以支持ping-pong
    if (*subSeqLengthPerProc < MAX_LENG_PER_UB_PROC && blockNumPerFullLine > 1) {
        *subSeqLengthPerProc = *subSeqLengthPerProc / 2; // (256)
    }
}

/**
 * 非2的幂次长度，为了折半求vmax，需要进行padding
 * totalBlockNum：当前总共需要处理的block数量
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::GetPaddingInfoForRowMax(int32_t totalBlockNum, int32_t *paddingBlockNum)
{
    // 满足最大长度倍数的部分不需要padding
    auto tailNum = totalBlockNum % BLOCK_NUM_FOR_VMAX;

    if (tailNum == 0) {
        *paddingBlockNum = 0;
        return;
    }

    int32_t totalBlock = 2;

    while (totalBlock < BLOCK_NUM_FOR_VMAX) {
        if (tailNum <= totalBlock) {
            break;
        }
        totalBlock *= 2;
    }

    *paddingBlockNum = totalBlock - tailNum;
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::PaddingForRowMaxTensor(
    int32_t totalBlockNum, int32_t paddingBlockNum, const LocalTensor<float> &paddingTensor, float value)
{
    if (paddingBlockNum == 0) {
        return;
    }

    auto tailNum = totalBlockNum % MAX_BLOCK_PER_ONE_PROC;

    AscendC::Duplicate(paddingTensor[tailNum * BASE_BLOCK_SIDE_LEN * rowPerTime], float(value), 64,
                       paddingBlockNum * 2 * rowPerTime, 1, 8);
    PipeBarrier<PIPE_V>();
}

/**
 * 当前处理的完整行在head中属于第几个完整行  --> 找到需要mask的块 （三角阵有两个块）
 * curLoopIndex：当前的大循环编号
 * cubeGroupBelong：所属的cube分组
 * fCubeGroupNum：F, cube总分组数量
 * blockNumPerHead：一个head包含的块行数量
 * *blockLineOffsetInOneHead：在Head中的完整行ID(三角阵中是拼接后的行号)
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::GetFullLineOffsetInOneHead(int32_t curLoopIndex, int32_t cubeGroupBelong,
                                                                         int32_t fCubeGroupNum, int32_t blockNumPerHead,
                                                                         int32_t *blockLineOffsetInOneHead)
{
    // 所有head依照 B*N顺序排列，每个loop能处理fCubeGroupNum个完整块行，当前处理的完整块行的编号 (全局处理的第x行)
    int32_t fullLineIdx = curLoopIndex * fCubeGroupNum + cubeGroupBelong;

    // 在Head内的第X个完整行
    *blockLineOffsetInOneHead = fullLineIdx % blockNumPerHead;
}


/**
 * 只有三角阵有tri block问题；分为左右两个section，需要分别计算softmax
 * tri block的位置由section_end表达
 * blockLineOffsetInOneHead: Head中第x个基本块行
 * *sectionOneStartBlockOffset: 第一段起始
 * *sectionTwoStartBlockOffset: 第二段起始
 * *sectionOneEndBlockOffset:   第一段结束（三角阵位置）
 * *sectionTwoEndBlockOffset):  第二段结束（三角阵位置）
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::GetTriMatrixSectionBlockOffset(
    int32_t blockLineOffsetInOneHead, int32_t blockNumPerFullLine, int32_t *sectionOneStartBlockOffset,
    int32_t *sectionTwoStartBlockOffset, int32_t *sectionOneEndBlockOffset, int32_t *sectionTwoEndBlockOffset)
{
    *sectionOneStartBlockOffset = 0;
    *sectionTwoStartBlockOffset = blockNumPerFullLine / 2 + 1 + blockLineOffsetInOneHead;

    *sectionOneEndBlockOffset = *sectionTwoStartBlockOffset - 1;
    *sectionTwoEndBlockOffset = blockNumPerFullLine - 1;
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::CaclMaxTensor(const LocalTensor<float> &tensorForCacl, int32_t blockNum)
{
    int32_t curBlockNum = blockNum;

    while (curBlockNum > 1) {
        // fp32
        max_v<ArchType::ASCEND_V220>(tensorForCacl, tensorForCacl,
                                     tensorForCacl[BASE_BLOCK_SIDE_LEN * curBlockNum * rowPerTime],
                                     curBlockNum * 2 * rowPerTime, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();

        curBlockNum = curBlockNum / 2;
    }

    max_v<ArchType::ASCEND_V220>(tensorForCacl, tensorForCacl, tensorForCacl[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                 2 * rowPerTime, 1, 1, 1, 8, 8, 8);
    PipeBarrier<PIPE_V>();
}

/**
 * 64个基本块以下求max
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ __attribute__((noinline)) void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::ProcessCaclMaxTensor(
    int32_t basicBlockNum, int32_t paddingBlockNum, bool ppFirstSection,
    const LocalTensor<float> &curTensorForAttnScore, const LocalTensor<float> &curTensorForRowmax,
    const LocalTensor<float> &tensorForCaclFinalRowmaxFp32)
{
    int32_t allBlockNum = basicBlockNum + paddingBlockNum;
    int32_t tailBlockNum = allBlockNum % BLOCK_NUM_FOR_VMAX;
    int32_t doneBlockNum = allBlockNum / BLOCK_NUM_FOR_VMAX * BLOCK_NUM_FOR_VMAX;
    bool fromBufForAttnScore = false;

    if (allBlockNum == 64) {
        // fp32
        max_v<ArchType::ASCEND_V220>(curTensorForRowmax, curTensorForAttnScore,
                                     curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * 32], 64, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        CaclMaxTensor(curTensorForRowmax, BLOCK_NUM_FOR_VMAX);
    } else if (allBlockNum >= 48) {
        // 48(0)\50(2)\52(4)\56(8)  fp32
        max_v<ArchType::ASCEND_V220>(curTensorForRowmax, curTensorForAttnScore,
                                     curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * 16], 32, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        max_v<ArchType::ASCEND_V220>(curTensorForRowmax, curTensorForRowmax,
                                     curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * 32], 32, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        max_v<ArchType::ASCEND_V220>(curTensorForRowmax, curTensorForRowmax,
                                     curTensorForRowmax[BASE_BLOCK_SIDE_LEN * 8], 16, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        CaclMaxTensor(curTensorForRowmax, 4);
    } else if (allBlockNum >= 32) {
        // 32(0)\34(2)\36(4)\40(8)  fp32
        max_v<ArchType::ASCEND_V220>(curTensorForRowmax, curTensorForAttnScore,
                                     curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * 16 * rowPerTime], 32 * rowPerTime, 1,
                                     1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        CaclMaxTensor(curTensorForRowmax, 8);
    } else if (allBlockNum >= 16) {
        // 16(0)\18(2)\20(4)\24(8)  fp32
        max_v<ArchType::ASCEND_V220>(curTensorForRowmax, curTensorForAttnScore,
                                     curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * 8 * rowPerTime], 16 * rowPerTime, 1, 1,
                                     1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        CaclMaxTensor(curTensorForRowmax, 4);
    }

    if (tailBlockNum == 8) {
        if (allBlockNum < 16) {
            max_v<ArchType::ASCEND_V220>(curTensorForRowmax,
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * doneBlockNum * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * (doneBlockNum + 4) * rowPerTime],
                                         8 * rowPerTime, 1, 1, 1, 8, 8, 8);
            PipeBarrier<PIPE_V>();

            max_v<ArchType::ASCEND_V220>(curTensorForRowmax, curTensorForRowmax,
                                         curTensorForRowmax[BASE_BLOCK_SIDE_LEN * 2 * rowPerTime], 4 * rowPerTime, 1, 1,
                                         1, 8, 8, 8);
            PipeBarrier<PIPE_V>();
        } else {
            max_v<ArchType::ASCEND_V220>(curTensorForRowmax[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * doneBlockNum * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * (doneBlockNum + 4) * rowPerTime],
                                         8 * rowPerTime, 1, 1, 1, 8, 8, 8);
            PipeBarrier<PIPE_V>();

            max_v<ArchType::ASCEND_V220>(curTensorForRowmax[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForRowmax[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForRowmax[BASE_BLOCK_SIDE_LEN * 3 * rowPerTime], 4 * rowPerTime, 1, 1,
                                         1, 8, 8, 8);
            PipeBarrier<PIPE_V>();

            max_v<ArchType::ASCEND_V220>(curTensorForRowmax[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForRowmax[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForRowmax[BASE_BLOCK_SIDE_LEN * 2 * rowPerTime], 2 * rowPerTime, 1, 1,
                                         1, 8, 8, 8);
            PipeBarrier<PIPE_V>();
        }
        // fp32  剩下128个FP32, 2组
        max_v<ArchType::ASCEND_V220>(curTensorForRowmax, curTensorForRowmax,
                                     curTensorForRowmax[BASE_BLOCK_SIDE_LEN * rowPerTime], 2 * rowPerTime, 1, 1, 1, 8,
                                     8, 8);
        PipeBarrier<PIPE_V>();
    } else if (tailBlockNum == 4) {
        if (allBlockNum < 16) {
            max_v<ArchType::ASCEND_V220>(curTensorForRowmax,
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * doneBlockNum * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * (doneBlockNum + 2) * rowPerTime],
                                         4 * rowPerTime, 1, 1, 1, 8, 8, 8);
            PipeBarrier<PIPE_V>();
        } else {
            max_v<ArchType::ASCEND_V220>(curTensorForRowmax[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * doneBlockNum * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * (doneBlockNum + 2) * rowPerTime],
                                         4 * rowPerTime, 1, 1, 1, 8, 8, 8);
            PipeBarrier<PIPE_V>();

            max_v<ArchType::ASCEND_V220>(curTensorForRowmax[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForRowmax[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForRowmax[BASE_BLOCK_SIDE_LEN * 2 * rowPerTime], 2 * rowPerTime, 1, 1,
                                         1, 8, 8, 8);
            PipeBarrier<PIPE_V>();
        }

        max_v<ArchType::ASCEND_V220>(curTensorForRowmax, curTensorForRowmax,
                                     curTensorForRowmax[BASE_BLOCK_SIDE_LEN * rowPerTime], 2 * rowPerTime, 1, 1, 1, 8,
                                     8, 8);
        PipeBarrier<PIPE_V>();
    } else if (tailBlockNum == 2) {
        if (allBlockNum < 16) {
            max_v<ArchType::ASCEND_V220>(curTensorForRowmax,
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * doneBlockNum * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * (doneBlockNum + 1) * rowPerTime],
                                         2 * rowPerTime, 1, 1, 1, 8, 8, 8);
            PipeBarrier<PIPE_V>();
        } else {
            max_v<ArchType::ASCEND_V220>(curTensorForRowmax[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * doneBlockNum * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * (doneBlockNum + 1) * rowPerTime],
                                         2 * rowPerTime, 1, 1, 1, 8, 8, 8);
            PipeBarrier<PIPE_V>();

            max_v<ArchType::ASCEND_V220>(curTensorForRowmax, curTensorForRowmax,
                                         curTensorForRowmax[BASE_BLOCK_SIDE_LEN * rowPerTime], 2 * rowPerTime, 1, 1, 1,
                                         8, 8, 8);
            PipeBarrier<PIPE_V>();
        }
    } // 没有其他分支了

    auto srcBuf = fromBufForAttnScore ? curTensorForAttnScore : curTensorForRowmax;

    if (ppFirstSection) {
        max_v<ArchType::ASCEND_V220>(tensorForCaclFinalRowmaxFp32, srcBuf, srcBuf[BASE_BLOCK_SIDE_LEN / 2], rowPerTime,
                                     1, 1, 1, 8, 8 * rowPerTime, 8 * rowPerTime);
        PipeBarrier<PIPE_V>();
    } else {
        max_v<ArchType::ASCEND_V220>(curTensorForRowmax, srcBuf, srcBuf[BASE_BLOCK_SIDE_LEN / 2], 1, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();

        max_v<ArchType::ASCEND_V220>(tensorForCaclFinalRowmaxFp32, tensorForCaclFinalRowmaxFp32, curTensorForRowmax, 1,
                                     1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
    }
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::CaclSumTensor(const LocalTensor<float> &tensorForCacl, int32_t blockNum)
{
    int32_t curBlockNum = blockNum;

    while (curBlockNum > 1) {
        // fp32
        add_v<ArchType::ASCEND_V220>(tensorForCacl, tensorForCacl,
                                     tensorForCacl[BASE_BLOCK_SIDE_LEN * curBlockNum * rowPerTime],
                                     curBlockNum * 2 * rowPerTime, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();

        curBlockNum = curBlockNum / 2;
    }

    add_v<ArchType::ASCEND_V220>(tensorForCacl, tensorForCacl, tensorForCacl[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                 2 * rowPerTime, 1, 1, 1, 8, 8, 8);
    PipeBarrier<PIPE_V>();
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ __attribute__((noinline)) void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::ProcessCaclSumTensor(
    int32_t basicBlockNum, int32_t paddingBlockNum, bool ppFirstSection,
    const LocalTensor<float> &curTensorForAttnScore, const LocalTensor<float> &curTensorForRowsum,
    const LocalTensor<float> &tensorForCaclFinalRowsumFp32)
{
    int32_t allBlockNum = basicBlockNum + paddingBlockNum;
    int32_t tailBlockNum = allBlockNum % BLOCK_NUM_FOR_VMAX;
    int32_t doneBlockNum = allBlockNum / BLOCK_NUM_FOR_VMAX * BLOCK_NUM_FOR_VMAX;
    bool fromBufForAttnScore = false;

    if (allBlockNum == 64) {
        // fp32
        add_v<ArchType::ASCEND_V220>(curTensorForRowsum, curTensorForAttnScore,
                                     curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * 32], 64, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        CaclSumTensor(curTensorForRowsum, BLOCK_NUM_FOR_VMAX);
    } else if (allBlockNum >= 48) {
        // 48(0)\50(2)\52(4)\56(8)  fp32
        add_v<ArchType::ASCEND_V220>(curTensorForRowsum, curTensorForAttnScore,
                                     curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * 16], 32, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        add_v<ArchType::ASCEND_V220>(curTensorForRowsum, curTensorForRowsum,
                                     curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * 32], 32, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        add_v<ArchType::ASCEND_V220>(curTensorForRowsum, curTensorForRowsum,
                                     curTensorForRowsum[BASE_BLOCK_SIDE_LEN * 8], 16, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        CaclSumTensor(curTensorForRowsum, 4);
    } else if (allBlockNum >= 32) {
        // 32(0)\34(2)\36(4)\40(8)  fp32
        add_v<ArchType::ASCEND_V220>(curTensorForRowsum, curTensorForAttnScore,
                                     curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * 16 * rowPerTime], 32 * rowPerTime, 1,
                                     1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        CaclSumTensor(curTensorForRowsum, 8);
    } else if (allBlockNum >= 16) {
        // 16(0)\18(2)\20(4)\24(8)  fp32
        add_v<ArchType::ASCEND_V220>(curTensorForRowsum, curTensorForAttnScore,
                                     curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * 8 * rowPerTime], 16 * rowPerTime, 1, 1,
                                     1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        CaclSumTensor(curTensorForRowsum, 4);
    }

    if (tailBlockNum == 8) {
        if (allBlockNum < 16) {
            add_v<ArchType::ASCEND_V220>(curTensorForRowsum,
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * doneBlockNum * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * (doneBlockNum + 4) * rowPerTime],
                                         8 * rowPerTime, 1, 1, 1, 8, 8, 8);
            PipeBarrier<PIPE_V>();

            add_v<ArchType::ASCEND_V220>(curTensorForRowsum, curTensorForRowsum,
                                         curTensorForRowsum[BASE_BLOCK_SIDE_LEN * 2 * rowPerTime], 4 * rowPerTime, 1, 1,
                                         1, 8, 8, 8);
            PipeBarrier<PIPE_V>();
        } else {
            add_v<ArchType::ASCEND_V220>(curTensorForRowsum[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * doneBlockNum * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * (doneBlockNum + 4) * rowPerTime],
                                         8 * rowPerTime, 1, 1, 1, 8, 8, 8);
            PipeBarrier<PIPE_V>();

            add_v<ArchType::ASCEND_V220>(curTensorForRowsum[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForRowsum[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForRowsum[BASE_BLOCK_SIDE_LEN * 3 * rowPerTime], 4 * rowPerTime, 1, 1,
                                         1, 8, 8, 8);
            PipeBarrier<PIPE_V>();

            add_v<ArchType::ASCEND_V220>(curTensorForRowsum[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForRowsum[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForRowsum[BASE_BLOCK_SIDE_LEN * 2 * rowPerTime], 2 * rowPerTime, 1, 1,
                                         1, 8, 8, 8);
            PipeBarrier<PIPE_V>();
        }
        // fp32  剩下128个FP32, 2组
        add_v<ArchType::ASCEND_V220>(curTensorForRowsum, curTensorForRowsum,
                                     curTensorForRowsum[BASE_BLOCK_SIDE_LEN * rowPerTime], 2 * rowPerTime, 1, 1, 1, 8,
                                     8, 8);
        PipeBarrier<PIPE_V>();
    } else if (tailBlockNum == 4) {
        if (allBlockNum < 16) {
            add_v<ArchType::ASCEND_V220>(curTensorForRowsum,
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * doneBlockNum * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * (doneBlockNum + 2) * rowPerTime],
                                         4 * rowPerTime, 1, 1, 1, 8, 8, 8);
            PipeBarrier<PIPE_V>();
        } else {
            add_v<ArchType::ASCEND_V220>(curTensorForRowsum[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * doneBlockNum * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * (doneBlockNum + 2) * rowPerTime],
                                         4 * rowPerTime, 1, 1, 1, 8, 8, 8);
            PipeBarrier<PIPE_V>();

            add_v<ArchType::ASCEND_V220>(curTensorForRowsum[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForRowsum[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForRowsum[BASE_BLOCK_SIDE_LEN * 2 * rowPerTime], 2 * rowPerTime, 1, 1,
                                         1, 8, 8, 8);
            PipeBarrier<PIPE_V>();
        }

        add_v<ArchType::ASCEND_V220>(curTensorForRowsum, curTensorForRowsum,
                                     curTensorForRowsum[BASE_BLOCK_SIDE_LEN * rowPerTime], 2 * rowPerTime, 1, 1, 1, 8,
                                     8, 8);
        PipeBarrier<PIPE_V>();
    } else if (tailBlockNum == 2) {
        if (allBlockNum < 16) {
            add_v<ArchType::ASCEND_V220>(curTensorForRowsum,
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * doneBlockNum * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * (doneBlockNum + 1) * rowPerTime],
                                         2 * rowPerTime, 1, 1, 1, 8, 8, 8);
            PipeBarrier<PIPE_V>();
        } else {
            add_v<ArchType::ASCEND_V220>(curTensorForRowsum[BASE_BLOCK_SIDE_LEN * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * doneBlockNum * rowPerTime],
                                         curTensorForAttnScore[BASE_BLOCK_SIDE_LEN * (doneBlockNum + 1) * rowPerTime],
                                         2 * rowPerTime, 1, 1, 1, 8, 8, 8);
            PipeBarrier<PIPE_V>();

            add_v<ArchType::ASCEND_V220>(curTensorForRowsum, curTensorForRowsum,
                                         curTensorForRowsum[BASE_BLOCK_SIDE_LEN * rowPerTime], 2 * rowPerTime, 1, 1, 1,
                                         8, 8, 8);
            PipeBarrier<PIPE_V>();
        }
    } // 没有其他分支了

    auto srcBuf = fromBufForAttnScore ? curTensorForAttnScore : curTensorForRowsum;

    if (ppFirstSection) {
        add_v<ArchType::ASCEND_V220>(tensorForCaclFinalRowsumFp32, srcBuf, srcBuf[BASE_BLOCK_SIDE_LEN / 2], rowPerTime,
                                     1, 1, 1, 8, 8 * rowPerTime, 8 * rowPerTime);
        PipeBarrier<PIPE_V>();
    } else {
        add_v<ArchType::ASCEND_V220>(curTensorForRowsum, srcBuf, srcBuf[BASE_BLOCK_SIDE_LEN / 2], 1, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();

        add_v<ArchType::ASCEND_V220>(tensorForCaclFinalRowsumFp32, tensorForCaclFinalRowsumFp32, curTensorForRowsum, 1,
                                     1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
    }
}
/**
 * 8192长度以下的序列计算max
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::ProcessLinePhaseOneForShortSeqMax(
    bool qkTriangle, PARAM_SHORT_SEQ_MAX Param, int32_t pingPongFlag, bool firstLine, __gm__ WORKSPACE_T *attnScoreGm,
    __gm__ INPUT_T *attnMaskGm, UB_FOR_SHORT_LEN_ATTN_SCORE &UbAttn)
{
    // 8192段序列，一定会遇到末尾三角阵
    auto totalBlockNum = Param.sectionOneBlockNum + Param.sectionTwoBlockNum;
    auto eventId = (pingPongFlag == 0 ? EVENT_ID0 : EVENT_ID1);

    // 原始atten score
    LocalTensor<float> curTensorForAttnScore =
        pingPongFlag == 0 ? UbAttn.tensorForLoadAttnScoreFp32[0] :
                            UbAttn.tensorForLoadAttnScoreFp32[MAX_LENG_PER_UB_PROC + BASE_BLOCK_SIDE_LEN * rowPerTime];

    LocalTensor<float> curTensorForSecondAttnScore =
        pingPongFlag == 0 ? UbAttn.tensorForStoreSecondAttnScoreFp32[0] :
                            UbAttn.tensorForStoreSecondAttnScoreFp32[MAX_LENG_PER_UB_PROC / 2];

    // 这里可以复用，最后的结果存在 buf_for_cacl_short_(second)_final_rowmax_fp16
    LocalTensor<float> curTensorForRowmax =
        pingPongFlag == 0 ? UbAttn.tensorForCaclRowmaxFp32[0] : UbAttn.tensorForCaclRowmaxFp32[MAX_LENG_PER_UB_PROC];

    // 得到最后128个最大值 (section one & tow 会连着存放)
    LocalTensor<float> curTensorForFinalRowmax =
        pingPongFlag == 0 ? UbAttn.tensorForCaclFinalRowmaxFp32[0] :
                            UbAttn.tensorForCaclFinalRowmaxFp32[BASE_BLOCK_SIDE_LEN * rowPerTime];
    LocalTensor<float> curTensorForSecondFinalRowmax =
        pingPongFlag == 0 ? UbAttn.tensorForCaclFinalRowmaxFp32[BASE_BLOCK_SIDE_LEN / 2 * rowPerTime] :
                            UbAttn.tensorForCaclFinalRowmaxFp32[BASE_BLOCK_SIDE_LEN * 3 / 2 * rowPerTime];

    // 32B对齐的最大值
    LocalTensor<float> curTensorForVbrcbRowmaxFp32 =
        pingPongFlag == 0 ? UbAttn.tensorForVbrcbRowmaxFp32[0] : UbAttn.tensorForVbrcbRowmaxFp32[BASE_BLOCK_SIDE_LEN];

    // mask
    LocalTensor<float> curTensorForMaskFp32 =
        pingPongFlag == 0 ? UbAttn.tensorForLoadOneBlockTriMaskFp32[0] :
                            UbAttn.tensorForLoadOneBlockTriMaskFp32[BASE_BLOCK_SIDE_LEN * rowPerTime];
    LocalTensor<INPUT_T> curTensorForMaskFp16 =
        curTensorForMaskFp32[64 * rowPerTime].template ReinterpretCast<INPUT_T>();

    // 存储最大值，最终需要返回到GM上
    LocalTensor<float> curTensorForRecordRowmax =
        pingPongFlag == 0 ? UbAttn.tensorForRecordRowmaxFp32[0] : UbAttn.tensorForRecordRowmaxFp32[512];
    GlobalTensor<float> tmpScoreGmTensor;
    tmpScoreGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(attnScoreGm + Param.sectionStartLineOffset));
    int32_t srcGap = rowPerTime == 1 ? 0 : maskSeqLength - BASE_BLOCK_SIDE_LEN;
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);

    AscendC::DataCopy(curTensorForAttnScore, tmpScoreGmTensor,
                      AscendC::DataCopyParams(totalBlockNum, BASE_BLOCK_SIDE_LEN * rowPerTime / 8,
                                              (BASE_BLOCK_DATA_NUM - rowPerTime * BASE_BLOCK_SIDE_LEN) / 8, 0));
    // 一定有两个tri matrix
    if (qkTriangle == true) {
        gm_to_ub<ArchType::ASCEND_V220, INPUT_T>(curTensorForMaskFp16, attnMaskGmTensor[Param.sectionMaskOffset],
                                                 0,                        // sid
                                                 rowPerTime,               // nBurst
                                                 BASE_BLOCK_SIDE_LEN / 16, // lenBurst
                                                 srcGap / 16,              // srcGap
                                                 0                         // dstGap
        );
    }
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);

    muls_v<ArchType::ASCEND_V220, float>(curTensorForAttnScore, curTensorForAttnScore, (float)scale,
                                         totalBlockNum * 2 * rowPerTime, // repeat
                                         1,                              // dstBlockStride
                                         1,                              // srcBlockStride
                                         8,                              // dstRepeatStride
                                         8                               // srcRepeatStride
    );
    PipeBarrier<PIPE_V>();
    if (qkTriangle == true) {
        // 处理tri matrix
        conv_v<ArchType::ASCEND_V220, INPUT_T, float>(curTensorForMaskFp32, curTensorForMaskFp16, rowPerTime * 2, 1, 1,
                                                      8, 4);
        PipeBarrier<PIPE_V>();

        // 128个fp32 --> 2个repeat
        muls_v<ArchType::ASCEND_V220, float>(curTensorForMaskFp32, curTensorForMaskFp32, (float)PADDING_FOR_MAX,
                                             2 * rowPerTime, // repeat
                                             1,              // dstBlockStride
                                             1,              // srcBlockStride
                                             8,              // dstRepeatStride
                                             8               // srcRepeatStride
        );
        PipeBarrier<PIPE_V>();

        add_v<ArchType::ASCEND_V220, float>(
            curTensorForAttnScore[rowPerTime * (Param.sectionOneBlockNum - 1) * BASE_BLOCK_SIDE_LEN],
            curTensorForAttnScore[rowPerTime * (Param.sectionOneBlockNum - 1) * BASE_BLOCK_SIDE_LEN],
            curTensorForMaskFp32,
            rowPerTime * 2, // repeat
            1,              // dstBlockStride
            1,              // src0BlockStride
            1,              // src1BlockStride
            8,              // dstRepeatStride
            8,              // src0RepeatStride
            8               // src1RepeatStride
        );
        PipeBarrier<PIPE_V>();

        // 直接写到目的地
        add_v<ArchType::ASCEND_V220, float>(
            curTensorForSecondAttnScore[rowPerTime * (Param.sectionTwoBlockNum - 1) * BASE_BLOCK_SIDE_LEN],
            curTensorForAttnScore[rowPerTime * (totalBlockNum - 1) * BASE_BLOCK_SIDE_LEN], curTensorForMaskFp32,
            rowPerTime * 2, // repeat
            1,              // dstBlockStride
            1,              // src0BlockStride
            1,              // src1BlockStride
            8,              // dstRepeatStride
            8,              // src0RepeatStride
            8               // src1RepeatStride
        );
        PipeBarrier<PIPE_V>();
    }

    // copy section tow
    if (Param.sectionTwoBlockNum > 1) {
        // 这个省不掉，相加一次折半的时候，可能会需要用到padding的值
        AscendC::Copy(curTensorForSecondAttnScore[0],
                      curTensorForAttnScore[Param.sectionOneBlockNum * BASE_BLOCK_SIDE_LEN * rowPerTime], 64,
                      rowPerTime * 2 * (Param.sectionTwoBlockNum - 1), {1, 1, 8, 8});
    }
    PipeBarrier<PIPE_V>();

    // 计算section one（每次都需要padding，因为会被section two污染）
    if (Param.sectionOnePaddingBlockNum > 0) {
        PaddingForRowMaxTensor(Param.sectionOneBlockNum, Param.sectionOnePaddingBlockNum, curTensorForAttnScore,
                               PADDING_FOR_MAX);
    }

    // 最后的结果在 curBufForFinalRowmax
    ProcessCaclMaxTensor(Param.sectionOneBlockNum, Param.sectionOnePaddingBlockNum, true, curTensorForAttnScore,
                         curTensorForRowmax, curTensorForFinalRowmax);

    if (qkTriangle == true) {
        // 计算section two （处理第一行的时候padding 0即可，因为所有行的padding都一样，不会被污染）
        if (Param.sectionTwoPaddingBlockNum > 0) {
            PaddingForRowMaxTensor(Param.sectionTwoBlockNum, Param.sectionTwoPaddingBlockNum,
                                   curTensorForSecondAttnScore, PADDING_FOR_MAX);
        }

        ProcessCaclMaxTensor(Param.sectionTwoBlockNum, Param.sectionTwoPaddingBlockNum, true,
                             curTensorForSecondAttnScore, curTensorForRowmax, curTensorForSecondFinalRowmax);

        // 接着计算max，有2个最大值(两段)
        cmax_v<ArchType::ASCEND_V220, float, ReduceOrder::ORDER_ONLY_VALUE>(
            curTensorForRecordRowmax[Param.recordRowmaxOffset], curTensorForFinalRowmax, rowPerTime * 2, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        brcb_v<ArchType::ASCEND_V220, uint32_t>(
            curTensorForVbrcbRowmaxFp32.ReinterpretCast<uint32_t>(),
            curTensorForRecordRowmax[Param.recordRowmaxOffset].template ReinterpretCast<uint32_t>(),
            1, // dstBlockStride
            8, // dstRepeatStride
            1  // repeat
        );
        PipeBarrier<PIPE_V>();
    } else {
        // 有2个最大值(两段)
        cmax_v<ArchType::ASCEND_V220, float, ReduceOrder::ORDER_ONLY_VALUE>(
            curTensorForRecordRowmax[Param.recordRowmaxOffset], curTensorForFinalRowmax, rowPerTime, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        brcb_v<ArchType::ASCEND_V220, uint32_t>(
            curTensorForVbrcbRowmaxFp32.ReinterpretCast<uint32_t>(),
            curTensorForRecordRowmax[Param.recordRowmaxOffset].template ReinterpretCast<uint32_t>(),
            1, // dstBlockStride
            8, // dstRepeatStride
            1  // repeat
        );
        PipeBarrier<PIPE_V>();
    }
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::ProcessLinePhaseOneForShortSeqExp(
    bool qkTriangle, PARAM_SHORT_SEQ_MAX Param, int32_t pingPongFlag, __gm__ WORKSPACE_T *attnScoreGm,
    __gm__ INPUT_T *attnMaskGm, UB_FOR_SHORT_LEN_ATTN_SCORE &UbAttn, int32_t offset)
{
    auto eventId = (pingPongFlag == 0 ? EVENT_ID0 : EVENT_ID1);

    LocalTensor<float> curTensorForAttnScore =
        pingPongFlag == 0 ? UbAttn.tensorForLoadAttnScoreFp32[0] :
                            UbAttn.tensorForLoadAttnScoreFp32[MAX_LENG_PER_UB_PROC + BASE_BLOCK_SIDE_LEN * rowPerTime];
    LocalTensor<float> curTensorForSecondAttnScore =
        pingPongFlag == 0 ? UbAttn.tensorForStoreSecondAttnScoreFp32[0] :
                            UbAttn.tensorForStoreSecondAttnScoreFp32[MAX_LENG_PER_UB_PROC / 2];

    LocalTensor<float> curTensorForVbrcbRowmaxFp32 =
        pingPongFlag == 0 ? UbAttn.tensorForVbrcbRowmaxFp32[0] : UbAttn.tensorForVbrcbRowmaxFp32[BASE_BLOCK_SIDE_LEN];
    // 用rowmax空间计算rowsum
    LocalTensor<float> curTensorForRowmax =
        pingPongFlag == 0 ? UbAttn.tensorForCaclRowmaxFp32[0] : UbAttn.tensorForCaclRowmaxFp32[MAX_LENG_PER_UB_PROC];
    LocalTensor<float> curTensorForFinalRowsum =
        pingPongFlag == 0 ? UbAttn.tensorForCaclFinalRowsumFp32[0] :
                            UbAttn.tensorForCaclFinalRowsumFp32[BASE_BLOCK_SIDE_LEN * rowPerTime];
    LocalTensor<float> curTensorForSecondFinalRowsum =
        pingPongFlag == 0 ? UbAttn.tensorForCaclFinalRowsumFp32[BASE_BLOCK_SIDE_LEN / 2 * rowPerTime] :
                            UbAttn.tensorForCaclFinalRowsumFp32[BASE_BLOCK_SIDE_LEN * 3 / 2 * rowPerTime];

    LocalTensor<float> curTensorForRecordRowsum =
        pingPongFlag == 0 ? UbAttn.tensorForRecordRowsumFp32[0] : UbAttn.tensorForRecordRowsumFp32[512];
    // fp16 result直接用exp空间
    LocalTensor<INPUT_T> curTensorForResultFp16 = curTensorForAttnScore.template ReinterpretCast<INPUT_T>();

    int32_t totalOffsetFp32 = Param.sectionStartLineOffset - offset + offset / 2;
    GlobalTensor<INPUT_T> tmpScoreGmTensorOut;
    tmpScoreGmTensorOut.SetGlobalBuffer(
        reinterpret_cast<__gm__ INPUT_T *>((__gm__ INPUT_T *)(attnScoreGm + totalOffsetFp32)));

    // 减去最大值
    if (rowPerTime == 1) {
        sub_v<ArchType::ASCEND_V220, float>(curTensorForAttnScore, curTensorForAttnScore, curTensorForVbrcbRowmaxFp32,
                                            Param.sectionOneBlockNum * 2, // repeat
                                            1,                            // dstBlockStride
                                            1,                            // src0BlockStride
                                            0,                            // src1BlockStride
                                            8,                            // dstRepeatStride
                                            8,                            // src0RepeatStride
                                            0                             // src1RepeatStride
        );
        PipeBarrier<PIPE_V>();
    } else {
        for (int32_t i = 0; i < 2 * rowPerTime; i++) {
            sub_v<ArchType::ASCEND_V220, float>(curTensorForAttnScore[i * 64], curTensorForAttnScore[i * 64],
                                                curTensorForVbrcbRowmaxFp32[(i / 2) * 8],
                                                Param.sectionOneBlockNum, // repeat
                                                1,                        // dstBlockStride
                                                1,                        // src0BlockStride
                                                0,                        // src1BlockStride
                                                32,                       // dstRepeatStride
                                                32,                       // src0RepeatStride
                                                0                         // src1RepeatStride
            );
            PipeBarrier<PIPE_V>();
        }
    }

    exp_v<ArchType::ASCEND_V220, float>(curTensorForAttnScore, curTensorForAttnScore,
                                        Param.sectionOneBlockNum * 2 * rowPerTime, 1, 1, 8, 8);
    PipeBarrier<PIPE_V>();

    if (Param.sectionOnePaddingBlockNum > 0) {
        PaddingForRowMaxTensor(Param.sectionOneBlockNum, Param.sectionOnePaddingBlockNum, curTensorForAttnScore, 0);
    }
    // 复用rowmax的计算空间
    ProcessCaclSumTensor(Param.sectionOneBlockNum, Param.sectionOnePaddingBlockNum, true, curTensorForAttnScore,
                         curTensorForRowmax, curTensorForFinalRowsum);
    if constexpr (IF_BF16) {
        convr_v<ArchType::ASCEND_V220, float, INPUT_T>(curTensorForResultFp16, curTensorForAttnScore,
                                                       Param.sectionOneBlockNum * 2 * rowPerTime, 1, 1, 4, 8);
    } else {
        conv_v<ArchType::ASCEND_V220, float, INPUT_T>(curTensorForResultFp16, curTensorForAttnScore,
                                                      Param.sectionOneBlockNum * 2 * rowPerTime, 1, 1, 4, 8);
    }
    PipeBarrier<PIPE_V>();

    if (qkTriangle == true) {
        if (rowPerTime == 1) {
            sub_v<ArchType::ASCEND_V220, float>(curTensorForSecondAttnScore, curTensorForSecondAttnScore,
                                                curTensorForVbrcbRowmaxFp32[8],
                                                Param.sectionTwoBlockNum * 2, // repeat
                                                1,                            // dstBlockStride
                                                1,                            // src0BlockStride
                                                0,                            // src1BlockStride
                                                8,                            // dstRepeatStride
                                                8,                            // src0RepeatStride
                                                0                             // src1RepeatStride
            );
            PipeBarrier<PIPE_V>();
        } else {
            for (int32_t i = 0; i < 2 * rowPerTime; i++) {
                sub_v<ArchType::ASCEND_V220, float>(curTensorForSecondAttnScore[i * 64],
                                                    curTensorForSecondAttnScore[i * 64],
                                                    curTensorForVbrcbRowmaxFp32[(i / 2) * 8 + rowPerTime * 8],
                                                    Param.sectionTwoBlockNum, // repeat
                                                    1,                        // dstBlockStride
                                                    1,                        // src0BlockStride
                                                    0,                        // src1BlockStride
                                                    32,                       // dstRepeatStride
                                                    32,                       // src0RepeatStride
                                                    0                         // src1RepeatStride
                );
                PipeBarrier<PIPE_V>();
            }
        }

        exp_v<ArchType::ASCEND_V220, float>(curTensorForSecondAttnScore, curTensorForSecondAttnScore,
                                            Param.sectionTwoBlockNum * 2 * rowPerTime, 1, 1, 8, 8);
        PipeBarrier<PIPE_V>();

        if (Param.sectionTwoPaddingBlockNum > 0) {
            PaddingForRowMaxTensor(Param.sectionTwoBlockNum, Param.sectionTwoPaddingBlockNum,
                                   curTensorForSecondAttnScore, 0);
        }
        ProcessCaclSumTensor(Param.sectionTwoBlockNum, Param.sectionTwoPaddingBlockNum, true,
                             curTensorForSecondAttnScore, curTensorForRowmax, curTensorForSecondFinalRowsum);

        if constexpr (IF_BF16) {
            convr_v<ArchType::ASCEND_V220, float, INPUT_T>(
                curTensorForResultFp16[Param.sectionOneBlockNum * BASE_BLOCK_SIDE_LEN * rowPerTime],
                curTensorForSecondAttnScore, Param.sectionTwoBlockNum * 2 * rowPerTime, 1, 1, 4, 8);
        } else {
            conv_v<ArchType::ASCEND_V220, float, INPUT_T>(
                curTensorForResultFp16[Param.sectionOneBlockNum * BASE_BLOCK_SIDE_LEN * rowPerTime],
                curTensorForSecondAttnScore, Param.sectionTwoBlockNum * 2 * rowPerTime, 1, 1, 4, 8);
        }
        PipeBarrier<PIPE_V>();
    }

    if (qkTriangle == true) {
        AscendC::RepeatReduceSum<float, false>(curTensorForRecordRowsum[Param.recordRowmaxOffset],
                                               curTensorForFinalRowsum, rowPerTime * 2, 0, 0, 1, 1, 8);
    } else {
        AscendC::RepeatReduceSum<float, false>(curTensorForRecordRowsum[Param.recordRowmaxOffset],
                                               curTensorForFinalRowsum, rowPerTime, 0, 0, 1, 1, 8);
    }

    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);

    AscendC::DataCopy(tmpScoreGmTensorOut, curTensorForResultFp16,
                      AscendC::DataCopyParams(
                          Param.sectionOneBlockNum + Param.sectionTwoBlockNum, BASE_BLOCK_SIDE_LEN * rowPerTime / 16, 0,
                          ((BASE_BLOCK_DATA_NUM - rowPerTime * BASE_BLOCK_SIDE_LEN) + BASE_BLOCK_DATA_NUM) / 16));

    SetFlag<HardEvent::MTE3_MTE2>(eventId);
}

/**
 * 长序列
 * 处理完整行的阶段一，计算出max值
 * 对于三角阵，分两段处理（本方法里面只处理一段），尾块是三角阵，需要掩码
 * 对于非三角阵，完整处理，没有三角阵问题
 * sparseFlag：记录max的下标
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::ProcessLinePhaseOneForMax(
    PARAM_LONG_SEQ_EXP Param, int32_t pingPongFlag, __gm__ WORKSPACE_T *attnScoreGm, __gm__ INPUT_T *attnMaskGm,
    UB_FOR_LONG_SEQ_ATTN_SCORE &UbAttn, bool sparseFlag)
{
    auto eventId = (pingPongFlag == 0 ? EVENT_ID0 : EVENT_ID1);
    bool firstFrag = Param.curFrag == 0;
    bool lastFrag = Param.curFrag == Param.totalFragNum - 1;

    // ping-pong， 确定ub上的地址
    LocalTensor<float> curTensorForAttnScore =
        pingPongFlag == 0 ? UbAttn.tensorForLoadAttnScoreFp32 : UbAttn.tensorForLoadAttnScoreFp32[MAX_LENG_PER_UB_PROC];
    LocalTensor<INPUT_T> curTensorForAttnScoreFp16 =
        curTensorForAttnScore[32 * BASE_BLOCK_SIDE_LEN].template ReinterpretCast<INPUT_T>();

    LocalTensor<float> curTensorForRowmax =
        pingPongFlag == 0 ? UbAttn.tensorForCaclRowmaxFp32 : UbAttn.tensorForCaclRowmaxFp32[MAX_LENG_PER_UB_PROC / 2];

    LocalTensor<float> curTensorForMask = pingPongFlag == 0 ?
                                              UbAttn.tensorForLoadOneBlockTriMaskFp32 :
                                              UbAttn.tensorForLoadOneBlockTriMaskFp32[BASE_BLOCK_SIDE_LEN];
    LocalTensor<INPUT_T> curTensorForMaskFp16 = curTensorForMask[64].template ReinterpretCast<INPUT_T>();

    LocalTensor<float> curTensorForHeadMask = pingPongFlag == 0 ?
                                                  UbAttn.tensorForLoadOneBlockTriMaskFp32[256] :
                                                  UbAttn.tensorForLoadOneBlockTriMaskFp32[BASE_BLOCK_SIDE_LEN + 256];
    LocalTensor<INPUT_T> curTensorForHeadMaskFp16 = curTensorForHeadMask[64].template ReinterpretCast<INPUT_T>();

    LocalTensor<float> curPpTensorForAttenScore;

    auto extraBufIndex = Param.totalFragNum - 1 - Param.curFrag;

    LocalTensor<float> ppTensorForLoadExtraFirstAttnScoreFp32 =
        pingPongFlag == 0 ? UbAttn.tensorForLoadExtraFirstAttnScoreFp32 :
                            UbAttn.tensorForLoadExtraFirstAttnScoreFp32[MAX_LENG_PER_UB_PROC];

    // 使用x个extra buf
    if (extraBufIndex >= 0) {
        curPpTensorForAttenScore = curTensorForAttnScore;
    } else {
        curTensorForAttnScore = ppTensorForLoadExtraFirstAttnScoreFp32;
        curPpTensorForAttenScore = ppTensorForLoadExtraFirstAttnScoreFp32;
    }

    WaitFlag<HardEvent::MTE3_MTE2>(eventId);

    // copy: gm -> ub
    GlobalTensor<float> tmpScoreGmTensor;
    tmpScoreGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(attnScoreGm + Param.sectionStartLineOffset));
    DataCopy(curTensorForAttnScore, tmpScoreGmTensor,
             DataCopyParams(Param.sectionBlockNum, BASE_BLOCK_SIDE_LEN / 8, BASIC_GAP / 8, 0));

    bool sparseTailMaskFlag =
        sparseFlag && (Param.applyTriMask == TRI_MATRIX_TAIL || Param.applyTriMask == TRI_MATRIX_HEAD_AND_TAIL);
    bool sparseHeadMaskFlag =
        sparseFlag && (Param.applyTriMask == TRI_MATRIX_HEAD || Param.applyTriMask == TRI_MATRIX_HEAD_AND_TAIL);
    // 尾块 && 三角阵，一定需要mask
    if ((Param.applyTriMask && Param.triMatrixNum > 0) || sparseTailMaskFlag) {
        gm_to_ub<ArchType::ASCEND_V220, INPUT_T>(curTensorForMaskFp16, attnMaskGmTensor[Param.sectionMaskOffset], 0, 1,
                                                 BASE_BLOCK_SIDE_LEN / 16, 0, 0);
    }
    if (sparseHeadMaskFlag) {
        gm_to_ub<ArchType::ASCEND_V220, INPUT_T>(curTensorForHeadMaskFp16,
                                                 attnMaskGmTensor[Param.sectionMaskOffset + 1], 0, 1,
                                                 BASE_BLOCK_SIDE_LEN / 16, 0, 0);
    }

    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);

    // repeat num 循环次数  (basic_block_num * 128 / 128)   ~~ FP32 * 2
    muls_v<ArchType::ASCEND_V220, float>(curTensorForAttnScore, curTensorForAttnScore, (float)scale,
                                         Param.sectionBlockNum * 2, 1, 1, 8, 8);
    PipeBarrier<PIPE_V>();

    if (Param.sectionPaddingBlockNum > 0) {
        PaddingForRowMaxTensor(Param.sectionBlockNum, Param.sectionPaddingBlockNum, curPpTensorForAttenScore,
                               PADDING_FOR_MAX);
    }

    if ((Param.applyTriMask && Param.triMatrixNum > 0) || sparseTailMaskFlag) {
        conv_v<ArchType::ASCEND_V220, INPUT_T, float>(curTensorForMask, curTensorForMaskFp16, 2, 1, 1, 8, 4);
        PipeBarrier<PIPE_V>();

        // 128个fp32 --> 2 repeats
        muls_v<ArchType::ASCEND_V220, float>(curTensorForMask, curTensorForMask, (float)PADDING_FOR_MAX, 2, 1, 1, 8, 8);
        PipeBarrier<PIPE_V>();

        // fp32, 2-repeat
        add_v<ArchType::ASCEND_V220, float>(curTensorForAttnScore[(Param.sectionBlockNum - 1) * BASE_BLOCK_SIDE_LEN],
                                            curTensorForAttnScore[(Param.sectionBlockNum - 1) * BASE_BLOCK_SIDE_LEN],
                                            curTensorForMask, 2, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
    }
    if (sparseHeadMaskFlag) {
        conv_v<ArchType::ASCEND_V220, INPUT_T, float>(curTensorForHeadMask, curTensorForHeadMaskFp16, 2, 1, 1, 8, 4);
        PipeBarrier<PIPE_V>();

        Duplicate(curTensorForMask, (float)1.0, 128); // 2*8*32B / 4B
        PipeBarrier<PIPE_V>();
        sub_v<ArchType::ASCEND_V220, float>(curTensorForHeadMask, curTensorForMask, curTensorForHeadMask, 2, 1, 1, 1, 8,
                                            8, 8);
        PipeBarrier<PIPE_V>();

        // 128个fp32 --> 2 repeats
        muls_v<ArchType::ASCEND_V220, float>(curTensorForHeadMask, curTensorForHeadMask, (float)PADDING_FOR_MAX, 2, 1,
                                             1, 8, 8);
        PipeBarrier<PIPE_V>();

        add_v<ArchType::ASCEND_V220, float>(curTensorForAttnScore, curTensorForAttnScore, curTensorForHeadMask, 2, 1, 1,
                                            1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
    }

    auto allBlockNum = Param.sectionBlockNum + Param.sectionPaddingBlockNum;
    int32_t tailBlockNum = allBlockNum % BLOCK_NUM_FOR_VMAX;
    int32_t doneBlockNum = allBlockNum / BLOCK_NUM_FOR_VMAX * BLOCK_NUM_FOR_VMAX;
    bool fromBufForAttnScore = false;

    LocalTensor<float> tensorForCaclFinalRowmaxFp32 = pingPongFlag == 0 ?
                                                          UbAttn.tensorForCaclFinalRowmaxFp32 :
                                                          UbAttn.tensorForCaclFinalRowmaxFp32[BASE_BLOCK_SIDE_LEN / 2];
    ProcessCaclMaxTensor(Param.sectionBlockNum, Param.sectionPaddingBlockNum, firstFrag, curTensorForAttnScore,
                         curTensorForRowmax, tensorForCaclFinalRowmaxFp32);

    if (lastFrag == false) {
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        // 不需要MTE3
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
    }
}

/**
 * sparseFlag：记录max的下标
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::ProcessLinePhaseOneForExp(
    PARAM_LONG_SEQ_EXP Param, int32_t pingPongFlag, __gm__ WORKSPACE_T *attnScoreGm, __gm__ INPUT_T *attnMaskGm,
    UB_FOR_LONG_SEQ_ATTN_SCORE &UbAttn, int32_t tmp, bool sparseFlag)
{
    auto eventId = pingPongFlag == 0 ? EVENT_ID0 : EVENT_ID1;
    // x-1个 extra空间
    bool needCopy = Param.curFrag < Param.totalFragNum - 1;

    LocalTensor<float> curTensorForAttnScore =
        pingPongFlag == 0 ? UbAttn.tensorForLoadAttnScoreFp32 : UbAttn.tensorForLoadAttnScoreFp32[MAX_LENG_PER_UB_PROC];
    LocalTensor<INPUT_T> curTensorForAttnScoreFp16 =
        curTensorForAttnScore[32 * BASE_BLOCK_SIDE_LEN].template ReinterpretCast<INPUT_T>();

    LocalTensor<float> curTensorForHeadMask = pingPongFlag == 0 ?
                                                  UbAttn.tensorForLoadOneBlockTriMaskFp32[256] :
                                                  UbAttn.tensorForLoadOneBlockTriMaskFp32[BASE_BLOCK_SIDE_LEN + 256];
    LocalTensor<INPUT_T> curTensorForHeadMaskFp16 = curTensorForHeadMask[64].template ReinterpretCast<INPUT_T>();
    LocalTensor<float> curTensorForMask = pingPongFlag == 0 ?
                                              UbAttn.tensorForLoadOneBlockTriMaskFp32 :
                                              UbAttn.tensorForLoadOneBlockTriMaskFp32[BASE_BLOCK_SIDE_LEN];
    LocalTensor<float> curTensorForRowmax =
        pingPongFlag == 0 ? UbAttn.tensorForCaclRowmaxFp32 : UbAttn.tensorForCaclRowmaxFp32[MAX_LENG_PER_UB_PROC / 2];
    LocalTensor<float> curTensorForCaclFinalRowsumFp32 =
        pingPongFlag == 0 ? UbAttn.tensorForCaclFinalRowsumFp32 :
                            UbAttn.tensorForCaclFinalRowsumFp32[BASE_BLOCK_SIDE_LEN / 2];
    auto extraBufIndex = Param.totalFragNum - 1 - Param.curFrag;

    // 使用x个extra buf
    if (extraBufIndex < 0) {
        curTensorForAttnScore = pingPongFlag == 0 ? UbAttn.tensorForLoadExtraFirstAttnScoreFp32 :
                                                    UbAttn.tensorForLoadExtraFirstAttnScoreFp32[MAX_LENG_PER_UB_PROC];
    }

    WaitFlag<HardEvent::MTE3_MTE2>(eventId);

    GlobalTensor<float> tmpAttnScoreGmTensor;
    tmpAttnScoreGmTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>((__gm__ float *)(attnScoreGm + Param.sectionStartLineOffset)));

    if (needCopy == true) {
        // 最后两次由于数据在UB上，不需要拷贝 （暂定仅最后一个在ub上）
        gm_to_ub<ArchType::ASCEND_V220, float>(curTensorForAttnScore, tmpAttnScoreGmTensor,
                                               0,                       // sid 一般0
                                               Param.sectionBlockNum,   // burst_number 搬运几块
                                               BASE_BLOCK_SIDE_LEN / 8, // FP32
                                               BASIC_GAP / 8,           // src stride 即burst stride   128*127/16
                                               0);

        if (Param.curFrag == 0 && sparseFlag) {
            gm_to_ub<ArchType::ASCEND_V220, INPUT_T>(curTensorForHeadMaskFp16,
                                                     attnMaskGmTensor[Param.sectionMaskOffset + 1], 0, 1,
                                                     BASE_BLOCK_SIDE_LEN / 16, 0, 0);
        }
    }

    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);

    // scale 求max先缩放
    if (needCopy == true) {
        if (Param.curFrag == 0 && sparseFlag) {
            conv_v<ArchType::ASCEND_V220, INPUT_T, float>(curTensorForHeadMask, curTensorForHeadMaskFp16, 2, 1, 1, 8,
                                                          4);
            PipeBarrier<PIPE_V>();
            Duplicate(curTensorForMask, (float)1.0, 128);
            PipeBarrier<PIPE_V>();
            sub_v<ArchType::ASCEND_V220, float>(curTensorForHeadMask, curTensorForMask, curTensorForHeadMask, 2, 1, 1,
                                                1, 8, 8, 8);
            PipeBarrier<PIPE_V>();
            // 128个fp32-->2 repeats
            muls_v<ArchType::ASCEND_V220, float>(curTensorForHeadMask, curTensorForHeadMask, float(PADDING_FOR_MAX), 2,
                                                 1, 1, 8, 8);
            PipeBarrier<PIPE_V>();
            add_v<ArchType::ASCEND_V220, float>(curTensorForAttnScore, curTensorForAttnScore, curTensorForHeadMask, 2,
                                                1, 1, 1, 8, 8, 8);
            PipeBarrier<PIPE_V>();
        }
        // repeat num 循环次数  (basic_block_num * 128 / 128)   ~~ FP32 * 2
        muls_v<ArchType::ASCEND_V220, float>(curTensorForAttnScore, curTensorForAttnScore, (float)scale,
                                             Param.sectionBlockNum * 2, 1, 1, 8, 8);
        PipeBarrier<PIPE_V>();
    }

    sub_v<ArchType::ASCEND_V220, float>(curTensorForAttnScore, curTensorForAttnScore,
                                        UbAttn.tensorForVbrcbRowmaxFp32[pingPongFlag == 0 ? 0 : 8],
                                        Param.sectionBlockNum * 2, 1, 1, 0, 8, 8, 0);
    PipeBarrier<PIPE_V>();

    // 原地覆盖
    exp_v<ArchType::ASCEND_V220, float>(curTensorForAttnScore, curTensorForAttnScore, Param.sectionBlockNum * 2, 1, 1,
                                        8, 8);
    PipeBarrier<PIPE_V>();

    if (Param.sectionPaddingBlockNum > 0) {
        auto tailNum = Param.sectionBlockNum % MAX_BLOCK_PER_ONE_PROC;
        AscendC::Duplicate(curTensorForAttnScore[tailNum * BASE_BLOCK_SIDE_LEN], (float)0, 64,
                           Param.sectionPaddingBlockNum * 2, 1, 8);
        PipeBarrier<PIPE_V>();
    }
    ProcessCaclSumTensor(Param.sectionBlockNum, Param.sectionPaddingBlockNum, Param.curFrag == Param.totalFragNum - 1,
                         curTensorForAttnScore, curTensorForRowmax, curTensorForCaclFinalRowsumFp32);

    LocalTensor<INPUT_T> curTensorForAttnScoreRes = curTensorForAttnScore.template ReinterpretCast<INPUT_T>();
    if constexpr (IF_BF16) {
        convr_v<ArchType::ASCEND_V220, float, INPUT_T>(curTensorForAttnScoreRes, curTensorForAttnScore,
                                                       Param.sectionBlockNum * 2, 1, 1, 4, 8);
    } else {
        conv_v<ArchType::ASCEND_V220, float, INPUT_T>(curTensorForAttnScoreRes, curTensorForAttnScore,
                                                      Param.sectionBlockNum * 2, 1, 1, 4, 8);
    }
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);

    int32_t totalOffsetFp32 = Param.sectionStartLineOffset - tmp + tmp / 2;
    GlobalTensor<INPUT_T> tmpScoreGmTensorOut;
    tmpScoreGmTensorOut.SetGlobalBuffer(
        reinterpret_cast<__gm__ INPUT_T *>((__gm__ INPUT_T *)(attnScoreGm + totalOffsetFp32)));
    DataCopy(
        tmpScoreGmTensorOut, curTensorForAttnScoreRes,
        DataCopyParams(Param.sectionBlockNum, BASE_BLOCK_SIDE_LEN / 16, 0, (BASIC_GAP + BASE_BLOCK_DATA_NUM) / 16));

    SetFlag<HardEvent::MTE3_MTE2>(eventId);
}

/**
 * rowsum的全信息
 * *eachCoreProcessLines：每个core处理的行数
 * *eachCoreOffsetLines：每个core偏移的行数
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::GetNormLocalInfo(
    int32_t qSeqLen, int32_t headNum, int32_t batchNum, int32_t *eachCoreProcessLines, int32_t *eachCoreOffsetLines)
{
    int32_t vectorNum = AscendC::GetBlockNum() * VEC_NUM_PER_CUBE;
    int32_t curVectorIdx = AscendC::GetBlockIdx();
    int32_t totalLines = qSeqLen * headNum * batchNum;

    *eachCoreProcessLines = totalLines / vectorNum;
    int32_t remainLines = totalLines % vectorNum;

    *eachCoreOffsetLines = *eachCoreProcessLines * curVectorIdx;

    if (remainLines > 0 && curVectorIdx < remainLines) {
        // 对尾数进行分配
        *eachCoreProcessLines += 1;
    }

    *eachCoreOffsetLines += curVectorIdx < remainLines ? curVectorIdx : remainLines;
}

/**
 * pvResultOffset：PV值的偏移
 * rowsumOffset：rowsum的偏移
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::ProcessForNormalize(
    int32_t linesPerLoop, int32_t pvResultOffset, int32_t rowsumOffset, int32_t pingPongFlag, UB_FOR_NORMALIZE &UbNorm,
    __gm__ float *__restrict__ gmCCube2, __gm__ INPUT_T *__restrict__ gm_cube2_out, __gm__ float *__restrict__ rowsumGm)
{
    auto eventId = (pingPongFlag == 0 ? EVENT_ID0 : EVENT_ID1);

    // 拷贝计算好的exp值 (interval/2 因为是fp32， ub空间申请时，按照fp16申请的)
    LocalTensor<float> fp32UbTensor =
        pingPongFlag ? UbNorm.tensorForLoadOFp32[UbNorm.oPingPongInterval] : UbNorm.tensorForLoadOFp32[0];

    // 拷贝由cube计算好的rowsum值 （一次处理）
    LocalTensor<float> rowsumUbTensor = pingPongFlag ? UbNorm.tensorForLoadRowsumFp32[UbNorm.rowsumPingPongInterval] :
                                                       UbNorm.tensorForLoadRowsumFp32[0];

    LocalTensor<float> rowsumBrcbUbTensor = pingPongFlag ?
                                                UbNorm.tensorForBrcbRowsumFp32[UbNorm.rowsumBrcbPingPongInterval] :
                                                UbNorm.tensorForBrcbRowsumFp32[0];

    auto rowsumCopyLines = (linesPerLoop + 7) / 8 * 8;
    GlobalTensor<float> tmpGmCCube2Tensor;
    tmpGmCCube2Tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gmCCube2 + pvResultOffset));
    GlobalTensor<float> tmpRowsumGmTensor;
    tmpRowsumGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rowsumGm + rowsumOffset));
    GlobalTensor<INPUT_T> tmpCCube2GmTensorOut;
    tmpCCube2GmTensorOut.SetGlobalBuffer(
        reinterpret_cast<__gm__ INPUT_T *>((__gm__ INPUT_T *)(gm_cube2_out + pvResultOffset)));
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);

    AscendC::DataCopy(fp32UbTensor, tmpGmCCube2Tensor,
                      AscendC::DataCopyParams(1, linesPerLoop * HEAD_DIM * 4 / 32, 0, 0));

    AscendC::DataCopy(rowsumUbTensor, tmpRowsumGmTensor, AscendC::DataCopyParams(1, rowsumCopyLines * 4 / 32, 0, 0));

    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);

    // 展开linesPerLoop个数据
    brcb_v<ArchType::ASCEND_V220, uint32_t>(rowsumBrcbUbTensor.template ReinterpretCast<uint32_t>(),
                                            rowsumUbTensor.template ReinterpretCast<uint32_t>(),
                                            1,           // dstBlockStride
                                            8,           // dstRepeatStride
                                            linesPerLoop // repeat
    );
    PipeBarrier<PIPE_V>();

    // 128个FP32 需要两个repeat； 因此，需要先算前半段，再算后半段
    AscendC::Div<float, false>(fp32UbTensor, fp32UbTensor, rowsumBrcbUbTensor, (uint64_t)0, linesPerLoop,
                               AscendC::BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();

    AscendC::Div<float, false>(fp32UbTensor[HEAD_DIM / 2], fp32UbTensor[HEAD_DIM / 2], rowsumBrcbUbTensor, (uint64_t)0,
                               linesPerLoop, AscendC::BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();

    LocalTensor<INPUT_T> fp16UbTensor = fp32UbTensor.template ReinterpretCast<INPUT_T>();
    if constexpr (IF_BF16) {
        convr_v<ArchType::ASCEND_V220, float, INPUT_T>(fp16UbTensor, fp32UbTensor, linesPerLoop * 2, 1, 1, 4, 8);
    } else {
        conv_v<ArchType::ASCEND_V220, float, INPUT_T>(fp16UbTensor, fp32UbTensor, linesPerLoop * 2, 1, 1, 4, 8);
    }
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);

    AscendC::DataCopy(tmpCCube2GmTensorOut, fp16UbTensor,
                      AscendC::DataCopyParams(1, linesPerLoop * HEAD_DIM * 2 / 32, 0, 0));

    SetFlag<HardEvent::MTE3_MTE2>(eventId);
}

/**
 * 定义函数，处理归一化（exp / rowsum）
 * 实际放在PV之后（P=softmax(QK_T)）, O=PV/rowsum
 * O.shape = (seq_len, headDim)，稠密阵（没有三角阵的考虑）
 *
 * maxProcLen: 一次处理的子序列最大长度 （FP32的元素个
 * curCoreProcessLines：总体需要处理的行数（ping + pong）
 * curCoreOffsetLines
 * UbNorm
 * gmCCube2: 归一化的输入和输出
 * rowsumGm
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::AttentionScoreNormalize(
    int32_t maxProcLen, int32_t curCoreProcessLines, int32_t curCoreOffsetLines, UB_FOR_NORMALIZE &UbNorm,
    __gm__ float *__restrict__ gmCCube2, __gm__ INPUT_T *__restrict__ gm_cube2_out, __gm__ float *__restrict__ rowsumGm)
{
    // 一个loop处理的行数 （单个ping或pong）；一定不会有余数； loop内最大的处理次数，最后一次ping-pong会小于等于该值
    int32_t linesPerLoop = maxProcLen / HEAD_DIM / 2;
    // ping-pong分两半，会有奇数行（有余数）
    int32_t halfLines = curCoreProcessLines / 2;

    // 最后一次ping、pong处理的行数；余数可能是0，表示刚好是整数倍
    int32_t lastPingLines = halfLines % linesPerLoop;
    int32_t lastPongLines = halfLines % linesPerLoop;

    int32_t pingPongTimes = halfLines / linesPerLoop * 2;
    int32_t pingPongTailLines = 0;

    if (lastPingLines > 0) {
        pingPongTimes += 2;
    }

    if (curCoreProcessLines % 2 > 0) {
        // 奇数时，ping-pong不等长 （在pong上加一个会更好）；=0表示刚好用完，所以不能增加额外的一行
        if (lastPongLines < linesPerLoop && lastPongLines != 0) {
            // 没到最长，pong时多处理一行， 此时不需要在多一次循环
            lastPongLines += 1;
        } else {
            // 否则当做尾行处理
            pingPongTailLines = 1;
            pingPongTimes += 1;
        }
    }

    int32_t lastPingPong = pingPongTimes - 2 - (pingPongTailLines > 0 ? 1 : 0);

    lastPingLines = (lastPingLines == 0 ? linesPerLoop : lastPingLines);
    lastPongLines = (lastPongLines == 0 ? linesPerLoop : lastPongLines);

    int32_t pingPongFlag = 0;

    // 需要计算的偏移地址
    int32_t pvResultOffset = curCoreOffsetLines * HEAD_DIM;
    int32_t rowsumOffset = curCoreOffsetLines;

    int32_t pingPvResultOffset = pvResultOffset;
    int32_t pongPvResultOffset = pvResultOffset + halfLines * HEAD_DIM;

    int32_t pingRowsumOffsetOffset = rowsumOffset;
    int32_t pongRowsumOffsetOffset = rowsumOffset + halfLines;

    int *curPvResultOffset = nullptr;
    int *curRowsumOffset = nullptr;

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int procTimes = 0; procTimes < pingPongTimes; procTimes++) {
        curPvResultOffset = (pingPongFlag == 0 ? &pingPvResultOffset : &pongPvResultOffset);
        curRowsumOffset = (pingPongFlag == 0 ? &pingRowsumOffsetOffset : &pongRowsumOffsetOffset);

        if (pingPongTailLines > 0 && procTimes == pingPongTimes - 1) {
            // 只有1行，pongPvResultOffset：需要计算偏移地址
            ProcessForNormalize(1, pongPvResultOffset, pongRowsumOffsetOffset, 1, UbNorm, gmCCube2, gm_cube2_out,
                                rowsumGm);
            continue;
        }

        auto curLines = linesPerLoop;
        if (procTimes >= lastPingPong) {
            if (pingPongFlag == 0) {
                curLines = lastPingLines;
            } else {
                curLines = lastPongLines;
            }
        }

        ProcessForNormalize(curLines, *curPvResultOffset, *curRowsumOffset, pingPongFlag, UbNorm, gmCCube2,
                            gm_cube2_out, rowsumGm);

        *curPvResultOffset += curLines * HEAD_DIM;
        *curRowsumOffset += curLines;

        pingPongFlag = 1 - pingPongFlag;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

/**
 * 短序列一次拷贝
 * qkTriangle: 是否倒三角
 * triMatrixMaskOffset: 128*128的三角阵中取第n行
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::AttentionScoreShortDoubleLineOne(
    bool qkTriangle, int32_t sectionOneBlockNums, int32_t sectionTwoBlockNums, int32_t triMatrixMaskOffset,
    int32_t eachVectorProcLineNum, int32_t localSectionOneStartLineOffset, int32_t localSectionTwoStartLineOffset,
    __gm__ WORKSPACE_T *__restrict__ attnScoreGm, __gm__ INPUT_T *__restrict__ attnMaskGm,
    UB_FOR_SHORT_LEN_ATTN_SCORE &UbAttn)
{
    // 一定是分两段计算rowmax，否则没有必要； 只能支持8192以下的序列，否则没法一次拷贝一个完整行
    PARAM_SHORT_SEQ_MAX ParamPingPong[2] = {0};

    ParamPingPong[0].sectionStartLineOffset = localSectionOneStartLineOffset;
    ParamPingPong[0].sectionOneBlockNum = sectionOneBlockNums;
    // 非三角阵为0
    ParamPingPong[0].sectionTwoBlockNum = sectionTwoBlockNums;
    ParamPingPong[0].recordRowmaxOffset = 0;

    // pong和ping相差一行或两行
    ParamPingPong[1].sectionStartLineOffset = localSectionOneStartLineOffset + BASE_BLOCK_SIDE_LEN * rowPerTime;
    ParamPingPong[1].sectionOneBlockNum = sectionOneBlockNums;
    // 非三角阵为0
    ParamPingPong[1].sectionTwoBlockNum = sectionTwoBlockNums;
    ParamPingPong[1].recordRowmaxOffset = 0;

    GetPaddingInfoForRowMax(sectionOneBlockNums, &ParamPingPong[0].sectionOnePaddingBlockNum);
    ParamPingPong[1].sectionOnePaddingBlockNum = ParamPingPong[0].sectionOnePaddingBlockNum;
    if (qkTriangle == true) {
        ParamPingPong[0].sectionMaskOffset = triMatrixMaskOffset;
        ParamPingPong[1].sectionMaskOffset = triMatrixMaskOffset + maskSeqLength * rowPerTime;
        GetPaddingInfoForRowMax(sectionTwoBlockNums, &ParamPingPong[0].sectionTwoPaddingBlockNum);
        ParamPingPong[1].sectionTwoPaddingBlockNum = ParamPingPong[0].sectionTwoPaddingBlockNum;
    }

    int32_t pingPongFlag = 0;

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

    for (int lines = 0; lines < eachVectorProcLineNum / PING_PONG_NUM / rowPerTime; lines++) {
        // 一定循环是2
        for (int oneLineProc = 0; oneLineProc < 2; oneLineProc++) {
            int32_t offset = lines * BASE_BLOCK_SIDE_LEN * PING_PONG_NUM * rowPerTime +
                             pingPongFlag * BASE_BLOCK_SIDE_LEN * rowPerTime;
            ProcessLinePhaseOneForShortSeqMax(qkTriangle, ParamPingPong[pingPongFlag], pingPongFlag, lines == 0,
                                              attnScoreGm, attnMaskGm, UbAttn);

            ProcessLinePhaseOneForShortSeqExp(qkTriangle, ParamPingPong[pingPongFlag], pingPongFlag, attnScoreGm,
                                              attnMaskGm, UbAttn, offset);
            pingPongFlag = 1 - pingPongFlag;
        }

        ParamPingPong[0].sectionStartLineOffset += BASE_BLOCK_SIZE_DOUBLE * rowPerTime;
        ParamPingPong[1].sectionStartLineOffset += BASE_BLOCK_SIZE_DOUBLE * rowPerTime;

        ParamPingPong[0].sectionMaskOffset += maskSeqLength * 2 * rowPerTime;
        ParamPingPong[1].sectionMaskOffset += maskSeqLength * 2 * rowPerTime;

        ParamPingPong[0].recordRowmaxOffset += 16;
        ParamPingPong[1].recordRowmaxOffset += 16;
    }

    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

/**
 * blockNumPerFullLine
 * subSeqLengthPerProc: 一次处理处理的长度
 * *pingBlockOffsetNum: ping起始块
 * *pongBlockOffsetNum: pong起始块
 * *tailBlockOffsetNum: tail起始块
 * *tailBlockNum:       tail的块数
 * *pingPongTimes:      pingpang的循环次数
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::GetUniRowmaxSeqInfoPerProc(
    int32_t blockNumPerFullLine, int32_t subSeqLengthPerProc, int32_t *pingBlockOffsetNum, int32_t *pongBlockOffsetNum,
    int32_t *tailBlockOffsetNum, int32_t *tailBlockNum, int32_t *pingPongTimes)
{
    // 完整行块的剩余尾块：pingpong需要偶数
    *tailBlockNum = blockNumPerFullLine % 2;
    *tailBlockOffsetNum = blockNumPerFullLine - *tailBlockNum;
    *pingBlockOffsetNum = 0;
    *pongBlockOffsetNum = (*tailBlockOffsetNum) / 2;

    auto totalSize = *pongBlockOffsetNum * BASE_BLOCK_SIDE_LEN;

    // ping和pong各算一次循环
    *pingPongTimes = totalSize / subSeqLengthPerProc * 2;

    // 分2个Section处理时，不是2的幂，未必能整除
    if (totalSize % subSeqLengthPerProc > 0) {
        // 并非是尾块  (尾块只有一个block )
        *pingPongTimes += 2;
    }
}

/**
 * 中序列
 * 处理完整行的阶段一，计算出max值
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::ProcessLinePhaseOneForMax(
    PARAM_MEDIUM_SEQ_EXP Param, int32_t pingPongFlag, bool firstProc, __gm__ WORKSPACE_T *__restrict__ attnScoreGm,
    __gm__ INPUT_T *__restrict__ attnMaskGm, UB_FOR_MDDIUM_SEQ_ATTN_SCORE &UbAttn, int32_t lines)
{
    auto eventId = (pingPongFlag == 0 ? EVENT_ID0 : EVENT_ID1);
    bool firstFrag = Param.curFrag == 0;
    bool lastFrag = Param.curFrag == Param.totalFragNum - 1;

    // ping-pong， 确定ub上的地址
    constexpr uint32_t maxLength = MAX_LENG_PER_UB_PROC > 4096 ? 4096 : MAX_LENG_PER_UB_PROC;
    LocalTensor<float> curTensorForAttnScore =
        pingPongFlag == 0 ? UbAttn.tensorForLoadAttnScoreFp32 : UbAttn.tensorForLoadAttnScoreFp32[maxLength];
    LocalTensor<INPUT_T> fp16CurTensorForAttnScore =
        curTensorForAttnScore[16 * BASE_BLOCK_SIDE_LEN].template ReinterpretCast<INPUT_T>();

    LocalTensor<float> curTensorForRowmax =
        pingPongFlag == 0 ? UbAttn.tensorForCaclRowmaxFp32 : UbAttn.tensorForCaclRowmaxFp32[maxLength / 2];
    LocalTensor<float> curTensorForMask = pingPongFlag == 0 ?
                                              UbAttn.tensorForLoadOneBlockTriMaskFp32 :
                                              UbAttn.tensorForLoadOneBlockTriMaskFp32[BASE_BLOCK_SIDE_LEN];
    LocalTensor<INPUT_T> curTensorForMaskFp16 = curTensorForMask[64].template ReinterpretCast<INPUT_T>();
    LocalTensor<float> curPpTensorForAttenScore = curTensorForAttnScore;
    LocalTensor<float> curTensorForFinalRowmax = UbAttn.tensorForCaclFinalRowmaxFp32;

    auto curBlockNum = 1;
    if (Param.tailBlock == false) {
        curBlockNum = Param.blockNumPerStep;
        if (lastFrag == true) {
            curBlockNum = Param.blockNumForLast;
        }
    }

    WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);

    // copy: gm -> ub
    GlobalTensor<float> tmpScoreGmTensor;
    tmpScoreGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(attnScoreGm + Param.sectionStartLineOffset));
    DataCopy(curTensorForAttnScore, tmpScoreGmTensor,
             DataCopyParams(curBlockNum, BASE_BLOCK_SIDE_LEN / 8, BASIC_GAP / 8, 0));

    // 尾块 && 三角阵，一定需要mask
    if (Param.applyTriMask && Param.triMatrixNum > 0) {
        gm_to_ub<ArchType::ASCEND_V220, INPUT_T>(curTensorForMaskFp16, attnMaskGmTensor[Param.sectionMaskOffset], 0, 1,
                                                 BASE_BLOCK_SIDE_LEN / 16, 0, 0);
    }

    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);

    auto paddingBlock = 0;

    if (lastFrag && Param.lastPaddingBlockNum > 0) {
        paddingBlock = Param.lastPaddingBlockNum;
        PaddingForRowMaxTensor(Param.blockNumForLast, Param.lastPaddingBlockNum, curPpTensorForAttenScore,
                               PADDING_FOR_MAX);
    }

    // 乘scale
    muls_v<ArchType::ASCEND_V220, float>(UbAttn.tensorForStoreOneLineAttnScoreFp32[Param.bufOffset],
                                         curTensorForAttnScore, (float)scale, (curBlockNum + paddingBlock) * 2, 1, 1, 8,
                                         8);
    PipeBarrier<PIPE_V>();

    if (Param.applyTriMask && Param.triMatrixNum > 0) {
        conv_v<ArchType::ASCEND_V220, INPUT_T, float>(curTensorForMask, curTensorForMaskFp16, 2, 1, 1, 8, 4);
        PipeBarrier<PIPE_V>();

        // 128个fp32 --> 2 repeats
        muls_v<ArchType::ASCEND_V220, float>(curTensorForMask, curTensorForMask, (float)PADDING_FOR_MAX, 2, 1, 1, 8, 8);
        PipeBarrier<PIPE_V>();

        // fp32, 2-repeat
        add_v<ArchType::ASCEND_V220, float>(
            UbAttn.tensorForStoreOneLineAttnScoreFp32[Param.bufOffset + (curBlockNum - 1) * BASE_BLOCK_SIDE_LEN],
            UbAttn.tensorForStoreOneLineAttnScoreFp32[Param.bufOffset + (curBlockNum - 1) * BASE_BLOCK_SIDE_LEN],
            curTensorForMask, 2, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
    }

    // ping-pong一行，所以第一次ping才是第一段
    ProcessCaclMaxTensor(curBlockNum, paddingBlock, firstFrag && firstProc,
                         UbAttn.tensorForStoreOneLineAttnScoreFp32[Param.bufOffset], curTensorForRowmax,
                         curTensorForFinalRowmax);

    // 非最后一次，需要MTE2
    if (lastFrag == false && lines == 0) {
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        SetFlag<HardEvent::MTE3_MTE2>(eventId);
    }
    SetFlag<HardEvent::V_MTE2>(eventId);
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::ProcessLinePhaseOneForExp(
    PARAM_MEDIUM_SEQ_EXP Param, int32_t pingPongFlag, bool firstProc, __gm__ WORKSPACE_T *__restrict__ attnScoreGm,
    __gm__ INPUT_T *__restrict__ attnMaskGm, UB_FOR_MDDIUM_SEQ_ATTN_SCORE &UbAttn, int32_t fp32Offset)
{
    constexpr uint32_t maxLength = MAX_LENG_PER_UB_PROC > 4096 ? 4096 : MAX_LENG_PER_UB_PROC;
    LocalTensor<float> curTensorForRowmax =
        pingPongFlag == 0 ? UbAttn.tensorForCaclRowmaxFp32 : UbAttn.tensorForCaclRowmaxFp32[maxLength / 2];
    auto eventId = pingPongFlag == 0 ? EVENT_ID0 : EVENT_ID1;

    bool lastFrag = Param.curFrag == Param.totalFragNum - 1;
    bool firstFrag = Param.curFrag == 0;

    auto curBlockNum = 1;
    auto paddingBlock = 0;
    if (Param.tailBlock == false) {
        curBlockNum = Param.blockNumPerStep;
        if (lastFrag == true) {
            curBlockNum = Param.blockNumForLast;
        }
    }

    // 当前处理的数据
    LocalTensor<float> tensorForAttnScore = UbAttn.tensorForStoreOneLineAttnScoreFp32[Param.bufOffset];

    sub_v<ArchType::ASCEND_V220, float>(tensorForAttnScore, tensorForAttnScore, UbAttn.tensorForVbrcbRowmaxFp32,
                                        curBlockNum * 2, 1, 1, 0, 8, 8, 0);
    PipeBarrier<PIPE_V>();

    // 原地覆盖
    exp_v<ArchType::ASCEND_V220, float>(tensorForAttnScore, tensorForAttnScore, curBlockNum * 2, 1, 1, 8, 8);
    PipeBarrier<PIPE_V>();

    if (lastFrag && Param.lastPaddingBlockNum > 0) {
        paddingBlock = Param.lastPaddingBlockNum;
        auto tailNum = Param.blockNumForLast % MAX_BLOCK_PER_ONE_PROC;
        AscendC::Duplicate(tensorForAttnScore[tailNum * BASE_BLOCK_SIDE_LEN], (float)0, 64,
                           Param.lastPaddingBlockNum * 2, 1, 8);
        PipeBarrier<PIPE_V>();
    }
    ProcessCaclSumTensor(curBlockNum, paddingBlock, firstFrag && firstProc, tensorForAttnScore, curTensorForRowmax,
                         UbAttn.tensorForCaclFinalRowsumFp32);

    LocalTensor<INPUT_T> fp16TensorForAttnScoreRes = tensorForAttnScore.template ReinterpretCast<INPUT_T>();
    if constexpr (IF_BF16) {
        convr_v<ArchType::ASCEND_V220, float, INPUT_T>(fp16TensorForAttnScoreRes, tensorForAttnScore, curBlockNum * 2,
                                                       1, 1, 4, 8);
    } else {
        conv_v<ArchType::ASCEND_V220, float, INPUT_T>(fp16TensorForAttnScoreRes, tensorForAttnScore, curBlockNum * 2, 1,
                                                      1, 4, 8);
    }
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);

    int32_t totalOffsetFp32 = Param.sectionStartLineOffset - fp32Offset + fp32Offset / 2;
    GlobalTensor<INPUT_T> tmpScoreGmTensorOut;
    tmpScoreGmTensorOut.SetGlobalBuffer(
        reinterpret_cast<__gm__ INPUT_T *>((__gm__ INPUT_T *)(attnScoreGm + totalOffsetFp32)));
    DataCopy(tmpScoreGmTensorOut, fp16TensorForAttnScoreRes,
             DataCopyParams(curBlockNum, BASE_BLOCK_SIDE_LEN / 16, 0, (BASIC_GAP + BASE_BLOCK_DATA_NUM) / 16));

    SetFlag<HardEvent::MTE3_MTE2>(eventId);
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::AttentionScoreSingleLine(
    bool qkTriangle, int32_t sectionLoopTimes, int32_t sectionOneBlockNums, int32_t sectionTwoBlockNums,
    int32_t triMatrixMaskOffset, int32_t eachVectorProcLineNum, int32_t localSectionOneStartLineOffset,
    int32_t localSectionTwoStartLineOffset, int32_t triMatrixNum, __gm__ WORKSPACE_T *__restrict__ attnScoreGm,
    __gm__ INPUT_T *__restrict__ attnMaskGm, UB_FOR_MDDIUM_SEQ_ATTN_SCORE &UbAttn)
{
    int32_t *curSectionBlockNums = &sectionOneBlockNums;

    // ping-pong每次处理得长度
    int32_t subSeqLengthPerProc;

    int32_t pingBlockOffsetNum = 0;
    int32_t pongBlockOffsetNum = 0;
    int32_t tailBlockOffsetNum = 0;
    int32_t tailBlockNum = 0;
    int32_t tailOffset = 0;
    int32_t pingPongTimes = 0;
    int32_t pingPongOffsetInLoop[2] = {0};

    bool lastTimePingOrPong = false;

    int32_t paddingBlockNumForSepSection = 0;
    int32_t curTriMatrixMaskOffset = 0;

    int32_t *curLocalSectionStartLineOffset = &localSectionOneStartLineOffset;

    int32_t pingPongFlag = 0;

    // ping-pong参数
    PARAM_MEDIUM_SEQ_EXP ParamPingPong[2] = {0};
    PARAM_MEDIUM_SEQ_EXP ParamTail = {0};

    // 尾块是固定的
    ParamTail.curFrag = 0;
    ParamTail.totalFragNum = 1;
    ParamTail.blockNumPerStep = 1;
    ParamTail.blockNumForLast = 1;
    ParamTail.tailBlock = true;
    // 补成2
    ParamTail.lastPaddingBlockNum = 1;

    auto maxBlock = MAX_BLOCK_PER_ONE_PROC / 2;
    auto maxLength = MAX_LENG_PER_UB_PROC / 2;

    ParamPingPong[0].blockNumPerStep = maxBlock;
    ParamPingPong[1].blockNumPerStep = maxBlock;

    ParamPingPong[0].tailBlock = false;
    ParamPingPong[1].tailBlock = false;

    // ping不会遭遇三角块
    ParamPingPong[0].applyTriMask = false;
    // 若有尾块（1块），一定是尾块
    ParamTail.applyTriMask = true;

    int32_t recordRowmaxOffset = 0;

    for (int sectionLoop = 0; sectionLoop < sectionLoopTimes; sectionLoop++) {
        auto sectionBlockNum = *curSectionBlockNums;
        auto sectionSeqLen = sectionBlockNum * BASE_BLOCK_SIDE_LEN;

        // 每个Section长度非2的幂，折半求max时，需要padding值
        if (sectionBlockNum > 1 && sectionBlockNum % 2 > 0) {
            // 保证偶数，为了能ping-pong （非2的幂）
            sectionSeqLen -= BASE_BLOCK_SIDE_LEN;
        }

        // 确定ping-pong的处理长度
        subSeqLengthPerProc = sectionSeqLen / 2;
        subSeqLengthPerProc = subSeqLengthPerProc > maxLength ? maxLength : subSeqLengthPerProc;

        GetUniRowmaxSeqInfoPerProc(sectionBlockNum, subSeqLengthPerProc, &pingBlockOffsetNum, &pongBlockOffsetNum,
                                   &tailBlockOffsetNum, &tailBlockNum, &pingPongTimes);

        // ping的padding数量
        GetPaddingInfoForRowMax(pongBlockOffsetNum, &ParamPingPong[0].lastPaddingBlockNum);

        ParamPingPong[1].lastPaddingBlockNum = ParamPingPong[0].lastPaddingBlockNum;

        ParamPingPong[0].blockNumForLast = pongBlockOffsetNum % maxBlock;
        if (ParamPingPong[0].blockNumForLast == 0) {
            // 最后一次时，会使用该参数
            ParamPingPong[0].blockNumForLast = maxBlock;
        }

        ParamPingPong[1].blockNumForLast = ParamPingPong[0].blockNumForLast;

        if (tailBlockNum > 0) {
            if (ParamPingPong[1].blockNumForLast < maxBlock) {
                // 等于0或者等于max_block， 说明没有额外空间，尽量减少ping-pong的次数，补充到pong
                ParamPingPong[1].blockNumForLast += 1;
                GetPaddingInfoForRowMax(pongBlockOffsetNum + 1, &ParamPingPong[1].lastPaddingBlockNum);
            } else {
                pingPongTimes += 1;
            }
        }

        curTriMatrixMaskOffset = triMatrixMaskOffset;

        ParamTail.sectionStartLineOffset = (sectionBlockNum - 1) * BASE_BLOCK_DATA_NUM;

        // tail加载ping上
        ParamPingPong[0].totalFragNum = pingPongTimes / 2 + pingPongTimes % 2;
        ParamPingPong[1].totalFragNum = pingPongTimes / 2;

        ParamPingPong[0].triMatrixNum = triMatrixNum;
        ParamPingPong[1].triMatrixNum = triMatrixNum;
        ParamTail.triMatrixNum = triMatrixNum;

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        // 流水变更
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);

        // 一个完整行循环
        for (int lines = 0; lines < eachVectorProcLineNum; lines++) {
            pingPongFlag = 0;

            // ping从行首开始，pong从行中间的某一块开始 （section loop = 1时，ping从第二段开始）
            ParamPingPong[0].sectionStartLineOffset = *curLocalSectionStartLineOffset;
            ParamPingPong[1].sectionStartLineOffset =
                *curLocalSectionStartLineOffset + pongBlockOffsetNum * BASE_BLOCK_DATA_NUM;
            ParamTail.sectionStartLineOffset =
                *curLocalSectionStartLineOffset + tailBlockOffsetNum * BASE_BLOCK_DATA_NUM;

            ParamPingPong[0].bufOffset = 0;
            ParamPingPong[1].bufOffset = MDDIUM_SEQ_THRESHOLD / 2;
            // 始终是最后一块
            ParamTail.bufOffset = MDDIUM_SEQ_THRESHOLD / 2 + (sectionBlockNum / 2) * BASE_BLOCK_SIDE_LEN;

            ParamPingPong[1].applyTriMask = false;

            // 一定循环是2
            for (int oneLineProc = 0; oneLineProc < pingPongTimes; oneLineProc++) {
                ParamPingPong[pingPongFlag].curFrag = oneLineProc / 2;

                // 尾块
                if (pingPongTimes % 2 == 1 && oneLineProc == pingPongTimes - 1) {
                    ParamTail.sectionMaskOffset = curTriMatrixMaskOffset;
                    ProcessLinePhaseOneForMax(ParamTail, pingPongFlag, oneLineProc == 0, attnScoreGm, attnMaskGm,
                                              UbAttn, lines);
                    continue;
                }

                // 最后一次ping-pong
                lastTimePingOrPong = oneLineProc >= pingPongTimes / 2 * 2 - 2;

                // 没有尾块
                if (lastTimePingOrPong && pingPongTimes % 2 == 0 && oneLineProc == pingPongTimes - 1) {
                    ParamPingPong[1].applyTriMask = true;
                }

                ParamPingPong[pingPongFlag].sectionMaskOffset = curTriMatrixMaskOffset;

                ProcessLinePhaseOneForMax(ParamPingPong[pingPongFlag], pingPongFlag, oneLineProc == 0, attnScoreGm,
                                          attnMaskGm, UbAttn, lines);

                ParamPingPong[pingPongFlag].bufOffset +=
                    (ParamPingPong[pingPongFlag].blockNumPerStep * BASE_BLOCK_SIDE_LEN);
                ParamPingPong[pingPongFlag].sectionStartLineOffset +=
                    (ParamPingPong[pingPongFlag].blockNumPerStep * BASE_BLOCK_DATA_NUM);

                pingPongFlag = 1 - pingPongFlag;
            }

            // 接着计算max (只有一个max)，有2个最大值(两段)
            cmax_v<ArchType::ASCEND_V220, float, ReduceOrder::ORDER_ONLY_VALUE>(
                UbAttn.tensorForRecordRowmaxFp32[recordRowmaxOffset], UbAttn.tensorForCaclFinalRowmaxFp32, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();

            brcb_v<ArchType::ASCEND_V220, uint32_t>(
                UbAttn.tensorForVbrcbRowmaxFp32.template ReinterpretCast<uint32_t>(),
                UbAttn.tensorForRecordRowmaxFp32[recordRowmaxOffset].template ReinterpretCast<uint32_t>(), 1, 8, 1);
            PipeBarrier<PIPE_V>();

            pingPongFlag = 0;

            ParamPingPong[0].sectionStartLineOffset = *curLocalSectionStartLineOffset;
            ParamPingPong[1].sectionStartLineOffset =
                *curLocalSectionStartLineOffset + pongBlockOffsetNum * BASE_BLOCK_DATA_NUM;

            ParamPingPong[0].bufOffset = 0;
            ParamPingPong[1].bufOffset = MDDIUM_SEQ_THRESHOLD / 2;

            // 计算exp
            for (int oneLineProc = 0; oneLineProc < pingPongTimes; oneLineProc++) {
                // 找到buf的偏移地址；拷贝数据的地址
                ParamPingPong[pingPongFlag].curFrag = oneLineProc / 2;
                int32_t fp32Offset = lines * BASE_BLOCK_SIDE_LEN;
                if (pingPongTimes % 2 == 1 && oneLineProc == 0) {
                    ProcessLinePhaseOneForExp(ParamTail, pingPongFlag, oneLineProc == 0, attnScoreGm, attnMaskGm,
                                              UbAttn, fp32Offset);

                    // 尾块
                    continue;
                }
                ProcessLinePhaseOneForExp(ParamPingPong[pingPongFlag], pingPongFlag, oneLineProc == 0, attnScoreGm,
                                          attnMaskGm, UbAttn, fp32Offset);

                ParamPingPong[pingPongFlag].sectionStartLineOffset +=
                    (ParamPingPong[pingPongFlag].blockNumPerStep * BASE_BLOCK_DATA_NUM);
                ParamPingPong[pingPongFlag].bufOffset +=
                    (ParamPingPong[pingPongFlag].blockNumPerStep * BASE_BLOCK_SIDE_LEN);

                pingPongFlag = 1 - pingPongFlag;
            }

            AscendC::RepeatReduceSum<float, false>(UbAttn.tensorForRecordRowsumFp32[recordRowmaxOffset],
                                                   UbAttn.tensorForCaclFinalRowsumFp32, 1, 0, 0, 1, 1, 8);
            PipeBarrier<PIPE_V>();

            recordRowmaxOffset += 8;

            *curLocalSectionStartLineOffset += BASE_BLOCK_SIDE_LEN;
            curTriMatrixMaskOffset += maskSeqLength;
        }

        // 流水变更
        int32_t flag = 0;
        for (int32_t i = 0; i < pingPongTimes; i++) {
            auto eventId = (flag == 0 ? EVENT_ID0 : EVENT_ID1);
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            flag = 1 - flag;
        }

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);

        curSectionBlockNums = &sectionTwoBlockNums;
        curLocalSectionStartLineOffset = &localSectionTwoStartLineOffset;
    }
}

/**
 * 单行ping-pong计算
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::AttentionScoreDoubleLine(
    bool qkTriangle, int32_t sectionLoopTimes, int32_t sectionOneBlockNums, int32_t sectionTwoBlockNums,
    int32_t triMatrixMaskOffset, int32_t eachVectorProcLineNum, int32_t localSectionOneStartLineOffset,
    int32_t localSectionTwoStartLineOffset, int32_t triMatrixNum, int32_t triMatrixOffset[],
    __gm__ WORKSPACE_T *__restrict__ attnScoreGm, __gm__ INPUT_T *__restrict__ attnMaskGm,
    UB_FOR_LONG_SEQ_ATTN_SCORE &UbAttn, bool sparseFlag)
{
    // section one 需要处理的块
    int32_t *curSectionBlockNums = &sectionOneBlockNums;
    // section的起始地址
    int32_t *curLocalSectionStartLineOffset = &localSectionOneStartLineOffset;

    int32_t pingPongFlag = 0;
    int32_t pingPongTimes = 0;

    int32_t tailBlockNum = 0;
    int32_t sectionPaddingBlockNum = 0;

    PARAM_LONG_SEQ_EXP ParamPingPong[2] = {0};

    int32_t rowmaxOffset = 0;
    int32_t rowsumOffset = 0;

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int sectionLoop = 0; sectionLoop < sectionLoopTimes; sectionLoop++) {
        LocalTensor curTensorForRecordRowmaxFp32 =
            sectionLoop == 0 ? UbAttn.tensorForRecordRowmaxFp32 : UbAttn.tensorForRecordRowmaxFp32[256];
        rowmaxOffset = 0;
        rowsumOffset = 0;

        pingPongTimes = (*curSectionBlockNums) / MAX_BLOCK_PER_ONE_PROC;
        tailBlockNum = (*curSectionBlockNums) % MAX_BLOCK_PER_ONE_PROC;

        // tail指的是，相对于64个block，剩余的部分
        if (tailBlockNum > 0) {
            pingPongTimes += 1;
            // 非三角阵不需要
            GetPaddingInfoForRowMax(*curSectionBlockNums, &sectionPaddingBlockNum);
        } else {
            sectionPaddingBlockNum = 0;
        }

        if (tailBlockNum == 0) {
            tailBlockNum = MAX_BLOCK_PER_ONE_PROC;
        }

        ParamPingPong[0].totalFragNum = pingPongTimes;
        ParamPingPong[1].totalFragNum = pingPongTimes;

        pingPongTimes *= 2; // ping一行，pong一行

        ParamPingPong[0].sectionMaskOffset = triMatrixMaskOffset;
        ParamPingPong[1].sectionMaskOffset = triMatrixMaskOffset + maskSeqLength;

        ParamPingPong[0].triMatrixNum = triMatrixNum;
        ParamPingPong[1].triMatrixNum = triMatrixNum;

        // ping一行，pong一行；eachVectorProcLineNum一定是偶数
        for (int lines = 0; lines < eachVectorProcLineNum / 2; lines++) {
            // 第一行
            ParamPingPong[0].sectionStartLineOffset = *curLocalSectionStartLineOffset;
            // 第二行
            ParamPingPong[1].sectionStartLineOffset = *curLocalSectionStartLineOffset + BASE_BLOCK_SIDE_LEN;

            ParamPingPong[0].sectionBlockNum = MAX_BLOCK_PER_ONE_PROC;
            ParamPingPong[1].sectionBlockNum = MAX_BLOCK_PER_ONE_PROC;
            ParamPingPong[0].sectionPaddingBlockNum = 0;
            ParamPingPong[1].sectionPaddingBlockNum = 0;

            ParamPingPong[0].applyTriMask = TRI_MATRIX_NONE;
            ParamPingPong[1].applyTriMask = TRI_MATRIX_NONE;

            pingPongFlag = 0;

            for (int oneLineProc = 0; oneLineProc < pingPongTimes; oneLineProc++) {
                ParamPingPong[pingPongFlag].curFrag = oneLineProc / 2;

                // 最后一段
                if (oneLineProc >= pingPongTimes - 2) {
                    ParamPingPong[pingPongFlag].sectionBlockNum = tailBlockNum;
                    ParamPingPong[pingPongFlag].sectionPaddingBlockNum = sectionPaddingBlockNum;
                    if (qkTriangle) {
                        ParamPingPong[pingPongFlag].applyTriMask = TRI_MATRIX_TAIL;
                    }
                    if (sparseFlag) {
                        ParamPingPong[pingPongFlag].sectionPaddingBlockNum = sectionPaddingBlockNum;
                    }
                }

                if (sparseFlag) {
                    if (pingPongTimes == 2) {
                        ParamPingPong[pingPongFlag].applyTriMask = TRI_MATRIX_HEAD_AND_TAIL;
                    } else {
                        if (oneLineProc <= 1) {
                            ParamPingPong[pingPongFlag].applyTriMask = TRI_MATRIX_HEAD;
                        }
                        if (oneLineProc >= pingPongTimes - 2) {
                            ParamPingPong[pingPongFlag].applyTriMask = TRI_MATRIX_TAIL;
                        }
                    }
                }

                ProcessLinePhaseOneForMax(ParamPingPong[pingPongFlag], pingPongFlag, attnScoreGm, attnMaskGm, UbAttn,
                                          sparseFlag);
                // -- 需要加tri-matrix （0-非三角阵; 1-三角阵, 非unirow; 2-三角阵, unirow）

                // 恢复mask
                ParamPingPong[pingPongFlag].applyTriMask = TRI_MATRIX_NONE;

                ParamPingPong[pingPongFlag].sectionStartLineOffset += BASE_BLOCK_DATA_NUM * MAX_BLOCK_PER_ONE_PROC;

                pingPongFlag = 1 - pingPongFlag;
            }

            // 计算两个max值 (ppBufForCaclFinalRowmaxFp32)
            cmax_v<ArchType::ASCEND_V220, float, ReduceOrder::ORDER_ONLY_VALUE>(
                curTensorForRecordRowmaxFp32[rowmaxOffset], UbAttn.tensorForCaclFinalRowmaxFp32, 2, 1, 1, 8);
            PipeBarrier<PIPE_V>();

            brcb_v<ArchType::ASCEND_V220, uint32_t>(
                UbAttn.tensorForVbrcbRowmaxFp32.template ReinterpretCast<uint32_t>(),
                curTensorForRecordRowmaxFp32[rowmaxOffset].template ReinterpretCast<uint32_t>(), 1, 8, 1);
            PipeBarrier<PIPE_V>();

            rowmaxOffset += 8;

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);

            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);

            // EXP
            pingPongFlag = 0;

            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

            for (int oneLineProc = pingPongTimes - 1; oneLineProc > -1; oneLineProc--) {
                ParamPingPong[pingPongFlag].curFrag = oneLineProc / 2;

                ParamPingPong[0].sectionBlockNum = MAX_BLOCK_PER_ONE_PROC;
                ParamPingPong[1].sectionBlockNum = MAX_BLOCK_PER_ONE_PROC;
                ParamPingPong[0].sectionPaddingBlockNum = 0;
                ParamPingPong[1].sectionPaddingBlockNum = 0;

                // 最后一段
                if (oneLineProc >= pingPongTimes - 2) {
                    ParamPingPong[pingPongFlag].sectionBlockNum = tailBlockNum;
                    ParamPingPong[pingPongFlag].sectionPaddingBlockNum = sectionPaddingBlockNum;
                }

                ParamPingPong[pingPongFlag].sectionStartLineOffset -= BASE_BLOCK_DATA_NUM * MAX_BLOCK_PER_ONE_PROC;

                int64_t tmp = lines * BASE_BLOCK_SIDE_LEN * 2 + pingPongFlag * BASE_BLOCK_SIDE_LEN;
                ProcessLinePhaseOneForExp(ParamPingPong[pingPongFlag], pingPongFlag, attnScoreGm, attnMaskGm, UbAttn,
                                          tmp, sparseFlag);

                pingPongFlag = 1 - pingPongFlag;
            }

            LocalTensor<float> curTensorForRecordRowsumFp32 =
                sectionLoop == 0 ? UbAttn.tensorForRecordRowsumFp32 : UbAttn.tensorForRecordRowsumFp32[256];
            AscendC::RepeatReduceSum<float, false>(curTensorForRecordRowsumFp32[rowsumOffset],
                                                   UbAttn.tensorForCaclFinalRowsumFp32, 2, 0, 0, 1, 1, 8);
            PipeBarrier<PIPE_V>();

            rowsumOffset += 8;
            *curLocalSectionStartLineOffset += BASE_BLOCK_SIZE_DOUBLE;

            ParamPingPong[0].sectionMaskOffset += maskSeqLength * PING_PONG_NUM;
            ParamPingPong[1].sectionMaskOffset += maskSeqLength * PING_PONG_NUM;
        }

        // 切换section two
        curLocalSectionStartLineOffset = &localSectionTwoStartLineOffset;
        curSectionBlockNums = &sectionTwoBlockNums;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

#endif

#endif // __VECTORFORWARD_H__