/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <numeric>
#include <algorithm>
#include <functional>
#include "mki/utils/assert/assert.h"
#include "mki/utils/checktensor/check_tensor.h"
#include "mki/utils/platform/platform_info.h"
#include "mla_tiling.h"

namespace AtbOps {

const int32_t NUM1 = 1;
const int32_t NUM2 = 2;
const int32_t NUM3 = 3;
const int32_t NUM4 = 4;
const int32_t NUM5 = 5;
const int32_t NUM6 = 6;
const int32_t NUM8 = 8;
const int32_t NUM16 = 16;
const int32_t NUM32 = 32;
const int32_t ROPE_DIM = 64;
const int32_t NUM256 = 256;
const int32_t NUM512 = 512;
const int32_t NUM576 = 576;
const float SPLITKV_SEQLEN = 2048;
const uint32_t BLOCK_WORKSPACE_SIZE = 32768;

int32_t CalcSplitNum(MLAInfo &mmInfo, int32_t blockDim, int32_t minKVSeqlen, int32_t blockSize)
{
    if (blockDim - mmInfo.flashDecodingTaskNum <= NUM4 || mmInfo.quantFlag) {
        return NUM1;
    }
    if (blockSize == 0 || blockDim == 0) {
        return NUM1;
    }
    int32_t minKVBlocks = (minKVSeqlen + blockSize - 1) / blockSize;
    for (int32_t splitNum = 2; splitNum <= NUM6; splitNum++) {
        if ((mmInfo.flashDecodingTaskNum * splitNum) % blockDim != 0) {
            continue;
        }
        int32_t repeatTimesPerBlock = mmInfo.flashDecodingTaskNum * splitNum / blockDim;
        if (minKVSeqlen / splitNum >= blockSize &&
            repeatTimesPerBlock + NUM6 <= minKVBlocks - (minKVBlocks + splitNum - 1) / splitNum * repeatTimesPerBlock) {
            return splitNum;
        } else {
            return NUM1;
        }
    }
    return NUM1;
}

Status BatchSeqSort(MLAInfo &mmInfo, uint32_t endIndex, uint32_t sortDim)
{
    if (sortDim == 0 || endIndex <= 0) {
        return Status::OkStatus();
    }
    std::sort(mmInfo.batchList.begin(), mmInfo.batchList.end());
    std::reverse(mmInfo.batchList.begin(), mmInfo.batchList.begin() + endIndex);
    uint32_t batchSortInfo = (mmInfo.batch + sortDim - 1) / sortDim;
    for (uint32_t sortIdx = 0; sortIdx < batchSortInfo; sortIdx++) {
        uint32_t startIndex = sortIdx * sortDim;
        uint32_t sortEnd = std::min(startIndex + sortDim, endIndex);
        if (sortIdx % NUM2 == 0) {
            std::sort(mmInfo.batchList.begin() + startIndex, mmInfo.batchList.begin() + sortEnd);
        } else {
            std::sort(mmInfo.batchList.begin() + startIndex,
                      mmInfo.batchList.begin() + sortEnd, std::greater<BatchNode>());
        }
    }
    return Status::OkStatus();
}

Status GetFlashDecodingInfo(MLAInfo &mmInfo, OpParam::MLA &param, uint32_t blockDim)
{
    if (blockDim == 0) {
        return Status::FailStatus(ERROR_INVALID_VALUE);
    }
    mmInfo.tailBatch = mmInfo.batch % blockDim;
    mmInfo.tailTaskNum = mmInfo.totalTaskNum % blockDim;
    mmInfo.flashDecodingTaskNum = mmInfo.quantFlag ? mmInfo.tailTaskNum : mmInfo.tailBatch;
    auto minKVSeqlen = std::min_element(param.kvSeqLen.begin(), param.kvSeqLen.end());
    auto minQSeqlen = mmInfo.qSeqLen != nullptr ? *std::min_element(param.qSeqLen.begin(), param.qSeqLen.end()) : 1;
    auto maxQSeqlen = mmInfo.qSeqLen != nullptr ? *std::max_element(param.qSeqLen.begin(), param.qSeqLen.end()) : 1;
    mmInfo.flashDecoding = !mmInfo.quantFlag && mmInfo.flashDecodingTaskNum != 0 && param.isRing == 0 &&
                           *minKVSeqlen >= SPLITKV_SEQLEN &&
                           ((minQSeqlen == NUM2 && maxQSeqlen == NUM2) ||
                           (minQSeqlen == 1 && maxQSeqlen == 1));
    if (!mmInfo.flashDecoding) {
        OP_TILING_CHECK_STATUS_RETURN(BatchSeqSort(mmInfo, mmInfo.batch, blockDim));
        return Status::OkStatus();
    }
    OP_TILING_CHECK_STATUS_RETURN(BatchSeqSort(mmInfo, mmInfo.batch - mmInfo.flashDecodingTaskNum, blockDim));
    mmInfo.splitKVNum = blockDim / mmInfo.flashDecodingTaskNum > 1 ?  blockDim / mmInfo.flashDecodingTaskNum :
                        CalcSplitNum(mmInfo, blockDim, *minKVSeqlen, mmInfo.blockSize);
    mmInfo.flashDecoding = mmInfo.splitKVNum == 1 ? false : true;
    int32_t taskNum = mmInfo.quantFlag ? mmInfo.totalTaskNum : mmInfo.batch;
    mmInfo.normalTaskNum = taskNum / blockDim * blockDim;
    MKI_LOG(INFO) << "flashDecoding is = " << mmInfo.flashDecoding;
    return Status::OkStatus();
}

Status GetMLANdInfo(const LaunchParam &launchParam, MLAInfo &mmInfo,
                    OpParam::MLA &param, uint32_t blockDim)
{
    auto kcacheShape = launchParam.GetInTensor(DIM_2).desc.dims;
    auto KDims = kcacheShape.size();
    auto tableShape = launchParam.GetInTensor(DIM_4).desc.dims;
    mmInfo.kNz = (kcacheShape.at(KDims - 1) == NUM16 || kcacheShape.at(KDims - 1) == NUM32) ? 1 : 0;
    if (mmInfo.kNz) {
        mmInfo.embeddingSize = static_cast<int32_t>(kcacheShape.at(DIM_3)) *
                            static_cast<int32_t>(kcacheShape.at(DIM_1));
        mmInfo.blockSize = static_cast<int32_t>(kcacheShape.at(DIM_2));
    } else {
        mmInfo.embeddingSize = static_cast<int32_t>(kcacheShape.at(DIM_3));
        mmInfo.blockSize = static_cast<int32_t>(kcacheShape.at(DIM_1));
    }
    mmInfo.numTokens = static_cast<int32_t>(param.kvSeqLen.size());
    mmInfo.numBlocks = static_cast<int32_t>(kcacheShape.at(DIM_0));
    mmInfo.maxNumBlocksPerQuery = static_cast<int32_t>(tableShape.at(DIM_1));
    mmInfo.tor = param.tor;
    mmInfo.kvSeqLen = param.kvSeqLen.data();
    mmInfo.qSeqLen = param.qSeqLen.data();
    param.kvHead = param.kvHead <= 0 ? param.headSize : param.kvHead;
    mmInfo.batch = static_cast<int32_t>(param.kvSeqLen.size());
    mmInfo.kvHeads = param.kvHead;
    mmInfo.numHeads = static_cast<int32_t>(param.headSize);
    mmInfo.maskType = static_cast<int32_t>(param.maskType);
    mmInfo.quantFlag = (static_cast<int32_t>(mmInfo.type) < NUM2) ? 0 : 1;
    auto maxQSeqlen = mmInfo.qSeqLen != nullptr ? *std::max_element(param.qSeqLen.begin(), param.qSeqLen.end()) : 1;
    mmInfo.mtpTp1Flag = ((param.headSize == M_LIMIT) ||
                           (launchParam.GetInTensor(DIM_0).desc.dtype == TENSOR_DTYPE_INT8 && maxQSeqlen > 1));
    if (mmInfo.mtpTp1Flag || static_cast<int32_t>(mmInfo.type) >= NUM2) {
        mmInfo.maskType = 0;
    }
    int32_t beginQ = 0;
    for (int32_t batchIdx = 0; batchIdx < mmInfo.batch; batchIdx++) {
        mmInfo.batchList.push_back(BatchNode(batchIdx, *(mmInfo.kvSeqLen + batchIdx), beginQ));
        beginQ = mmInfo.qSeqLen == nullptr ? beginQ + 1 : beginQ + *(mmInfo.qSeqLen + batchIdx);
    }
    mmInfo.totalTaskNum = mmInfo.qSeqLen != nullptr ?
                          std::accumulate(mmInfo.qSeqLen, mmInfo.qSeqLen + mmInfo.batch, static_cast<int32_t>(0)) :
                          mmInfo.batch;
    if (mmInfo.mtpTp1Flag) {
        OP_TILING_CHECK_STATUS_RETURN(GetFlashDecodingInfo(mmInfo, param, blockDim));
    }
    return Status::OkStatus();
}

Status GetMLAInfo(const LaunchParam &launchParam, MLAInfo &mmInfo, OpParam::MLA &param, uint32_t blockDim)
{
    OP_TILING_CHECK_STATUS_RETURN(GetMLANdInfo(launchParam, mmInfo, param, blockDim));
    return Status::OkStatus();
}

Status GetTilingKeyTypeBase(MLAInfo &mmInfo, const Tensor &qTensor, const Tensor &qRopeTensor)
{
    if (qTensor.desc.dtype == TENSOR_DTYPE_BF16) {
        mmInfo.type = TilingKeyType::TILING_BF16_DATA;
    } else if (qTensor.desc.dtype == TENSOR_DTYPE_FLOAT16) {
        mmInfo.type = TilingKeyType::TILING_HALF_DATA;
    } else if (qRopeTensor.desc.dtype == TENSOR_DTYPE_FLOAT16) {
        mmInfo.type = TilingKeyType::TILING_INT8_HALF_DATA;
    } else {
        mmInfo.type = TilingKeyType::TILING_INT8_BF16_DATA;
    }
    return Status::OkStatus();
}

Status GenTilingKey(MLAInfo &mmInfo, KernelInfo &kernelInfo, OpParam::MLA &param)
{
    uint32_t dataType = static_cast<int32_t>(mmInfo.type);
    uint32_t tilingKey = dataType + (mmInfo.kNz << NUM4) + (mmInfo.mtpTp1Flag << NUM2) +
                         (param.isRing << NUM5) + (mmInfo.flashDecoding << NUM6);
    kernelInfo.SetTilingId(tilingKey);
    MKI_LOG(INFO) << "TILING KEY IS = " << tilingKey;
    return Status::OkStatus();
}

Status MLATiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto param = AnyCast<OpParam::MLA>(launchParam.GetParam());
    auto qTensor = launchParam.GetInTensor(DIM_0);
    auto qRopeTensor = launchParam.GetInTensor(DIM_1);

    MLAInfo mmInfo;
    GetTilingKeyTypeBase(mmInfo, qTensor, qRopeTensor);
    uint32_t blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    Status ret1 = GetMLAInfo(launchParam, mmInfo, param, blockDim);
    OP_TILING_CHECK_STATUS_RETURN(ret1);
    uint32_t *tilingParam = reinterpret_cast<uint32_t *>(kernelInfo.GetTilingHostAddr());
    uint64_t tilingSize = kernelInfo.GetTilingSize();
    Status ret = GetMLATilingParam(launchParam, mmInfo, blockDim, tilingParam, tilingSize);
    OP_TILING_CHECK_STATUS_RETURN(ret);
    uint32_t dataLenHalf = sizeof(uint16_t);
    uint32_t dataLenFloat = sizeof(float);
    uint32_t dataLenInt = sizeof(int32_t);
    uint64_t basicWorkSpaceHalf = blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenHalf;
    uint64_t basicWorkSpaceFloat = blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenFloat;
    uint64_t basicWorkSpaceInt8 = blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenInt;
    uint64_t oCoreWorkSpaceSize = mmInfo.flashDecoding && mmInfo.mtpTp1Flag ?
        WORKSPACE_BLOCK_SIZE_DB * mmInfo.flashDecodingTaskNum * mmInfo.splitKVNum * dataLenFloat * 2 : 0;
    uint64_t lWorkSpaceSize = mmInfo.flashDecoding && mmInfo.mtpTp1Flag ?
        mmInfo.numHeads * mmInfo.flashDecodingTaskNum * mmInfo.splitKVNum * dataLenFloat * 2 * 8 : 0;
    if (mmInfo.quantFlag) {
        uint64_t sWorkSpaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceFloat * 2 : basicWorkSpaceFloat;
        uint64_t pWorkSpaceSize = basicWorkSpaceInt8;
        uint64_t oTempWorkSpaceSize = basicWorkSpaceInt8 * 2;
        kernelInfo.GetScratchSizes() = {sWorkSpaceSize, sWorkSpaceSize, pWorkSpaceSize,
                                        oTempWorkSpaceSize, basicWorkSpaceFloat, oCoreWorkSpaceSize, lWorkSpaceSize};
    } else {
        uint64_t sWorkSpaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceFloat * 4 : basicWorkSpaceFloat * 2;
        uint64_t pWorkSpaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceHalf * 4 : basicWorkSpaceHalf * 2;
        uint64_t oTempWorkSpaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceFloat * 4 : basicWorkSpaceFloat * 2;
        uint64_t goWorkSpaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceFloat * 2 : basicWorkSpaceFloat;
        kernelInfo.GetScratchSizes() = {sWorkSpaceSize, NUM512, pWorkSpaceSize, oTempWorkSpaceSize,
                                        goWorkSpaceSize, oCoreWorkSpaceSize, lWorkSpaceSize};
    }
    Status ret2 = GenTilingKey(mmInfo, kernelInfo, param);
    OP_TILING_CHECK_STATUS_RETURN(ret2);
    kernelInfo.SetBlockDim(blockDim);
    MKI_LOG(INFO) << "launchBufferSize = " << tilingSize << " block dim = " << blockDim;
    return Status::OkStatus();
}

inline Status PrefillPreCheck(OpParam::MLA &param)
{
    MKI_CHECK(param.headSize > 0, "headSize is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK((param.kvHead > 0 && param.kvHead <= param.headSize && param.headSize % param.kvHead == 0) ||
                  (param.kvHead == 0),
              "kvHead is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(param.qSeqLen.size() == param.kvSeqLen.size(), "qSeqLen size Not equal to kvSeqLen",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(param.qSeqLen.data() != nullptr, "qSeq cannot be nullptr",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(param.kvSeqLen.data() != nullptr, "kvSeq cannot be nullptr",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}

Status GetSwaMaskInfo(MLAInfo &mmInfo, OpParam::MLA &param, const Tensor tensorMask)
{
    auto maxQSeq = std::max_element(param.qSeqLen.begin(), param.qSeqLen.end());
    if (*maxQSeq != 1) {
        auto maskShape = tensorMask.desc.dims;
        auto maskDim = maskShape.size();
        int32_t maxSeq = maskShape.at(maskDim - 1);
        mmInfo.maxSeqLen = maxSeq;
        MKI_CHECK(maskDim == DIM_2, "maskdim invalid",
            return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(maskShape.at(0) <= maskShape.at(1),
            "mask shape invalid, maxkv should be larger than maxq",
            return Status::FailStatus(ERROR_INVALID_VALUE));
    }
    return Status::OkStatus();
}

inline Status GetPrefiillMaskInfo(MLAInfo &mmInfo, OpParam::MLA &param,
                                  const Tensor &tensorMask)
{
    auto maskType = param.maskType;
    if (maskType == OpParam::MLA::MASK_TYPE_NONE || maskType == OpParam::MLA::MASK_TYPE_CAUSAL_MASK) {
        MKI_LOG(INFO) << "No Check for nomask or causal mask";
        return Status::OkStatus(); 
    }
    auto maskShape = tensorMask.desc.dims;
    auto maxKvSeq = std::max_element(param.kvSeqLen.begin(), param.kvSeqLen.end());
    MKI_LOG(INFO) << "max kv seq" << *maxKvSeq;
    if (maskType == OpParam::MLA::MASK_TYPE_SWA_NORM) {
        auto ret = GetSwaMaskInfo(mmInfo, param, tensorMask);
        return ret;
    }
    auto maskDim = maskShape.size();
    int32_t maxSeq = maskShape.at(maskDim - 1);
    mmInfo.maxSeqLen = maxSeq;
    MKI_CHECK(maskType == OpParam::MLA::MASK_TYPE_MASK_FREE,
                "mask type invalid",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(maskDim == DIM_2, "maskdim invalid",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(maskShape.at(1) == NORM_CMP_MASK_LEN, "compress mask shape should be 512, 512",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}

inline Status MLAPrefillPostCheck(MLAInfo &mmInfo, OpParam::MLA &param)
{
    MKI_CHECK(mmInfo.batch > 0 && mmInfo.batch <= ND_BATCH_LIMIT, "batch is invalid",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(static_cast<int32_t>(param.qSeqLen.size()) == mmInfo.batch, "SeqLen size is invalid",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.embeddingSize <= MAX_EMBEDDING && mmInfo.embeddingSize > 0, "headdimQ is invalid",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.embeddingSizeV <= MAX_EMBEDDING && mmInfo.embeddingSizeV > 0, "headdimV is invalid",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK((mmInfo.maskType == OpParam::MLA::MASK_TYPE_SWA_NORM) == (mmInfo.windowSize > 0),
                "swa windowSize should be greater than 0", return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}

inline void PrefillLog(MLAInfo &mmInfo, OpParam::MLA &param)
{
    MKI_LOG(INFO) << "batch is: " << mmInfo.batch << " kvMaxSeq is: " << mmInfo.maxKvSeqLen
                  << " maskMaxSeq is: " << mmInfo.maxSeqLen << " head is: " << mmInfo.numHeads
                  << " qSeq  is: " << *(mmInfo.qSeqLen) << " kvSeq is: " << *(mmInfo.kvSeqLen)
                  << " tor is : " << mmInfo.tor
                  << " kv_head is: " << mmInfo.kvHeads << " embed is: " << mmInfo.embeddingSize
                  << " maskStrid is: " << mmInfo.maskStride << " headStride is: " << mmInfo.headStride
                  << " mask type is " << mmInfo.maskType;
}

void MLAPrefillFillInfo(MLAInfo &mmInfo, OpParam::MLA &param, int32_t batch, int32_t embed)
{
    mmInfo.batch = batch;
    mmInfo.numHeads = param.headSize;
    mmInfo.embeddingSize = embed;
    mmInfo.embeddingSizeV = embed - ROPE_DIM;
    mmInfo.qSeqLen = param.qSeqLen.data();
    mmInfo.kvSeqLen = param.kvSeqLen.data();
    mmInfo.tor = param.tor;
    mmInfo.kvHeads = param.kvHead == 0 ? param.headSize : param.kvHead;
    mmInfo.maskType = static_cast<uint32_t>(param.maskType);
    mmInfo.windowSize = static_cast<uint32_t>(param.windowSize);
}

Status InitInfo(MLAInfo &mmInfo, OpParam::MLA &param)
{
    OP_TILING_CHECK_STATUS_RETURN(PrefillPreCheck(param));
    int32_t batch = 0;
    SVector<int64_t> kcacheShape;
    SVector<int64_t> vcacheShape;
    kcacheShape = mmInfo.tensors.kCache.desc.dims;
    int64_t queryDim = mmInfo.tensors.query.desc.dims.size();
    int32_t embed = 0; // headdim
    if (queryDim == DIM_3) {
        embed = (mmInfo.tensors.query.desc.dims).at(2) +
                    (mmInfo.tensors.queryRope.desc.dims).at(2);
    } else {
        embed = ((mmInfo.tensors.query.desc.dims).at(1) +
                    (mmInfo.tensors.queryRope.desc.dims).at(1)) / param.headSize;
    }
    auto kvSeqLen = param.kvSeqLen;
    batch = kvSeqLen.size();
    auto maxKvSeq = *std::max_element(param.kvSeqLen.begin(), param.kvSeqLen.end());
    mmInfo.maxKvSeqLen = maxKvSeq;
    OP_TILING_CHECK_STATUS_RETURN(GetPrefiillMaskInfo(mmInfo, param, mmInfo.tensors.mask));
    MLAPrefillFillInfo(mmInfo, param, batch, embed);
    vcacheShape = mmInfo.tensors.vCache.desc.dims;
    auto vcacheDim = vcacheShape.size();
    MKI_CHECK((vcacheShape[vcacheDim - 1] == mmInfo.embeddingSizeV) ||
                  (vcacheShape[vcacheDim - 1] == mmInfo.kvHeads * mmInfo.embeddingSizeV),
              "headdimV is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    OP_TILING_CHECK_STATUS_RETURN(MLAPrefillPostCheck(mmInfo, param));
    PrefillLog(mmInfo, param);
    return Status::OkStatus();
}

Status GenMlaPrefillTilingKey(MLAInfo &mmInfo, KernelInfo &kernelInfo, OpParam::MLA &param)
{
    // currently only support fp16/bf16 maskfree prefill or bf16 prefill kernel
    uint32_t dataType = static_cast<int32_t>(mmInfo.type);
    uint32_t tilingKey = dataType + (mmInfo.maskType << NUM1);
    kernelInfo.SetTilingId(tilingKey);
    MKI_LOG(INFO) << "MLA Prefill TILING KEY IS = " << tilingKey;
    return Status::OkStatus();
}

Status MLAPrefillTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    MLAInfo mmInfo;
    uint32_t index = 0;
    MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MLA), "OpParam is invalid",
              return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
    auto param = AnyCast<OpParam::MLA>(launchParam.GetParam());
    mmInfo.tensors.query = launchParam.GetInTensor(index++);
    mmInfo.tensors.queryRope = launchParam.GetInTensor(index++);
    mmInfo.tensors.kCache = launchParam.GetInTensor(index++);
    mmInfo.tensors.kCacheRope = launchParam.GetInTensor(index++);
    mmInfo.tensors.vCache = launchParam.GetInTensor(index++);
    mmInfo.tensors.mask = launchParam.GetInTensor(index++);
    mmInfo.tensors.alibiCoeff = launchParam.GetInTensor(index++);
    Status ret = InitInfo(mmInfo, param);
    OP_TILING_CHECK_STATUS_RETURN(ret);
    uint32_t blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    MKI_CHECK(blockDim > 0, "blockDim cannot <= 0", return Status::FailStatus(ERROR_INVALID_VALUE));
    mmInfo.blockDim = blockDim;
    uint8_t *tilingHost = kernelInfo.GetTilingHostAddr();
    uint64_t tilingSize = kernelInfo.GetTilingSize();
    uint32_t *tilingParam = reinterpret_cast<uint32_t *>(tilingHost);
    ret = GetMLAPrefillTilingParam(mmInfo, blockDim, tilingParam, tilingSize);
    OP_TILING_CHECK_STATUS_RETURN(ret);
    GetTilingKeyTypeBase(mmInfo, mmInfo.tensors.query, mmInfo.tensors.queryRope);
    ret = GenMlaPrefillTilingKey(mmInfo, kernelInfo, param);
    OP_TILING_CHECK_STATUS_RETURN(ret);
    uint64_t dataLenFloat = sizeof(float);
    uint64_t sSize = static_cast<uint64_t>(blockDim) * static_cast<uint64_t>(BLOCK_WORKSPACE_SIZE) * NUM16 * dataLenFloat;
    kernelInfo.GetScratchSizes() = {sSize, sSize, sSize, sSize * 2}; // oTmp/S/P
    kernelInfo.SetBlockDim(blockDim);
    return Status::OkStatus();
}

} // namespace AtbOps
