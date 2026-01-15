/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <algorithm>
#include <numeric>
#include "mki/utils/assert/assert.h"
#include "mki/utils/checktensor/check_tensor.h"
#include "mki/utils/platform/platform_info.h"
#include "ring_mla_tiling.h"

namespace AtbOps {

const int32_t NUM1 = 1;
const int32_t NUM2 = 2;
const int32_t NUM3 = 3;
const int32_t NUM4 = 4;
const int32_t NUM5 = 5;
const int32_t NUM16 = 16;
const int32_t NUM32 = 32;
const int32_t NUM64 = 64;
const int32_t NUM256 = 256;
const int32_t NUM512 = 512;
const int32_t NUM576 = 576;
const float SPLITKV_RATION = 0.8;
const int32_t DEFAULT_HEADDIM = 192;

Status GetMLANdInfo(const LaunchParam &launchParam, RINGMLAInfo &mmInfo,
                    OpParam::RINGMLA &param)
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
    mmInfo.mtpTp1Flag = (mmInfo.numHeads == M_LIMIT); // quant not support
    if (mmInfo.mtpTp1Flag) {
        mmInfo.maskType = 0;
    }
    if (static_cast<int32_t>(mmInfo.type) >= NUM2) {
        mmInfo.maskType = 0;
    }
    if (mmInfo.qSeqLen != nullptr) {
        mmInfo.totalTaskNum = std::accumulate(mmInfo.qSeqLen, mmInfo.qSeqLen + mmInfo.batch, static_cast<int32_t>(0));
    } else {
        mmInfo.totalTaskNum = mmInfo.batch;
    }
    return Status::OkStatus();
}

Status GetRINGMLAInfo(const LaunchParam &launchParam, RINGMLAInfo &mmInfo, OpParam::RINGMLA &param)
{
    OP_TILING_CHECK_STATUS_RETURN(GetMLANdInfo(launchParam, mmInfo, param));
    return Status::OkStatus();
}

Status GetTilingKeyTypeBase(RINGMLAInfo &mmInfo, const Tensor &qTensor, const Tensor &qRopeTensor)
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

Status GenTilingKey(RINGMLAInfo &mmInfo, KernelInfo &kernelInfo, OpParam::RINGMLA &param)
{
    uint32_t dataType = static_cast<int32_t>(mmInfo.type);
    uint32_t tilingKey = dataType + (mmInfo.kNz << NUM4) + (mmInfo.mtpTp1Flag << NUM2) + (param.isRing << NUM5);
    kernelInfo.SetTilingId(tilingKey);
    MKI_LOG(INFO) << "TILING KEY IS = " << tilingKey;
    return Status::OkStatus();
}

Status RINGMLATiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto param = AnyCast<OpParam::RINGMLA>(launchParam.GetParam());
    auto qTensor = launchParam.GetInTensor(DIM_0);
    auto qRopeTensor = launchParam.GetInTensor(DIM_1);

    RINGMLAInfo mmInfo;
    GetTilingKeyTypeBase(mmInfo, qTensor, qRopeTensor);
    Status ret1  = GetRINGMLAInfo(launchParam, mmInfo, param);
    uint32_t blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    uint32_t *tilingParam = reinterpret_cast<uint32_t *>(kernelInfo.GetTilingHostAddr());
    uint64_t tilingSize = kernelInfo.GetTilingSize();
    Status ret = GetRINGMLATilingParam(launchParam, mmInfo, blockDim, tilingParam, tilingSize);
    OP_TILING_CHECK_STATUS_RETURN(ret);
    uint32_t dataLenHalf = sizeof(uint16_t);
    uint32_t dataLenFloat = sizeof(float);
    uint32_t dataLenInt = sizeof(int32_t);
    uint64_t basicWorkSpaceHalf = blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenHalf;
    uint64_t basicWorkSpaceFloat = blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenFloat;
    uint64_t basicWorkSpaceInt8 = blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenInt;
    bool isQuant = (static_cast<int32_t>(mmInfo.type) < NUM2) ? 0 : 1;
    if (isQuant) {
        uint64_t sWorkSpaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceFloat * 2 : basicWorkSpaceFloat;
        uint64_t pWorkSpaceSize = basicWorkSpaceInt8;
        uint64_t oTempWorkSpcaceSize = basicWorkSpaceInt8 * 2;
        kernelInfo.GetScratchSizes() = {sWorkSpaceSize, sWorkSpaceSize, pWorkSpaceSize,
                                        oTempWorkSpcaceSize, basicWorkSpaceFloat};
    } else {
        uint64_t sWorkSpaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceFloat * 4 : basicWorkSpaceFloat * 2;
        uint64_t pWorkSpaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceHalf * 4 : basicWorkSpaceHalf * 2;
        uint64_t oTempWorkSpcaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceFloat * 4 : basicWorkSpaceFloat * 2;
        uint64_t goWorkSpcaceSize = mmInfo.mtpTp1Flag ? basicWorkSpaceFloat * 2 : basicWorkSpaceFloat;
        kernelInfo.GetScratchSizes() = {sWorkSpaceSize, NUM512, pWorkSpaceSize, oTempWorkSpcaceSize,
                                        goWorkSpcaceSize};
    }
    Status ret2 = GenTilingKey(mmInfo, kernelInfo, param);
    OP_TILING_CHECK_STATUS_RETURN(ret2);
    kernelInfo.SetBlockDim(blockDim);
    MKI_LOG(INFO) << "launchBufferSize = " << tilingSize << " block dim = " << blockDim;
    return Status::OkStatus();
}

inline Status PrefillPreCheck(OpParam::RINGMLA &param)
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
    MKI_CHECK(param.kTensorList.size() == param.vTensorList.size(), "k and v cache have different batch number",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}

Status GetAlibiMaskInfo(RINGMLAInfo &mmInfo, OpParam::RINGMLA &param, const Tensor &tensorMask,
                        int32_t maxSeq)
{
    auto maskShape = tensorMask.desc.dims;
    auto maskDim = maskShape.size();
    MKI_CHECK(maskDim <= DIM_4 && maskDim >= DIM_2, "maskdim invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    if (maskDim == DIM_3) { // [h, ms, ms]
        mmInfo.maskStride = 0;
        mmInfo.headStride = static_cast<uint32_t>(maxSeq);
    } else if (maskDim == DIM_4) { // [bs,1,ms,ms]  [bs,headnum,ms,ms]
        MKI_CHECK(maskShape.at(DIM_2) * maskShape.at(1) <= UINT32_MAX, "maxSeq * headnum can not large than UINT32_MAX",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        mmInfo.maskStride = maskShape.at(1) * maskShape.at(DIM_2);
        mmInfo.headStride = static_cast<uint32_t>(maxSeq);
    } else if (maskDim == DIM_2) {
        MKI_CHECK(maxSeq == LONG_SEQ_ALIBI_LEN, "alibi mask shape must be [256, 256] for long seq opt",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
    }
    MKI_CHECK(static_cast<uint64_t>(mmInfo.headStride) * static_cast<uint32_t>(maxSeq) <= UINT32_MAX,
              "maxSeq can not exceed UINT32_MAX", return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}

inline Status GetPrefiillMaskInfo(RINGMLAInfo &mmInfo, OpParam::RINGMLA &param,
                                  const Tensor &tensorMask)
{
    auto &tensorAlibiCoeff = mmInfo.tensors.alibiCoeff;
    auto maskShape = tensorMask.desc.dims;
    auto maskType = param.maskType;
    auto maxQSeq = std::max_element(param.qSeqLen.begin(), param.qSeqLen.end());
    auto maxKvSeq = std::max_element(param.kvSeqLen.begin(), param.kvSeqLen.end());
    MKI_LOG(INFO) << "max kv seq" << *maxKvSeq;
    if (maskType != OpParam::RINGMLA::MASK_TYPE_NONE) {
        auto maskDim = maskShape.size();
        int32_t maxSeq = maskShape.at(maskDim - 1);
        mmInfo.maxSeqLen = maxSeq;
        if (!(maxSeq == NORM_CMP_MASK_LEN && param.isTriuMask != 0) && CheckEmptyTensor(tensorAlibiCoeff)) {
            MKI_CHECK(*maxQSeq <= maxSeq, "qSeqLen larger than maxSeqLen",
                      return Status::FailStatus(ERROR_INVALID_VALUE));
            MKI_CHECK(*maxKvSeq <= maxSeq, "kvSeqLen larger than maxSeqLen",
                      return Status::FailStatus(ERROR_INVALID_VALUE));
        }
        if (maskType == OpParam::RINGMLA::MASK_TYPE_ALIBI) {
            auto ret = GetAlibiMaskInfo(mmInfo, param, tensorMask, maxSeq);
            OP_TILING_CHECK_STATUS_RETURN(ret);
        } else {
            MKI_CHECK(maskDim <= DIM_3 && maskDim >= DIM_2, "maskdim invalid",
                      return Status::FailStatus(ERROR_INVALID_VALUE));
            MKI_CHECK(maskShape.at(1) <= UINT32_MAX, "maxShape can not exceed UINT32_MAX",
                      return Status::FailStatus(ERROR_INVALID_VALUE));
            if (maskDim == DIM_2) { // [ms,ms]
                mmInfo.maskStride = 0;
                mmInfo.headStride = 0;
            } else if (maskDim == DIM_3) { // [bs,ms,ms]
                mmInfo.maskStride = maskShape.at(1);
                mmInfo.headStride = 0;
            }
            if (maxSeq == NORM_CMP_MASK_LEN && param.isTriuMask != 0) {
                mmInfo.isLongSeq = 1;
            }
        }
    }
    return Status::OkStatus();
}

inline Status MLAPrefillPostCheck(RINGMLAInfo &mmInfo, OpParam::RINGMLA &param)
{
    MKI_CHECK(mmInfo.batch > 0 && mmInfo.batch <= ND_BATCH_LIMIT, "batch is invalid",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(static_cast<int32_t>(param.qSeqLen.size()) == mmInfo.batch, "SeqLen size is invalid",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.isTriuMask == 0 || mmInfo.isTriuMask == 1, "param isTriuMask is invalid",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.embeddingSize <= MAX_EMBEDDING && mmInfo.embeddingSize > 0, "headdimQ is invalid",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.embeddingSizeV <= MAX_EMBEDDING && mmInfo.embeddingSizeV > 0, "headdimV is invalid",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}

inline void PrefillLog(RINGMLAInfo &mmInfo, OpParam::RINGMLA &param)
{
    MKI_LOG(INFO) << "batch is: " << mmInfo.batch << " kvMaxSeq is: " << mmInfo.maxKvSeqLen
                  << " maskMaxSeq is: " << mmInfo.maxSeqLen << " head is: " << mmInfo.numHeads
                  << " qSeq  is: " << *(mmInfo.qSeqLen) << " kvSeq is: " << *(mmInfo.kvSeqLen)
                  << " tor is : " << mmInfo.tor
                  << " kv_head is: " << mmInfo.kvHeads << " embed is: " << mmInfo.embeddingSize
                  << " maskStrid is: " << mmInfo.maskStride << " headStride is: " << mmInfo.headStride << " isTriuMask "
                  << mmInfo.isTriuMask << " mask type is " << mmInfo.maskType <<  " longseq is " << mmInfo.isLongSeq
                  << "isRing is" << mmInfo.isRing;
}

void MLAPrefillFillInfo(RINGMLAInfo &mmInfo, OpParam::RINGMLA &param, size_t batch, int32_t embed)
{
    mmInfo.batch = static_cast<int32_t>(batch);
    mmInfo.numHeads = param.headSize;
    mmInfo.embeddingSize = embed;
    mmInfo.embeddingSizeV = embed - NUM64;
    mmInfo.qSeqLen = param.qSeqLen.data();
    mmInfo.kvSeqLen = param.kvSeqLen.data();
    mmInfo.tor = param.tor;
    mmInfo.kvHeads = param.kvHead == 0 ? param.headSize : param.kvHead;
    mmInfo.isTriuMask = param.isTriuMask;
    mmInfo.kTensorList = param.kTensorList;
    mmInfo.vTensorList = param.vTensorList;
    mmInfo.maskType = static_cast<uint32_t>(param.maskType);
    mmInfo.quantType = static_cast<uint32_t>(param.quantType);
    mmInfo.isRing =  param.isRing;
}

Status InitInfo(RINGMLAInfo &mmInfo, OpParam::RINGMLA &param)
{
    OP_TILING_CHECK_STATUS_RETURN(PrefillPreCheck(param));
    size_t batch = 0;
    SVector<int64_t> kcacheShape;
    SVector<int64_t> vcacheShape;
    kcacheShape = mmInfo.tensors.kCache.desc.dims;
    int32_t embed = DEFAULT_HEADDIM;

    int32_t maxKvSeqLen = 0;
    if (kcacheShape.size() == DIM_3) {
        for (std::size_t i = 0; i < param.kvSeqLen.size(); i++) {
            maxKvSeqLen = std::max(maxKvSeqLen, static_cast<int32_t>(param.kvSeqLen[i]));
        }
    } else if (kcacheShape.size() == DIM_4) {
        maxKvSeqLen = static_cast<int32_t>(kcacheShape.at(2));
    }
    mmInfo.maxKvSeqLen = maxKvSeqLen; // MaxSeqLen is the 2st dimension of K
    batch = param.kvSeqLen.size();

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

Status RINGMLAPrefillTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    RINGMLAInfo mmInfo;
    uint32_t index = 0;
    MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::RINGMLA), "OpParam is invalid",
              return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
    auto param = AnyCast<OpParam::RINGMLA>(launchParam.GetParam());
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
    ret = GetRINGMLAPrefillTilingParam(mmInfo, blockDim, tilingParam, tilingSize);
    OP_TILING_CHECK_STATUS_RETURN(ret);
    uint64_t dataLenFloat = sizeof(float);
    uint64_t sSize = static_cast<uint64_t>(blockDim) * static_cast<uint64_t>(32768) * 16 * dataLenFloat;
    kernelInfo.GetScratchSizes() = {sSize, sSize, sSize, sSize * 2}; // oTmp/S/P
    kernelInfo.SetBlockDim(blockDim);
    return Status::OkStatus();
}

} // namespace AtbOps
