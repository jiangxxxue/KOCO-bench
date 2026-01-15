/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <mki/base/operation_base.h>
#include <mki/utils/log/log.h>
#include <mki/utils/const/op_const.h>
#include <mki_loader/op_register.h>
#include <mki/utils/checktensor/check_tensor.h>
#include "atbops/params/params.h"

static constexpr int32_t LONG_SEQ_LEN = 128;
static constexpr int32_t SWA_COMPRESS_MASK_SIZE = 512;

namespace AtbOps {
using namespace Mki;
class UnpadFlashAttentionNzOperation : public OperationBase {
public:
    explicit UnpadFlashAttentionNzOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::UnpadFlashAttentionNz), "OpParam is invalid",
                  return nullptr);
        auto param = AnyCast<OpParam::UnpadFlashAttentionNz>(launchParam.GetParam());
        switch (param.type) {
            case OpParam::UnpadFlashAttentionNz::UNPAD_ALIBI_FLASH_ATTENTION_NZ:
            case OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_ENCODER:
            case OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ:
            case OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_ENCODER_NOCACHE:
                return GetKernelByName("UnpadFlashAttentionNzEncoderKernel");
            case OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_DECODER:
                return GetKernelByName("UnpadFlashAttentionNzDecoderKernel");
            default:
                break;
        }
        return nullptr;
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return DIM_7; // 7 inputs
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return DIM_1; // 1 output
    }

private:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::UnpadFlashAttentionNz), "OpParam is invalid",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        auto param = AnyCast<OpParam::UnpadFlashAttentionNz>(launchParam.GetParam());
        MKI_LOG(INFO) << "infer shape param: " << param.headSize << ", type:" << param.type;
        switch (param.type) {
            case OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ:
            case OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_ENCODER:
            case OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_DECODER:
            case OpParam::UnpadFlashAttentionNz::UNPAD_ALIBI_FLASH_ATTENTION_NZ:
                return InferShapeUnpadFlashAttentionNz(launchParam, outTensors);
            case OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_ENCODER_NOCACHE:
                return InferShapeUnpadFlashAttentionNzEncoderNocache(launchParam, outTensors);
            default:
                break;
        }
        return Status::OkStatus();
    }

    Status InferShapeUnpadFlashAttentionNz(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const
    {
        MKI_CHECK(CheckUnpadFlashAttentionNz(launchParam), "Failed to check launch param",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        outTensors[DIM_0].desc = tensorQ.desc;

        return Status::OkStatus();
    }

    Status InferShapeUnpadFlashAttentionNzEncoderNocache(const LaunchParam &launchParam,
        SVector<Tensor> &outTensors) const
    {
        MKI_CHECK(CheckUnpadFlashAttentionNzEncoderNocache(launchParam), "Failed to check run info",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check run info"));
        auto &tensorQ = launchParam.GetInTensor(DIM_0);
        outTensors[DIM_0].desc = tensorQ.desc;

        return Status::OkStatus();
    }

    bool CheckUnpadFAKvBatchwiseNz(const OpParam::UnpadFlashAttentionNz &param, const Tensor &tensorQ) const
    {
        uint32_t batch = param.kvSeqLen.size();
        MKI_CHECK(param.kTensorList.size() == batch && param.vTensorList.size() == batch,
            "kv batch num invalid", return false);
        auto kShape = param.kTensorList[0].desc.dims;
        MKI_CHECK(kShape.size() == 3, "kv batch shape size wrong", return false);
        for (size_t i = 0; i < batch; ++i) {
            MKI_CHECK(param.kTensorList[i].desc.dims == kShape && param.vTensorList[i].desc.dims == kShape,
                         "kv batch shape inconsistent, should all be [hiddenSize / 16, maxSeq, 16]", return false);
            MKI_CHECK(param.kTensorList[i].data != nullptr && param.vTensorList[i].data != nullptr,
                         "kv batchwise ptr cannot be nullptr", return false);
            MKI_CHECK(param.kTensorList[i].desc.dtype == TENSOR_DTYPE_FLOAT16,
                         "k dtype invalid, should be either fp16", return false);
            MKI_CHECK(param.vTensorList[i].desc.dtype == TENSOR_DTYPE_FLOAT16,
                         "v dtype invalid, should be either fp16", return false);
            MKI_CHECK(param.kTensorList[i].desc.dtype == tensorQ.desc.dtype &&
                         param.vTensorList[i].desc.dtype == tensorQ.desc.dtype,
                         "Q K V dtype should be the same", return false);
            MKI_CHECK(param.kTensorList[i].desc.dims[DIM_2] == 16, "K Shape should be in nz format", return false);
            MKI_CHECK(param.vTensorList[i].desc.dims[DIM_2] == 16, "V Shape should be in nz format", return false);
        }
        auto kvSeqLen = param.kvSeqLen;
        auto maxKvSeqlenIter = std::max_element(kvSeqLen.begin(), kvSeqLen.end());
        MKI_CHECK(kShape[1] >= *maxKvSeqlenIter,
                     "batchwise maxSeqlen should exceed real kv maxlen", return false);
        return true;
    }

    struct ShapeParam {
        int32_t maxQ;
        int32_t maxKv;
        uint32_t batch;
    };

    bool FindMask(std::vector<std::pair<SVector<int64_t>, bool>> &pairs, SVector<int64_t> &curShape,
                  bool nz) const
    {
        auto target =
            std::find_if(pairs.begin(), pairs.end(), [curShape, nz](std::pair<SVector<int64_t>, bool> iter) {
                if (!iter.second || curShape.size() != iter.first.size()) {
                    return false;
                }
                uint32_t count = 0;
                for (int32_t i = curShape.size() - 1; i >= 0; i--, count++) {
                    // batch, head应该完全一致，maxQ和maxKv保证能够覆盖
                    if (count < (nz ? DIM_3 : DIM_2)) {
                        if (iter.first[i] > curShape[i]) {
                            return false;
                        }
                    } else {
                        if (iter.first[i] != curShape[i]) {
                            return false;
                        }
                    }
                }
                return true;
            });
        return target != pairs.end() && (!nz || (*target).first[DIM_3] == FP16_ALIGN_NUM);
    }

    bool CheckMaskPre(const Tensor &tensorMask, const Tensor &q) const
    {
        MKI_CHECK(q.desc.dtype == tensorMask.desc.dtype, "mask data type not consitent with q", return false);
        MKI_CHECK(tensorMask.desc.dtype == TENSOR_DTYPE_FLOAT16,
                     "Input4 dtype should be float16", return false);
        return true;
    }

    bool CheckNdMask(const Tensor &tensorMask, Tensor &q, const ShapeParam &shapePara,
                     const OpParam::UnpadFlashAttentionNz &param) const
    {
        auto maxQ = shapePara.maxQ;
        auto maxKv = shapePara.maxKv;
        auto batch = shapePara.batch;
        auto headSize = param.headSize;
        constexpr int32_t longSeqAlibiLen = 256;
        auto currentShape = tensorMask.desc.dims;
        auto sz = currentShape.size();
        MKI_CHECK(sz >= DIM_2, "mask invalid, please check.", return false);
        auto maskLen = currentShape[sz - 1];
        auto alibi = param.maskType == OpParam::UnpadFlashAttentionNz::MASK_TYPE_ALIBI;
        auto norm = param.maskType == OpParam::UnpadFlashAttentionNz::MASK_TYPE_NORM;
        auto lookAhead = param.maskType == OpParam::UnpadFlashAttentionNz::MASK_TYPE_LOOK_AHEAD;
        auto isLongSeq = (param.isTriuMask == 1) && (maskLen == LONG_SEQ_LEN);
        auto kvHead = param.kvHead == 0 ? headSize : param.kvHead;
        auto isAlibiCompress = maskLen == LONG_SEQ_LEN && currentShape[sz - DIM_2] != maskLen && alibi;
        std::vector<std::pair<SVector<int64_t>, bool>> supports = {
            {{maxQ, maxKv}, true},
            {{LONG_SEQ_LEN, LONG_SEQ_LEN}, isLongSeq},
            {{batch, LONG_SEQ_LEN, LONG_SEQ_LEN}, isLongSeq},
            {{longSeqAlibiLen, longSeqAlibiLen}, alibi && sz == DIM_2},
            {{q.desc.dims[DIM_0], maxKv}, lookAhead},
            {{batch, maxQ, maxKv}, norm},
            {{headSize, maxQ, maxKv}, alibi},
            {{static_cast<int32_t>(batch) / kvHead, maxQ, maxKv}, norm && param.compressHead},
            {{headSize, maxQ, maxKv}, alibi && param.compressHead},
            {{headSize, maxQ, LONG_SEQ_LEN}, isAlibiCompress},
            {{batch, headSize, maxQ, maxKv}, true},
            {{static_cast<int32_t>(batch) / kvHead, headSize, maxQ, maxKv}, alibi && param.compressHead},
        };
        // 保证mask一定能覆盖S，核内不会出现异常，用户保证1.避免多传;2.数值正常
        MKI_CHECK(FindMask(supports, currentShape, false), "current mask shape is unsupported!", return false);
        return true;
    }

    bool CheckNzMask(const Tensor &tensorMask, const ShapeParam &shapePara,
        const OpParam::UnpadFlashAttentionNz &param) const
    {
        auto headSize = param.headSize;
        constexpr int32_t longSeqAlibiLen = 256;
        auto currentShape = tensorMask.desc.dims;
        auto sz = currentShape.size();
        MKI_CHECK(sz == DIM_4, "mask invalid, please check.", return false);
        auto maskLen = currentShape[DIM_2];
        auto alibi = param.maskType == OpParam::UnpadFlashAttentionNz::MASK_TYPE_ALIBI;
        auto isLongSeq = (param.isTriuMask == 1) && (maskLen == LONG_SEQ_LEN);
        constexpr int32_t MAX_SAFE_VALUE = FP16_ALIGN_NUM - 1;
        MKI_CHECK(shapePara.maxQ <= INT32_MAX - MAX_SAFE_VALUE,
                "shapePara.maxQ is too large, please check", return false);
        MKI_CHECK(shapePara.maxKv <= INT32_MAX - MAX_SAFE_VALUE,
                "shapePara.maxKv is too large, please check", return false);
        auto maxNzQ = (shapePara.maxQ + FP16_ALIGN_NUM - 1) / FP16_ALIGN_NUM * FP16_ALIGN_NUM;
        auto maxKv = (shapePara.maxKv + FP16_ALIGN_NUM - 1) / FP16_ALIGN_NUM * FP16_ALIGN_NUM;
        auto batch = shapePara.batch;
        auto isAlibiCompress =
            currentShape[DIM_1] * currentShape[DIM_3] == LONG_SEQ_LEN && currentShape[DIM_2] > LONG_SEQ_LEN && alibi;
        auto alibiDim2 = currentShape[DIM_1] * currentShape[DIM_3] == longSeqAlibiLen &&
                         currentShape[DIM_2] == longSeqAlibiLen && alibi;
        auto isSwaCompress = param.maskType == OpParam::UnpadFlashAttentionNz::MASK_TYPE_SWA_COMPRESS;
        std::vector<std::pair<SVector<int64_t>, bool>> supports = {
            {{1, maxKv / FP16_ALIGN_NUM, maxNzQ, FP16_ALIGN_NUM}, true},
            {{1, longSeqAlibiLen / FP16_ALIGN_NUM, longSeqAlibiLen, FP16_ALIGN_NUM}, alibiDim2},
            {{1, LONG_SEQ_LEN / FP16_ALIGN_NUM, LONG_SEQ_LEN, FP16_ALIGN_NUM}, isLongSeq},
            {{batch, maxKv / FP16_ALIGN_NUM, maxNzQ, FP16_ALIGN_NUM}, true},
            {{headSize, LONG_SEQ_LEN / FP16_ALIGN_NUM, maxNzQ, FP16_ALIGN_NUM}, isAlibiCompress},
            {{headSize, maxKv / FP16_ALIGN_NUM, maxNzQ, FP16_ALIGN_NUM}, alibi},
            {{static_cast<int32_t>(batch) * headSize, maxKv / FP16_ALIGN_NUM, maxNzQ, FP16_ALIGN_NUM}, alibi},
            {{1, SWA_COMPRESS_MASK_SIZE / FP16_ALIGN_NUM, SWA_COMPRESS_MASK_SIZE, FP16_ALIGN_NUM}, isSwaCompress},
        };
        // 保证mask一定能覆盖S，核内不会出现异常，用户保证1.避免多传;2.数值正常
        MKI_CHECK(FindMask(supports, currentShape, true), "current mask shape is unsupported!", return false);
        return true;
    }

    bool CheckMask(const LaunchParam &launchParam, Tensor q, Tensor mask) const
    {
        auto param = AnyCast<OpParam::UnpadFlashAttentionNz>(launchParam.GetParam());
        MKI_CHECK((param.maskType == OpParam::UnpadFlashAttentionNz::MASK_TYPE_NONE
                || (param.type == OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_DECODER
                && param.maskType == OpParam::UnpadFlashAttentionNz::MASK_TYPE_SWA_NORM))
                ^ (!CheckEmptyTensor(mask)),
            "mask type inconsistent", return false); // 当Mask Type为None，或Decoder场景为SWA时，mask_tensor为空
        if (CheckEmptyTensor(mask)) {
            return true;
        }
        MKI_CHECK_NO_LOG(CheckMaskPre(mask, q), return false);
        auto head = param.headSize;
        auto qSeqLen = param.qSeqLen;
        auto kvSeqLen = param.kvSeqLen;
        if (param.type == OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ && kvSeqLen.size() == 0) {
            auto batch = qSeqLen.size() > 0 ? qSeqLen.size() : launchParam.GetInTensor(DIM_4).desc.dims[0];
            auto currentShape = mask.desc.dims;
            MKI_CHECK(currentShape.size() == DIM_4, "mask invalid, please check.", return false);
            MKI_CHECK(currentShape[DIM_3] == FP16_ALIGN_NUM, "not nz mask", return false);
            MKI_CHECK(currentShape[DIM_2] >= FP16_ALIGN_NUM, "not nz decoder mask", return false);
            auto cs0 = currentShape[DIM_0];
            MKI_CHECK((cs0 == 1) || (cs0 == head) || (cs0 == static_cast<int64_t>(batch)) ||
                             (cs0 == static_cast<int64_t>(static_cast<int32_t>(batch) * head)),
                         "not nz decoder mask", return false);
        } else {
            // batch
            uint32_t batch = kvSeqLen.size();
            MKI_CHECK(batch > 0, "batch invalid, please check", return false);
            // head
            MKI_CHECK(head > 0, "head invalid, please check", return false);
            // maxKvSeqlen
            auto maxKvSeqlenIter = std::max_element(kvSeqLen.begin(), kvSeqLen.end());
            MKI_CHECK(maxKvSeqlenIter != kvSeqLen.end() && *maxKvSeqlenIter > 0, "kvSeqlen invalid, please check",
                         return false);
            auto minKvSeqlenIter = std::min_element(kvSeqLen.begin(), kvSeqLen.end());
            MKI_CHECK((minKvSeqlenIter != kvSeqLen.end() && *minKvSeqlenIter >= 0),
                         "kvSeqlen min value invalid, please check", return false);
            // maxQSeqlen
            auto maxQSeqlenIter = std::max_element(qSeqLen.begin(), qSeqLen.end());
            auto maxQ = maxQSeqlenIter != qSeqLen.end() ? *maxQSeqlenIter : 1;
            MKI_CHECK(maxQ > 0, "qSeqlen max value invalid, please check", return false);
            auto minQSeqlenIter = std::min_element(qSeqLen.begin(), qSeqLen.end());
            MKI_CHECK((minQSeqlenIter == qSeqLen.end()) ||
                             ((minQSeqlenIter != qSeqLen.end() && *minQSeqlenIter >= 0)),
                         "qSeqlen min value invalid, please check", return false);
            MKI_LOG(INFO) << "[batch, head, maxQ, maxKv]: [" << batch << ", " << head << ", " << maxQ << ", "
                          << *maxKvSeqlenIter << "]";
            MKI_CHECK(*maxKvSeqlenIter >= maxQ, "maxQ & maxKv inconsistent.", return false);
            ShapeParam shapePara = {maxQ, *maxKvSeqlenIter, batch};
            return CheckNzMask(mask, shapePara, param);
        }
        return true;
    }

    bool CheckUnpadFlashAttentionNz(const LaunchParam &launchParam) const
    {
        auto param = AnyCast<OpParam::UnpadFlashAttentionNz>(launchParam.GetParam());
        auto &tensorQ = launchParam.GetInTensor(DIM_0); // Q.shape = [batch, hiddenSize/16, ntokens/16, 16]
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16,
            "Input0 dtype " << GetStrWithDType(tensorQ.desc.dtype) << " invalid, should be float16",
            return false);
        auto &tensorKcache = launchParam.GetInTensor(DIM_1);
        auto &tensorVcache = launchParam.GetInTensor(DIM_2);
        if (CheckEmptyTensor(tensorKcache) || CheckEmptyTensor(tensorVcache)) {
            MKI_CHECK(CheckEmptyTensor(tensorKcache) && CheckEmptyTensor(tensorVcache),
                " Normal k and v should both be empty tensor if batches are split", return false);
            MKI_CHECK(CheckUnpadFAKvBatchwiseNz(param, tensorQ), " Kv batchwise settings invalid", return false);
        } else {
            MKI_CHECK(tensorKcache.desc.dims.size() == DIM_5,
                "Input2 dim num " << tensorKcache.desc.dims.size() << " invalid, should be " << DIM_5,
                return false);
            MKI_CHECK(tensorVcache.desc.dims.size() == DIM_5,
                "Input3 dim num " << tensorVcache.desc.dims.size() << " invalid, should be " << DIM_5,
                return false);
            MKI_CHECK(tensorKcache.desc.dims[DIM_4] == 16, " K Shape should be in nz format", return false);
            MKI_CHECK(tensorVcache.desc.dims[DIM_4] == 16, " V Shape should be in nz format", return false);
            MKI_CHECK(tensorKcache.desc.dtype == TENSOR_DTYPE_FLOAT16,
                "Input1 dtype " << GetStrWithDType(tensorKcache.desc.dtype) << " invalid, should be float16",
                return false);
            MKI_CHECK(tensorVcache.desc.dtype == TENSOR_DTYPE_FLOAT16,
                "Input2 dtype " << GetStrWithDType(tensorVcache.desc.dtype) << " invalid, should be float16",
                return false);
        }
        MKI_CHECK(tensorQ.desc.dims.size() == DIM_4,
            "Input1 dim num " << tensorQ.desc.dims.size() << " invalid, should be " << DIM_4, return false);
        MKI_CHECK(tensorQ.desc.dims[DIM_3] == 16, "Q Shape should be in nz format", return false);
        auto &tensorLayerId = launchParam.GetInTensor(DIM_3);
        MKI_CHECK(tensorLayerId.desc.dims.size() == 1, "Input4 dim num should be 1", return false);
        MKI_CHECK(!(param.type == OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_DECODER
                && param.maskType == OpParam::UnpadFlashAttentionNz::MASK_TYPE_SWA_COMPRESS),
            "SWA Compress feature is not supported in decoder currently", return false);
        // check mask
        MKI_CHECK(CheckMask(launchParam, tensorQ, launchParam.GetInTensor(DIM_4)),
            "mask invalid", return false);
        return true;
    }

    bool CheckUnpadFlashAttentionNzEncoderNocache(const LaunchParam &launchParam) const
    {
        auto param = AnyCast<OpParam::UnpadFlashAttentionNz>(launchParam.GetParam());
        auto &tensorQ = launchParam.GetInTensor(DIM_0); // Q.shape = [1, hiddenSize/16, ntokens/16, 16]
        MKI_CHECK(tensorQ.desc.dtype == TENSOR_DTYPE_FLOAT16,
            "Input0 dtype " << GetStrWithDType(tensorQ.desc.dtype) << " invalid, should be float16",
            return false);
        auto &tensorKcache = launchParam.GetInTensor(DIM_1); // K.shape = [1, hiddenSize/16, ntokens/16, 16]
        MKI_CHECK(tensorKcache.desc.dtype == TENSOR_DTYPE_FLOAT16,
            "Input1 dtype " << GetStrWithDType(tensorKcache.desc.dtype) << " invalid, should be float16",
            return false);
        auto &tensorVcache = launchParam.GetInTensor(DIM_2); // V.shape = [1, hiddenSize/16, ntokens/16, 16]
        MKI_CHECK(tensorVcache.desc.dtype == TENSOR_DTYPE_FLOAT16,
            "Input2 dtype " << GetStrWithDType(tensorVcache.desc.dtype) << " invalid, should be float16",
            return false);
        MKI_CHECK(tensorQ.desc.dims.size() == DIM_4,
            "Input1 dim num " << tensorQ.desc.dims.size() << " invalid, should be " << DIM_4,
            return false);
        MKI_CHECK(tensorKcache.desc.dims.size() == DIM_4,
            "Input2 dim num " << tensorKcache.desc.dims.size() << " invalid, should be " << DIM_4,
            return false);
        MKI_CHECK(tensorVcache.desc.dims.size() == DIM_4,
            "Input3 dim num " << tensorVcache.desc.dims.size() << " invalid, should be " << DIM_4,
            return false);
        MKI_CHECK(tensorQ.desc.dims[DIM_3] == 16,
            "Q Shape should be in nz format", return false);
        MKI_CHECK(tensorKcache.desc.dims[DIM_3] == 16,
            "K Shape should be in nz format", return false);
        MKI_CHECK(tensorVcache.desc.dims[DIM_3] == 16,
            "V Shape should be in nz format", return false);
        // check mask
        MKI_CHECK(CheckMask(launchParam, tensorQ, launchParam.GetInTensor(DIM_4)),
            "mask invalid", return false);
        return true;
    }
};

REG_OPERATION(UnpadFlashAttentionNzOperation);
} //    namespace AtbOps