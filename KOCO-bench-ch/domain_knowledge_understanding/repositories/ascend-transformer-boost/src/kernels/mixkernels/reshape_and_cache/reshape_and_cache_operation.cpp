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

namespace AtbOps {
using namespace Mki;
constexpr uint32_t HEAD_SIZE_LIMITED = 16 * 1024; // head_size上限
constexpr uint32_t MIN_CAL_FACTOR = 32;     // rope模式下最后一维应是32的倍数
constexpr uint32_t ZERO = 0;
constexpr uint32_t INT8NZ = 32;
constexpr uint32_t FP16NZ = 16;

class ReshapeAndCacheOperation : public OperationBase {
public:
    explicit ReshapeAndCacheOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::ReshapeAndCache),
            "OpParam is invalid", return nullptr);
        auto param = AnyCast<OpParam::ReshapeAndCache>(launchParam.GetParam());
        switch (param.type) {
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND:
                return GetKernelByName("ReshapeAndCacheNdKernel");
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS:
                return GetKernelByName("ReshapeAndCacheCompressKernel");
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS_ROPE:
                return GetKernelByName("ReshapeAndCacheCompressRopeKernel");
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_NZ:
                return GetKernelByName("ReshapeAndCacheNzKernel");
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND_SISO:
                return GetKernelByName("ReshapeAndCacheNdSisoKernel");
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_OMNI_COMPRESS:
                return GetKernelByName("ReshapeAndCacheOmniCompressKernel");
            default:
                break;
        }
        MKI_LOG(ERROR) << "Unsupport reshape_and_cache type " << param.type;
        return nullptr;
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::ReshapeAndCache), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::ReshapeAndCache>(specificParam);
        switch (param.type) {
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND_SISO:
                return DIM_3;
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND:
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_NZ:
                return DIM_5;
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS:
                return DIM_7;
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS_ROPE:
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_OMNI_COMPRESS:
                return DIM_8;
            default:
                break;
        }
        return DIM_1;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::ReshapeAndCache), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::ReshapeAndCache>(specificParam);
        switch (param.type) {
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND_SISO:
                return DIM_1;
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND:
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_NZ:
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS:
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS_ROPE:
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_OMNI_COMPRESS:
                return DIM_2;
            default:
                break;
        }
        return DIM_1;
    }

    bool CheckCommonReshapeAndCache(const LaunchParam &launchParam) const
    {
        MKI_CHECK(launchParam.GetInTensorCount() >= 5, "input num invalid", return false);
        auto &key = launchParam.GetInTensor(DIM_0); // K.shape = [num_tokens, num_heads, head_size]
        auto &value = launchParam.GetInTensor(DIM_1); // V.shape = [num_tokens, num_heads, head_size]
        auto &keyCache = launchParam.GetInTensor(DIM_2);
        auto &valueCache = launchParam.GetInTensor(DIM_3);
        auto &slot = launchParam.GetInTensor(DIM_4);

        MKI_CHECK(!CheckEmptyTensor(key), "1st inTensor should not be null tensor ", return false);
        MKI_CHECK(!CheckEmptyTensor(value), "2nd inTensor should not be null tensor ", return false);
        MKI_CHECK(!CheckEmptyTensor(keyCache), "3rd inTensor should not be null tensor ", return false);
        MKI_CHECK(!CheckEmptyTensor(valueCache), "4th inTensor should not be null tensor ", return false);
        MKI_CHECK(!CheckEmptyTensor(slot), "5th inTensor should not be null tensor ", return false);

        auto indtypeK = key.desc.dtype;
        auto indtypeV = value.desc.dtype;
        MKI_CHECK(keyCache.desc.dtype == indtypeK && valueCache.desc.dtype == indtypeV,
            "K&KCache/V&VCache's dtype must be same", return false);
        MKI_CHECK(slot.desc.dtype == TENSOR_DTYPE_INT32, "slot should be int32", return false);

        MKI_CHECK(key.desc.dims.size() == DIM_3 && value.desc.dims.size() == DIM_3,
            "K&V's dim num should be " << DIM_3, return false);
        MKI_CHECK(keyCache.desc.dims.size() == DIM_4 && valueCache.desc.dims.size() == DIM_4,
            "KCache&VCache's dim num should be " << DIM_4, return false);
        MKI_CHECK(slot.desc.dims.size() == DIM_1, "slot's dim num should be " << DIM_1, return false);

        auto numTokens = key.desc.dims[DIM_0];
        auto numHeads = key.desc.dims[DIM_1];
        auto numBlocks = keyCache.desc.dims[DIM_0];
        auto blockSize = keyCache.desc.dims[DIM_1];
        auto headSizeK = key.desc.dims[DIM_2];
        auto headSizeV = value.desc.dims[DIM_2];

        if (headSizeK != headSizeV) {
            MKI_LOG(WARN) << "headSizeK is not equal to headSizeV";
        }

        MKI_CHECK(value.desc.dims[DIM_0] == numTokens && value.desc.dims[DIM_1] == numHeads,
            "dim_0 and dim_1 of K/V should be same", return false);
        MKI_CHECK(valueCache.desc.dims[DIM_0] == numBlocks &&
            valueCache.desc.dims[DIM_2] == keyCache.desc.dims[DIM_2],
            "dim_0, dim_2 of KCache/VCache should be same", return false);
        auto param = AnyCast<OpParam::ReshapeAndCache>(launchParam.GetParam());
        if (param.type != OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_NZ) {
            MKI_CHECK(indtypeK == indtypeV,
                "K&KCache&V&VCache's dtype must be same", return false);
            MKI_CHECK(valueCache.desc.dims[DIM_1] == blockSize,
                "dim_1 of KCache/VCache should be same", return false);
        }
        return true;
    }

    Status CheckReshapeAndCacheNdNz(const LaunchParam &launchParam) const
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 5, "input num invalid",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        auto &key = launchParam.GetInTensor(DIM_0);
        auto &value = launchParam.GetInTensor(DIM_1);
        auto &keyCache = launchParam.GetInTensor(DIM_2);
        auto &valueCache = launchParam.GetInTensor(DIM_3);
        auto &slot = launchParam.GetInTensor(DIM_4);
        auto numTokens = key.desc.dims[DIM_0];
        auto numHeads = key.desc.dims[DIM_1];
        auto headSizeK = key.desc.dims[DIM_2];
        auto headSizeV = value.desc.dims[DIM_2];
        auto offsetK = key.desc.offset;
        auto offsetV = value.desc.offset;

        auto indtype = key.desc.dtype;
        MKI_CHECK((indtype == TENSOR_DTYPE_FLOAT16 || indtype == TENSOR_DTYPE_BF16 || indtype == TENSOR_DTYPE_INT8),
            "K&V&KCache&VCache should be either int8 OR float16 OR bfloat16",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(slot.desc.dims[DIM_0] == numTokens, "slot's dim_0 should be same as numTokens",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(offsetK >= ZERO && offsetV >= ZERO, "offset should be greater than or equal 0",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        auto param = AnyCast<OpParam::ReshapeAndCache>(launchParam.GetParam());
        if (param.type == OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND) {
            MKI_CHECK(keyCache.desc.dims[DIM_2] == numHeads && keyCache.desc.dims[DIM_3] == headSizeK
                && valueCache.desc.dims[DIM_3] == headSizeV,
                "dim_2 and dim_3 of KCache&VCache should be same as K&V",
                return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        }
        if (param.type == OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_NZ) {
            if (indtype == TENSOR_DTYPE_INT8) {
                MKI_CHECK((keyCache.desc.dims[DIM_3] == INT8NZ
                    && numHeads * headSizeK == keyCache.desc.dims[DIM_1] * INT8NZ),
                    "keyCache's lastDim should be 32 when int8", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
            } else {
                MKI_CHECK((keyCache.desc.dims[DIM_3] == FP16NZ
                    && numHeads * headSizeK == keyCache.desc.dims[DIM_1] * FP16NZ),
                "keyCache's lastDim should be 16 when fp16/bf16", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
            }
            MKI_CHECK(valueCache.desc.dims[DIM_3] == FP16NZ
                && numHeads * headSizeV == valueCache.desc.dims[DIM_1] * FP16NZ,
                "valueCache's lastDim should be aligned to 16", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        }
        return Status::OkStatus();
    }

    bool CheckReshapeAndCacheCommonCompress(const LaunchParam &launchParam) const
    {
        auto &key = launchParam.GetInTensor(DIM_0);
        auto &value = launchParam.GetInTensor(DIM_1);
        auto &slot = launchParam.GetInTensor(DIM_4);
        auto &wins = launchParam.GetInTensor(DIM_5);
        auto &seqLens = launchParam.GetInTensor(DIM_6);

        MKI_CHECK(!CheckEmptyTensor(wins), "6th inTensor should not be null tensor ", return false);
        MKI_CHECK(!CheckEmptyTensor(seqLens), "7th inTensor should not be null tensor ", return false);

        MKI_CHECK(wins.desc.dims.size() == 1, "6th inTensor dims must be 1", return false);
        MKI_CHECK(seqLens.desc.dims.size() == 1, "7th inTensor dims must be 1", return false);

        auto batch = seqLens.desc.dims[DIM_0];
        auto numHeads = key.desc.dims[DIM_1];
        auto headSize = key.desc.dims[DIM_2];
        MKI_CHECK(headSize % 32 == 0,
            "headSize 3rd-dim of KV should be a multiple of 32.", return false);
        MKI_CHECK(headSize == value.desc.dims[DIM_2],
            "headSize 3rd-dim of KV should be same.", return false);
        MKI_CHECK(launchParam.GetInTensor(DIM_2).desc.dims[DIM_3] == headSize
            && launchParam.GetInTensor(DIM_3).desc.dims[DIM_3] == headSize,
            "headSize 4rd-dim of KVCache should be same.", return false);
        MKI_CHECK(launchParam.GetInTensor(DIM_2).desc.dims[DIM_2] == DIM_1,
            "numHeads 3rd-dim of KVCache should be 1.", return false);
        MKI_CHECK(slot.desc.dims[DIM_0] == numHeads * batch,
            "Shape of slot[" << slot.desc.dims[DIM_0] << "] invalid, should be " << numHeads * batch, return false);
        MKI_CHECK(wins.desc.dims[DIM_0] == numHeads * batch,
            "Shape of wins[" << wins.desc.dims[DIM_0] << "] invalid, should be " << numHeads * batch, return false);
        return true;
    }

    Status CheckReshapeAndCacheCompress4Alibi(const LaunchParam &launchParam) const
    {
        MKI_CHECK(CheckReshapeAndCacheCommonCompress(launchParam),
            "Failed to check common compress launch param", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensorCount() == 7,
            "input num invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        auto &key = launchParam.GetInTensor(DIM_0);
        auto indtype = key.desc.dtype;
        MKI_CHECK((indtype == TENSOR_DTYPE_FLOAT16 || indtype == TENSOR_DTYPE_BF16 || indtype == TENSOR_DTYPE_INT8),
            "K&V&KCache&VCache should be either int8 OR float16 OR bfloat16",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        return Status::OkStatus();
    }

    Status CheckReshapeAndCacheCompress4RoPE(const LaunchParam &launchParam) const
    {
        MKI_CHECK(CheckReshapeAndCacheCommonCompress(launchParam),
            "Failed to check common compress launch param", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensorCount() == 8,
            "input num invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        auto &offsetIdx = launchParam.GetInTensor(DIM_7);
        MKI_CHECK(!CheckEmptyTensor(offsetIdx),
            "8th inTensor should not be null tensor ", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(offsetIdx.desc.dims.size() == 1,
            "8th inTensor dims must be 1", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        auto &key = launchParam.GetInTensor(DIM_0);
        auto &seqLens = launchParam.GetInTensor(DIM_6);
        auto indtype = key.desc.dtype;
        MKI_CHECK((indtype == TENSOR_DTYPE_FLOAT16 || indtype == TENSOR_DTYPE_BF16),
            "K&V&KCache&VCache should be either float16 OR bfloat16",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        
        auto batch = seqLens.desc.dims[DIM_0];
        auto numHeads = key.desc.dims[DIM_1];
        auto headSize = key.desc.dims[DIM_2];
        MKI_CHECK(headSize % MIN_CAL_FACTOR == 0 && headSize != 0 && headSize <= HEAD_SIZE_LIMITED,
            "headSize shape is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(offsetIdx.desc.dims[DIM_0] == numHeads * batch,
            "Shape of offsetIdx[" << offsetIdx.desc.dims[DIM_0] << "] invalid, should be " << numHeads * batch,
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        return Status::OkStatus();
    }

    bool CheckReshapeAndCacheNDSISO(const LaunchParam &launchParam) const
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::ReshapeAndCache),
            "OpParam is invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 3,
            "input num invalid", return false);
        auto &key = launchParam.GetInTensor(DIM_0); // K.shape = [num_tokens, num_heads, head_size]
        auto &keyCache = launchParam.GetInTensor(DIM_1);
        auto &slot = launchParam.GetInTensor(DIM_2);
        auto &offsetK = key.desc.offset;

        auto indtype = key.desc.dtype;
        MKI_CHECK(keyCache.desc.dtype == indtype,
            "K&KCache's dtype must be same", return false);
        MKI_CHECK((indtype == TENSOR_DTYPE_FLOAT16 || indtype == TENSOR_DTYPE_BF16 || indtype == TENSOR_DTYPE_INT8),
            "K&KCache should be both float16 OR int8 OR bfloat16", return false);
        MKI_CHECK(slot.desc.dtype == TENSOR_DTYPE_INT32,
            "slot should be int32", return false);
        MKI_CHECK(offsetK >= ZERO, "offset should be greater than or equal 0", return false);

        MKI_CHECK(key.desc.dims.size() == DIM_3,
            "K's dim num should be " << DIM_3, return false);
        MKI_CHECK(keyCache.desc.dims.size() == DIM_4,
            "KCache's dim num should be " << DIM_4, return false);
        MKI_CHECK(slot.desc.dims.size() == DIM_1,
            "slot's dim num should be " << DIM_1, return false);

        MKI_CHECK(!CheckEmptyTensor(key), "1st inTensor should not be null tensor ", return false);
        MKI_CHECK(!CheckEmptyTensor(keyCache), "2rd inTensor should not be null tensor ", return false);
        MKI_CHECK(!CheckEmptyTensor(slot), "3th inTensor should not be null tensor ", return false);

        auto numTokens = key.desc.dims[DIM_0];
        auto numHeads = key.desc.dims[DIM_1];
        auto headSizeK = key.desc.dims[DIM_2];

        MKI_CHECK(keyCache.desc.dims[DIM_2] == numHeads && keyCache.desc.dims[DIM_3] == headSizeK,
            "dim_2 and dim_3 of KCache and dim_1 and dim_2 of key should be same", return false);
        MKI_CHECK(slot.desc.dims[DIM_0] == numTokens,
            "slot's dim_0 should be same as numTokens", return false);

        return true;
    }
    Status CheckReshapeAndCacheOmniCompress4RoPE(const LaunchParam &launchParam) const
    {
        MKI_CHECK(CheckReshapeAndCacheCommonCompress(launchParam),
            "Failed to check common compress launch param", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(launchParam.GetInTensorCount() == 8,
            "input num invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        auto &offsetIdx = launchParam.GetInTensor(DIM_7);
        MKI_CHECK(!CheckEmptyTensor(offsetIdx),
            "8th inTensor should not be null tensor ", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(offsetIdx.desc.dims.size() == 1,
            "8th inTensor dims must be 1", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        auto &key = launchParam.GetInTensor(DIM_0);
        auto &seqLens = launchParam.GetInTensor(DIM_6);
        auto indtype = key.desc.dtype;
        MKI_CHECK((indtype == TENSOR_DTYPE_FLOAT16 || indtype == TENSOR_DTYPE_BF16),
            "K&V&KCache&VCache should be either float16 OR bfloat16",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        
        auto batch = seqLens.desc.dims[DIM_0];
        auto numHeads = key.desc.dims[DIM_1];
        auto headSize = key.desc.dims[DIM_2];
        MKI_CHECK(headSize % MIN_CAL_FACTOR == 0 && headSize != 0 && headSize <= HEAD_SIZE_LIMITED,
            "headSize shape is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(offsetIdx.desc.dims[DIM_0] == numHeads * batch,
            "Shape of offsetIdx[" << offsetIdx.desc.dims[DIM_0] << "] invalid, should be " << numHeads * batch,
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        return Status::OkStatus();
    }
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::ReshapeAndCache),
            "OpParam is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        auto param = AnyCast<OpParam::ReshapeAndCache>(launchParam.GetParam());
        // 仅检查siso算子
        if (param.type == OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND_SISO) {
            MKI_CHECK(CheckReshapeAndCacheNDSISO(launchParam),
                "Failed to check launch param",
                return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
            auto &tensorKeyCacheIn = launchParam.GetInTensor(DIM_1);
            outTensors[DIM_0] = tensorKeyCacheIn;
            return Status::OkStatus();
        }

        // 1. 检查多种reshape算子 公共部分
        MKI_CHECK(CheckCommonReshapeAndCache(launchParam), "Failed to check common launch param",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        auto &tensorKeyCacheIn = launchParam.GetInTensor(DIM_2);
        auto &tensorValueCacheIn = launchParam.GetInTensor(DIM_3);
        outTensors[DIM_0] = tensorKeyCacheIn;
        outTensors[DIM_1] = tensorValueCacheIn;

        // 2. 检查各自reshape算子 私有部分
        switch (param.type) {
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND:
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_NZ:
                return CheckReshapeAndCacheNdNz(launchParam);
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS:
                return CheckReshapeAndCacheCompress4Alibi(launchParam);
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS_ROPE:
                return CheckReshapeAndCacheCompress4RoPE(launchParam);
            case OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_OMNI_COMPRESS:
                return CheckReshapeAndCacheOmniCompress4RoPE(launchParam);
            default:
                return Status::FailStatus(ERROR_ATTR_INVALID_TYPE,
                    "Failed to check reshape param, type of specificParam is invalid");
        }

        return Status::OkStatus();
    }
};
REG_OPERATION(ReshapeAndCacheOperation);
} //    namespace AtbOps
