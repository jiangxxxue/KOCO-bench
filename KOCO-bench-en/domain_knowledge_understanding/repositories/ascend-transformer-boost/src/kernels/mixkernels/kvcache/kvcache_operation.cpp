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
#include "atbops/params/params.h"

namespace AtbOps {
using namespace Mki;
class KVCacheOperation : public OperationBase {
public:
    explicit KVCacheOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::KVCache),
                  "OpParam is invalid", return nullptr);
        auto param = launchParam.GetParam<OpParam::KVCache>();
        switch (param.type) {
            case OpParam::KVCache::KVCACHE_ND:
                return GetKernelByName("KVCacheNdKernel");
            case OpParam::KVCache::KVCACHE_NZ:
                return GetKernelByName("KVCacheNzKernel");
            case OpParam::KVCache::KVCACHE_DYNAMIC_BATCH:
                return GetKernelByName("KVCacheNdDynamicBatchKernel");
            case OpParam::KVCache::KVCACHE_ND_PARAMS:
                return GetKernelByName("KVCacheNdParamsKernel");
            case OpParam::KVCache::KVCACHE_NZ_PARAMS:
                return GetKernelByName("KVCacheNzParamsKernel");
            case OpParam::KVCache::KVCACHE_DYNAMIC_BATCH_PARAMS:
                return GetKernelByName("KVCacheNdDynamicBatchParamsKernel");
            default:
                break;
        }
        MKI_LOG(ERROR) << "Unsupport KVCache type " << param.type;
        return nullptr;
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::KVCache), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::KVCache>(specificParam);
        if (param.type == OpParam::KVCache::KVCACHE_ND_PARAMS
            || param.type == OpParam::KVCache::KVCACHE_DYNAMIC_BATCH_PARAMS
            || param.type == OpParam::KVCache::KVCACHE_NZ_PARAMS) {
            return DIM_3;
        }
        return DIM_5; // KvCache has 5 inputs: newkv, layer_id, cache_in, token_offset, seqlen
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::KVCache), "OpParam is invalid", return 0);
        return 1;
    }

    bool CheckKVCacheParams(const LaunchParam &launchParam, bool dyncBatchFlag, Tensor& cacheIn) const
    {
        auto param = AnyCast<OpParam::KVCache>(launchParam.GetParam());
        MKI_CHECK(param.seqLen.size() == param.tokenOffset.size(),
            "batch in seq len and token are not equal", return false);
        MKI_CHECK(dyncBatchFlag ? (cacheIn.desc.dims[DIM_0] >= static_cast<int64_t>(param.tokenOffset.size())) :
                                    (cacheIn.desc.dims[DIM_1] == static_cast<int64_t>(param.tokenOffset.size())),
            "batch in cache should be greater or equal to in seq len and token", return false);
        return true;
    }

    bool CheckKVCache(const LaunchParam &launchParam, bool dyncBatchFlag, const Tensor& layerId, Tensor& cacheIn) const
    {
        auto tokenOffset = launchParam.GetInTensor(DIM_3);
        MKI_CHECK(tokenOffset.desc.dtype == TENSOR_DTYPE_INT32 || tokenOffset.desc.dtype == TENSOR_DTYPE_UINT32,
                  "input token offset data type error", return false);
        MKI_CHECK(tokenOffset.desc.format == TENSOR_FORMAT_ND, "token offset data format error", return false);
        MKI_CHECK(tokenOffset.desc.dims.size() == DIM_1,  "input token offset data dims error", return false);

        auto seqLen = launchParam.GetInTensor(DIM_4);
        MKI_CHECK(seqLen.desc.dtype == TENSOR_DTYPE_INT32 || seqLen.desc.dtype == TENSOR_DTYPE_UINT32,
                  "input seq len data type error", return false);
        MKI_CHECK(seqLen.desc.format == TENSOR_FORMAT_ND, "input seq len data format error", return false);
        MKI_CHECK(seqLen.desc.dims.size() == DIM_1, "input seq len data dims error", return false);

        MKI_CHECK(seqLen.desc.dims[DIM_0] == tokenOffset.desc.dims[DIM_0],
            "batch in seq len and token are not equal", return false);
        MKI_CHECK(dyncBatchFlag ? (cacheIn.desc.dims[DIM_0] >= tokenOffset.desc.dims[DIM_0]) :
                                     (cacheIn.desc.dims[DIM_1] == tokenOffset.desc.dims[DIM_0]),
            "batch in cache should be greater or equal to in seq len and token", return false);

        MKI_CHECK(layerId.desc.dtype == seqLen.desc.dtype && layerId.desc.dtype == tokenOffset.desc.dtype,
                  "layerId seqLen tokenOffset data type are not same", return false);
        return true;
    }

    bool CheckKVCacheCommon(const LaunchParam &launchParam) const
    {
        auto param = AnyCast<OpParam::KVCache>(launchParam.GetParam());
        auto newKV = launchParam.GetInTensor(DIM_0);
        bool nzFlag = (newKV.desc.format == TENSOR_FORMAT_FRACTAL_NZ);
        TensorDType dtype = newKV.desc.dtype;
        MKI_CHECK(dtype == TENSOR_DTYPE_FLOAT16 || dtype == TENSOR_DTYPE_BF16 || dtype == TENSOR_DTYPE_INT8,
            "input new kv data type error", return false);
        MKI_CHECK(newKV.desc.format == TENSOR_FORMAT_ND || nzFlag, "input new kv data format error", return false);
        MKI_CHECK(newKV.desc.dims.size() == (nzFlag ? DIM_4 : DIM_2), "input new kv data dims error: " <<
            newKV.desc.dims.size(), return false);

        auto layerId = launchParam.GetInTensor(DIM_1);
        MKI_CHECK(layerId.desc.dtype == TENSOR_DTYPE_INT32 || layerId.desc.dtype == TENSOR_DTYPE_UINT32,
                  "input layer id data type error", return false);
        MKI_CHECK(layerId.desc.format == TENSOR_FORMAT_ND, "input layer id data format error", return false);
        MKI_CHECK(layerId.desc.dims.size() == DIM_1, "input layer id data dims error", return false);

        auto cacheIn = launchParam.GetInTensor(DIM_2);
        nzFlag = (nzFlag && (cacheIn.desc.format == TENSOR_FORMAT_FRACTAL_NZ));
        bool dyncBatchFlag = (cacheIn.desc.dims.size() == DIM_3);
        MKI_CHECK(cacheIn.desc.dtype == dtype, "cacheIn and newKV are of different dtype", return false);
        MKI_CHECK(cacheIn.desc.format == TENSOR_FORMAT_ND || nzFlag, "input cache data format error", return false);
        MKI_CHECK(cacheIn.desc.dims.size() == (nzFlag ? DIM_5 : (dyncBatchFlag ? DIM_3 : DIM_4)),
            "input cache data dims error", return false);
        MKI_CHECK((nzFlag ? (newKV.desc.dims[DIM_1] == cacheIn.desc.dims[DIM_2]) :
            (newKV.desc.dims[DIM_1] == (dyncBatchFlag ? cacheIn.desc.dims[DIM_2] : cacheIn.desc.dims[DIM_3]))),
            "hiddenSize in newKv and cacheIn are not equal", return false);

        if (param.type == OpParam::KVCache::KVCACHE_ND_PARAMS
            || param.type == OpParam::KVCache::KVCACHE_DYNAMIC_BATCH_PARAMS
            || param.type == OpParam::KVCache::KVCACHE_NZ_PARAMS) {
            return CheckKVCacheParams(launchParam, dyncBatchFlag, cacheIn);
        }
        return CheckKVCache(launchParam, dyncBatchFlag, layerId, cacheIn);
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::KVCache),
            "OpParam is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "OpParam is invalid"));
        auto param = AnyCast<OpParam::KVCache>(launchParam.GetParam());
        MKI_LOG(INFO) << "infer shape type:" << param.type;
        MKI_CHECK(CheckKVCacheCommon(launchParam), "Failed to check launch param",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        auto &tensorcacheIn = launchParam.GetInTensor(DIM_2);
        outTensors[DIM_0].desc = tensorcacheIn.desc;

        return Status::OkStatus();
    }
};

REG_OPERATION(KVCacheOperation);
} // namespace AtbOps
