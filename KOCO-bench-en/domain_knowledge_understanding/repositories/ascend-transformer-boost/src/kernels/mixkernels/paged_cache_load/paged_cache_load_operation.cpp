/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

class PagedCacheLoadOperation : public OperationBase {
public:
    explicit PagedCacheLoadOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::PagedCacheLoad),
            "OpParam is invalid", return nullptr);
        auto param = AnyCast<OpParam::PagedCacheLoad>(launchParam.GetParam());
        switch (param.type) {
            case OpParam::PagedCacheLoad::PAGED_CACHE_LOAD_NZ:
                return GetKernelByName("PagedCacheLoadNzKernel");
            case OpParam::PagedCacheLoad::PAGED_CACHE_LOAD_ND:
                return GetKernelByName("PagedCacheLoadNdKernel");
            default:
                break;
        }
        MKI_LOG(ERROR) << "Unsupport PagedCacheLoad type " << param.type;
        return nullptr;
    }

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::PagedCacheLoad), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::PagedCacheLoad>(specificParam);
        switch (param.type) {
            case OpParam::PagedCacheLoad::PAGED_CACHE_LOAD_NZ:
            case OpParam::PagedCacheLoad::PAGED_CACHE_LOAD_ND:
                return DIM_7;
            default:
                break;
        }
        return DIM_1;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::PagedCacheLoad), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::PagedCacheLoad>(specificParam);
        switch (param.type) {
            case OpParam::PagedCacheLoad::PAGED_CACHE_LOAD_NZ:
            case OpParam::PagedCacheLoad::PAGED_CACHE_LOAD_ND:
                return DIM_2;
            default:
                break;
        }
        return DIM_1;
    }

    bool CheckCommonPagedCacheLoad(const LaunchParam &launchParam) const
    {
        auto &keyCache = launchParam.GetInTensor(DIM_0);
        auto &valueCache = launchParam.GetInTensor(DIM_1);
        auto &blockTables = launchParam.GetInTensor(DIM_2);
        auto &contextLens = launchParam.GetInTensor(DIM_3);
        auto &key = launchParam.GetInTensor(DIM_4);
        auto &value = launchParam.GetInTensor(DIM_5);

        MKI_CHECK(!CheckEmptyTensor(keyCache), "1st inTensor should not be null tensor ", return false);
        MKI_CHECK(!CheckEmptyTensor(valueCache), "2nd inTensor should not be null tensor ", return false);
        MKI_CHECK(!CheckEmptyTensor(blockTables), "3rd inTensor should not be null tensor ", return false);
        MKI_CHECK(!CheckEmptyTensor(contextLens), "4rd inTensor should not be null tensor ", return false);
        MKI_CHECK(!CheckEmptyTensor(key), "5rd inTensor should not be null tensor ", return false);
        MKI_CHECK(!CheckEmptyTensor(value), "6rd inTensor should not be null tensor ", return false);

        auto inDtype = keyCache.desc.dtype;
        MKI_CHECK(valueCache.desc.dtype == inDtype, "KCache&VCache's dtype must be same", return false);

        MKI_CHECK(blockTables.desc.dtype == TENSOR_DTYPE_INT32, "blockTables should be int32", return false);
        MKI_CHECK(contextLens.desc.dtype == TENSOR_DTYPE_INT32, "contextLens should be int32", return false);

        MKI_CHECK(blockTables.desc.dims.size() == DIM_2, "blockTables's dim num should be " << DIM_2, return false);
        MKI_CHECK(contextLens.desc.dims.size() == DIM_1, "contextLens's dim num should be " << DIM_1, return false);

        auto param = AnyCast<OpParam::PagedCacheLoad>(launchParam.GetParam());
        if (!param.cuSeqLens) {
            MKI_CHECK(blockTables.desc.dims[DIM_0] == contextLens.desc.dims[DIM_0],
                "contextLens's dim_0 must be same as blockTables's dim_0", return false);
        } else {
            MKI_CHECK((blockTables.desc.dims[DIM_0]+1) == contextLens.desc.dims[DIM_0],
                "contextLens's dim_0 must be same as blockTables's dim_0+1", return false);
        }
        if (param.hasSeqStarts) {
            auto &seqStarts = launchParam.GetInTensor(DIM_6);
            MKI_CHECK(!CheckEmptyTensor(seqStarts), "7rd inTensor should not be null tensor ", return false);
            MKI_CHECK(seqStarts.desc.dtype == TENSOR_DTYPE_INT32, "seqStarts should be int32", return false);
            MKI_CHECK(seqStarts.desc.dims.size() == DIM_1, "seqStarts's dim num should be " << DIM_1, return false);
            MKI_CHECK(blockTables.desc.dims[DIM_0] == seqStarts.desc.dims[DIM_0],
                "seqStarts's dim_0 must be same as blockTables's dim_0", return false);
        }

        return true;
    }

    Status CheckPagedCacheLoadNd(const LaunchParam &launchParam) const
    {
        auto &keyCache = launchParam.GetInTensor(DIM_0);
        auto &valueCache = launchParam.GetInTensor(DIM_1);

        auto &key = launchParam.GetOutTensor(DIM_0);
        auto &value = launchParam.GetOutTensor(DIM_1);

        auto inDtype = keyCache.desc.dtype;
        MKI_CHECK((inDtype == TENSOR_DTYPE_FLOAT16 || inDtype == TENSOR_DTYPE_BF16 || inDtype == TENSOR_DTYPE_INT8),
            "K&V&KCache&VCache should be either int8 OR float16 OR bfloat16",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        MKI_CHECK(keyCache.desc.dims.size() == DIM_4 && valueCache.desc.dims.size() == DIM_4,
            "KCache&VCache's dim num should be " << DIM_4, return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        auto numBlocks = keyCache.desc.dims[DIM_0];
        auto blockSize = keyCache.desc.dims[DIM_1];
        MKI_CHECK(valueCache.desc.dims[DIM_0] == numBlocks && valueCache.desc.dims[DIM_1] == blockSize,
            "dim_0, dim_1 of KCache/VCache should be same", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        auto numHeads = keyCache.desc.dims[DIM_2];
        auto headSizeK = keyCache.desc.dims[DIM_3];
        auto headSizeV = valueCache.desc.dims[DIM_3];
        constexpr int32_t baiscBlockSize = 32;
        MKI_CHECK(numHeads * headSizeK * GetTensorElementSize(inDtype) % baiscBlockSize == 0,
            "dim_2*dim_3 of KCache should be 32B aligned", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(numHeads * headSizeV * GetTensorElementSize(inDtype) % baiscBlockSize == 0,
            "dim_2*dim_3 of VCache should be 32B aligned", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        MKI_CHECK(key.desc.dims.size() == DIM_3 && value.desc.dims.size() == DIM_3,
            "K&V's dim num should be " << DIM_3, return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(key.desc.dtype == inDtype && value.desc.dtype == inDtype,
            "K&V's dtype must be same as KCache&VCache", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(key.desc.dims[DIM_0] == value.desc.dims[DIM_0],
            "K&V's dim_0 must be same", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        MKI_CHECK(key.desc.dims[DIM_1] == keyCache.desc.dims[DIM_2] && key.desc.dims[DIM_2] == keyCache.desc.dims[DIM_3],
            "dim_1/2 of K must be same as dim_2/3 of KCache", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(value.desc.dims[DIM_1] == valueCache.desc.dims[DIM_2] && value.desc.dims[DIM_2] == valueCache.desc.dims[DIM_3],
            "dim_1/2 of V must be same as dim_2/3 of VCache", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        return Status::OkStatus();
    }

    Status CheckPagedCacheLoadNz(const LaunchParam &launchParam) const
    {
        auto &keyCache = launchParam.GetInTensor(DIM_0);
        auto &valueCache = launchParam.GetInTensor(DIM_1);

        auto &key = launchParam.GetOutTensor(DIM_0);
        auto &value = launchParam.GetOutTensor(DIM_1);

        auto inDtype = keyCache.desc.dtype;
        MKI_CHECK((inDtype == TENSOR_DTYPE_FLOAT16 || inDtype == TENSOR_DTYPE_BF16 || inDtype == TENSOR_DTYPE_INT8),
            "K&V&KCache&VCache should be either int8 OR float16 OR bfloat16",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        MKI_CHECK(keyCache.desc.dims.size() == DIM_4 && valueCache.desc.dims.size() == DIM_4,
            "KCache&VCache's dim num should be " << DIM_4, return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        auto numBlocks = keyCache.desc.dims[DIM_0];
        auto blockSize = keyCache.desc.dims[DIM_2];
        MKI_CHECK(valueCache.desc.dims[DIM_0] == numBlocks && valueCache.desc.dims[DIM_2] == blockSize,
            "dim_0, dim_2 of KCache/VCache should be same", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        MKI_CHECK(key.desc.dims.size() == DIM_2 && value.desc.dims.size() == DIM_2,
            "K&V's dim num should be " << DIM_2, return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(key.desc.dtype == inDtype && value.desc.dtype == inDtype,
            "K&V's dtype must be same as KCache&VCache", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
        MKI_CHECK(key.desc.dims[DIM_0] == value.desc.dims[DIM_0],
            "K&V's dim_0 must be same", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        uint32_t eleNumAligned = 32; // int8
        if (inDtype == TENSOR_DTYPE_FLOAT16 || inDtype == TENSOR_DTYPE_BF16) {
            eleNumAligned = 16; // fp16, bf16
        }
        MKI_CHECK(keyCache.desc.dims[DIM_3] == eleNumAligned && valueCache.desc.dims[DIM_3] == eleNumAligned
            && key.desc.dims[DIM_1] == keyCache.desc.dims[DIM_1] * eleNumAligned
            && value.desc.dims[DIM_1] == valueCache.desc.dims[DIM_1] * eleNumAligned,
            "NZ format tensor dim should be aligned to eleNumAligned",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        return Status::OkStatus();
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::PagedCacheLoad),
            "OpParam is invalid", return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        // check common part
        MKI_CHECK(CheckCommonPagedCacheLoad(launchParam), "Failed to check common launch param",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR));

        // generate output
        auto &tensorKeyIn = launchParam.GetInTensor(DIM_4);
        auto &tensorValueIn = launchParam.GetInTensor(DIM_5);
        outTensors[DIM_0] = tensorKeyIn;
        outTensors[DIM_1] = tensorValueIn;

        // check private part
        auto param = AnyCast<OpParam::PagedCacheLoad>(launchParam.GetParam());
        switch (param.type) {
            case OpParam::PagedCacheLoad::PAGED_CACHE_LOAD_NZ:
                return CheckPagedCacheLoadNz(launchParam);
            case OpParam::PagedCacheLoad::PAGED_CACHE_LOAD_ND:
                return CheckPagedCacheLoadNd(launchParam);
            default:
                return Status::FailStatus(ERROR_ATTR_INVALID_TYPE,
                    "Failed to check reshape param, type of specificParam is invalid");
        }
        return Status::OkStatus();
    }
};
REG_OPERATION(PagedCacheLoadOperation);
} // namespace AtbOps
