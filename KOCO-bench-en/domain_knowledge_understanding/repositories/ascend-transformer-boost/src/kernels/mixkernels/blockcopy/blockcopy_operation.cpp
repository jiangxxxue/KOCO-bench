/*
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
class BlockCopyOperation : public OperationBase {
public:
    explicit BlockCopyOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::BlockCopy), "OpParam is invalid", return 0);
        return DIM_5;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::BlockCopy), "OpParam is invalid", return 0);
        return DIM_2;
    }

    bool CheckBlockCopy(const LaunchParam &launchParam) const
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::BlockCopy),
            "OpParam is invalid", return false);
        
        MKI_CHECK(launchParam.GetInTensorCount() == 5, "input num invalid", return false);

        return CheckKVCache(launchParam) && CheckSrcBlockList(launchParam) && CheckDistBlockIndices(launchParam);
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(CheckBlockCopy(launchParam), "Failed to check launch param",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        
        auto& keyCacheIn = launchParam.GetInTensor(DIM_0);
        auto& valueCacheIn = launchParam.GetInTensor(DIM_1);
        outTensors[DIM_0] = keyCacheIn;
        outTensors[DIM_1] = valueCacheIn;
        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::BlockCopy),
            "OpParam is invalid", return nullptr);
        return GetKernelByName("BlockCopyKernel");
    }

private:
    bool CheckKVCache(const LaunchParam &launchParam) const
    {
        auto &keyCache = launchParam.GetInTensor(DIM_0); // K.shape = [block_count, block_size, num_heads, head_size]
        auto &valueCache = launchParam.GetInTensor(DIM_1); // V.shape = [block_count, block_size, num_heads, head_size]

        MKI_CHECK(!CheckEmptyTensor(keyCache), "1st inTensor should not be null tensor ", return false);
        MKI_CHECK(!CheckEmptyTensor(valueCache), "2nd inTensor should not be null tensor ", return false);

        auto inDtype = keyCache.desc.dtype;
        MKI_CHECK(valueCache.desc.dtype == inDtype, "K&V's dtype must be same", return false);
        MKI_CHECK(inDtype == TENSOR_DTYPE_FLOAT16 || inDtype == TENSOR_DTYPE_BF16 ||
            inDtype == TENSOR_DTYPE_INT8, "K&V should be both float16 OR int8 OR bfloat16", return false);

        MKI_CHECK(keyCache.desc.dims == valueCache.desc.dims, "K&V's shape should be same", return false);
        MKI_CHECK(keyCache.desc.dims.size() == DIM_4, "K&V's dim num should be " << DIM_4, return false);

        return true;
    }

    bool CheckSrcBlockList(const LaunchParam &launchParam) const
    {
        auto &srcLocation = launchParam.GetInTensor(DIM_2);
        auto &cumSum = launchParam.GetInTensor(DIM_4);

        MKI_CHECK(!CheckEmptyTensor(srcLocation), "3rd inTensor should not be null tensor ", return false);
        MKI_CHECK(!CheckEmptyTensor(cumSum), "5th inTensor should not be null tensor ", return false);

        MKI_CHECK(srcLocation.desc.dtype == TENSOR_DTYPE_INT32, "src's dtype should be int32", return false);
        MKI_CHECK(cumSum.desc.dtype == TENSOR_DTYPE_INT32, "cumSum's dtype should be int32", return false);

        MKI_CHECK(srcLocation.desc.dims == cumSum.desc.dims, "src&cumSum's shape should be same", return false);
        MKI_CHECK(srcLocation.desc.dims.size() == DIM_1, "src&&cumSum's dim num should be " << DIM_1, return false);

        return true;
    }

    bool CheckDistBlockIndices(const LaunchParam &launchParam) const
    {
        auto &distLocation = launchParam.GetInTensor(DIM_3);

        MKI_CHECK(!CheckEmptyTensor(distLocation), "4th inTensor should not be null tensor ", return false);

        MKI_CHECK(distLocation.desc.dtype == TENSOR_DTYPE_INT32, "dist's dtype should be int32", return false);

        MKI_CHECK(distLocation.desc.dims.size() == DIM_1, "dist's dim num should be " << DIM_1, return false);

        return true;
    }
};

REG_OPERATION(BlockCopyOperation);
} // namespace AtbOps
