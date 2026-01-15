/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/base/kernel_base.h>
#include <mki_loader/op_register.h>
#include <mki/types.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include "kernels/elewise/simple_broadcast/tiling/simple_broadcast_tiling.h"

namespace AsdOps {
using namespace Mki;
class BroadcastKernel : public KernelBase {
public:
    explicit BroadcastKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() > 1, "in tensor num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "out tensor num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(BroadcastTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        BroadcastInfo broadcastInfo;
        FillBroadCastInfoImpl(launchParam, broadcastInfo);
        return SimplyBroadcastTiling(broadcastInfo, launchParam, kernelInfo_);
    }

protected:
    virtual void FillBroadCastInfoImpl(const LaunchParam &launchParam, BroadcastInfo &broadcastInfo) const = 0;
};

class AtbQuantPerChannelKernel : public BroadcastKernel {
public:
    explicit AtbQuantPerChannelKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : BroadcastKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(BroadcastKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 3, "in tensor num is invalid", return false);
        TensorDType dtype = launchParam.GetInTensor(TENSOR_DST_IDX).desc.dtype;
        MKI_CHECK(dtype == TENSOR_DTYPE_FLOAT16 || dtype == TENSOR_DTYPE_BF16,
            "in tensor dtype is invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(TENSOR_SCALE_IDX).desc.dtype == dtype,
            "in tensor dtype is invalid", return false);
        const Tensor &tensor = launchParam.GetInTensor(TENSOR_OFFSET_IDX);
        if (!CheckEmptyTensor(tensor)) {
            MKI_CHECK(tensor.desc.dtype == TENSOR_DTYPE_INT8, "in tensor2 dtype invalid", return false);
        }
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT8,
            "out tensor dtype invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        BroadcastInfo broadcastInfo;
        FillBroadCastInfoImpl(launchParam, broadcastInfo);
        return QuantPerChannelTiling(broadcastInfo, launchParam, kernelInfo_);
    }

protected:
    void FillBroadCastInfoImpl(const LaunchParam &launchParam, BroadcastInfo &broadcastInfo) const override
    {
        TensorDType inTensorDtype = launchParam.GetInTensor(TENSOR_DST_IDX).desc.dtype;
        TensorDType outTensorDtype = launchParam.GetOutTensor(TENSOR_DST_IDX).desc.dtype;
        broadcastInfo.broadcastBytesPerElem = GetTensorElementSize(inTensorDtype) +
                                              GetTensorElementSize(outTensorDtype);\
        broadcastInfo.broadcastBytesPerElem *= 2; // 2：常开double buffer

        TensorDType scaleDtype = launchParam.GetInTensor(1).desc.dtype;
        TensorDType offsetDtype = TENSOR_DTYPE_INT8;
        broadcastInfo.normalBytesPerElem = GetTensorElementSize(scaleDtype) +
                                           GetTensorElementSize(offsetDtype) +
                                           2 * sizeof(TENSOR_DTYPE_FLOAT16); // 2：两个中间缓存buffer

        // BF16计算会先转换成FP32，需要额外的UB空间
        if (inTensorDtype == TENSOR_DTYPE_BF16) {
            broadcastInfo.broadcastBytesPerElem += sizeof(float); // x->fp32
            broadcastInfo.normalBytesPerElem += 2 * sizeof(float); // scale&offset -> fp32
        }
    }
};
REG_KERNEL_BASE(AtbQuantPerChannelKernel);

class AtbDequantPerChannelKernel : public BroadcastKernel {
public:
    explicit AtbDequantPerChannelKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : BroadcastKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(BroadcastKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 3, "in tensor num is invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(TENSOR_DST_IDX).desc.dtype == TENSOR_DTYPE_INT8,
            "in tensor dtype is invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(TENSOR_SCALE_IDX).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "in tensor dtype is invalid", return false);
        const Tensor &tensor = launchParam.GetInTensor(TENSOR_OFFSET_IDX);
        if (!CheckEmptyTensor(tensor)) {
            MKI_CHECK(tensor.desc.dtype == TENSOR_DTYPE_INT8, "in tensor2 dtype invalid", return false);
        }
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "out tensor dtype invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        BroadcastInfo broadcastInfo;
        FillBroadCastInfoImpl(launchParam, broadcastInfo);
        return DequantPerChannelTiling(broadcastInfo, launchParam, kernelInfo_);
    }

protected:
    void FillBroadCastInfoImpl(const LaunchParam &launchParam, BroadcastInfo &broadcastInfo) const override
    {
        TensorDType inTensorDtype = launchParam.GetInTensor(TENSOR_DST_IDX).desc.dtype;
        TensorDType outTensorDtype = launchParam.GetOutTensor(TENSOR_DST_IDX).desc.dtype;
        broadcastInfo.broadcastBytesPerElem = GetTensorElementSize(inTensorDtype) +
                                              GetTensorElementSize(outTensorDtype);
        broadcastInfo.broadcastBytesPerElem *= 2; // 2：常开double buffer

        TensorDType scaleDtype = launchParam.GetInTensor(1).desc.dtype;
        TensorDType offsetDtype = TENSOR_DTYPE_INT8;
        broadcastInfo.normalBytesPerElem = GetTensorElementSize(scaleDtype) +
                                           GetTensorElementSize(offsetDtype) +
                                           2 * sizeof(TENSOR_DTYPE_FLOAT16); // 2：两个中间缓存buffer
    }
};
REG_KERNEL_BASE(AtbDequantPerChannelKernel);
} // namespace AsdOps