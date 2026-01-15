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
#include <mki/utils/log/log.h>
#include <mki/utils/checktensor/check_tensor.h>
#include "asdops/params/params.h"
#include "kernels/norm/common/common_tiling_data.h"
#include "kernels/norm/postrmsnorm/tiling/post_rms_norm_tiling.h"

namespace {
static const float THRESHOLD = 2e-38;
} // namespace

namespace AsdOps {
class PreRmsNormKernel : public KernelBase {
public:
    explicit PreRmsNormKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(RmsNormCommonTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return PostRmsNormTiling(launchParam, kernelInfo_);
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Norm), "norm: param type invalid",
                     return false);
        AsdOps::OpParam::Norm param = AnyCast<OpParam::Norm>(launchParam.GetParam());
        AsdOps::OpParam::Norm::NormType type = param.normType;
        MKI_CHECK(type == AsdOps::OpParam::Norm::RMS_NORM, "rmsnorm: param type invalid", return false);
        MKI_CHECK(param.epsilon >= THRESHOLD, "epsilon invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 4, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 2, "output num invalid", return false);
        const Tensor &tensorX = launchParam.GetInTensor(TENSOR_X_IDX);
        const Tensor &tensorBias = launchParam.GetInTensor(TENSOR_BIAS_IDX);
        const Tensor &tensorResIn = launchParam.GetInTensor(TENSOR_RES_IN_IDX);
        const Tensor &tensorGamma = launchParam.GetInTensor(TENSOR_GAMMA_IDX);
        const Tensor &tensorY = launchParam.GetOutTensor(TENSOR_Y_IDX);
        const Tensor &tensorResOut = launchParam.GetOutTensor(TENSOR_RES_OUT_IDX);
        MKI_CHECK(tensorX.desc.format == TENSOR_FORMAT_ND, "tensorX format invalid", return false);
        MKI_CHECK(tensorX.desc.dtype == TENSOR_DTYPE_FLOAT16 || tensorX.desc.dtype == TENSOR_DTYPE_BF16,
                  "tensorX dtype invalid", return false);
        MKI_CHECK(tensorResIn.desc.format == tensorX.desc.format, "tensorResIn format invalid", return false);
        MKI_CHECK(tensorResIn.desc.dtype == tensorX.desc.dtype, "tensorResIn dtype invalid", return false);
        MKI_CHECK(tensorGamma.desc.format == tensorX.desc.format, "tensorGamma format invalid", return false);
        MKI_CHECK(tensorGamma.desc.dtype == tensorX.desc.dtype, "tensorGamma dtype invalid", return false);
        MKI_CHECK(tensorY.desc.format == tensorX.desc.format, "tensorY format invalid", return false);
        MKI_CHECK(tensorY.desc.dtype == tensorX.desc.dtype, "tensorY dtype invalid", return false);
        MKI_CHECK(tensorResOut.desc.format == tensorX.desc.format, "tensorResOut format invalid", return false);
        MKI_CHECK(tensorResOut.desc.dtype == tensorX.desc.dtype, "tensorResOut dtype invalid", return false);
        const SVector<int64_t> &xDims = tensorX.desc.dims;
        const SVector<int64_t> &resInDims = tensorResIn.desc.dims;
        const SVector<int64_t> &gmmaDims = tensorGamma.desc.dims;
        if (xDims.empty() || gmmaDims.empty() || resInDims.size() != xDims.size()) {
            MKI_LOG(ERROR) << "inDims is empty or inDims size is invalid!"
                           << " xDimsSize : " << xDims.size() << ", resInDims : " << resInDims.size();
            return false;
        }
        size_t xLastIdx = xDims.size() - 1;
        size_t gLastIdx = gmmaDims.size() - 1;
        size_t resInLastIdx = resInDims.size() - 1;
        if (xDims[xLastIdx] != resInDims[resInLastIdx] || xDims[xLastIdx] != gmmaDims[gLastIdx]) {
            MKI_LOG(ERROR) << "last dimension not the same!" << " xDims[xLastIdx] : " << xDims[xLastIdx]
                           << " resInDims[lastDim] : " << resInDims[resInLastIdx]
                           << " gmmaDims[lastDim] : " << gmmaDims[gLastIdx];
            return false;
        }
        if (!CheckEmptyTensor(tensorBias)) {
            const SVector<int64_t> &biasDims = tensorBias.desc.dims;
            MKI_CHECK(tensorBias.desc.dtype == tensorX.desc.dtype, "in tensorBias dtype invalid", return false);
            MKI_CHECK(tensorBias.desc.format == tensorX.desc.format, "tensorBias format invalid", return false);
            MKI_CHECK(biasDims[biasDims.size() - 1] == xDims[xLastIdx], "the last dimension should be the same",
                         return false);
        }
        return true;
    }
};
REG_KERNEL_BASE(PreRmsNormKernel);
} // namespace AsdOps