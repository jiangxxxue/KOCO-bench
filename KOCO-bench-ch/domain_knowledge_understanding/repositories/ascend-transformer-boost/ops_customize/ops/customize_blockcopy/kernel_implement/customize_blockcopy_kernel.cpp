/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/base/kernel_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include "include/customizeblockcopy.h"
#include "mixkernels/utils/common.h"
#include "tiling/customize_blockcopy_tiling.h"
#include "tiling/tiling_data.h"

namespace AtbOps {
using namespace Mki;
static constexpr uint32_t TENSOR_INPUT_NUM = 5;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 2;
/**
 * @class CustomizeBlockCopyKernel
 * @brief 封装具体的 kernel 实现，负责调用 tiling 生成与 kernel 初始化。
 */
class CustomizeBlockCopyKernel : public KernelBase {
public:
    explicit CustomizeBlockCopyKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
        MKI_LOG(INFO) << "CustomizeBlockCopyKernel construct ";
    }
    /**
     * @brief 判断该 kernel 是否支持当前的 launchParam。
     * @param[in] launchParam    运行时输入信息
     * @return bool              支持则返回 true，否则 false
     */
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "in tensor num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "out tensor num invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::CustomizeBlockCopy),
                  "CustomizeBlockCopy: param type invalid", return false);
        return true;
    }
    /**
     * @brief 返回 tilingData 在 device 端的字节大小，用于分配 buffer。
     * @param[in] launchParam    运行时输入信息
     * @return uint64_t          tilingData 大小
     */
    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        return sizeof(CustomizeBlockCopyTilingData);
    }
    /**
     * @brief 根据 launchParam 调用 CustomizeBlockCopyTiling 并填充 kernelInfo_。
     * @param[in] launchParam    运行时输入信息
     * @return Status            返回状态
     */
    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = CustomizeBlockCopyTiling(launchParam, kernelInfo_);
        MKI_CHECK_NO_LOG(status.Ok(), return status);

        return Status::OkStatus();
    }
};
REG_KERNEL_BASE(CustomizeBlockCopyKernel); // Kernel注册
} // namespace AtbOps