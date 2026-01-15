/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "asdops/params/scatter_elements_v2.h"

#include <mki/base/kernel_base.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/platform/platform_info.h>
#include <mki_loader/op_register.h>

#include "kernels/scatter_elements_v2/tiling/scatter_elements_v2_tiling.h"

namespace AsdOps {
using namespace Mki;

static constexpr char NONE[] = "none";
static constexpr char ADD[] = "add";
template <TensorDType INPUT_TYPE, TensorDType INDICE_TYPE, const char *REDUCTION>
class ScatterElementsV2Kernel : public KernelBase {
public:
    explicit ScatterElementsV2Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        // 检查参数类型是否为 ScatterElemens
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::ScatterElementsV2),
                  "ScatterElements: param type invalid", return false);
        MKI_CHECK(PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B,
                  "ScatterElements operator only supported ASCEND_910B platform", return false);
        // 检查输入张量数量是否为 3（input, indices, updates）
        MKI_CHECK(launchParam.GetInTensorCount() == 3, "input num invalid", return false);
        // 检查输出张量数量是否为 1（output）
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        // 调用 ScatterElementsV2 的通用分片函数
        return ScatterElementsV2CommonTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

// ScatterElementsV2Kernel
using ScatterElementsV2Int32Int32NoneKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_INT32, TENSOR_DTYPE_INT32, NONE>;
using ScatterElementsV2Int32Int32AddKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_INT32, TENSOR_DTYPE_INT32, ADD>;
using ScatterElementsV2Int32Int64NoneKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_INT32, TENSOR_DTYPE_INT64, NONE>;
using ScatterElementsV2Int32Int64AddKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_INT32, TENSOR_DTYPE_INT64, ADD>;

using ScatterElementsV2Bfloat16Int32NoneKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_BF16, TENSOR_DTYPE_INT32, NONE>;
using ScatterElementsV2Bfloat16Int32AddKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_BF16, TENSOR_DTYPE_INT32, ADD>;
using ScatterElementsV2Bfloat16Int64NoneKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_BF16, TENSOR_DTYPE_INT64, NONE>;
using ScatterElementsV2Bfloat16Int64AddKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_BF16, TENSOR_DTYPE_INT64, ADD>;

using ScatterElementsV2Uint8Int32NoneKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_UINT8, TENSOR_DTYPE_INT32, NONE>;
using ScatterElementsV2Uint8Int32AddKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_UINT8, TENSOR_DTYPE_INT32, ADD>;
using ScatterElementsV2Uint8Int64NoneKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_UINT8, TENSOR_DTYPE_INT64, NONE>;
using ScatterElementsV2Uint8Int64AddKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_UINT8, TENSOR_DTYPE_INT64, ADD>;

using ScatterElementsV2Int8Int32NoneKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_INT8, TENSOR_DTYPE_INT32, NONE>;
using ScatterElementsV2Int8Int32AddKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_INT8, TENSOR_DTYPE_INT32, ADD>;
using ScatterElementsV2Int8Int64NoneKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_INT8, TENSOR_DTYPE_INT64, NONE>;
using ScatterElementsV2Int8Int64AddKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_INT8, TENSOR_DTYPE_INT64, ADD>;

using ScatterElementsV2Float32Int32NoneKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_FLOAT, TENSOR_DTYPE_INT32, NONE>;
using ScatterElementsV2Float32Int32AddKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_FLOAT, TENSOR_DTYPE_INT32, ADD>;
using ScatterElementsV2Float32Int64NoneKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_FLOAT, TENSOR_DTYPE_INT64, NONE>;
using ScatterElementsV2Float32Int64AddKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_FLOAT, TENSOR_DTYPE_INT64, ADD>;

using ScatterElementsV2Float16Int32NoneKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_FLOAT16, TENSOR_DTYPE_INT32, NONE>;
using ScatterElementsV2Float16Int32AddKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_FLOAT16, TENSOR_DTYPE_INT32, ADD>;
using ScatterElementsV2Float16Int64NoneKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_FLOAT16, TENSOR_DTYPE_INT64, NONE>;
using ScatterElementsV2Float16Int64AddKernel = ScatterElementsV2Kernel<TENSOR_DTYPE_FLOAT16, TENSOR_DTYPE_INT64, ADD>;

REG_KERNEL_BASE(ScatterElementsV2Int32Int32NoneKernel);
REG_KERNEL_BASE(ScatterElementsV2Int32Int32AddKernel);
REG_KERNEL_BASE(ScatterElementsV2Int32Int64NoneKernel);
REG_KERNEL_BASE(ScatterElementsV2Int32Int64AddKernel);
REG_KERNEL_BASE(ScatterElementsV2Bfloat16Int32NoneKernel);
REG_KERNEL_BASE(ScatterElementsV2Bfloat16Int32AddKernel);
REG_KERNEL_BASE(ScatterElementsV2Bfloat16Int64NoneKernel);
REG_KERNEL_BASE(ScatterElementsV2Bfloat16Int64AddKernel);
REG_KERNEL_BASE(ScatterElementsV2Uint8Int32NoneKernel);
REG_KERNEL_BASE(ScatterElementsV2Uint8Int32AddKernel);
REG_KERNEL_BASE(ScatterElementsV2Uint8Int64NoneKernel);
REG_KERNEL_BASE(ScatterElementsV2Uint8Int64AddKernel);
REG_KERNEL_BASE(ScatterElementsV2Int8Int32NoneKernel);
REG_KERNEL_BASE(ScatterElementsV2Int8Int32AddKernel);
REG_KERNEL_BASE(ScatterElementsV2Int8Int64NoneKernel);
REG_KERNEL_BASE(ScatterElementsV2Int8Int64AddKernel);
REG_KERNEL_BASE(ScatterElementsV2Float32Int32NoneKernel);
REG_KERNEL_BASE(ScatterElementsV2Float32Int32AddKernel);
REG_KERNEL_BASE(ScatterElementsV2Float32Int64NoneKernel);
REG_KERNEL_BASE(ScatterElementsV2Float32Int64AddKernel);
REG_KERNEL_BASE(ScatterElementsV2Float16Int32NoneKernel);
REG_KERNEL_BASE(ScatterElementsV2Float16Int32AddKernel);
REG_KERNEL_BASE(ScatterElementsV2Float16Int64NoneKernel);
REG_KERNEL_BASE(ScatterElementsV2Float16Int64AddKernel);
} // namespace AsdOps