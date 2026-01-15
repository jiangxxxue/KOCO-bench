/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
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
#include "include/customizeblockcopy.h"


namespace AtbOps {
using namespace Mki;
/**
 * @class CustomizeBlockCopyOperation
 * @brief Mki 层的自定义 CustomizeBlockCopy 算子，实现参数检查、形状推理及 Kernel 选择。
 */
class CustomizeBlockCopyOperation : public OperationBase {
public:
    explicit CustomizeBlockCopyOperation(const std::string &opName) noexcept : OperationBase(opName) {}
    /**
     * @brief 返回算子期望的输入张量数量。
     * @param[in] specificParam  包含 op 参数的 Any 对象，应为 OpParam::CustomizeBlockCopy 类型。
     * @return int64_t          固定返回 5；检查失败时返回 0。
     */
    int64_t GetInputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::CustomizeBlockCopy), "OpParam is invalid", return 0);
        return DIM_5;
    }
    /**
     * @brief 返回算子期望的输出张量数量。
     * @param[in] specificParam  包含 op 参数的 Any 对象，应为 OpParam::CustomizeBlockCopy 类型。
     * @return int64_t          固定返回 2（in-place 返回 K/V）；检查失败时返回 0。
     */
    int64_t GetOutputNum(const Any &specificParam) const override
    {
        MKI_CHECK(specificParam.Type() == typeid(OpParam::CustomizeBlockCopy), "OpParam is invalid", return 0);
        return DIM_2;
    }
    /**
     * @brief 对 LaunchParam 中的参数和输入张量进行完整性检查。
     *
     * 依次校验：
     *  - Param 类型是否为 OpParam::CustomizeBlockCopy
     *  - 输入张量数量是否为 5
     *  - K/V Cache、src 索引、dst 索引列表格式是否合法
     *
     * @param[in] launchParam  包含运行时所有输入张量和参数的上下文
     * @return bool            全部检查通过返回 true，否则 false
     */
    bool CheckBlockCopy(const LaunchParam &launchParam) const
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::CustomizeBlockCopy), "OpParam is invalid",
                  return false);

        MKI_CHECK(launchParam.GetInTensorCount() == 5, "input num invalid", return false);

        return CheckKVCache(launchParam) && CheckSrcBlockList(launchParam) && CheckDistBlockIndices(launchParam);
    }
    /**
     * @brief 在推理阶段设置输出张量形状，同输入的 K/V Cache 保持一致。
     *
     * 1. 调用 CheckBlockCopy 检查参数合法性
     * 2. 将 inTensors[0]、inTensors[1] 直接赋值给 outTensors[0]、outTensors[1]
     *
     * @param[in]  launchParam   运行时输入张量及参数
     * @param[out] outTensors    填充后的输出张量列表
     * @return Status            成功返回 OkStatus；失败返回带错误码的 FailStatus
     */
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(CheckBlockCopy(launchParam), "Failed to check launch param",
                  return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));

        auto &keyCacheIn = launchParam.GetInTensor(DIM_0);
        auto &valueCacheIn = launchParam.GetInTensor(DIM_1);
        outTensors[DIM_0] = keyCacheIn;
        outTensors[DIM_1] = valueCacheIn;
        return Status::OkStatus();
    }
    /**
     * @brief 选择最合适的 Kernel 实例执行本操作。
     *
     * 1. 调用 IsConsistent 检查参数合法性和形状一致性
     * 2. 检查 Param 类型
     * 3. 返回名为 "CustomizeBlockCopyKernel" 的 Kernel 对象
     *
     * @param[in] launchParam  运行时输入张量及参数
     * @return Kernel*         成功返回指向 Kernel 的指针；失败返回 nullptr
     */
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(IsConsistent(launchParam), "Failed to check consistent", return nullptr);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::CustomizeBlockCopy), "OpParam is invalid",
                  return nullptr);
        return GetKernelByName("CustomizeBlockCopyKernel");
    }

private:
    /**
     * @brief 检查 K/V Cache 张量的合法性。
     *
     * - 非空且 dtype 相同
     * - dtype 为 float16、bfloat16 或 int8
     * - 维度数量为 4，且 shape 一致
     *
     * @param[in] launchParam  运行时上下文
     * @return bool            合法返回 true，否则 false
     */
    bool CheckKVCache(const LaunchParam &launchParam) const
    {
        auto &keyCache = launchParam.GetInTensor(DIM_0);   // K.shape = [block_count, block_size, num_heads, head_size]
        auto &valueCache = launchParam.GetInTensor(DIM_1); // V.shape = [block_count, block_size, num_heads, head_size]

        MKI_CHECK(!CheckEmptyTensor(keyCache), "1st inTensor should not be null tensor ", return false);
        MKI_CHECK(!CheckEmptyTensor(valueCache), "2nd inTensor should not be null tensor ", return false);

        auto inDtype = keyCache.desc.dtype;
        MKI_CHECK(valueCache.desc.dtype == inDtype, "K&V's dtype must be same", return false);
        MKI_CHECK(inDtype == TENSOR_DTYPE_FLOAT16 || inDtype == TENSOR_DTYPE_BF16 || inDtype == TENSOR_DTYPE_INT8,
                  "K&V should be both float16 OR int8 OR bfloat16", return false);

        MKI_CHECK(keyCache.desc.dims == valueCache.desc.dims, "K&V's shape should be same", return false);
        MKI_CHECK(keyCache.desc.dims.size() == DIM_4, "K&V's dim num should be " << DIM_4, return false);

        return true;
    }
    /**
     * @brief 检查 src block 列表及 cumSum 张量的合法性。
     *
     * - 非空
     * - dtype 为 int32
     * - shape/dims 与 cumSum 匹配
     *
     * @param[in] launchParam  运行时上下文
     * @return bool            合法返回 true，否则 false
     */
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
    /**
     * @brief 检查 dst block 列表张量的合法性。
     *
     * - 非空
     * - dtype 为 int32
     * - 维度数为 1
     *
     * @param[in] launchParam  运行时上下文
     * @return bool            合法返回 true，否则 false
     */
    bool CheckDistBlockIndices(const LaunchParam &launchParam) const
    {
        auto &distLocation = launchParam.GetInTensor(DIM_3);

        MKI_CHECK(!CheckEmptyTensor(distLocation), "4th inTensor should not be null tensor ", return false);

        MKI_CHECK(distLocation.desc.dtype == TENSOR_DTYPE_INT32, "dist's dtype should be int32", return false);

        MKI_CHECK(distLocation.desc.dims.size() == DIM_1, "dist's dim num should be " << DIM_1, return false);

        return true;
    }
};

REG_OPERATION(CustomizeBlockCopyOperation); // Operation注册
} // namespace AtbOps
