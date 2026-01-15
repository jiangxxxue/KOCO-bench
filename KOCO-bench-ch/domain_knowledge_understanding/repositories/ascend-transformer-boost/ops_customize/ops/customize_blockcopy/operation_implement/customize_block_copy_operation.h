/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file customize_block_copy_operation.h
 * @brief 定义 CustomizeBlockCopyOperation 类，实现 ATB 高层 API 的形状检查、
 *        参数校验以及 Runner 创建逻辑。
 */
#ifndef CUSTOMIZE_BLOCK_COPY_BLOCK_COPYOPERATION_H
#define CUSTOMIZE_BLOCK_COPY_BLOCK_COPYOPERATION_H
#include "customize_blockcopy/kernel_implement/include/customizeblockcopy.h"
#include "atb/operation/operation_base.h"
#include "customize_op_params.h"

namespace atb {
/**
 * @class CustomizeBlockCopyOperation
 * @brief ATB 高层封装的 CustomizeBlockCopy 操作，将输入/输出张量描述检查、
 *        参数合法性检查和 Runner 创建流程串联起来。
 */
class CustomizeBlockCopyOperation : public OperationBase {
public:
    explicit CustomizeBlockCopyOperation(const customize::BlockCopyParam &param);
    ~CustomizeBlockCopyOperation() override;
    /**
     * @brief 获取期望的输入张量数量（固定为 5: KCache、VCache、srcIdx、dstIdx、cumSum）。
     * @return uint32_t  输入张量数量
     */
    uint32_t GetInputNum() const override;
    /**
     * @brief 获取期望的输出张量数量（in-place，不返回额外张量）。
     * @return uint32_t  输出张量数量，固定为 0
     */
    uint32_t GetOutputNum() const override;
    /**
     * @brief 对输入 TensorDesc 列表进行完整的维度和形状检查。
     *
     * - K/V Cache 必须同形并且 dimNum = 4
     * - srcIdx/dstIdx 的长度 ≤ blockCount
     * - cumSum 与 srcIdx 形状相同
     *
     * @param[in] inTensorDescs  输入 TensorDesc 列表
     * @return Status            检查通过返回 NO_ERROR，否则返回错误码
     */
    Status InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const override;
    /**
     * @brief 对运行时的输入 Tensor 进行类型、大小等检查。
     *
     * @param[in] inTensors   输入 Tensor 列表
     * @param[out] outTensors  输出 Tensor 列表（in-place，此处忽略）
     * @return Status         检查通过返回 NO_ERROR，否则返回错误码
     */
    Status SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const override;

protected:
    /**
     * @brief 推理阶段设置输出 TensorDesc，本操作 in-place，不修改输出。
     *
     * @param[in] inTensorDescs   输入 TensorDesc 列表
     * @param[out] outTensorDescs 输出 TensorDesc 列表
     * @return Status             始终返回 NO_ERROR
     */
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    /**
     * @brief 创建对应的 OpsRunner 实例，用于后续执行阶段调度。
     * @param[in] context  ATB 执行上下文，包含 Stream 等信息
     * @return std::shared_ptr<Runner>  实例化的 CustomizeBlockCopyOpsRunner
     */
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;

private:
    customize::BlockCopyParam param_;
};
} // namespace atb
#endif