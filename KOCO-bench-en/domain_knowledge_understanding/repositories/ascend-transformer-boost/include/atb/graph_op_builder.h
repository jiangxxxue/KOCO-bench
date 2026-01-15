/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_GRAPH_OP_H
#define ATB_GRAPH_OP_H
#include <string>
#include <memory>
#include "atb/operation.h"

//!
//! \file graph_op_builder.h
//!
//! \brief 定义图算子的构建方法
//!

//!
//! \namespace atb
//!
//! \brief 加速库的命名空间.
//!
namespace atb {

//!
//! \class GraphOpBuilder.
//!
//! \brief 图算子创建类，主要用于简化图算子创建过程.
//!
//! GraphOpBuilder类会通过Operation的输入输出关系组建出算子的拓扑图，并最终完成GraphOp的创建，优化了之前手动定义tensor id的组图方式.
//!
class GraphOpBuilder {
public:
    //! \brief 构造函数.
    GraphOpBuilder();

    //! \brief 析构函数.
    virtual ~GraphOpBuilder();

    //!
    //! \brief 初始化图算子.
    //!
    //! 定义图算子的名称，输入输出及shape传导规则.
    //!
    //! \param opName 图算子的名称
    //!
    //! \param inferShapeFunc 图算子的shape传导规则
    //!
    //! \param inTensorNames 输入tensor名称
    //!
    //! \param outTensorNames 输出tensor名称
    //!
    //! \return 状态值.如果设置成功，返回NO_ERROR.
    //!
    virtual Status Init(const std::string &opName, const InferShapeFunc &inferShapeFunc,
                        const SVector<std::string> &inTensorNames, const SVector<std::string> &outTensorNames) = 0;

    //!
    //! \brief 改变输入tensor的shape.
    //!
    //! \param srcTensorName 输入tensor的名称
    //!
    //! \param reshapeFunc shape修改规则
    //!
    //! \param viewTensorName shape修改后的tensor名称
    //!
    //! \return 状态值.如果设置成功，返回NO_ERROR.
    //!
    virtual Status Reshape(const std::string &srcTensorName, const ReshapeFunc &reshapeFunc,
                           const std::string &viewTensorName) = 0;

    //!
    //! \brief 向图中添加算子.
    //!
    //! \param operation 要添加的算子
    //!
    //! \param inTensorNames 被添加算子的输入tensor名称
    //!
    //! \param outTensorNames 被添加算子的输出tensor名称
    //!
    //! \return 状态值.如果设置成功，返回NO_ERROR.
    //!
    virtual Status AddOperation(Operation *operation, const SVector<std::string> &inTensorNames,
                                const SVector<std::string> &outTensorNames) = 0;

    //!
    //! \brief 创建图算子.
    //!
    //! \return 返回被创建的图算子，失败返回空指针.
    //!
    virtual Operation *Build() = 0;

    //!
    //! \brief 创建并向图中添加算子.
    //!
    //! \param opParam 要添加的算子参数
    //!
    //! \param inTensorNames 被添加算子的输入tensor名称
    //!
    //! \param outTensorNames 被添加算子的输出tensor名称
    //!
    //! \return 状态值.如果设置成功，返回NO_ERROR.
    //!
    template <class OpParam>
    Status AddOperation(const OpParam &opParam, const SVector<std::string> &inTensorNames,
                        const SVector<std::string> &outTensorNames)
    {
        Operation *operation = nullptr;
        Status st = CreateOperation(opParam, &operation);
        if (st != NO_ERROR) {
            return st;
        }
        st =  AddOperation(operation, inTensorNames, outTensorNames);
        if (st != NO_ERROR) {
            if (operation != nullptr) {
                DestroyOperation(operation);
            }
        }
        return st;
    }
};

//!
//! \brief 创建图算子构建器.
//!
//! \param builder 返回被创建的构建器
//!
//! \return 状态值.如果设置成功，返回NO_ERROR.
//!
Status CreateGraphOpBuilder(GraphOpBuilder **builder);

//!
//! \brief 销毁图算子构建器.
//!
//! \param builder 待销毁的构建器
//!
//! \return 状态值.如果设置成功，返回NO_ERROR.
//!
Status DestroyGraphOpBuilder(GraphOpBuilder *builder);
} // namespace atb
#endif