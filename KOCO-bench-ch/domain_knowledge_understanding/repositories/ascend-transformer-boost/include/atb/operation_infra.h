/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_OPERATION_INFRA_H
#define ATB_OPERATION_INFRA_H
#include <memory>
#include "atb/operation.h"
#include "atb/context.h"

//!
//! \file operation_infra.h
//!
//! \brief 定义加速库OperationInfra类
//!

namespace atb {
class OperationImpl;

//!
//! \class OperationInfra.
//!
//! \brief 加速库OperationInfra类.
//!
//! 该接口类定义了加速库插件算子的基类，基类提供了一些内置公共接口供用户编写自己的插件算子
//!
class OperationInfra : public Operation {
public:
    //! \brief 构造函数.
    OperationInfra();

    //! \brief 拷贝构造函数.
    //!
    //! \param other 拷贝的对象
    OperationInfra(const OperationInfra &other);

    //! \brief 赋值构造函数.
    //!
    //! \param other 赋值的对象

    //! \return 对象本身
    OperationInfra& operator = (const OperationInfra &other);

    //! \brief 析构函数.
    ~OperationInfra() override;

    //! \brief 设置该Operation需要使用的stream的id，id为context中的stream序号，该接口需要配合GetExecuteStream接口使用。
    //!
    //! \param streamId 需要设置的stream的id
    //!
    void SetExecuteStreamId(uint32_t streamId);

    //!
    //! \brief 获取当前Operation使用的stream的id
    //!
    //! \return stream的id，id为context中的stream序号
    //!
    uint32_t GetExecuteStreamId() const;

    //!
    //! \brief 获取当前Operation使用的stream
    //!
    //! \param context Operation使用的Context
    //!
    //! \return 当前Operation使用的stream
    //!
    aclrtStream GetExecuteStream(Context *context);

private:
    std::unique_ptr<OperationImpl> impl_;
};
}
#endif