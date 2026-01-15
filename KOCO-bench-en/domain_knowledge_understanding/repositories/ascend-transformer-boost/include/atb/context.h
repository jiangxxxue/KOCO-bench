/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_CONTEXT_H
#define ATB_CONTEXT_H
#include <functional>
#include <acl/acl.h>
#include "atb/types.h"

//!
//! \file context.h
//!
//! \brief 定义加速库上下文类
//!

//!
//! \namespace atb
//!
//! \brief 加速库的命名空间.
//!
namespace atb {

//!
//! \enum ExecuteType
//!
//! \brief 下发接口形式枚举，通过Context选择加速库算子下发接口的形式, 支持单段下发和使用分线程两段式下发.
//!
enum ExecuteType : int {
    EXECUTE_NORMAL = 0, //!< 直接下发
    EXECUTE_PRELAUNCH,  //!< 用于分线程下发，第一段下发
    EXECUTE_LAUNCH,     //!< 用于分线程下发，第二段下发
};

//!
//! \enum LaunchMode
//!
//! \brief 算子下发模式枚举，通过Context选择算子下发的模式，支持单算子下发与整图下发
//!
enum LaunchMode : int {
    KERNEL_LAUNCH_MODE = 0, //!< 单算子下发模式
    GRAPH_LAUNCH_MODE       //!< 整图下发模式
};

//!
//! \class Context.
//!
//! \brief 加速库上下文类，主要用于管理Operation运行所需要的全局资源.
//!
//! Context类会管理任务流队列比如Operation执行以及TilingCopy,管理tiling内存的申请与释放.
//!
class Context {
public:
    //! \brief 默认构造函数.
    Context() = default;

    //! \brief 默认析构函数.
    virtual ~Context() = default;

    //!
    //! \brief 将传入stream队列设置为当前执行队列.
    //!
    //! 将传入stream队列设置为当前执行队列,然后再去执行对应的Operation.
    //!
    //! \param stream 传入的stream队列
    //!
    //! \return 状态值.如果设置成功，返回NO_ERROR.
    //!
    virtual Status SetExecuteStream(aclrtStream stream) = 0;

    //!
    //! \brief 获取当前执行stream队列.
    //!
    //! \return 执行流队列
    //!
    virtual aclrtStream GetExecuteStream() const = 0;

    //!
    //! \brief 设置异步拷贝tiling信息功能.
    //!
    //! 设置异步拷贝tiling信息功能是否开启，如果是，则创建stream和event来进行tiling拷贝过程.
    //!
    //! \param enable 传入的标志，bool类型
    //!
    //! \return 状态值.如果设置成功，返回NO_ERROR.
    //!
    virtual Status SetAsyncTilingCopyStatus(bool enable) = 0;

    //!
    //! \brief 获取tiling拷贝状态.
    //!
    //! \return 如果获取成功，返回True.
    //!
    virtual bool GetAsyncTilingCopyStatus() const = 0;

    //!
    //! \brief 设置实际的执行流，Operation执行时会根据配置的streamId从Context匹配对应的实际执行流
    //!
    //! \param streams 需要设置的一组stream
    //!
    //! \return 状态值，如果设置成功，返回NO_ERROR
    virtual Status SetExecuteStreams(const std::vector<aclrtStream> &streams) = 0;

    //!
    //! \brief 获取Context中当前设置的一组执行流
    //!
    //! \return Context当前设置的一组执行流
    virtual std::vector<aclrtStream> GetExecuteStreams() = 0;

    //!
    //! \brief 设置Execute的类型
    //!
    //! \param type ExecuteType类型
    //!
    //! \return 状态值，如果设置成功，返回NO_ERROR
    virtual Status SetExecuteType(ExecuteType type) = 0;

    //!
    //! \brief 获取当前context Execute的类型
    //!
    //! \return 获取到的ExecuteType类型
    virtual ExecuteType GetExecuteType() = 0;

    //!
    //! \brief 设置算子下发模式
    //!
    //! \param mode 算子下发的模式类型
    //!
    //! \return 状态值，如果设置成功，返回NO_ERROR
    virtual Status SetLaunchMode(LaunchMode mode) = 0;

    //!
    //! \brief 返回当前的算子下发模式
    //!
    //! \return 当前的算子下发模式
    virtual LaunchMode GetLaunchMode() = 0;
};

//!
//! \brief 创建上下文.
//!
//! 在当前进程或线程中显式创建一个Context.
//!
//! \param context 传入的context
//!
//! \return 状态值.如果设置成功，返回NO_ERROR.
//!
Status CreateContext(Context **context);

//!
//! \brief 创建上下文.
//!
//! 在当前进程或线程中显式创建一个由用户管理Tiling内存的Context.
//!
//! \param context 传入的context
//！
//! \param alloc 传入的Tiling内存分配方法
//!
//! \param dealloc 传入的Tiling内存释放方法
//!
//! \return 状态值.如果设置成功，返回NO_ERROR.
//!
Status CreateContext(Context **context, const std::function<void*(size_t)>& alloc, const std::function<void(void*)>& dealloc);

//!
//! \brief 销毁上下文.
//!
//! 销毁上下文中所有的资源.
//!
//! \param context 传入的context
//!
//! \return 状态值.如果设置成功，返回NO_ERROR.
//!
Status DestroyContext(Context *context);
} // namespace atb
#endif