/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_COMM_H
#define ATB_COMM_H
#include "atb/types.h"
//!
//! \file comm.h
//!
//! \brief 定义加速库通信域相关接口
//!

//!
//! \namespace atb
//!
//! \brief 加速库的命名空间.
//!
namespace atb {
//! \brief 通信域指针
//!
using HcclComm = void*;
//!
//! \namespace Comm
//!
//! \brief 通信域相关接口的命名空间
//!
namespace Comm {
//!
//! \brief 创建HCCL通信域
//!
//! \param[in] rank 该进程的rank
//! \param[in] rankRoot 主卡,默认是0
//! \param[in] rankSize 卡数
//! \param[out] commName 通信域名称
//!
//! \return 返回通信域指针
//!
HcclComm CreateHcclComm(int32_t rank, int32_t rankRoot, int32_t rankSize, char *commName);

//!
//! \brief 通过rankTableFile创建HCCL通信域
//! \param[in] rank 该进程的rank
//! \param[in] rankTableFile 通信域配置文件
//! \param[in] rankSize 卡数
//! \param[out] commName 通信域名称
//!
//! \return 返回通信域指针
//!
HcclComm CreateHcclCommByRankTableFile(int32_t rank, int32_t rankSize, const char *rankTableFile,
                                       char *commName);

//!
//! \brief 创建HCCL多机通信域
//! \param[in] rankTableFile 通信域配置文件
//! \param[in] subCommRankId 本rank在子通信域中的rank id
//! \param[in] rankIds 子通信域中rank在全局通信域中的rank id组成的数组
//! \param[in] subCommId 当前子通信域标识
//! \param[in] hcclBufferSize 子通信域共享数据的缓存区大小
//! \param[out] commName 通信域名称
//!
//! \return 返回通信域指针
//!
HcclComm CreateHcclCrossMulitComm(const char *rankTableFile, uint32_t subCommRankId, std::vector<uint32_t> &rankIds,
                                  uint64_t subCommId, uint32_t hcclBufferSize, char *commName);

//!
//! \brief 销毁指定的HCCL通信域
//!
//! \param comm 传入通信域指针
//!
//! \return  状态值.如果销毁成功，返回NO_ERROR
//!
Status DestoryHcclComm(HcclComm comm);
}; // namespace Comm
} // namespace atb
#endif