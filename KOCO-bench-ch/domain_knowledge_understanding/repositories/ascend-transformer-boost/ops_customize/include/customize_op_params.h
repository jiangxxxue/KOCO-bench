/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CUSTOMIZE_OPPARAM_H
#define CUSTOMIZE_OPPARAM_H
#include <cstdint>
#include <string>
#include <limits>
#include <hccl/hccl_types.h>
#include <acl/acl.h>
#include "atb/svector.h"

//!
//! \file customize_op_params.h
//!
//! \brief 定义加速库所有用户自定义算子参数
//!

//!
//! \namespace atb
//!
//! \brief 加速库命名空间.
//!
namespace atb {

namespace customize {

//!
//! \struct BlockCopyParam (用户自定义目录的示例算子)
//!
//! \brief 将KVCache里通过src indices指定的block数据copy到dst indices指定的block位置上。
//!
struct BlockCopyParam {
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[16] = {0};
};
} // namespace customize
} // namespace atb
#endif