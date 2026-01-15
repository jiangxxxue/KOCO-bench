/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifdef __DAV_C220_VEC__

#include "lccl_op.h"

LCCL_TYPE_AIV_FUNC(LCCL_ALLGATHER_FUNC_AUTO_DEF);
LCCL_TYPE_AIV_FUNC(LCCL_REDUCE_SCATTER_FUNC_AUTO_DEF);
LCCL_TYPE_AIV_FUNC(LCCL_ALL2ALL_FUNC_AUTO_DEF);

#ifdef ENABLE_LCCL_MIX
LCCL_BROADCAST_FUNC_AUTO_DEF(_mix_aiv)
#else
LCCL_BROADCAST_FUNC_AUTO_DEF()
#endif
#endif
#ifdef __DAV_C220_CUBE__

#include "lccl_op.h"
LCCL_TYPE_AIC_FUNC(LCCL_ALLGATHER_FUNC_AUTO_DEF);
LCCL_TYPE_AIC_FUNC(LCCL_REDUCE_SCATTER_FUNC_AUTO_DEF);
LCCL_TYPE_AIC_FUNC(LCCL_ALL2ALL_FUNC_AUTO_DEF);
#ifdef ENABLE_LCCL_MIX
LCCL_BROADCAST_FUNC_AUTO_DEF(_mix_aic)
#else
LCCL_BROADCAST_FUNC_AUTO_DEF()
#endif
#endif