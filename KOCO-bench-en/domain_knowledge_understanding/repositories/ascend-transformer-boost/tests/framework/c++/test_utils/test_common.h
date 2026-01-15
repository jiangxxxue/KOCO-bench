/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ATB_UNIT_TEST_COMMON_H
#define ATB_UNIT_TEST_COMMON_H

#include <ATen/ATen.h>
#include <string>
#include <mki/utils/SVector/SVector.h>
#include <fstream>
#include <regex>
#include "atb/utils/log.h"

constexpr int TIME = 50;

at::IntArrayRef ToIntArrayRef(const Mki::SVector<int64_t> &src);
at::IntArrayRef ToIntArrayRef(const std::vector<int64_t> &src);
int64_t Prod(const Mki::SVector<int64_t> &vec);
void LogMemUsage();

#endif