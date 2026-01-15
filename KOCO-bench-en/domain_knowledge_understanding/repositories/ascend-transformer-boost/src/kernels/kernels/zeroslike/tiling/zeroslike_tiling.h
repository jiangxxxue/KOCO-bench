/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendOpCommonLib is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifndef ASCEND_OPS_ZEROS_LIKE_TILING_H
#define ASCEND_OPS_ZEROS_LIKE_TILING_H

#include "mki/bin_handle.h"
#include "mki/kernel_info.h"
#include "mki/launch_param.h"

namespace AsdOps {
Mki::Status ZerosLikeTiling(const std::string &kernelName, const Mki::LaunchParam &launchParam,
                            Mki::KernelInfo &kernelInfo, const Mki::BinHandle &binHandle);
} // namespace AsdOps
#endif // ASCEND_OPS_ZEROSLIKE_TILING_H
