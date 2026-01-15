/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_STOREUTIL_H
#define ATB_STOREUTIL_H

#include <string>
#include <mki/tensor.h>
#include <mki/utils/bin_file/bin_file.h>
#include <mki/launch_param.h>
#include "atb/types.h"
#include "atb/utils/runner_variant_pack.h"

namespace atb {
class StoreUtil {
public:
    static void WriteFile(const uint8_t *data, uint64_t dataLen, const std::string &filePath);
    static void SaveVariantPack(aclrtStream stream, const VariantPack &variantPack, const std::string &dirPath);
    static void SaveVariantPack(aclrtStream stream, const RunnerVariantPack &runnerVariantPack,
                                const std::string &dirPath);
    static void SaveLaunchParam(aclrtStream stream, const Mki::LaunchParam &launchParam, const std::string &dirPath);

private:
    static void CopyTensorData(Mki::BinFile &binFile, const void *deviceData, uint64_t dataSize);
    static void SaveTensor(const Mki::Tensor &tensor, const std::string &filePath) __attribute__((weak));
    static void SaveTensor(const Tensor &tensor, const std::string &filePath) __attribute__((weak));
    static void SaveTensor(const std::string &format, const std::string &dtype, const std::string &dims,
                           const void *deviceData, uint64_t dataSize, const std::string &filePath);
};
} // namespace atb
#endif