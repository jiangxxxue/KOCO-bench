/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_TBE_TILING_RUNNER_KERNEL_H
#define ASCEND_OPS_TBE_TILING_RUNNER_KERNEL_H

#include <memory>

#include <mki/types.h>
#include <mki/kernel_info.h>
#include <mki/bin_handle.h>
#include <mki/utils/status/status.h>
#include <mki/utils/SVector/SVector.h>

namespace AsdOpsGeRt {
using namespace Mki;
class TbeTilingRunnerImpl;
class TbeTilingRunner {
public:
    TbeTilingRunner();
    ~TbeTilingRunner() = default;

    TbeTilingRunner &SetName(const char *opType);
    TbeTilingRunner &SetKernelName(const std::string kernelName);

    TbeTilingRunner &AddInput(TensorDType dtype, TensorFormat format,
                              const SVector<int64_t> &dims);
    TbeTilingRunner &AddConstInput(TensorDType dtype, TensorFormat format,
                                   std::initializer_list<int64_t> dims, const void *data, size_t size);
    TbeTilingRunner &AddConstInput(TensorDType dtype, TensorFormat format,
                                   const SVector<int64_t> &dims, const void *data, size_t size);
    TbeTilingRunner &AddOutput(TensorDType dtype, TensorFormat format,
                               const SVector<int64_t> &dims);

    TbeTilingRunner &AddAttrBool(bool value);
    TbeTilingRunner &AddAttrInt(int32_t attr);
    TbeTilingRunner &AddAttrInt64(int64_t attr);
    TbeTilingRunner &AddAttrFloat(float attr);
    TbeTilingRunner &AddAttrStr(const char *attr);
    TbeTilingRunner &AddAttrIntList(const int64_t *attr, const size_t num);

    Status GetTilingData(uint8_t *tilingData, uint64_t tilingDataLen, const BinHandle &binHandle);

    uint32_t GetBlockDim();
    uint32_t GetIntercoreSync();
    uint64_t GetTilingId();
    uint64_t GetTilingSize();
    void GetWorkSpace(SVector<uint64_t, 8> &workspace); // 8 小容量SVECTOR

private:
    std::shared_ptr<TbeTilingRunnerImpl> impl_;
};
} // namespace AsdOpsGeRt

namespace AsdOps {
using namespace Mki;
Status GetTilingFromRunner(KernelInfo &kernelInfo, AsdOpsGeRt::TbeTilingRunner &runner, const BinHandle &binHandle);
} // namespace AsdOps


#endif