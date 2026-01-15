/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCAL_COC_KERNEL_ARGS_H
#define LCAL_COC_KERNEL_ARGS_H

#include <string>
#include "tiling_args.h"
#include "lcal_comm.h"
#include "lcoc_args.h"

namespace Lcal {
struct CoCKernelArgs {
    void *matrixA = nullptr;
    void *matrixB = nullptr;
    void *bias = nullptr;
    void *gamma = nullptr;
    void *output = nullptr;
    void *midOutput = nullptr;
    void *workspace = nullptr;
    void *dequantScale = nullptr;
    void *dequantOffset = nullptr;
    void *quantScale = nullptr;
    void *quantOffset = nullptr;
    void *commArgsPtr = nullptr;
    uint64_t fftsAddr = 0;

    void *numLocalTokensPerExpertPtr = nullptr;
    void *numGlobalTokensPerLocalExpertPtr = nullptr;
    void *globalTokensPerLocalExpertMatrixPtr = nullptr;
    CoCTilingData *pCocTiling = nullptr;
    CoCKernelParam cocKernelParam = {};
    int SetFFTSAddr();
    void SetInputPkgArgs(CoCInputPkg &inputPkg);
    void SetOutputPkgArgs(CoCOutputPkg &outputPkg);
    void SetWorkspacePtrArg(void *workspacePtr);
    void SetParamDescArgs(const CoCParamDesc &paramDesc);
    void SetCommArgs(const LcalComm &comm);
    void SetCoCTilingDataArgs(const CoCTilingData &tilingData);
    std::string ParamToString();
};

}

#endif // LCAL_COC_KERNEL_ARGS_H
