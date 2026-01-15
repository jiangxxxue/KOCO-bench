/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coc_kernel_args.h"
#include <sstream>
#include <string>
#include <mki/utils/log/log.h>
#include <mki/utils/rt/other/other.h>
#include "tiling.h"
#include "lcal_internal.h"
using namespace Mki;

namespace Lcal {
int CoCKernelArgs::SetFFTSAddr()
{
    uint32_t fftsLen;
    int error = MkiRtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    if (error != MKIRT_SUCCESS) {
        MKI_LOG(ERROR) << "MkiRtGetC2cCtrlAddr err";
        return LCAL_ERROR_MKIRT;
    }
    return LCAL_SUCCESS;
}

void CoCKernelArgs::SetInputPkgArgs(CoCInputPkg &inputPkg)
{
    matrixA = inputPkg.matrixA;
    matrixB = inputPkg.matrixB;
    bias = inputPkg.bias;
    gamma = inputPkg.gamma;
    dequantScale = inputPkg.dequantScale;
    dequantOffset = inputPkg.dequantOffset;
    quantScale = inputPkg.quantScale;
    quantOffset = inputPkg.quantOffset;
    numLocalTokensPerExpertPtr = inputPkg.num_local_tokens_per_expert;
    numGlobalTokensPerLocalExpertPtr = inputPkg.num_global_tokens_per_local_expert;
    globalTokensPerLocalExpertMatrixPtr = inputPkg.global_tokens_per_expert_matrix;
}

void CoCKernelArgs::SetOutputPkgArgs(CoCOutputPkg &outputPkg)
{
    output = outputPkg.output;
    midOutput = outputPkg.midOutput;
}

void CoCKernelArgs::SetWorkspacePtrArg(void *workspacePtr)
{
    workspace = workspacePtr;
}

void CoCKernelArgs::SetParamDescArgs(const CoCParamDesc &paramDesc)
{
    cocKernelParam.quantInfo = paramDesc.quantInfo;
    cocKernelParam.twoDimTPInfo = paramDesc.twoDimTPInfo;
    cocKernelParam.postInfo = paramDesc.postInfo;
    cocKernelParam.weightNz = paramDesc.mmInfo.weightNz;
    cocKernelParam.moeInfo = paramDesc.moeInfo;
}

void CoCKernelArgs::SetCommArgs(const LcalComm &comm)
{
    commArgsPtr = comm.GetCommArgsPtr();
}

void CoCKernelArgs::SetCoCTilingDataArgs(const CoCTilingData &tilingData)
{
    pCocTiling = &(cocKernelParam.cocTilingData);
    cocKernelParam.cocTilingData = tilingData;
}

std::string CoCKernelArgs::ParamToString()
{
    std::string quantInfoString = "[QuantInfo]: dequantGranularity=" +
                                  std::to_string(cocKernelParam.quantInfo.dequantGranularity) + "\n";
    auto moeInfo = cocKernelParam.moeInfo;
    std::string moeInfoString =
           std::string("[MoeInfo]: local_expert_nums=") + std::to_string(moeInfo.local_expert_nums) +
           ", EP=" + std::to_string(static_cast<int>(moeInfo.EP)) +
           ", TP=" + std::to_string(static_cast<int>(moeInfo.TP))  +
           ", maxOutputSize=" + std::to_string(moeInfo.maxOutputSize) +
           ", isMoe=" + std::to_string(static_cast<int>(moeInfo.isMoe)) + "\n";
    std::string weightNzInfoString = "[weightNz]: weightNz=" +
                                      std::to_string(cocKernelParam.weightNz) + "\n";
    std::string tilingInfoString = cocKernelParam.cocTilingData.ToString();
    return quantInfoString + moeInfoString + weightNzInfoString + tilingInfoString;
}
}
