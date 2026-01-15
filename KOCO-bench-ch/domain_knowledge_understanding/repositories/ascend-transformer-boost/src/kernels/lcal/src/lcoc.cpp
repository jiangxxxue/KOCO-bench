/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <lcoc.h>
#include <chrono>
#include <mutex>
#include <map>
#include <thread>
#include <tiling.h>
#include <lcoc_func.h>
#include <lcoc_args.h>
#include <lcoc_workspace.h>
#include <tiling_func.h>
#include "lcal_internal.h"
#include "mki/utils/log/log.h"
#include "mki/utils/env/env.h"
#include "profiling/report_timing.h"
#include "runtime/rt_ffts.h"

using namespace std;
using namespace chrono;
namespace Lcal {
bool CheckLcalComm(const LcalComm *lcalComm)
{
    if (lcalComm == nullptr) {
        MKI_LOG(ERROR) << "The lcalComm is nullptr!";
        return false;
    }

    auto rank = lcalComm->GetRank();
    auto rankSize = lcalComm->GetRankSize();
    auto coreNum = lcalComm->GetPhysicalInfo().coreNum;
    std::vector<std::tuple<std::string, int, int, int>> paramCheckList = {
        {"rankSize", rankSize, PARAM_CHECK_MIN_VALUE_ONE, LCAL_MAX_RANK_SIZE},
        {"rank", rank, PARAM_CHECK_MIN_VALUE_ZERO, rankSize - 1},
        {"coreNum", coreNum, PARAM_CHECK_MIN_VALUE_ONE, PARAM_CHECK_MAX_VALUE},
    };
    return CheckParamScopeList(paramCheckList);
}

bool CheckLcalType(LcalType lcalType)
{
    if (lcalType < LcalType::PURE_MATMUL || lcalType >= LcalType::LCAL_TYPE_MAX) {
        MKI_LOG(ERROR) << "The lcalType:" << int(lcalType)
        << " must be in [" << int(LcalType::PURE_MATMUL) << ", " << int(LcalType::LCAL_TYPE_MAX) << ")!";
        return false;
    }
    return true;
}

bool Check2DTPType(LcalType lcalType)
{
    return lcalType == LcalType::ALL_GATHER_MATMUL_REDUCE_SCATTER;
}

bool CheckMOEType(LcalType lcalType)
{
    return (lcalType >= LcalType::ALLTOALLV_ALLGATHER_MATMUL) &&
           (lcalType <= LcalType::MATMUL_REDUCESCATTER_ALLTOALLVC_HIDDEN);
}

bool CheckCoCParamDesc(LcalType lcalType, const CoCParamDesc &paramDesc)
{
    if (COC_TYPE2ELE_SIZE.find(paramDesc.dataTypeDesc) == COC_TYPE2ELE_SIZE.end()) {
        MKI_LOG(ERROR) << "The dataTypeDesc:" << paramDesc.dataTypeDesc << " is not support yet!";
        return false;
    }
    if (paramDesc.op != HCCL_REDUCE_SUM) {
        MKI_LOG(ERROR) << "The ReduceOp:" << paramDesc.op << " is not support yet!";
        return false;
    }

    auto batchSize = paramDesc.mmInfo.batchSize;
    auto m = paramDesc.mmInfo.m;
    auto n = paramDesc.mmInfo.n;
    auto k = paramDesc.mmInfo.k;
    std::vector<std::tuple<std::string, int, int, int>> paramCheckList = {
        {"batchSize", batchSize, PARAM_CHECK_MIN_VALUE_ONE, PARAM_CHECK_MIN_VALUE_ONE},
        {"m", m, INPUT_PARAM_DEFAULT_VALUE, MAX_M_VALUE},
        {"n", n, PARAM_CHECK_MIN_VALUE_ONE, MAX_N_VALUE},
        {"k", k, PARAM_CHECK_MIN_VALUE_ONE, MAX_K_VALUE},
    };
    if (Check2DTPType(lcalType)) {
        auto agDim = paramDesc.twoDimTPInfo.agDim;
        auto rsDim = paramDesc.twoDimTPInfo.rsDim;
        paramCheckList.emplace_back("agDim", agDim, PARAM_CHECK_MIN_VALUE_ONE, PARAM_CHECK_MAX_VALUE);
        paramCheckList.emplace_back("rsDim", rsDim, PARAM_CHECK_MIN_VALUE_ONE, PARAM_CHECK_MAX_VALUE);
    }
    if (CheckMOEType(lcalType)) {
        auto ep = paramDesc.moeInfo.EP;
        auto tp = paramDesc.moeInfo.TP;
        auto localExpertNums = paramDesc.moeInfo.local_expert_nums;
        paramCheckList.emplace_back("ep", ep, PARAM_CHECK_MIN_VALUE_ONE, PARAM_CHECK_MAX_VALUE);
        paramCheckList.emplace_back("tp", tp, PARAM_CHECK_MIN_VALUE_ONE, PARAM_CHECK_MAX_VALUE);
        paramCheckList.emplace_back("localExpertNums", localExpertNums,
                                    PARAM_CHECK_MIN_VALUE_ONE, PARAM_CHECK_MAX_VALUE);
    }
    return CheckParamScopeList(paramCheckList);
}

bool Lcoc::CheckInputParam(LcalType lcalType, const CoCTiling &tiling, const CoCParamDesc &paramDesc) const
{
    if (!CheckLcalComm(comm_)) {
        return false;
    }
    if (!CheckLcalType(lcalType)) {
        return false;
    }
    if (!CheckCoCTiling(tiling)) {
        return false;
    }
    if (!CheckCoCParamDesc(lcalType, paramDesc)) {
        return false;
    }
    return true;
}

void Lcoc::SetTaskParam(LcalType lcalType, const CoCParamDesc &paramDesc, const LcalComm &comm)
{
    taskParam_.rank = comm.GetRank();
    taskParam_.rankSize = comm.GetRankSize();
    taskParam_.blockDim = comm.GetPhysicalInfo().coreNum;
    taskParam_.chipName = comm.GetPhysicalInfo().chipName;
    taskParam_.cocParamDesc = paramDesc;
    taskParam_.lcalType = lcalType;
    taskParam_.bufferSize = comm.GetBufferSize();
}

void Lcoc::SetLcocParam(LcalType lcalType, const CoCParamDesc &paramDesc)
{
    SetTaskParam(lcalType, paramDesc, *comm_);
    tilingSuccess_ = false;
}

CoCTilingFunc *CreateCoCTilingFunc(LcalType lcalType)
{
    bool isDeterministic = false;
    const char *lcocDeterministic = Mki::GetEnv("LCCL_DETERMINISTIC");
    std::string lcocDeterministicStr = lcocDeterministic == nullptr ? "" : lcocDeterministic;
    if (lcocDeterministicStr == "1" || lcocDeterministicStr == "true") {
        isDeterministic = true;
    }
    CoCTilingFunc *pTilingFunc = nullptr;
    switch (lcalType) {
        case LcalType::ALL_GATHER_MATMUL:
            pTilingFunc = new (std::nothrow) CoCAllGatherMatmulTilingFunc();
            break;
        case LcalType::ALL_GATHER_MATMUL_V2:
            pTilingFunc = new (std::nothrow) CoCAllGatherMatmulV2TilingFunc();
            break;
        case LcalType::MATMUL_REDUCE_SCATTER:
            pTilingFunc = new (std::nothrow) CoCMatmulReduceScatterTilingFunc();
            break;
        case LcalType::MATMUL_ALL_REDUCE:
            if (isDeterministic) {
                pTilingFunc = new (std::nothrow) CoCMatmulAllReduceDeterTilingFunc();
            } else {
                pTilingFunc = new (std::nothrow) CoCMatmulAllReduceTilingFunc();
            }
            break;
        case LcalType::ALL_GATHER_MATMUL_REDUCE_SCATTER:
            pTilingFunc = new (std::nothrow) CoCAllgatherMatmulReduceScatterTilingFunc();
            break;
        case LcalType::ALLTOALLV_ALLGATHER_MATMUL:
            pTilingFunc = new (std::nothrow) CoCAllToAllAllGatherMatmulTilingFunc();
            break;
        case LcalType::ALLTOALLVC_ALLGATHER_MATMUL_HIDDEN:
            pTilingFunc = new (std::nothrow) CoCAllToAllAllGatherMatmulHiddenTilingFunc();
            break;
        case LcalType::MATMUL_REDUCESCATTER_ALLTOALLVC_HIDDEN:
            pTilingFunc = new (std::nothrow) CoCMatmulReduceScatterAllToAllHiddenTilingFunc();
            break;
        default:
            pTilingFunc = new (std::nothrow) CoCTilingFunc();
    }
    return pTilingFunc;
}

Lcoc::~Lcoc() {}

Lcoc::Lcoc(LcalComm *comm) : comm_(comm) {}

Lcoc::Lcoc(LcalComm &comm) : comm_(&comm) {}

int Lcoc::SetParam(LcalType lcalType, const CoCTiling &tiling, const CoCParamDesc &paramDesc)
{
    // 参数检查
    if (!CheckInputParam(lcalType, tiling, paramDesc)) {
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    // 设置LCOC初始化参数
    SetLcocParam(lcalType, paramDesc);
    // 创建Tiling函数
    CoCTilingFunc *pTilingFunc = CreateCoCTilingFunc(lcalType);
    if (pTilingFunc == nullptr) {
        PrintErrorLog(lcalType, "Create CoCTilingFunc failed!");
        return LCAL_ERROR_INTERNAL;
    }
    // 生成Tiling策略参数
    CoCTilingData tilingData = pTilingFunc->GenerateTiling(taskParam_, tiling);
    // 检查Tiling策略参数是否合法
    bool tilingCheckRes = pTilingFunc->CheckTiling(taskParam_);
    if (!tilingCheckRes) {
        PrintErrorLog(lcalType, "Tiling check failed!");
        // 释放TilingFunc
        delete pTilingFunc;
        pTilingFunc = nullptr;
        return LCAL_ERROR_INTERNAL;
    }
    // 赋值Tiling参数
    tiling_ = tilingData;
    // 设置成功标志
    tilingSuccess_ = true;
    // 释放TilingFunc
    delete pTilingFunc;
    pTilingFunc = nullptr;
    return LCAL_SUCCESS;
}

int Lcoc::LaunchOperator(CoCInputPkg &inputPkg, CoCOutputPkg &outputPkg, void *workspace, aclrtStream stream)
{
    CoCKernelArgs args;
    int error = args.SetFFTSAddr();
    if (error != LCAL_SUCCESS) {
        return error;
    }
    auto paramDesc = taskParam_.cocParamDesc;
    args.SetInputPkgArgs(inputPkg);
    args.SetOutputPkgArgs(outputPkg);
    args.SetWorkspacePtrArg(workspace);
    args.SetParamDescArgs(paramDesc);
    args.SetCommArgs(*comm_);
    args.SetCoCTilingDataArgs(tiling_);
    MKI_LOG(DEBUG) << "[" << LCAL_TYPE2NAME.at(taskParam_.lcalType) << "]:" << args.ParamToString();
    return ComputeOverComm(taskParam_.lcalType, args, COC_TYPE2HCCL_TYPE.at(paramDesc.dataTypeDesc), stream);
}

bool Lcoc::CheckBasic(const CoCInputPkg &inputPkg, const CoCOutputPkg &outputPkg, LcalType lcalType) const
{
    (void) outputPkg;
    if (!tilingSuccess_) {
        std::string str = "Tiling error. Please check whether the 'Lcoc::SetParam' method has been called, "
                          "or verify if the tiling parameter is valid.";
        PrintErrorLog(lcalType, str);
        return false;
    }
    if (taskParam_.lcalType != lcalType) {
        std::string str = "lcalType of Lcoc::SetParam doesn't match launch function.";
        PrintErrorLog(lcalType, str);
        return false;
    }
    if (COC_TYPE2HCCL_TYPE.find(taskParam_.cocParamDesc.dataTypeDesc) == COC_TYPE2HCCL_TYPE.end()) {
        std::string str = "invalid dataTypeDesc";
        PrintErrorLog(lcalType, str);
        return false;
    }
    if (inputPkg.matrixA == nullptr || inputPkg.matrixB == nullptr) {
        std::string str = "inputPkg.matrixA or inputPkg.matrixB is nullptr";
        PrintErrorLog(lcalType, str);
        return false;
    }
    return true;
}

int Lcoc::AllGatherMatmul(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace, aclrtStream stream)
{
    LcalType lcalType = LcalType::ALL_GATHER_MATMUL;
    if (!CheckBasic(inputPkg, outputPkg, lcalType)) {
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    ReportTiming report("LcocAllGatherMatmul", true);
    return LaunchOperator(inputPkg, outputPkg, workspace, stream);
}

int Lcoc::AllGatherMatmulV2(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace, aclrtStream stream)
{
    LcalType lcalType = LcalType::ALL_GATHER_MATMUL_V2;
    if (!CheckBasic(inputPkg, outputPkg, lcalType)) {
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    ReportTiming report("LcocAllGatherMatmulV2", true);
    return LaunchOperator(inputPkg, outputPkg, workspace, stream);
}

int Lcoc::MatmulReduceScatter(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace, aclrtStream stream)
{
    LcalType lcalType = LcalType::MATMUL_REDUCE_SCATTER;
    if (!CheckBasic(inputPkg, outputPkg, lcalType)) {
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    if (taskParam_.cocParamDesc.mmInfo.m % taskParam_.rankSize != 0) {
        if (taskParam_.rank == 0) {
            MKI_LOG(ERROR) << "MatmulReduceScatter: input tensor must be the same size as output size times world size";
        }
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    ReportTiming report("LcocMatmulReduceScatter", true);
    return LaunchOperator(inputPkg, outputPkg, workspace, stream);
}

int Lcoc::MatmulAllReduce(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace, aclrtStream stream)
{
    LcalType lcalType = LcalType::MATMUL_ALL_REDUCE;
    if (!CheckBasic(inputPkg, outputPkg, lcalType)) {
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    ReportTiming report("LcocMatmulAllReduce", true);
    return LaunchOperator(inputPkg, outputPkg, workspace, stream);
}

int Lcoc::PureMatmul(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace, aclrtStream stream)
{
    LcalType lcalType = LcalType::PURE_MATMUL;
    if (!CheckBasic(inputPkg, outputPkg, lcalType)) {
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    ReportTiming report("LcocPureMatmul", true);
    return LaunchOperator(inputPkg, outputPkg, workspace, stream);
}

int Lcoc::AllGatherMatmulReduceScatter(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace,
                                       aclrtStream stream)
{
    LcalType lcalType = LcalType::ALL_GATHER_MATMUL_REDUCE_SCATTER;
    if (!CheckBasic(inputPkg, outputPkg, lcalType)) {
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    ReportTiming report("LcocAllGatherMatmulReduceScatter", true);
    return LaunchOperator(inputPkg, outputPkg, workspace, stream);
}

int Lcoc::AllToAllVAllGatherMatmul(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace,
    aclrtStream stream)
{
    LcalType lcalType = LcalType::ALLTOALLV_ALLGATHER_MATMUL;
    if (!CheckBasic(inputPkg, outputPkg, lcalType)) {
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    return LaunchOperator(inputPkg, outputPkg, workspace, stream);
}

int Lcoc::MatmulReduceScatterAllToAllVHidden(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace,
    aclrtStream stream)
{
    LcalType lcalType = LcalType::MATMUL_REDUCESCATTER_ALLTOALLVC_HIDDEN;
    if (!CheckBasic(inputPkg, outputPkg, lcalType)) {
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    return LaunchOperator(inputPkg, outputPkg, workspace, stream);
}
 
int Lcoc::AllToAllVAllGatherMatmulHidden(CoCInputPkg inputPkg, CoCOutputPkg outputPkg, void *workspace,
    aclrtStream stream)
{
    LcalType lcalType = LcalType::ALLTOALLVC_ALLGATHER_MATMUL_HIDDEN;
    if (!CheckBasic(inputPkg, outputPkg, lcalType)) {
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    return LaunchOperator(inputPkg, outputPkg, workspace, stream);
}
LcalComm *Lcoc::GetComm()
{
    return comm_;
}

MatMulInfo &Lcoc::GetMatMulInfo()
{
    return taskParam_.cocParamDesc.mmInfo;
}

void Lcoc::GetTiling(CoCTiling &tiling)
{
    tiling = tiling_;
}


bool IsMatrixAligned(const int64_t &m, const int64_t &n, const bool &transpose, int nElemAlign)
{
    if (nElemAlign == 0) {
        return false;
    }
    return (transpose ? m : n) % nElemAlign == 0;
}

int64_t Lcoc::GetWorkspaceSize()
{
    LcalType lcalType = taskParam_.lcalType;
    auto cocParamDesc = taskParam_.cocParamDesc;
    bool isDeterministic = (GetComm()->GetCommArgs()->extraFlag & ExtraFlag::DETERMINISTIC) != 0;
    CoCDataTypeDesc dataType = cocParamDesc.dataTypeDesc;
    const MatMulInfo &mmInfo = cocParamDesc.mmInfo;
    const QuantInfo &quantInfo = cocParamDesc.quantInfo;
    const MoeInfo& moeInfo = cocParamDesc.moeInfo;
    bool hasQuant = quantInfo.quantGranularity != QuantGranularity::QUANT_GRANULARITY_UNDEFINED;
    bool hasDequant = quantInfo.dequantGranularity != QuantGranularity::QUANT_GRANULARITY_UNDEFINED;
    int32_t eleSize = COC_TYPE2ELE_SIZE.at(dataType);
    int32_t nElemAlign = Lcal::ALIGN_BYTES / eleSize;
    int32_t mAlign = AlignUp(mmInfo.m, nElemAlign);
    int32_t nAlign = AlignUp(mmInfo.n, nElemAlign);
    int32_t kAlign = AlignUp(mmInfo.k, nElemAlign);
    int32_t maxOutputSize = moeInfo.maxOutputSize;

    bool hasAAlign = hasQuant || (!IsMatrixAligned(mmInfo.m, mmInfo.k, mmInfo.transA, nElemAlign) && mmInfo.m != 1);

    bool hasBAlign = (!mmInfo.weightNz) && ((hasDequant && !mmInfo.isInt8)
                     || (!IsMatrixAligned(mmInfo.k, mmInfo.n, mmInfo.transB, nElemAlign)));

    int32_t accumRankSize = taskParam_.lcalType == LcalType::ALL_GATHER_MATMUL ? taskParam_.rankSize : 0;

    bool hasAccum = dataType == CoCDataTypeDesc::INT8INT8_INT32_BF16;
    bool hasDequantParam = (quantInfo.dequantGranularity == QuantGranularity::PER_TOKEN ||
                            quantInfo.dequantGranularity == QuantGranularity::PER_TENSOR);
    bool hasFormatDequantScale = (quantInfo.dequantGranularity == QuantGranularity::PER_CHANNEL);
    bool isMoe = false;
    if (lcalType == LcalType::ALLTOALLV_ALLGATHER_MATMUL ||
        lcalType == LcalType::ALLTOALLVC_ALLGATHER_MATMUL_HIDDEN ||
        lcalType == LcalType::MATMUL_REDUCESCATTER_ALLTOALLVC_HIDDEN) {
            isMoe = true;
    }
    bool isAlltoallVc =
        lcalType == LcalType::ALLTOALLV_ALLGATHER_MATMUL || lcalType == LcalType::ALLTOALLVC_ALLGATHER_MATMUL_HIDDEN ||
        lcalType == LcalType::MATMUL_REDUCESCATTER_ALLTOALLVC_HIDDEN;

    uint64_t dequantWorkSpaceSize = GetDequantWorkSpaceSize(lcalType, tiling_.withSerialMode, mmInfo.m, mmInfo.n,
        tiling_.m0, tiling_.n0, tiling_.pValue, tiling_.nLoop, taskParam_.rankSize, taskParam_.blockDim, maxOutputSize);
    LcalWorkspaceInfo lcalWorkspaceInfo = GetLcalWorkspaceInfo(0, mmInfo.batchSize, mmInfo.m, mmInfo.k,
        mmInfo.n, mAlign, kAlign, nAlign, mmInfo.transA, mmInfo.transB, eleSize, hasAAlign, hasBAlign,
        accumRankSize, hasAccum, dequantWorkSpaceSize, hasDequantParam, hasFormatDequantScale, isDeterministic,
        isMoe, isAlltoallVc, moeInfo.EP, moeInfo.local_expert_nums, maxOutputSize);

    MKI_LOG(DEBUG) << "[Lcoc Workspace]: " << "m=" << mmInfo.m << ", k=" << mmInfo.k << ", n=" << mmInfo.n
        << ", mAlign=" << mAlign << ", kAlign=" << kAlign << ", nAlign=" << nAlign << ", transA=" << mmInfo.transA
        << ", transB=" << mmInfo.transB << ", eleSize=" << eleSize << ", hasAAlign=" << hasAAlign
        << ", hasBAlign=" << hasBAlign << ", accumRankSize=" << accumRankSize << ", hasAccum=" << hasAccum
        << ", dequantWorkSpaceSize=" << dequantWorkSpaceSize << ", hasDequantParam=" << hasDequantParam
        << ", hasFormatDequantScale=" << hasFormatDequantScale << ", isDeterministic=" << isDeterministic
        << ", isMoe=" << isMoe << ", isAlltoallVc=" << isAlltoallVc << ", moeInfo.EP=" << static_cast<int>(moeInfo.EP)
        << ", moeInfo.local_expert_nums=" << moeInfo.local_expert_nums
        << ", maxOutputSize=" << maxOutputSize << ", workspaceSize=" << lcalWorkspaceInfo.workspaceSize;
    return lcalWorkspaceInfo.workspaceSize;
}
}
