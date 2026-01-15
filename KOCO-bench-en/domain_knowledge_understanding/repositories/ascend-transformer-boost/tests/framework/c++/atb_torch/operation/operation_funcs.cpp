/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "operation_funcs.h"
#include <nlohmann/json.hpp>
#include <functional>
#include "atb/utils/log.h"
#include "atb/infer_op_params.h"
#include "atb/train_op_params.h"
#include "chatglm6b/layer/chatglm6blayer_encoder_operation.h"
#include "chatglm6b/operation/chatglm6b_lmhead_operation.h"
#include "llama65b/layer/llama65b_layer_operation.h"
#include "common_graph/common_graph_operation.h"
#include "view_graph/view_graph_operation.h"
#include "llama65b/layer/llama65b_layer_mlp.h"
#include "post_process/post_process_opration.h"
#include "llama65b/layer/llama65b_layer_mlp_graph_builder.h"

using OperationCreateFunc = std::function<atb::Status(const nlohmann::json &paramJson, atb::Operation **op)>;

static atb::Status AllReduceOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::AllReduceParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.find("allReduceType") != paramJson.end()) {
        param.allReduceType = paramJson["allReduceType"].get<std::string>();
    }
    if (paramJson.contains("commMode")) {
        param.commMode = atb::infer::CommMode(paramJson["commMode"].get<int>());
    }
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.find("commDomain") != paramJson.end()) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    if (paramJson.contains("quantType")) {
        param.quantType = atb::infer::AllReduceParam::QuantType(paramJson["quantType"].get<int>());
    }
    if (paramJson.contains("outDataType")) {
        param.outDataType = aclDataType(paramJson["outDataType"].get<int32_t>());
    }
    ATB_LOG(INFO) << "AllReduceParam rank:" << param.rank << ", rankSize:" << param.rankSize
                  << ", rankRoot:" << param.rankRoot << ", backend:" << param.backend
                  << ", commDomain:" << param.commDomain << ", quantType:" << param.quantType
                  << ", outDataType:" << param.outDataType;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status BroadcastOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::BroadcastParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.contains("commMode")) {
        param.commMode = atb::infer::CommMode(paramJson["commMode"].get<int>());
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.find("commDomain") != paramJson.end()) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    ATB_LOG(INFO) << "BroadcastParam rank:" << param.rank << ", rankSize:" << param.rankSize
                  << ", rankRoot:" << param.rankRoot << ", backend:" << param.backend
                  << ", commDomain:" << param.commDomain;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status ReduceScatterOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::ReduceScatterParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.contains("commMode")) {
        param.commMode = atb::infer::CommMode(paramJson["commMode"].get<int>());
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.find("reduceType") != paramJson.end()) {
        param.reduceType = paramJson["reduceType"].get<std::string>();
    }
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.find("commDomain") != paramJson.end()) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    ATB_LOG(INFO) << "ReduceScatterParam rank:" << param.rank << ", rankSize:" << param.rankSize
                  << ", rankRoot:" << param.rankRoot << ", backend:" << param.backend
                  << ", reduceType:" << param.reduceType << ", commDomain:" << param.commDomain;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status ReduceScatterVOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::ReduceScatterVParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("sendCounts") != paramJson.end()) {
        for (auto item : paramJson["sendCounts"]) {
            param.sendCounts.push_back(item.get<int64_t>());
        }
    }
    if (paramJson.find("sdispls") != paramJson.end()) {
        for (auto item : paramJson["sdispls"]) {
            param.sdispls.push_back(item.get<int64_t>());
        }
    }
    if (paramJson.find("recvCount") != paramJson.end()) {
        param.recvCount = paramJson["recvCount"].get<int64_t>();
    }
    if (paramJson.find("reduceType") != paramJson.end()) {
        param.reduceType = paramJson["reduceType"].get<std::string>();
    }
    if (paramJson.contains("commMode")) {
        param.commMode = atb::infer::CommMode(paramJson["commMode"].get<int>());
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.find("commDomain") != paramJson.end()) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    ATB_LOG(INFO) << "ReduceScatterVParam rank:" << param.rank << ", rankSize:" << param.rankSize
                  << ", rankRoot:" << param.rankRoot << ", backend:" << param.backend
                  << ", reduceType:" << param.reduceType << ", commDomain:" << param.commDomain;
    return CreateOperation(param, op);
}

static atb::Status AllGatherOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::AllGatherParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.contains("commMode")) {
        param.commMode = atb::infer::CommMode(paramJson["commMode"].get<int>());
    }
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.find("commDomain") != paramJson.end()) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    ATB_LOG(INFO) << "AllGatherParam rank:" << param.rank << ", rankSize:" << param.rankSize
                  << ", rankRoot:" << param.rankRoot << ", backend:" << param.backend
                  << ", commDomain:" << param.commDomain;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status AllGatherVOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::AllGatherVParam param;
    if (paramJson.find("rank") != paramJson.end()) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.find("rankSize") != paramJson.end()) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.contains("commMode")) {
        param.commMode = atb::infer::CommMode(paramJson["commMode"].get<int>());
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.find("commDomain") != paramJson.end()) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    ATB_LOG(INFO) << "AllGatherParam rank:" << param.rank << ", rankSize:" << param.rankSize
                  << ", rankRoot:" << param.rankRoot << ", backend:" << param.backend
                  << ", commDomain:" << param.commDomain;
    return CreateOperation(param, op);
}

static atb::Status BlockCopyOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::BlockCopyParam param;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status DynamicNTKOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::DynamicNTKParam param;

    if (paramJson.contains("outputType")) {
        param.outDataType = aclDataType(paramJson["outputType"].get<int32_t>());
    }

    ATB_LOG(INFO) << "DynamicNTKParam  outputType:" << param.outDataType;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status GroupTopkOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::GroupTopkParam param;

    if (paramJson.contains("groupNum")) {
        param.groupNum = paramJson["groupNum"].get<int32_t>();
    }

    if (paramJson.contains("k")) {
        param.k = paramJson["k"].get<int32_t>();
    }
    if (paramJson.contains("groupMultiFlag")) {
        param.groupMultiFlag = atb::infer::GroupTopkParam::GroupMultiFlag(paramJson["groupMultiFlag"].get<uint16_t>());
    }

    if (paramJson.contains("n")) {
        param.n = paramJson["n"].get<uint16_t>();
    }

    ATB_LOG(INFO) << "GroupTopkParam groupNum:" << param.groupNum << ", k:" << param.k
                  << ", groupMultiFlag:" << param.groupMultiFlag << ", n:" << param.n;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status LinearParallelOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::LinearParallelParam param;
    if (paramJson.find("transWeight") != paramJson.end()) {
        param.transWeight = paramJson["transWeight"].get<bool>();
    }
    if (paramJson.find("hasResidual") != paramJson.end()) {
        param.hasResidual = paramJson["hasResidual"].get<bool>();
    }
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.contains("commMode")) {
        param.commMode = atb::infer::CommMode(paramJson["commMode"].get<int>());
    }
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.contains("type")) {
        param.type = atb::infer::LinearParallelParam::ParallelType(paramJson["type"].get<int>());
    }
    if (paramJson.contains("keepIntermediate")) {
        param.keepIntermediate = paramJson["keepIntermediate"].get<bool>();
    }
    if (paramJson.contains("quantType")) {
        param.quantType = atb::infer::LinearParallelParam::QuantType(paramJson["quantType"].get<int>());
    }
    if (paramJson.contains("quantGroupSize")) {
        param.quantGroupSize = paramJson["quantGroupSize"].get<int32_t>();
    }
    if (paramJson.contains("outDataType")) {
        param.outDataType = aclDataType(paramJson["outDataType"].get<int32_t>());
    }
    if (paramJson.find("commDomain") != paramJson.end()) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    if (paramJson.find("twoDimTPInfo") != paramJson.end()) {
        const auto &twoDimTPInfo = paramJson["twoDimTPInfo"];
        if (twoDimTPInfo.contains("agDim")) {
            param.twoDimTPInfo.agDim = twoDimTPInfo["agDim"].get<uint16_t>();
        }
        if (twoDimTPInfo.contains("rsDim")) {
            param.twoDimTPInfo.rsDim = twoDimTPInfo["rsDim"].get<uint16_t>();
        }
        if (twoDimTPInfo.contains("innerDimIsAg")) {
            param.twoDimTPInfo.innerDimIsAg = twoDimTPInfo["innerDimIsAg"].get<uint8_t>();
        }
    }
    if (paramJson.find("moeInfo") != paramJson.end()) {
        const auto &moeInfo = paramJson["moeInfo"];
        if (moeInfo.contains("localExpertNums")) {
            param.moeInfo.localExpertNums = moeInfo["localExpertNums"].get<int16_t>();
        }
        if (moeInfo.contains("epSize")) {
            param.moeInfo.epSize = static_cast<int8_t>(moeInfo["epSize"].get<int16_t>());
        }
        if (moeInfo.contains("tpSize")) {
            param.moeInfo.tpSize = static_cast<int8_t>(moeInfo["tpSize"].get<int16_t>());
        }
    }
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    ATB_LOG(INFO) << "LinearParallelParam  rank:" << param.rank << ", rankSize:" << param.rankSize
                  << ", rankRoot:" << param.rankRoot << ", backend:" << param.backend
                  << ", transWeight:" << param.transWeight << ", commMode:" << param.commMode << ", type:" << param.type
                  << ", keepIntermediate:" << param.keepIntermediate << ", quantType:" << param.quantType
                  << ", quantGroupSize:" << param.quantGroupSize << ", commDomain:" << param.commDomain
                  << ", twoDimTPInfo.agDim:" << param.twoDimTPInfo.agDim
                  << ", twoDimTPInfo.rsDim:" << param.twoDimTPInfo.rsDim
                  << ", twoDimTPInfo.innerDimIsAg:" << param.twoDimTPInfo.innerDimIsAg
                  << ", moeInfo.localExpertNums:" << param.moeInfo.localExpertNums
                  << ", moeInfo.epSize:" << param.moeInfo.epSize
                  << ", moeInfo.tpSize:" << param.moeInfo.tpSize;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status CumsumOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::CumsumParam param;
    for (auto item : paramJson["axes"]) {
        param.axes.push_back(item.get<int64_t>());
        ATB_LOG(FATAL) << "axes:" << param.axes.at(0);
    }
    if (paramJson.contains("exclusive")) {
        param.exclusive = paramJson["exclusive"].get<bool>();
    }
    if (paramJson.contains("reverse")) {
        param.reverse = paramJson["reverse"].get<bool>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status ConcatOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::ConcatParam param;
    ATB_LOG(INFO) << "ConcatParam axis:" << param.concatDim;
    if (paramJson.contains("concatDim")) {
        param.concatDim = paramJson["concatDim"].get<int>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status SplitOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::SplitParam param;
    if (paramJson.contains("splitDim")) {
        param.splitDim = paramJson["splitDim"].get<int>();
    }
    if (paramJson.contains("splitNum")) {
        param.splitNum = paramJson["splitNum"].get<int>();
    }
    if (paramJson.contains("splitSizes")) {
        for (auto item : paramJson["splitSizes"]) {
            param.splitSizes.push_back(item.get<int32_t>());
        }
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status GatherOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::GatherParam param;
    ATB_LOG(INFO) << "GatherParam axis:" << param.axis;
    ATB_LOG(INFO) << "GatherParam batchDims:" << param.batchDims;
    if (paramJson.contains("axis")) {
        param.axis = paramJson["axis"].get<int64_t>();
    }
    if (paramJson.contains("batchDims")) {
        param.batchDims = paramJson["batchDims"].get<int64_t>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}


static atb::Status GroupedMatmulInplaceAddOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::GroupedMatmulInplaceAddParam param;
    if (paramJson.contains("transposeA")) {
        param.transposeA = paramJson["transposeA"].get<bool>();
    }
    if (paramJson.contains("transposeB")) {
        param.transposeB = paramJson["transposeB"].get<bool>();
    }
    ATB_LOG(INFO) << "GroupedMatmulInplaceAddParam  transposeA:" << param.transposeA
                  << ", transposeB:" << param.transposeB;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status LinearOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::LinearParam param;
    if (paramJson.contains("transposeA")) {
        param.transposeA = paramJson["transposeA"].get<bool>();
    }
    if (paramJson.contains("transposeB")) {
        param.transposeB = paramJson["transposeB"].get<bool>();
    }
    if (paramJson.contains("hasBias")) {
        param.hasBias = paramJson["hasBias"].get<bool>();
    }
    if (paramJson.contains("outDataType")) {
        param.outDataType = aclDataType(paramJson["outDataType"].get<int32_t>());
    }
    if (paramJson.contains("enAccum")) {
        param.enAccum = paramJson["enAccum"].get<bool>();
    }
    if (paramJson.contains("matmulType")) {
        param.matmulType = atb::infer::LinearParam::MatmulType(paramJson["matmulType"].get<int>());
    }
    if (paramJson.contains("quantMode")) {
        param.quantMode = atb::infer::LinearParam::QuantMode(paramJson["quantMode"].get<int>());
    }
    ATB_LOG(INFO) << "LinearParam  transposeA:" << param.transposeA << ", transposeB:" << param.transposeB
                  << ", hasBias:" << param.hasBias << ", outDataType:" << param.outDataType
                  << ", enAccum: " << param.enAccum << ", matmulType: " << param.matmulType
                  << ", quantMode" << param.quantMode;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status SelfAttentionOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::SelfAttentionParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("qScale")) {
        param.qScale = paramJson["qScale"].get<float>();
    }
    if (paramJson.contains("qkScale")) {
        param.qkScale = paramJson["qkScale"].get<float>();
    }
    if (paramJson.contains("kvHeadNum")) {
        param.kvHeadNum = paramJson["kvHeadNum"].get<int>();
    }
    if (paramJson.contains("maskType")) {
        param.maskType = atb::infer::SelfAttentionParam::MaskType(paramJson["maskType"].get<int32_t>());
    }
    if (paramJson.contains("calcType")) {
        param.calcType = atb::infer::SelfAttentionParam::CalcType(paramJson["calcType"].get<int32_t>());
    }
    if (paramJson.contains("clampType")) {
        param.clampType = atb::infer::SelfAttentionParam::ClampType(paramJson["clampType"].get<int32_t>());
    }
    if (paramJson.contains("kernelType")) {
        param.kernelType = atb::infer::SelfAttentionParam::KernelType(paramJson["kernelType"].get<int32_t>());
    }
    if (paramJson.contains("clampMin")) {
        param.clampMin = paramJson["clampMin"].get<float>();
    }
    if (paramJson.contains("clampMax")) {
        param.clampMax = paramJson["clampMax"].get<float>();
    }
    if (paramJson.contains("isTriuMask")) {
        param.isTriuMask = paramJson["isTriuMask"].get<uint32_t>();
    }
    if (paramJson.contains("kvcacheCfg")) {
        param.kvcacheCfg = atb::infer::SelfAttentionParam::KvCacheCfg(paramJson["kvcacheCfg"].get<int32_t>());
    }
    if (paramJson.contains("scaleType")) {
        param.scaleType = atb::infer::SelfAttentionParam::ScaleType(paramJson["scaleType"].get<int32_t>());
    }
    if (paramJson.contains("quantType")) {
        param.quantType = atb::infer::SelfAttentionParam::QuantType(paramJson["quantType"].get<int32_t>());
    }
    if (paramJson.contains("outDataType")) {
        param.outDataType = aclDataType(paramJson["outDataType"].get<int32_t>());
    }
    if (paramJson.contains("inputLayout")) {
        param.inputLayout = atb::infer::InputLayout(paramJson["inputLayout"].get<int32_t>());
    }
    if (paramJson.contains("mlaVHeadSize")) {
        param.mlaVHeadSize = paramJson["mlaVHeadSize"].get<int>();
    }
    if (paramJson.contains("windowSize")) {
        param.windowSize = paramJson["windowSize"].get<uint32_t>();
    }
    if (paramJson.contains("cacheType")) {
        param.cacheType = atb::infer::SelfAttentionParam::CacheType(paramJson["cacheType"].get<int32_t>());
    }
    if (paramJson.contains("batchRunStatusEnable")) {
        param.batchRunStatusEnable = paramJson["batchRunStatusEnable"].get<bool>();
    }
    ATB_LOG(INFO) << "SelfAttentionParam headNum:" << param.headNum << ", qScale:" << param.qScale
                  << ", qkScale:" << param.qkScale << ", maskType:" << param.maskType
                  << ", kvHeadNum:" << param.kvHeadNum << ", calcType:" << param.calcType
                  << ", clampType:" << param.clampType << ", kvcacheCfg:" << param.kvcacheCfg
                  << ", scaleType:" << param.scaleType << ", inputLayout:" << param.inputLayout
                  << ", mlaVHeadSize:" << param.mlaVHeadSize << ", windowSize:" << param.windowSize
                  << ", cacheType: " << param.cacheType;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status RelayAttentionOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::RelayAttentionParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("qkScale")) {
        param.qkScale = paramJson["qkScale"].get<float>();
    }
    if (paramJson.contains("kvHeadNum")) {
        param.kvHeadNum = paramJson["kvHeadNum"].get<int>();
    }
    if (paramJson.contains("maskType")) {
        param.maskType = atb::infer::RelayAttentionParam::MaskType(paramJson["maskType"].get<int32_t>());
    }
    ATB_LOG(INFO) << "RelayAttentionParam headNum:" << param.headNum << ", qkScale:" << param.qkScale
                  << ", maskType:" << param.maskType << ", kvHeadNum:" << param.kvHeadNum;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status SetValueOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::SetValueParam param;
    for (auto item : paramJson["starts"]) {
        param.starts.push_back(item.get<int>());
        ATB_LOG(INFO) << "starts:" << param.starts.at(0);
    }
    for (auto item : paramJson["ends"]) {
        param.ends.push_back(item.get<int>());
        ATB_LOG(INFO) << "ends:" << param.ends.at(0);
    }
    for (auto item : paramJson["strides"]) {
        param.strides.push_back(item.get<int>());
        ATB_LOG(INFO) << "strides:" << param.strides.at(0);
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::infer::SortParam SortParamFromJson(const nlohmann::json &paramJson)
{
    atb::infer::SortParam param;
    for (auto item : paramJson["num"]) {
        param.num.push_back(item.get<int>());
        ATB_LOG(INFO) << "num:" << param.num.at(0);
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return param;
}

static atb::Status SortOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    return CreateOperation(SortParamFromJson(paramJson), op);
}

static atb::Status SortOperationUpdate(const nlohmann::json &paramJson, atb::Operation *op)
{
    return UpdateOperationParam(op, SortParamFromJson(paramJson));
}

static atb::Status TransposeOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::TransposeParam param;
    for (auto item : paramJson["perm"]) {
        param.perm.push_back(item.get<int>());
    }
    ATB_LOG(INFO) << "transpose(" << param.perm << ")";
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status LinearSparseOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::LinearSparseParam param;
    if (paramJson.contains("transposeA")) {
        param.transposeA = paramJson["transposeA"].get<bool>();
    }
    if (paramJson.contains("transposeB")) {
        param.transposeB = paramJson["transposeB"].get<bool>();
    }
    if (paramJson.contains("tilingK")) {
        param.tilingK = paramJson["tilingK"].get<uint32_t>();
    }
    if (paramJson.contains("tilingN")) {
        param.tilingN = paramJson["tilingN"].get<uint32_t>();
    }
    ATB_LOG(INFO) << "LinearSparseParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB
                  << ", tilingK:" << param.tilingK << ", tilingN:" << param.tilingN;

    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status ActivationOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::ActivationParam param;
    if (paramJson.contains("activationType")) {
        param.activationType = atb::infer::ActivationType(paramJson["activationType"].get<int32_t>());
    }
    if (paramJson.contains("scale")) {
        param.scale = paramJson["scale"].get<float>();
    }
    if (paramJson.contains("geluMode")) {
        param.geluMode = atb::infer::ActivationParam::GeLUMode(paramJson["geluMode"].get<int32_t>());
    }
    if (paramJson.contains("dim")) {
        param.dim = atb::infer::ActivationParam::GeLUMode(paramJson["dim"].get<int32_t>());
    }
    ATB_LOG(INFO) << "ActivationParam activationType:" << param.activationType << ", scale:" << param.scale;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}


static atb::Status RopeOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::RopeParam param;
    if (paramJson.contains("rotaryCoeff")) {
        param.rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    }
    if (paramJson.contains("cosFormat")) {
        param.cosFormat = paramJson["cosFormat"].get<int>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status RopeQConcatOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::RopeQConcatParam param;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status KvCacheOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::KvCacheParam param;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status PagedAttentionOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::PagedAttentionParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.qkScale = paramJson["qkScale"].get<float>();
    param.kvHeadNum = paramJson["kvHeadNum"].get<int>();

    if (paramJson.contains("maskType")) {
        param.maskType = atb::infer::PagedAttentionParam::MaskType(paramJson["maskType"].get<int32_t>());
    }

    if (paramJson.contains("quantType")) {
        param.quantType = atb::infer::PagedAttentionParam::QuantType(paramJson["quantType"].get<int32_t>());
    }

    if (paramJson.contains("hasQuantOffset")) {
        param.hasQuantOffset = paramJson["hasQuantOffset"].get<bool>();
    }

    if (paramJson.contains("compressType")) {
        param.compressType = atb::infer::PagedAttentionParam::CompressType(paramJson["compressType"].get<int32_t>());
    }
    if (paramJson.contains("calcType")) {
        param.calcType = atb::infer::PagedAttentionParam::CalcType(paramJson["calcType"].get<int32_t>());
    }
    if (paramJson.contains("scaleType")) {
        param.scaleType = atb::infer::PagedAttentionParam::ScaleType(paramJson["scaleType"].get<int32_t>());
    }
    if (paramJson.contains("inputLayout")) {
        param.inputLayout = atb::infer::InputLayout(paramJson["inputLayout"].get<int32_t>());
    }
    if (paramJson.contains("outDataType")) {
        param.outDataType = aclDataType(paramJson["outDataType"].get<int32_t>());
    }
    if (paramJson.contains("mlaVHeadSize")) {
        param.mlaVHeadSize = paramJson["mlaVHeadSize"].get<int>();
    }
    if (paramJson.contains("batchRunStatusEnable")) {
        param.batchRunStatusEnable = paramJson["batchRunStatusEnable"].get<bool>();
    }
    ATB_LOG(INFO) << "PagedAttentionOperationCreate headNum:" << param.headNum << ", scale:" << param.qkScale
                  << ", kvHeadNum:" << param.kvHeadNum << ", quantType:" << param.quantType
                  << ", hasQuantOffset:" << param.hasQuantOffset << ", compressType:" << param.compressType
                  << ", calcType:" << param.calcType << ", scaleType:" << param.scaleType
                  << ", outDataType:" << param.outDataType << ", inputLayout:" << param.inputLayout
                  << ", mlaVHeadSize:" << param.mlaVHeadSize << ", batchRunStatusEnable:" << param.batchRunStatusEnable;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status ReshapeAndCacheOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::ReshapeAndCacheParam param;
    if (paramJson.contains("compressType")) {
        param.compressType = atb::infer::ReshapeAndCacheParam::CompressType(paramJson["compressType"].get<int32_t>());
    }
    if (paramJson.contains("kvCacheCfg")) {
        param.kvCacheCfg = atb::infer::ReshapeAndCacheParam::KvCacheCfg(paramJson["kvCacheCfg"].get<int32_t>());
    }
    ATB_LOG(INFO) << "ReshapeAndCacheOperation Create compressType:" << param.compressType
                  << ", kvCacheCfg:" << param.kvCacheCfg;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status ReshapeAndCacheWithStrideOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::ReshapeAndCacheWithStrideParam param;
    if (paramJson.contains("compressType")) {
        param.compressType =
            atb::infer::ReshapeAndCacheWithStrideParam::CompressType(paramJson["compressType"].get<int32_t>());
    }
    if (paramJson.contains("kvCacheCfg")) {
        param.kvCacheCfg =
            atb::infer::ReshapeAndCacheWithStrideParam::KvCacheCfg(paramJson["kvCacheCfg"].get<int32_t>());
    }
    ATB_LOG(INFO) << "ReshapeAndCacheWithStrideOperation Create compressType:" << param.compressType
                  << ", kvCacheCfg:" << param.kvCacheCfg;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}
static atb::infer::FillParam FillParamFromJson(const nlohmann::json &paramJson)
{
    atb::infer::FillParam param;
    if (paramJson.contains("withMask")) {
        param.withMask = paramJson["withMask"].get<bool>();
    }
    if (paramJson.contains("value")) {
        for (auto item : paramJson["value"]) {
            param.value.push_back(item.get<float>());
        }
    }
    if (paramJson.contains("outDim")) {
        for (auto item : paramJson["outDim"]) {
            param.outDim.push_back(item.get<int32_t>());
        }
    }
    ATB_LOG(INFO) << "FillParam withMask:" << param.withMask << ", value:" << param.value
                  << ", outDim:" << param.outDim;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return param;
}

static atb::Status FillOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    return CreateOperation(FillParamFromJson(paramJson), op);
}

static atb::Status FillOperationUpdate(const nlohmann::json &paramJson, atb::Operation *op)
{
    return UpdateOperationParam(op, FillParamFromJson(paramJson));
}

static atb::Status RepeatOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::RepeatParam param;
    for (auto item : paramJson["multiples"]) {
        param.multiples.push_back(item.get<int64_t>());
    }
    ATB_LOG(INFO) << "RepeatParam multiples:" << param.multiples;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status SliceOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::SliceParam param;
    for (auto item : paramJson["offsets"]) {
        param.offsets.push_back(item.get<int64_t>());
    }
    for (auto item : paramJson["size"]) {
        param.size.push_back(item.get<int64_t>());
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status SoftmaxOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::SoftmaxParam param;
    for (auto item : paramJson["axes"]) {
        param.axes.push_back(item.get<int64_t>());
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status OnehotOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::OnehotParam param;
    if (paramJson.contains("axis")) {
        param.axis = paramJson["axis"].get<int64_t>();
    }
    if (paramJson.contains("depth")) {
        param.depth = paramJson["depth"].get<int64_t>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status ElewiseOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::ElewiseParam param;
    param.elewiseType = paramJson["elewiseType"].get<atb::infer::ElewiseParam::ElewiseType>();
    if (paramJson.contains("mulsParam")) {
        const auto &mulsParam = paramJson["mulsParam"];
        if (mulsParam.contains("varAttr")) {
            param.mulsParam.varAttr = mulsParam["varAttr"].get<float>();
        }
        if (mulsParam.contains("rsv")) {
            for (size_t i = 0; i < mulsParam["rsv"].size(); i++) {
                param.mulsParam.rsv[i] = mulsParam["rsv"].at(i).get<int8_t>();
            }
        }
    }
    if (paramJson.contains("outTensorType")) {
        param.outTensorType = paramJson["outTensorType"].get<aclDataType>();
    }
    if (paramJson.contains("quantParam")) {
        const auto &quantParam = paramJson["quantParam"];
        if (quantParam.contains("inputScale")) {
            param.quantParam.inputScale = quantParam["inputScale"].get<float>();
        }
        if (quantParam.contains("inputOffset")) {
            param.quantParam.inputOffset = quantParam["inputOffset"].get<int>();
        }
        if (quantParam.contains("asymmetric")) {
            param.quantParam.asymmetric = quantParam["asymmetric"].get<bool>();
        }
        if (quantParam.contains("rsv")) {
            for (size_t i = 0; i < quantParam["rsv"].size(); i++) {
                param.quantParam.rsv[i] = quantParam["rsv"].at(i).get<int8_t>();
            }
        }
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::infer::TopkToppSamplingParam TopkToppSamplingParamFromJson(const nlohmann::json &paramJson)
{
    atb::infer::TopkToppSamplingParam param;
    if (paramJson.contains("topkToppSamplingType")) {
        param.topkToppSamplingType =
            paramJson["topkToppSamplingType"].get<atb::infer::TopkToppSamplingParam::TopkToppSamplingType>();
    }
    if (paramJson.contains("randSeeds")) {
        for (auto item : paramJson["randSeeds"]) {
            param.randSeeds.push_back(item.get<uint32_t>());
        }
    }
    if (paramJson.contains("randSeed")) {
        param.randSeed = paramJson["randSeed"].get<uint32_t>();
    }
    if (paramJson.contains("topk")) {
        param.topk = paramJson["topk"].get<uint32_t>();
    }
    if (paramJson.contains("logProbsSize")) {
        param.logProbsSize = paramJson["logProbsSize"].get<int32_t>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return param;
}

static atb::Status TopkToppSamplingOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    return CreateOperation(TopkToppSamplingParamFromJson(paramJson), op);
}

static atb::Status TopkToppSamplingOperationUpdate(const nlohmann::json &paramJson, atb::Operation *op)
{
    return UpdateOperationParam(op, TopkToppSamplingParamFromJson(paramJson));
}

static atb::Status PadOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::PadParam param;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status NonzeroOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::NonzeroParam param;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status IndexAddOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::IndexAddParam param;
    param.indexType = paramJson["indexType"].get<atb::infer::IndexAddParam::IndexType>();
    if (paramJson.contains("axis")) {
        param.axis = paramJson["axis"].get<int64_t>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status UnpadOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::UnpadParam param;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::train::GenAttentionMaskParam GetGenAttentionMaskParamFromJson(const nlohmann::json &paramJson)
{
    atb::train::GenAttentionMaskParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int32_t>();
        ATB_LOG(INFO) << "param.headNum:" << param.headNum;
    }
    for (auto item : paramJson["seqLen"]) {
        param.seqLen.push_back(item.get<int>());
    }
    ATB_LOG(INFO) << "param.seqLen:" << param.seqLen;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return param;
}

static atb::Status GenAttentionMaskOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    return CreateOperation(GetGenAttentionMaskParamFromJson(paramJson), op);
}

static atb::Status GenAttentionMaskOperationUpdate(const nlohmann::json &paramJson, atb::Operation *op)
{
    return UpdateOperationParam(op, GetGenAttentionMaskParamFromJson(paramJson));
}

static atb::train::RopeGradParam GetRopeGradParamFromJson(const nlohmann::json &paramJson)
{
    atb::train::RopeGradParam param;
    for (auto item : paramJson["qSeqLen"]) {
        param.qSeqLen.push_back(item.get<int>());
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return param;
}

static atb::Status RopeGradOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    return CreateOperation(GetRopeGradParamFromJson(paramJson), op);
}

static atb::Status RopeGradOperationUpdate(const nlohmann::json &paramJson, atb::Operation *op)
{
    return UpdateOperationParam(op, GetRopeGradParamFromJson(paramJson));
}

static atb::Status RmsNormBackwardOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::train::RmsNormBackwardParam param;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status LayerNormOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::LayerNormParam param;
    if (paramJson.contains("layerType")) {
        param.layerType = atb::infer::LayerNormParam::LayerNormType(paramJson["layerType"].get<int32_t>());
    }
    if (param.layerType == atb::infer::LayerNormParam::LAYER_NORM_NORM) {
        const auto &normParam = paramJson["normParam"];
        if (normParam.contains("epsilon")) {
            param.normParam.epsilon = normParam["epsilon"].get<float>();
        }
        if (normParam.contains("quantType")) {
            param.normParam.quantType = atb::infer::QuantType(normParam["quantType"].get<int32_t>());
        }
        if (normParam.contains("beginNormAxis")) {
            param.normParam.beginNormAxis = normParam["beginNormAxis"].get<int32_t>();
        }
        if (normParam.contains("beginParamsAxis")) {
            param.normParam.beginParamsAxis = normParam["beginParamsAxis"].get<int32_t>();
        }
        if (normParam.contains("dynamicQuantType")) {
            param.normParam.dynamicQuantType =
                atb::infer::DynamicQuantType(normParam["dynamicQuantType"].get<int32_t>());
        }
        if (normParam.contains("rsv")) {
            for (size_t i = 0; i < normParam["rsv"].size(); i++) {
                param.normParam.rsv[i] = normParam["rsv"].at(i).get<int8_t>();
            }
        }
    }
    if (param.layerType == atb::infer::LayerNormParam::LAYER_NORM_PRENORM) {
        const auto &preNormParam = paramJson["preNormParam"];
        if (preNormParam.contains("epsilon")) {
            param.preNormParam.epsilon = preNormParam["epsilon"].get<float>();
        }
        if (preNormParam.contains("quantType")) {
            param.preNormParam.quantType = atb::infer::QuantType(preNormParam["quantType"].get<int32_t>());
        }
        if (preNormParam.contains("opMode")) {
            param.preNormParam.opMode = preNormParam["opMode"].get<size_t>();
        }
        if (preNormParam.contains("zoomScaleValue")) {
            param.preNormParam.zoomScaleValue = preNormParam["zoomScaleValue"].get<float>();
        }
        if (preNormParam.contains("rsv")) {
            for (size_t i = 0; i < preNormParam["rsv"].size(); i++) {
                param.preNormParam.rsv[i] = preNormParam["rsv"].at(i).get<int8_t>();
            }
        }
    }
    if (param.layerType == atb::infer::LayerNormParam::LAYER_NORM_POSTNORM) {
        const auto &postNormParam = paramJson["postNormParam"];
        if (postNormParam.contains("epsilon")) {
            param.postNormParam.epsilon = postNormParam["epsilon"].get<float>();
        }
        if (postNormParam.contains("quantType")) {
            param.postNormParam.quantType = atb::infer::QuantType(postNormParam["quantType"].get<int32_t>());
        }
        if (postNormParam.contains("opMode")) {
            param.postNormParam.opMode = postNormParam["opMode"].get<size_t>();
        }
        if (postNormParam.contains("zoomScaleValue")) {
            param.postNormParam.zoomScaleValue = postNormParam["zoomScaleValue"].get<float>();
        }
        if (postNormParam.contains("rsv")) {
            for (size_t i = 0; i < postNormParam["rsv"].size(); i++) {
                param.postNormParam.rsv[i] = postNormParam["rsv"].at(i).get<int8_t>();
            }
        }
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status LayerNormWithStrideOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::LayerNormWithStrideParam param;
    if (paramJson.contains("layerType")) {
        param.layerType = atb::infer::LayerNormWithStrideParam::LayerNormType(paramJson["layerType"].get<int32_t>());
    }
    if (param.layerType == atb::infer::LayerNormWithStrideParam::LAYER_NORM_NORM) {
        const auto &normParam = paramJson["normParam"];
        if (normParam.contains("epsilon")) {
            param.normParam.epsilon = normParam["epsilon"].get<float>();
        }
        if (normParam.contains("quantType")) {
            param.normParam.quantType = atb::infer::QuantType(normParam["quantType"].get<int32_t>());
        }
        if (normParam.contains("beginNormAxis")) {
            param.normParam.beginNormAxis = normParam["beginNormAxis"].get<int32_t>();
        }
        if (normParam.contains("beginParamsAxis")) {
            param.normParam.beginParamsAxis = normParam["beginParamsAxis"].get<int32_t>();
        }
        if (normParam.contains("dynamicQuantType")) {
            param.normParam.dynamicQuantType =
                atb::infer::DynamicQuantType(normParam["dynamicQuantType"].get<int32_t>());
        }
        if (normParam.contains("rsv")) {
            for (size_t i = 0; i < normParam["rsv"].size(); i++) {
                param.normParam.rsv[i] = normParam["rsv"].at(i).get<int8_t>();
            }
        }
    }
    if (param.layerType == atb::infer::LayerNormWithStrideParam::LAYER_NORM_PRENORM) {
        const auto &preNormParam = paramJson["preNormParam"];
        if (preNormParam.contains("epsilon")) {
            param.preNormParam.epsilon = preNormParam["epsilon"].get<float>();
        }
        if (preNormParam.contains("quantType")) {
            param.preNormParam.quantType = atb::infer::QuantType(preNormParam["quantType"].get<int32_t>());
        }
        if (preNormParam.contains("opMode")) {
            param.preNormParam.opMode = preNormParam["opMode"].get<size_t>();
        }
        if (preNormParam.contains("zoomScaleValue")) {
            param.preNormParam.zoomScaleValue = preNormParam["zoomScaleValue"].get<float>();
        }
        if (preNormParam.contains("rsv")) {
            for (size_t i = 0; i < preNormParam["rsv"].size(); i++) {
                param.preNormParam.rsv[i] = preNormParam["rsv"].at(i).get<int8_t>();
            }
        }
    }
    if (param.layerType == atb::infer::LayerNormWithStrideParam::LAYER_NORM_POSTNORM) {
        const auto &postNormParam = paramJson["postNormParam"];
        if (postNormParam.contains("epsilon")) {
            param.postNormParam.epsilon = postNormParam["epsilon"].get<float>();
        }
        if (postNormParam.contains("quantType")) {
            param.postNormParam.quantType = atb::infer::QuantType(postNormParam["quantType"].get<int32_t>());
        }
        if (postNormParam.contains("opMode")) {
            param.postNormParam.opMode = postNormParam["opMode"].get<size_t>();
        }
        if (postNormParam.contains("zoomScaleValue")) {
            param.postNormParam.zoomScaleValue = postNormParam["zoomScaleValue"].get<float>();
        }
        if (postNormParam.contains("rsv")) {
            for (size_t i = 0; i < postNormParam["rsv"].size(); i++) {
                param.postNormParam.rsv[i] = postNormParam["rsv"].at(i).get<int8_t>();
            }
        }
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status RmsNormOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::RmsNormParam param;
    if (paramJson.contains("layerType")) {
        param.layerType = atb::infer::RmsNormParam::RmsNormType(paramJson["layerType"].get<int32_t>());
    }
    if (param.layerType == atb::infer::RmsNormParam::RMS_NORM_NORM) {
        const auto &normParam = paramJson["normParam"];
        if (normParam.contains("epsilon")) {
            param.normParam.epsilon = normParam["epsilon"].get<float>();
        }
        if (normParam.contains("quantType")) {
            param.normParam.quantType = atb::infer::QuantType(normParam["quantType"].get<int32_t>());
        }
        if (normParam.contains("layerNormEps")) {
            param.normParam.layerNormEps = normParam["layerNormEps"].get<double>();
        }
        if (normParam.contains("rstd")) {
            param.normParam.rstd = normParam["rstd"].get<bool>();
        }
        if (normParam.contains("precisionMode")) {
            param.normParam.precisionMode = normParam["precisionMode"].get<atb::infer::RmsNormParam::PrecisionMode>();
        }
        if (normParam.contains("modelType")) {
            param.normParam.modelType = normParam["modelType"].get<atb::infer::RmsNormParam::ModelType>();
        }
        if (normParam.contains("dynamicQuantType")) {
            param.normParam.dynamicQuantType =
                atb::infer::DynamicQuantType(normParam["dynamicQuantType"].get<int32_t>());
        }
        if (normParam.contains("rsv")) {
            for (size_t i = 0; i < normParam["rsv"].size(); i++) {
                param.normParam.rsv[i] = normParam["rsv"].at(i).get<int8_t>();
            }
        }
    }
    if (param.layerType == atb::infer::RmsNormParam::RMS_NORM_PRENORM) {
        const auto &preNormParam = paramJson["preNormParam"];
        if (preNormParam.contains("epsilon")) {
            param.preNormParam.epsilon = preNormParam["epsilon"].get<float>();
        }
        if (preNormParam.contains("quantType")) {
            param.preNormParam.quantType = atb::infer::QuantType(preNormParam["quantType"].get<int32_t>());
        }
        if (preNormParam.contains("hasBias")) {
            param.preNormParam.hasBias = preNormParam["hasBias"].get<bool>();
        }
        if (preNormParam.contains("rsv")) {
            for (size_t i = 0; i < preNormParam["rsv"].size(); i++) {
                param.preNormParam.rsv[i] = preNormParam["rsv"].at(i).get<int8_t>();
            }
        }
    }
    if (param.layerType == atb::infer::RmsNormParam::RMS_NORM_POSTNORM) {
        const auto &postNormParam = paramJson["postNormParam"];
        ATB_LOG(INFO) << "postNormParam:" << postNormParam;
        if (postNormParam.contains("epsilon")) {
            param.postNormParam.epsilon = postNormParam["epsilon"].get<float>();
        }
        if (postNormParam.contains("quantType")) {
            ATB_LOG(INFO) << "postNormParam:" << postNormParam;
            param.postNormParam.quantType = atb::infer::QuantType(postNormParam["quantType"].get<int32_t>());
        }
        if (postNormParam.contains("hasBias")) {
            param.postNormParam.hasBias = postNormParam["hasBias"].get<bool>();
        }
        if (postNormParam.contains("rsv")) {
            for (size_t i = 0; i < postNormParam["rsv"].size(); i++) {
                param.postNormParam.rsv[i] = postNormParam["rsv"].at(i).get<int8_t>();
            }
        }
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status RmsNormWithStrideOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::RmsNormWithStrideParam param;
    if (paramJson.contains("layerType")) {
        param.layerType = atb::infer::RmsNormWithStrideParam::RmsNormType(paramJson["layerType"].get<int32_t>());
    }
    if (param.layerType == atb::infer::RmsNormWithStrideParam::RMS_NORM_NORM) {
        const auto &normParam = paramJson["normParam"];
        if (normParam.contains("epsilon")) {
            param.normParam.epsilon = normParam["epsilon"].get<float>();
        }
        if (normParam.contains("quantType")) {
            param.normParam.quantType = atb::infer::QuantType(normParam["quantType"].get<int32_t>());
        }
        if (normParam.contains("layerNormEps")) {
            param.normParam.layerNormEps = normParam["layerNormEps"].get<double>();
        }
        if (normParam.contains("rstd")) {
            param.normParam.rstd = normParam["rstd"].get<bool>();
        }
        if (normParam.contains("precisionMode")) {
            param.normParam.precisionMode =
                normParam["precisionMode"].get<atb::infer::RmsNormWithStrideParam::PrecisionMode>();
        }
        if (normParam.contains("modelType")) {
            param.normParam.modelType = normParam["modelType"].get<atb::infer::RmsNormWithStrideParam::ModelType>();
        }
        if (normParam.contains("dynamicQuantType")) {
            param.normParam.dynamicQuantType =
                atb::infer::DynamicQuantType(normParam["dynamicQuantType"].get<int32_t>());
        }
        if (normParam.contains("rsv")) {
            for (size_t i = 0; i < normParam["rsv"].size(); i++) {
                param.normParam.rsv[i] = normParam["rsv"].at(i).get<int8_t>();
            }
        }
    }
    if (param.layerType == atb::infer::RmsNormWithStrideParam::RMS_NORM_PRENORM) {
        const auto &preNormParam = paramJson["preNormParam"];
        if (preNormParam.contains("epsilon")) {
            param.preNormParam.epsilon = preNormParam["epsilon"].get<float>();
        }
        if (preNormParam.contains("quantType")) {
            param.preNormParam.quantType = atb::infer::QuantType(preNormParam["quantType"].get<int32_t>());
        }
        if (preNormParam.contains("hasBias")) {
            param.preNormParam.hasBias = preNormParam["hasBias"].get<bool>();
        }
        if (preNormParam.contains("rsv")) {
            for (size_t i = 0; i < preNormParam["rsv"].size(); i++) {
                param.preNormParam.rsv[i] = preNormParam["rsv"].at(i).get<int8_t>();
            }
        }
    }
    if (param.layerType == atb::infer::RmsNormWithStrideParam::RMS_NORM_POSTNORM) {
        const auto &postNormParam = paramJson["postNormParam"];
        ATB_LOG(INFO) << "postNormParam:" << postNormParam;
        if (postNormParam.contains("epsilon")) {
            param.postNormParam.epsilon = postNormParam["epsilon"].get<float>();
        }
        if (postNormParam.contains("quantType")) {
            ATB_LOG(INFO) << "postNormParam:" << postNormParam;
            param.postNormParam.quantType = atb::infer::QuantType(postNormParam["quantType"].get<int32_t>());
        }
        if (postNormParam.contains("hasBias")) {
            param.postNormParam.hasBias = postNormParam["hasBias"].get<bool>();
        }
        if (postNormParam.contains("rsv")) {
            for (size_t i = 0; i < postNormParam["rsv"].size(); i++) {
                param.postNormParam.rsv[i] = postNormParam["rsv"].at(i).get<int8_t>();
            }
        }
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::train::StridedBatchMatmulParam GetStridedBatchMatmulParamFromJson(const nlohmann::json &paramJson)
{
    atb::train::StridedBatchMatmulParam param;
    if (paramJson.contains("transA")) {
        param.transposeA = paramJson["transA"].get<int32_t>();
        ATB_LOG(INFO) << "param.transposeA:" << param.transposeA;
    }
    if (paramJson.contains("transB")) {
        param.transposeB = paramJson["transB"].get<int32_t>();
        ATB_LOG(INFO) << "param.transposeB:" << param.transposeB;
    }
    if (paramJson.contains("batch")) {
        param.batch = paramJson["batch"].get<int32_t>();
        ATB_LOG(INFO) << "param.batch:" << param.batch;
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int32_t>();
        ATB_LOG(INFO) << "param.headNum:" << param.headNum;
    }
    for (auto item : paramJson["m"]) {
        param.m.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["n"]) {
        param.n.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["k"]) {
        param.k.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["lda"]) {
        param.lda.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["ldb"]) {
        param.ldb.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["ldc"]) {
        param.ldc.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["strideA"]) {
        param.strideA.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["strideB"]) {
        param.strideB.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["strideC"]) {
        param.strideC.push_back(item.get<int32_t>());
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return param;
}

static atb::Status StridedBatchMatmulOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    return CreateOperation(GetStridedBatchMatmulParamFromJson(paramJson), op);
}

static atb::Status StridedBatchMatmulOperationUpdate(const nlohmann::json &paramJson, atb::Operation *op)
{
    return UpdateOperationParam(op, GetStridedBatchMatmulParamFromJson(paramJson));
}

static atb::Status ChatGlm6BLayerEncoderOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb_speed::ChatGlm6BLayerEncoderParam param;
    param.layerNormEps = paramJson["layerNormEps"].get<double>();
    param.headNum = paramJson["headNum"].get<int>();
    param.headDim = paramJson["headDim"].get<int>();
    param.qScale = paramJson["qScale"].get<float>();
    param.qkScale = paramJson["qkScale"].get<float>();
    param.residualAddScale = paramJson["residualAddScale"].get<float>();
    ATB_LOG(INFO) << "ChatGlm6BLayerEncoderParam layerNormEps:" << param.layerNormEps << ", headNum:" << param.headNum
                  << ", headDim:" << param.headDim << ", qScale:" << param.qScale << ", qkScale:" << param.qkScale
                  << ", residualAddScale:" << param.residualAddScale;
    return atb_speed::CreateChatGlm6BLayerEncoderOperation(param, op);
}

static atb::Status Chatglm6BLmHeadOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb_speed::ChatGlm6BLmHeadParam param;
    ATB_LOG(INFO) << "ChatGlm6BLmHeadParam";
    return atb_speed::CreateChatGlm6BLmHeadOperation(param, op);
}

static atb::Status AsStridedOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::AsStridedParam param;
    for (auto item : paramJson["size"]) {
        param.size.push_back(item.get<int64_t>());
    }
    for (auto item : paramJson["stride"]) {
        param.stride.push_back(item.get<int64_t>());
    }
    for (auto item : paramJson["offset"]) {
        param.offset.push_back(item.get<int64_t>());
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status MultinomialOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::MultinomialParam param;
    if (paramJson.contains("numSamples")) {
        param.numSamples = paramJson["numSamples"].get<uint32_t>();
        ATB_LOG(INFO) << "param.numSamples:" << param.numSamples;
    }
    if (paramJson.contains("randSeed")) {
        param.randSeed = paramJson["randSeed"].get<uint32_t>();
        ATB_LOG(INFO) << "param.randSeed:" << param.randSeed;
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status ReduceOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::ReduceParam param;
    param.reduceType = paramJson["reduceType"].get<atb::infer::ReduceParam::ReduceType>();
    for (auto item : paramJson["axis"]) {
        param.axis.push_back(item.get<int64_t>());
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status WhereOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::WhereParam param;
    ATB_LOG(INFO) << "WhereParam: NULL";
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status TransdataOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::TransdataParam param;
    if (paramJson.contains("transdataType")) {
        param.transdataType = atb::infer::TransdataParam::TransdataType(paramJson["transdataType"].get<int>());
    }
    if (paramJson.contains("outCrops")) {
        param.outCrops.clear();
        for (auto item : paramJson["outCrops"]) {
            param.outCrops.push_back(item.get<int64_t>());
        }
    }
    ATB_LOG(INFO) << "TransdataParam transdataType:" << param.transdataType << ", outCrops:" << param.outCrops;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::train::FastSoftMaxParam GetFastSoftMaxParamFromJson(const nlohmann::json &paramJson)
{
    atb::train::FastSoftMaxParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int32_t>();
    }
    if (paramJson.contains("qSeqLen")) {
        for (auto item : paramJson["qSeqLen"]) {
            param.qSeqLen.push_back(item.get<int32_t>());
        }
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return param;
}

static atb::Status FastSoftMaxOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    return CreateOperation(GetFastSoftMaxParamFromJson(paramJson), op);
}

static atb::Status FastSoftMaxOperationUpdate(const nlohmann::json &paramJson, atb::Operation *op)
{
    return UpdateOperationParam(op, GetFastSoftMaxParamFromJson(paramJson));
}

static atb::train::FastSoftMaxGradParam GetFastSoftMaxGradParamFromJson(const nlohmann::json &paramJson)
{
    atb::train::FastSoftMaxGradParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int32_t>();
    }
    if (paramJson.contains("qSeqLen")) {
        for (auto item : paramJson["qSeqLen"]) {
            param.qSeqLen.push_back(item.get<int32_t>());
        }
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return param;
}

static atb::Status FastSoftMaxGradOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    return CreateOperation(GetFastSoftMaxGradParamFromJson(paramJson), op);
}

static atb::Status FastSoftMaxGradOperationUpdate(const nlohmann::json &paramJson, atb::Operation *op)
{
    return UpdateOperationParam(op, GetFastSoftMaxGradParamFromJson(paramJson));
}

static atb::Status GatingOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::GatingParam param;
    if (paramJson.contains("cumSumNum")) {
        param.cumSumNum = paramJson["cumSumNum"].get<int32_t>();
    }
    if (paramJson.contains("topkExpertNum")) {
        param.topkExpertNum = paramJson["topkExpertNum"].get<int32_t>();
    }
    if (paramJson.contains("deviceExpert")) {
        for (auto item : paramJson["deviceExpert"]) {
            param.deviceExpert.push_back(item.get<int32_t>());
        }
    }
    if (paramJson.contains("cumSumInt64")) {
        for (auto item : paramJson["cumSumInt64"]) {
            param.cumSumInt64 = paramJson["cumSumInt64"].get<bool>();
        }
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status Llama65BLayerOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb_speed::LlamaLayerFusionParallelParam param;
    if (paramJson.contains("rmsNormEps")) {
        param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("headDim")) {
        param.headDim = paramJson["headDim"].get<int>();
    }
    if (paramJson.contains("rank")) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.contains("rankSize")) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.contains("qkScale")) {
        param.qkScale = paramJson["qkScale"].get<float>();
    }
    if (paramJson.contains("rotaryCoeff")) {
        param.rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    }
    if (paramJson.contains("transpose")) {
        param.transpose = paramJson["transpose"].get<bool>();
    }
    if (paramJson.contains("batchRunStatusEnable")) {
        param.batchRunStatusEnable = paramJson["batchRunStatusEnable"].get<bool>();
    }
    if (paramJson.contains("coderType")) {
        param.coderType = paramJson["coderType"].get<int>();
    }
    if (paramJson.contains("isTriuMask")) {
        param.isTriuMask = paramJson["isTriuMask"].get<int>();
    }
    ATB_LOG(INFO) << "Llama65BLayerOperationCreate rmsNormEps:" << param.rmsNormEps << ", rank:" << param.rank;
    return atb_speed::CreateLlama65BLayerOperation(param, op);
}

static atb::train::UnpadWithHiddenStateParam GetUnpadWithHiddenStateParamFromJson(const nlohmann::json &paramJson)
{
    atb::train::UnpadWithHiddenStateParam param;
    if (paramJson.contains("qSeqLen")) {
        for (auto item : paramJson["qSeqLen"]) {
            param.qSeqLen.push_back(item.get<int32_t>());
        }
    }
    if (paramJson.contains("maxSeqLen")) {
        param.maxSeqLen = paramJson["maxSeqLen"].get<int32_t>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return param;
}

static atb::Status UnpadWithHiddenStateOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    return CreateOperation(GetUnpadWithHiddenStateParamFromJson(paramJson), op);
}

static atb::Status UnpadWithHiddenStateOperationUpdate(const nlohmann::json &paramJson, atb::Operation *op)
{
    return UpdateOperationParam(op, GetUnpadWithHiddenStateParamFromJson(paramJson));
}

static atb::train::PadWithHiddenStateParam GetPadWithHiddenStateParamFromJson(const nlohmann::json &paramJson)
{
    atb::train::PadWithHiddenStateParam param;
    if (paramJson.contains("qSeqLen")) {
        for (auto item : paramJson["qSeqLen"]) {
            param.qSeqLen.push_back(item.get<int32_t>());
        }
    }
    if (paramJson.contains("maxSeqLen")) {
        param.maxSeqLen = paramJson["maxSeqLen"].get<int32_t>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return param;
}

static atb::Status PadWithHiddenStateOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    return CreateOperation(GetPadWithHiddenStateParamFromJson(paramJson), op);
}

static atb::Status PadWithHiddenStateOperationUpdate(const nlohmann::json &paramJson, atb::Operation *op)
{
    return UpdateOperationParam(op, GetPadWithHiddenStateParamFromJson(paramJson));
}

static atb::Status CommonGraphOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb_speed::CommonGraphParam param;
    return atb_speed::CreateCommonGraphOperation(param, op);
}

static atb::Status ViewGraphOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb_speed::ViewGraphParam param;
    return atb_speed::CreateViewGraphOperation(param, op);
}

static atb::Status Llama65BMlpLayerCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb_speed::LlamaMlpParam param;
    if (paramJson.contains("transpose")) {
        param.transpose = paramJson["transpose"].get<bool>();
    }
    return atb_speed::CreateLlamaMlpOperation(param, op);
}

static atb::Status PostProcessOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb_speed::PostProcessParam param;
    param.temperature = paramJson["temperature"].get<double>();
    param.topK = paramJson["topK"].get<uint32_t>();
    param.randSeed = paramJson["randSeed"].get<uint32_t>();
    ATB_LOG(INFO) << "SampleLayerCreate: temperature:" << param.temperature << ", topK:" << param.topK
                  << ", randSeed:" << param.randSeed;
    return CreatePostProcessOperation(param, op);
}

static atb::Status Llama65BMlpLayerGraphBuilderCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb_speed::LlamaMlpParamGb param;
    if (paramJson.contains("transpose")) {
        param.transpose = paramJson["transpose"].get<bool>();
    }
    return atb_speed::CreateLlamaMlpOperationByGraphOpBuilder(param, op);
}

static atb::Status SendOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::SendParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    param.rankRoot = paramJson["rankRoot"].get<int>();
    param.destRank = paramJson["destRank"].get<uint32_t>();
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.contains("commMode")) {
        param.commMode = atb::infer::CommMode(paramJson["commMode"].get<int>());
    }
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.find("commDomain") != paramJson.end()) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    ATB_LOG(INFO) << "SendParam rank:" << param.rank << ", rankSize:" << param.rankSize
                  << ", rankRoot:" << param.rankRoot << ", destrank:" << param.destRank << ", backend:" << param.backend
                  << ", commDomain:" << param.commDomain;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status RecvOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::RecvParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    param.rankRoot = paramJson["rankRoot"].get<int>();
    param.srcRank = paramJson["srcRank"].get<uint32_t>();
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.contains("commMode")) {
        param.commMode = atb::infer::CommMode(paramJson["commMode"].get<int>());
    }
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.find("commDomain") != paramJson.end()) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    ATB_LOG(INFO) << "RecvParam rank:" << param.rank << ", rankSize:" << param.rankSize
                  << ", rankRoot:" << param.rankRoot << ", srcRank:" << param.srcRank << ", backend:" << param.backend
                  << ", commDomain:" << param.commDomain;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status all_to_all_operationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::AllToAllParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.contains("commMode")) {
        param.commMode = atb::infer::CommMode(paramJson["commMode"].get<int>());
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.find("commDomain") != paramJson.end()) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    if (paramJson.find("transpose") != paramJson.end()) {
        param.transpose = paramJson["transpose"].get<bool>();
    }
    ATB_LOG(INFO) << "AllToAllParam rank:" << param.rank << ", rankSize:" << param.rankSize
                  << ", rankRoot:" << param.rankRoot << ", backend:" << param.backend
                  << ", commDomain:" << param.commDomain << ", transpose:" << param.transpose;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status all_to_allv_operationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::AllToAllVParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    for (auto item : paramJson["sendCounts"]) {
        param.sendCounts.push_back(item.get<int64_t>());
    }
    for (auto item : paramJson["sdispls"]) {
        param.sdispls.push_back(item.get<int64_t>());
    }
    for (auto item : paramJson["recvCounts"]) {
        param.recvCounts.push_back(item.get<int64_t>());
    }
    for (auto item : paramJson["rdispls"]) {
        param.rdispls.push_back(item.get<int64_t>());
    }
    if (paramJson.contains("commMode")) {
        param.commMode = atb::infer::CommMode(paramJson["commMode"].get<int>());
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.find("commDomain") != paramJson.end()) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    ATB_LOG(INFO) << "AllToAllVParam rank:" << param.rank << ", rankSize:" << param.rankSize
                  << ", rankRoot:" << param.rankRoot << ", backend:" << param.backend
                  << ", commDomain:" << param.commDomain;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status all_to_allvv2_operationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::AllToAllVV2Param param;
    if (paramJson.contains("rank")) {
        param.rank = paramJson["rank"].get<int>();
    }
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.contains("commMode")) {
        param.commMode = atb::infer::CommMode(paramJson["commMode"].get<int>());
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.find("commDomain") != paramJson.end()) {
        param.commDomain = paramJson["commDomain"].get<std::string>();
    }
    ATB_LOG(INFO) << "AllToAllVV2Param rank:" << param.rank << ", rankSize:" << param.rankSize
                  << ", rankRoot:" << param.rankRoot << ", backend:" << param.backend
                  << ", commDomain:" << param.commDomain;
    return CreateOperation(param, op);
}

static atb::train::LaserAttentionParam GetLaserAttentionParamFromJson(const nlohmann::json &paramJson)
{
    atb::train::LaserAttentionParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("inputLayout")) {
        param.inputLayout = paramJson["inputLayout"].get<std::string>();
    }
    if (paramJson.contains("scaleValue")) {
        param.scaleValue = paramJson["scaleValue"].get<float>();
    }
    if (paramJson.contains("keepProb")) {
        param.keepProb = paramJson["keepProb"].get<float>();
    }
    if (paramJson.contains("preTokens")) {
        param.preTokens = paramJson["preTokens"].get<int>();
    }
    if (paramJson.contains("nextTokens")) {
        param.nextTokens = paramJson["nextTokens"].get<int>();
    }
    if (paramJson.contains("sparseMode")) {
        param.sparseMode = paramJson["sparseMode"].get<int>();
    }
    if (paramJson.contains("innerPrecise")) {
        param.innerPrecise = paramJson["innerPrecise"].get<int>();
    }
    ATB_LOG(INFO) << "LaserAttentionParam headNum: " << param.headNum << ", inputLayout: " << param.inputLayout
                  << ", scaleValue: " << param.scaleValue << ", keepProb: " << param.keepProb
                  << ", preTokens: " << param.preTokens << ", nextTokens: " << param.nextTokens
                  << ", sparseMode: " << param.sparseMode << ", innerPrecise: " << param.innerPrecise;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return param;
}

static atb::Status LaserAttentionOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    return CreateOperation(GetLaserAttentionParamFromJson(paramJson), op);
}

static atb::Status LaserAttentionOperationUpdate(const nlohmann::json &paramJson, atb::Operation *op)
{
    return UpdateOperationParam(op, GetLaserAttentionParamFromJson(paramJson));
}

static atb::train::LaserAttentionGradParam GetLaserAttentionGradParamFromJson(const nlohmann::json &paramJson)
{
    atb::train::LaserAttentionGradParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("inputLayout")) {
        param.inputLayout = paramJson["inputLayout"].get<std::string>();
    }
    if (paramJson.contains("scaleValue")) {
        param.scaleValue = paramJson["scaleValue"].get<float>();
    }
    if (paramJson.contains("keepProb")) {
        param.keepProb = paramJson["keepProb"].get<float>();
    }
    if (paramJson.contains("preTokens")) {
        param.preTokens = paramJson["preTokens"].get<int>();
    }
    if (paramJson.contains("nextTokens")) {
        param.nextTokens = paramJson["nextTokens"].get<int>();
    }
    if (paramJson.contains("sparseMode")) {
        param.sparseMode = paramJson["sparseMode"].get<int>();
    }
    if (paramJson.contains("innerPrecise")) {
        param.innerPrecise = paramJson["innerPrecise"].get<int>();
    }
    ATB_LOG(INFO) << "LaserAttentionGradParam headNum: " << param.headNum << ", inputLayout: " << param.inputLayout
                  << ", scaleValue: " << param.scaleValue << ", keepProb: " << param.keepProb
                  << ", preTokens: " << param.preTokens << ", nextTokens: " << param.nextTokens
                  << ", sparseMode: " << param.sparseMode << ", innerPrecise: " << param.innerPrecise;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return param;
}

static atb::Status LaserAttentionGradOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    return CreateOperation(GetLaserAttentionGradParamFromJson(paramJson), op);
}

static atb::Status LaserAttentionGradOperationUpdate(const nlohmann::json &paramJson, atb::Operation *op)
{
    return UpdateOperationParam(op, GetLaserAttentionGradParamFromJson(paramJson));
}

static atb::Status GroupedMatmulWithRoutingOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    ATB_LOG(INFO) << "GroupedMatmulOperationCreate";
    atb::infer::GroupedMatmulWithRoutingParam param;
    if (paramJson.contains("transposeB")) {
        param.transposeB = paramJson["transposeB"].get<bool>();
    }
    if (paramJson.contains("topK")) {
        param.topK = paramJson["topK"].get<int32_t>();
    }
    if (paramJson.contains("groupedMatmulType")) {
        param.groupedMatmulType =
            atb::infer::GroupedMatmulWithRoutingParam::GroupedMatmulType(paramJson["groupedMatmulType"].get<int>());
    }
    if (paramJson.contains("outDataType")) {
        param.outDataType = static_cast<aclDataType>(paramJson["outDataType"].get<int>());
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status CohereLayerNormOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    ATB_LOG(INFO) << "CohereLayerNormOperationCreate";
    atb::infer::CohereLayerNormParam param;
    if (paramJson.contains("epsilon")) {
        param.epsilon = paramJson["epsilon"].get<float>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status GatherPreRmsNormOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    ATB_LOG(INFO) << "GatherPreRmsNormOperationCreate";
    atb::infer::GatherPreRmsNormParam param;
    if (paramJson.contains("epsilon")) {
        param.epsilon = paramJson["epsilon"].get<float>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status SwigluQuantOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::SwigluQuantParam param;
    param.quantType = paramJson["quantType"].get<atb::infer::SwigluQuantParam::QuantType>();
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status NormRopeReshapeOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    ATB_LOG(INFO) << "NormRopeReshapeOperationCreate";
    atb::infer::NormRopeReshapeParam param;
    if (paramJson.contains("epsilon")) {
        param.epsilon = paramJson["epsilon"].get<float>();
    }
    if (paramJson.contains("precisionMode")) {
        param.precisionMode = paramJson["precisionMode"].get<uint32_t>();
    }
    if (paramJson.contains("rotaryCoeff")) {
        param.rotaryCoeff = paramJson["rotaryCoeff"].get<uint32_t>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status FusedAddTopkDivOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    ATB_LOG(INFO) << "FusedAddTopkDivOperationCreate";
    atb::infer::FusedAddTopkDivParam param;
    if (paramJson.contains("groupNum")) {
        param.groupNum = paramJson["groupNum"].get<uint32_t>();
    }
    if (paramJson.contains("groupTopk")) {
        param.groupTopk = paramJson["groupTopk"].get<uint32_t>();
    }
    if (paramJson.contains("n")) {
        param.n = paramJson["n"].get<uint32_t>();
    }
    if (paramJson.contains("k")) {
        param.k = paramJson["k"].get<uint32_t>();
    }
    if (paramJson.contains("activationType")) {
        param.activationType = paramJson["activationType"].get<atb::infer::ActivationType>();
    }
    if (paramJson.contains("isNorm")) {
        param.isNorm = paramJson["isNorm"].get<bool>();
    }
    if (paramJson.contains("scale")) {
        param.scale = paramJson["scale"].get<float>();
    }
    if (paramJson.contains("enableExpertMapping")) {
        param.enableExpertMapping = paramJson["enableExpertMapping"].get<bool>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<uint8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status MlaPreprocessOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    ATB_LOG(INFO) << "MlaPreprocessOperationCreate";
    atb::infer::MlaPreprocessParam param;
    if (paramJson.contains("wdqDim")) {
        param.wdqDim = paramJson["wdqDim"].get<uint32_t>();
    }
    if (paramJson.contains("qRopeDim")) {
        param.qRopeDim = paramJson["qRopeDim"].get<uint32_t>();
    }
    if (paramJson.contains("kRopeDim")) {
        param.kRopeDim = paramJson["kRopeDim"].get<uint32_t>();
    }
    if (paramJson.contains("epsilon")) {
        param.epsilon = paramJson["epsilon"].get<float>();
    }
    if (paramJson.contains("qRotaryCoeff")) {
        param.qRotaryCoeff = paramJson["qRotaryCoeff"].get<int32_t>();
    }
    if (paramJson.contains("kRotaryCoeff")) {
        param.kRotaryCoeff = paramJson["kRotaryCoeff"].get<int32_t>();
    }
    if (paramJson.contains("transposeWdq")) {
        param.transposeWdq = paramJson["transposeWdq"].get<bool>();
    }
    if (paramJson.contains("transposeWuq")) {
        param.transposeWuq = paramJson["transposeWuq"].get<bool>();
    }
    if (paramJson.contains("transposeWuk")) {
        param.transposeWuk = paramJson["transposeWuk"].get<bool>();
    }
    if (paramJson.contains("cacheMode")) {
        param.cacheMode = paramJson["cacheMode"].get<atb::infer::MlaPreprocessParam::CacheMode>();
    }
    if (paramJson.contains("quantMode")) {
        param.quantMode = paramJson["quantMode"].get<atb::infer::MlaPreprocessParam::QuantMode>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status MultiLatentAttentionOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::MultiLatentAttentionParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("qkScale")) {
        param.qkScale = paramJson["qkScale"].get<float>();
    }
    if (paramJson.contains("kvHeadNum")) {
        param.kvHeadNum = paramJson["kvHeadNum"].get<int>();
    }
    if (paramJson.contains("maskType")) {
        param.maskType = atb::infer::MultiLatentAttentionParam::MaskType(paramJson["maskType"].get<int32_t>());
    }
    if (paramJson.contains("calcType")) {
        param.calcType = atb::infer::MultiLatentAttentionParam::CalcType(paramJson["calcType"].get<int32_t>());
    }
    if (paramJson.contains("cacheMode")) {
        param.cacheMode = atb::infer::MultiLatentAttentionParam::CacheMode(paramJson["cacheMode"].get<uint8_t>());
    }
    if (paramJson.contains("windowSize")) {
        param.windowSize = atb::infer::MultiLatentAttentionParam::MaskType(paramJson["windowSize"].get<int32_t>());
    }
    ATB_LOG(INFO) << "PagedAttentionOperationCreate headNum:" << param.headNum << ", scale:" << param.qkScale
                  << ", kvHeadNum:" << param.kvHeadNum << ", maskType:" << param.maskType
                  << ", calcType:" << param.calcType << ", cacheMode:" << param.cacheMode
                  << ", windowSize:" << param.windowSize;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status ScatterElementsV2OperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::ScatterElementsV2Param param;
    ATB_LOG(INFO) << "ScatterElementsV2Param axis:" << param.axis;
    ATB_LOG(INFO) << "ScatterElementsV2Param reduction:" << param.reduction;
    if (paramJson.contains("axis")) {
        param.axis = paramJson["axis"].get<int32_t>();
    }
    if (paramJson.contains("reduction")) {
        param.reduction = paramJson["reduction"].get<atb::infer::ScatterElementsV2Param::ReductionType>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status ReshapeAndCacheOmniOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    ATB_LOG(INFO) << "ReshapeAndCacheOmniOperationCreate";
    atb::infer::ReshapeAndCacheOmniParam param;
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status RazorFusionAttentionOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    ATB_LOG(INFO) << "RazorFusionAttentionOperationCreate";
    atb::infer::RazorFusionAttentionParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int32_t>();
    }
    if (paramJson.contains("kvHeadNum")) {
        param.kvHeadNum = paramJson["kvHeadNum"].get<int32_t>();
    }
    if (paramJson.contains("qkScale")) {
        param.qkScale = paramJson["qkScale"].get<float>();
    }
    if (paramJson.contains("razorLen")) {
        param.razorLen = paramJson["razorLen"].get<int32_t>();
    }
    if (paramJson.contains("preTokens")) {
        param.preTokens = paramJson["preTokens"].get<int32_t>();
    }
    if (paramJson.contains("nextTokens")) {
        param.nextTokens = paramJson["nextTokens"].get<int32_t>();
    }
    if (paramJson.contains("tileQ")) {
        param.tileQ = paramJson["tileQ"].get<int32_t>();
    }
    if (paramJson.contains("tileKv")) {
        param.tileKv = paramJson["tileKv"].get<int32_t>();
    }
    if (paramJson.contains("textQLen")) {
        param.textQLen = paramJson["textQLen"].get<int32_t>();
    }
    if (paramJson.contains("textKvLen")) {
        param.textKvLen = paramJson["textKvLen"].get<int32_t>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<uint8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status FaUpdateOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::FaUpdateParam param;
    if (paramJson.contains("faUpdateType")) {
        param.faUpdateType = paramJson["faUpdateType"].get<atb::infer::FaUpdateParam::FaUpdateType>();
    }
    if (paramJson.contains("sp")) {
        param.sp = paramJson["sp"].get<uint32_t>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<uint8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status PagedCacheLoadOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    ATB_LOG(INFO) << "PagedCacheLoadOperationCreate";
    atb::infer::PagedCacheLoadParam param;
    if (paramJson.contains("kvCacheCfg")) {
        param.kvCacheCfg = atb::infer::PagedCacheLoadParam::KvCacheCfg(paramJson["kvCacheCfg"].get<int8_t>());
    }
    if (paramJson.contains("isSeqLensCumsumMode")) {
        param.isSeqLensCumsumMode = paramJson["isSeqLensCumsumMode"].get<bool>();
    }
    if (paramJson.contains("hasSeqStarts")) {
        param.hasSeqStarts = paramJson["hasSeqStarts"].get<bool>();
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<int8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status GmmDeqSwigluQuantGmmDeqOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    ATB_LOG(INFO) << "GmmDeqSwigluQuantGmmDeqOperationCreate";
    atb::infer::GmmDeqSwigluQuantGmmDeqParam param;
    if (paramJson.contains("outputType")) {
        param.outputType = paramJson["outputType"];
    }
    if (paramJson.contains("groupListType")) {
        param.groupListType = paramJson["groupListType"];
    }
    if (paramJson.contains("weightUpPermuteType")) {
        param.weightUpPermuteType = paramJson["weightUpPermuteType"];
    }
    if (paramJson.contains("transposeWeightUp")) {
        param.transposeWeightUp = paramJson["transposeWeightUp"];
    }
    if (paramJson.contains("transposeWeightDown")) {
        param.transposeWeightDown = paramJson["transposeWeightDown"];
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<uint8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status MmDeqSwigluQuantMmDeqOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    ATB_LOG(INFO) << "MmDeqSwigluQuantMmDeqOperationCreate";
    atb::infer::MmDeqSwigluQuantMmDeqParam param;
    if (paramJson.contains("outputType")) {
        param.outputType = paramJson["outputType"];
    }
    if (paramJson.contains("weightUpPermuteType")) {
        param.weightUpPermuteType = paramJson["weightUpPermuteType"];
    }
    if (paramJson.contains("transposeWeightUp")) {
        param.transposeWeightUp = paramJson["transposeWeightUp"];
    }
    if (paramJson.contains("transposeWeightDown")) {
        param.transposeWeightDown = paramJson["transposeWeightDown"];
    }
    if (paramJson.contains("rsv")) {
        for (size_t i = 0; i < paramJson["rsv"].size(); i++) {
            param.rsv[i] = paramJson["rsv"].at(i).get<uint8_t>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status RingMLAOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::RingMLAParam param;
    if (paramJson.contains("calcType")) {
        param.calcType = atb::infer::RingMLAParam::CalcType(paramJson["calcType"].get<int32_t>());
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("kvHeadNum")) {
        param.kvHeadNum = paramJson["kvHeadNum"].get<int>();
    }
    if (paramJson.contains("qkScale")) {
        param.qkScale = paramJson["qkScale"].get<float>();
    }
    if (paramJson.contains("kernelType")) {
        param.kernelType = atb::infer::RingMLAParam::KernelType(paramJson["kernelType"].get<int32_t>());
    }
    if (paramJson.contains("maskType")) {
        param.maskType = atb::infer::RingMLAParam::MaskType(paramJson["maskType"].get<int32_t>());
    }
    if (paramJson.contains("inputLayout")) {
        param.inputLayout = atb::infer::InputLayout(paramJson["inputLayout"].get<int32_t>());
    }
    return CreateOperation(param, op);
}

std::map<std::string, OperationCreateFunc> g_funcMap = {
    {"AllReduceOperation", &AllReduceOperationCreate},
    {"BroadcastOperation", &BroadcastOperationCreate},
    {"AllGatherOperation", &AllGatherOperationCreate},
    {"AllGatherVOperation", &AllGatherVOperationCreate},
    {"GatherOperation", &GatherOperationCreate},
    {"ConcatOperation", &ConcatOperationCreate},
    {"SplitOperation", &SplitOperationCreate},
    {"CumsumOperation", &CumsumOperationCreate},
    {"TransposeOperation", &TransposeOperationCreate},
    {"SelfAttentionOperation", &SelfAttentionOperationCreate},
    {"SetValueOperation", &SetValueOperationCreate},
    {"SortOperation", &SortOperationCreate},
    {"ActivationOperation", &ActivationOperationCreate},
    {"KvCacheOperation", &KvCacheOperationCreate},
    {"PagedAttentionOperation", &PagedAttentionOperationCreate},
    {"ReshapeAndCacheOperation", &ReshapeAndCacheOperationCreate},
    {"RopeOperation", &RopeOperationCreate},
    {"RopeQConcatOperation", &RopeQConcatOperationCreate},
    {"FillOperation", &FillOperationCreate},
    {"RepeatOperation", &RepeatOperationCreate},
    {"SliceOperation", &SliceOperationCreate},
    {"SoftmaxOperation", &SoftmaxOperationCreate},
    {"OnehotOperation", &OnehotOperationCreate},
    {"ElewiseOperation", &ElewiseOperationCreate},
    {"TopkToppSamplingOperation", &TopkToppSamplingOperationCreate},
    {"LinearOperation", &LinearOperationCreate},
    {"LinearParallelOperation", &LinearParallelOperationCreate},
    {"LinearSparseOperation", &LinearSparseOperationCreate},
    {"PadOperation", &PadOperationCreate},
    {"UnpadOperation", &UnpadOperationCreate},
    {"GenAttentionMaskOperation", &GenAttentionMaskOperationCreate},
    {"LayerNormOperation", &LayerNormOperationCreate},
    {"LayerNormWithStrideOperation", &LayerNormWithStrideOperationCreate},
    {"RmsNormOperation", &RmsNormOperationCreate},
    {"RmsNormWithStrideOperation", &RmsNormWithStrideOperationCreate},
    {"RmsNormBackwardOperation", &RmsNormBackwardOperationCreate},
    {"RopeGradOperation", &RopeGradOperationCreate},
    {"ChatGlm6BLayerEncoderOperation", &ChatGlm6BLayerEncoderOperationCreate},
    {"AsStridedOperation", &AsStridedOperationCreate},
    {"MultinomialOperation", &MultinomialOperationCreate},
    {"ReduceOperation", &ReduceOperationCreate},
    {"ReduceScatterOperation", &ReduceScatterOperationCreate},
    {"WhereOperation", &WhereOperationCreate},
    {"TransdataOperation", &TransdataOperationCreate},
    {"FastSoftMaxOperation", &FastSoftMaxOperationCreate},
    {"FastSoftMaxGradOperation", &FastSoftMaxGradOperationCreate},
    {"StridedBatchMatmulOperation", &StridedBatchMatmulOperationCreate},
    {"Llama65BLayerOperation", &Llama65BLayerOperationCreate},
    {"UnpadWithHiddenStateOperation", &UnpadWithHiddenStateOperationCreate},
    {"PadWithHiddenStateOperation", &PadWithHiddenStateOperationCreate},
    {"NonzeroOperation", &NonzeroOperationCreate},
    {"IndexAddOperation", &IndexAddOperationCreate},
    {"CommonGraphOperation", &CommonGraphOperationCreate},
    {"Llama65BMlpLayer", &Llama65BMlpLayerCreate},
    {"ViewGraphOperation", &ViewGraphOperationCreate},
    {"PostProcessOperation", &PostProcessOperationCreate},
    {"Llama65BMlpLayerGraphBuilder", &Llama65BMlpLayerGraphBuilderCreate},
    {"GatingOperation", &GatingOperationCreate},
    {"SendOperation", &SendOperationCreate},
    {"RecvOperation", &RecvOperationCreate},
    {"ReduceScatterVOperation", &ReduceScatterVOperationCreate},
    {"BlockCopyOperation", &BlockCopyOperationCreate},
    {"AllToAllOperation", &all_to_all_operationCreate},
    {"AllToAllVOperation", &all_to_allv_operationCreate},
    {"AllToAllVV2Operation", &all_to_allvv2_operationCreate},
    {"DynamicNTKOperation", &DynamicNTKOperationCreate},
    {"LaserAttentionOperation", &LaserAttentionOperationCreate},
    {"LaserAttentionGradOperation", &LaserAttentionGradOperationCreate},
    {"GroupTopkOperation", &GroupTopkOperationCreate},
    {"GroupedMatmulWithRoutingOperation", &GroupedMatmulWithRoutingOperationCreate},
    {"GroupedMatmulInplaceAddOperation", &GroupedMatmulInplaceAddOperationCreate},
    {"RelayAttentionOperation", &RelayAttentionOperationCreate},
    {"CohereLayerNormOperation", &CohereLayerNormOperationCreate},
    {"GatherPreRmsNormOperation", &GatherPreRmsNormOperationCreate},
    {"SwigluQuantOperation", &SwigluQuantOperationCreate},
    {"NormRopeReshapeOperation", &NormRopeReshapeOperationCreate},
    {"FusedAddTopkDivOperation", &FusedAddTopkDivOperationCreate},
    {"MlaPreprocessOperation", &MlaPreprocessOperationCreate},
    {"MultiLatentAttentionOperation", &MultiLatentAttentionOperationCreate},
    {"ReshapeAndCacheOmniOperation", &ReshapeAndCacheOmniOperationCreate},
    {"ReshapeAndCacheWithStrideOperation", &ReshapeAndCacheWithStrideOperationCreate},
    {"RazorFusionAttentionOperation", &RazorFusionAttentionOperationCreate},
    {"FaUpdateOperation", &FaUpdateOperationCreate},
    {"PagedCacheLoadOperation", &PagedCacheLoadOperationCreate},
    {"ScatterElementsV2Operation", &ScatterElementsV2OperationCreate},
    {"GmmDeqSwigluQuantGmmDeqOperation", &GmmDeqSwigluQuantGmmDeqOperationCreate},
    {"MmDeqSwigluQuantMmDeqOperation", &MmDeqSwigluQuantMmDeqOperationCreate},
    {"RingMLAOperation", &RingMLAOperationCreate},
};

atb::Status CreateOperation(const std::string &opName, const std::string &param, atb::Operation **operation)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);

    auto it = g_funcMap.find(opName);
    if (it == g_funcMap.end()) {
        ATB_LOG(ERROR) << "not support opName:" << opName;
        return atb::ERROR_INVALID_PARAM;
    }
    try {
        return it->second(paramJson, operation);
    } catch (const std::exception &e) {
        ATB_LOG(INFO) << "param is :" << param;
        ATB_LOG(ERROR) << opName << " parse json fail, error:" << e.what();
    }
    return atb::ERROR_INVALID_PARAM;
}

using OperationUpdateFunc = std::function<atb::Status(const nlohmann::json &paramJson, atb::Operation *op)>;

std::map<std::string, OperationUpdateFunc> g_update_funcMap = {
    {"FastSoftMaxOperation", &FastSoftMaxOperationUpdate},
    {"FastSoftMaxGradOperation", &FastSoftMaxGradOperationUpdate},
    {"GenAttentionMaskOperation", &GenAttentionMaskOperationUpdate},
    {"PadWithHiddenStateOperation", &PadWithHiddenStateOperationUpdate},
    {"RopeGradOperation", &RopeGradOperationUpdate},
    {"StridedBatchMatmulOperation", &StridedBatchMatmulOperationUpdate},
    {"UnpadWithHiddenStateOperation", &UnpadWithHiddenStateOperationUpdate},
    {"LaserAttentionOperation", &LaserAttentionOperationUpdate},
    {"LaserAttentionGradOperation", &LaserAttentionGradOperationUpdate},
    {"FillOperation", &FillOperationUpdate},
    {"SortOperation", &SortOperationUpdate},
    {"TopkToppSamplingOperation", &TopkToppSamplingOperationUpdate},
};

atb::Status UpdateOperationParam(const std::string &opName, const std::string &param, atb::Operation *operation)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);

    auto it = g_update_funcMap.find(opName);
    if (it == g_update_funcMap.end()) {
        ATB_LOG(ERROR) << "not support opName:" << opName;
        return atb::ERROR_INVALID_PARAM;
    }
    try {
        return it->second(paramJson, operation);
    } catch (const std::exception &e) {
        ATB_LOG(INFO) << "param is :" << param;
        ATB_LOG(ERROR) << opName << " parse json fail, error:" << e.what();
    }
    return atb::ERROR_INVALID_PARAM;
}
