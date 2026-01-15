/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tbe_tiling_runner.h"

#include <securec.h>
#include <mutex>
#include <utility>

#include <mki/tensor.h>
#include <mki/types.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/env/env.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_configs.h>
#include <mki/utils/platform/platform_manager.h>
#include <mki/utils/rt/rt.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/params.h"
#include "compute_node_info.h"
#include "continuous_vector.h"
#include "graph/any_value.h"
#include "graph/ge_tensor.h"
#include "graph/debug/ge_util.h"
#include "kernel_run_context.h"
#include "kernel_run_context_builder.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry_base.h"
#include "runtime_attrs_def.h"

namespace {
const std::vector<std::string> CORE_TYPE_VEC {
    "AiCore",
    "AiCore",
    "VectorCore",
    "AiCore",
    "MIX",
};
}

namespace AsdOpsGeRt {
static std::unordered_map<std::string, KernelContextHolder> g_tilingParseCache;
std::mutex g_mutexTilingParseCache;

class AsdOpsFePlatformInfosManager {
public:
    static std::pair<bool, fe::PlatFormInfos> GetPlatFormInfos()
    {
        static std::once_flag initedFlag;
        static fe::PlatFormInfos platformInfo;
        static bool isSuccess;

        std::call_once(initedFlag, [&]() { isSuccess = InitPlatformInfo(platformInfo); });
        return std::make_pair(isSuccess, platformInfo);
    }

    static bool InitPlatformInfo(fe::PlatFormInfos &platformInfo)
    {
        const uint32_t maxLen = 100;
        std::string version;
        MKI_CHECK(Mki::MkiRtDeviceGetSocVersion(version, maxLen) == MKIRT_SUCCESS,
            "failed to get soc version", return false);
        std::string socVersion(version);
        MKI_LOG(DEBUG) << "tiling runner get soc version: " << socVersion;
        Mki::PlatformManager &platformManager = Mki::PlatformManager::Instance();
        MKI_CHECK(platformManager.InitializePlatformManager() == Mki::PLATFORM_SUCCESS,
            "failed to initialize platform manager", return false);
        Mki::PlatformConfigs platformConfigs;
        MKI_CHECK(platformManager.GetPlatformConfigs(socVersion, platformConfigs) == Mki::PLATFORM_SUCCESS,
            "failed to get platform information", return false);
        MKI_CHECK(platformInfo.Init(), "failed to initialize platformInfos", return false);
        AdaptPlatformInfos(platformInfo, platformConfigs);
        return true;
    }

    static void AdaptPlatformInfos(fe::PlatFormInfos &platformInfo, Mki::PlatformConfigs &platformConfigs)
    {
        std::map<std::string, std::map<std::string, std::string>> platformResMap = platformConfigs.GetPlatformSpecMap();

        platformInfo.SetFixPipeDtypeMap(platformConfigs.GetFixPipeDtypeMap());
        platformInfo.SetAICoreIntrinsicDtype(platformConfigs.GetAICoreIntrinsicDtype());
        platformInfo.SetVectorCoreIntrinsicDtype(platformConfigs.GetVectorCoreIntrinsicDtype());
        for (auto &[label, res]: platformResMap) {
            platformInfo.SetPlatformRes(label, res);
        }
    }
};

class TbeTilingRunnerImpl {
public:
    TbeTilingRunnerImpl() = default;
    ~TbeTilingRunnerImpl() = default;

    void SetName(const char *opType) { opType_ = opType; }

    void SetKernelName(const std::string kernelName) { kernelName_ = kernelName; }

    void AddInput(Mki::TensorDType dtype, Mki::TensorFormat format, const Mki::SVector<int64_t> &dims)
    {
        AddTensor(dtype, format, dims, inputs_);
    }

    void AddConstInput(Mki::TensorDType dtype, Mki::TensorFormat format,
                       const Mki::SVector<int64_t> &dims, const void *data, size_t size)
    {
        Shape shape;
        for (const auto dim : dims) {
            (void)shape.AppendDim(dim);
        }
        size_t totalSize = 0UL;
        auto tensorHolder = Tensor::CreateFollowing(GeDataType(dtype), size, totalSize);
        MKI_CHECK(tensorHolder != nullptr, "tensorHolder is nullptr", return);
        auto tensor = reinterpret_cast<Tensor *>(tensorHolder.get());
        if (memcpy_s(tensor->GetData<uint8_t>(), totalSize - sizeof(Tensor), data, size) != EOK) {
            MKI_LOG(ERROR) << "Failed to add const input";
            return;
        }
        tensor->MutableOriginShape() = shape;
        tensor->MutableStorageShape() = shape;
        tensor->SetDataType(GeDataType(dtype));
        tensor->SetStorageFormat(GeFormat(format));
        tensor->SetOriginFormat(GeFormat(format));
        contextComponent_.indexToTensors.emplace_back(inputs_.size(), std::move(tensorHolder));
        AddTensor(dtype, format, dims, inputs_);
    }

    void AddOutput(Mki::TensorDType dtype, Mki::TensorFormat format, const Mki::SVector<int64_t> &dims)
    {
        AddTensor(dtype, format, dims, outputs_);
    }

    void AddAttrBool(bool value)
    {
        auto data = ge::ComGraphMakeUnique<uint8_t[]>(sizeof(uint8_t));
        auto ret = memcpy_s(data.get(), sizeof(uint8_t), &value, sizeof(uint8_t));
        MKI_CHECK(ret == EOK, "failed to copy attr bool", return);
        attrs_.emplace_back(std::make_pair(std::move(data), sizeof(uint8_t)));
    }

    void AddAttrInt64(int64_t attr)
    {
        auto data = ge::ComGraphMakeUnique<uint8_t[]>(sizeof(int64_t));
        auto ret = memcpy_s(data.get(), sizeof(int64_t), &attr, sizeof(int64_t));
        MKI_CHECK(ret == EOK, "failed to copy attr int64", return);
        attrs_.emplace_back(std::make_pair(std::move(data), sizeof(int64_t)));
    }

    void AddAttrFloat(float attr)
    {
        auto data = ge::ComGraphMakeUnique<uint8_t[]>(sizeof(float));
        auto ret = memcpy_s(data.get(), sizeof(float), &attr, sizeof(float));
        MKI_CHECK(ret == EOK, "failed to copy attr float", return);
        attrs_.emplace_back(std::make_pair(std::move(data), sizeof(float)));
    }

    void AddAttrStr(const char *attr)
    {
        size_t dataLen = strlen(attr) + 1;
        auto data = ge::ComGraphMakeUnique<uint8_t[]>(dataLen);
        auto ret = memcpy_s(data.get(), dataLen, attr, dataLen);
        MKI_CHECK(ret == EOK, "failed to copy attr float", return);
        attrs_.emplace_back(std::make_pair(std::move(data), dataLen));
    }

    void AddAttrIntList(const int64_t *attr, const size_t num)
    {
        size_t totalSize = 0;
        auto data = ContinuousVector::Create<int64_t>(num, totalSize);
        MKI_CHECK(data != nullptr, "failed to create ContinuousVector", return);
        auto dataVec = reinterpret_cast<ContinuousVector *>(data.get());
        dataVec->SetSize(num);
        auto ret = memcpy_s(dataVec->MutableData(), sizeof(int64_t) * num, attr, sizeof(int64_t) * num);
        MKI_CHECK(ret == EOK, "failed to copy attr list int", return);
        attrs_.emplace_back(std::make_pair(std::move(data), totalSize));
    }

    Mki::Status GetTilingParseContextHolder(KernelContextHolder &tilingParseContextHolder,
        std::unique_ptr<uint8_t[]> &computeNode)
    {
        {
            std::lock_guard<std::mutex> lck(g_mutexTilingParseCache);
            auto it = g_tilingParseCache.find(kernelName_);
            if (it == g_tilingParseCache.end()) {
                MKI_LOG(DEBUG) << "tactic " << kernelName_ << " is first tiling parse";
                g_tilingParseCache[kernelName_] = BuildTilingParseContextHolder(computeNode);
                MKI_CHECK(g_tilingParseCache[kernelName_].context_ != nullptr,
                    "failed to build tiling parse context", return Mki::Status::FailStatus(1));
                MKI_CHECK((opImpl_->tiling_parse)(g_tilingParseCache[kernelName_].context_) == ge::GRAPH_SUCCESS,
                    "failed to run tiling parse", return Mki::Status::FailStatus(1));
            }
        }
        tilingParseContextHolder.context_ = g_tilingParseCache[kernelName_].context_;
        MKI_CHECK(((tilingParseContextHolder.context_)->GetOutputPointer<void **>(0)) != nullptr,
            "OutputPointer is nullptr", return Mki::Status::FailStatus(1));

        return Mki::Status::OkStatus();
    }

    Mki::Status GetTilingData(uint8_t *tilingData, uint64_t tilingDataLen, const BinHandle &binHandle)
    {
        MKI_CHECK(tilingData != nullptr, "tilingData invalid", return Mki::Status::FailStatus(1));
        MKI_CHECK(tilingDataLen != 0, "tilingData invalid", return Mki::Status::FailStatus(1));
        MKI_CHECK(InitKernelAttrs(binHandle), "failed to init tactic attrs", return Mki::Status::FailStatus(1));
        MKI_CHECK(InitPlatformInfo(), "failed to init platform info", return Mki::Status::FailStatus(1));
        opImpl_ = gert::OpImplRegistry::GetInstance().GetOpImpl(opType_);
        MKI_CHECK(opImpl_ != nullptr, "failed to find tiling entry", return Mki::Status::FailStatus(1));
        auto computeNodePtr = CreateComputeNode();
        MKI_CHECK(computeNodePtr != nullptr, "compute node is nullptr", return Mki::Status::FailStatus(1));

        KernelContextHolder tilingParseContextHolder;
        auto status = GetTilingParseContextHolder(tilingParseContextHolder, computeNodePtr);
        MKI_CHECK(status.Ok(), "failed to get tiling parse context", return Mki::Status::FailStatus(1));

        KernelContextHolder tilingContextHolder = BuildTilingContextHolder(computeNodePtr,
            *((tilingParseContextHolder.context_)->GetOutputPointer<void **>(0)), tilingDataLen);
        MKI_CHECK(tilingContextHolder.context_ != nullptr,
            "failed to build tiling context", return Mki::Status::FailStatus(1));

        auto tilingContext = reinterpret_cast<TilingContext *>(tilingContextHolder.context_);
        MKI_CHECK(opImpl_->tiling(tilingContext) == ge::GRAPH_SUCCESS,
            "failed to run tiling", return Mki::Status::FailStatus(1));

        auto rawTilingData = tilingContext->GetRawTilingData();
        MKI_CHECK(rawTilingData != nullptr, "failed to get rawtilingdata", return Mki::Status::FailStatus(1));
        auto ret = memcpy_s(tilingData, tilingDataLen, rawTilingData->GetData(), rawTilingData->GetDataSize());
        MKI_CHECK(ret == EOK, "failed to copy tilingdata", return Mki::Status::FailStatus(1));

        contextComponent_.blockDim = tilingContext->GetBlockDim();
        contextComponent_.tilingId = tilingContext->GetTilingKey();
        contextComponent_.tilingSize = rawTilingData->GetDataSize();

        return Mki::Status::OkStatus();
    }

    uint32_t GetBlockDim()
    {
        MKI_LOG(INFO) << kernelName_ << " BlockDim " << contextComponent_.blockDim;
        return contextComponent_.blockDim;
    }

    uint32_t GetIntercoreSync()
    {
        MKI_LOG(INFO) << kernelName_ << " IntercoreSync " << intercoreSync_;
        return intercoreSync_;
    }

    uint64_t GetTilingId()
    {
        MKI_LOG(INFO) << kernelName_ << " TilingId " << contextComponent_.tilingId;
        return contextComponent_.tilingId;
    }

    uint64_t GetTilingSize()
    {
        MKI_LOG(INFO) << kernelName_ << " TilingSize " << contextComponent_.tilingSize;
        return contextComponent_.tilingSize;
    }

    void GetWorkSpace(Mki::SVector<uint64_t, 8> &workspace) // 8 小容量SVECTOR
    {
        uint8_t *wksp = contextComponent_.workspaceSize.get();
        auto workspaceInfo = reinterpret_cast<gert::TypedContinuousVector<size_t> *>(wksp);
        MKI_CHECK(workspaceInfo && workspaceInfo->GetData(), "failed to get workspace info", return);

        size_t workspaceNum = workspaceInfo->GetSize();
        const size_t *workspaceSize = workspaceInfo->GetData();
        MKI_LOG(INFO) << kernelName_ << " workspace num " << workspaceNum;
        for (size_t i = 0; i < workspaceNum; i++) {
            size_t bufferSize = workspaceSize[i];
            MKI_LOG(DEBUG) << "size[" << i << "] " << bufferSize;
            workspace.push_back(bufferSize);
        }
    }

private:
    bool InitKernelAttrs(const BinHandle &binHandle)
    {
        // compileInfo
        compileInfo_ = binHandle.GetKernelCompileInfo();
        MKI_CHECK(compileInfo_ != nullptr, "compile info is nullptr", return false);
        // coreType
        int32_t coreTypeIdx = binHandle.GetKernelCoreType();
        MKI_CHECK(coreTypeIdx > -1, "core type is empty", return false);
        coreType_ = CORE_TYPE_VEC[coreTypeIdx];
        // intercoreSync
        intercoreSync_ = binHandle.GetIntercoreSync();
        // taskRatio
        cubeRatio_ = binHandle.GetCubeRatio();
        vectorRatio_ = binHandle.GetCubeRatio();

        return true;
    }

    bool InitPlatformInfo()
    {
        auto [isSuccess, platformInfo] = AsdOpsFePlatformInfosManager::GetPlatFormInfos();
        MKI_CHECK(isSuccess, "failed to get PlatFormInfos", return false);
        platformInfo_ = platformInfo;
        platformInfo_.SetCoreNumByCoreType(coreType_);

        if (coreType_ == "MIX" && (cubeRatio_ != 0 || vectorRatio_ != 0)) {
            uint32_t cubeCoreNum = platformInfo_.GetCoreNumByType("AiCore");
            uint32_t vectorCoreNum = platformInfo_.GetCoreNumByType("VectorCore");
            cubeCoreNum = (cubeRatio_ == 0) ? std::numeric_limits<uint32_t>::max() : (cubeCoreNum / cubeRatio_);
            vectorCoreNum = (vectorRatio_ == 0) ? std::numeric_limits<uint32_t>::max() : (vectorCoreNum / vectorRatio_);
            uint32_t coreNum = (cubeCoreNum < vectorCoreNum) ? cubeCoreNum : vectorCoreNum;
            if (coreNum == 0) {
                MKI_LOG(WARN) << "invalid coreNum for MIX with ratio: " << cubeRatio_ << ":" << vectorRatio_
                              << ", use 1 instead";
                coreNum = 1;
            }
            platformInfo_.SetCoreNum(coreNum);
        }

        return true;
    }

    void AddTensor(Mki::TensorDType dtype, Mki::TensorFormat format, const Mki::SVector<int64_t> &dims,
                   std::vector<std::pair<std::unique_ptr<ge::GeTensorDesc>, std::unique_ptr<Shape>>> &tensors) const
    {
        auto desc = ge::ComGraphMakeUnique<ge::GeTensorDesc>();
        MKI_CHECK(desc != nullptr, "desc is nullptr", return);
        desc->SetDataType(GeDataType(dtype));
        desc->SetFormat(GeFormat(format));
        desc->SetOriginFormat(GeFormat(format));
        auto shape = ge::ComGraphMakeUnique<Shape>();
        MKI_CHECK(shape != nullptr, "shape is nullptr", return);
        for (const auto dim : dims) {
            (void)shape->AppendDim(dim);
        }
        tensors.emplace_back(std::make_pair(std::move(desc), std::move(shape)));
    }

    ge::DataType GeDataType(Mki::TensorDType dtype) const
    {
        switch (dtype) {
            case Mki::TENSOR_DTYPE_FLOAT: return ge::DT_FLOAT;
            case Mki::TENSOR_DTYPE_FLOAT16: return ge::DT_FLOAT16;
            case Mki::TENSOR_DTYPE_INT8: return ge::DT_INT8;
            case Mki::TENSOR_DTYPE_INT32: return ge::DT_INT32;
            case Mki::TENSOR_DTYPE_UINT8: return ge::DT_UINT8;
            case Mki::TENSOR_DTYPE_INT16: return ge::DT_INT16;
            case Mki::TENSOR_DTYPE_UINT16: return ge::DT_UINT16;
            case Mki::TENSOR_DTYPE_UINT32: return ge::DT_UINT32;
            case Mki::TENSOR_DTYPE_INT64: return ge::DT_INT64;
            case Mki::TENSOR_DTYPE_UINT64: return ge::DT_UINT64;
            case Mki::TENSOR_DTYPE_DOUBLE: return ge::DT_DOUBLE;
            case Mki::TENSOR_DTYPE_BOOL: return ge::DT_BOOL;
            case Mki::TENSOR_DTYPE_STRING: return ge::DT_STRING;
            case Mki::TENSOR_DTYPE_COMPLEX64: return ge::DT_COMPLEX64;
            case Mki::TENSOR_DTYPE_COMPLEX128: return ge::DT_COMPLEX128;
            case Mki::TENSOR_DTYPE_BF16: return ge::DT_BF16;
            default:
                // ERROR LOG
                break;
        }
        return ge::DT_MAX;
    }

    ge::Format GeFormat(Mki::TensorFormat format) const
    {
        switch (format) {
            case Mki::TENSOR_FORMAT_NCHW: return ge::FORMAT_NCHW;
            case Mki::TENSOR_FORMAT_NHWC: return ge::FORMAT_NHWC;
            case Mki::TENSOR_FORMAT_ND: return ge::FORMAT_ND;
            case Mki::TENSOR_FORMAT_NC1HWC0: return ge::FORMAT_NC1HWC0;
            case Mki::TENSOR_FORMAT_FRACTAL_Z: return ge::FORMAT_FRACTAL_Z;
            case Mki::TENSOR_FORMAT_NC1HWC0_C04: return ge::FORMAT_NC1HWC0_C04;
            case Mki::TENSOR_FORMAT_HWCN: return ge::FORMAT_HWCN;
            case Mki::TENSOR_FORMAT_NDHWC: return ge::FORMAT_NDHWC;
            case Mki::TENSOR_FORMAT_FRACTAL_NZ: return ge::FORMAT_FRACTAL_NZ;
            case Mki::TENSOR_FORMAT_NCDHW: return ge::FORMAT_NCDHW;
            case Mki::TENSOR_FORMAT_NDC1HWC0: return ge::FORMAT_NDC1HWC0;
            case Mki::TENSOR_FORMAT_FRACTAL_Z_3D: return ge::FORMAT_FRACTAL_Z_3D;
            default:
                // ERROR LOG
                break;
        }
        return ge::FORMAT_MAX;
    }

    std::unique_ptr<uint8_t[]> CreateComputeNode()
    {
        size_t attrSize = sizeof(RuntimeAttrsDef);
        size_t attrNum = attrs_.size();
        attrSize += sizeof(size_t) * attrNum;
        for (size_t i = 0; i < attrNum; ++i) {
            attrSize += attrs_[i].second;
        }
        auto attrPtr = ge::ComGraphMakeUnique<uint8_t[]>(attrSize);
        MKI_CHECK(attrPtr != nullptr, "attrPtr is nullptr", return nullptr);
        auto attrsDef = reinterpret_cast<RuntimeAttrsDef *>(attrPtr.get());
        attrsDef->attr_num = attrNum;
        auto memret = memset_s(attrsDef->reserved_, sizeof(attrsDef->reserved_), 0, sizeof(attrsDef->reserved_));
        if (memret != EOK) {
            MKI_LOG(ERROR) << "Memset failed, result:" << memret;
            return nullptr;
        }
        size_t currentOffset = sizeof(RuntimeAttrsDef) + sizeof(size_t) * attrsDef->attr_num;
        auto attrPos = attrPtr.get();
        for (size_t i = 0; i < attrs_.size(); ++i) {
            attrsDef->offset[i] = currentOffset;
            auto ret = memcpy_s(attrPos + currentOffset, attrSize - currentOffset,
                                attrs_[i].first.get(), attrs_[i].second);
            if (ret != EOK) {
                MKI_LOG(ERROR) << "Failed to copy attr to AttrDef";
                return nullptr;
            }
            currentOffset += attrs_[i].second;
        }

        size_t inputNum = inputs_.size();
        size_t outputNum = outputs_.size();
        size_t computeNodeSize = sizeof(ComputeNodeInfo) + (inputNum + outputNum) * sizeof(CompileTimeTensorDesc);
        size_t totalSize = computeNodeSize + attrSize;
        auto computeNodePtr = ge::ComGraphMakeUnique<uint8_t[]>(totalSize);
        MKI_CHECK(computeNodePtr != nullptr, "computeNodePtr is nullptr", return nullptr);
        auto computeNodeDef = reinterpret_cast<ComputeNodeInfo *>(computeNodePtr.get());
        computeNodeDef->Init(0, inputNum, outputNum, opType_, opType_);
        for (size_t i = 0; i < inputNum; i++) {
            auto td = computeNodeDef->MutableInputTdInfo(i);
            MKI_CHECK(td != nullptr, "td is nullptr", return nullptr);
            td->SetDataType(inputs_[i].first->GetDataType());
            td->SetOriginFormat(inputs_[i].first->GetFormat());
            td->SetStorageFormat(inputs_[i].first->GetFormat());
        }
        for (size_t i = 0; i < outputNum; i++) {
            auto td = computeNodeDef->MutableOutputTdInfo(i);
            MKI_CHECK(td != nullptr, "td is nullptr", return nullptr);
            td->SetDataType(outputs_[i].first->GetDataType());
            td->SetOriginFormat(outputs_[i].first->GetFormat());
            td->SetStorageFormat(outputs_[i].first->GetFormat());
        }
        auto attr = computeNodeDef->MutableAttrs();
        const auto offset = ge::PtrToPtr<RuntimeAttrs, uint8_t>(attr) - computeNodePtr.get();
        auto ret = memcpy_s(ge::PtrToPtr<RuntimeAttrs, uint8_t>(attr), (totalSize - offset), attrPtr.get(), attrSize);
        if (ret != EOK) {
            MKI_LOG(ERROR) << "Failed to copy AttrDef to ComputeNode";
            return nullptr;
        }
        return computeNodePtr;
    }

    KernelContextHolder BuildTilingParseContextHolder(std::unique_ptr<uint8_t[]> &computeNode)
    {
        const size_t inputSize = 3; // TilingParse has 3 inputs
        const size_t outputSize = 1; // TilingParse has 1 output
        KernelContextHolder holder;
        size_t size = sizeof(KernelRunContext) + sizeof(Chain *) * (inputSize + outputSize);
        holder.context_holder_ = ge::ComGraphMakeUnique<uint8_t[]>(size);
        MKI_CHECK(holder.context_holder_ != nullptr, "context holder is nullptr", return holder);
        holder.context_ = ge::PtrToPtr<uint8_t, KernelContext>(holder.context_holder_.get());
        auto kernelRunContext = holder.context_->GetContext();
        kernelRunContext->input_size = inputSize;  // TilingParse has 3 inputs
        kernelRunContext->output_size = outputSize; // TilingParse has 1 output
        kernelRunContext->compute_node_info = ge::PtrToPtr<uint8_t, ComputeNodeInfo>(computeNode.get());
        kernelRunContext->output_start = &(kernelRunContext->values[kernelRunContext->input_size]);
        holder.value_holder_.resize(kernelRunContext->input_size + kernelRunContext->output_size);
        for (size_t i = 0UL; i < holder.value_holder_.size(); ++i) {
            kernelRunContext->values[i] = ge::PtrToPtr<Chain, AsyncAnyValue>(&holder.value_holder_[i]);
        }

        size_t i = 0;
        holder.value_holder_[i++].Set(const_cast<char *>(compileInfo_), nullptr);
        holder.value_holder_[i++].Set(reinterpret_cast<void *>(&platformInfo_), nullptr);
        holder.value_holder_[i++].Set(const_cast<char *>(opType_), nullptr);

        holder.value_holder_[i++].Set(opImpl_->compile_info_creator(), opImpl_->compile_info_deleter);

        return holder;
    }

    KernelContextHolder BuildTilingContextHolder(std::unique_ptr<uint8_t[]> &computeNode, void *compileInfo,
                                                 uint32_t tilingSize)
    {
        // prepare contextComponent_
        KernelContextHolder holder;
        size_t inputNum = inputs_.size();
        size_t outputNum = outputs_.size();
        for (size_t i = 0; i < inputNum; i++) {
            StorageShape storageShape;
            storageShape.MutableStorageShape() = *(inputs_[i].second);
            storageShape.MutableOriginShape() = *(inputs_[i].second);
            contextComponent_.storageShapes.emplace_back(storageShape);
        }
        for (size_t i = 0; i < outputNum; i++) {
            StorageShape storageShape;
            storageShape.MutableStorageShape() = *(outputs_[i].second);
            storageShape.MutableOriginShape() = *(outputs_[i].second);
            contextComponent_.storageShapes.emplace_back(storageShape);
        }

        contextComponent_.tilingData = TilingData::CreateCap(tilingSize);
        MKI_CHECK(contextComponent_.tilingData != nullptr, "tilingData is nullptr", return holder);
        contextComponent_.workspaceSize = ContinuousVector::Create<size_t>(kWorkspaceHolerSize_);
        MKI_CHECK(contextComponent_.workspaceSize != nullptr, "workspaceSize is nullptr", return holder);
        std::vector<void *> tilingContextInputs(contextComponent_.storageShapes.size() + kSize_, nullptr);
        for (size_t i = 0UL; i < contextComponent_.indexToTensors.size(); ++i) {
            tilingContextInputs[contextComponent_.indexToTensors[i].first] =
                reinterpret_cast<Tensor *>(contextComponent_.indexToTensors[i].second.get());
        }
        for (size_t i = 0UL; i < contextComponent_.storageShapes.size(); ++i) {
            if (tilingContextInputs[i] == nullptr) {
                tilingContextInputs[i] = &contextComponent_.storageShapes[i];
            }
        }
        tilingContextInputs[contextComponent_.storageShapes.size()] = compileInfo;
        tilingContextInputs[contextComponent_.storageShapes.size() + 1] = reinterpret_cast<void *>(&platformInfo_);

        // prepare kernelruncontext
        size_t contextSize = sizeof(KernelRunContext) + sizeof(Chain *) * (tilingContextInputs.size() + 5); // output 5
        holder.context_holder_ = ge::ComGraphMakeUnique<uint8_t[]>(contextSize);
        MKI_CHECK(holder.context_holder_ != nullptr, "context holder  is nullptr", return holder);
        holder.context_ = ge::PtrToPtr<uint8_t, KernelContext>(holder.context_holder_.get());
        auto kernelRunContext = holder.context_->GetContext();
        kernelRunContext->input_size = tilingContextInputs.size();
        kernelRunContext->output_size = 5; // TilingContext has 5 outputs
        kernelRunContext->compute_node_info = ge::PtrToPtr<uint8_t, ComputeNodeInfo>(computeNode.get());
        kernelRunContext->output_start = &(kernelRunContext->values[kernelRunContext->input_size]);
        holder.value_holder_.resize(kernelRunContext->input_size + kernelRunContext->output_size);
        for (size_t i = 0UL; i < holder.value_holder_.size(); ++i) {
            kernelRunContext->values[i] = ge::PtrToPtr<Chain, AsyncAnyValue>(&holder.value_holder_[i]);
        }
        for (size_t i = 0UL; i < tilingContextInputs.size(); ++i) {
            holder.value_holder_[i].Set(tilingContextInputs[i], nullptr);
        }

        size_t i = tilingContextInputs.size();
        holder.value_holder_[i++].Set(nullptr, nullptr);
        holder.value_holder_[i++].Set(nullptr, nullptr);
        holder.value_holder_[i++].Set(&contextComponent_.atomicFlag, nullptr);
        holder.value_holder_[i++].Set(contextComponent_.tilingData.get(), nullptr);
        holder.value_holder_[i++].Set(contextComponent_.workspaceSize.get(), nullptr);

        return holder;
    }

private:
    struct ContextComponent {
        std::vector<StorageShape> storageShapes;
        std::vector<std::pair<uint32_t, std::unique_ptr<uint8_t[]>>> indexToTensors;
        std::unique_ptr<uint8_t[]> tilingData;
        std::unique_ptr<uint8_t[]> workspaceSize;
        bool atomicFlag = true;
        // tiling extrainfo
        uint32_t blockDim = 0;
        uint64_t tilingId = 0;
        uint64_t tilingSize = 0;
    };

    const size_t kSize_ = 3UL;
    const size_t kWorkspaceHolerSize_ = 8UL;
    const char *opType_ = "DefaultImpl";
    const char *compileInfo_{nullptr};
    std::string kernelName_;
    std::string coreType_ = "";
    uint32_t intercoreSync_ = 0;
    uint32_t cubeRatio_ = 0;
    uint32_t vectorRatio_ = 0;
    fe::PlatFormInfos platformInfo_;
    const OpImplRegistry::OpImplFunctions *opImpl_ = nullptr;
    ContextComponent contextComponent_;
    std::vector<std::pair<std::unique_ptr<ge::GeTensorDesc>, std::unique_ptr<Shape>>> inputs_;
    std::vector<std::pair<std::unique_ptr<ge::GeTensorDesc>, std::unique_ptr<Shape>>> outputs_;
    std::vector<std::pair<std::unique_ptr<uint8_t[]>, size_t>> attrs_;
};

TbeTilingRunner::TbeTilingRunner() : impl_(std::make_shared<TbeTilingRunnerImpl>()) {}

TbeTilingRunner &TbeTilingRunner::SetName(const char *opType)
{
    impl_->SetName(opType);
    return *this;
}

TbeTilingRunner &TbeTilingRunner::SetKernelName(const std::string kernelName)
{
    impl_->SetKernelName(kernelName);
    return *this;
}

TbeTilingRunner &TbeTilingRunner::AddInput(Mki::TensorDType dtype, Mki::TensorFormat format,
                                           const Mki::SVector<int64_t> &dims)
{
    impl_->AddInput(dtype, format, dims);
    return *this;
}

TbeTilingRunner &TbeTilingRunner::AddConstInput(Mki::TensorDType dtype, Mki::TensorFormat format,
                                                std::initializer_list<int64_t> dims, const void *data, size_t size)
{
    Mki::SVector<int64_t> dims1(dims);
    impl_->AddConstInput(dtype, format, dims1, data, size);
    return *this;
}

TbeTilingRunner &TbeTilingRunner::AddConstInput(Mki::TensorDType dtype, Mki::TensorFormat format,
                                                const Mki::SVector<int64_t> &dims, const void *data, size_t size)
{
    impl_->AddConstInput(dtype, format, dims, data, size);
    return *this;
}

TbeTilingRunner &TbeTilingRunner::AddOutput(Mki::TensorDType dtype, Mki::TensorFormat format,
                                            const Mki::SVector<int64_t> &dims)
{
    impl_->AddOutput(dtype, format, dims);
    return *this;
}

TbeTilingRunner &TbeTilingRunner::AddAttrBool(bool value)
{
    impl_->AddAttrBool(value);
    return *this;
}

TbeTilingRunner &TbeTilingRunner::AddAttrInt(int32_t attr)
{
    impl_->AddAttrInt64(attr);
    return *this;
}

TbeTilingRunner &TbeTilingRunner::AddAttrStr(const char *attr)
{
    impl_->AddAttrStr(attr);
    return *this;
}

TbeTilingRunner &TbeTilingRunner::AddAttrInt64(int64_t attr)
{
    impl_->AddAttrInt64(attr);
    return *this;
}

TbeTilingRunner &TbeTilingRunner::AddAttrFloat(float attr)
{
    impl_->AddAttrFloat(attr);
    return *this;
}

TbeTilingRunner &TbeTilingRunner::AddAttrIntList(const int64_t *attr, const size_t num)
{
    impl_->AddAttrIntList(attr, num);
    return *this;
}

Mki::Status TbeTilingRunner::GetTilingData(uint8_t *tilingData, uint64_t tilingDataLen, const BinHandle &binHandle)
{
    return impl_->GetTilingData(tilingData, tilingDataLen, binHandle);
}

uint32_t TbeTilingRunner::GetBlockDim()
{
    return impl_->GetBlockDim();
}

uint32_t TbeTilingRunner::GetIntercoreSync()
{
    return impl_->GetIntercoreSync();
}

uint64_t TbeTilingRunner::GetTilingId()
{
    return impl_->GetTilingId();
}

uint64_t TbeTilingRunner::GetTilingSize()
{
    return impl_->GetTilingSize();
}

void TbeTilingRunner::GetWorkSpace(Mki::SVector<size_t, 8> &workspace) // 8 小容量SVECTOR
{
    impl_->GetWorkSpace(workspace);
}
} // namespace AsdOpsGeRt

namespace AsdOps {
Mki::Status GetTilingFromRunner(KernelInfo &kernelInfo, AsdOpsGeRt::TbeTilingRunner &runner, const BinHandle &binHandle)
{
    auto status = runner.GetTilingData(kernelInfo.GetTilingHostAddr(), kernelInfo.GetTilingSize(), binHandle);
    MKI_CHECK(status.Ok(), "failed to run tiling runner", return status);

    kernelInfo.SetBlockDim(runner.GetBlockDim());
    kernelInfo.SetTilingId(runner.GetTilingId());
    kernelInfo.SetTilingUsedSize(runner.GetTilingSize());
    if (runner.GetIntercoreSync() == 1) {
        kernelInfo.SetHwsyncIdx(0);
    }
    runner.GetWorkSpace(kernelInfo.GetScratchSizes());

    return Mki::Status::OkStatus();
}
} // namespace AsdOps
