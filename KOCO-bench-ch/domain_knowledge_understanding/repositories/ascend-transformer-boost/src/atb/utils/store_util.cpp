/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/store_util.h"
#include <sys/stat.h>
#include <acl/acl_rt.h>
#include <mki/launch_param.h>
#include <mki/utils/file_system/file_system.h>
#include "atb/utils/log.h"
#include "atb/utils/config.h"
#include "atb/utils/probe.h"
#include "atb/utils/disk_util.h"
#include "atb/utils/tensor_util.h"

namespace atb {
static const char *TENSOR_FILE_NAME_EXT = ".bin";

void StoreUtil::WriteFile(const uint8_t *data, uint64_t dataLen, const std::string &filePath)
{
    if (data == nullptr) {
        ATB_LOG(ERROR) << "WriteFile error! data is nullptr!";
        return;
    }
    std::string dirPath = Mki::FileSystem::DirName(filePath);
    if (dirPath.empty()) {
        ATB_LOG(ERROR) << "get dir name fail, path:" << filePath;
        return;
    }
    if (!Mki::FileSystem::Exists(dirPath)) {
        if (!Mki::FileSystem::Makedirs(dirPath, S_IRWXU | S_IRGRP | S_IXGRP)) {
            ATB_LOG(ERROR) << "Make dir fail, path:" << dirPath;
            return;
        }
    }

    if (!DiskUtil::IsDiskAvailable(dirPath)) {
        ATB_LOG(WARN) << "not write " << filePath << " for disk not vailable";
        return;
    }

    bool ret = Mki::FileSystem::WriteFile(reinterpret_cast<const char *>(data), dataLen, filePath);
    if (!ret) {
        ATB_LOG(ERROR) << "write file fail, path:" << filePath;
        return;
    }
    ATB_LOG(INFO) << "write filePath:" << filePath << " success";
}

void StoreUtil::SaveVariantPack(aclrtStream stream, const VariantPack &variantPack, const std::string &dirPath)
{
    if (Probe::IsSaveTensorData()) {
        int ret = aclrtSynchronizeStream(stream);
        ATB_LOG_IF(ret != 0, ERROR) << "aclrtSynchronizeStream fail, ret:" << ret;
    }

    for (size_t i = 0; i < variantPack.inTensors.size(); ++i) {
        std::string fileName = "intensor" + std::to_string(i) + TENSOR_FILE_NAME_EXT;
        std::string filePath = Mki::FileSystem::Join({dirPath, fileName});
        SaveTensor(variantPack.inTensors.at(i), filePath);
    }

    for (size_t i = 0; i < variantPack.outTensors.size(); ++i) {
        std::string fileName = "outtensor" + std::to_string(i) + TENSOR_FILE_NAME_EXT;
        std::string filePath = Mki::FileSystem::Join({dirPath, fileName});
        SaveTensor(variantPack.outTensors.at(i), filePath);
    }
}

void StoreUtil::SaveVariantPack(aclrtStream stream, const RunnerVariantPack &runnerVariantPack,
                                const std::string &dirPath)
{
    if (Probe::IsSaveTensorData()) {
        int ret = aclrtSynchronizeStream(stream);
        ATB_LOG_IF(ret != 0, ERROR) << "aclrtSynchronizeStream fail, ret:" << ret;
    }

    for (size_t i = 0; i < runnerVariantPack.inTensors.size(); ++i) {
        std::string fileName = "intensor" + std::to_string(i) + TENSOR_FILE_NAME_EXT;
        std::string filePath = Mki::FileSystem::Join({dirPath, fileName});
        SaveTensor(runnerVariantPack.inTensors.at(i), filePath);
    }

    for (size_t i = 0; i < runnerVariantPack.outTensors.size(); ++i) {
        std::string fileName = "outtensor" + std::to_string(i) + TENSOR_FILE_NAME_EXT;
        std::string filePath = Mki::FileSystem::Join({dirPath, fileName});
        SaveTensor(runnerVariantPack.outTensors.at(i), filePath);
    }
}

void StoreUtil::SaveLaunchParam(aclrtStream stream, const Mki::LaunchParam &launchParam, const std::string &dirPath)
{
    if (Probe::IsSaveTensorData()) {
        int ret = aclrtSynchronizeStream(stream);
        ATB_LOG_IF(ret != 0, ERROR) << "aclrtSynchronizeStream fail, ret:" << ret;
    }

    for (size_t i = 0; i < launchParam.GetInTensorCount(); ++i) {
        std::string fileName = "intensor" + std::to_string(i) + TENSOR_FILE_NAME_EXT;
        std::string filePath = Mki::FileSystem::Join({dirPath, fileName});
        SaveTensor(launchParam.GetInTensor(i), filePath);
    }

    for (size_t i = 0; i < launchParam.GetOutTensorCount(); ++i) {
        std::string fileName = "outtensor" + std::to_string(i) + TENSOR_FILE_NAME_EXT;
        std::string filePath = Mki::FileSystem::Join({dirPath, fileName});
        SaveTensor(launchParam.GetOutTensor(i), filePath);
    }
}

void StoreUtil::SaveTensor(const Mki::Tensor &tensor, const std::string &filePath)
{
    try {
        std::vector<char> hostData(tensor.dataSize);
        int ret =
            aclrtMemcpy(hostData.data(), tensor.dataSize, tensor.data, tensor.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << "aclrtMemcpy failed! ret:" << ret;
            return;
        }
        Probe::SaveTensor(std::to_string(tensor.desc.format), std::to_string(tensor.desc.dtype),
                          TensorUtil::AsdOpsDimsToString(tensor.desc.dims), hostData.data(), tensor.dataSize, filePath);
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) << "save tensor throw an error: " << e.what();
        return;
    }
}

void StoreUtil::SaveTensor(const Tensor &tensor, const std::string &filePath)
{
    try {
        std::vector<char> hostData(tensor.dataSize);
        int ret = aclrtMemcpy(hostData.data(), tensor.dataSize, tensor.deviceData, tensor.dataSize,
                              ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << "aclrtMemcpy failed! ret:" << ret;
            return;
        }
        Probe::SaveTensor(std::to_string(tensor.desc.format), std::to_string(tensor.desc.dtype),
                          TensorUtil::ShapeToString(tensor.desc.shape), hostData.data(), tensor.dataSize, filePath);
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) << "save tensor throw an error: " << e.what();
        return;
    }
}

void StoreUtil::CopyTensorData(Mki::BinFile &binFile, const void *deviceData, uint64_t dataSize)
{
    try {
        std::vector<char> hostData(dataSize);
        int st = aclrtMemcpy(hostData.data(), dataSize, deviceData, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
        ATB_LOG_IF(st != 0, ERROR) << "aclrtMemcpy device to host fail for save atbTensor, ret:" << st;
        Mki::Status ret = binFile.AddObject("data", hostData.data(), dataSize);
        if (!ret.Ok()) {
            ATB_LOG(ERROR) << "copy tensor data to binfile error, " << ret.ToString();
            return;
        }
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) << "save tensor throw an error: " << e.what();
        return;
    }
}

void StoreUtil::SaveTensor(const std::string &format, const std::string &dtype, const std::string &dims,
                           const void *deviceData, uint64_t dataSize, const std::string &filePath)
{
    if (!DiskUtil::IsDiskAvailable(filePath)) {
        ATB_LOG(WARN) << "not write " << filePath << " for disk not vailable";
        return;
    }

    Mki::BinFile binFile;
    binFile.AddAttr("format", format);
    binFile.AddAttr("dtype", dtype);
    binFile.AddAttr("dims", dims);
    if (deviceData) {
        CopyTensorData(binFile, deviceData, dataSize);
    } else {
        ATB_LOG(WARN) << "save atbTensor " << filePath << " data is empty";
    }

    Mki::Status st = binFile.Write(filePath);
    if (st.Ok()) {
        ATB_LOG(INFO) << "save tensor success, filePath:" << filePath;
    } else {
        ATB_LOG(ERROR) << "save tensor fail, filePath:" << filePath;
    }
}
} // namespace atb