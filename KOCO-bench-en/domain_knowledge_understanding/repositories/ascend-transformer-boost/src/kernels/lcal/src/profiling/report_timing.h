/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef REPORT_TIMING_H
#define REPORT_TIMING_H
#include <cstring>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#include <securec.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <toolchain/slog.h>
#include <toolchain/prof_api.h>
#include <toolchain/prof_common.h>
#include <hccl_types.h>
#include <mki/utils/log/log.h>

namespace Lcal {
class ReportTiming {
public:
    static constexpr uint64_t PROF_TASK_TIME_DUMP = 0x0000100000000ULL;
    ReportTiming() = delete;
    explicit ReportTiming(const char *opName, int commDomain, int64_t count = 0,
                          HcclDataType dataType = HCCL_DATA_TYPE_RESERVED)
        : opName_(opName), typeMix_(false), count_(count), dataType_(dataType)
    {
        InitProfiling(commDomain);
    }

    explicit ReportTiming(const char *opName, uint32_t blockDim)
        : opName_(opName), blockDim_(blockDim), typeMix_(true)
    {
        InitProfiling(0);
    }

    explicit ReportTiming(const char *opName, const int32_t rankId, const bool isReporting, uint8_t *dumpAddr,
        const aclrtStream stream) : opName_(opName), rankId_(rankId), isReporting_(isReporting),
        dumpAddr_(dumpAddr), stream_(stream)
    {
        moduleId_ = DUMP_MODULE_ID;
        InitProfiling(0);
    }

    ~ReportTiming()
    {
        MKI_LOG(DEBUG) << "ReportTiming " << __LINE__ << " ~ReportTiming() " <<
            " isReporting_:" << isReporting_ << " profEnable_:" << profEnable_;
        if (profEnable_ && isReporting_) {
            ReportMsprofData();
        }

        if (!isReporting_) {
            ProfilingStatus(RESET_STATUS);
        }
    }

    void InitProfiling(int commDomain)
    {
        if (ProfilingStatus() == -1) {
            ProfilingStatus(0);
            MKI_LOG(INFO) << "MsprofRegisterCallback start!";
            if (MsprofRegisterCallback(moduleId_, ProfHandle) != 0) {
                MKI_LOG(ERROR) << "MsprofRegisterCallback fail!";
            }
        }

        MKI_LOG(DEBUG) << "InitProfiling " << __LINE__ << " ProfilingStatus():" << ProfilingStatus() <<
            " isReporting_:" << isReporting_;
        if (ProfilingStatus() > 0) {
            ParamsInit(commDomain);
        }
        MKI_LOG(DEBUG) << "InitProfiling " << __LINE__ << " ProfilingStatus():" << ProfilingStatus() <<
            " isReporting_:" << isReporting_ << " profEnable_:" << profEnable_;
    }

    static int64_t ProfilingStatus(int64_t setValue = -1)
    {
        static int64_t profilingStatus = -1;
        if (setValue == RESET_STATUS) {
            profilingStatus = -1;
        } else if (setValue != -1) {
            profilingStatus = setValue;
        }
        return profilingStatus;
    }

    void ParamsInit(int commDomain)
    {
        profEnable_ = true;
        std::string groupName = std::to_string(commDomain);
        groupHash_ = MsprofGetHashId(groupName.c_str(), strlen(groupName.c_str()));

        std::string naStr = "NA";
        naHash_ = MsprofGetHashId(naStr.c_str(), strlen(naStr.c_str()));

        nameHash_ = MsprofGetHashId(opName_, strlen(opName_));
        beginTime_ = MsprofSysCycleTime();
    }

    void ReportMsprofData()
    {
        tid_ = GetCurrentThreadId();
        if (tid_ == -1) {
            MKI_LOG(ERROR) << "GetCurrentThreadId error!" << " name: " << opName_;
            return;
        }
        endTime_ = MsprofSysCycleTime();

        MKI_LOG(DEBUG) << "ReportMsprofData " << ProfilingStatus() << " dumpAddr_ is " <<
            (dumpAddr_ == nullptr ? "" : "not") << " nullptr ";

        if (ProfilingStatus() != PROF_TASK_TIME_DUMP || dumpAddr_ == nullptr) {
            CallMsprofReportHostNodeApi();
            CallMsprofReportHostLcclOpApi();
            CallMsprofReportHostLcclOpInfo();
            CallMsprofReportHostNodeBasicInfo();
            CallMsprofReportContextIdInfo();
        } else {
            CallMsprofReportDumpApi();
        }
    }

    void CallMsprofReportDumpApi() const
    {
        constexpr uint32_t dumpCoreCnt = 75;
        constexpr uint32_t dumpSizePerCore = 1 * 1024 * 1024;
        constexpr uint32_t dumpWorkspaceSize = dumpCoreCnt * dumpSizePerCore;

        MKI_LOG(DEBUG) << "LcclReporting dump rankId " << rankId_;
        uint8_t *devProfData = dumpAddr_;
        size_t profLen = dumpWorkspaceSize;

        std::vector<uint8_t> buffer(profLen, 0);
        int ret = 0;
        ret = aclrtMemcpyAsync(&buffer[0], profLen, devProfData, profLen, ACL_MEMCPY_DEVICE_TO_HOST, stream_);
        if (ret != 0) {
            MKI_LOG(ERROR) << "aclrtMemcpyAsync dump data failed";
        }
        ret = aclrtSynchronizeStream(stream_);
        if (ret != 0) {
            MKI_LOG(ERROR) << "aclrtSynchronizeStream dump data failed";
        }

        constexpr int32_t logLimit = 2;
        constexpr int32_t logFirstLimit = 10;
        constexpr int32_t profLevel = 3000;
        MsprofAdditionalInfo t;
        t.level = profLevel;
        t.type = 0;
        t.threadId = 0;
        t.dataLen = sizeof(LcclDumpLogInfo);
        t.timeStamp = 0;
        for (uint32_t coreId = 0; coreId < dumpCoreCnt; ++coreId) {
            LcclDumpUnion *u = reinterpret_cast<LcclDumpUnion *>(&buffer[coreId * dumpSizePerCore]);
            LcclDumpBlockInfo *b = &(u->blockInfo);
            LcclDumpLogInfo *l = &((u + 1)->logInfo);

            int32_t logLen = (dumpSizePerCore - b->dumpOffset) / sizeof(LcclDumpUnion) - 1;
            for (int32_t logInfoIdx = 0; logInfoIdx < logLen; ++logInfoIdx) {
                LcclDumpLogInfo *logInfo = l + logInfoIdx;
                auto ret = memcpy_s(t.data, sizeof(LcclDumpLogInfo), logInfo, sizeof(LcclDumpLogInfo));
                if (ret != 0) {
                    MKI_LOG(ERROR) << "LcclReporting report memcpy_s err " << ret;
                }
                if ((logInfoIdx < logLimit) || (logInfoIdx < logFirstLimit && rankId_ == 0 && coreId == 0)) {
                    MKI_LOG(DEBUG) << "LcclReporting report: rankId=" << rankId_ << ", coreId=" << coreId  <<
                        ", curLog=" << logInfoIdx << "/" << logLen <<
                        "; LcclDumpLogInfo: logId=" << logInfo->logId << ", blockId=" << logInfo->blockId <<
                        ", syscyc=" << logInfo->syscyc << ", curPc=" << logInfo->curPc  <<
                        ", operationType=" << logInfo->operationType;
                }
                MsprofReportAdditionalInfo(0, &t, sizeof(MsprofAdditionalInfo));
            }
        }
    }

    void CallMsprofReportHostNodeApi() const
    {
        MsprofApi reporterData{};
        reporterData.level = MSPROF_REPORT_NODE_LEVEL;
        reporterData.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
        reporterData.threadId = static_cast<uint32_t>(tid_);
        reporterData.beginTime = beginTime_;
        reporterData.endTime = endTime_;
        reporterData.itemId = nameHash_;

        auto ret = MsprofReportApi(true, &reporterData);
        if (ret != 0) {
            MKI_LOG(ERROR) << "CallMsprofReportHostNodeApi error! code: " << ret << " name: " << opName_;
        }
    }

    void CallMsprofReportHostLcclOpApi() const
    {
        if (typeMix_) {
            return;
        }
        MsprofApi reporterData{};
        reporterData.level = MSPROF_REPORT_HCCL_NODE_LEVEL;
        reporterData.type = MSPROF_REPORT_HCCL_MASTER_TYPE;
        reporterData.threadId = static_cast<uint32_t>(tid_);
        reporterData.beginTime = beginTime_;
        reporterData.endTime = endTime_;
        reporterData.itemId = nameHash_;

        auto ret = MsprofReportApi(true, &reporterData);
        if (ret != 0) {
            MKI_LOG(ERROR) << "CallMsprofReportHostLcclOpApi error! code: " << ret << " name: " << opName_;
        }
    }

    void CallMsprofReportHostLcclOpInfo() const
    {
        if (typeMix_) {
            return;
        }
        MsprofCompactInfo reporterData = {};
        reporterData.level = MSPROF_REPORT_NODE_LEVEL;
        reporterData.type = MSPROF_REPORT_NODE_HCCL_OP_INFO_TYPE;
        reporterData.threadId = static_cast<uint32_t>(tid_);
        reporterData.dataLen = sizeof(MsprofHCCLOPInfo);
        reporterData.timeStamp = beginTime_ + 1;

        reporterData.data.hcclopInfo.relay = 0;
        reporterData.data.hcclopInfo.retry = 0;
        reporterData.data.hcclopInfo.dataType = dataType_;
        reporterData.data.hcclopInfo.algType = naHash_;
        reporterData.data.hcclopInfo.count = count_;
        reporterData.data.hcclopInfo.groupName = groupHash_;

        auto ret = MsprofReportCompactInfo(static_cast<uint32_t>(true),
            static_cast<void *>(&reporterData), static_cast<uint32_t>(sizeof(MsprofCompactInfo)));
        if (ret != 0) {
            MKI_LOG(ERROR) << "CallMsprofReportHostLcclOpInfo error! code: " << ret << " name: " << opName_;
        }
    }

    void CallMsprofReportHostNodeBasicInfo() const
    {
        if (ProfilingStatus() == PROF_TASK_TIME_L0) {
            return;
        }
        MsprofCompactInfo reporterData{};

        reporterData.level = MSPROF_REPORT_NODE_LEVEL;
        reporterData.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
        reporterData.threadId = static_cast<uint32_t>(tid_);
        reporterData.dataLen = sizeof(MsprofNodeBasicInfo);
        reporterData.timeStamp = endTime_;

        reporterData.data.nodeBasicInfo.opName = nameHash_;
        reporterData.data.nodeBasicInfo.opType = nameHash_;
        reporterData.data.nodeBasicInfo.blockDim = ((blockDim_ & 0x0000FFFU) | 0x20000U);

        auto ret = MsprofReportCompactInfo(static_cast<uint32_t>(true),
                                           static_cast<void *>(&reporterData),
                                           static_cast<uint32_t>(sizeof(MsprofCompactInfo)));
        if (ret != 0) {
            MKI_LOG(ERROR) << "CallMsprofReportHostNodeBasicInfo error! code: " << ret << " name: " << opName_;
        }
    }

    void CallMsprofReportContextIdInfo() const
    {
        if (!typeMix_) {
            return;
        }

        MsprofAdditionalInfo additionalInfo = {};
        additionalInfo.magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
        additionalInfo.level = MSPROF_REPORT_NODE_LEVEL;
        additionalInfo.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
        additionalInfo.timeStamp = beginTime_ + 1;
        additionalInfo.threadId = static_cast<uint32_t>(tid_);
        additionalInfo.dataLen = sizeof(MsprofContextIdInfo);

        MsprofContextIdInfo info = {};
        info.opName = nameHash_;
        info.ctxIdNum = 1;
        info.ctxIds[0] = 0;

        int ret = memcpy_s(additionalInfo.data, MSPROF_ADDTIONAL_INFO_DATA_LENGTH, &info, sizeof(MsprofContextIdInfo));
        MKI_LOG_IF(ret != EOK, ERROR) << "memcpy_s Error! Error Code: " << ret;

        auto retReport = MsprofReportAdditionalInfo(static_cast<uint32_t>(true),
                                                    static_cast<void *>(&additionalInfo),
                                                    static_cast<uint32_t>(sizeof(MsprofAdditionalInfo)));
        if (retReport != 0) {
            MKI_LOG(ERROR) << "ProfReportAdditionalInfo error!" << " name: " << opName_;
        }
    }

    static int32_t GetCurrentThreadId()
    {
        int32_t tid = static_cast<int32_t>(syscall(SYS_gettid));
        if (tid == -1) {
            MKI_LOG(ERROR) << "get tid failed, errno: " << errno;
        }
        return tid;
    }

    static int32_t ProfHandle(uint32_t type, void *data, uint32_t len)
    {
        if (data == nullptr) {
            MKI_LOG(ERROR) << "ProfHandle failed! data is nullptr!";
            return -1;
        }
        if (type != PROF_CTRL_SWITCH) {
            MKI_LOG(ERROR) << "ProfHandle failed! ProfCtrlType is not correct!";
            return -1;
        }
        if (len < sizeof(MsprofCommandHandle)) {
            MKI_LOG(ERROR) << "ProfHandle failed! dataSize is not correct!";
            return -1;
        }
        MsprofCommandHandle *profilerConfig = static_cast<MsprofCommandHandle*>(data);
        const uint32_t profType = profilerConfig->type;
        const uint64_t profSwitch = profilerConfig->profSwitch;
        if (profType == PROF_COMMANDHANDLE_TYPE_START) {
            MKI_LOG(INFO) << "Open Profiling Switch " << std::hex << profSwitch << std::dec;
            if ((profSwitch & PROF_TASK_TIME_L0) != PROF_CTRL_INVALID) {
                ProfilingStatus(PROF_TASK_TIME_L0);
                MKI_LOG(DEBUG) << "Profiling Level0 Enable";
            }
            if ((profSwitch & PROF_TASK_TIME_L1) != PROF_CTRL_INVALID) {
                ProfilingStatus(PROF_TASK_TIME_L1);
                MKI_LOG(DEBUG) << "Profiling Level1 Enable";
            }
            if ((profSwitch & PROF_TASK_TIME_DUMP) != PROF_CTRL_INVALID) {
                ProfilingStatus(PROF_TASK_TIME_DUMP);
                MKI_LOG(DEBUG) << "Profiling dump Enable";
            }
        }
        if (profType == PROF_COMMANDHANDLE_TYPE_STOP) {
            MKI_LOG(INFO) << "Close Profiling Switch";
            ProfilingStatus(0);
        }
        return 0;
    }

private:
    static constexpr uint64_t PROF_TASK_TIME_L0 = 0x00000800ULL;
    static constexpr uint64_t PROF_TASK_TIME_L1 = 0x00000002ULL;
    static constexpr int32_t DUMP_MODULE_ID = 61;
    static constexpr int32_t RESET_STATUS = -2;
    uint64_t beginTime_ = 0;
    uint64_t endTime_ = 0;
    const char *opName_ = nullptr;
    uint32_t blockDim_ = 0;
    uint64_t nameHash_ = 0;
    uint64_t groupHash_ = 0;
    uint64_t naHash_ = 0;
    bool typeMix_ = false;
    long tid_ = 0;
    bool profEnable_ = false;
    int64_t count_ = 0;
    uint8_t dataType_ = HCCL_DATA_TYPE_RESERVED;
    int32_t rankId_ = 0;
    bool isReporting_ = true;
    uint8_t *dumpAddr_ = nullptr;
    aclrtStream stream_ = nullptr;
    int32_t moduleId_ = INVLID_MOUDLE_ID;
};
}
#endif