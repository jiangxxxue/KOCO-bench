/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCAL_COMM_H
#define LCAL_COMM_H

#include <vector>
#include <string>

#include <hccl.h>
#include "lcal_types.h"
#include "lcal_api.h"
#include "comm_args.h"

namespace Lcal {
constexpr int IPC_NAME_SIZE = 65;
constexpr int SINGLE_MACHINE_910B2C_RANK_SIZE = 16;

class LcalSockExchange;
class LcalComm {
public:
    LcalComm(int rank, int rankSize);
    LcalComm(int rank, int rankSize, int bufferSize);
    LcalComm(int rank, int rankSize, int commDomain, int bufferSize, int isEnableMagic);
    LcalComm(int rank, int rankSize, LcalUniqueId commId);
    ~LcalComm();
    LcalComm(const LcalComm &) = delete;
    LcalComm &operator=(const LcalComm &) = delete;
    int Init();
    int InitThread(const std::string &uid = "default");
    int GetRank() const;
    int GetRankSize() const;
    int GetCommSize() const;
    int GetBufferSize() const;
    const PhysicalInfo &GetPhysicalInfo() const;
    GM_ADDR GetCommArgsPtr() const;
    CommArgs* GetCommArgs();
    std::string PrintDFX();
    friend class Lccl;
    friend class Lcoc;
    friend class LcclTest;

private:
    int SetMemoryName(std::string &name);
    int SetIpcPidSdid(std::string &name, const uint32_t *pids, const int64_t *sdids) const;
    int OpenIpcMem(const char names[LCAL_MAX_RANK_SIZE][IPC_NAME_SIZE]);
    int GetDev();
    int GetDevThread(const std::string &uid = "");
    int EnablePeerAccess();
    int InitCommMem();
    int InitCommon();
    void CloseIpcMem();
    void FreePeerMem(GM_ADDR &mem) const;
    int InitMem();
    int GetSidId(int64_t sdids[LCAL_MAX_RANK_SIZE], int rankSize);
    int GetPid(uint32_t *pids);
    int GetName(std::string &name, char names[LCAL_MAX_RANK_SIZE][IPC_NAME_SIZE]) const;
    int SyncCommArgs();
    int InitDumpAddr();

private:
    int rank_ = 0;  // global rank id
    int rankSize_ = 0;  // global rank size
    int commSize_ = 0;  // local LcalComm size
    int localRank_ = -1;
    int localRankSize_ = -1;
    int devId_ = 0;
    int64_t magic_ = 1;
    bool inited_ = false;
    bool ipcMemInited_ = false;
    std::string uid_ = {};
    std::vector<int> devList_ = {};
    std::vector<int> rankList_ = {};
    int commDomain_ = {};
    int bufferSize_ = LCAL_COMM_BUFFER_SIZE;

    // shared ping pong buff，这个地址就是一开始申请在HBM上的，所以host上可以取到，但不能直接修改。
    GM_ADDR peerMem_[LCAL_MAX_RANK_SIZE] = {};
    PhysicalInfo physicalInfo_ = {};
    CommArgs commArgs_ = {};    // host侧
    GM_ADDR commArgsPtr_ = nullptr; // device侧
    LcalUniqueId commId_ = {};
    LcalSockExchange *socketExchange_ = nullptr;
    bool deterministic_ = false;
    bool isEnableMsprofOp_ = false;
    bool isEnableMix_ = false;
};
} // Lcal

#endif // LCAL_COMM_H
