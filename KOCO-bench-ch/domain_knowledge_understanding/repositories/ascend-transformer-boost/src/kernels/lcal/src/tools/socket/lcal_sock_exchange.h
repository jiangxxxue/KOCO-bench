/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCCL_SOCK_EXCHANGE_H
#define LCCL_SOCK_EXCHANGE_H

#include <vector>
#include <string>
#include <memory>
#include <securec.h>

#include <sys/socket.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <arpa/inet.h>

#include "mki/utils/log/log.h"

#include "lcal_types.h"
#include "lcal_api.h"

namespace Lcal {

union LcalSocketAddress {
    struct sockaddr sa;
    struct sockaddr_in sin;
    struct sockaddr_in6 sin6;
};

constexpr uint64_t LCAL_MAGIC = 0xdddd0000dddd0000;

struct LcalBootstrapHandle {
    uint64_t magic;
    union LcalSocketAddress addr;
};

union LcalBootstrap {
    LcalBootstrapHandle handle;
    LcalUniqueId uid;
};

int BootstrapGetUniqueId(LcalBootstrapHandle &handle, int commDomain);

class LcalSockExchange {
public:
    LcalSockExchange(int rank, int rankSize, std::vector<int> &rankList, int commDomain);
    LcalSockExchange(int rank, int rankSize, LcalUniqueId lcalCommId);
    ~LcalSockExchange();

    template <typename T> int AllGather(const T *sendBuf, size_t sendCount, T *recvBuf)
    {
        if (!isInit_ && Prepare() != LCAL_SUCCESS) {
            return LCAL_ERROR_INTERNAL;
        }
        isInit_ = true;

        if (!IsServer()) {
            return ClientSendRecv(sendBuf, sendCount, recvBuf);
        } else {
            return ServerRecvSend(sendBuf, sendCount, recvBuf);
        }
    }

    int GetNodeNum();

    static bool CheckValid(LcalUniqueId lcalCommId)
    {
        LcalBootstrap id {};
        id.uid = lcalCommId;
        return id.handle.magic == LCAL_MAGIC;
    }

private:
    void GetIpAndPort();
    int Prepare();
    int Listen();
    int Accept();
    int StartSecureTunnel();
    void Close(int &fd) const;
    int Connect();
    int AcceptConnection(int fd, sockaddr_in &clientAddr, socklen_t *sinSize) const;
    void Cleanup();
    bool IsServer() const;
    static bool CheckErrno(int ioErrno)
    {
        return ((ioErrno == EAGAIN) || (ioErrno == EWOULDBLOCK) || (ioErrno == EINTR));
    }

    template <typename T> int Send(int fd, const T *sendBuf, size_t sendSize, int flag) const
    {
        do {
            auto ret = send(fd, sendBuf, sendSize, flag);
            if (ret < 0) {
                if (CheckErrno(errno)) {
                    MKI_LOG(ERROR) << "send failed: " << strerror(errno);
                    continue;
                }
                MKI_LOG(DEBUG) << "Send failed: " << strerror(errno);
            }
            return ret;
        } while (true);
    }

    template <typename T> int Recv(int fd, T *recvBuf, size_t recvSize, int flag) const
    {
        do {
            auto ret = recv(fd, recvBuf, recvSize, flag);
            if (ret < 0) {
                if (CheckErrno(errno)) {
                    MKI_LOG(ERROR) << "recv failed: " << strerror(errno);
                    continue;
                }
                MKI_LOG(DEBUG) << "recv failed: " << strerror(errno);
            }
            return ret;
        } while (true);
    }

    template <typename T> int ClientSendRecv(const T *sendBuf, size_t sendSize, T *recvBuf)
    {
        if (Send(fd_, sendBuf, sendSize * sizeof(T), 0) <= 0) {
            MKI_LOG(ERROR) << "Client side " << rank_ << " send buffer failed";
            return LCAL_ERROR_INTERNAL;
        }

        if (Recv(fd_, recvBuf, sendSize * rankSize_ * sizeof(T), MSG_WAITALL) <= 0) {
            MKI_LOG(ERROR) << "Client side " << rank_ << " recv buffer failed ";
            return LCAL_ERROR_INTERNAL;
        }

        return LCAL_SUCCESS;
    }

    template <typename T> int ServerRecvSend(const T *sendBuf, size_t sendSize, T *recvBuf)
    {
        auto ret = memcpy_s(recvBuf, sendSize * sizeof (T), sendBuf, sendSize * sizeof (T));
        if (ret != EOK) {
            MKI_LOG(ERROR) << "Failed to copy sendBuf to recvBuf.";
            return LCAL_ERROR_INTERNAL;
        }

        for (int i = 1; i < rankSize_; ++i) {
            if (Recv(clientFds_[i], recvBuf + i * sendSize, sendSize * sizeof(T), MSG_WAITALL) <= 0) {
                MKI_LOG(ERROR) << "Server side recv rank " << i << " buffer failed";
                return LCAL_ERROR_INTERNAL;
            }
        }

        for (int i = 1; i < rankSize_; ++i) {
            if (Send(clientFds_[i], recvBuf, sendSize * rankSize_ * sizeof(T), 0) <= 0) {
                MKI_LOG(ERROR) << "Server side send rank " << i << " buffer failed";
                return LCAL_ERROR_INTERNAL;
            }
        }

        return LCAL_SUCCESS;
    }

    pid_t pid_ = 0;
    int rank_ = 0;
    int rankSize_ = 0;
    int fd_ = -1;
    std::vector<int> clientFds_ = {};
    bool isInit_ = false;
    std::vector<int> rankList_ = {};
    int commDomain_ = -1;
    std::string ip_ = "";
    uint16_t port_ = 0;
    LcalBootstrap lcalCommId_ = {};
};
}

#endif