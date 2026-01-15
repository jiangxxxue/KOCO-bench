/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "lcal_sock_exchange.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <csignal>
#include <cerrno>
#include <cstring>
#include <set>
#include <string>
#include <fstream>
#include <sstream>

#include <sys/types.h>
#include <sys/socket.h>

#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#include <mki/utils/env/env.h>

using namespace std;
namespace Lcal {
const string LCAL_LOCAL_SOCK_IP = "127.0.0.1";
constexpr uint16_t LCAL_DEFAULT_SOCK_PORT = 10067;
constexpr uint32_t LCAL_MAX_BACK_LOG = 65535;

int ParseIpAndPort(const char* input, string &ip, uint16_t &port)
{
    if (input == nullptr) {
        return LCAL_INVALID_VALUE;
    }
    string inputStr(input);
    size_t colonPos = inputStr.find(':');
    if (colonPos == string::npos) {
        MKI_LOG(ERROR) << "Input string does not contain a colon separating IP and port.";
        return LCAL_ERROR_INTERNAL;
    }

    ip = inputStr.substr(0, colonPos);
    std::string portStr = inputStr.substr(colonPos + 1);

    std::istringstream portStream(portStr);
    portStream >> port;
    if (portStream.fail() || portStream.bad()) {
        MKI_LOG(ERROR) << "Invalid port number.";
        return LCAL_ERROR_INTERNAL;
    }
    return LCAL_SUCCESS;
}

LcalSockExchange::~LcalSockExchange()
{
    Cleanup();
}

LcalSockExchange::LcalSockExchange(int rank, int rankSize, std::vector<int> &rankList, int commDomain)
    : rank_(rank), rankSize_(rankSize), rankList_(rankList), commDomain_(commDomain)
{
}

LcalSockExchange::LcalSockExchange(int rank, int rankSize, LcalUniqueId lcalCommId)
    : rank_(rank), rankSize_(rankSize)
{
    lcalCommId_.uid = lcalCommId;
}

int LcalSockExchange::GetNodeNum()
{
    if (!isInit_ && Prepare() != LCAL_SUCCESS) {
        return LCAL_ERROR_INTERNAL;
    }
    isInit_ = true;
    const string filePath = "/proc/sys/kernel/random/boot_id";
    ifstream fileStream(filePath);
    stringstream buffer;
    if (fileStream) {
        buffer << fileStream.rdbuf();
        fileStream.close();
    }
    const std::string uuid = buffer.str();
    MKI_LOG(DEBUG) << "rank:" << rank_ << " UUID " << uuid;

    set<string> uuidSet {};
    uuidSet.insert(uuid);
    int nodeNum = -1;
    if (IsServer()) {
        for (int i = 1; i < rankSize_; ++i) {
            if (Recv(clientFds_[i], const_cast<__caddr_t>(uuid.data()), uuid.size(), 0) <= 0) {
                MKI_LOG(ERROR) << "Server side recv rank " << i << " buffer failed";
                return LCAL_ERROR_INTERNAL;
            }
            uuidSet.insert(uuid);
        }
        nodeNum = static_cast<int>(uuidSet.size());
        for (int i = 1; i < rankSize_; ++i) {
            if (Send(clientFds_[i], &nodeNum, sizeof(int), 0) <= 0) {
                MKI_LOG(ERROR) << "Server side send rank " << i << " buffer failed";
                return LCAL_ERROR_INTERNAL;
            }
        }
    } else {
        if (Send(fd_, uuid.data(), uuid.size(), 0) <= 0) {
            MKI_LOG(ERROR) << "Client side " << rank_ << " send buffer failed";
            return LCAL_ERROR_INTERNAL;
        }
        if (Recv(fd_, &nodeNum, sizeof(int), 0) <= 0) {
            MKI_LOG(ERROR) << "Client side " << rank_ << " recv buffer failed ";
            return LCAL_ERROR_INTERNAL;
        }
    }
    return nodeNum;
}

void LcalSockExchange::GetIpAndPort()
{
    const char* env = Mki::GetEnv("LCAL_COMM_ID");

    if (env == nullptr or ParseIpAndPort(env, ip_, port_) != LCAL_SUCCESS) {
        ip_ = LCAL_LOCAL_SOCK_IP;
        port_ = LCAL_DEFAULT_SOCK_PORT;
    }
    port_ += commDomain_;
    lcalCommId_.handle.addr.sin.sin_family = AF_INET;
    lcalCommId_.handle.addr.sin.sin_addr.s_addr = inet_addr(LCAL_LOCAL_SOCK_IP.c_str());
    lcalCommId_.handle.addr.sin.sin_port = htons(port_);
    MKI_LOG(DEBUG) << "curRank: " << rank_ << " commDomain: " << commDomain_ << " ip: " << ip_ << " port: " << port_;
}

int LcalSockExchange::Prepare()
{
    if (lcalCommId_.handle.magic != LCAL_MAGIC) {
        GetIpAndPort();
    }
    if (!IsServer()) {
        if (ip_ != LCAL_LOCAL_SOCK_IP) {
            MKI_LOG(ERROR) << "Multi-machine is not supported at the moment";
            return LCAL_ERROR_INTERNAL;
        }
        return Connect();
    }

    clientFds_.resize(rankSize_, -1);
    if (Listen() != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "Listen Failed!";
        return LCAL_ERROR_INTERNAL;
    }

    if (Accept() != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "Accept Failed!";
        return LCAL_ERROR_INTERNAL;
    }

    return LCAL_SUCCESS;
}

int LcalSockExchange::Listen()
{
    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) {
        MKI_LOG(ERROR) << "Server side create socket failed";
        return LCAL_ERROR_INTERNAL;
    }

    int reuse = 1;
    if (setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(int)) < 0) {
        MKI_LOG(ERROR) << "Server side set reuseaddr failed";
        return LCAL_ERROR_INTERNAL;
    }

    struct sockaddr *addrPtr = &lcalCommId_.handle.addr.sa;
    if (bind(fd_, addrPtr, sizeof(struct sockaddr)) < 0) {
        MKI_LOG(ERROR) << "Server side bind " << ntohs(lcalCommId_.handle.addr.sin.sin_port) << " failed";
        return LCAL_ERROR_INTERNAL;
    }

    if (listen(fd_, LCAL_MAX_BACK_LOG) < 0) {
        MKI_LOG(ERROR) << "Server side listen " << ntohs(lcalCommId_.handle.addr.sin.sin_port) << " failed";
        return LCAL_ERROR_INTERNAL;
    }
    MKI_LOG(INFO) << "The server is listening! ip: "<< inet_ntoa(lcalCommId_.handle.addr.sin.sin_addr)
        << " port: " <<  ntohs(lcalCommId_.handle.addr.sin.sin_port);

    return LCAL_SUCCESS;
}

int LcalSockExchange::AcceptConnection(int fd, sockaddr_in& clientAddr, socklen_t *sinSize) const
{
    int clientFd;
    LcalSocketAddress clientAddrPtr;
    clientAddrPtr.sin = clientAddr;

    do {
        clientFd = accept(fd, &clientAddrPtr.sa, sinSize);
        if (clientFd < 0) {
            if (!CheckErrno(errno)) {
                MKI_LOG(ERROR) << "Server side accept failed" << strerror(errno);
                return -1;
            }
            MKI_LOG(DEBUG) << "accept failed: " << strerror(errno);
            continue;
        }
        break;
    } while (true);

    return clientFd;
}

int LcalSockExchange::Accept()
{
    struct sockaddr_in clientAddr;
    socklen_t sinSize = sizeof(struct sockaddr_in);

    for (int i = 1; i < rankSize_; ++i) {
        int fd = AcceptConnection(fd_, clientAddr, &sinSize);
        if (fd < 0) {
            MKI_LOG(ERROR) << "AcceptConnection failed";
            return LCAL_ERROR_INTERNAL;
        }

        int rank = 0;
        if (Recv(fd, &rank, sizeof(rank), 0) <= 0) {
            MKI_LOG(ERROR) << "Server side recv rank id failed";
            return LCAL_ERROR_INTERNAL;
        }

        if (rank >= rankSize_ || rank <= 0 || clientFds_[rank] >= 0) {
            MKI_LOG(ERROR) << "Server side recv invalid rank id " << rank;
            return LCAL_ERROR_INTERNAL;
        }

        MKI_LOG(DEBUG) << "Server side recv rank id " << rank;
        clientFds_[rank] = fd;
    }

    return LCAL_SUCCESS;
}

void LcalSockExchange::Close(int &fd) const
{
    if (fd == -1) {
        return;
    }

    if (close(fd) < 0) {
        MKI_LOG(WARN) << "failed to close fd:" << fd;
        return;
    }

    fd = -1;
}

int LcalSockExchange::Connect()
{
    MKI_LOG(DEBUG) << "Client side " << rank_ << " begin to connect";

    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) {
        MKI_LOG(ERROR) << "Client side " << rank_ << " create socket failed";
        return LCAL_ERROR_INTERNAL;
    }

    int sleepTimeS = 1;
    int maxRetryCount = 1800; // 增加retry时间为30分钟
    int retryCount = 0;
    bool success = false;
    struct sockaddr *addrPtr = &lcalCommId_.handle.addr.sa;
    while (retryCount < maxRetryCount) {
        if (connect(fd_, addrPtr, sizeof(struct sockaddr)) < 0) {
            if (errno == ECONNREFUSED) {
                MKI_LOG(DEBUG) << "Client side " << rank_ << " try connect " << (retryCount + 1) << " times refused";
                retryCount++;
                sleep(sleepTimeS);
                continue;
            }
            if (errno != EINTR) {
                MKI_LOG(ERROR) << "Client side " << rank_ << " connect failed: " << strerror(errno);
                break;
            }
            MKI_LOG(DEBUG) << "Client side " << rank_ << " try connect failed: " << strerror(errno);
            continue;
        }
        success = true;
        break;
    }

    if (!success) {
        MKI_LOG(ERROR) << "Client side " << rank_ << " connect failed";
        return LCAL_ERROR_INTERNAL;
    }

    if (Send(fd_, &rank_, sizeof(rank_), 0) <= 0) {
        MKI_LOG(ERROR) << "Client side " << rank_ << " send rank failed";
        return LCAL_ERROR_INTERNAL;
    }

    return LCAL_SUCCESS;
}

bool LcalSockExchange::IsServer() const
{
    return rank_ == 0;
}

void LcalSockExchange::Cleanup()
{
    if (fd_ >= 0) {
        Close(fd_);
    }

    if (clientFds_.empty()) {
        return;
    }

    for (int i = 1; i < rankSize_; ++i) {
        if (clientFds_[i] >= 0) {
            Close(clientFds_[i]);
        }
    }
    if (pid_ > 0) {
        kill(pid_, SIGINT);
        int status;
        waitpid(pid_, &status, 0);
        MKI_LOG(DEBUG) << "child process resources cleaned up";
    }
}

int GetAddrFromString(LcalSocketAddress* ua, const char* ipPortPair)
{
    std::string ip;
    uint16_t port;
    int ret = ParseIpAndPort(ipPortPair, ip, port);
    if (ret != LCAL_SUCCESS) {
        MKI_LOG(ERROR) << "lcal ParseIpAndPort failed!";
        return LCAL_ERROR_INTERNAL;
    }
    ua->sin.sin_family = AF_INET;
    ua->sin.sin_addr.s_addr = inet_addr(ip.c_str());
    ua->sin.sin_port = htons(port);
    return LCAL_SUCCESS;
}

int BootstrapGetServerIp(LcalSocketAddress& handle)
{
    char hostname[256];

    if (gethostname(hostname, sizeof(hostname)) < 0) {
        MKI_LOG(ERROR) << "ERROR: Failed to get hostname.";
        return LCAL_ERROR_INTERNAL;
    }

    struct hostent *hostEntry = gethostbyname(hostname);
    if (hostEntry == nullptr) {
        MKI_LOG(ERROR) << "ERROR: Failed to get host entry." ;
        return LCAL_ERROR_INTERNAL;
    }

    const char* ip = inet_ntoa(*reinterpret_cast<struct in_addr*>(hostEntry->h_addr_list[0]));
    if (ip == nullptr) {
        MKI_LOG(ERROR) << "ERROR: Failed to convert IP address.";
        return LCAL_ERROR_INTERNAL;
    }

    auto ret = memset_s(&handle, sizeof(handle), 0, sizeof(handle));
    if (ret != EOK) {
        MKI_LOG(ERROR) << "Failed to memset_s handle in BootstrapGetServerIp.";
        return LCAL_ERROR_INTERNAL;
    }
    handle.sin.sin_family = AF_INET;
    handle.sin.sin_addr.s_addr = inet_addr(ip);
    handle.sin.sin_port = 0;

    return LCAL_SUCCESS;
}

int BootstrapGetUniqueId(struct LcalBootstrapHandle& handle, int commDomain)
{
    auto ret = memset_s(&handle, sizeof(LcalBootstrapHandle), 0, sizeof(LcalBootstrapHandle));
    if (ret != EOK) {
        MKI_LOG(ERROR) << "Failed to memset_s handle in BootstrapGetUniqueId.";
        return LCAL_ERROR_INTERNAL;
    }

    const char* env = Mki::GetEnv("LCAL_COMM_ID");
    if (env) {
        MKI_LOG(INFO) << "LCAL_COMM_ID set by environment to " << env;
        if (GetAddrFromString(&handle.addr, env) != LCAL_SUCCESS) {
            MKI_LOG(WARN) << ("Invalid LCAL_COMM_ID, please use format: <ipv4>:<port>");
            return LCAL_INVALID_VALUE;
        }
    } else {
        int bootRet = BootstrapGetServerIp(handle.addr);
        if (bootRet != LCAL_SUCCESS) {
            MKI_LOG(ERROR) << "lcal BootstrapGetIpPort failed!";
            return LCAL_ERROR_INTERNAL;
        }
    }
    int dev;
    int aclRet = aclrtGetDevice(&dev);
    if (aclRet != ACL_SUCCESS) {
        MKI_LOG(ERROR) << "ERROR: GetDevice.";
        return LCAL_ERROR_INTERNAL;
    }
    handle.addr.sin.sin_port = htons(LCAL_DEFAULT_SOCK_PORT + dev + commDomain);
    handle.magic = LCAL_MAGIC;

    return LCAL_SUCCESS;
}
}