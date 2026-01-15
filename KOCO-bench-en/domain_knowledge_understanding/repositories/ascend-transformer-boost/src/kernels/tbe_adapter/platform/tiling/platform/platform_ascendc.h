/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file platform_ascendc.h
 * \brief
 */

#ifndef PLATFORM_ASCENDC_H
#define PLATFORM_ASCENDC_H

#include <cstdint>
#include <mutex>

#define ASCENDC_ASSERT(cond, behavior) \
    do {                               \
        if (!(cond)) {                 \
            behavior;                  \
            raise(SIGABRT);            \
        }                              \
    } while (0)
namespace fe {
class PlatFormInfos;
}

namespace platform_ascendc {
enum class CoreMemType {
    L0_A = 0,
    L0_B = 1,
    L0_C = 2,
    L1 = 3,
    L2 = 4,
    UB = 5,
    HBM = 6,
    RESERVED
};

enum class SocVersion {
    ASCEND910 = 0,  // Ascend910A, Ascend910B
    ASCEND910B,    // Ascend910B1~4, Ascend910B2C, Ascend910_93 Serials
    ASCEND310P,    // Ascend310P1, Ascend310P3
    ASCEND310B,    // Ascend310B1, Ascend310B2, Ascend310B3, Ascend310B4
    ASCEND910_95,  // ASCEND910_95, __DAV_C310__
    ASCEND910_55,  // Ascend910_55, __DAV_310R6__
    AS31XM1,
    ASCEND031,
    ASCEND035,
    ASCEND310,
    ASCEND610,
    ASCEND610Lite,
    ASCEND910_93,
    BS9SX1A,
    BS9SX2A,
    HI3796CV300CS,
    HI3796CV300ES,
    MC61AM21A,
    SD3403,
    KIRIN9010,
    RESERVED_VERSION = 99999
};

class PlatformAscendC {
public:
    PlatformAscendC() = delete;
    ~PlatformAscendC() = default;
    explicit PlatformAscendC(fe::PlatFormInfos *platformInfo): platformInfo_(platformInfo) {}
    /**
     * Get Core Number
     * On Ascend910B MIX model, return AICore number
     * @return core number by core type
     */
    uint32_t GetCoreNum(void) const;
    /**
     * Get Core Number AiCore
     * @return ai_core_num
     */
    uint32_t GetCoreNumAic(void) const;
    /**
     * Get Core Number VectorCore
     * @return vector_core_num
     */
    uint32_t GetCoreNumAiv(void) const;
    /**
     * Get Core Number VectorCore for m200
     * @return vector_core_num if m200, otherwise 0
     */
    uint32_t GetCoreNumVector(void) const;
    /**
     * Calc task schedule block dim
    * @sliceNum number slice of data division
     * @aicCoreNum value of GetCoreNumAic() if used cube API, otherwise 0
     * @aivCoreNum value of GetCoreNumAiv() if used vector API, otherwise 0
     * @return task schedule block dim
     */
    uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum) const;
    /**
     * Get Work Space Size
     * @return work sapce size by chip type
     */
    uint32_t GetLibApiWorkSpaceSize(void) const;
    uint32_t GetResCubeGroupWorkSpaceSize(void) const;
    uint32_t GetResGroupBarrierWorkSpaceSize(void) const;
    void GetCoreMemSize(const CoreMemType &memType, uint64_t &size) const;
    void GetCoreMemBw(const CoreMemType &memType, uint64_t &bwSize) const;
    /**
     * Get Soc Version Enum
     * @return Enum SocVersion
     */
    SocVersion GetSocVersion(void) const;

private:
    fe::PlatFormInfos *platformInfo_;
    fe::PlatFormInfos* GetPlatFormInfo(void) const;
};
class PlatformAscendCManager {
public:
    static PlatformAscendC* GetInstance()
    {
        const std::lock_guard<std::mutex> lock(platformInitMtx);
        if (platformInfo == nullptr) {
            PlatformAscendCManagerInit(nullptr);
            if (platformInfo == nullptr) {
                return nullptr;
            }
        }
        return platformInfo;
    }
    static PlatformAscendC* GetInstance(const char *customSocVersion)
    {
        const std::lock_guard<std::mutex> lock(platformInitMtx);
        if (platformInfo == nullptr) {
            PlatformAscendCManagerInit(customSocVersion);
            if (platformInfo == nullptr) {
                return nullptr;
            }
        }
        return platformInfo;
    }
private:
    static PlatformAscendC *platformInfo;
    static std::mutex platformInitMtx;
    static PlatformAscendC* PlatformAscendCManagerInit(const char *customSocVersion);
    static SocVersion SocVersionMap(const char *socVersionStr);
    static fe::PlatFormInfos* PlatformAscendCInit(const char *customSocVersion);
    PlatformAscendCManager();
    ~PlatformAscendCManager() = default;
};
}
#endif
