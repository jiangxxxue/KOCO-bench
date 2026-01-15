/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lcoc_func.h"
#include "lcoc_args.h"
#include "mki/utils/log/log.h"

using namespace std;
namespace Lcal {
    // 校验参数取值范围在[min, max]内,当max=-1时，表示参数取值范围在[min, +∞)
    bool CheckParamScope(const std::string &name, const int &value, const int &min, const int &max)
    {
        if (value < min || (max != PARAM_CHECK_MAX_VALUE && value > max)) {
            if (max == PARAM_CHECK_MAX_VALUE) {
                MKI_LOG(ERROR) << "The " << name << ":" << value << " must equal or greater than " << min << "!";
            } else {
                MKI_LOG(ERROR) << "The " << name << ":" << value << " must be in [" << min << ", " << max << "]!";
            }
            return false;
        }
        return true;
    }

    bool CheckParamScopeList(std::vector<std::tuple<std::string, int, int, int>> paramCheckList)
    {
        for (auto &param : paramCheckList) {
            auto name = std::get<0>(param);
            auto value = std::get<1>(param);
            auto min = std::get<2>(param);
            auto max = std::get<3>(param);
            if (value == INPUT_PARAM_DEFAULT_VALUE) {
                continue;
            }
            if (!CheckParamScope(name, value, min, max)) {
                return false;
            }
        }
        return true;
    }

    bool CheckParamAlign(const std::string &name, const int &value, const int &align)
    {
        if (align == 0) {
            return false;
        }
        if (value % align != 0) {
            MKI_LOG(ERROR) << "The " << name << ":" << value << " must be aligned by " << align << "!";
            return false;
        }
        return true;
    }

    void PrintErrorLog(LcalType lcalType, const std::string &log)
    {
        MKI_LOG(ERROR) << "[" + LCAL_TYPE2NAME.at(lcalType) + "]: " << log;
    }

    bool CheckParamPowerOfTwo(const std::string &name, int value)
    {
        if (value <= 0) {
            MKI_LOG(ERROR) << "The " << name << ":" << value << " must be greater than zero!";
            return false;
        }
        if ((static_cast<unsigned int>(value) & (static_cast<unsigned int>(value) - 1)) != 0) {
            MKI_LOG(ERROR) << "The " << name << ":" << value << " must be power of two!";
            return false;
        }
        return true;
    }

    int64_t GetAlignedMatrixSize(const int64_t &batchSize, const int64_t &m, const int64_t &n, const bool &transpose,
                                 int nElemAlign)
    {
        if (nElemAlign == 0) {
            return false;
        }
        int64_t nRow = transpose ? n : m;
        int64_t nCol = transpose ? m : n;
        int64_t nColAlign = (nCol + nElemAlign - 1) / nElemAlign * nElemAlign;
        return batchSize * nRow * nColAlign;
    }

}