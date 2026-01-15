/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "customize_operation_ir_cfg.h"

#include <memory>
#include <string>
#include <mki/utils/status/status.h>
#include "atb/utils/log.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"

namespace atb {
CustomizeOperationIrCfg::CustomizeOperationIrCfg()
{
    InitOperationIrCfg();
}

CustomizeOperationIrCfg::~CustomizeOperationIrCfg() {}

void CustomizeOperationIrCfg::InitOperationIrCfg()
{
    std::string iniFilePath =
        GetSingleton<Config>().GetAtbHomePath() + "/configs/customize_ops_configs/customize_ops_info.ini";
    Mki::Status status = opIrCfg_.Load(iniFilePath);
    if (!status.Ok()) {
        ATB_LOG(ERROR) << "Load customize_ops_info.ini failed!";
        return;
    }
    ATB_LOG(INFO) << "Load customize_ops_info.ini success!";
}

Mki::OperationIr *CustomizeOperationIrCfg::GetOperationIr(const std::string &opKey)
{
    return opIrCfg_.GetOperationIr(opKey);
}
} //  namespace atb