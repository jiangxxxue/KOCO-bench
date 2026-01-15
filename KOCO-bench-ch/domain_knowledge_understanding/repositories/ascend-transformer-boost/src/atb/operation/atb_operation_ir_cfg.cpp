/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/operation/atb_operation_ir_cfg.h"

#include <memory>
#include <string>
#include <mki/utils/status/status.h>
#include "atb/utils/log.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"

namespace atb {
AtbOperationIrCfg::AtbOperationIrCfg()
{
    InitOperationIrCfg();
}

AtbOperationIrCfg::~AtbOperationIrCfg() {}

void AtbOperationIrCfg::InitOperationIrCfg()
{
    std::string iniFilePath = GetSingleton<Config>().GetAtbHomePath() + "/configs/ops_configs/atb_ops_info.ini";
    Mki::Status status = opIrCfg_.Load(iniFilePath);
    if (!status.Ok()) {
        ATB_LOG(ERROR) << "Load atb_ops_info.ini failed!";
        return;
    }
    ATB_LOG(INFO) << "Load atb_ops_info.ini success!";
}

Mki::OperationIr *AtbOperationIrCfg::GetOperationIr(const std::string &opKey)
{
    return opIrCfg_.GetOperationIr(opKey);
}
} //  namespace atb