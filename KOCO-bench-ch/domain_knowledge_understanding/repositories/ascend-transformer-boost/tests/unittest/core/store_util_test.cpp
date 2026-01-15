/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <vector>
#include <string>
#include <string.h>
#include <cstdlib>
#include <gtest/gtest.h>
#include "atb/utils/store_util.h"
#include <atb/utils/log.h>
#include <mki/utils/file_system/file_system.h>
#include <mki/launch_param.h>
#include <acl/acl.h>
#include <iostream>
#include <fstream>
using namespace std;
 
bool isFileExist(std::string fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}
 
TEST(TestStoreUtil, TestWriteFile)
{
    std::vector<uint8_t> &tilingData = *(new std::vector<uint8_t>(10, 1));
    std::string dirPath = "/home/test";
    std::string filePath = dirPath + "/test_tilingdata.bin";
    atb::StoreUtil::WriteFile(tilingData.data(), tilingData.size(), filePath);
    bool isExist = isFileExist(filePath);
    EXPECT_EQ(isExist, true);
    Mki::FileSystem::DeleteFile(filePath);
}
