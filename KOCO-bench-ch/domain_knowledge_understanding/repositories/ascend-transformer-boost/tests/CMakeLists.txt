#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

include_directories(
    $ENV{ASCEND_HOME_PATH}/include
)

add_subdirectory(framework)
if(USE_UNIT_TEST OR USE_ALL_TEST)
    add_subdirectory(unittest)
    add_subdirectory(cinterface)
endif()
if(USE_FUZZ_TEST)
    add_subdirectory(apitest/fuzztest)
endif()