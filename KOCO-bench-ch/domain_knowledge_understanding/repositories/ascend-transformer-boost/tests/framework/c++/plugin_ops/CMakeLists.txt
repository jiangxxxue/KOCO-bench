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
    $ENV{ATB_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include
)

link_directories(
    $ENV{ATB_HOME_PATH}/lib
    $ENV{ASCEND_HOME_PATH}/lib64
)

file(GLOB_RECURSE ATB_PLUGIN_OPERATION "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(atb_test_pluginop STATIC ${ATB_PLUGIN_OPERATION})
target_compile_options(atb_test_pluginop PUBLIC "-Wno-unused-parameter" "-Wno-unused-variable" "-Wno-unused-function")
target_compile_options(atb_test_pluginop PUBLIC "-Wno-maybe-uninitialized" "-Wno-missing-field-initializers")
target_compile_options(atb_test_pluginop PUBLIC "-Wno-sign-compare" "-Wno-enum-compare" "-Wno-attributes")
target_link_libraries(atb_test_pluginop PUBLIC atb ascendcl opapi nnopbase)
install(TARGETS atb_test_pluginop DESTINATION lib)