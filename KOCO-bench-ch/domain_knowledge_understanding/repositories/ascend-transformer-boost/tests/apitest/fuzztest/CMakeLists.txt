#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
add_executable(atb_fuzztest ${SOURCE_FILES})
target_include_directories(atb_fuzztest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include)
target_link_directories(atb_fuzztest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib)
target_include_directories(atb_fuzztest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/secodefuzz/include)
target_link_directories(atb_fuzztest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/secodefuzz/lib)
set_target_properties(atb_fuzztest PROPERTIES LINK_FLAGS "-Wl,-rpath-link,${PROJECT_SOURCE_DIR}/3rdparty/asdops/lib/")
target_link_libraries(atb_fuzztest PRIVATE atb_test_utils -lSecodefuzz -lgtest -lgtest_main)
target_link_libraries(atb_fuzztest PRIVATE torch_python torch_npu)
install(TARGETS atb_fuzztest DESTINATION bin)
