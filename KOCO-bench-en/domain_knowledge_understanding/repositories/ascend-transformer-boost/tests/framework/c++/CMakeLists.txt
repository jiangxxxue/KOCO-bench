#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

file(GLOB_RECURSE TEST_UTILS_SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/test_utils/*.cpp")
 
add_library(atb_test_utils ${TEST_UTILS_SOURCE_FILES})
target_include_directories(atb_test_utils PUBLIC ${PROJECT_SOURCE_DIR}/tests/framework/c++)
target_compile_options(atb_test_utils PUBLIC "-Wno-unused-parameter" "-Wno-unused-variable" "-Wno-unused-function")
target_compile_options(atb_test_utils PUBLIC "-Wno-maybe-uninitialized" "-Wno-missing-field-initializers")
target_compile_options(atb_test_utils PUBLIC "-Wno-sign-compare" "-Wno-enum-compare" "-Wno-attributes")
target_link_libraries(atb_test_utils PUBLIC atb atb_train torch torch_cpu c10)
 
file(GLOB_RECURSE ATB_TORCH_SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/atb_torch/*.cpp")
file(GLOB_RECURSE ATB_LAYER_OPERATION "${CMAKE_CURRENT_LIST_DIR}/layer_ops/*.cpp")
add_library(atb_test_framework SHARED ${ATB_TORCH_SOURCE_FILES} ${ATB_LAYER_OPERATION})
target_include_directories(atb_test_framework PUBLIC ${PROJECT_SOURCE_DIR}/tests/framework/c++/layer_ops)
target_link_libraries(atb_test_framework PUBLIC atb_test_utils)
target_link_libraries(atb_test_framework PUBLIC torch_python torch_npu)
install(TARGETS atb_test_framework DESTINATION lib)

add_subdirectory(kernels/autogen)

file(GLOB_RECURSE KERNELS_TEST_UTILS_SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/kernels/test_utils/*.cpp")
add_library(atb_kernels_test_utils ${KERNELS_TEST_UTILS_SOURCE_FILES})
target_include_directories(atb_kernels_test_utils PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(atb_kernels_test_utils PUBLIC ${PROJECT_SOURCE_DIR}/tests/framework/c++/kernels/autogen)
target_link_libraries(atb_kernels_test_utils PUBLIC mki_test_autogen mki torch torch_cpu c10 c_sec)
target_link_options(atb_kernels_test_utils PUBLIC LINKER:--allow-shlib-undefined)
target_compile_options(atb_kernels_test_utils PUBLIC "-Wno-unused-parameter" "-Wno-unused-variable" "-Wno-unused-function")
