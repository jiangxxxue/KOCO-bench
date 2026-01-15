# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

option(USE_MSSANITIZER "USE_MSSANITIZER" OFF)
option(USE_MSDEBUG "USE_MSDEBUG" OFF)
message(STATUS "USE_MSSANITIZER:${USE_MSSANITIZER}")
message(STATUS "USE_MSDEBUG:${USE_MSDEBUG}")

set(OPS_PROJECT_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(OPS_THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/3rdparty")
set(MKI_PACKAGE_DIR "${OPS_THIRD_PARTY_DIR}/mki")
set(MKI_SCRIPT_DIR "${MKI_PACKAGE_DIR}/scripts")

file(GLOB ALL_OPS_IMPL_DIRS
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/ops/*/operation_implement"
    "${CMAKE_CURRENT_LIST_DIR}/ops/*/operation_implement/tiling"
)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/ops
    ${ALL_OPS_IMPL_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/include
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/kernels/include
    ${CMAKE_SOURCE_DIR}/src/kernels/tiling
    ${CMAKE_SOURCE_DIR}/src/kernels/mixkernels
    ${CMAKE_SOURCE_DIR}/3rdparty/mki/include
)

add_subdirectory(ops)

file(GLOB_RECURSE CUSTOMIZE_SRCS
     "${CMAKE_CURRENT_SOURCE_DIR}/ops/*/operation_implement/*.cpp"
)

add_library(atb_customize SHARED ${CUSTOMIZE_SRCS})
add_library(atb_customize_static STATIC ${CUSTOMIZE_SRCS})

set(CUSTOMIZE_DEPS
    mki
    customize_ops
    ascendcl
    profapi
    pthread
    acl_op_compiler
)

target_link_libraries(atb_customize PUBLIC ${CUSTOMIZE_DEPS})
target_link_libraries(atb_customize_static PUBLIC ${CUSTOMIZE_DEPS})

install(TARGETS atb_customize atb_customize_static DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/customize)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/ops_customize/customize_ops_configs DESTINATION ./configs)

if (BUILD_CUSTOMIZE_OPS_TEST)
    set(GTEST_PATH "${CMAKE_SOURCE_DIR}/3rdparty/googletest")
    find_package(GTest 1.13.0 REQUIRED
                PATHS "${GTEST_PATH}"
                NO_DEFAULT_PATH)
    enable_testing()
    add_subdirectory(ops/customize_blockcopy/tests)
endif()