#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

cmake_minimum_required(VERSION 3.15)
project("atb")
set(CMAKE_CXX_STANDARD 17)

IF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    SET(ARCH aarch64)
ELSEIF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    SET(ARCH x86_64)
ENDIF ()

option(BUILD_TEST_FRAMEWORK "BUILD_TEST_FRAMEWORK" OFF)
option(BUILD_PYBIND "Build Python bindings for ATB Python Module" ON)
option(BUILD_CUSTOMIZE_OPS "Build user-customize operators" OFF)
option(BUILD_CUSTOMIZE_OPS_TEST "Build user-customize operators tests files" OFF)
option(USE_UNIT_TEST "USE_UNIT_TEST" OFF)
option(USE_PYTHON_TEST "USE_PYTHON_TEST" OFF)
option(USE_FUZZ_TEST "USE_FUZZ_TEST" OFF)
option(USE_CSV_OPS_TEST "USE_CSV_OPS_TEST" OFF)
option(USE_INFRA_TEST "USE_INFRA_TEST" OFF)
option(USE_TORCH_ATB_TEST "USE_TORCH_ATB_TEST" OFF)
option(USE_CXX11_ABI "USE_CXX11_ABI" ON)
option(USE_ASAN "USE_ASAN" OFF)
option(USE_MSSANITIZER "USE_MSSANITIZER" OFF)

message(STATUS "BUILD_TEST_FRAMEWORK:${BUILD_TEST_FRAMEWORK}")
message(STATUS "BUILD_PYBIND:${BUILD_PYBIND}")
message(STATUS "BUILD_CUSTOMIZE_OPS:${BUILD_CUSTOMIZE_OPS}")
message(STATUS "BUILD_CUSTOMIZE_OPS_TEST:${BUILD_CUSTOMIZE_OPS_TEST}")
message(STATUS "USE_UNIT_TEST:${USE_UNIT_TEST}")
message(STATUS "USE_PYTHON_TEST:${USE_PYTHON_TEST}")
message(STATUS "USE_FUZZ_TEST:${USE_FUZZ_TEST}")
message(STATUS "USE_CSV_OPS_TEST:${USE_CSV_OPS_TEST}")
message(STATUS "USE_INFRA_TEST:${USE_INFRA_TEST}")
message(STATUS "USE_TORCH_ATB_TEST:${USE_TORCH_ATB_TEST}")
message(STATUS "USE_CXX11_ABI:${USE_CXX11_ABI}")
message(STATUS "USE_ASAN:${USE_ASAN}")
message(STATUS "USE_MSSANITIZER:${USE_MSSANITIZER}")

set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fstack-protector-all")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wl,--build-id=none")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
if(USE_ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -fsanitize-coverage=trace-pc")
endif()
IF(CMAKE_BUILD_TYPE STREQUAL "Release")
    ADD_LINK_OPTIONS(-s)
ELSE()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DEBUG")
ENDIF()

set(LD_FLAGS_GLOBAL "-rdynamic -ldl -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -Wl,--build-id=none")
set(CMAKE_SHARED_LINKER_FLAGS "${LD_FLAGS_GLOBAL} -fexceptions")
set(CMAKE_EXE_LINKER_FLAGS "${LD_FLAGS_GLOBAL} -pie -fPIE")
if(USE_CXX11_ABI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(cxx_abi 1)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(cxx_abi 0)
endif()

set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/output/atb/cxx_abi_${cxx_abi}")
message(STATUS "CMAKE_INSTALL_PREFIX:${CMAKE_INSTALL_PREFIX}")

if(BUILD_PYBIND AND NOT USE_CXX11_ABI)
    add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/pybind11)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/kernels/include
    ${PROJECT_SOURCE_DIR}/src/kernels/lcal/include
    ${PROJECT_SOURCE_DIR}/src/kernels/lcal/include/lcoc
    ${PROJECT_SOURCE_DIR}/src/kernels/lcal/include/lcoc/tiling
    ${PROJECT_SOURCE_DIR}/3rdparty/mki/include
    ${PROJECT_SOURCE_DIR}/3rdparty/nlohmannJson/include
    $ENV{ASCEND_HOME_PATH}/include/aclnn
    $ENV{PYTHON_INCLUDE_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/include
    $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
    $ENV{PYTORCH_NPU_INSTALL_PATH}/include
    $ENV{ASCEND_HOME_PATH}/${ARCH}-linux/include/hccl/)

link_directories(
    ${PROJECT_SOURCE_DIR}/3rdparty/mki/lib
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{PYTHON_LIB_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/lib
    $ENV{PYTORCH_NPU_INSTALL_PATH}/lib
    ${PROJECT_SOURCE_DIR}/output/atb/cxx_abi_${cxx_abi}/lib)

if(BUILD_TEST_FRAMEWORK OR USE_UNIT_TEST OR USE_PYTHON_TEST OR USE_FUZZ_TEST OR USE_CSV_OPS_TEST OR USE_INFRA_TEST OR USE_ALL_TEST)
    if(USE_FUZZ_TEST OR USE_ALL_TEST)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    endif()
    set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
    add_subdirectory(tests)
endif()
add_subdirectory(src)
if (BUILD_CUSTOMIZE_OPS)
    add_subdirectory(ops_customize)
endif()

install(FILES ${PROJECT_SOURCE_DIR}/scripts/set_env.sh DESTINATION ./..)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/ops_configs DESTINATION ./configs)
install(FILES ${PROJECT_SOURCE_DIR}/3rdparty/mki/lib/libmki.so DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/atb DESTINATION include)
