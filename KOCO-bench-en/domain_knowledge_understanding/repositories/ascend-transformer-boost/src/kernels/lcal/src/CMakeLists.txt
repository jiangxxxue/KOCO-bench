#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
set_source_files_properties(lcal_internal.cpp PROPERTIES COMPILE_FLAGS "-O3")

set(LCAL_SOURCE_FILE lcal_comm.cpp lccl.cpp
        lcal_internal.cpp lcal_internal.h lcal_wrap.cpp
        tools/socket/lcal_sock_exchange.h
        tools/socket/lcal_sock_exchange.cpp
        coc_kernel_args.h coc_kernel_args.cpp lcoc.cpp lcoc_func.cpp
)
file(GLOB TILING_SOURCE_FILE tiling/*.cpp)
list(APPEND LCAL_SOURCE_FILE ${TILING_SOURCE_FILE})

add_library(lcal SHARED ${LCAL_SOURCE_FILE})
add_library(lcal_static STATIC ${LCAL_SOURCE_FILE})
set_target_properties(lcal_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(lcal ascendcl runtime profapi c_sec mki)
target_link_libraries(lcal_static ascendcl runtime profapi c_sec mki)

message(STATUS "LCAL USE_MSSANITIZER = ${USE_MSSANITIZER}")
set(LCAL_CCE_PATH "${CMAKE_CURRENT_BINARY_DIR}/lcal_cce.o")
if(USE_MSSANITIZER)
math(EXPR LCAL_1OP_BIN_SIZE "128 * 1024 * 1024")
add_definitions(-DUSE_MSSANITIZER)
else()
math(EXPR LCAL_1OP_BIN_SIZE "5 * 1024 * 1024")
endif()

add_definitions(-DLCAL_1OP_BIN_SIZE=${LCAL_1OP_BIN_SIZE})

add_subdirectory(kernels)
add_subdirectory(ascendc_kernels)

add_custom_command(
        OUTPUT ${LCAL_CCE_PATH}
        COMMAND cat ascendc_kernels/lccl_op.o kernels/lcoc_op.o > ${LCAL_CCE_PATH}
        COMMAND echo "concat op..."
        DEPENDS lccl_op lcoc_op
)

set_source_files_properties(
        lcal_internal.cpp
        PROPERTIES
        OBJECT_DEPENDS ${LCAL_CCE_PATH}
)
install(TARGETS lcal LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(TARGETS lcal_static DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)



