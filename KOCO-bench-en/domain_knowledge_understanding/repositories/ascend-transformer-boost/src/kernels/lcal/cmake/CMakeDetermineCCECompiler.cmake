#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
set(PRIVATE_CCEC_PATH ${CMAKE_SOURCE_DIR}/3rdparty/compiler)
find_program(CMAKE_CCE_COMPILER
        NAMES "ccec"
        HINTS "${PRIVATE_CCEC_PATH}/ccec_compiler/bin"
        HINTS "${ASCEND_HOME_PATH}/${ARCH}-linux/ccec_compiler/bin"
        DOC "CCE Compiler"
)
find_program(CMAKE_CCE_LINKER
        NAMES "ld.lld"
        HINTS "${PRIVATE_CCEC_PATH}/ccec_compiler/bin"
        HINTS "${ASCEND_HOME_PATH}/${ARCH}-linux/ccec_compiler/bin"
        DOC "CCE Linker"
)
message(STATUS "CMAKE_CCE_COMPILER: " ${CMAKE_CCE_COMPILER})
message(STATUS "CMAKE_PLATFORM_INFO_DIR: "${CMAKE_PLATFORM_INFO_DIR})
configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeCCECompiler.cmake.in
        ${CMAKE_PLATFORM_INFO_DIR}/CMakeCCECompiler.cmake 
        @ONLY
)
set(CMAKE_CCE_SOURCE_FILE_EXTENSIONS cce;cpp)
set(CMAKE_CCE_COMPILER_ENV_VAR "CCEC")

