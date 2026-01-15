#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
include(CMakeCommonLanguageInclude)
include(Compiler/CMakeCommonCompilerMacros)
set(CMAKE_INCLUDE_FLAG_CCE "-I")
if(UNIX)
    set(CMAKE_CCE_OUTPUT_EXTENSION .o)
else()
    set(CMAKE_CCE_OUTPUT_EXTENSION .obj)
endif()
set(CMAKE_DEPFILE_FLAGS_CCE "-MD -MT <DEP_TARGET> -MF <DEP_FILE>")
set(CMAKE_CCE_DEPFILE_FORMAT gcc)
set(CMAKE_CCE_DEPENDS_USE_COMPILER TRUE)
if(NOT CMAKE_CCE_COMPILE_OBJECT)
    set(CMAKE_CCE_COMPILE_OBJECT
            "${CMAKE_CCE_COMPILER} -xcce <DEFINES>\
        <INCLUDES>${__IMPLICIT_INCLUDES} ${_CMAKE_CCE_BUILTIN_INCLUDE_PATH}\
        <FLAGS> ${_CMAKE_COMPILE_AS_CCE_FLAG} ${_CMAKE_CCE_COMPILE_OPTIONS}\
        ${_CMAKE_CCE_COMMON_COMPILE_OPTIONS} -o <OBJECT> -c <SOURCE>")
endif()

