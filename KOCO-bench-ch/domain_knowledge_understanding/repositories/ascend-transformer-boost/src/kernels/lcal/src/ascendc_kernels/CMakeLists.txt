#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
include(../ascendc.cmake)
include_directories(.)
option(ENABLE_LCCL_910A5_OP "ENABLE lccl_910A5_op library and compile options" OFF)

file(GLOB_RECURSE KERNEL_FILES *.cpp)
set_source_files_properties(${KERNEL_FILES} PROPERTIES LANGUAGE CCE)

# 常规算子
add_library(normal_lccl_op1_tmp OBJECT
            lccl_op1.cpp
            sync_collectives.h
            collectives.h
)

add_library(normal_lccl_op2_tmp OBJECT
            lccl_op2.cpp
            sync_collectives.h
            collectives.h
)
# 设置编译选项
target_compile_options(normal_lccl_op1_tmp PRIVATE 
        ${CCE_COMPILE_OPTION}
        --cce-aicore-arch=${AIV_ARCH}
)
# 设置编译选项
target_compile_options(normal_lccl_op2_tmp PRIVATE 
        ${CCE_COMPILE_OPTION}
        --cce-aicore-arch=${AIV_ARCH}
)

add_custom_target(normal_lccl_op1
        COMMAND ${CMAKE_CCE_LINKER} -m aicorelinux -Ttext=0
        "CMakeFiles/normal_lccl_op1_tmp.dir/lccl_op1.cpp.o"
        ${SANITIZER_DEPEND_LIBS}
        --static -o "CMakeFiles/normal_lccl_op1.dir/lccl_op1.cpp.o"   --allow-multiple-definition
)
add_dependencies(normal_lccl_op1 normal_lccl_op1_tmp)
add_custom_target(normal_lccl_op2
        COMMAND ${CMAKE_CCE_LINKER} -m aicorelinux -Ttext=0
        "CMakeFiles/normal_lccl_op2_tmp.dir/lccl_op2.cpp.o"
        ${SANITIZER_DEPEND_LIBS}
        --static -o "CMakeFiles/normal_lccl_op2.dir/lccl_op2.cpp.o"   --allow-multiple-definition
)
add_dependencies(normal_lccl_op2 normal_lccl_op2_tmp)

# 带dump的mix算子
add_library(dump_lccl_op1_tmp_aic OBJECT
            lccl_op1.cpp
            sync_collectives.h
            collectives.h
)
add_library(dump_lccl_op1_tmp_aiv OBJECT
            lccl_op1.cpp
            sync_collectives.h
            collectives.h
)
target_compile_options(dump_lccl_op1_tmp_aic PRIVATE
        ${CCE_COMPILE_OPTION}
        --cce-aicore-arch=${AIC_ARCH}
        -DENABLE_LCCL_DUMP
        -DENABLE_LCCL_MIX
)
target_compile_options(dump_lccl_op1_tmp_aiv PRIVATE
        ${CCE_COMPILE_OPTION}
        --cce-aicore-arch=${AIV_ARCH}
        --cce-long-call=true
        -DENABLE_LCCL_DUMP
        -DENABLE_LCCL_MIX
)

add_custom_target(dump_lccl_op1
        COMMAND ${CMAKE_CCE_LINKER} -m aicorelinux -Ttext=0
        "CMakeFiles/dump_lccl_op1_tmp_aic.dir/lccl_op1.cpp.o"
        "CMakeFiles/dump_lccl_op1_tmp_aiv.dir/lccl_op1.cpp.o"
        ${SANITIZER_DEPEND_LIBS}
        --static -o "CMakeFiles/dump_lccl_op1.dir/lccl_op1.cpp.o"   --allow-multiple-definition
)
add_dependencies(dump_lccl_op1 dump_lccl_op1_tmp_aic dump_lccl_op1_tmp_aiv)

add_library(dump_lccl_op2_tmp_aic OBJECT
            lccl_op2.cpp
            sync_collectives.h
            collectives.h
)
add_library(dump_lccl_op2_tmp_aiv OBJECT
            lccl_op2.cpp
            sync_collectives.h
            collectives.h
)
target_compile_options(dump_lccl_op2_tmp_aic PRIVATE
        ${CCE_COMPILE_OPTION}
        --cce-aicore-arch=${AIC_ARCH}
        -DENABLE_LCCL_DUMP
        -DENABLE_LCCL_MIX
)
target_compile_options(dump_lccl_op2_tmp_aiv PRIVATE
        ${CCE_COMPILE_OPTION}
        --cce-aicore-arch=${AIV_ARCH}
        --cce-long-call=true
        -DENABLE_LCCL_DUMP
        -DENABLE_LCCL_MIX
)

add_custom_target(dump_lccl_op2
        COMMAND ${CMAKE_CCE_LINKER} -m aicorelinux -Ttext=0
        "CMakeFiles/dump_lccl_op2_tmp_aic.dir/lccl_op2.cpp.o"
        "CMakeFiles/dump_lccl_op2_tmp_aiv.dir/lccl_op2.cpp.o"
        ${SANITIZER_DEPEND_LIBS}
        --static -o "CMakeFiles/dump_lccl_op2.dir/lccl_op2.cpp.o"   --allow-multiple-definition
)
add_dependencies(dump_lccl_op2 dump_lccl_op2_tmp_aic dump_lccl_op2_tmp_aiv)

# 不带dump的mix算子

add_library(mix_lccl_op1_tmp_aic OBJECT
            lccl_op1.cpp
            sync_collectives.h
            collectives.h
)
add_library(mix_lccl_op1_tmp_aiv OBJECT
            lccl_op1.cpp
            sync_collectives.h
            collectives.h
)
target_compile_options(mix_lccl_op1_tmp_aic PRIVATE
        ${CCE_COMPILE_OPTION}
        --cce-aicore-arch=${AIC_ARCH}
        -DENABLE_LCCL_MIX
)
target_compile_options(mix_lccl_op1_tmp_aiv PRIVATE
        ${CCE_COMPILE_OPTION}
        --cce-aicore-arch=${AIV_ARCH}
        --cce-long-call=true
        -DENABLE_LCCL_MIX
)

add_custom_target(mix_lccl_op1
        COMMAND ${CMAKE_CCE_LINKER} -m aicorelinux -Ttext=0
        "CMakeFiles/mix_lccl_op1_tmp_aic.dir/lccl_op1.cpp.o"
        "CMakeFiles/mix_lccl_op1_tmp_aiv.dir/lccl_op1.cpp.o"
        ${SANITIZER_DEPEND_LIBS}
        --static -o "CMakeFiles/mix_lccl_op1.dir/lccl_op1.cpp.o"   --allow-multiple-definition
)
add_dependencies(mix_lccl_op1 mix_lccl_op1_tmp_aic mix_lccl_op1_tmp_aiv)

add_library(mix_lccl_op2_tmp_aic OBJECT
            lccl_op2.cpp
            sync_collectives.h
            collectives.h
)
add_library(mix_lccl_op2_tmp_aiv OBJECT
            lccl_op2.cpp
            sync_collectives.h
            collectives.h
)
target_compile_options(mix_lccl_op2_tmp_aic PRIVATE
        ${CCE_COMPILE_OPTION}
        --cce-aicore-arch=${AIC_ARCH}
        -DENABLE_LCCL_MIX
)
target_compile_options(mix_lccl_op2_tmp_aiv PRIVATE
        ${CCE_COMPILE_OPTION}
        --cce-aicore-arch=${AIV_ARCH}
        --cce-long-call=true
        -DENABLE_LCCL_MIX
)

add_custom_target(mix_lccl_op2
        COMMAND ${CMAKE_CCE_LINKER} -m aicorelinux -Ttext=0
        "CMakeFiles/mix_lccl_op2_tmp_aic.dir/lccl_op2.cpp.o"
        "CMakeFiles/mix_lccl_op2_tmp_aiv.dir/lccl_op2.cpp.o"
        ${SANITIZER_DEPEND_LIBS}
        --static -o "CMakeFiles/mix_lccl_op2.dir/lccl_op2.cpp.o"   --allow-multiple-definition
)
add_dependencies(mix_lccl_op2 mix_lccl_op2_tmp_aic mix_lccl_op2_tmp_aiv)

add_custom_target(lccl_op
        COMMAND echo "generating lccl op ... ENABLE_LCCL_910A5_OP=${ENABLE_LCCL_910A5_OP}"
        COMMAND rm -f lccl_op.o
        COMMAND find CMakeFiles -name "*.o" ! -path "*tmp*" | sort | xargs -I {} sed '1s/^/DDDD/' {} >> lccl_op.o
        COMMAND truncate -c -s ${LCAL_1OP_BIN_SIZE} lccl_op.o
        COMMAND rm -f ${LCAL_CCE_PATH}
)
add_dependencies(lccl_op dump_lccl_op1 dump_lccl_op2 mix_lccl_op1 mix_lccl_op2 normal_lccl_op1 normal_lccl_op2)
