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
set(OP_NAMES pure_matmul matmul_allreduce matmul_reduce_scatter allgather_matmul allgather_matmul_reduce_scatter alltoallv_allgather_matmul matmul_reduce_scatter_alltoallv)
 
file(GLOB KERNEL_FILES *.cpp)
set_source_files_properties(${KERNEL_FILES} PROPERTIES LANGUAGE CCE)
file(GLOB KERNEL_FILES2 *.cce)
set_source_files_properties(${KERNEL_FILES2} PROPERTIES LANGUAGE CCE)

foreach(OP_NAME IN LISTS OP_NAMES)
    add_library(lcoc_${OP_NAME}_aic_obj OBJECT coc_${OP_NAME}.cce)
    target_compile_options(lcoc_${OP_NAME}_aic_obj PRIVATE
            ${CCE_COMPILE_OPTION}
            --cce-aicore-arch=${AIC_ARCH}
    )
    add_library(lcoc_${OP_NAME}_aiv_obj OBJECT coc_${OP_NAME}.cce)
    target_compile_options(lcoc_${OP_NAME}_aiv_obj PRIVATE
            ${CCE_COMPILE_OPTION}
            --cce-aicore-arch=${AIV_ARCH}
    )
    add_custom_target(${OP_NAME}_o
            COMMAND ${CMAKE_CCE_LINKER} -m aicorelinux -Ttext=0
            "CMakeFiles/lcoc_${OP_NAME}_aic_obj.dir/coc_${OP_NAME}*.o"
            "CMakeFiles/lcoc_${OP_NAME}_aiv_obj.dir/coc_${OP_NAME}*.o"
            ${SANITIZER_DEPEND_LIBS}
            --static -o "lcal_coc_${OP_NAME}.o"   --allow-multiple-definition
            COMMAND truncate -c -s ${LCAL_1OP_BIN_SIZE} "lcal_coc_${OP_NAME}.o"
    )
endforeach()
# 生成文件名列表，每个都带有 .o 后缀
set(OUTPUT_FILES "")
foreach(OP_NAME IN LISTS OP_NAMES)
    list(APPEND OUTPUT_FILES "lcal_coc_${OP_NAME}.o")
endforeach()

add_custom_target(lcoc_op
    COMMAND cat ${OUTPUT_FILES} > lcoc_op.o
)
foreach(OP_NAME IN LISTS OP_NAMES)
    add_dependencies(${OP_NAME}_o lcoc_${OP_NAME}_aic_obj lcoc_${OP_NAME}_aiv_obj)
    add_dependencies(lcoc_op ${OP_NAME}_o)
endforeach()