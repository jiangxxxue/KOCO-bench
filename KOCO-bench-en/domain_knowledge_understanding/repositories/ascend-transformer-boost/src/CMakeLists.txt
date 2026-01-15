#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

set(ops_train_directory ${CMAKE_CURRENT_LIST_DIR}/ops/ops_train)
set(ops_infer_directory ${CMAKE_CURRENT_LIST_DIR}/ops/ops_infer)
set(ops_common_directory ${CMAKE_CURRENT_LIST_DIR}/ops/ops_common)
set(atb_directory ${CMAKE_CURRENT_LIST_DIR}/atb)
set(MSTX_PATH $ENV{ASCEND_HOME_PATH}/tools/mstx/include)
set(ATB_INCLUDE_DIR $ENV{ASCEND_HOME_PATH}/include)

add_compile_options(-Wfloat-equal -fno-common)

file(GLOB_RECURSE INFER_OP_SOURCE "${ops_infer_directory}/*.cpp")
file(GLOB_RECURSE TRAIN_OP_SOURCE "${ops_train_directory}/*.cpp")
file(GLOB_RECURSE COMMON_OP_SOURCE "${ops_common_directory}/*.cpp")
file(GLOB_RECURSE ATB_FRAMEWORK_SOURCE "${atb_directory}/*.cpp")

add_subdirectory(kernels)

if(BUILD_PYBIND AND NOT USE_CXX11_ABI)
    add_subdirectory(torch_atb)
endif()

add_library(atb SHARED ${INFER_OP_SOURCE} ${ATB_FRAMEWORK_SOURCE} ${COMMON_OP_SOURCE} ${C_INTERFACE_SOURCE})
add_library(atb_static STATIC ${INFER_OP_SOURCE} ${ATB_FRAMEWORK_SOURCE} ${COMMON_OP_SOURCE} ${C_INTERFACE_SOURCE})
add_library(atb_train SHARED ${TRAIN_OP_SOURCE} ${COMMON_OP_SOURCE})
add_library(atb_train_static STATIC ${TRAIN_OP_SOURCE} ${COMMON_OP_SOURCE})
if(USE_ASAN)
    target_link_libraries(atb PRIVATE asan)
    target_link_libraries(atb_train PRIVATE asan)
endif()
target_link_libraries(atb PUBLIC dl mki asdops atb_mixops ascendcl profapi lcal hccl pthread acl_op_compiler nnopbase)
target_link_libraries(atb_train PUBLIC atb)
target_include_directories(atb PUBLIC ${MSTX_PATH} ${ATB_INCLUDE_DIR})
target_include_directories(atb_static PUBLIC ${MSTX_PATH} ${ATB_INCLUDE_DIR})
target_include_directories(atb_train PUBLIC ${MSTX_PATH} ${ATB_INCLUDE_DIR})
target_include_directories(atb_train_static PUBLIC ${MSTX_PATH} ${ATB_INCLUDE_DIR})

install(TARGETS atb atb_static atb_train atb_train_static DESTINATION lib)