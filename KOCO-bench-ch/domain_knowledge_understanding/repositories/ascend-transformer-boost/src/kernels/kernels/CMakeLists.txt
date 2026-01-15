# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(NAMESPACE "AsdOps")
set(OP_LIST_YAML_DIR "${OPS_PROJECT_ROOT_DIR}/configs/kernels")
set(OPS_CONFIG_CMAKE_DIR "${CMAKE_BINARY_DIR}/cmake/kernels")

include(${MKI_PACKAGE_DIR}/cmake/host_config.cmake)
include(${MKI_PACKAGE_DIR}/cmake/kernel_config.cmake)

include_directories($ENV{ASCEND_HOME_PATH}/include
                    $ENV{ASCEND_HOME_PATH}/include/aclnn)
                    
if(NOT EXISTS ${OP_LIST_YAML_DIR}/op_list.yaml)
    file(MAKE_DIRECTORY ${OP_LIST_YAML_DIR})
    execute_process(COMMAND python3 ${MKI_PACKAGE_DIR}/scripts/op_list_utils.py -s ${CMAKE_CURRENT_LIST_DIR} -d ${OP_LIST_YAML_DIR}
                    ERROR_VARIABLE RESULT_INFO
                    RESULT_VARIABLE RESULT)
    if(NOT RESULT EQUAL 0)
        message(FATAL_ERROR "create op_list.yaml failed, error code: ${RESULT}, error info:\n${RESULT_INFO}")
    endif()
endif()

file(MAKE_DIRECTORY ${OPS_CONFIG_CMAKE_DIR})
set(OP_LIST_CMD_PARAMS "'${OP_LIST_YAML_DIR}/op_list.yaml', '${OPS_CONFIG_CMAKE_DIR}/op_build.cmake'")
execute_process(COMMAND python3 -c "import op_list_utils\nop_list_utils.build_cmake_options(${OP_LIST_CMD_PARAMS})"
                WORKING_DIRECTORY ${MKI_PACKAGE_DIR}/scripts
                ERROR_VARIABLE RESULT_INFO
                RESULT_VARIABLE RESULT)
if(NOT RESULT EQUAL 0)
    message(FATAL_ERROR "create op_build.cmake failed, error code: ${RESULT}, error info:\n${RESULT_INFO}")
endif()

include(${OPS_CONFIG_CMAKE_DIR}/op_build.cmake)

 
file(GLOB_RECURSE CMAKELISTS_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt")
string(REPLACE "/CMakeLists.txt" "" CMAKELISTS_PATHS "${CMAKELISTS_PATHS}")
foreach(SUB_DIR ${CMAKELISTS_PATHS})
    message(STATUS "add subdirectory: ${SUB_DIR}")
    add_subdirectory(${SUB_DIR})
endforeach()

set_source_files_properties(${BINARY_SRC_LIST} PROPERTIES GENERATED TRUE)

add_custom_target(BINARY_SRC_TARGET ALL
    DEPENDS ${BINARY_SRC_LIST} ${BINARY_TARGET_LIST}
)

# build attribute_utils.cpp for canndev ops infer code
file(GLOB ATTIBUTES_DEPENDS_FILES "${PROJECT_SOURCE_DIR}/src/kernels/include/asdops/params/*.h")
set(ATTRIBUTE_UTILS_CPP ${CMAKE_CURRENT_LIST_DIR}/attribute_utils.cpp)
add_custom_command(
    OUTPUT ${ATTRIBUTE_UTILS_CPP}
    DEPENDS ${ATTIBUTES_DEPENDS_FILES}
    COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/build_attribute_utils.py --namespace ${NAMESPACE}
                                                                           --params_path ${PROJECT_SOURCE_DIR}/src/kernels/include/asdops/params
                                                                           --dest_file_path ${ATTRIBUTE_UTILS_CPP}
)
add_custom_target(OPS_ATTRIBUTE_UTILS_CPP_TARGET DEPENDS ${ATTRIBUTE_UTILS_CPP})
set_source_files_properties(${ATTRIBUTE_UTILS_CPP} PROPERTIES GENERATED TRUE)

#########################
# 设置target名称
#########################
add_library(asdops SHARED ${BINARY_SRC_LIST} param_to_json.cpp ops.cpp ${ATTRIBUTE_UTILS_CPP})
add_dependencies(asdops BINARY_SRC_TARGET)
target_link_libraries(asdops PRIVATE ${ops_objects} tbe_adapter mki)

add_library(asdops_static STATIC $<TARGET_OBJECTS:asdops>)
add_dependencies(asdops_static BINARY_SRC_TARGET)
target_link_libraries(asdops_static PRIVATE ${ops_objects} tbe_adapter mki)

add_executable(asdops_sym_check sym_check.cpp)
target_link_libraries(asdops_sym_check PRIVATE tbe_adapter asdops mki)

install(TARGETS asdops DESTINATION lib)
install(TARGETS asdops_static DESTINATION lib)
