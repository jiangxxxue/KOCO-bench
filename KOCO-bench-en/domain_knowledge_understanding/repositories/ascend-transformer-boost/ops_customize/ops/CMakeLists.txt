set(NAMESPACE "AtbOps")
set(OP_LIST_YAML_DIR "${OPS_PROJECT_ROOT_DIR}/configs/ops")
set(OPS_CONFIG_CMAKE_DIR "${CMAKE_BINARY_DIR}/cmake/ops")

include(${MKI_PACKAGE_DIR}/cmake/host_config.cmake)
include(${MKI_PACKAGE_DIR}/cmake/kernel_config.cmake)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_SOURCE_DIR}/src/kernels
    ${CMAKE_SOURCE_DIR}/src/kernels/include
    ${CMAKE_SOURCE_DIR}/src/kernels/mixkernels
    ${CMAKE_SOURCE_DIR}/src/kernels/mixkernels/utils
    ${CMAKE_SOURCE_DIR}/src/kernels/mixkernels/utils/common/kernel
    $ENV{ASCEND_HOME_PATH}/include
)

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
 
file(GLOB_RECURSE CMAKELISTS_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/*/kernel_implement/CMakeLists.txt")
string(REPLACE "/CMakeLists.txt" "" CMAKELISTS_PATHS "${CMAKELISTS_PATHS}")
foreach(SUB_DIR ${CMAKELISTS_PATHS})
    message(STATUS "add subdirectory: ${SUB_DIR}")
    add_subdirectory(${SUB_DIR})
endforeach()

set_source_files_properties(${BINARY_SRC_LIST} PROPERTIES GENERATED TRUE)

add_custom_target(CUSTOMIZE_BINARY_SRC_TARGET ALL
    DEPENDS ${BINARY_SRC_LIST} ${BINARY_TARGET_LIST}
)

list(APPEND BINARY_SRC_LIST ${CMAKE_CURRENT_LIST_DIR}/param_to_json.cpp ${CMAKE_CURRENT_LIST_DIR}/ops.cpp)
add_library(customize_ops SHARED ${BINARY_SRC_LIST})
add_dependencies(customize_ops CUSTOMIZE_BINARY_SRC_TARGET)
target_link_libraries(customize_ops PRIVATE ${ops_objects} mki tiling_api register)
target_link_libraries(customize_ops PRIVATE
    -Wl,--whole-archive
    exp_mixops_static
    -Wl,--no-whole-archive
)

add_library(customize_ops_static STATIC $<TARGET_OBJECTS:customize_ops>)
add_dependencies(customize_ops_static CUSTOMIZE_BINARY_SRC_TARGET)
target_link_libraries(customize_ops_static PRIVATE ${ops_objects} mki tiling_api register)
target_link_libraries(customize_ops_static PRIVATE
    -Wl,--whole-archive
    exp_mixops_static
    -Wl,--no-whole-archive
)

add_executable(ops_sym_check sym_check.cpp)
target_link_libraries(ops_sym_check PRIVATE customize_ops mki)

install(TARGETS customize_ops DESTINATION lib)
install(TARGETS customize_ops_static DESTINATION lib)
