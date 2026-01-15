set(dynamic_ntk_src
    ${CMAKE_CURRENT_LIST_DIR}/dynamic_ntk_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dynamic_ntk/dynamic_ntk_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dynamic_ntk/tiling/dynamic_ntk_tiling.cpp
)

add_operation(DynamicNTKOperation "${dynamic_ntk_src}")

add_kernel(dynamic_ntk ascend910b vector
    dynamic_ntk/op_kernel/dynamic_ntk.cpp
    DynamicNTKKernel)
add_kernel(dynamic_ntk ascend310p vector
    dynamic_ntk/op_kernel/dynamic_ntk.cpp
    DynamicNTKKernel)