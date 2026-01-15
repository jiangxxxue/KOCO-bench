set(index_srcs
    ${CMAKE_CURRENT_LIST_DIR}/index_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/add/index_add_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/add_valid/index_add_valid_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/add/tiling/index_add_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/add_valid/tiling/index_add_valid_tiling.cpp
)

add_operation(IndexOperation "${index_srcs}")

add_kernel(index_add_valid ascend910b vector
    add_valid/op_kernel/index_add_valid.cpp
    IndexAddValidF16Kernel)