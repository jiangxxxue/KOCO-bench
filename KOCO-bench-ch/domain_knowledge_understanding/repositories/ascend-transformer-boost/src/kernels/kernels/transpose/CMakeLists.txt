set(transpose_srcs
    ${CMAKE_CURRENT_LIST_DIR}/transpose_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/transpose_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/transpose_tiling.cpp
)

add_operation(TransposeOperation "${transpose_srcs}")