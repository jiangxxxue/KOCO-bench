set(reverse_srcs
    ${CMAKE_CURRENT_LIST_DIR}/reverse_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reverse/reverse_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/reverse_tiling.cpp
)

add_operation(ReverseOperation "${reverse_srcs}")