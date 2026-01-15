set(softmax_srcs
    ${CMAKE_CURRENT_LIST_DIR}/softmax_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/softmax_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/softmax_tiling.cpp
)

add_operation(SoftmaxOperation "${softmax_srcs}")