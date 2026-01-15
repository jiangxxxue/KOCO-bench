set(asstrided_srcs
    ${CMAKE_CURRENT_LIST_DIR}/asstrided_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/asstrided_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/asstrided_tiling.cpp
)

add_operation(AsStridedOperation "${asstrided_srcs}")