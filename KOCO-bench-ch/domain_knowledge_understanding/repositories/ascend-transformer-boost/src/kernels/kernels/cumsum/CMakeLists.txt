set(cumsum_srcs
    ${CMAKE_CURRENT_LIST_DIR}/cumsum_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cumsum_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/cumsum_tiling.cpp
)

add_operation(CumsumOperation "${cumsum_srcs}")
