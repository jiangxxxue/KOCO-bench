set(fill_srcs
    ${CMAKE_CURRENT_LIST_DIR}/fill_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fill/fill_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/masked_fill/masked_fill_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/fill_tiling.cpp
)

add_operation(FillOperation "${fill_srcs}")