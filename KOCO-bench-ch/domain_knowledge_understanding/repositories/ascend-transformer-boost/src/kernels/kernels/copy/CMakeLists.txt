set(copy_srcs
    ${CMAKE_CURRENT_LIST_DIR}/copy_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/copy_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/copy_tiling.cpp
)

add_operation(CopyOperation "${copy_srcs}")