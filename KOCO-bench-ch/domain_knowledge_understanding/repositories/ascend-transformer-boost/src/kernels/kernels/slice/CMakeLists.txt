set(slice_srcs
    ${CMAKE_CURRENT_LIST_DIR}/slice_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/slice_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/slice_tiling.cpp
)

add_operation(SliceOperation "${slice_srcs}")