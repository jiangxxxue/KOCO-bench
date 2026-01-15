set(gather_srcs
    ${CMAKE_CURRENT_LIST_DIR}/gather_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gather_kernel/gather_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/gather_tiling.cpp
)

add_operation(GatherOperation "${gather_srcs}")