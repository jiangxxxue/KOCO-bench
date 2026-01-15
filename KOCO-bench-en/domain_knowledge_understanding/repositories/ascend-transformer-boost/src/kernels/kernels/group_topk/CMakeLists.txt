set(group_topk_srcs
    ${CMAKE_CURRENT_LIST_DIR}/group_topk_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/group_topk_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/group_topk_tiling.cpp
)

add_operation(GroupTopkOperation "${group_topk_srcs}")

add_kernel(group_topk ascend910b vector
    op_kernel/group_topk.cpp
    GroupTopkKernel)