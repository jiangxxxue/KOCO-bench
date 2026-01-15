set(blockcopy_srcs
    ${CMAKE_CURRENT_LIST_DIR}/blockcopy_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/blockcopy_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/blockcopy_tiling.cpp
)

add_operation(BlockCopyOperation "${blockcopy_srcs}")

add_kernel(blockcopy ascend910b vector
    op_kernel/blockcopy.cpp
    BlockCopyKernel)

add_kernel(blockcopy ascend310p vector
    op_kernel/blockcopy_310p.cpp
    BlockCopyKernel)