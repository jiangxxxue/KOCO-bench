set(customize_blockcopy_srcs
    ${CMAKE_CURRENT_LIST_DIR}/customize_blockcopy_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/customize_blockcopy_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/customize_blockcopy_tiling.cpp
)

add_operation(CustomizeBlockCopyOperation "${customize_blockcopy_srcs}")

add_kernel(customize_blockcopy ascend910b vector
    op_kernel/customize_blockcopy.cpp
    CustomizeBlockCopyKernel)

add_kernel(customize_blockcopy ascend310p vector
    op_kernel/customize_blockcopy_310p.cpp
    CustomizeBlockCopyKernel)