set(gmm_add_srcs
    ${CMAKE_CURRENT_LIST_DIR}/gmm_add_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gmm_add_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/gmm_add_tiling.cpp
)

add_operation(GmmAddOperation "${gmm_add_srcs}")

add_kernel(gmm_add ascend910b cube
    op_kernel/gmm_add.cce
    GmmAddKernel)
