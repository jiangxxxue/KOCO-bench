set(nonzero_src
    ${CMAKE_CURRENT_LIST_DIR}/nonzero_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/nonzero/nonzero_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/nonzero/tiling/nonzero_tiling.cpp
)

add_operation(NonzeroOperation "${nonzero_src}")

add_kernel(nonzero ascend910b vector
    nonzero/op_kernel/nonzero.cpp
    NonzeroKernel)