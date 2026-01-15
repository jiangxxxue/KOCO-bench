set(multinomial_src
    ${CMAKE_CURRENT_LIST_DIR}/multinomial_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/multinomial/multinomial_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/multinomial/tiling/multinomial_tiling.cpp
)

add_operation(MultinomialOperation "${multinomial_src}")

add_kernel(multinomial ascend910b vector
    multinomial/op_kernel/multinomial.cpp
    MultinomialKernel)
add_kernel(multinomial ascend310p vector
    multinomial/op_kernel/multinomial.cpp
    MultinomialKernel)