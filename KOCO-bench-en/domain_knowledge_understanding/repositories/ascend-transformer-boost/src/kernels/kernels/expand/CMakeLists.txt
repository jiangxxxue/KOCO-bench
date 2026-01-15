set(expand_srcs
    ${CMAKE_CURRENT_LIST_DIR}/expand_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/expand/expand_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/expand_tiling.cpp
)

add_operation(ExpandOperation "${expand_srcs}")