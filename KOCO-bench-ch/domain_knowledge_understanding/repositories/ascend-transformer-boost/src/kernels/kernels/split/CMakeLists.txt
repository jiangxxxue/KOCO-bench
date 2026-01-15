set(split_srcs
    ${CMAKE_CURRENT_LIST_DIR}/split_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/split/split_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/split_v/split_v_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/split/tiling/split_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/split_v/tiling/split_v_tiling.cpp
)

add_operation(SplitOperation "${split_srcs}")
