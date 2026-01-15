set(zeroslike_srcs
    ${CMAKE_CURRENT_LIST_DIR}/zeroslike_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/zeroslike_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/zeroslike_tiling.cpp
)

add_operation(ZerosLikeOperation "${zeroslike_srcs}")