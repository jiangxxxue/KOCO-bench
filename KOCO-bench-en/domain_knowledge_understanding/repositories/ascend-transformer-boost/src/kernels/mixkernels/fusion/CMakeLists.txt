set(fusion_srcs
    ${CMAKE_CURRENT_LIST_DIR}/fusion_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fusion_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/fusion_tiling.cpp
)

add_operation(FusionOperation "${fusion_srcs}")
