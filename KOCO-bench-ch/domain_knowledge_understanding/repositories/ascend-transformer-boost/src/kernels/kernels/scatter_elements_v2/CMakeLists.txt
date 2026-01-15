# 设置 scatter 算子的源文件
set(scatter_elements_v2_srcs
        ${CMAKE_CURRENT_LIST_DIR}/scatter_elements_v2_operation.cpp
        ${CMAKE_CURRENT_LIST_DIR}/scatter_elements_v2_kernel.cpp
        ${CMAKE_CURRENT_LIST_DIR}/tiling/scatter_elements_v2_tiling.cpp
)

# 添加 scatter 算子
add_operation(ScatterElementsV2Operation "${scatter_elements_v2_srcs}")
