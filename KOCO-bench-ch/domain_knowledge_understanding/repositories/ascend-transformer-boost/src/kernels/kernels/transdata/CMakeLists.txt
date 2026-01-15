set(transdata_srcs
    ${CMAKE_CURRENT_LIST_DIR}/transdata_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/transdata_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/transdata_tiling.cpp
)

add_operation(TransdataOperation "${transdata_srcs}")
