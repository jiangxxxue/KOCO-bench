set(fa_update_srcs
    ${CMAKE_CURRENT_LIST_DIR}/faUpdate_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/decode_update/decode_update_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/decode_update/tiling/decode_update_tiling.cpp
)

add_operation(FaUpdateOperation "${fa_update_srcs}")

add_kernel(decode_update ascend910b vector
    decode_update/op_kernel/decode_update.cpp
    DecodeUpdateKernel)
