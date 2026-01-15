execute_process(COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/op_desc_json_auto.py
                                --namespace AtbOps
                                --params_path ${PROJECT_SOURCE_DIR}/src/kernels/include/atbops)
execute_process(COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/op_desc_json_auto.py
                                --namespace AsdOps
                                --params_path ${PROJECT_SOURCE_DIR}/src/kernels/include/asdops)
add_library(mki_test_autogen SHARED
            ${CMAKE_CURRENT_LIST_DIR}/op_desc_json_atbops.cpp
            ${CMAKE_CURRENT_LIST_DIR}/op_desc_json_asdops.cpp
            ${CMAKE_CURRENT_LIST_DIR}/op_desc_json.cpp)
target_compile_options(mki_test_autogen PUBLIC "-Wno-unused-parameter" "-Wno-unused-variable" "-Wno-unused-function")
target_compile_options(mki_test_autogen PUBLIC "-Wno-maybe-uninitialized" "-Wno-missing-field-initializers")
target_compile_options(mki_test_autogen PUBLIC "-Wno-sign-compare" "-Wno-enum-compare" "-Wno-attributes")
target_link_libraries(mki_test_autogen PRIVATE asdops atb_mixops mki)

install(TARGETS mki_test_autogen DESTINATION lib)
install(FILES ${PROJECT_SOURCE_DIR}/3rdparty/mki/tests/libmki_torch.so DESTINATION lib)