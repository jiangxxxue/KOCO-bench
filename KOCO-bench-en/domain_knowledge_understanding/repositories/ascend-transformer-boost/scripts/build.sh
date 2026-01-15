#!/bin/bash
#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

set -e
SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR/..
export CODE_ROOT=$(pwd)
export CACHE_DIR=$CODE_ROOT/build
OUTPUT_DIR=$CODE_ROOT/output
THIRD_PARTY_DIR=$CODE_ROOT/3rdparty

ARCH=$(arch)
COMPILE_OPTIONS=""
COMPILE_VERBOSE=""
CMAKE_BUILD_TYPE=Release
USE_CXX11_ABI=""
USE_ASAN=OFF
USE_MSSANITIZER=OFF
USE_MSDEBUG=OFF
USE_ASCENDC_DUMP=OFF
SKIP_BUILD=OFF
BUILD_TBE_ADAPTER=OFF
CSVOPSTEST_OPTIONS=""
BUILD_PYBIND=ON
SRC_ONLY=OFF
MKI_BUILD_MODE=Test
VERSION="8.0.0"
LOG_PATH="/var/log/cann_atb_log/"
LOG_NAME="cann_atb_install.log"

BUILD_OPTION_LIST="help default testframework unittest kernelunittest pythontest torchatbtest kernelpythontest csvopstest fuzztest infratest hitest alltest clean gendoc customizeops"
BUILD_CONFIGURE_LIST=("--verbose" "--use_cxx11_abi=0" "--use_cxx11_abi=1"
    "--asan" "--skip_build" "--csvopstest_options=.*" "--debug" "--clean-first" "--msdebug" "--ascendc_dump" "--mssanitizer" "--no-pybind"
    "--src-only" "--customizeops_tests")

function fn_build_googletest()
{
    if [ -d "$THIRD_PARTY_DIR/googletest/lib" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    [[ ! -d "googletest" ]] && git clone --branch v1.13.0 --depth 1 https://github.com/google/googletest.git
    cd googletest
    sed -i '34d' CMakeLists.txt
    rm -rf build && mkdir build && cd build
    if [ "$USE_CXX11_ABI" == "ON" ]
    then
        sed -i '33 a add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)' ../CMakeLists.txt
    else
        sed -i '33 a add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)' ../CMakeLists.txt
    fi
    cmake .. -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/googletest -DCMAKE_SKIP_RPATH=TRUE -DCMAKE_CXX_FLAGS="-fPIC"
    cmake --build . --parallel $(nproc)
    cmake --install . > /dev/null
    [[ -d "$THIRD_PARTY_DIR/googletest/lib64" ]] && cp -rf $THIRD_PARTY_DIR/googletest/lib64 $THIRD_PARTY_DIR/googletest/lib
    echo "Googletest is successfully installed to $THIRD_PARTY_DIR/googletest"
}

function fn_build_stub()
{
    if [ -f "$THIRD_PARTY_DIR/cpp-stub/src/stub.h" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    git clone --depth 1 https://github.com/coolxv/cpp-stub.git
}

function fn_build_doxygen()
{
    if [ -d "$THIRD_PARTY_DIR/doxygen/bin" ]; then
        return 0
    fi
    sys_info=$(awk -F= '/^NAME/{print $2}' /etc/os-release)
    if [ "$sys_info" == *"EulerOS"* ]; then
        yum -y install flex bison
    elif [ "$sys_info" == *"Ubuntu"* ]; then
        apt-get -y install flex bison
    fi
    cd $THIRD_PARTY_DIR
    if [ ! -d "$THIRD_PARTY_DIR/doxygen" ]; then
        [[ ! -f "doxygen-1.9.3.src.tar.gz" ]] && wget --no-check-certificate https://github.com/doxygen/doxygen/releases/download/Release_1_9_3/doxygen-1.9.3.src.tar.gz
        tar -xzvf doxygen-1.9.3.src.tar.gz
        mv doxygen-1.9.3 doxygen
    fi
    cd doxygen
    rm -rf build && mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/doxygen
    cmake --build . --parallel $(nproc)
    cmake --install . > /dev/null
}

function fn_build_sphinx()
{
    [[ "$COVERAGE_TYPE" != "" ]] && return 0
    pip install -U sphinx
    pip install sphinx_rtd_theme
    pip install myst_parser
    pip install breathe
}

function fn_gen_doc()
{
    if [ "${SKIP_BUILD}" == "ON" ]; then
        echo "info: skip generate doc because SKIP_BUILD is on."
        return 0
    fi

    cd $CODE_ROOT
    branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || git rev-parse HEAD)
    export LD_LIBRARY_PATH=$CODE_ROOT/3rdparty/graphviz/lib:$LD_LIBRARY_PATH
    local doxyfile=$CODE_ROOT/docs/Doxyfile
    local doxygen_output_dir=$CODE_ROOT/docs/$branch
    [[ -f "$doxyfile" ]] && rm -rf $doxyfile
    [[ -d "$doxygen_output_dir" ]] && rm -rf $doxygen_output_dir
    mkdir -p $doxygen_output_dir
    $THIRD_PARTY_DIR/doxygen/bin/doxygen -g $doxyfile
    sed -i "s#PROJECT_NAME           =.*#PROJECT_NAME           = "AscendTransformerBoost"#g" $doxyfile
    sed -i "s#PROJECT_NUMBER         =.*#PROJECT_NUMBER         = $branch#g" $doxyfile
    sed -i "s#OUTPUT_DIRECTORY       =.*#OUTPUT_DIRECTORY       = $doxygen_output_dir#g" $doxyfile
    sed -i "s#OUTPUT_LANGUAGE        =.*#OUTPUT_LANGUAGE        = Chinese#g" $doxyfile
    sed -i "s#INPUT                  =.*#INPUT                  = $CODE_ROOT/include/atb $CODE_ROOT/docs/doxygen/atb_document.txt#g" $doxyfile
    sed -i "s#RECURSIVE              =.*#RECURSIVE              = YES#g" $doxyfile
    sed -i "s#USE_MDFILE_AS_MAINPAGE =.*#USE_MDFILE_AS_MAINPAGE = $CODE_ROOT/README.md#g" $doxyfile
    sed -i "s#HTML_EXTRA_STYLESHEET  =.*#HTML_EXTRA_STYLESHEET  = $CODE_ROOT/docs/doxygen/custom.css#g" $doxyfile
    sed -i "s#GENERATE_LATEX         =.*#GENERATE_LATEX         = NO#g" $doxyfile
    sed -i "s#HAVE_DOT               =.*#HAVE_DOT               = NO#g" $doxyfile
    sed -i "s#USE_MATHJAX            =.*#USE_MATHJAX            = YES#g" $doxyfile
    sed -i "s#WARN_NO_PARAMDOC       =.*#WARN_NO_PARAMDOC       = YES#g" $doxyfile
    sed -i "s#GENERATE_TREEVIEW      =.*#GENERATE_TREEVIEW      = YES#g" $doxyfile
    sed -i "s#WARN_AS_ERROR          =.*#WARN_AS_ERROR          = YES#g" $doxyfile
    sed -i "s#GENERATE_XML           =.*#GENERATE_XML           = YES#g" $doxyfile
    $THIRD_PARTY_DIR/doxygen/bin/doxygen $doxyfile
    [[ "$COVERAGE_TYPE" != "" ]] && return 0
    local sphinx_output_dir=$CODE_ROOT/docs/$branch/guide
    [[ -d "$sphinx_output_dir" ]] && rm -rf $sphinx_output_dir
    mkdir -p $sphinx_output_dir
    sphinx-build -M html $CODE_ROOT/docs $sphinx_output_dir
}

function fn_build_mki()
{
    if [ "$MKI_BUILD_MODE" == "Test" -a -d "$THIRD_PARTY_DIR/mki/tests" ]; then
        return 0
    fi
    if [ "$MKI_BUILD_MODE" == "Dev" -a -d "$THIRD_PARTY_DIR/mki/include" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    if [ ! -d "Mind-KernelInfra" ]; then
        branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || echo "commit_id")
        [[ "$branch" == *br_personal* || "$branch" == "commit_id" || "$branch" == *revert-mr* ]] && branch=master
        echo  "current branch for mki: $branch"
        git clone --branch $branch --depth 1 https://gitee.com/ascend/Mind-KernelInfra.git
    fi
    cd Mind-KernelInfra
    echo  "current commid id of Mind-KernelInfra: $(git rev-parse HEAD)"
    if [ "$USE_CXX11_ABI" == "ON" ];then
        build_options="$build_options --use_cxx11_abi=1"
    else
        build_options="$build_options --use_cxx11_abi=0"
    fi
    if [ "$MKI_BUILD_MODE" == "Test" ]; then
        echo "mki build by Test mode"
        mkdir -p $THIRD_PARTY_DIR/Mind-KernelInfra/3rdparty
        cp -r $THIRD_PARTY_DIR/nlohmannJson $THIRD_PARTY_DIR/Mind-KernelInfra/3rdparty
        build_type=testframework
    elif [ "$CMAKE_BUILD_TYPE" == "Release" ]; then
        echo "mki build by Develop mode"
        build_type=dev
    else
        echo "mki build by Debug mode"
        build_type=debug
    fi
    if [ "$USE_MSDEBUG" == "ON" ]; then
        build_options="$build_options --msdebug"
    fi
    if [ "$USE_ASCENDC_DUMP" == "ON" ]; then
        build_options="$build_options --ascendc_dump"
    fi
    build_options="$build_options --output=$THIRD_PARTY_DIR $COMPILE_VERBOSE"
    bash scripts/build.sh $build_type $build_options
}

function fn_build_catlass()
{
    if [ -d "$THIRD_PARTY_DIR/catlass" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    branch=catlass-v1-stable
    echo  "current branch for catlass: $branch"
    git clone --branch $branch --depth 1 https://gitee.com/ascend/catlass.git
}

function fn_build_nlohmann_json()
{
    if [ -d "$THIRD_PARTY_DIR/nlohmannJson" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    git clone --branch v3.11.3 --depth 1 https://github.com/nlohmann/json.git
    mv json nlohmannJson
}

function fn_build_pybind11()
{
    if [ -d "$THIRD_PARTY_DIR/pybind11" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    git clone --branch v2.10.3 --depth 1 https://github.com/pybind/pybind11.git
}

function fn_build_secodefuzz()
{
    FUZZ_DEST_PATH=$THIRD_PARTY_DIR/secodefuzz
    if [ -d "$FUZZ_DEST_PATH/lib/" ]; then
        return 0
    fi
    cd $CACHE_DIR
    rm -rf secodefuzz
    git clone -b v2.4.5 --depth=1 https://szv-open.codehub.huawei.com/innersource/Fuzz/secodefuzz.git
    cd secodefuzz
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --target Secodefuzz --parallel $(nproc)
    mkdir -p $FUZZ_DEST_PATH/include
    mkdir -p $FUZZ_DEST_PATH/lib
    mv ../Secodefuzz/secodeFuzz.h $FUZZ_DEST_PATH/include
    mv libSecodefuzz.a $FUZZ_DEST_PATH/lib
}

function fn_build_cann_dependency()
{
    [[ -d "$THIRD_PARTY_DIR/compiler" ]] && rm -rf $THIRD_PARTY_DIR/compiler
    mkdir -p $THIRD_PARTY_DIR/compiler
    CCEC_COMPILER_DIR=$THIRD_PARTY_DIR/compiler/ccec_compiler
    TIKCPP_DIR=$THIRD_PARTY_DIR/compiler/tikcpp
    ln -s $ASCEND_HOME_PATH/compiler/ccec_compiler $CCEC_COMPILER_DIR
    ln -s $ASCEND_HOME_PATH/compiler/tikcpp $TIKCPP_DIR
}

function fn_build_3rdparty_for_test()
{
    if [ "${SKIP_BUILD}" == "ON" ]; then
        echo "info: skip atb test 3rdparty because SKIP_BUILD is on."
        return 0
    fi
    fn_build_googletest
    fn_build_stub
}

function fn_get_cxx_abi_string()
{
    if [ "${USE_CXX11_ABI}" == "ON" ]; then
        echo "cxx_abi_1"
    else
        echo "cxx_abi_0"
    fi
}

function fn_copy_tbe_adapter()
{
    if [ -z $ATB_BUILD_DEPENDENCY_PATH ]; then
        ATB_BUILD_DEPENDENCY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/$(fn_get_cxx_abi_string)
        echo "warning: ATB_BUILD_DEPENDENCY_PATH is not set. Using default path: /usr/local/Ascend/nnal/atb/latest/atb/$(fn_get_cxx_abi_string)"
    fi

    if [ ! -f $ATB_BUILD_DEPENDENCY_PATH/lib/libtbe_adapter.so ]; then
        echo "error:$ATB_BUILD_DEPENDENCY_PATH/lib/libtbe_adapter.so dose not exist, please set correct ATB_BUILD_DEPENDENCY_PATH"
        return 0
    fi

    LOCAL_ABI=$(fn_get_cxx_abi_string)

    LOCAL_TBE_ADAPTER_PATH=$CODE_ROOT/output/atb/$LOCAL_ABI
    TARGET_ABI=$(basename "$ATB_BUILD_DEPENDENCY_PATH")
    if [ "${TARGET_ABI}" != "${LOCAL_ABI}" ];then
        echo "$ATB_BUILD_DEPENDENCY_PATH use $TARGET_ABI, but $LOCAL_TBE_ADAPTER_PATH use $LOCAL_ABI, abi error."
        return 0
    fi

    if [ "${ATB_BUILD_DEPENDENCY_PATH}" != "${LOCAL_TBE_ADAPTER_PATH}" ]; then
        if [ ! -d "${LOCAL_TBE_ADAPTER_PATH/lib}" ];then
            mkdir -p $LOCAL_TBE_ADAPTER_PATH/lib
        fi
        cp ${ATB_BUILD_DEPENDENCY_PATH}/lib/libtbe_adapter.so $LOCAL_TBE_ADAPTER_PATH/lib
    fi
}

function fn_build_tbe_dependency()
{
    LOCAL_ABI=$(fn_get_cxx_abi_string)
    if [ -f $OUTPUT_DIR/atb/$LOCAL_ABI/lib/libtbe_adapter.so ];then
        echo "libtbe_adapter.so is already exist, skip build process."
        return 0
    fi

    #copy from nnal
    fn_copy_tbe_adapter
}
function fn_build_3rdparty_for_compile()
{
    fn_build_nlohmann_json
    fn_build_mki
    fn_build_catlass
    fn_build_cann_dependency
    fn_build_tbe_dependency
    if [ "$BUILD_PYBIND" == "ON" -a "$USE_CXX11_ABI" != "ON" ]; then
        fn_build_pybind11
    fi
    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DBUILD_TBE_ADAPTER=$BUILD_TBE_ADAPTER"
}

function fn_build_3rdparty_for_doc()
{
    if [ "${SKIP_BUILD}" == "ON" ]; then
        echo "info: skip atb doc 3rdparty because SKIP_BUILD is on."
        return 0
    fi
    fn_build_doxygen
    fn_build_sphinx
}

function export_atb_env()
{
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH}
    source $OUTPUT_DIR/atb/set_env.sh
}

function export_atb_hitest_env()
{
    export isOverlappedCompile=0
    export PlatformToken=BOARD
    export gcovmode=0
    export TimerPolicy=1         #是否开启线程定时器采集覆盖率数据, 1: 开启, 0: 不开启
    export TimeInterval=60       #线程定时器，各隔多少秒采集一次数据，60表示60秒
    export SignalPolicy=1        #是否开启发信号采集覆盖率，1: 开启, 0: 不开启
    export SignalNUM=34          #kill -34 pid 采覆盖率， kill -44 pid 重置覆盖率
    export lltwrapper_cfg=0      # 0: 普通模式，4: 无OS极简模式(单模块), 5: 无OS通用模式
    export HITEST_AGENT_INSIDE=1 # 1: 使用内嵌agent.o, 0: 不使用内嵌agent.o
    export USE_HLLT_COVERAGE=1
    export USE_HLLT_TESTCASE=0
    export simplemode=0	                  # 0: 非精简模式, 1: 精简模式
    export ncs_coverage_stub_mold=1       # 0: 非计数模式, 1: 计数模式
    export hitest_disable_cfg=0           # 是否需要导出CFG控制流图, 默认导出 0:导出  1:不导出
    export hitest_disable_dfg=1           # 是否需要导出污点数据, 默认不导出 0:导出  1:不导出
    export lltcovRootpath=/opt/covdata	  #(测试执行环境中，覆盖率数据保存的根目录)
    export PATH=/home/slave1/hitest/linux_avatar_${ARCH}:$PATH	        #(添加工具包到PATH环境变量)
    export LD_LIBRARY_PATH=/home/slave1/hitest/linux_avatar_${ARCH}:$LD_LIBRARY_PATH	    #(添加工具包到LD_LIBRARY_PATH环境变量)
    export LLT_INCLUDEPATH=${CODE_ROOT}/core  #(指定需要插桩的代码目录，用冒号:间隔，可指定多个代码目录)
    export LLT_EXCLUDEPATH=${CODE_ROOT}/tests #指定需要排除不插桩的路径，即，指定的路径下的所有源码都不会被插桩
    #清空上一次构建的模型数据
    rm -rf /home/slave1/hitest/linux_avatar_${ARCH}/apache-tomcat-8.0.39/webapps/base/covstub
    rm -rf /home/slave1/hitest/linux_avatar_${ARCH}/apache-tomcat-8.0.39/webapps/base/covdata
}

function generate_atb_version_info()
{
    branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || echo $branch)
    commit_id=$(git rev-parse HEAD)
    touch $OUTPUT_DIR/version.info
    cat>$OUTPUT_DIR/version.info<<EOF
    Platform : ${ARCH}
    branch : ${branch}
    commit id : ${commit_id}
EOF
}

function fn_init_env()
{
    res=$(python3 -c "import torch" &> /dev/null || echo "torch_not_exist")
    if [ "$res" == "torch_not_exist" ]; then
        echo "Warning: Torch is not installed!"
        [[ "$USE_CXX11_ABI" == "" ]] && USE_CXX11_ABI=ON
    fi
    if [ "$USE_CXX11_ABI" == "" ]; then
        if [ $(python3 -c 'import torch; print(torch.compiled_with_cxx11_abi())') == "True" ]; then
            USE_CXX11_ABI=ON
        else
            USE_CXX11_ABI=OFF
        fi
    fi
    export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
    export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
    export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
    export PYTORCH_NPU_INSTALL_PATH="$(python3 -c 'import torch, torch_npu, os; print(os.path.dirname(os.path.abspath(torch_npu.__file__)))')"
    export LD_LIBRARY_PATH=$PYTORCH_INSTALL_PATH/../torch.libs:$LD_LIBRARY_PATH

    echo "PYTHON_INCLUDE_PATH=$PYTHON_INCLUDE_PATH"
    echo "PYTHON_LIB_PATH=$PYTHON_LIB_PATH"
    echo "PYTORCH_INSTALL_PATH=$PYTORCH_INSTALL_PATH"
    echo "PYTORCH_NPU_INSTALL_PATH=$PYTORCH_NPU_INSTALL_PATH"
    echo "USE_CXX11_ABI=$USE_CXX11_ABI"
}

function rename_whl_file() {
    python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    version_suffix="cp${python_version//./}"
    echo "DEBUG: Current Python version: $python_version, setting wheel file tag: $version_suffix"
    for whl in ./whl/*.whl; do
        if [[ "$whl" == *"py3"* ]]; then
            new_name="${whl/py3/${version_suffix}}"
            echo "DEBUG: Renaming $whl to $new_name"
            mv "$whl" "$new_name"
        fi
    done
}

function fn_gen_atb_whl()
{
    cd $CODE_ROOT/output
    cp -rf $CODE_ROOT/pyproject.toml .
    mkdir -p ./torch_atb
    cp -rf $CODE_ROOT/torch_atb/* ./torch_atb
    cp -rf ./atb/cxx_abi_0/* ./torch_atb
    pip wheel --no-deps --no-build-isolation --wheel-dir ./whl .
    rename_whl_file
    rm -rf ./torch_atb
    rm -rf ./torch_atb.egg-info
    rm -rf ./pyproject.toml
    rm -rf ./build
}

function fn_make_run_package()
{
    mkdir -p $OUTPUT_DIR/scripts
    mkdir -p $CODE_ROOT/ci/$ARCH
    cp $CODE_ROOT/scripts/install.sh $OUTPUT_DIR
    cp $CODE_ROOT/scripts/set_env.sh $OUTPUT_DIR
    cp $CODE_ROOT/scripts/uninstall.sh $OUTPUT_DIR/scripts
    cp $CODE_ROOT/scripts/filelist.csv $OUTPUT_DIR/scripts
    sed -i "s/ATBPKGARCH/${ARCH}/" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/install.sh
    sed -i "s!LOG_PATH_PLACEHOLDER!${LOG_PATH}!" $OUTPUT_DIR/install.sh
    sed -i "s!LOG_NAME_PLACEHOLDER!${LOG_NAME}!" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/scripts/uninstall.sh
    sed -i "s!LOG_PATH_PLACEHOLDER!${LOG_PATH}!" $OUTPUT_DIR/scripts/uninstall.sh
    sed -i "s!LOG_NAME_PLACEHOLDER!${LOG_NAME}!" $OUTPUT_DIR/scripts/uninstall.sh
    sed -i 's|${atb_path}|${atb_path}/latest/atb|g' $OUTPUT_DIR/set_env.sh

    chmod +x $OUTPUT_DIR/*
    makeself_dir=${ASCEND_HOME_PATH}/toolkit/tools/op_project_templates/ascendc/customize/cmake/util/makeself/
    ${makeself_dir}/makeself.sh --header ${makeself_dir}/makeself-header.sh \
       --help-header $CODE_ROOT/scripts/help.info --gzip --complevel 4 --nomd5 --sha256 --chown \
        ${OUTPUT_DIR} $CODE_ROOT/ci/$ARCH/Ascend-cann-atb_${VERSION}_linux-${ARCH}.run "Ascend-cann-atb-api" ./install.sh

    mv $CODE_ROOT/ci/$ARCH $OUTPUT_DIR
    echo "Ascend-cann-atb_${VERSION}_linux-${ARCH}.run is successfully generated in $OUTPUT_DIR"
}

function fn_build()
{
    if [ "${ASCEND_HOME_PATH}" == "" ]; then
        echo "error: build failed because ASCEND_HOME_PATH is null, please source cann set_env.sh first."
        exit 1
    fi
    if [ "${SKIP_BUILD}" == "ON" ]; then
        echo "info: skip atb build because SKIP_BUILD is on."
        return 0
    fi
    if [ "${USE_ASCENDC_DUMP}" == "OFF" ] && [[ -f "$THIRD_PARTY_DIR/mki/lib/libmki.so" ]]; then
        if readelf -d $THIRD_PARTY_DIR/mki/lib/libmki.so | grep -q "libascend_dump.so"; then
            rm -rf $THIRD_PARTY_DIR/mki
            [[ -d "$CACHE_DIR" ]] && rm -rf $CACHE_DIR && mkdir $CACHE_DIR
        fi
    fi
    fn_build_3rdparty_for_compile
    cd $CACHE_DIR
    if [ "$CMAKE_CXX_COMPILER_LAUNCHER" == "" -a command -v ccache &> /dev/null ]; then
        COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    fi
    if [ "${USE_ASCENDC_DUMP}" == "ON" ]; then
        COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_ASCENDC_DUMP=ON"
    fi
    echo "COMPILE_OPTIONS:$COMPILE_OPTIONS"
    cmake $CODE_ROOT $COMPILE_OPTIONS
    cmake --build . --parallel $COMPILE_VERBOSE
    cmake --install .

    if [ "$BUILD_PYBIND" == "ON" -a "$USE_CXX11_ABI" != "ON" ]; then
        fn_gen_atb_whl
    fi
}

function pack_testframework()
{
    cd $CODE_ROOT/output
    rm -f atb_testframework*.tar.gz
    pack_name=atb_testframework_${ARCH}_$(date +%Y%m%d%H%M).tar.gz
    tar -zcf $pack_name ./atb
    echo "$pack_name is generated in path $CODE_ROOT/output"
}

function pack_hitest()
{
    mkdir -p opt/HiTest/apache-tomcat-8.0.39/webapps/covroot
    cp -rf /home/slave1/hitest/linux_avatar_${ARCH}/apache-tomcat-8.0.39/webapps/base/covstub opt/HiTest/apache-tomcat-8.0.39/webapps/covroot
    tar -zcf hitest_atb-${ARCH}_"$(date +%Y%m%d%H)".tgz opt/HiTest/apache-tomcat-8.0.39/webapps/covroot/covstub
    rm -rf opt
    mkdir -p $CODE_ROOT/output/HiTest
    mv hitest_atb-${ARCH}_"$(date +%Y%m%d%H)".tgz $CODE_ROOT/output/HiTest
    mv atb_testframework*.tar.gz $CODE_ROOT/output/HiTest
    tar -zcf high_level_test.tar.gz $CODE_ROOT/tests/high_level_test
    mv high_level_test.tar.gz $CODE_ROOT/output/HiTest
}

function fn_run_unittest()
{
    export_atb_env
    export LD_LIBRARY_PATH=$PYTORCH_INSTALL_PATH/lib:$PYTORCH_NPU_INSTALL_PATH/lib:$LD_LIBRARY_PATH
    echo "run $ATB_HOME_PATH/bin/atb_unittest"
    $ATB_HOME_PATH/bin/atb_unittest --gtest_output=xml:test_detail.xml
}

function fn_run_kernel_unittest()
{
    export_atb_env
    export LD_LIBRARY_PATH=$PYTORCH_INSTALL_PATH/lib:$PYTORCH_NPU_INSTALL_PATH/lib:$LD_LIBRARY_PATH
    echo "run $ATB_HOME_PATH/bin/kernels_unittest"
    $ATB_HOME_PATH/bin/kernels_unittest --gtest_output=xml:kernel_test_detail.xml
}

function fn_run_kernel_cinterfacetest()
{
    export_atb_env
    export LD_LIBRARY_PATH=$PYTORCH_INSTALL_PATH/lib:$PYTORCH_NPU_INSTALL_PATH/lib:$LD_LIBRARY_PATH
    echo "run $ATB_HOME_PATH/bin/atb_cinterface"
    $ATB_HOME_PATH/bin/atb_cinterface
}


function fn_run_fuzztest()
{
    export_atb_env
    export LD_LIBRARY_PATH=$PYTORCH_INSTALL_PATH/lib:$PYTORCH_NPU_INSTALL_PATH/lib:$LD_LIBRARY_PATH
    echo "run $ATB_HOME_PATH/bin/atb_fuzztest"
    $ATB_HOME_PATH/bin/atb_fuzztest
}

function fn_build_coverage()
{
    GCOV_DIR=$CACHE_DIR/gcov
    PYYTHON_FILTER_TOOL=$CODE_ROOT/tests/framework/python/GcovFilterTool.py
    LCOV_PATH=$CACHE_DIR
    GENHTML_PATH=$CACHE_DIR

    if command -v lcov &> /dev/null; then
        LCOV_PATH=$(which lcov)
    else
        echo "Cannot access lcov from commandline, try /usr/local/lcov/bin/lcov"
        LCOV_PATH=/usr/local/lcov/bin/lcov
    fi

    if command -v genhtml &> /dev/null; then
        GENHTML_PATH=$(which genhtml)
    else
        echo "Cannot access genhtml from commandline, try /usr/local/lcov/bin/genhtml"
        GENHTML_PATH=/usr/local/lcov/bin/genhtml
    fi

    if [ -d "$GCOV_DIR" ]
    then
        rm -rf $GCOV_DIR
    fi
    mkdir $GCOV_DIR

    if [ "$COVERAGE_TYPE" == "ALLTEST" ];then
        fn_run_csvopstest
        fn_run_unittest
        fn_run_pythontest
        fn_run_torchatbtest
        fn_run_kernel_cinterfacetest
    fi
    if [ "$COVERAGE_TYPE" == "FUZZTEST" ];then
        fn_run_fuzztest
    fi

    cd $GCOV_DIR
    $LCOV_PATH -c --directory ./.. --output-file tmp_coverage.info --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt
    $LCOV_PATH -r tmp_coverage.info '*/3rdparty/*' '*torch/*' '*c10/*' '*ATen/*' '*/c++/7*' '*tests/*' '*tools/*' '/usr/*' '/opt/*' '*models/*' '*build/*' '*output/*' -output-file test_coverage.info --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt
    $LCOV_PATH -a test_coverage.info -o main_coverage.info --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt
    python3 $PYYTHON_FILTER_TOOL --input ./main_coverage.info --output ./final.info --root $CACHE_DIR --debug 1 >> $GCOV_DIR/log.txt
    $GENHTML_PATH --branch-coverage final.info -o cover_result --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt
    tail -n 4 $GCOV_DIR/log.txt
    cd ..
    tar -czf gcov.tar.gz gcov
    mv gcov.tar.gz $OUTPUT_DIR/atb/
    cd $OUTPUT_DIR/atb/
    tar -zcf csvopstest.tar.gz csvopstest
}

function fn_run_torchatbtest()
{
    export_atb_env
    fn_install_torch_atb
    cd $CODE_ROOT/tests/apitest/torch_atb_test
    for testdir in $(ls -d *_test); do
        echo "Running tests in $testdir"
        for testfile in $(find "$testdir" -name "*_test.py"); do
            file_only="${testfile##*/}"
            python3 -m unittest discover -s $CODE_ROOT/tests/apitest -p "$file_only" -v
        done
    done
}

function fn_run_pythontest()
{
    export_atb_env
    cd $CODE_ROOT/tests/apitest/opstest/python/
    rm -rf ./kernel_meta*
    for i in $(ls -d operations/*/); do
        if [[ $(find $i -name __init__.py) != "" ]];then
            python3 -m unittest discover -s ./$i -p "*test*.py";
        fi
    done
}

function fn_run_kernel_pythontest()
{
    export_atb_env
    cd $CODE_ROOT/tests/apitest/kernelstest/
    rm -rf ./kernel_meta*
    for i in $(ls -d */); do
        python3 -m unittest discover -s ./$i -p "*test*.py"
    done
}

function fn_run_csvopstest()
{
    export_atb_env
    cd $CODE_ROOT/tests/framework/python/CsvOpsTestTool
    python3 ./atb_csv_ops_test.py $CSVOPSTEST_OPTIONS
}

function fn_run_infratest()
{
    export_atb_env
    find $CODE_ROOT/tests/infratest -type f -name "*.py" -exec python3 {} \;
}

function fn_install_torch_atb()
{
    py_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    py_major_version=${py_version%%.*}
    py_minor_version=${py_version##*.}

    if [ "$py_major_version" != "3" ] || { [ "$py_minor_version" != "9" ] && [ "$py_minor_version" != "10" ] && [ "$py_minor_version" != "11" ]; }; then
        echo "ERROR: Unsupported Python version. Only Python 3.9, 3.10, and 3.11 are supported."
        exit 1
    fi

    wheel_file="torch_atb-0.0.1-cp${py_major_version}${py_minor_version}-none-any.whl"
    wheel_path="${OUTPUT_DIR}/whl/${wheel_file}"

    if [ ! -f "$wheel_path" ]; then
        echo "ERROR: Wheel file ${wheel_file} not found at ${wheel_path}."
        exit 1
    fi

    if pip3 show torch_atb > /dev/null 2>&1; then
        echo "INFO: torch_atb is already installed, force-reinstalling..."
        if pip3 install --force-reinstall "$wheel_path" > /dev/null 2>&1; then
            echo "INFO: torch_atb reinstallation succeeded!"
        else
            echo "ERROR: torch_atb reinstallation failed!"
            exit 1
        fi
    else
        echo "INFO: torch_atb not found, installing..."
        if pip3 install "$wheel_path" > /dev/null 2>&1; then
            echo "INFO: torch_atb installation succeeded!"
        else
            echo "ERROR: torch_atb installation failed!"
            exit 1
        fi
    fi
}

function fn_main()
{
    if [[ "$BUILD_OPTION_LIST" =~ "$1" ]]; then
        if [[ -z "$1" ]]; then
            arg1="default"
        else
            arg1=$1
            shift
        fi
    else
        cfg_flag=0
        for item in ${BUILD_CONFIGURE_LIST[*]}; do
            if [[ "$1" =~ $item ]]; then
                cfg_flag=1
                break 1
            fi
        done
        if [[ "$cfg_flag" == 1 ]]; then
            arg1="default"
        else
            echo "argument $1 is unknown, please type build.sh help for more imformation"
            exit 1
        fi
    fi

    until [[ -z "$1" ]]
    do {
        arg2=$1
        case "${arg2}" in
        "--use_cxx11_abi=1")
            USE_CXX11_ABI=ON
            ;;
        "--use_cxx11_abi=0")
            USE_CXX11_ABI=OFF
            ;;
        "--verbose")
            COMPILE_VERBOSE="--verbose"
            ;;
        "--asan")
            USE_ASAN=ON
            CMAKE_BUILD_TYPE=Debug
            ;;
        "--skip_build")
            SKIP_BUILD=ON
            ;;
        --csvopstest_options=*)
            arg2=${arg2#*=}
            if [ ! -z "$arg2" ];then
                CSVOPSTEST_OPTIONS=$arg2
            fi
            ;;
        "--debug")
            CMAKE_BUILD_TYPE=Debug
            ;;
        "--msdebug")
            USE_MSDEBUG=ON
            CHIP_TYPE=$(npu-smi info -m | grep -oE 'Ascend\s*\S+' | head -n 1 | tr -d ' ' | tr '[:upper:]' '[:lower:]')
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_MSDEBUG=ON -DCHIP_TYPE=${CHIP_TYPE}"
            ;;
        "--ascendc_dump")
            if [[ -f "$THIRD_PARTY_DIR/mki/lib/libmki.so" ]]; then
                if ! readelf -d $THIRD_PARTY_DIR/mki/lib/libmki.so | grep -q "libascend_dump.so"; then
                    rm -rf $THIRD_PARTY_DIR/mki
                fi
            fi
            USE_ASCENDC_DUMP=ON
            ;;
        "--mssanitizer")
            USE_MSSANITIZER=ON
            ;;
        "--clean-first")
            [[ -d "$CACHE_DIR" ]] && rm -rf $CACHE_DIR
            [[ -d "$OUTPUT_DIR" ]] && rm -rf $OUTPUT_DIR
            [[ -d "$THIRD_PARTY_DIR" ]] && rm -rf $THIRD_PARTY_DIR
            ;;
        "--no-pybind")
            BUILD_PYBIND=OFF
            ;;
        "--src-only")
            SRC_ONLY=ON
            ;;
        "--customizeops_tests")
            fn_build_googletest
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DBUILD_CUSTOMIZE_OPS_TEST=ON"
            ;;
        esac
        shift
    }
    done

    [[ ! -d "$CACHE_DIR" ]] && mkdir $CACHE_DIR
    [[ ! -d "$OUTPUT_DIR" ]] && mkdir $OUTPUT_DIR
    [[ ! -d "$THIRD_PARTY_DIR" ]] && mkdir $THIRD_PARTY_DIR
    fn_init_env

    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
    -DUSE_CXX11_ABI=$USE_CXX11_ABI -DUSE_ASAN=$USE_ASAN -DBUILD_PYBIND=$BUILD_PYBIND -DUSE_MSSANITIZER=$USE_MSSANITIZER \
    -DPACKAGE_COMPILE=OFF"
    case "${arg1}" in
        "default")
            MKI_BUILD_MODE=Dev
            fn_build
            generate_atb_version_info
            fn_make_run_package
            ;;
        "testframework")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DBUILD_TEST_FRAMEWORK=ON"
            fn_build
            generate_atb_version_info
            pack_testframework
            ;;
        "unittest")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_UNIT_TEST=ON"
            fn_build_3rdparty_for_test
            fn_build
            fn_run_kernel_cinterfacetest
            fn_run_unittest
            ;;
        "kernelunittest")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_UNIT_TEST=ON"
            fn_build_3rdparty_for_test
            fn_build
            fn_run_kernel_unittest
            ;;
        "pythontest")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_PYTHON_TEST=ON"
            fn_build
            fn_run_pythontest
            ;;
        "kernelpythontest")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_PYTHON_TEST=ON"
            fn_build
            fn_run_kernel_pythontest
            ;;
        "torchatbtest")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_TORCH_ATB_TEST=ON"
            fn_build
            fn_run_torchatbtest
            ;;
        "csvopstest")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_CSV_OPS_TEST=ON"
            fn_build
            fn_run_csvopstest
            ;;
        "infratest")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_INFRA_TEST=ON"
            fn_build
            fn_run_infratest
            ;;
        "hitest")
            export_atb_hitest_env
            export CMAKE_CXX_COMPILER_LAUNCHER=hitestwrapper
            export CMAKE_CXX_LINKER_LAUNCHER=hitestwrapper
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DBUILD_TEST_FRAMEWORK=ON"
            fn_build
            generate_atb_version_info
            pack_testframework
            pack_hitest
            ;;
        "fuzztest")
            COVERAGE_TYPE="FUZZTEST"
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_FUZZ_TEST=ON"
            RANDOM_SEED=0
            TEST_TIME=30
            python3 $CODE_ROOT/tests/apitest/fuzztest/generate_operation_fuzz_test.py $CODE_ROOT $RANDOM_SEED $TEST_TIME
            fn_build_3rdparty_for_test
            fn_build_secodefuzz
            fn_build
            fn_build_coverage
            ;;
        "alltest")
            COVERAGE_TYPE="ALLTEST"
            fn_build_3rdparty_for_doc
            fn_gen_doc
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_ALL_TEST=ON"
            CSVOPSTEST_OPTIONS="-i $CODE_ROOT/tests/apitest/opstest/csv/ -o $CODE_ROOT/output/atb/csvopstest -tt Function"
            fn_build_3rdparty_for_test
            fn_build
            fn_build_coverage
            ;;
        "clean")
            [[ -d "$CACHE_DIR" ]] && rm -rf $CACHE_DIR
            [[ -d "$OUTPUT_DIR" ]] && rm -rf $OUTPUT_DIR
            [[ -d "$THIRD_PARTY_DIR" ]] && rm -rf $THIRD_PARTY_DIR
            echo "clean all build history"
            ;;
        "gendoc")
            fn_build_3rdparty_for_doc
            fn_gen_doc
            ;;
        "customizeops")
            MKI_BUILD_MODE=Dev
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DBUILD_CUSTOMIZE_OPS=ON"
            fn_build
            generate_atb_version_info
            fn_make_run_package
            ;;
        *)
            echo "Usage: "
            echo "run build.sh help|default|testframework|unittest|kernelunittest|pythontest|kernelpythontest|torchatbtest|csvopstest|infratest|fuzztest|alltest|clean|gendoc|customizeops| --debug|--verbose|--use_cxx11_abi=0|--use_cxx11_abi=1|--skip_build|--msdebug|--ascendc_dump|--mssanitizer|--csvopstest_options=<options>|--clean-first|--no-pybind"
            ;;
    esac
}

fn_main "$@"
