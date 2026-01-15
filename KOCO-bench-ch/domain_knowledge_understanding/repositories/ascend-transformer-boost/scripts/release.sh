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

function fn_build_mki()
{
    if [ ! -d "$THIRD_PARTY_DIR"/Mind-KernelInfra ]; then
        [[ ! -d "$THIRD_PARTY_DIR" ]] && mkdir $THIRD_PARTY_DIR
        cd $THIRD_PARTY_DIR
        branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || echo "commit_id") 
        [[ "$branch" == *br_personal* || "$branch" == "commit_id" ]] && branch=master
        echo  "current branch for atb and mki: $branch"
        git clone --branch $branch --depth 1 https://szv-open.codehub.huawei.com/OpenBaize/Ascend/Mind-KernelInfra.git
    else
        [[ -d "$THIRD_PARTY_DIR"/Mind-KernelInfra/build ]] && rm -rf $THIRD_PARTY_DIR/Mind-KernelInfra/build
        [[ -d "$THIRD_PARTY_DIR"/Mind-KernelInfra/output ]] && rm -rf $THIRD_PARTY_DIR/Mind-KernelInfra/output
    fi

    cd $THIRD_PARTY_DIR/Mind-KernelInfra
    echo  "current commid id of Mind-KernelInfra: $(git rev-parse HEAD)"
    if [ "$USE_CXX11_ABI" == "ON" ];then
        build_options="--use_cxx11_abi=1"
    else
        build_options="--use_cxx11_abi=0"
    fi

    build_options_atbops="$build_options --output=$THIRD_PARTY_DIR"
    bash scripts/build.sh release $build_options_atbops
    cd $THIRD_PARTY_DIR
    tar -xf mki.tar.gz
    rm -f mki.tar.gz
}

function fn_init_makeself()
{
    if [ -d "$ATB_DIR/opensource/makeself" ]; then
        echo "makeself exist in $ATB_DIR, not build it"
        return
    fi
    [[ ! -d "$ATB_DIR/opensource" ]] && mkdir $ATB_DIR/opensource
    cd $ATB_DIR/opensource/
    rm -rf makeself
    git clone -b release-2.5.0 https://github.com/megastep/makeself.git
}

function fn_build_cann_dependency()
{
    CCEC_COMPILER_DIR=$THIRD_PARTY_DIR/compiler/ccec_compiler
    TIKCPP_DIR=$THIRD_PARTY_DIR/compiler/tikcpp
    [[ -d "$THIRD_PARTY_DIR/compiler" ]] && rm -rf $THIRD_PARTY_DIR/compiler
    mkdir -p $THIRD_PARTY_DIR/compiler
    ln -s $ASCEND_HOME_PATH/compiler/ccec_compiler $CCEC_COMPILER_DIR
    ln -s $ASCEND_HOME_PATH/compiler/tikcpp $TIKCPP_DIR
}

function fn_build_nlohmann_json()
{
    if [ -d "$THIRD_PARTY_DIR/nlohmannJson" ]; then
        return $?
    fi
    [[ ! -d "$CACHE_DIR" ]] && mkdir $CACHE_DIR
    [[ ! -d "$THIRD_PARTY_DIR" ]] && mkdir $THIRD_PARTY_DIR
    cd $CACHE_DIR
    git clone --branch v3.11.3 --depth 1 https://github.com/nlohmann/json.git
    mv json nlohmannJson
    mv ./nlohmannJson $THIRD_PARTY_DIR
}

function fn_build_pybind11()
{
    if [ -d "$THIRD_PARTY_DIR/pybind11" ]; then
        return $?
    fi
    cd $THIRD_PARTY_DIR
    git clone --branch v2.10.3 --depth 1 https://github.com/pybind/pybind11.git 
}
 
function fn_build_catlass()
{
    if [ -d "$THIRD_PARTY_DIR/catlass" ]; then
        return $?
    fi
    cd $THIRD_PARTY_DIR
    branch=catlass-v1-stable
    echo "current branch for catlass: $branch"
    git clone --branch $branch --depth 1 https://gitee.com/ascend/catlass.git
}

function fn_init_env()
{
    res=$(python3 -c "import torch" &> /dev/null || echo "torch_not_exist")
    if [ "$res" == "torch_not_exist" ]; then
        echo "Warning: Torch is not installed!"
        [[ "$USE_CXX11_ABI" == "" ]] && USE_CXX11_ABI=ON
        echo "USE_CXX11_ABI=$USE_CXX11_ABI"
        return 0
    fi

    PYTORCH_VERSION="$(python3 -c 'import torch; print(torch.__version__)')"
    export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
    export LD_LIBRARY_PATH=$PYTORCH_INSTALL_PATH/../torch.libs:$LD_LIBRARY_PATH
    if [ -z "${PYTORCH_NPU_INSTALL_PATH}" ]; then
        export PYTORCH_NPU_INSTALL_PATH=$(pip show torch-npu | grep "Location" | awk '{print $2}')/torch_npu
    fi
    
    echo "PYTORCH_INSTALL_PATH=$PYTORCH_INSTALL_PATH"
    echo "PYTORCH_NPU_INSTALL_PATH=$PYTORCH_NPU_INSTALL_PATH"
    export LD_LIBRARY_PATH=$PYTORCH_INSTALL_PATH/lib:$PYTORCH_NPU_INSTALL_PATH/lib:$LD_LIBRARY_PATH
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

function fn_build()
{
    [[ -d "$CACHE_DIR" ]] && rm -rf $CACHE_DIR
    mkdir $CACHE_DIR
    cd $CACHE_DIR
    fn_init_env
    echo "COMPILE_OPTIONS:$COMPILE_OPTIONS"
    cmake $CODE_ROOT $COMPILE_OPTIONS
    cmake --build . --parallel $(nproc)
    cmake --install .
    if [ "$USE_CXX11_ABI" != "ON" ]; then
        fn_gen_atb_whl
    fi
}

function config_atb_version()
{
    if [ -z "$VERSION" ]; then
        echo "PackageName is not set, use default setting!"
        VERSION='1.0.RC1'
        echo "PackageName: $VERSION"
    fi
    if [ -z "$VERSION_B" ]; then
        echo "CANNVersion is not set, use default setting!"
        VERSION_B='1.0.RC1'
        echo "CANNVersion: $VERSION_B"
    fi
    if [[ "${VERSION}" =~ ^[1-9]\.[0-9]\. ]]; then
        sed -i "s/ATBVERSION/${VERSION_B}/" $ATB_DIR/src/atb/utils/utils.cpp
    else
        echo "VERSION is invalid"
        exit 1
    fi
}

function fn_make_run_package()
{
    if [ $( uname -a | grep -c -i "x86_64" ) -ne 0 ]; then
        echo "it is system of x86_64"
        ARCH="x86_64"
    elif [ $( uname -a | grep -c -i "aarch64" ) -ne 0 ]; then
        echo "it is system of aarch64"
        ARCH="aarch64"
    else
        echo "it is not system of aarch64 or x86_64"
        exit 1
    fi
    branch=$(git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || echo $branch)
    commit_id=$(git rev-parse HEAD)
    touch $OUTPUT_DIR/version.info
    cat>$OUTPUT_DIR/version.info<<EOF
    Ascend-cann-atb : ${VERSION}
    Ascend-cann-atb Version : ${VERSION_B}
    Platform : ${ARCH}
    branch : ${branch}
    commit id : ${commit_id}
EOF

    mkdir -p $OUTPUT_DIR/scripts
    mkdir -p $RELEASE_DIR/$ARCH
    cp $ATB_DIR/scripts/install.sh $OUTPUT_DIR
    cp $ATB_DIR/scripts/set_env.sh $OUTPUT_DIR
    cp $ATB_DIR/scripts/uninstall.sh $OUTPUT_DIR/scripts
    cp $ATB_DIR/scripts/filelist.csv $OUTPUT_DIR/scripts
    sed -i "s/ATBPKGARCH/${ARCH}/" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/install.sh
    sed -i "s!LOG_PATH_PLACEHOLDER!${LOG_PATH}!" $OUTPUT_DIR/install.sh
    sed -i "s!LOG_NAME_PLACEHOLDER!${LOG_NAME}!" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${VERSION}!" $OUTPUT_DIR/scripts/uninstall.sh
    sed -i "s!LOG_PATH_PLACEHOLDER!${LOG_PATH}!" $OUTPUT_DIR/scripts/uninstall.sh
    sed -i "s!LOG_NAME_PLACEHOLDER!${LOG_NAME}!" $OUTPUT_DIR/scripts/uninstall.sh
    sed -i 's|${atb_path}|${atb_path}/latest/atb|g' $OUTPUT_DIR/set_env.sh

    chmod +x $OUTPUT_DIR/*
    $ATB_DIR/opensource/makeself/makeself.sh --header $ATB_DIR/opensource/makeself/makeself-header.sh \
       --help-header $ATB_DIR/scripts/help.info --gzip --complevel 4 --nomd5 --sha256 --chown \
        ${OUTPUT_DIR} $RELEASE_DIR/$ARCH/Ascend-cann-atb_${VERSION}_linux-${ARCH}.run "Ascend-cann-atb-api" ./install.sh

    rm -rf $OUTPUT_DIR/*
    mv $RELEASE_DIR/$ARCH $OUTPUT_DIR
    echo "Ascend-cann-atb_${VERSION}_linux-${ARCH}.run is successfully generated in $OUTPUT_DIR"
}

function fn_build_tbe_dependency()
{
    CANNDEV_DIR=$THIRD_PARTY_DIR/canndev
    METADEF_DIR=$THIRD_PARTY_DIR/metadef
    API_DIR=$THIRD_PARTY_DIR/api
    CANN_OPS_DIR=$THIRD_PARTY_DIR/cann-ops-adv
    export ASCEND_KERNEL_PATH=$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/kernel
    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DBUILD_TBE_ADAPTER=ON"

    # release
    if [ ! -d "$CANNDEV_DIR" ];then
        echo "Failed to find canndev"
        exit 1
    fi
    if [ ! -d "$API_DIR" ];then
        echo "Failed to find api"
        exit 1
    fi
    if [ ! -d "$CANN_OPS_DIR" ];then
        echo "Failed to find cann-ops-adv"
        exit 1
    fi
    cp -r $CODE_ROOT/src/kernels/tbe_adapter/stubs/include/canndev $THIRD_PARTY_DIR
    cp -r $CODE_ROOT/src/kernels/tbe_adapter/stubs/include/api $THIRD_PARTY_DIR
    if [ ! -d "$METADEF_DIR" ];then
        echo "Failed to find metadef"
        exit 1
    fi
    return
}

function fn_main()
{
    if [[ "$1" == "pack" ]]; then
        arg1="pack"
        shift
    else
        arg1="compile"
        shift
    fi

    until [[ -z "$1" ]]
    do {
        case "$1" in
        "--PackageName="*)
            VERSION="${1#*=}"
            echo "PackageName set to: $VERSION"
            ;;
        "--CANNVersion="*)
            VERSION_B="${1#*=}"
            echo "CANNVersion set to: $VERSION_B"
            ;;
        "--use_cxx11_abi=1")
            USE_CXX11_ABI=ON
            ;;
        "--use_cxx11_abi=0")
            USE_CXX11_ABI=OFF
            ;;
        "--build_customize_ops")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DBUILD_CUSTOMIZE_OPS=ON"
            ;;
        "--local_release_compile")
            LOCAL_RELEASE_COMPILE=ON
        esac
        shift
    }
    done

    case "${arg1}" in
        "pack")
            config_atb_version
            fn_init_makeself
            fn_make_run_package
            ;;
        "compile")
            [[ -d "$THIRD_PARTY_DIR"/asdops ]] && rm -rf $THIRD_PARTY_DIR/asdops
            [[ -d "$THIRD_PARTY_DIR"/mki ]] && rm -rf $THIRD_PARTY_DIR/mki
            fn_build_mki
            fn_build_catlass
            fn_build_cann_dependency
            fn_build_tbe_dependency
            [[ "$USE_CXX11_ABI" == "ON" ]] && COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_CXX11_ABI=ON"
            [[ "$USE_CXX11_ABI" == "OFF" ]] && COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_CXX11_ABI=OFF"
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Release \
            -DLOCAL_RELEASE_COMPILE=$LOCAL_RELEASE_COMPILE -DPACKAGE_COMPILE=ON"
            config_atb_version
            fn_build_nlohmann_json
            fn_build_pybind11
            fn_build
            ;;
    esac
}

set -e
fn_init_env
SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR
cd ..
export CODE_ROOT=$(pwd)
export CACHE_DIR=$CODE_ROOT/build
OUTPUT_DIR=$CODE_ROOT/output
THIRD_PARTY_DIR=$CODE_ROOT/3rdparty
LOG_PATH="/var/log/cann_atb_log/"
LOG_NAME="cann_atb_install.log"
ATB_DIR=$CODE_ROOT
RELEASE_DIR=$CODE_ROOT/ci/release
LOCAL_RELEASE_COMPILE=OFF

cann_default_install_path="/usr/local/Ascend/ascend-toolkit"

set +e
if [ -d "${cann_default_install_path}" ];then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    export ASCEND_HOME_PATH=/home/slave1/Ascend/ascend-toolkit/latest
    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
fi
set -e
fn_main "$@"
