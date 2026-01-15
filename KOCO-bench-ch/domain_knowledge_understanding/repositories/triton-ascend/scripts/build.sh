#!/bin/bash

set -ex

if [ $# -lt 5 ]; then
  echo "Usage: $0 <TRITON_ASCEND_ROOT> <LLVM_ROOT> <TRITON_ASCEND_VERSION> <BUILD_MODE> <IS_MANYLINUX>"
  exit 1
fi

TRITON_ASCEND_ROOT=$1
LLVM_ROOT=$2
TRITON_ASCEND_VERSION=$3
BUILD_MODE=$4
IS_MANYLINUX_VAL=$5

case "$BUILD_MODE" in
  "develop")
    echo "Building in editable/development mode..."
    cmd="pip install --no-build-isolation --editable ."
    ;;
  "install")
    echo "Building in normal install mode..."
    cmd="pip install --no-build-isolation ."
    ;;
  "bdist_wheel")
    echo "Building wheel package..."
    cmd="python -m build --wheel"
    ;;
  "clean")
    echo "Cleaning build artifacts..."
    python3 setup.py clean
    ;;
  *)
    echo "Error: Unknown BUILD_MODE: ${BUILD_MODE}"
    echo "Valid options: develop, install, bdist_wheel"
    exit 1
    ;;
esac

TRITON_ASCEND_PROJECT_ROOT=$(readlink -f "${TRITON_ASCEND_ROOT}/..")

cd $TRITON_ASCEND_PROJECT_ROOT

LLVM_SYSPATH=${LLVM_ROOT} \
TRITON_PLUGIN_DIRS=${TRITON_ASCEND_ROOT} \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_WHEEL_NAME="triton_ascend" \
TRITON_VERSION=${TRITON_ASCEND_VERSION} \
IS_MANYLINUX=${IS_MANYLINUX_VAL} \
${cmd}
