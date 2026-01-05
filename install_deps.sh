#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Installation script for Dynamo build dependencies
# This script installs all necessary libraries to make cargo build pass

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
UCX_VERSION="${UCX_VERSION:-v1.19.0}"
NIXL_REF="${NIXL_REF:-0.7.1}"
GDRCOPY_REF="${GDRCOPY_REF:-v2.5.1}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
UCX_INSTALL_PATH="${UCX_INSTALL_PATH:-/usr/local/ucx}"
NIXL_INSTALL_PATH="${NIXL_INSTALL_PATH:-/opt/nvidia/nvda_nixl}"

# Detect architecture
ARCH_NAME="x86_64-linux-gnu"
if [ "$(uname -m)" != "amd64" ] && [ "$(uname -m)" != "x86_64" ]; then
    ARCH_NAME="aarch64-linux-gnu"
fi

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

install_system_dependencies() {
    echo_info "Installing system dependencies..."

    apt-get update

    # Install all required packages
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        autoconf \
        automake \
        libtool \
        meson \
        ninja-build \
        clang \
        libclang-dev \
        protobuf-compiler \
        git \
        wget \
        curl \
        python3-dev \
        python3-pip \
        pybind11-dev \
        python3-pybind11 \
        libibverbs-dev \
        libibverbs1 \
        ibverbs-providers \
        ibverbs-utils \
        libibumad-dev \
        libibumad3 \
        libnuma-dev \
        libnuma1 \
        librdmacm-dev \
        librdmacm1 \
        rdma-core \
        rdma-core-devel 2>/dev/null || true \
        pkg-config \
        ca-certificates

    echo_info "System dependencies installed successfully"
}

install_python_build_tools() {
    echo_info "Installing Python build tools..."
    pip3 install --no-cache-dir meson ninja pybind11 || true
}

build_gdrcopy() {
    if [ -d /usr/local/include/gdrapi.h ]; then
        echo_info "gdrcopy already installed, skipping..."
        return 0
    fi

    echo_info "Building and installing gdrcopy ${GDRCOPY_REF}..."

    local TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    git clone --depth 1 --branch ${GDRCOPY_REF} https://github.com/NVIDIA/gdrcopy.git
    cd gdrcopy

    # Build and install
    make prefix=/usr/local CUDA=${CUDA_PATH} all install

    # Try to build kernel module if headers are available
    if [ -d /lib/modules/$(uname -r)/build ]; then
        echo_info "Building gdrcopy kernel module..."
        make -C src gdrdrv || echo_warn "Failed to build kernel module (this is optional)"
    else
        echo_warn "Kernel headers not found, skipping kernel module build"
    fi

    cd /
    rm -rf "$TEMP_DIR"

    echo_info "gdrcopy installed successfully"
}

build_ucx() {
    if [ -d "${UCX_INSTALL_PATH}" ] && [ -f "${UCX_INSTALL_PATH}/bin/ucx_info" ]; then
        echo_info "UCX already installed at ${UCX_INSTALL_PATH}, skipping..."
        return 0
    fi

    echo_info "Building and installing UCX ${UCX_VERSION}..."

    local TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    git clone --depth 1 --branch ${UCX_VERSION} https://github.com/openucx/ucx.git
    cd ucx

    ./autogen.sh

    # Configure with all necessary options
    ./contrib/configure-release \
        --prefix=${UCX_INSTALL_PATH} \
        --enable-shared \
        --disable-static \
        --disable-doxygen-doc \
        --enable-optimizations \
        --enable-cma \
        --enable-devel-headers \
        --with-cuda=${CUDA_PATH} \
        --with-verbs \
        --with-dm \
        --with-gdrcopy=/usr/local \
        --enable-mt

    make -j$(nproc)
    make install

    # Configure library path
    echo "${UCX_INSTALL_PATH}/lib" > /etc/ld.so.conf.d/ucx.conf
    echo "${UCX_INSTALL_PATH}/lib/ucx" >> /etc/ld.so.conf.d/ucx.conf
    ldconfig

    cd /
    rm -rf "$TEMP_DIR"

    echo_info "UCX installed successfully to ${UCX_INSTALL_PATH}"
}

build_nixl() {
    if [ -d "${NIXL_INSTALL_PATH}" ] && [ -f "${NIXL_INSTALL_PATH}/include/nixl.h" ]; then
        echo_info "NIXL already installed at ${NIXL_INSTALL_PATH}, skipping..."
        return 0
    fi

    echo_info "Building and installing NIXL ${NIXL_REF}..."

    # Check architecture limitations
    if [ "$ARCH_NAME" != "x86_64-linux-gnu" ]; then
        echo_error "NIXL is not supported on ${ARCH_NAME} architecture"
        return 1
    fi

    local TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    git clone --depth 1 --branch ${NIXL_REF} https://github.com/ai-dynamo/nixl.git
    cd nixl

    mkdir -p build
    meson setup build/ \
        --prefix=${NIXL_INSTALL_PATH} \
        --buildtype=release \
        -Dcudapath_lib="${CUDA_PATH}/lib64" \
        -Dcudapath_inc="${CUDA_PATH}/include" \
        -Ducx_path="${UCX_INSTALL_PATH}"

    cd build
    ninja
    ninja install

    # Configure library paths
    # NIXL installs to lib64 but we need to symlink to arch-specific directory
    mkdir -p "${NIXL_INSTALL_PATH}/lib/${ARCH_NAME}"
    if [ -d "${NIXL_INSTALL_PATH}/lib64" ]; then
        cp -r "${NIXL_INSTALL_PATH}/lib64/"* "${NIXL_INSTALL_PATH}/lib/${ARCH_NAME}/" 2>/dev/null || true
    fi

    echo "${NIXL_INSTALL_PATH}/lib/${ARCH_NAME}" > /etc/ld.so.conf.d/nixl.conf
    echo "${NIXL_INSTALL_PATH}/lib/${ARCH_NAME}/plugins" >> /etc/ld.so.conf.d/nixl.conf
    echo "${NIXL_INSTALL_PATH}/lib64" >> /etc/ld.so.conf.d/nixl.conf
    echo "${NIXL_INSTALL_PATH}/lib64/plugins" >> /etc/ld.so.conf.d/nixl.conf
    ldconfig

    cd /
    rm -rf "$TEMP_DIR"

    echo_info "NIXL installed successfully to ${NIXL_INSTALL_PATH}"
}

print_environment_setup() {
    echo ""
    echo_info "Installation complete! Please add the following to your environment:"
    echo ""
    echo "export NIXL_PREFIX=${NIXL_INSTALL_PATH}"
    echo "export NIXL_LIB_DIR=${NIXL_INSTALL_PATH}/lib/${ARCH_NAME}"
    echo "export NIXL_PLUGIN_DIR=${NIXL_INSTALL_PATH}/lib/${ARCH_NAME}/plugins"
    echo "export LD_LIBRARY_PATH=\${NIXL_LIB_DIR}:\${NIXL_PLUGIN_DIR}:${UCX_INSTALL_PATH}/lib:${UCX_INSTALL_PATH}/lib/ucx:\${LD_LIBRARY_PATH}"
    echo "export PATH=${UCX_INSTALL_PATH}/bin:\${PATH}"
    echo ""
    echo_info "You can add these to your ~/.bashrc or run them in your current shell"
    echo ""
}

main() {
    echo_info "Starting Dynamo dependency installation..."
    echo_info "This script will install UCX and NIXL libraries required for cargo build"
    echo ""

    check_root

    # Check for CUDA
    if [ ! -d "${CUDA_PATH}" ]; then
        echo_error "CUDA not found at ${CUDA_PATH}"
        echo_error "Please install CUDA or set CUDA_PATH environment variable"
        exit 1
    fi
    echo_info "Found CUDA at ${CUDA_PATH}"

    # Install dependencies in order
    install_system_dependencies
    install_python_build_tools
    build_gdrcopy
    build_ucx
    build_nixl

    print_environment_setup

    echo_info "All dependencies installed successfully!"
    echo_info "You can now run: cargo build --locked --profile dev"
}

# Run main function
main "$@"
