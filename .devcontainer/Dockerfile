# Docker image for running hip on Nvida GPU, with:
# cmake, pkg v0.5.0, shfmt,
# clang 18, clang-format, clangd, omp,
# HIP 6.2, CUDA 12.4.1,
# vim, helix, Go, git

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt update -y && \
    apt install -y lsb-release wget && \
    wget https://github.com/mvdan/sh/releases/download/v3.7.0/shfmt_v3.7.0_linux_amd64 -O /usr/bin/shfmt && \
    chmod u+x /usr/bin/shfmt && \
    wget https://github.com/genshen/pkg/releases/download/v0.5.0/pkg-linux-amd64 -O /usr/bin/pkg && \
    chmod u+x /usr/bin/pkg && \
    DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common gnupg vim && \
    apt install -y golang-go cmake

RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" >> /etc/apt/sources.list && \
    echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" >> /etc/apt/sources.list && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc && \
    apt update -y && \
    apt install -y clang-18 lldb-18 lld-18 libc++-18-dev libc++abi-18-dev libomp-18-dev clang-tools-18 \
        libclang-common-18-dev libclang-18-dev libclang1-18 clang-format-18 python3-clang-18 clangd-18 clang-tidy-18 && \
    apt install -y git && \
    apt clean

# https://github.com/ROCm/HIP/issues/3521
# https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html (6.2.41134)
RUN export HIP_PLATFORM='nvidia' \
    && wget https://repo.radeon.com/amdgpu-install/6.2/ubuntu/jammy/amdgpu-install_6.2.60200-1_all.deb \
    && apt install -y ./amdgpu-install_6.2.60200-1_all.deb \
    && apt update \
    && apt install -y hip-dev hipcc-nvidia \
    && rm ./amdgpu-install_6.2.60200-1_all.deb \
    && ln -s /opt/rocm/bin/* /usr/local/bin/ \
    && apt clean

# install helix and git-lfs
RUN add-apt-repository ppa:maveonair/helix-editor && apt update && apt install -y helix \
    && apt clean \
    && wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz -O /tmp/git-lfs-linux-amd64-v3.5.1.tar.gz \
    && tar zxf /tmp/git-lfs-linux-amd64-v3.5.1.tar.gz \
    && mv git-lfs-3.5.1/git-lfs /usr/local/bin/ \
    && rm -rf /tmp/git-lfs-linux-amd64-v3.5.1.tar.gz git-lfs-3.5.1/ \
    && ln -s /usr/bin/clangd-18 /usr/local/bin/clangd \
    && ln -s /usr/bin/clang-format-18  /usr/local/bin/clang-format

ENV CC="clang-18" \
    CXX="clang++-18" \
    HIP_PLATFORM='nvidia'

WORKDIR /workspace
