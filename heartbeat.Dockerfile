# syntax=docker/dockerfile:1.10.0-labs
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# === BEGIN templates/args.Dockerfile ===
##########################
#### Build Arguments #####
##########################
# TARGETARCH is set automatically by Docker BuildKit for every --platform build.
# It must NOT be declared in the global scope (before any FROM) — doing so shadows
# the automatic per-platform value that BuildKit injects.
#
# In each stage that needs it, re-declare with:  ARG TARGETARCH
#
# ARCH_ALT (x86_64 / aarch64) is computed inline in RUN steps:
#   ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64")
ARG DEVICE=cuda
ARG HEARTBEAT_BASE_IMAGE

# Python/CUDA configuration
ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=13.0
ARG CUDA_MAJOR=${CUDA_VERSION%%.*}

# Base and runtime images configuration
ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base
ARG BASE_IMAGE_TAG=25.11-cuda13.0-devel-ubuntu24.04
ARG RUNTIME_IMAGE=vllm/vllm-openai
ARG RUNTIME_IMAGE_TAG=v0.26.0-ubuntu2404

# wheel builder image selection

ARG WHEEL_BUILDER_IMAGE=quay.io/pypa/manylinux_2_28_x86_64

# Build configuration
ARG ENABLE_KVBM=true
ARG CARGO_BUILD_JOBS

ARG NATS_VERSION=v2.12.14
ARG ETCD_VERSION=v3.5.33

ARG ENABLE_MEDIA_FFMPEG=false
ARG FFMPEG_VERSION=8.1.2
ARG LIBVPX_REF=v1.14.1
ARG ENABLE_GPU_MEMORY_SERVICE=true

# SCCACHE configuration
ARG USE_SCCACHE
ARG SCCACHE_VERSION=v0.14.0
ARG SCCACHE_BUCKET=""
ARG SCCACHE_REGION=""

# NIXL configuration
ARG NIXL_UCX_REF=v1.21.0
ARG NIXL_REF=v1.3.2

ARG NIXL_GDRCOPY_REF=v2.5.2
ARG NIXL_LIBFABRIC_REPO=https://github.com/ofiwg/libfabric.git
ARG NIXL_LIBFABRIC_REF=v2.4.0
ARG HWLOC_VERSION=2.12.2

ARG MAX_JOBS=10
# FlashInfer cubin/jit-cache version used by the vLLM installer.
ARG FLASHINF_REF=v0.6.14

ARG VLLM_OMNI_REF=v0.26.0rc1

# If left blank, then we will fallback to vLLM defaults
ARG DEEPGEMM_REF=""

# aws-sdk-cpp tag for the NIXL OBJ / S3 backend (built in wheel_builder).
ARG AWS_SDK_CPP_VERSION=1.11.760
# ModelExpress Python client for model loading (optional)
ARG MODELEXPRESS_VERSION=0.5.0

# --- Base Image Stages

# === BEGIN templates/dynamo_base.Dockerfile ===
##################################
########## Base Image ############
##################################

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS dynamo_base

ARG TARGETARCH

USER root
WORKDIR /opt/dynamo

# Install sccache into the base image so downstream stages can COPY it
# instead of downloading from GitHub (avoids 502 errors under parallel builds)
ARG SCCACHE_VERSION
RUN ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    wget --tries=3 --waitretry=5 \
        "https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/sccache-${SCCACHE_VERSION}-${ARCH_ALT}-unknown-linux-musl.tar.gz" && \
    tar -xzf "sccache-${SCCACHE_VERSION}-${ARCH_ALT}-unknown-linux-musl.tar.gz" && \
    mv "sccache-${SCCACHE_VERSION}-${ARCH_ALT}-unknown-linux-musl/sccache" /usr/local/bin/ && \
    rm -rf sccache*

# Install uv package manager. It lives in a directory of its own, prepended to
# PATH, because some bases ship a uv earlier on PATH than /usr/local/bin
# (/root/.local/bin, /opt/venv/bin). All stages share one uv cache mount and uv
# rejects cache entries written by a newer version, so the pinned copy has to be
# the one that runs. Holding only uv/uvx keeps the prepend from shadowing
# anything else, notably a framework venv's python.
COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /uvx /opt/uv/bin/
ENV PATH=/opt/uv/bin:${PATH}

# Install NATS server
ARG NATS_VERSION
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    wget --tries=3 --waitretry=5 https://github.com/nats-io/nats-server/releases/download/${NATS_VERSION}/nats-server-${NATS_VERSION}-${TARGETARCH}.deb && \
    dpkg -i nats-server-${NATS_VERSION}-${TARGETARCH}.deb && rm nats-server-${NATS_VERSION}-${TARGETARCH}.deb

# Install etcd
ARG ETCD_VERSION
RUN wget --tries=3 --waitretry=5 https://github.com/etcd-io/etcd/releases/download/$ETCD_VERSION/etcd-$ETCD_VERSION-linux-${TARGETARCH}.tar.gz -O /tmp/etcd.tar.gz && \
    mkdir -p /usr/local/bin/etcd && \
    tar -xvf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1 && \
    rm /tmp/etcd.tar.gz
ENV PATH=/usr/local/bin/etcd/:$PATH

# Rust Setup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.96.1

# Install Rust — ARCH_ALT (x86_64/aarch64) is derived from TARGETARCH at build time
RUN ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    RUSTARCH="${ARCH_ALT}-unknown-linux-gnu" && \
    wget --tries=3 --waitretry=5 "https://static.rust-lang.org/rustup/archive/1.28.1/${RUSTARCH}/rustup-init" && \
    chmod +x rustup-init && \
    ./rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host ${RUSTARCH} && \
    rm rustup-init && \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME

# === BEGIN templates/wheel_builder.Dockerfile ===
##################################
##### Wheel Build Image ##########
##################################

##################################
##### wheel_builder_base #########
##################################
# Shared base for all wheel builds: tools, system deps, and native libraries (except nixl).

FROM ${WHEEL_BUILDER_IMAGE} AS wheel_builder_base

# Redeclare ARGs for this stage
ARG TARGETARCH
ARG CARGO_BUILD_JOBS
ARG DEVICE

WORKDIR /workspace

# Compliance: always create the rust license-harvest dir so the licenses stage's
# `COPY --from=wheel_builder /opt/dynamo/rust-licenses` never fails, even for
# targets that build no wheels. runtime_wheel_builder populates it post-build.
RUN mkdir -p /opt/dynamo/rust-licenses

# Copy CUDA from base stage
COPY --from=dynamo_base /usr/local/cuda /usr/local/cuda
COPY --from=dynamo_base /etc/ld.so.conf.d/hpcx.conf /etc/ld.so.conf.d/hpcx.conf

# Set environment variables first so they can be used in COPY commands
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16} \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    CARGO_TARGET_DIR=/opt/dynamo/target \
    PATH=/usr/local/cargo/bin:$PATH

# Copy artifacts from base stage
COPY --from=dynamo_base $RUSTUP_HOME $RUSTUP_HOME
COPY --from=dynamo_base $CARGO_HOME $CARGO_HOME

# Install system dependencies
# Cache dnf downloads; sharing=locked avoids dnf/rpm races with concurrent builds.
# --setopt=tsflags=nocontexts: skip SELinux file-context labeling. The manylinux
# image lacks the SELinux policy store that some compute nodes expect; without
# this flag, dnf fails with "ValueError: SELinux policy is not managed".
RUN --mount=type=cache,target=/var/cache/dnf,sharing=locked \
    dnf install -y --setopt=tsflags=nocontexts almalinux-release-synergy && \
    dnf config-manager --set-enabled powertools && \
    dnf install -y --setopt=tsflags=nocontexts \
        # Autotools (required for UCX, libfabric ./autogen.sh and ./configure)
        autoconf \
        automake \
        libtool \
        make \
        # RPM build tools (required for gdrcopy's build-rpm-packages.sh)
        rpm-build \
        rpm-sign \
        # Build tools
        cmake \
        ninja-build \
        clang-devel \
        # Install GCC toolset 14 (CUDA compatible, max version 14)
        gcc-toolset-14-gcc \
        gcc-toolset-14-gcc-c++ \
        gcc-toolset-14-binutils \
        flex \
        wget \
        # Kernel module build dependencies
        dkms \
        # Protobuf support
        protobuf-compiler \
        # RDMA/InfiniBand support (required for UCX build with --with-verbs)
        libibverbs \
        libibverbs-devel \
        rdma-core \
        rdma-core-devel \
        libibumad \
        libibumad-devel \
        librdmacm-devel \
        numactl-devel \
        # Libfabric support
        libcurl-devel \
        openssl-devel \
        libuuid-devel \
        zlib-devel

# Default comes from context.yaml; keep it in sync with upstream NIXL's
# contrib/Dockerfile.manylinux. NIXL v1.0.x needs newer hwloc than RHEL8 ships.
ARG HWLOC_VERSION
RUN cd /tmp && \
    HWLOC_SERIES="$(echo "${HWLOC_VERSION}" | cut -d. -f1,2)" && \
    wget -q "https://download.open-mpi.org/release/hwloc/v${HWLOC_SERIES}/hwloc-${HWLOC_VERSION}.tar.gz" && \
    tar -xzf "hwloc-${HWLOC_VERSION}.tar.gz" && \
    cd "hwloc-${HWLOC_VERSION}" && \
    ./configure --prefix=/usr/local --disable-nvml && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig && \
    rm -rf "/tmp/hwloc-${HWLOC_VERSION}" "/tmp/hwloc-${HWLOC_VERSION}.tar.gz"

# Set GCC toolset 14 as the default compiler (CUDA requires GCC <= 14)
ENV PATH="/opt/rh/gcc-toolset-14/root/usr/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/rh/gcc-toolset-14/root/usr/lib64:${LD_LIBRARY_PATH}" \
    CC="/opt/rh/gcc-toolset-14/root/usr/bin/gcc" \
    CXX="/opt/rh/gcc-toolset-14/root/usr/bin/g++"

# Ensure a modern protoc is available (required for --experimental_allow_proto3_optional)
RUN set -eux; \
    ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64"); \
    PROTOC_VERSION=25.3; \
    case "${ARCH_ALT}" in \
      x86_64) PROTOC_ZIP="protoc-${PROTOC_VERSION}-linux-x86_64.zip" ;; \
      aarch64) PROTOC_ZIP="protoc-${PROTOC_VERSION}-linux-aarch_64.zip" ;; \
      *) echo "Unsupported architecture: ${ARCH_ALT}" >&2; exit 1 ;; \
    esac; \
    wget --tries=3 --waitretry=5 -O /tmp/protoc.zip "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/${PROTOC_ZIP}"; \
    rm -f /usr/local/bin/protoc /usr/bin/protoc; \
    unzip -o /tmp/protoc.zip -d /usr/local bin/protoc include/*; \
    chmod +x /usr/local/bin/protoc; \
    ln -s /usr/local/bin/protoc /usr/bin/protoc; \
    protoc --version

# Point build tools explicitly at the modern protoc
ENV PROTOC=/usr/local/bin/protoc

# Install uv package manager, ahead of the copy manylinux bundles in
# /usr/local/bin. See dynamo_base.Dockerfile for why it gets its own directory.
COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /uvx /opt/uv/bin/
ENV PATH=/opt/uv/bin:${PATH}

ENV CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH:-} \
    NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

# Create virtual environment for building wheels
ARG PYTHON_VERSION
ENV VIRTUAL_ENV=/workspace/.venv
# Cache uv downloads; uv handles its own locking for this cache.
# pyyaml: needed by the compliance NOTICES-bundling steps below (overrides.py
# imports yaml at module scope); the system python3 doesn't ship it.
RUN --mount=type=cache,id=uv-root-0.12.0,target=/root/.cache/uv,sharing=shared \
    export UV_CACHE_DIR=/root/.cache/uv UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 && \
    uv venv ${VIRTUAL_ENV} --python $PYTHON_VERSION && \
    uv pip install --upgrade meson pybind11 patchelf maturin[patchelf] tomlkit pyyaml

ARG NIXL_UCX_REF

ARG NIXL_GDRCOPY_REF

# Build and install gdrcopy
RUN ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    git clone --depth 1 --branch ${NIXL_GDRCOPY_REF} https://github.com/NVIDIA/gdrcopy.git && \
    cd gdrcopy/packages && \
    CUDA=/usr/local/cuda ./build-rpm-packages.sh && \
    rpm -Uvh gdrcopy-kmod-*.el8.noarch.rpm && \
    rpm -Uvh gdrcopy-*.el8.${ARCH_ALT}.rpm && \
    rpm -Uvh gdrcopy-devel-*.el8.noarch.rpm

# sccache binary is pre-installed in dynamo_base; stage it off-PATH so
# Meson doesn't auto-detect it as a CUDA compiler launcher
# (https://github.com/mesonbuild/meson/issues/11118).
# When USE_SCCACHE=true the RUN below symlinks it onto PATH before install.
COPY --from=dynamo_base /usr/local/bin/sccache /opt/sccache/sccache

ARG USE_SCCACHE
ARG SCCACHE_BUCKET
ARG SCCACHE_REGION
COPY container/use-sccache.sh /tmp/use-sccache.sh
RUN if [ "$USE_SCCACHE" = "true" ]; then \
        ln -s /opt/sccache/sccache /usr/local/bin/sccache && \
        /tmp/use-sccache.sh install; \
    fi

# Compliance: native source archives drop here. RUN git clone / wget …tar lines
# in the wheel_builder pipeline preserve their resulting archive at
# /tmp/native-sources/<name>-<version>.tar.gz so the per-image `sources_collect`
# stage can COPY them out for OSRB submission. Created here unconditionally
# (cheap) so the COPY always succeeds even when no native source builds run
# for this framework.
RUN mkdir -p /tmp/native-sources

# Compliance source-archival pattern (do NOT add ARG ENABLE_SOURCE_ARCHIVAL
# at this scope — it would invalidate every downstream layer when the flag
# flips between PR builds and post-merge builds).
#
# When future work adds cargo-vendor / go-mod-vendor / native source-tree
# preservation, declare the ARG INLINE in the smallest possible scope,
# immediately before the gated RUN, e.g.:
#
#     ARG ENABLE_SOURCE_ARCHIVAL=false
#     RUN if [ "$ENABLE_SOURCE_ARCHIVAL" = "true" ]; then \
#           cargo vendor --locked --manifest-path /opt/dynamo/Cargo.toml \
#               /tmp/native-sources/rust-vendor; \
#         fi
#
# This way the cache invalidation is contained to one RUN layer (the gated
# one), not the rest of wheel_builder_base. shared-build-image.yml passes
# ENABLE_SOURCE_ARCHIVAL=true via extra_build_args on push / release /
# workflow_dispatch events; PR builds get the default "false" and skip.

# Set SCCACHE environment variables (RUSTC_WRAPPER is set dynamically by
# setup-env only when the sccache server starts successfully)
ENV SCCACHE_BUCKET=${USE_SCCACHE:+${SCCACHE_BUCKET}} \
    SCCACHE_REGION=${USE_SCCACHE:+${SCCACHE_REGION}}

# Build FFmpeg for every framework's video-encode path, SGLang included. The
# build is VP9-only (libvpx) — it contains no H.264, H.265, or AAC encoder in
# any form — so SGLang's video-generation handler gets a VP9 encoder to write
# with, matching vLLM/TRT-LLM.
# Build FFmpeg so libs are available for Rust checks in CI.
# We build the ffmpeg CLI with the libvpx_vp9 encoder so Python code can encode
# video without the GPL-licensed binary shipped by imageio-ffmpeg.
# Stays LGPL-only AND royalty-free: --disable-gpl --disable-nonfree are preserved,
# and no H.264/H.265/AAC codec is built in any form. Video encode is VP9 only
# (libvpx, BSD). NVENC is intentionally NOT enabled: the hardware H.264 encoder is
# still a distributable H.264 codec surface, so it is omitted entirely — see the
# post-build guard below that fails the build if any H.264 surface reappears.
#
# MEDIA CODEC ALLOWLIST: the in-tree libavcodec should carry only
# the media formats we actually build and use, not ffmpeg's full default decoder
# set. A blanket --disable-decoders/--disable-demuxers/--disable-parsers plus a
# narrow allowlist keeps the shipped libav*.so limited to that set. The allowlist
# covers exactly two paths: (1) the encode CLI ingesting rawvideo frames from
# imageio over a pipe and encoding with libvpx_vp9, and (2) the Rust media-ffmpeg
# VideoDecoder decoding VP8/VP9 in mp4/webm/mkv (test fixtures are VP9-in-mp4).
# No H.264 parser/decoder/encoder is enabled — H.264 is not built at all. Image
# decode does not use ffmpeg (it goes through the Rust `image` crate), so no
# still-image decoders are enabled here.
# The `fd` protocol is enabled alongside `pipe`: `ffmpeg -i -` reads stdin via
# the `fd:` protocol on ffmpeg 8.x (not `pipe:`), so omitting it breaks the
# imageio encode path with "Protocol not found. Did you mean file:fd:?". Both
# are pure fd/stream I/O and carry no codec implementation.
#
# Combined with the 8.1 -> 8.1.2 bump below (an upstream maintenance release),
# this also trims the decoder surface to what we ship.
# Do not delete the source tarball for legal reasons.
ARG FFMPEG_VERSION
ARG LIBVPX_REF
RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}} && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    if [ "$DEVICE" = "xpu" ] || [ "$DEVICE" = "cpu" ]; then \
    apt-get update -y && apt-get install -y build-essential pkg-config xz-utils git yasm; \
    apt-get clean && rm -rf /var/lib/apt/lists/*; \
    elif [ "$DEVICE" = "cuda" ]; then \
    dnf install -y --setopt=tsflags=nocontexts pkg-config xz git yasm; \
    fi && \
    # No nv-codec-headers: NVENC/NVDEC are not built, so the NVIDIA codec API
    # headers are not needed. This keeps H.264 (incl. the h264_nvenc HW encoder)
    # out of the in-tree ffmpeg entirely.
    cd /tmp && \
    # libvpx: BSD-licensed VP9 encoder needed for the WebM output path. Built from
    # source so we don't need to track distro package names (libvpx-dev on Debian
    # vs libvpx-devel via EPEL on RHEL/manylinux).
    git clone --depth 1 --branch ${LIBVPX_REF} https://chromium.googlesource.com/webm/libvpx.git && \
    cd libvpx && \
    ./configure --prefix=/usr/local --enable-shared --disable-static --disable-examples --disable-unit-tests --disable-tools --disable-docs && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd /tmp && \
    # Retry the ffmpeg fetch in a shell loop: curl's own --retry does not cover
    # SSL/connection-level failures (e.g. `curl: (35) SSL_ERROR_SYSCALL`) unless
    # --retry-all-errors is used, which needs curl >= 7.71 (the manylinux build
    # base ships 7.61). The loop retries on ANY failure and is version-agnostic.
    for attempt in 1 2 3 4 5; do \
        curl --retry 3 --retry-delay 5 --retry-connrefused --connect-timeout 30 -fL \
            -o ffmpeg-${FFMPEG_VERSION}.tar.xz \
            https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz && break; \
        echo "ffmpeg download attempt ${attempt}/5 failed; retrying in 10s" >&2; \
        sleep 10; \
    done && \
    test -s ffmpeg-${FFMPEG_VERSION}.tar.xz && \
    tar xf ffmpeg-${FFMPEG_VERSION}.tar.xz && \
    cd ffmpeg-${FFMPEG_VERSION} && \
    ./configure \
        --prefix=/usr/local \
        --disable-gpl \
        --disable-nonfree \
        --disable-doc \
        --disable-static \
        --disable-x86asm \
        --disable-network \
        --disable-bsfs \
        --disable-devices \
        --disable-libdrm \
        --enable-shared \
        --enable-libvpx \
        --disable-encoders \
        --enable-encoder=libvpx_vp9 \
        --disable-decoders \
        --enable-decoder=vp8,vp9,rawvideo \
        --disable-muxers \
        --enable-muxer=mov,mp4,matroska,webm \
        --disable-demuxers \
        --enable-demuxer=mov,matroska,rawvideo \
        --disable-parsers \
        --enable-parser=vp8,vp9 \
        --disable-protocols \
        --enable-protocol=file,pipe,fd && \
    make -j$(nproc) && \
    make install && \
    # Compliance guard: fail the build if any royalty-bearing / HW codec surface
    # leaked into the in-tree ffmpeg. By construction this build is VP9-only, so a
    # match here means a config regression. Check the implementation-carrying
    # surfaces (encoders/decoders/parsers), not -codecs (lists names even when no
    # implementation is built) and not -bsfs: bitstream filters (e.g.
    # aac_adtstoasc, h264_mp4toannexb) only reframe an already-encoded stream, are
    # pulled in as mov/mp4 muxer dependencies, and carry no codec implementation.
    for surface in encoders decoders parsers; do \
        if /usr/local/bin/ffmpeg -hide_banner "-${surface}" 2>/dev/null \
             | grep -qiE 'h\.?264|h\.?265|hevc|(^| )aac|nvenc|cuvid|nvdec'; then \
            echo "ERROR: in-tree ffmpeg exposes a disallowed codec via -${surface}" >&2; \
            /usr/local/bin/ffmpeg -hide_banner "-${surface}" 2>/dev/null \
             | grep -iE 'h\.?264|h\.?265|hevc|(^| )aac|nvenc|cuvid|nvdec' >&2; \
            exit 1; \
        fi; \
    done && \
    /tmp/use-sccache.sh show-stats "FFMPEG" && \
    ldconfig && \
    mkdir -p /usr/local/src/ffmpeg && \
    find /tmp/ffmpeg-${FFMPEG_VERSION} \( -name config.log -o -name config.status \) -delete && \
    mv /tmp/ffmpeg-${FFMPEG_VERSION}* /usr/local/src/ffmpeg/

# Build and install UCX
RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    cd /usr/local/src && \
    git clone https://github.com/openucx/ucx.git && \
    cd ucx &&  \
    git checkout $NIXL_UCX_REF &&	 \
    # The intel/llm-scaler xe-GDR patch (ucx-v1.12.0.patch) is upstream since
    # UCX v1.21.0 (ib_md.c xe srcversion check, ze_copy_md.c HOST bit); restore
    # the fetch + git apply for DEVICE=xpu if this ref ever drops below v1.21.0.
    ./autogen.sh &&      \
    if [ "$DEVICE" = "xpu" ]; then \
     ./contrib/configure-release     \
        --prefix=/usr/local/ucx     \
        --with-ze                   \
        --enable-shared             \
        --disable-static            \
        --disable-doxygen-doc       \
        --enable-optimizations      \
        --enable-cma                \
        --enable-devel-headers      \
        --with-verbs                \
        --with-dm                   \
        --with-efa                  \
        --without-cuda              \
        --enable-mt;                 \
    elif [ "$DEVICE" = "cuda" ]; then \
     ./contrib/configure-release     \
        --prefix=/usr/local/ucx     \
        --enable-shared             \
        --disable-static            \
        --disable-doxygen-doc       \
        --enable-optimizations      \
        --enable-cma                \
        --enable-devel-headers      \
        --with-cuda=/usr/local/cuda \
        --with-verbs                \
        --with-dm                   \
        --with-gdrcopy=/usr/local   \
        --with-efa                  \
        --enable-mt;                 \
    elif [ "$DEVICE" = "cpu" ]; then  \
     ./contrib/configure-release     \
        --prefix=/usr/local/ucx     \
        --enable-shared             \
        --disable-static            \
        --disable-doxygen-doc       \
        --enable-optimizations      \
        --enable-cma                \
        --enable-devel-headers      \
        --with-verbs                \
        --without-cuda              \
        --enable-mt;                 \
     fi && \
     make -j &&                      \
     make -j install-strip &&        \
     /tmp/use-sccache.sh show-stats "UCX" && \
     echo "/usr/local/ucx/lib" > /etc/ld.so.conf.d/ucx.conf && \
     echo "/usr/local/ucx/lib/ucx" >> /etc/ld.so.conf.d/ucx.conf && \
     ldconfig

ARG NIXL_LIBFABRIC_REPO
ARG NIXL_LIBFABRIC_REF
RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    cd /usr/local/src && \
    git clone "${NIXL_LIBFABRIC_REPO}" && \
    cd libfabric && \
    git checkout $NIXL_LIBFABRIC_REF && \
    ./autogen.sh && \
    ./configure --prefix="/usr/local/libfabric" \
                --disable-verbs \
                --disable-psm3 \
                --disable-opx \
                --disable-usnic \
                --disable-rstream \
                --enable-efa \
                --with-cuda=/usr/local/cuda \
                --enable-cuda-dlopen \
                --with-gdrcopy \
                --enable-gdrcopy-dlopen && \
    make -j$(nproc) && \
    make install && \
    /tmp/use-sccache.sh show-stats "LIBFABRIC" && \
    echo "/usr/local/libfabric/lib" > /etc/ld.so.conf.d/libfabric.conf && \
    ldconfig

ENV PKG_CONFIG_PATH="/usr/local/libfabric/lib/pkgconfig:${PKG_CONFIG_PATH}"

# Build and install AWS SDK C++ (required for NIXL OBJ backend / S3 support)
ARG AWS_SDK_CPP_VERSION
RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env cmake); \
    fi && \
    git clone --recurse-submodules --depth 1 --branch ${AWS_SDK_CPP_VERSION} \
        https://github.com/aws/aws-sdk-cpp.git /tmp/aws-sdk-cpp && \
    mkdir -p /tmp/aws-sdk-cpp/build && \
    cd /tmp/aws-sdk-cpp/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_ONLY="s3" \
        -DENABLE_TESTING=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=ON && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /tmp/aws-sdk-cpp && \
    ldconfig && \
    /tmp/use-sccache.sh show-stats "AWS SDK C++"

##################################
##### runtime_wheel_builder ######
##################################
# Builds ai-dynamo, ai-dynamo-runtime, and gpu_memory_service wheels, sans nixl.

FROM wheel_builder_base AS runtime_wheel_builder

# Copy source code (order matters for layer caching)
COPY .cargo/ /opt/dynamo/.cargo/
COPY pyproject.toml README.md LICENSE Cargo.toml Cargo.lock rust-toolchain.toml hatch_build.py /opt/dynamo/
COPY lib/ /opt/dynamo/lib/
COPY components/ /opt/dynamo/components/

# Build ai-dynamo (pure Python) and ai-dynamo-runtime (maturin) wheels
ARG USE_SCCACHE

ARG ENABLE_MEDIA_FFMPEG

RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    --mount=type=cache,target=/root/.cargo/registry,sharing=shared \
    --mount=type=cache,target=/root/.cargo/git,sharing=shared \
    --mount=type=cache,id=uv-root-0.12.0,target=/root/.cache/uv,sharing=shared \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export UV_CACHE_DIR=/root/.cache/uv && \
    export SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}} && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env cmake); \
    fi && \
    mkdir -p ${CARGO_TARGET_DIR} && \
    source ${VIRTUAL_ENV}/bin/activate && \
    cd /opt/dynamo && \
    uv build --wheel --out-dir /opt/dynamo/dist && \
    cd /opt/dynamo/lib/bindings/python && \
    if [ "$ENABLE_MEDIA_FFMPEG" = "true" ]; then \
        maturin build --release --features "media-ffmpeg,kv-indexer,slot-tracker,select-service,mm-routing,aic-forward-pass" --out /opt/dynamo/dist; \
    else \
        maturin build --release --features "kv-indexer,slot-tracker,select-service,mm-routing,aic-forward-pass" --out /opt/dynamo/dist; \
    fi && \
    /tmp/use-sccache.sh show-stats "Dynamo Runtime"

# Compliance: harvest each crate's real LICENSE files from the cargo registry
# source cache so the rust NOTICES generator can inline upstream license text
# (the runtime image keeps only the compiled wheel). Keyed "<name>-<version>"
# to match generators/rust.py. Best-effort: unreadable/absent files are skipped
# and the generator falls back to canonical SPDX text. cargo's registry lives
# under CARGO_HOME and/or the cache-mounted /root/.cargo — scan both.
RUN --mount=type=cache,target=/root/.cargo/registry,sharing=shared \
    for src in "${CARGO_HOME}/registry/src" /root/.cargo/registry/src; do \
        [ -d "$src" ] || continue; \
        find "$src" -mindepth 2 -maxdepth 2 -type d | while IFS= read -r crate; do \
            dest="/opt/dynamo/rust-licenses/$(basename "$crate")"; \
            for lf in "$crate"/LICENSE* "$crate"/LICENCE* "$crate"/COPYING* "$crate"/NOTICE* "$crate"/UNLICENSE*; do \
                [ -e "$lf" ] || continue; \
                mkdir -p "$dest" && cp "$lf" "$dest/" 2>/dev/null || true; \
            done; \
        done; \
    done; \
    echo "rust license harvest: $(find /opt/dynamo/rust-licenses -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l) crates with license files"; \
    true

# Compliance: bundle the human-readable third-party Rust NOTICES into the
# maturin wheels themselves (PEP 639 <dist-info>/licenses/), using the harvested
# crate license texts. The wheel already carries maturin's CycloneDX SBOM (the
# machine-readable inventory); this adds the texts the redistributed wheel's
# MIT/BSD/Apache attribution clauses require. Best-effort + non-fatal: a failure
# leaves the wheel with its SBOM intact rather than breaking the build.
# Must run with the build venv's python: bundle_wheel_notices imports
# compliance.overrides, which needs pyyaml (installed in the venv above);
# the bare system python3 lacks it and the step would no-op with a warning.
COPY container/compliance /opt/compliance
RUN set -u; injected=0; \
    for whl in /opt/dynamo/dist/ai_dynamo_runtime*.whl; do \
        [ -e "$whl" ] || continue; \
        PYTHONPATH=/opt ${VIRTUAL_ENV}/bin/python3 -m compliance.bundle_wheel_notices \
            --wheel "$whl" --licenses-dir /opt/dynamo/rust-licenses -v \
            && injected=$((injected+1)) || echo "::warning::wheel NOTICES bundling failed for $whl (SBOM retained)"; \
    done; \
    echo "wheel NOTICES bundled into $injected wheel(s)"

# Compliance source archival: vendor the workspace lockfile for the OSRB
# bundle. Gated on ENABLE_SOURCE_ARCHIVAL so PR builds skip the ~200-400 MB
# vendor pull. The vendor tree is consumed downstream by each runtime
# template's sources_collect stage, which filters against the installed
# wheels' embedded SBOMs to keep only the third-party crates we actually
# ship. Stay scoped to one RUN layer (cache invalidation contained).
ARG ENABLE_SOURCE_ARCHIVAL=false
# Mount cargo registry + git caches so re-runs don't re-download the
# ~750 crates from crates.io every build. `sharing=shared` lets parallel
# builds (e.g. multiple frameworks in CI) read the same cache concurrently.
RUN --mount=type=cache,target=/root/.cargo/registry,sharing=shared \
    --mount=type=cache,target=/root/.cargo/git,sharing=shared \
    if [ "$ENABLE_SOURCE_ARCHIVAL" = "true" ]; then \
        mkdir -p /tmp/dynamo-vendor-full && \
        cd /opt/dynamo && \
        cargo vendor --locked /tmp/dynamo-vendor-full > /dev/null && \
        cp Cargo.toml Cargo.lock /tmp/dynamo-vendor-full/ ; \
    fi

# Build gpu-memory-service wheel → /opt/dynamo/dist/gpu_memory_service*.whl (small C++ extension, fast build -- all targets, all frameworks)

# Build gpu_memory_service wheel (C++ extension only needs Python headers, no CUDA/torch)
ARG ENABLE_GPU_MEMORY_SERVICE
RUN --mount=type=cache,id=uv-root-0.12.0,target=/root/.cache/uv,sharing=shared \
    if [ "$ENABLE_GPU_MEMORY_SERVICE" = "true" ]; then \
        export UV_CACHE_DIR=/root/.cache/uv && \
        source ${VIRTUAL_ENV}/bin/activate && \
        uv build --wheel --out-dir /opt/dynamo/dist /opt/dynamo/lib/gpu_memory_service; \
    fi

##################################
##### wheel_builder ##############
##################################

# Builds NIXL (native + Python wheel) and NIXL-linked extension wheels, then
# consolidates all wheels.
# Runtime templates COPY from this stage.
# Note: XPU triggers this path even when the framework section lacks nixl_ref,
# because no upstream XPU runtime image ships pre-built NIXL.

FROM wheel_builder_base AS wheel_builder

# Build and install nixl
ARG TARGETARCH
ARG DEVICE
ARG NIXL_REF
ARG USE_SCCACHE

ARG CUDA_MAJOR

RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    source ${VIRTUAL_ENV}/bin/activate && \
    git clone "https://github.com/ai-dynamo/nixl.git" && \
    cd nixl && \
    git checkout ${NIXL_REF} && \
    if [ "$DEVICE" = "cuda" ]; then \
        PKG_NAME="nixl-cu${CUDA_MAJOR}"; \
    else \
        PKG_NAME="nixl-${DEVICE}"; \
    fi && \
    ./contrib/tomlutil.py --wheel-name $PKG_NAME pyproject.toml && \
    mkdir build && \
    if [ "$DEVICE" = "cuda" ]; then \
        meson setup build/ --prefix=/opt/nvidia/nvda_nixl --buildtype=release \
            -Dcudapath_lib="/usr/local/cuda/lib64" \
            -Dcudapath_inc="/usr/local/cuda/include" \
            -Ducx_path="/usr/local/ucx" \
            -Dlibfabric_path="/usr/local/libfabric"; \
    elif [ "$DEVICE" = "xpu" ]; then \
        meson setup build/ --prefix=/opt/intel/intel_nixl --buildtype=release \
            -Ducx_path="/usr/local/ucx"; \
    elif [ "$DEVICE" = "cpu" ]; then \
        meson setup build/ --prefix=/opt/nvidia/nvda_nixl --buildtype=release \
            -Ducx_path="/usr/local/ucx"; \
    fi && \
    cd build && \
    ninja && \
    ninja install && \
    /tmp/use-sccache.sh show-stats "NIXL"

ENV NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib64 \
    NIXL_PLUGIN_DIR=/opt/nvidia/nvda_nixl/lib64/plugins \
    NIXL_PREFIX=/opt/nvidia/nvda_nixl

ENV LD_LIBRARY_PATH=${NIXL_LIB_DIR}:${NIXL_PLUGIN_DIR}:/usr/local/ucx/lib:/usr/local/ucx/lib/ucx:${LD_LIBRARY_PATH}

RUN echo "$NIXL_LIB_DIR" > /etc/ld.so.conf.d/nixl.conf && \
    echo "$NIXL_PLUGIN_DIR" >> /etc/ld.so.conf.d/nixl.conf && \
    ldconfig

# Build NIXL wheel → /opt/dynamo/dist/nixl/nixl*.whl (C++ transport library, all targets)
ARG PYTHON_VERSION
RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    --mount=type=cache,id=uv-root-0.12.0,target=/root/.cache/uv,sharing=shared \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export UV_CACHE_DIR=/root/.cache/uv && \
    export SCCACHE_S3_KEY_PREFIX="${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}}" && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env); \
    fi && \
    cd /workspace/nixl && \
    uv build . --wheel --out-dir /opt/dynamo/dist/nixl --python $PYTHON_VERSION

# Copy source code (order matters for layer caching)
COPY .cargo/ /opt/dynamo/.cargo/
COPY pyproject.toml README.md LICENSE Cargo.toml Cargo.lock rust-toolchain.toml hatch_build.py /opt/dynamo/
COPY lib/ /opt/dynamo/lib/
COPY components/ /opt/dynamo/components/

# Build kvbm wheel (with nixl linkage via auditwheel repair)
ARG ENABLE_KVBM
RUN --mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \
    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \
    --mount=type=cache,target=/root/.cargo/registry,sharing=shared \
    --mount=type=cache,target=/root/.cargo/git,sharing=shared \
    --mount=type=cache,id=uv-root-0.12.0,target=/root/.cache/uv,sharing=shared \
    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \
    export UV_CACHE_DIR=/root/.cache/uv && \
    export SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX:-${TARGETARCH}} && \
    ARCH_ALT=$([ "${TARGETARCH}" = "amd64" ] && echo "x86_64" || echo "aarch64") && \
    if [ "$USE_SCCACHE" = "true" ]; then \
        eval $(/tmp/use-sccache.sh setup-env cmake); \
    fi && \
    mkdir -p ${CARGO_TARGET_DIR} && \
    source ${VIRTUAL_ENV}/bin/activate && \
    if [ "$ENABLE_KVBM" = "true" ]; then \
        cd /opt/dynamo/lib/bindings/kvbm && \
        KVBM_FEATURES=""; \
        if [ "$DEVICE" = "cuda" ]; then KVBM_FEATURES="--features nccl"; fi && \
        maturin build --release ${KVBM_FEATURES} --out target/wheels && \
        if [ "$DEVICE" = "cuda" ]; then \
            auditwheel repair \
                --exclude libnixl.so \
                --exclude libnixl_build.so \
                --exclude libnixl_common.so \
                --exclude 'lib*.so*' \
                --plat manylinux_2_28_${ARCH_ALT} \
                --wheel-dir /opt/dynamo/dist \
                target/wheels/*.whl; \
        elif [ "$DEVICE" = "xpu" ] || [ "$DEVICE" = "cpu" ]; then \
            cp target/wheels/*.whl /opt/dynamo/dist/; \
        fi; \
    fi && \
    /tmp/use-sccache.sh show-stats "Dynamo KVBM"

# Consolidate all wheels from the runtime wheel builder stage
COPY --from=runtime_wheel_builder /opt/dynamo/dist/ /opt/dynamo/dist/

# Compliance: bundle third-party Rust NOTICES into the kvbm wheel built in this
# stage (the ai-dynamo-runtime wheel was already bundled in runtime_wheel_builder
# and arrives consolidated above). Harvest kvbm's crate licenses from the cargo
# registry, then inject into its auditwheel-repaired wheel. Best-effort/non-fatal.
# Venv python required: compliance.overrides needs pyyaml, absent from the
# system python3 (see the runtime_wheel_builder bundling step).
COPY container/compliance /opt/compliance
RUN --mount=type=cache,target=/root/.cargo/registry,sharing=shared \
    set -u; \
    for src in "${CARGO_HOME}/registry/src" /root/.cargo/registry/src; do \
        [ -d "$src" ] || continue; \
        find "$src" -mindepth 2 -maxdepth 2 -type d | while IFS= read -r crate; do \
            dest="/opt/dynamo/rust-licenses/$(basename "$crate")"; \
            for lf in "$crate"/LICENSE* "$crate"/LICENCE* "$crate"/COPYING* "$crate"/NOTICE* "$crate"/UNLICENSE*; do \
                [ -e "$lf" ] || continue; mkdir -p "$dest" && cp "$lf" "$dest/" 2>/dev/null || true; \
            done; \
        done; \
    done; \
    for whl in /opt/dynamo/dist/kvbm*.whl; do \
        [ -e "$whl" ] || continue; \
        PYTHONPATH=/opt ${VIRTUAL_ENV}/bin/python3 -m compliance.bundle_wheel_notices \
            --wheel "$whl" --licenses-dir /opt/dynamo/rust-licenses -v \
            || echo "::warning::kvbm wheel NOTICES bundling failed (SBOM retained)"; \
    done; \
    echo "kvbm wheel NOTICES step done"

# --- SGLang Stages

# --- No SGLANG stages included

# --- VLLM Stages

# === BEGIN templates/vllm_runtime.Dockerfile ===
##################################
########## Runtime Image #########
##################################

FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS pre_runtime

ARG PYTHON_VERSION
ARG ENABLE_KVBM
ARG ENABLE_GPU_MEMORY_SERVICE
ARG VLLM_OMNI_REF
ARG NIXL_REF

ARG CUDA_MAJOR

ARG MODELEXPRESS_VERSION

WORKDIR /workspace

ENV DYNAMO_HOME=/opt/dynamo
ENV HOME=/home/dynamo

ENV PATH=/usr/local/bin/etcd:${PATH}

# Expose libnixl.so from the upstream nixl-cu${CUDA_MAJOR} PyPI wheel through a
# stable prefix so non-Python consumers use the same NIXL copy that Python imports.
# This keeps Rust nixl-sys dlopen("libnixl.so") from falling into stub mode in
# processes that do not import the nixl Python package first.
ARG SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/dist-packages
ENV NIXL_PREFIX=/opt/dynamo/nixl \
    NIXL_LIB_DIR=/opt/dynamo/nixl \
    NIXL_PLUGIN_DIR=/opt/dynamo/nixl/plugins
COPY --chmod=755 container/deps/vllm/install_nixl_from_wheel.sh /usr/local/bin/install_nixl_from_wheel
RUN install_nixl_from_wheel \
    --cuda-major "${CUDA_MAJOR}" \
    --site-packages "${SITE_PACKAGES}" \
    --prefix "${NIXL_PREFIX}" \
    --skip-headers
ENV LD_LIBRARY_PATH=${NIXL_LIB_DIR}:${NIXL_PLUGIN_DIR}:${LD_LIBRARY_PATH:-}

# Install NATS and ETCD
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/
COPY --from=dynamo_base /opt/uv/bin/uv /opt/uv/bin/uvx /opt/uv/bin/
ENV PATH=/opt/uv/bin:${PATH}

# Bring base-image OS packages up to the current patch releases published in
# the distro archives. --only-upgrade skips anything not already installed, so
# no new packages are added; versions are left unpinned so a cache-busted
# rebuild picks up the newest patch level (BuildKit reuses this layer otherwise).
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --only-upgrade \
        dirmngr \
        gnupg \
        gnupg-utils \
        gnupg2 \
        gpg \
        gpg-agent \
        gpgconf \
        gpgsm \
        gpgv \
        keyboxd \
        libssl3t64 \
        openssl && \
    rm -rf /var/lib/apt/lists/*

# Create dynamo user with group 0 for OpenShift compatibility.
# Pin -u 1000 explicitly: the vllm/vllm-openai >=0.22 image ships a `vllm` user at
# UID 2000, so after freeing 1000 (ubuntu) useradd would otherwise auto-assign the
# next-highest UID (2001) and fail the `id -u dynamo` == 1000 assertion below.
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -u 1000 -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache/vllm /opt/dynamo \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /home/dynamo/.cache/vllm /opt/dynamo /workspace \
    # Arbitrary OpenShift UIDs need to create the vLLM and Triton caches under $HOME.
    && chmod g+rwx /home/dynamo /home/dynamo/.cache /home/dynamo/.cache/vllm \
    && mkdir -p /etc/profile.d \
    && echo 'umask 002' > /etc/profile.d/00-umask.sh

# FlashInfer creates package-local cubin symlinks at runtime. Grant group 0
# write access so arbitrary OpenShift UIDs can initialize the cubin cache.
RUN SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    CUBINS_DIR="$SITE_PACKAGES/flashinfer_cubin/cubins" && \
    if [ -d "$CUBINS_DIR" ]; then \
        find "$CUBINS_DIR" -type d -exec chmod g+rwx {} + ; \
    fi

# Copy attribution files and wheels
COPY --chmod=664 --chown=dynamo:0 LICENSE /workspace/
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

# Install device-specific NIXL wheels for non-CUDA devices.
# These are custom-built in wheel_builder and required for dev builds to link against NIXL libraries.

# Keep the upstream Python solve intact: install only Dynamo-owned wheels and
# suppress transitive dependency resolution unless a later validation proves a
# missing package must be added explicitly.

# Install Dynamo runtime wheels and optional KVBM/GMS wheels.
# Use --no-deps to prevent dependency conflicts (e.g., KVBM downgrading nixl).
RUN --mount=type=cache,id=uv-root-0.12.0,target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install --system --no-deps /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl && \
    uv pip install --system --no-deps /opt/dynamo/wheelhouse/ai_dynamo*any.whl && \
    if [ "${ENABLE_KVBM}" = "true" ]; then \
        KVBM_WHEEL=$(ls /opt/dynamo/wheelhouse/kvbm*.whl 2>/dev/null | head -1); \
        if [ -n "$KVBM_WHEEL" ]; then uv pip install --system --no-deps "$KVBM_WHEEL"; fi; \
    fi && \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then uv pip install --system --no-deps "$GMS_WHEEL"; fi; \
    fi

# Launch-script examples use jq for readable curl output like the upstream omni
# image. SoX is intentionally NOT installed: vLLM-Omni replaced its sox audio path
# with a pure-numpy peak_normalize() (vllm_omni/utils/audio.py), pysox isn't
# installed, and nothing shells out to the sox binary — so `sox`/`libsox-fmt-all`
# were dead weight that only dragged in a GPL-2.0+ codec cluster (sox, libsox*,
# libao*, libmad0, libid3tag0, libltdl7) we'd then be redistributing. SoX is
# inherently GPL (no LGPL replacement), so the compliant fix is to not ship it.
# (sglang_runtime.Dockerfile is the reference codec-compliance pattern.)
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        jq; \
    rm -rf /var/lib/apt/lists/*

# Layer the released vLLM-Omni package matching the pinned upstream ref while
# constraining packages already solved in the upstream vLLM image.
RUN --mount=type=bind,source=./container/deps/vllm/protected_packages.txt,target=/tmp/vllm_omni_protected_packages.txt \
    --mount=type=bind,source=./container/deps/vllm/install_vllm_omni.sh,target=/tmp/install_vllm_omni.sh \
    --mount=type=cache,id=uv-root-0.12.0,target=/root/.cache/uv,sharing=locked \
    set -eux; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    export VLLM_OMNI_TARGET_DEVICE=cuda; \
    bash /tmp/install_vllm_omni.sh

# Install only the ModelExpress client package. --no-deps preserves the upstream
# vLLM runtime dependency stack. google-crc32c is imported eagerly by the MX
# vLLM loader (>=0.5.0) and is not guaranteed in the base image, so install it
# alongside and verify the import path at build time.
RUN --mount=type=cache,id=uv-root-0.12.0,target=/root/.cache/uv,sharing=locked \
    set -eux; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    uv pip install --system --no-deps \
        "modelexpress==${MODELEXPRESS_VERSION}"; \
    uv pip install --system "google-crc32c>=1.5.0"; \
    python3 -c "import modelexpress.engines.vllm"

# The upstream vllm/vllm-openai base image ships a GPL/GPL-3.0 ffmpeg built
# against libx264/libx265/libmp3lame. Purge ONLY the explicitly-named ffmpeg +
# codec packages and replace them with the LGPL-only in-tree ffmpeg built in
# wheel_builder (--disable-gpl --disable-nonfree; H.264 via NVENC, VP9 via
# libvpx). PyAV, torchaudio, torchvision, soundfile and Pillow all bundle their
# own libraries and do not link the system ffmpeg/codecs, so removing them is
# safe. dpkg-query keeps the match robust across base-image/arch version
# suffixes (e.g. libavcodec58 vs 60).
#
# This grep is the COMPLETE, auditable set of what leaves the image: there is
# deliberately NO apt-get autoremove, so the removal can never cascade into
# unrelated auto-installed packages. That matters because the base image marks
# both the gcc/g++/make toolchain (torch.inductor/Triton JIT shell out to it at
# runtime) and the CUDA math libs (libcublas/libcusolver/libcusparse — the torch
# wheels here ship no bundled cublas and load the system copies) as
# auto-installed. A bare `autoremove --purge` sweeps all of those as "orphaned",
# which broke runtime JIT (missing C compiler) in the 1.3.0 rc image. Any
# LGPL/BSD media libs left orphaned (libva, libvdpau, ...) are license-clean
# dead weight, not a compliance issue.
RUN set -eux; \
    purge=$(dpkg-query -W -f='${Package}\n' 2>/dev/null \
        | grep -E '^(ffmpeg|libav[a-z]|libsw[a-z]|libpostproc|libx264|libx265|libmp3lame|libaom|libdav1d|libvpx|libtheora|libvorbis|libopus|libsoxr|libcaca|libcdio|libzvbi|libgme|libvidstab|libdc1394|libraw1394|libiec61883|libtwolame|libshine|libsrt[0-9]|libudfread|libsvtav1|libbs2b|librubberband|libchromaprint|libcodec2|libgsm|libass[0-9]|libbluray|libxvidcore|libflite)' \
        || true); \
    if [ -n "$purge" ]; then \
        DEBIAN_FRONTEND=noninteractive apt-get purge -y $purge; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# Regression guard for the codec purge above: torch.inductor/Triton JIT shell
# out to a host C/C++ compiler at runtime, so a missing toolchain only surfaces
# on the first compile in production. Reproduce that compile path at build time
# (CPU-only) so a missing compiler aborts the build instead of shipping.
RUN --mount=type=bind,source=./container/deps/vllm/validate_torch_compile_smoke.py,target=/tmp/validate_torch_compile_smoke.py,readonly \
    python3 /tmp/validate_torch_compile_smoke.py

# Copy the LGPL ffmpeg from wheel_builder: versioned shared libs (libav*.so*,
# libsw*.so*) + libvpx + the LGPL CLI binary that imageio/diffusers target via
# IMAGEIO_FFMPEG_EXE. Ungated by enable_media_ffmpeg because the base GPL ffmpeg
# was just purged, so the LGPL CLI must always be present for the omni
# video-export path to have something to encode with.
RUN --mount=type=bind,from=wheel_builder,source=/usr/local/,target=/tmp/usr/local/ \
    mkdir -p /usr/local/lib/pkgconfig && \
    cp -rnL /tmp/usr/local/include/libav* /tmp/usr/local/include/libsw* /usr/local/include/ && \
    cp -nL /tmp/usr/local/lib/libav*.so* /tmp/usr/local/lib/libsw*.so* /usr/local/lib/ && \
    cp -nL /tmp/usr/local/lib/lib*vpx*.so* /usr/local/lib/ 2>/dev/null || true && \
    cp -nL /tmp/usr/local/lib/pkgconfig/libav*.pc /tmp/usr/local/lib/pkgconfig/libsw*.pc /usr/local/lib/pkgconfig/ && \
    cp -nL /tmp/usr/local/bin/ffmpeg /usr/local/bin/ffmpeg && \
    cp -r /tmp/usr/local/src/ffmpeg /usr/local/src/ && \
    ldconfig
ENV IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg

# Positive codec guard: the shipped ffmpeg MUST expose the VP9 encoder and MUST
# NOT expose any H.264/H.265/AAC/NVENC encoder. A missing/broken copy (no VP9)
# or a codec regression fails the build here rather than at runtime — closing the
# gap where an image with no working encoder passed every PR gate.
RUN set -eu; \
    ff="${IMAGEIO_FFMPEG_EXE:-ffmpeg}"; \
    "$ff" -hide_banner -encoders 2>/dev/null | grep -qiE 'libvpx[-_]vp9' \
      || { echo "ERROR: shipped ffmpeg ($ff) has no VP9 encoder" >&2; exit 1; }; \
    if "$ff" -hide_banner -encoders 2>/dev/null \
         | grep -iE 'h\.?264|h\.?265|hevc|(^| )aac|nvenc|cuvid|nvdec'; then \
        echo "ERROR: shipped ffmpeg ($ff) exposes an H.264/H.265/AAC/NVENC encoder" >&2; \
        exit 1; \
    fi

# Replace the upstream vllm/vllm-openai image's imageio-ffmpeg (which ships a
# GPL-encumbered prebuilt ffmpeg binary in <site-packages>/imageio_ffmpeg/binaries/)
# with a source install that leaves no binary on disk. On cuda, IMAGEIO_FFMPEG_EXE
# (set above) points imageio at the LGPL CLI copied from wheel_builder. The
# --no-binary directive lives in the requirements file itself.

RUN --mount=type=bind,source=./container/deps/requirements.vllm.txt,target=/tmp/requirements.vllm.txt \
    --mount=type=cache,id=uv-root-0.12.0,target=/root/.cache/uv,sharing=locked \
    export UV_CACHE_DIR=/root/.cache/uv && \
    uv pip install --system \
        --reinstall-package imageio-ffmpeg --reinstall-package PyNvVideoCodec \
        --no-deps --requirement /tmp/requirements.vllm.txt

# Remove the vLLM source tree shipped in the base image to avoid pytest
# collection conflicts (duplicate conftest plugin registration) and stale
# tool scripts referencing files not present in Dynamo's build context.
RUN rm -rf /workspace/vllm

# Transitional: this exists only until the base image stops shipping these.
# The permanent statement of what may ship is
# tests/dependencies/test_no_software_video_codecs.py -- when this purge
# becomes redundant, delete it and keep those assertions.
# Remove the codec-bearing video-DECODE wheels inherited from the vllm-openai
# base. Each bundles its own full ffmpeg carrying software H.264/H.265/AAC;
# PyAV and decord additionally ship GPL libx264/libx265. Dynamo's vLLM component
# imports none of the removed wheels, so they are unused decode-side dead weight.
# (PyNvVideoCodec is KEPT for NVDEC hardware decode -- see the note below.) The in-tree
# LGPL ffmpeg + imageio-ffmpeg installed above are intentionally KEPT for the
# omni video-encode path, which uses the royalty-free VP9 (libvpx_vp9) encoder —
# no H.264 is built. Direct rm makes the removal robust regardless of how the
# base image's pip is configured; the guards fail the build if any of them survive.
RUN set -eux; \
    python3 -m pip uninstall --yes \
        av decord decord2 opencv-python opencv-python-headless torchcodec \
        || true; \
    SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    rm -rf \
        "${SITE_PACKAGES}"/av "${SITE_PACKAGES}"/av-*.dist-info "${SITE_PACKAGES}"/av.libs \
        "${SITE_PACKAGES}"/cv2 "${SITE_PACKAGES}"/opencv_python*.dist-info "${SITE_PACKAGES}"/opencv_python*.libs \
        "${SITE_PACKAGES}"/decord "${SITE_PACKAGES}"/decord-*.dist-info "${SITE_PACKAGES}"/decord.libs \
        "${SITE_PACKAGES}"/decord2 "${SITE_PACKAGES}"/decord2-*.dist-info "${SITE_PACKAGES}"/decord2.libs \
        "${SITE_PACKAGES}"/torchcodec "${SITE_PACKAGES}"/torchcodec-*.dist-info \
        /root/.cache/pip; \
    # Guard EVERY purged wheel: the uninstall above is `|| true`, so a package
    # that survived (rename, new bundling) would otherwise pass silently. decord2
    # imports as `decord`, so both are covered by the one check.
    ! python3 -c "import cv2" 2>/dev/null; \
    ! python3 -c "import av" 2>/dev/null; \
    ! python3 -c "import decord" 2>/dev/null; \
    ! python3 -c "import torchcodec" 2>/dev/null

# PyNvVideoCodec is KEPT (removed from the purge above) but UPGRADED to >=2.2.0 by
# the requirements install: the base image's 2.0.4 bundles a full FFmpeg (incl.
# libavcodec) that the codec gate rejects, while 2.2.0 bundles only libavutil +
# libavformat (container demux, no software codec). It provides the built-in
# H.264/H.265 video-input path (NVDEC hardware decode via libnvcuvid;
# common/multimodal/nvdec_decoder.py) and requires the driver "video" capability
# at runtime or it cannot import; set it in the image and ensure the K8s
# pod/runtimeClass does not drop it.
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

# Regression guard for the --no-deps ModelExpress install above: resolve and
# invoke the vllm.general_plugins entry points exactly as vLLM does at every
# startup, so a missing transitive dependency fails the build here instead of
# at pod startup. Runs after every package/library install in this stage
# (including the XPU apt step and the cuda codec purge above) so the check is
# order-independent and sees the final image state.
RUN python3 -c "from importlib.metadata import entry_points; \
eps = [ep for ep in entry_points(group='vllm.general_plugins') if ep.name == 'modelexpress']; \
assert eps, 'modelexpress vllm.general_plugins entry point not found'; \
[ep.load()() for ep in eps]"

USER dynamo

# Copy the workspace surface needed by the current vLLM pre-merge test image.
# Keep optional framework trees like planner out of /workspace so the upstream
# runtime does not look like a fully-expanded generic image.
COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=775 --chown=dynamo:0 dev /workspace/dev
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/common /workspace/components/src/dynamo/common
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/frontend /workspace/components/src/dynamo/frontend
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/vllm /workspace/components/src/dynamo/vllm
COPY --chown=dynamo:0 lib /workspace/lib

# Setup launch banner in common directory accessible to all users
USER root
RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen && \
    chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc

USER dynamo

ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

# Reset the upstream "vllm serve" entrypoint so the derived runtime behaves
# like other Dynamo images and can execute arbitrary commands directly.
ENTRYPOINT []

# === BEGIN templates/compliance.Dockerfile ===
#
# Inline-compliance Dockerfile stages, shared by the vllm / sglang / trtllm
# runtime templates.
#
# This template emits four stages in fixed order:
#
#   1. licenses          -- runs compliance.generators against the
#                           previously-defined build stage, validates output
#                           against policy, and stages the unified /legal tree
#                           (flat NOTICES-<Eco>.txt + osrb-deps.csv + osrb.cdx.json).
#   2. compliance_artifact -- FROM scratch; exposes the unified /legal tree for
#                           CI extraction as a single `-compliance` artifact.
#                           (Named *_artifact to avoid colliding with the
#                           `compliance` build-context the deploy Dockerfiles use.)
#   3. sources_collect   -- gated on ENABLE_SOURCE_ARCHIVAL; runs
#                           compliance.collect_sources to produce /sources.zip.
#   4. sources_archive   -- FROM scratch; exposes /sources.zip.
#
# The caller (each per-framework runtime template) is expected to:
#   - have defined `pre_runtime` already
#   - end with its own final stage (typically `runtime`) that does
#     `COPY --from=licenses /legal /legal` to inherit NOTICES.
#
# Jinja variables consumed:
#
#   compliance_base_stage     -- "pre_runtime"; set by
#                                container/render.py:_render_context().
#   compliance_baseline_sbom  -- filename under base_sboms/ (or empty string
#                                if no baseline captured); set by
#                                _render_context() from `framework`/
#                                `device_key`.
#   compliance_ecosystems     -- comma-separated --ecosystem list for the
#                                licenses stage. planner drops dpkg (distroless,
#                                ships no builder Debian packages); other targets
#                                get python,rust,dpkg,native. Set by
#                                _render_context().
#   compliance_source_ecosystem_flags -- repeated --ecosystem flags for the
#                                sources_collect stage; per-target likewise.
#   framework, target, make_efa -- already in render context; control
#                                  ecosystem flags + EFA native attribution.

#######################################
########## Compliance: licenses #######
#######################################
#
# Runs every per-ecosystem generator under container/compliance/generators/
# against the parent build stage's filesystem, applies the license policy
# gate, and exposes /legal/ + /sboms/ for the next two stages to fan out.
#
# Per-framework variations:
#   - sglang uses `--site-packages "$(... sysconfig ...)"` because the
#     upstream image installs into system Python via
#     `pip install --break-system-packages`, not a venv.
#   - native always runs (image filter "{framework}-{target}[-efa]"), attributing
#     the from-source binaries the python/rust/dpkg scanners miss (ffmpeg, libvpx,
#     UCX, NIXL, gdrcopy, libfabric, etcd, nats-server) per native_packages.yaml's
#     per-image `images:` lists; make_efa adds the EFA-only entries.

FROM pre_runtime AS licenses

USER root
RUN mkdir -p /legal /sboms
COPY --chown=root:0 container/compliance /opt/compliance
ENV PYTHONPATH=/opt

# Real crate LICENSE files harvested from the cargo registry in wheel_builder
# (empty when none were harvested -- the rust generator then falls back to
# canonical SPDX text). Keyed "<name>-<version>". wheel_builder_base always
# creates the dir, so this COPY never fails even for wheel-less targets.
COPY --from=wheel_builder /opt/dynamo/rust-licenses /tmp/rust-licenses

# BASELINE_SBOM_FILE: the per-arch baseline SBOM *stem* (e.g.
# "cuda@2ab6381d") under /opt/compliance/base_sboms/. We append
# "-${TARGETARCH}.cdx.json" so each platform of a multi-arch build subtracts
# its OWN-arch floor — the amd64 baseline would otherwise under-attribute a
# package present in the amd64 base but not the arm64 base that we install on
# arm64. Rendered from context.yaml's baseline_sbom by render.py; empty when no
# baseline is captured (NOTICES then cover the full image — correct but
# unfiltered).
ARG BASELINE_SBOM_FILE="cuda@2ab6381d"
ARG TARGETARCH
# Resolve where this image's Python packages live at runtime rather than per
# framework: venv-based images export VIRTUAL_ENV (trtllm, vllm xpu/cpu, dev),
# while images that install into system Python leave it unset (vllm cuda via
# `pip --system`, sglang via `pip --break-system-packages`). Pick the matching
# generator flag so it always finds the deps — passing an empty
# `--venv ${VIRTUAL_ENV}` is what broke system-Python images.
RUN if [ -n "${VIRTUAL_ENV:-}" ]; then PKG_ARG="--venv ${VIRTUAL_ENV}"; else PKG_ARG="--site-packages $(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; fi && \
    python3 -m compliance.generators \
    --ecosystem python,rust,dpkg,native \
    ${PKG_ARG} \
    --rust-licenses-dir /tmp/rust-licenses \
    --output-dir /legal \
    --policy /opt/compliance/policy/licenses.toml \
    --native-yaml /opt/compliance/native_packages.yaml \
    --native-image vllm-runtime \
    ${BASELINE_SBOM_FILE:+--subtract-sbom /opt/compliance/base_sboms/${BASELINE_SBOM_FILE}-${TARGETARCH}.cdx.json} \
    -v
# Policy gate runs on the single unified CSV (its `ecosystem` column scopes each
# row), replacing the per-ecosystem loop. Non-zero exit fails the build.
RUN python3 -m compliance.policy.validate \
        --policy /opt/compliance/policy/licenses.toml \
        --input /legal/osrb-deps.csv

# Media-codec allowlist gate: scans THIS stage's filesystem (==
# the shipped image tree, since licenses is FROM pre_runtime) and fails the build
# if a media-codec library/binary (a third-party libav*, libx264/265, or a stray
# or imageio-bundled ffmpeg) ships outside our in-tree allowlist or a reasoned
# exception. Feeds the generated delta SBOM in too, for an ffmpeg-version floor.
# Files, not just the SBOM, because statically-bundled codec .so's don't appear
# as components.
#
# CUDA only for now. The XPU image is not currently published on NGC, so it does
# not yet require the codecs to be removed. It would also fail this gate today:
# the purge the gate depends on sits in the `device == "cuda"` block of
# vllm_runtime.Dockerfile, so the XPU image still carries its base image's media
# stack. Enable both together if that changes.

RUN python3 -m compliance.scan_codecs \
        --root / \
        --policy /opt/compliance/policy/codec_policy.yaml \
        --sbom /legal/osrb.cdx.json \
        --image vllm-runtime \
        --fail-on-findings -v

#######################################
####### Compliance: artifact ##########
#######################################
#
# Single FROM-scratch stage exposing the unified compliance tree for CI
# extraction (one `-compliance` artifact): flat NOTICES-<Eco>.txt + the unified
# osrb-deps.csv (with Notes) + osrb.cdx.json (delta CycloneDX). Export is bounded
# by these files' size (a few MB) regardless of runtime image size.

FROM scratch AS compliance_artifact
COPY --from=licenses /legal/ /

#######################################
########## Compliance: sources ########
#######################################
#
# Collects third-party source archives on top of the runtime baseline.
# Gated on ENABLE_SOURCE_ARCHIVAL -- default off so PR builds stay fast;
# CI flips it on for nightly + release/*.*.* branch pushes (see
# .github/workflows/post-merge-ci.yml and nightly-ci.yml).

FROM pre_runtime AS sources_collect

USER root
RUN mkdir -p /sources /opt/compliance /opt/native-sources /opt/dynamo-vendor-full
COPY --chown=root:0 container/compliance /opt/compliance
ENV PYTHONPATH=/opt
COPY --from=wheel_builder /tmp/native-sources/ /opt/native-sources/
COPY --from=wheel_builder /tmp/dynamo-vendor-full/ /opt/dynamo-vendor-full/

ARG ENABLE_SOURCE_ARCHIVAL=false
ARG BASELINE_SBOM_FILE="cuda@2ab6381d"
ARG TARGETARCH
RUN if [ "$ENABLE_SOURCE_ARCHIVAL" = "true" ]; then \
if [ -n "${VIRTUAL_ENV:-}" ]; then RUST_PKG_ARG="--rust-venv ${VIRTUAL_ENV}"; else RUST_PKG_ARG="--rust-site-packages $(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; fi && \
        python3 -m compliance.collect_sources \
            --ecosystem dpkg --ecosystem rust --ecosystem native \
            --output-zip /sources.zip \
            --sources-root /sources \
            --native-source-dir /opt/native-sources \
            ${RUST_PKG_ARG} \
            --rust-vendor-full /opt/dynamo-vendor-full \
            ${BASELINE_SBOM_FILE:+--baseline-sbom /opt/compliance/base_sboms/${BASELINE_SBOM_FILE}-${TARGETARCH}.cdx.json} \
            -v ; \
    else \
        python3 -c "import zipfile; zipfile.ZipFile('/sources.zip','w').close()" ; \
    fi

FROM scratch AS sources_archive
COPY --from=sources_collect /sources.zip /sources.zip

# === END templates/compliance.Dockerfile ===

FROM pre_runtime AS runtime

COPY --from=licenses /legal /legal

# --- TRTLLM Stages

# --- No TRTLLM stages included

# --- Development Stages

# --- No development stages included
# Morph production overlay: keep the exact qualified DSV4 engine image and
# replace only Dynamo's compiled runtime plus the vLLM publisher source.
ARG HEARTBEAT_BASE_IMAGE
FROM ${HEARTBEAT_BASE_IMAGE} AS heartbeat_overlay
USER root
COPY --from=runtime_wheel_builder /opt/dynamo/dist/ai_dynamo_runtime-*.whl /tmp/dynamo-heartbeat/
COPY --from=runtime_wheel_builder /opt/dynamo/components/src/dynamo/vllm/publisher.py /tmp/dynamo-heartbeat/publisher.py
RUN python3 -m pip install --no-deps --force-reinstall /tmp/dynamo-heartbeat/ai_dynamo_runtime-*.whl \
    && target=$(python3 -c 'import pathlib, dynamo.vllm.publisher as p; print(pathlib.Path(p.__file__))' 2>/dev/null | tail -n1) \
    && install -m 0644 /tmp/dynamo-heartbeat/publisher.py "$target" \
    && rm -rf /tmp/dynamo-heartbeat \
    && python3 -c 'from importlib.metadata import version; assert version("ai-dynamo") == "1.4.0"; assert version("ai-dynamo-runtime") == "1.4.0"; import dynamo.vllm.publisher as p; assert "self.num_gpu_block" in open(p.__file__).read(); print("Dynamo idle load heartbeat overlay OK")'
