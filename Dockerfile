#
#   docker build -t aqwenator .
#   docker run --gpus all -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
#       aqwenator Qwen/Qwen3.5-9B
#
# The model must already be in the mounted Hugging Face cache (or pass a directory).
# Building needs no GPU. GPU_ARCH is the oldest architecture the kernels are compiled for;
# the driver compiles them for the device it finds at start-up.
ARG CUDA_VERSION=13.0.3
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build
ARG RUST_TOOLCHAIN=1.99.1
# Must match the cuda-rust rev in Cargo.toml.
ARG CUDA_RUST_REV=8f5f2a5d6c9a5d16a75036fbcc639f751b68d537
ARG GPU_ARCH=sm_80

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git gnupg build-essential pkg-config \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && . /etc/os-release \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/${VERSION_CODENAME}/ llvm-toolchain-${VERSION_CODENAME}-21 main" > /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends clang-21 libclang-21-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    LIBCLANG_PATH=/usr/lib/llvm-21/lib \
    PATH=/root/.cargo/bin:/usr/lib/llvm-21/bin:$PATH

RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --default-toolchain none --profile minimal \
    && rustup toolchain install ${RUST_TOOLCHAIN} --profile minimal -c rust-src -c rustc-dev -c llvm-tools \
    && cargo +${RUST_TOOLCHAIN} install --git https://github.com/NVIDIA/cuda-rust.git --rev ${CUDA_RUST_REV} --locked cargo-oxide

WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/src/target \
    cargo oxide build --arch ${GPU_ARCH} -- --release --locked -p aq-server -p aq-engine --bins \
    && mkdir /out \
    && cp target/release/aq-serve target/release/aq-bench target/release/aq-check target/release/aq-ceiling /out/

# The binaries alone (`--target binaries --output type=local,dest=bin`); they need
# libcublasLt.so.13 and the NVIDIA driver on the host.
FROM scratch AS binaries
COPY --from=build /out/ /

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}
# cuBLASLt is the only toolkit library the engine links; libcuda comes from the host driver.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcublas-13-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /usr/local/bin/
EXPOSE 8000
ENTRYPOINT ["aq-serve"]
