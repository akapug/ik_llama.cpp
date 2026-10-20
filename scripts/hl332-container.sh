#!/usr/bin/env bash
# Container-only dependencies and synthetic CPU controls; no device admission.
set -euo pipefail
out=$1
cores=$2
owner_uid=$3
owner_gid=$4
trap 'chown -R "$owner_uid:$owner_gid" "$out"' EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends build-essential cmake libvulkan-dev glslc spirv-headers ca-certificates python3
cmake --version > "$out/toolchain.txt"
g++ --version >> "$out/toolchain.txt"
/usr/local/cuda/bin/nvcc --version >> "$out/toolchain.txt"
glslc --version >> "$out/toolchain.txt"
common=(
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF
  '-DCMAKE_C_FLAGS=-march=znver1 -mtune=znver1'
  '-DCMAKE_CXX_FLAGS=-march=znver1 -mtune=znver1'
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF
  -DGGML_EXPERT_CHUNKING=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_IQK_FA_ALL_QUANTS=ON
  -DGGML_NCCL=OFF -DGGML_CCACHE=OFF -DGGML_OPENMP=ON
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
)
# The CPU-only executable has no GPU registry entries. This is a control build,
# not the mixed deliverable and not a substitute for its compilation.
cmake -S /source -B "$out/cpu" "${common[@]}" -DGGML_CUDA=OFF -DGGML_VULKAN=OFF 2>&1 | tee "$out/cpu-configure.log"
cmake --build "$out/cpu" --parallel "$cores" --target test-backend-ops 2>&1 | tee "$out/cpu-build.log"
"$out/cpu/bin/test-backend-ops" --hl332-cpu 2>&1 | tee "$out/cpu-controls.log"
# Negative controls must fail: absent backend and a zero-matched operation.
if "$out/cpu/bin/test-backend-ops" --hl332-required HL332-absent > "$out/absent-backend.log" 2>&1; then
  printf 'FAIL: missing backend was accepted\n' >&2; exit 1
fi
if "$out/cpu/bin/test-backend-ops" test -b CPU -o HL332-absent > "$out/zero-cases.log" 2>&1; then
  printf 'FAIL: zero cases were accepted\n' >&2; exit 1
fi
cmake -S /source -B "$out/mixed" "${common[@]}" \
  -DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_HIPBLAS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ 2>&1 | tee "$out/mixed-configure.log"
cmake --build "$out/mixed" --parallel "$cores" --target llama-server test-backend-ops 2>&1 | tee "$out/mixed-build.log"
# Do not run even the mixed executable here: this slot has no device custody.
mkdir "$out/pkg" "$out/pkg/bin" "$out/pkg/lib"
cp "$out/mixed/bin/llama-server" "$out/mixed/bin/test-backend-ops" "$out/pkg/bin/"
for lib in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do
  cp -L "/usr/local/cuda/lib64/$lib" "$out/pkg/lib/"
done
ldd "$out/mixed/bin/test-backend-ops" > "$out/link-closure.txt"
find "$out/pkg" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/package-sha256.txt"
printf 'BUILD_AND_SYNTHETIC_CPU_PASS\n' > "$out/result.txt"
