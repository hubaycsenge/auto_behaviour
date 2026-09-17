#!/usr/bin/env bash
# Build llama.cpp with CUDA for ABC's vlm_llamacpp engine.
#
# This is what unlocks the two thirds of the cluster vLLM cannot use: the GTX
# 1080 nodes are below vLLM's compute-capability floor entirely, and Qwen3-VL's
# vision path does not work on the 2080 Ti / Titan RTX nodes under vLLM.
#
#   srun -p medium --gres=gpu:1 --pty setup/install_llamacpp.sh
set -euo pipefail

prefix="${ABC_LLAMA_PREFIX:-$HOME/abc_llama}"
src="$prefix/llama.cpp"

command -v cmake  >/dev/null || { echo "cmake is required" >&2; exit 1; }
command -v nvcc   >/dev/null || echo "WARNING: nvcc not found; building CPU-only"

mkdir -p "$prefix"
if [ ! -d "$src" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$src"
else
  git -C "$src" pull --ff-only
fi

# Build for every architecture on this cluster: Pascal (61), Turing (75),
# Ampere (80 for A100, 86 for 3090/A4000). One binary then runs anywhere.
cmake -S "$src" -B "$src/build" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="61;75;80;86" \
  -DLLAMA_CURL=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$src/build" --config Release -j "$(nproc)" --target llama-server llama-mtmd-cli

mkdir -p "$prefix/bin"
cp "$src/build/bin/llama-server" "$prefix/bin/"
cp "$src/build/bin/llama-mtmd-cli" "$prefix/bin/" 2>/dev/null || true

echo
echo "== done =="
echo "Add to PATH:  export PATH=$prefix/bin:\$PATH"
echo "Then fetch a model:  setup/fetch_gguf.sh"
