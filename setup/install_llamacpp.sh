#!/usr/bin/env bash
# Build llama.cpp with CUDA for ABC's vlm_llamacpp engine.
#
# This is what unlocks the two thirds of the cluster vLLM cannot use: the GTX
# 1080 nodes are below vLLM's compute-capability floor entirely, and Qwen3-VL's
# vision path does not work on the 2080 Ti / Titan RTX nodes under vLLM.
#
#   setup/install_llamacpp.sh              build with CUDA
#   setup/install_llamacpp.sh --cpu        build without CUDA
#   setup/install_llamacpp.sh --jobs 4     cap build parallelism
#
# RUN THIS ON THE LOGIN NODE, not under srun. On this cluster the CUDA toolkit
# is installed only on the login node -- the compute nodes have no nvcc and no
# git -- so the build has to happen where the compiler is. The resulting binary
# runs anywhere, because it is built for every GPU architecture in the cluster.
set -euo pipefail

prefix="${ABC_LLAMA_PREFIX:-$HOME/abc_llama}"
src="$prefix/llama.cpp"
venv="${ABC_VENV:-$HOME/abc_env}"
want_cuda=1
jobs=""

while [ $# -gt 0 ]; do
  case "$1" in
    --cpu)    want_cuda=0; shift ;;
    --jobs)   jobs="$2"; shift 2 ;;
    --prefix) prefix="$2"; src="$prefix/llama.cpp"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# -- cmake ------------------------------------------------------------------
# Debian's cmake is absent on these nodes. PyPI ships it as a wheel with the
# real binary inside, so the ABC virtualenv is a perfectly good place to get
# one without root. It must be invoked by absolute path: cmake derives its
# module root from argv[0], and a relative path breaks that.
cmake="$(command -v cmake || true)"
if [ -z "$cmake" ]; then
  cmake="$(ls -1 "$venv"/lib/python*/site-packages/cmake/data/bin/cmake 2>/dev/null | head -1 || true)"
fi
if [ -z "$cmake" ]; then
  echo "-- cmake not found; installing it into $venv"
  if [ ! -x "$venv/bin/python" ]; then
    echo "   no virtualenv at $venv. Run setup/install_server.sh first, or set ABC_VENV." >&2
    exit 1
  fi
  "$venv/bin/python" -m pip install -q cmake
  cmake="$(ls -1 "$venv"/lib/python*/site-packages/cmake/data/bin/cmake 2>/dev/null | head -1 || true)"
fi
[ -n "$cmake" ] || { echo "cmake is still unavailable; install it and retry" >&2; exit 1; }
cmake="$(cd "$(dirname "$cmake")" && pwd)/$(basename "$cmake")"
echo "-- cmake: $cmake ($("$cmake" --version | head -1))"

command -v git >/dev/null || {
  echo "git is required and not on PATH. On this cluster git exists on the login" >&2
  echo "node but not on the compute nodes -- run this script on the login node." >&2
  exit 1
}

# -- CUDA -------------------------------------------------------------------
cuda_args=()
if [ "$want_cuda" = 1 ]; then
  if ! command -v nvcc >/dev/null; then
    cat >&2 <<MSG
nvcc is not on PATH, so a CUDA build is not possible here.

On this cluster the CUDA toolkit is installed on the LOGIN NODE only; the
compute nodes have neither nvcc nor git. If you launched this with srun, run it
directly on the login node instead. The binary it produces is built for every
GPU architecture in the cluster, so it runs on the compute nodes afterwards.

To build without GPU support anyway:  setup/install_llamacpp.sh --cpu
MSG
    exit 1
  fi

  # CUDA pins the host compiler it will accept, and Ubuntu's default gcc is
  # usually newer than that. Read the limit out of the toolkit's own header and
  # pick a matching compiler rather than letting nvcc fail halfway through.
  host_config="$(dirname "$(command -v nvcc)")/../include/crt/host_config.h"
  max_gcc=""
  if [ -f "$host_config" ]; then
    max_gcc="$(grep -oP '#if __GNUC__ > \K[0-9]+' "$host_config" | head -1 || true)"
  fi
  host_cxx=""
  if [ -n "$max_gcc" ] && [ "$(gcc -dumpversion | cut -d. -f1)" -gt "$max_gcc" ]; then
    for v in $(seq "$max_gcc" -1 9); do
      if command -v "g++-$v" >/dev/null; then host_cxx="$(command -v "g++-$v")"; break; fi
    done
    if [ -z "$host_cxx" ]; then
      echo "CUDA here accepts gcc <= $max_gcc but only gcc $(gcc -dumpversion) is installed." >&2
      echo "Install g++-$max_gcc, or build with --cpu." >&2
      exit 1
    fi
    echo "-- CUDA accepts gcc <= $max_gcc; using $host_cxx for device code"
    cuda_args+=("-DCMAKE_CUDA_HOST_COMPILER=$host_cxx")
  fi

  # Every GPU architecture on this cluster: Pascal (GTX 1080), Turing (2080 Ti,
  # Titan RTX), Ampere (A100 = 80, 3090/A4000 = 86). One binary runs on all.
  cuda_args+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61;75;80;86")
  echo "-- CUDA: $(nvcc --version | tail -2 | head -1)"
else
  echo "-- building without CUDA"
fi

# The login node is shared and has been pushed into swap by parallel builds
# before, so be modest by default rather than using every core.
if [ -z "$jobs" ]; then
  jobs="$(nproc)"
  [ "$jobs" -gt 6 ] && jobs=6
fi

# -- build ------------------------------------------------------------------
mkdir -p "$prefix"
if [ ! -d "$src" ]; then
  echo "-- cloning llama.cpp"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$src"
else
  echo "-- updating llama.cpp"
  git -C "$src" pull --ff-only
fi

echo "-- configuring"
"$cmake" -S "$src" -B "$src/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF \
  "${cuda_args[@]}"

echo "-- building with $jobs job(s)"
"$cmake" --build "$src/build" --config Release -j "$jobs" \
  --target llama-server llama-mtmd-cli

mkdir -p "$prefix/bin"
cp "$src/build/bin/llama-server" "$prefix/bin/"
cp "$src/build/bin/llama-mtmd-cli" "$prefix/bin/" 2>/dev/null || true

# Make it persistent for anyone sourcing the ABC environment.
if [ -f "$venv/abc-env.sh" ] && ! grep -q "abc_llama" "$venv/abc-env.sh"; then
  echo "export PATH=\"$prefix/bin:\$PATH\"" >> "$venv/abc-env.sh"
  echo "-- added $prefix/bin to $venv/abc-env.sh"
fi

cat <<MSG

== done ==
Binary:  $prefix/bin/llama-server
On PATH: export PATH="$prefix/bin:\$PATH"   (already added to abc-env.sh)

Next:    setup/fetch_gguf.sh     downloads a model + its mmproj projector
         abc check               vlm_llamacpp should now report available
MSG
