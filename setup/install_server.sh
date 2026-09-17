#!/usr/bin/env bash
# Install ABC's server-side environment on the cluster.
#
#   setup/install_server.sh                 core only (mock engine, BORIS, SLURM)
#   setup/install_server.sh --vllm          + vLLM video-LLM engine   (Ampere: nipg38/10/32)
#   setup/install_server.sh --pose          + YOLO/ByteTrack pose engine
#   setup/install_server.sh --audio         + CLAP zero-shot audio classifier
#   setup/install_server.sh --all           everything above
#
# The heavy extras are separate because they pull in torch: install them from a
# node that has the GPU you intend to use, so pip picks a matching CUDA wheel.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${ABC_VENV:-$HOME/abc_env}"
want_vllm=0 want_pose=0 want_audio=0

for arg in "$@"; do
  case "$arg" in
    --vllm)  want_vllm=1 ;;
    --pose)  want_pose=1 ;;
    --audio) want_audio=1 ;;
    --all)   want_vllm=1; want_pose=1; want_audio=1 ;;
    --venv=*) venv="${arg#*=}" ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

echo "== ABC server install =="
echo "   checkout: $root"
echo "   venv:     $venv"

# Debian's python3-venv is missing on these nodes, so bootstrap pip by hand.
if [ ! -x "$venv/bin/python" ]; then
  echo "-- creating the virtualenv"
  python3 -m venv "$venv" 2>/dev/null || {
    echo "   (ensurepip unavailable; bootstrapping pip manually)"
    python3 -m venv --without-pip "$venv"
    curl -sS https://bootstrap.pypa.io/get-pip.py -o "$venv/get-pip.py"
    "$venv/bin/python" "$venv/get-pip.py" -q
    rm -f "$venv/get-pip.py"
  }
fi

pip="$venv/bin/python -m pip"
$pip install -q --upgrade pip wheel

echo "-- core dependencies"
# PyAV carries its own ffmpeg: these nodes have none, and nothing here may
# depend on a system codec being present.
$pip install -q "av>=11" "Pillow>=10" "numpy>=1.24"

if [ "$want_audio" = 1 ]; then
  echo "-- audio engine (CLAP zero-shot classifier)"
  $pip install -q "torch" "transformers>=4.40"
fi

if [ "$want_pose" = 1 ]; then
  echo "-- pose engine (YOLO + ByteTrack)"
  $pip install -q "ultralytics>=8.3"
fi

if [ "$want_vllm" = 1 ]; then
  echo "-- vLLM engine"
  cc="$("$venv/bin/python" - <<'PY'
try:
    import torch
    print(int(sum(x * y for x, y in zip(torch.cuda.get_device_capability(0), (10, 1)))))
except Exception:
    print(0)
PY
)"
  if [ "$cc" != 0 ] && [ "$cc" -lt 80 ]; then
    echo "   WARNING: this GPU is compute capability $((cc / 10)).$((cc % 10))."
    echo "   vLLM needs 8.0+ for bfloat16, and Qwen3-VL's vision path does not work"
    echo "   on Turing under vLLM at all. Use --llamacpp on this node instead."
  fi
  $pip install -q "vllm>=0.10"
fi

cat > "$venv/abc-env.sh" <<ENV
# Source this to put ABC on PATH.
export PATH="$root/bin:\$PATH"
export ABC_VENV="$venv"
export HF_HOME="\${HF_HOME:-$HOME/.cache/huggingface}"
ENV

echo
echo "== done =="
echo "Add ABC to your PATH:    source $venv/abc-env.sh"
echo "Then check the install:  abc check"
