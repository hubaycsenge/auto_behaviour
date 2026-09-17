#!/usr/bin/env bash
# Download a GGUF vision model and its mmproj projector for vlm_llamacpp.
#
# A vision model is TWO files: the language weights and a separate mmproj
# projector. llama.cpp loads the language model happily without the projector
# and then never sees a frame, so ABC refuses to start unless both are set.
#
#   setup/fetch_gguf.sh                       Qwen2.5-VL-7B Q4_K_M (~5 GB)
#   setup/fetch_gguf.sh --model <repo> --file <name> --mmproj <name>
set -euo pipefail

dest="${ABC_GGUF_DIR:-$HOME/abc_models}"
repo="ggml-org/Qwen2.5-VL-7B-Instruct-GGUF"
file="Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
mmproj="mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"

while [ $# -gt 0 ]; do
  case "$1" in
    --model)  repo="$2"; shift 2 ;;
    --file)   file="$2"; shift 2 ;;
    --mmproj) mmproj="$2"; shift 2 ;;
    --dest)   dest="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$dest"
base="https://huggingface.co/$repo/resolve/main"
for name in "$file" "$mmproj"; do
  if [ -f "$dest/$name" ]; then
    echo "already have $name"
    continue
  fi
  echo "downloading $name"
  curl -fL --progress-bar "$base/$name" -o "$dest/$name.part"
  mv "$dest/$name.part" "$dest/$name"
done

cat <<MSG

== done ==
Set these as the vlm_llamacpp engine options in the client (Engines > Options):

  "model_path":  "$dest/$file"
  "mmproj_path": "$dest/$mmproj"
MSG
