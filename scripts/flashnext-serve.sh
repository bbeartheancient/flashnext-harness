#!/usr/bin/env bash
# llama-server for Qwen3.8-Flash-Next on 2x Arc Pro B70 (SYCL).
# Proven launch config - see docs/flashnext_recipe.md
# ("Phase 2 - launch (PROVEN ...)") for the failure modes behind every flag.
set -eo pipefail

# Required environment:
#   LLAMA_BIN  - path to the llama-server binary built with the Flash-Next patch
#                (e.g. $LLAMA_CPP_DIR/build-sycl/bin/llama-server)
#   MODEL_DIR  - directory containing the Flash-Next GGUF
#   MODEL      - full path to the GGUF (default: $MODEL_DIR/flashnext-q4km-pleq6k.gguf)
LLAMA_BIN="${LLAMA_BIN:?set LLAMA_BIN to your patched llama-server binary}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the directory holding the GGUF}"
MODEL="${MODEL:-$MODEL_DIR/flashnext-q4km-pleq6k.gguf}"

# setvars.sh fails under systemd's clean env; export its LD_LIBRARY_PATH directly.
ONEAPI="${ONEAPI_ROOT:-/opt/intel/oneapi}"
export LD_LIBRARY_PATH="$ONEAPI/tcm/1.5/lib:$ONEAPI/umf/1.1/lib:$ONEAPI/tbb/2023.1/lib/intel64/gcc4.8:$ONEAPI/mkl/2026.1/lib:$ONEAPI/dnnl/2026.0/lib:$ONEAPI/compiler/2026.1/opt/compiler/lib:$ONEAPI/compiler/2026.1/lib:$ONEAPI/ccl/2022.1/lib:${LD_LIBRARY_PATH:-}"
export ONEAPI_DEVICE_SELECTOR="level_zero:*"

exec "$LLAMA_BIN" \
  -m "$MODEL" \
  --device SYCL0,SYCL1 --tensor-split 1,1 \
  -ngl 99 \
  -ot "per_layer_token_embd=CPU,blk\.[0-9]*[048]\.ffn_.*_exps=CPU" \
  -c 49152 -fa on -t 14 -np 1 \
  --embeddings -b 1536 --ubatch-size 1536 \
  --host 127.0.0.1 --port 8001
