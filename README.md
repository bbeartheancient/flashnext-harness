# flashnext-harness

A working deployment stack for **Qwen3.8-Flash-Next** (125B MoE, GGUF) on
**2x Intel Arc Pro B70 (32 GB each)** via llama.cpp's SYCL backend, plus the
**NeoHarness** agentic harness (BPE: Belief/Progress/Experience state on a
bicameral n-gram buffer) and the **AFI SYCL/ESIMD FWHT kernels** used by the
latent lane.

## Contents

- `llama.cpp-patches/flashnext-mtp-hiddenstates.patch` — diff against
  llama.cpp base commit `9723942` (17 files): Qwen3.8-Flash-Next
  (`qwen4exp`) architecture support, MTP speculative decoding, and the
  server `/hidden_states` endpoint.
- `scripts/flashnext-serve.sh` — proven 2x B70 launch config
  (tensor-split, expert offloads, 48k ctx). Paths via `$LLAMA_BIN` /
  `$MODEL_DIR`.
- `docs/flashnext_recipe.md` — full recipe: download, GGUF
  convert/quantize, build, launch, and the failure modes behind each flag.
- `neoharness/`, `tests/`, `examples/`, `scripts/bench_*.py` — the
  NeoHarness agentic harness (numpy-only host side; see
  `README.neoharness.md`).
- `fabric/` — FastAPI orchestrator + LCARS webGUI (port 18093) including
  the BPE bridge (`fabric/bpe_bridge.py`) that injects NeoHarness state
  into OpenCode sessions.
- `kernels/m1multitronic-sycl/` — SYCL FWHT (SLM / XMX / ESIMD), QK-int8,
  dequant and RTU kernels with CMake build (intel/llvm v7 preset for B70).

## Quickstart

1. **Build llama.cpp with SYCL + the patch**

   ```sh
   git clone https://github.com/ggml-org/llama.cpp
   cd llama.cpp && git checkout 9723942
   git apply /path/to/flashnext-harness/llama.cpp-patches/flashnext-mtp-hiddenstates.patch
   cmake -B build-sycl -DGGML_SYCL=ON -DCMAKE_BUILD_TYPE=Release
   cmake --build build-sycl --target llama-server -j
   ```

   Requires oneAPI (Level Zero). See `docs/flashnext_recipe.md`.

2. **Get the model and convert to GGUF** (details in
   `docs/flashnext_recipe.md`):

   ```sh
   export MODEL_DIR=$HOME/models/flashnext
   hf download Qwen/Qwen3.8-Flash-Next --local-dir "$MODEL_DIR/flashnext-bf16"
   python convert_hf_to_gguf.py "$MODEL_DIR/flashnext-bf16" \
     --outfile "$MODEL_DIR/flashnext-bf16.gguf" --outtype bf16
   # then quantize + the PLE-Q6K tensor override per the recipe
   ```

3. **Serve on both B70s**

   ```sh
   export LLAMA_BIN=/path/to/llama.cpp/build-sycl/bin/llama-server
   bash scripts/flashnext-serve.sh    # listens on 127.0.0.1:8001
   ```

4. **Run the NeoHarness loop**

   ```sh
   python -m neoharness "your goal" --workspace DIR --model <id>
   # or the OpenAI-compatible server:
   python -m neoharness serve --port 8393
   ```

   Tests: `python -m pytest tests -q` (offline, no GPU).

5. **Kernels** (optional, for the latent lane):

   ```sh
   cd kernels/m1multitronic-sycl
   bash scripts/with-intel-llvm.sh cmake --preset llvm7
   bash scripts/with-intel-llvm.sh cmake --build --preset llvm7
   ```

   Needs intel/llvm v7.0.0 (`AFI_DPCPP_ROOT` overrides the default
   `$HOME/intel-llvm-7.0.0`). `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`.

## Docs

- `docs/flashnext_recipe.md` — the proven Flash-Next launch recipe.
- `README.neoharness.md` — the NeoHarness design doc (BPE harness,
  agentic layer rules, echo/transport gates).
- `kernels/m1multitronic-sycl/` — kernel build via CMake presets;
  `scripts/env.sh` carries the Battlemage/B70 runtime environment.

## License

MIT (see `LICENSE`). llama.cpp itself is MIT; the patch applies to an
MIT-licensed tree.
