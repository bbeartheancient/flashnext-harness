#!/usr/bin/env bash
# Track 0 kernel env: intel/llvm v7.0.0 (DPC++ 7 / oneAPI 2026.0 grain)
# on top of the installed Level Zero / IGC from oneAPI 2025.3.
#
# Do NOT use this to restart Comfy / llama.cpp.pinned. Those stay on
# /opt/intel/oneapi 2025.3 (Alchemist-era pin). v7 is ABI-breaking.
#
#   bash scripts/with-intel-llvm.sh cmake --preset llvm7
#   bash scripts/with-intel-llvm.sh cmake --build --preset llvm7
#   bash scripts/with-intel-llvm.sh ./build-llvm7/afi-inventory -o inventory/b70.json
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM="${AFI_DPCPP_ROOT:-$HOME/intel-llvm-7.0.0}"
if [[ ! -x "$LLVM/bin/clang++" ]]; then
    echo "intel/llvm missing at $LLVM (set AFI_DPCPP_ROOT). See docs/b70-inventory.md" >&2
    exit 1
fi
# setvars.sh reads possibly-unset ICD vars.
set +u
# shellcheck disable=SC1091
source "${ROOT}/scripts/env.sh"
set -u
export PATH="$LLVM/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM/lib:${LD_LIBRARY_PATH:-}"
export AFI_DPCPP_ROOT="$LLVM"
export CXX="${CXX:-clang++}"
export CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-clang++}"
if [ "$#" -eq 0 ]; then
    echo "usage: $0 <command> [args...]" >&2
    echo "clang++=$(command -v clang++)  sycl-ls=$(command -v sycl-ls)" >&2
    echo "AFI_DPCPP_ROOT=$AFI_DPCPP_ROOT" >&2
    exit 2
fi
exec "$@"
