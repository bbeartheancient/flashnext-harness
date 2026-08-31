# source from bash:  source scripts/env.sh
# fish:              bash scripts/with-oneapi.sh <cmd>
#                     or:  bass source "$AFI_ROOT/scripts/env.sh"

if [ -f /opt/intel/oneapi/setvars.sh ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh >/dev/null
fi

export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export SYCL_DEVICE_FILTER="${SYCL_DEVICE_FILTER:-level_zero:gpu}"
export ZES_ENABLE_SYSMAN="${ZES_ENABLE_SYSMAN:-1}"

# Battlemage / B70 (seated 2026-08-19; audited vs Puget B70 multi-GPU guide
# 2026-08-22). A770 did not need these.
# Track 0 joint_matrix needs intel/llvm v7, not this 2025.3 pin:
#   bash scripts/with-intel-llvm.sh …
export NEO_ReadDeviceBinaryBuiltins=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CCL_TOPO_P2P_ACCESS=0
# Xe2: regular command lists; immediate mode unstable on Battlemage
# (Puget B70 guide appendix).
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMAND_LISTS="${SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMAND_LISTS:-0}"
# NOTE: Puget sets ZES_ENABLE_SYSMAN=0 for vLLM container init stability;
# we keep 1 because the fabric Pulse/hwmon scrape depends on it. If a
# service shows sysman init errors, override =0 for that service only.

export AFI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AFI_HOA64_ROOT="${AFI_HOA64_ROOT:-$HOME/hoa64}"
export AFI_ARRAYFIRE_ROOT="${AFI_ARRAYFIRE_ROOT:-$HOME/arrayfire}"
export PYTHONPATH="${AFI_ROOT}/python:${HOME}:${PYTHONPATH:-}"

if command -v icpx >/dev/null 2>&1; then
    export CXX="${CXX:-icpx}"
    export CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-icpx}"
fi
