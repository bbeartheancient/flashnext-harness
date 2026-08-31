#pragma once

#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace afi {

enum class FwhtImpl {
    Auto,  // XMX joint_matrix when the device advertises a 8×16×16 tile
    Slm,   // XVE add/sub butterflies in SLM (A770 + B70 fallback)
    Xmx,   // H.16 DPAS atom + tail butterflies (B70 / intel-llvm v7)
    Esimd, // GRF-resident row kernel, bf16 only (docs: GRF-FWHT rewrite)
};

// Element type of the src/dst buffers. Half-native = fp16/bf16 in, fp32
// compute in-kernel, same half type out (no device-side cast pass).
enum class FwhtElem {
    F32, // fp32 in/out (host-oracle path; existing checks/bench)
    F16, // fp16 in/out, fp32 internal (half-native)
    BF16,// bf16 in/out, fp32 internal (half-native)
};

const char *fwht_impl_name(FwhtImpl impl);

// True when ext_intel_matrix is present and a BF16 tile with N=K=16 exists.
bool fwht_xmx_ok(const sycl::device &dev);

// Host reference — same butterflies as python/afi/fwht.py.
void fwht_host(float *rows, int n, int64_t n_rows, float scale);

// Device FWHT. n must be 16, 32, 64, 128, or 256. scale is usually 1/sqrt(n).
// Auto uses XMX on B70 (llvm v7) and SLM otherwise.
// wait=true (default) drains the queue — C++ checks need it. The torch op
// passes false and stays on the current XPU stream.
void fwht_device(sycl::queue &q, const float *src, float *dst, int n,
                 int64_t n_rows, float scale, FwhtImpl impl = FwhtImpl::Auto,
                 bool wait = true);

// Typed src/dst FWHT. elem selects the buffer element type; the butterfly
// math stays fp32 (or the DPAS leaf for the H.16 atom in XMX mode). This
// is the half-native in/out path the torch op uses for fp16/bf16 tensors —
// no separate cast-in / cast-out kernels.
void fwht_device_typed(sycl::queue &q, const void *src, void *dst, int n,
                       int64_t n_rows, float scale, FwhtElem elem,
                       FwhtImpl impl = FwhtImpl::Auto, bool wait = true);

inline float ortho_scale(int n) { return 1.0f / std::sqrt(static_cast<float>(n)); }

// Tunable XMX launch config (Layer 2 autotuner). TN=TK=16 stay fixed: the
// DPAS B tile is a square H(16) atom, and (16,16) is the only square tile
// B70 advertises for fp16/bf16. tm/sg are kernel-binary params (template
// instantiations); wg_n/wg_m/slm_pad/prefetch are runtime launch params.
struct FwhtXmxConfig {
    int tm = 8;       // DPAS m: rows per joint_matrix op, {1,2,4,8}
    int sg = 16;      // sub-group size, {16, 32}
    int wg_n = 0;     // subgroups per WG along the row; 0 = n/16 (legacy)
    int wg_m = 1;     // row-tiles (of tm rows) per work-group
    int slm_pad = 0;  // extra elements on the SLM tile leading dim
    int prefetch = 0; // double-buffer global->SLM across wg_m tiles
};

// True when cfg is launchable for width n on dev (tile, SLM budget, sub-group
// size, staging register bound for prefetch). Does not JIT — a config can
// still fail at launch (e.g. sg=32 joint_matrix); callers catch that.
bool fwht_xmx_config_ok(const sycl::device &dev, const FwhtXmxConfig &cfg,
                        int n);

// Tuned launch-config winner for elem @ width n (Layer 2 autotuner, full
// lattice on BMG-G31 2026-08-20 at rows=28/2048/4096). Row-count-aware: small
// (< 1024 rows) picks wide work-groups for decode; larger rows use the prefill
// winner. Falls back to the legacy hardcoded config (tm=8, sg=16, wg_n=0) for
// widths/elems without a measured winner.
FwhtXmxConfig fwht_default_xmx_config(FwhtElem elem, int n, int64_t rows);

// Tuned XMX FWHT for fp16/bf16 buffers. Returns false without launching when
// the config is illegal (autotuner domain wipe); throws only on device errors.
bool fwht_device_xmx_tuned(sycl::queue &q, const void *src, void *dst, int n,
                           int64_t n_rows, float scale, FwhtElem elem,
                           const FwhtXmxConfig &cfg, bool wait = true);

inline bool fwht_supported(int n) {
    return n == 16 || n == 32 || n == 64 || n == 128 || n == 256;
}

} // namespace afi
