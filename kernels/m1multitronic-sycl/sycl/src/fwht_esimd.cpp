#include "afi/fwht_esimd.hpp"
#include "afi/fwht.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <type_traits>

namespace esimd = sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;

namespace afi {
namespace {

// One ESIMD thread = one half-precision (bf16/fp16) row of length N. The
// whole transform is log2(N) in-register butterfly stages over region
// views; fp32 arithmetic; single end-scale (== per-stage 1/sqrt(2),
// compounded). No SLM, no barriers, no DPAS — pure bandwidth. LSC messages
// cap at 64 elements, so loads/stores are chunked via compile-time
// recursion while the transform runs on the full-width simd.
// Unique kernel-name type per (elem, N) instantiation.
template <typename T, int N> struct FwhtEsimdName {};

template <typename T, int C, int N>
inline std::enable_if_t<(C * 64 >= N)>
load_chunks(esimd::simd<float, N> &, const uint16_t *) {}

template <typename T, int C, int N>
inline std::enable_if_t<(C * 64 < N)>
load_chunks(esimd::simd<float, N> &v, const uint16_t *p) {
    constexpr int OFF = C * 64;
    constexpr int CNT = (N - OFF >= 64) ? 64 : (N - OFF);
    auto part = xesimd::lsc_block_load<T, CNT>(
        reinterpret_cast<const T *>(p + OFF));
    const esimd::simd<float, CNT> pf = part; // exact value conversion
    v.template select<CNT, 1>(OFF) = pf;
    load_chunks<T, C + 1, N>(v, p);
}

template <typename T, int C, int N>
inline std::enable_if_t<(C * 64 >= N)>
store_chunks(uint16_t *, esimd::simd<float, N>) {}

template <typename T, int C, int N>
inline std::enable_if_t<(C * 64 < N)>
store_chunks(uint16_t *p, esimd::simd<float, N> v) {
    constexpr int OFF = C * 64;
    constexpr int CNT = (N - OFF >= 64) ? 64 : (N - OFF);
    // NOTE: by-value — select() requires a non-const lvalue.
    const esimd::simd<float, CNT> pf = v.template select<CNT, 1>(OFF);
    const esimd::simd<T, CNT> pb = pf; // RNE
    xesimd::lsc_block_store<T, CNT>(reinterpret_cast<T *>(p + OFF), pb);
    store_chunks<T, C + 1, N>(p, v);
}

// Butterfly stages as compile-time recursion: region parameters must be
// constant expressions, so the stage size H is a template argument. Each
// stage pairs x[i] with x[i+H] over every block of size 2H — expressed as a
// [G x 2H] 2D region view with left/right half-column slices (G = N/2H).
// Stage order and fp32 arithmetic match the host oracle exactly
// (fwht_host_row / python/afi/fwht.py); scale applied after all stages.
template <int H, int N>
inline std::enable_if_t<(H >= N)> stages(esimd::simd<float, N> &) {}

template <int H, int N>
inline std::enable_if_t<(H < N)> stages(esimd::simd<float, N> &v) {
    constexpr int G = N / (2 * H);
    if constexpr (G == 1) {
        // Terminal stage: degenerate [1 x N] 2D views miscompose on
        // BMG-G31 + llvm v7; plain contiguous half-selects are exact.
        auto lo = v.template select<N / 2, 1>(0);
        auto hi = v.template select<N / 2, 1>(N / 2);
        const esimd::simd<float, N / 2> a = lo;
        const esimd::simd<float, N / 2> b = hi;
        lo = a + b;
        hi = a - b;
    } else {
        auto m = v.template bit_cast_view<float, G, 2 * H>();
        auto lo = m.template select<G, 1, H, 1>(0, 0);
        auto hi = m.template select<G, 1, H, 1>(0, H);
        const esimd::simd<float, G * H> a = lo;
        const esimd::simd<float, G * H> b = hi;
        lo = a + b;
        hi = a - b;
    }
    stages<H * 2, N>(v);
}

template <typename T, int N>
void launch_half(sycl::queue &q, const void *src, void *dst, int64_t n_rows,
                 float scale) {
    q.parallel_for<FwhtEsimdName<T, N>>(
        sycl::range<1>(size_t(n_rows)),
        [=](sycl::id<1> r) SYCL_ESIMD_KERNEL {
            const int64_t row = static_cast<int64_t>(r.get(0));
            esimd::simd<float, N> v;
            load_chunks<T, 0, N>(v,
                                 static_cast<const uint16_t *>(src) + row * N);
            stages<1, N>(v);
            v *= scale;
            store_chunks<T, 0, N>(static_cast<uint16_t *>(dst) + row * N, v);
        });
}

} // namespace

bool fwht_esimd_ok(const void *src, const void *dst, int n) {
    if (!fwht_supported(n)) return false;
    const auto sa = reinterpret_cast<std::uintptr_t>(src);
    const auto da = reinterpret_cast<std::uintptr_t>(dst);
    // lsc block ops need at least DW alignment.
    return (sa & 0x3) == 0 && (da & 0x3) == 0;
}

bool fwht_device_esimd_bf16(sycl::queue &q, const void *src, void *dst,
                            int n, int64_t n_rows, float scale, bool wait) {
    return fwht_device_esimd_half(q, src, dst, n, n_rows, scale,
                                  FwhtElem::BF16, wait);
}

bool fwht_device_esimd_half(sycl::queue &q, const void *src, void *dst,
                            int n, int64_t n_rows, float scale, FwhtElem elem,
                            bool wait) {
    if (n_rows <= 0) {
        if (wait) q.wait_and_throw();
        return true;
    }
    if (elem != FwhtElem::BF16 && elem != FwhtElem::F16) return false;
    if (!fwht_esimd_ok(src, dst, n)) return false;

#define AFI_ESIMD_LAUNCH(T_, N_) launch_half<T_, N_>(q, src, dst, n_rows, scale)
#define AFI_ESIMD_SWITCH(T_)                                                   \
    switch (n) {                                                               \
    case 16: AFI_ESIMD_LAUNCH(T_, 16); break;                                  \
    case 32: AFI_ESIMD_LAUNCH(T_, 32); break;                                  \
    case 64: AFI_ESIMD_LAUNCH(T_, 64); break;                                  \
    case 128: AFI_ESIMD_LAUNCH(T_, 128); break;                                \
    case 256: AFI_ESIMD_LAUNCH(T_, 256); break;                                \
    default: return false;                                                     \
    }
    if (elem == FwhtElem::BF16) {
        AFI_ESIMD_SWITCH(sycl::ext::oneapi::bfloat16)
    } else {
        AFI_ESIMD_SWITCH(sycl::half)
    }
#undef AFI_ESIMD_SWITCH
#undef AFI_ESIMD_LAUNCH

    if (wait) q.wait_and_throw();
    return true;
}

} // namespace afi
