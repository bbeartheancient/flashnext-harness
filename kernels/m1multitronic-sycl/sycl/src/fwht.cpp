#include "afi/fwht.hpp"
#include "afi/fwht_esimd.hpp"

#include <sycl/ext/oneapi/bfloat16.hpp>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace afi {
namespace {

// B70 advertises BF16 and FP16 8×16×16. FP16 has more mantissa; ±1 is
// exact in both. The half-native paths use HALF for fp16 input and BF16
// for bf16 input (the A-tile copies the input dtype); the fp32 path still
// rotates through a FP16 leaf. QuaRot-style rotate matches the host butterfly.
// leaf_t of the XMX kernel = T when T is fp16/bf16, else FP16.
namespace matrix = sycl::ext::oneapi::experimental::matrix;

constexpr int kTm = 8;
constexpr int kTn = 16;
constexpr int kTk = 16;
constexpr int kSg = 16;

// Sylvester H(16)_{i,j} = (-1)^{popcount(i & j)}. Matches python/afi/fwht.py.
constexpr int8_t sylvester_pm1(int i, int j) {
    int x = i & j;
    int bits = 0;
    for (int b = 0; b < 8; ++b)
        bits += (x >> b) & 1;
    return (bits & 1) ? static_cast<int8_t>(-1) : static_cast<int8_t>(1);
}
static_assert(sylvester_pm1(0, 0) == 1 && sylvester_pm1(1, 1) == -1,
              "Sylvester H2");

void fwht_host_row(float *y, int n, float scale) {
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += 2 * h) {
            for (int j = 0; j < h; ++j) {
                const float a = y[i + j];
                const float b = y[i + j + h];
                y[i + j] = a + b;
                y[i + j + h] = a - b;
            }
        }
    }
    if (scale != 1.0f) {
        for (int i = 0; i < n; ++i) y[i] *= scale;
    }
}

// Element type → half/double-width leaf: DPAS leaf for T = fp16/bf16 is T
// itself; the fp32 path rotates through a FP16 H.16 atom.
template <typename T>
using leaf_t = std::conditional_t<std::is_same_v<T, float>, sycl::half, T>;
static_assert(std::is_same_v<leaf_t<float>, sycl::half>);
static_assert(std::is_same_v<leaf_t<sycl::half>, sycl::half>);
static_assert(std::is_same_v<leaf_t<sycl::ext::oneapi::bfloat16>, sycl::ext::oneapi::bfloat16>);

template <typename T, int N>
class FwhtSlmKernel {
public:
    FwhtSlmKernel(const T *src, T *dst, int64_t n_rows, float scale,
                  sycl::local_accessor<float, 1> slm)
        : src_(src), dst_(dst), n_rows_(n_rows), scale_(scale), slm_(slm) {}

    void operator()(sycl::nd_item<1> it) const {
        const int64_t row = static_cast<int64_t>(it.get_group(0));
        const int lid = static_cast<int>(it.get_local_id(0));
        if (row >= n_rows_) return;

        slm_[lid] = static_cast<float>(src_[row * N + lid]);
        sycl::group_barrier(it.get_group());

        for (int h = 1; h < N; h *= 2) {
            const int stride = h * 2;
            if ((lid % stride) < h) {
                const float a = slm_[lid];
                const float b = slm_[lid + h];
                slm_[lid] = a + b;
                slm_[lid + h] = a - b;
            }
            sycl::group_barrier(it.get_group());
        }

        dst_[row * N + lid] = static_cast<T>(slm_[lid] * scale_);
    }

private:
    const T *src_;
    T *dst_;
    int64_t n_rows_;
    float scale_;
    sycl::local_accessor<float, 1> slm_;
};

// XMX path: one work-group owns kTm rows of length N. N/16 subgroups each
// apply H.16 via DPAS on one 16-wide chunk (no serial convert/store loop).
// Tail butterflies h ≥ 16 stay fp32 in SLM.
template <typename T, int N>
class FwhtXmxKernel {
public:
    using leaf = leaf_t<T>;

    FwhtXmxKernel(const T *src, T *dst, int64_t n_rows, float scale,
                  sycl::local_accessor<float, 1> tile,
                  sycl::local_accessor<leaf, 1> tile_h,
                  sycl::local_accessor<leaf, 1> h_slm)
        : src_(src), dst_(dst), n_rows_(n_rows), scale_(scale), tile_(tile),
          tile_h_(tile_h), h_slm_(h_slm) {}

    [[sycl::reqd_sub_group_size(kSg)]]
    void operator()(sycl::nd_item<1> it) const {
        const int lid = static_cast<int>(it.get_local_id(0));
        const int64_t row0 = static_cast<int64_t>(it.get_group(0)) * kTm;
        auto sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int chunk =
            static_cast<int>(sg.get_group_linear_id()) * kTn;

        if (sg.get_group_linear_id() == 0) {
            for (int j = 0; j < kTn; ++j) {
                h_slm_[lane * kTn + j] =
                    leaf(static_cast<float>(sylvester_pm1(lane, j)));
            }
        }
        for (int r = 0; r < kTm; ++r) {
            const int64_t row = row0 + r;
            const T v = (row < n_rows_) ? src_[row * N + lid] : T(0);
            tile_[r * N + lid] = static_cast<float>(v);
            tile_h_[r * N + lid] = static_cast<leaf>(v);
        }
        sycl::group_barrier(it.get_group());

        matrix::joint_matrix<sycl::sub_group, leaf, matrix::use::b, kTk, kTn,
                             matrix::layout::row_major>
            B;
        matrix::joint_matrix_load(
            sg, B, h_slm_.template get_multi_ptr<sycl::access::decorated::no>(),
            kTn);

        auto a_ptr =
            tile_h_.template get_multi_ptr<sycl::access::decorated::no>() +
            chunk;
        auto c_ptr =
            tile_.template get_multi_ptr<sycl::access::decorated::no>() + chunk;

        matrix::joint_matrix<sycl::sub_group, leaf, matrix::use::a, kTm, kTk,
                             matrix::layout::row_major>
            A;
        matrix::joint_matrix<sycl::sub_group, float, matrix::use::accumulator,
                             kTm, kTn>
            C;
        matrix::joint_matrix_load(sg, A, a_ptr, N);
        matrix::joint_matrix_fill(sg, C, 0.0f);
        matrix::joint_matrix_mad(sg, C, A, B, C);
        matrix::joint_matrix_store(sg, C, c_ptr, N, matrix::layout::row_major);
        sycl::group_barrier(it.get_group());

        for (int h = kTn; h < N; h *= 2) {
            const int stride = h * 2;
            if ((lid % stride) < h) {
                for (int r = 0; r < kTm; ++r) {
                    const float a = tile_[r * N + lid];
                    const float b = tile_[r * N + lid + h];
                    tile_[r * N + lid] = a + b;
                    tile_[r * N + lid + h] = a - b;
                }
            }
            sycl::group_barrier(it.get_group());
        }

        for (int r = 0; r < kTm; ++r) {
            const int64_t row = row0 + r;
            if (row >= n_rows_) continue;
            dst_[row * N + lid] = static_cast<T>(tile_[r * N + lid] * scale_);
        }
    }

private:
    const T *src_;
    T *dst_;
    int64_t n_rows_;
    float scale_;
    sycl::local_accessor<float, 1> tile_;
    sycl::local_accessor<leaf, 1> tile_h_;
    sycl::local_accessor<leaf, 1> h_slm_;
};

template <typename T, int N>
void launch_slm(sycl::queue &q, const T *src, T *dst, int64_t n_rows,
                float scale) {
    q.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(N), h);
        h.parallel_for(sycl::nd_range<1>({static_cast<size_t>(n_rows) * N}, {N}),
                       FwhtSlmKernel<T, N>(src, dst, n_rows, scale, slm));
    });
}

template <typename T, int N>
void launch_xmx(sycl::queue &q, const T *src, T *dst, int64_t n_rows,
                float scale) {
    const size_t groups =
        static_cast<size_t>((n_rows + kTm - 1) / kTm);
    q.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> tile(sycl::range<1>(kTm * N), h);
        sycl::local_accessor<leaf_t<T>, 1> tile_h(sycl::range<1>(kTm * N), h);
        sycl::local_accessor<leaf_t<T>, 1> h_slm(sycl::range<1>(kTk * kTn), h);
        h.parallel_for(sycl::nd_range<1>({groups * static_cast<size_t>(N)},
                                        {static_cast<size_t>(N)}),
                       FwhtXmxKernel<T, N>(src, dst, n_rows, scale, tile,
                                           tile_h, h_slm));
    });
}

template <typename T>
void launch_slm_n(sycl::queue &q, const T *src, T *dst, int n, int64_t n_rows,
                  float scale) {
    switch (n) {
    case 16: launch_slm<T, 16>(q, src, dst, n_rows, scale); return;
    case 32: launch_slm<T, 32>(q, src, dst, n_rows, scale); return;
    case 64: launch_slm<T, 64>(q, src, dst, n_rows, scale); return;
    case 128: launch_slm<T, 128>(q, src, dst, n_rows, scale); return;
    case 256: launch_slm<T, 256>(q, src, dst, n_rows, scale); return;
    default:
        throw std::invalid_argument("fwht: n must be 16/32/64/128/256, got " +
                                    std::to_string(n));
    }
}

template <typename T>
void launch_xmx_n(sycl::queue &q, const T *src, T *dst, int n, int64_t n_rows,
                  float scale) {
    switch (n) {
    case 16: launch_xmx<T, 16>(q, src, dst, n_rows, scale); return;
    case 32: launch_xmx<T, 32>(q, src, dst, n_rows, scale); return;
    case 64: launch_xmx<T, 64>(q, src, dst, n_rows, scale); return;
    case 128: launch_xmx<T, 128>(q, src, dst, n_rows, scale); return;
    case 256: launch_xmx<T, 256>(q, src, dst, n_rows, scale); return;
    default:
        throw std::invalid_argument("fwht: n must be 16/32/64/128/256, got " +
                                    std::to_string(n));
    }
}

// ---------------------------------------------------------------------------
// Tuned XMX kernel (Layer 2 autotuner). TN=TK=16 stay fixed: the DPAS B tile
// is a square H(16) atom and (16,16) is the only square fp16/bf16 tile B70
// advertises. TM/SG are template params (kernel-binary); wg_n/wg_m/pad/pf are
// runtime launch params. One WG covers wg_m row-tiles of TM rows with wg_n
// subgroups of SG lanes; subgroup s applies H(16) to chunks s, s+wg_n, ... .
// Staging/stores are flat strided loops so any wg_n*SG thread count works.
// prefetch=1 overlaps the next tile's global loads with the current tile's
// compute via a per-thread register buffer (bounded by kMaxPre).
template <typename T, int N, int TM, int SG>
class FwhtXmxTunedKernel {
public:
    using leaf = leaf_t<T>;
    static constexpr int kMaxPre = 8;

    FwhtXmxTunedKernel(const T *src, T *dst, int64_t n_rows, float scale,
                       int wg_n, int wg_m, int pad, int prefetch,
                       sycl::local_accessor<float, 1> tile,
                       sycl::local_accessor<leaf, 1> tile_h,
                       sycl::local_accessor<leaf, 1> h_slm)
        : src_(src), dst_(dst), n_rows_(n_rows), scale_(scale), wg_n_(wg_n),
          wg_m_(wg_m), ldm_(N + pad), prefetch_(prefetch), tile_(tile),
          tile_h_(tile_h), h_slm_(h_slm) {}

    [[sycl::reqd_sub_group_size(SG)]]
    void operator()(sycl::nd_item<1> it) const {
        const int lid = static_cast<int>(it.get_local_id(0));
        const int nthreads = wg_n_ * SG;
        auto sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int sid = static_cast<int>(sg.get_group_linear_id());
        const int64_t row0 = static_cast<int64_t>(it.get_group(0)) *
                             (TM * wg_m_);

        if (sid == 0 && lane < 16) {
            for (int j = 0; j < 16; ++j) {
                h_slm_[lane * 16 + j] =
                    leaf(static_cast<float>(sylvester_pm1(lane, j)));
            }
        }

        auto stage = [&](int t, const leaf *pre) {
            for (int e = lid, k = 0; e < TM * N; e += nthreads, ++k) {
                const int r = e / N, c = e % N;
                const int64_t row = row0 + t * TM + r;
                const leaf v = pre ? pre[k]
                                   : ((row < n_rows_) ? leaf(src_[row * N + c])
                                                      : leaf(0));
                tile_[r * ldm_ + c] = static_cast<float>(v);
                tile_h_[r * ldm_ + c] = v;
            }
        };

        stage(0, nullptr);
        sycl::group_barrier(it.get_group());

        matrix::joint_matrix<sycl::sub_group, leaf, matrix::use::b, 16, 16,
                             matrix::layout::row_major>
            B;
        matrix::joint_matrix_load(
            sg, B, h_slm_.template get_multi_ptr<sycl::access::decorated::no>(),
            16);

        leaf pre[kMaxPre];
        for (int t = 0; t < wg_m_; ++t) {
            if (prefetch_ && t + 1 < wg_m_) {
                for (int e = lid, k = 0; e < TM * N; e += nthreads, ++k) {
                    const int r = e / N, c = e % N;
                    const int64_t row = row0 + (t + 1) * TM + r;
                    pre[k] = (row < n_rows_) ? leaf(src_[row * N + c])
                                             : leaf(0);
                }
            }

            for (int chunk = sid * 16; chunk < N; chunk += wg_n_ * 16) {
                auto a_ptr =
                    tile_h_.template get_multi_ptr<sycl::access::decorated::no>() +
                    chunk;
                auto c_ptr =
                    tile_.template get_multi_ptr<sycl::access::decorated::no>() +
                    chunk;
                matrix::joint_matrix<sycl::sub_group, leaf, matrix::use::a, TM,
                                     16, matrix::layout::row_major>
                    A;
                matrix::joint_matrix<sycl::sub_group, float,
                                     matrix::use::accumulator, TM, 16>
                    C;
                matrix::joint_matrix_load(sg, A, a_ptr, ldm_);
                matrix::joint_matrix_fill(sg, C, 0.0f);
                matrix::joint_matrix_mad(sg, C, A, B, C);
                matrix::joint_matrix_store(sg, C, c_ptr, ldm_,
                                           matrix::layout::row_major);
            }
            sycl::group_barrier(it.get_group());

            for (int h = 16; h < N; h *= 2) {
                const int stride = h * 2;
                for (int col = lid; col < N; col += nthreads) {
                    if ((col % stride) < h) {
                        for (int r = 0; r < TM; ++r) {
                            float *rowp = &tile_[r * ldm_];
                            const float a = rowp[col];
                            const float b = rowp[col + h];
                            rowp[col] = a + b;
                            rowp[col + h] = a - b;
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
            }

            for (int e = lid; e < TM * N; e += nthreads) {
                const int r = e / N, c = e % N;
                const int64_t row = row0 + t * TM + r;
                if (row < n_rows_) {
                    dst_[row * N + c] =
                        static_cast<T>(tile_[r * ldm_ + c] * scale_);
                }
            }
            sycl::group_barrier(it.get_group());

            if (t + 1 < wg_m_) {
                stage(t + 1, prefetch_ ? pre : nullptr);
                sycl::group_barrier(it.get_group());
            }
        }
    }

private:
    const T *src_;
    T *dst_;
    int64_t n_rows_;
    float scale_;
    int wg_n_;
    int wg_m_;
    int ldm_;
    int prefetch_;
    sycl::local_accessor<float, 1> tile_;
    sycl::local_accessor<leaf, 1> tile_h_;
    sycl::local_accessor<leaf, 1> h_slm_;
};

template <typename T, int N, int TM, int SG>
void launch_xmx_tuned(sycl::queue &q, const T *src, T *dst, int64_t n_rows,
                      float scale, const FwhtXmxConfig &cfg) {
    const int wg_n = cfg.wg_n > 0 ? cfg.wg_n : N / 16;
    const size_t local = static_cast<size_t>(wg_n) * SG;
    const size_t tile_elems = static_cast<size_t>(TM) * (N + cfg.slm_pad);
    const size_t groups = static_cast<size_t>(
        (n_rows + static_cast<int64_t>(TM) * cfg.wg_m - 1) /
        (static_cast<int64_t>(TM) * cfg.wg_m));
    q.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> tile(sycl::range<1>(tile_elems), h);
        sycl::local_accessor<leaf_t<T>, 1> tile_h(sycl::range<1>(tile_elems),
                                                  h);
        sycl::local_accessor<leaf_t<T>, 1> h_slm(sycl::range<1>(16 * 16), h);
        h.parallel_for(sycl::nd_range<1>({groups * local}, {local}),
                       FwhtXmxTunedKernel<T, N, TM, SG>(
                           src, dst, n_rows, scale, wg_n, cfg.wg_m,
                           cfg.slm_pad, cfg.prefetch, tile, tile_h, h_slm));
    });
}

template <typename T, int N>
void launch_xmx_tuned_ts(sycl::queue &q, const T *src, T *dst, int64_t n_rows,
                         float scale, const FwhtXmxConfig &cfg) {
    const int key = cfg.tm * 64 + cfg.sg;
    switch (key) {
    case 1 * 64 + 16: launch_xmx_tuned<T, N, 1, 16>(q, src, dst, n_rows, scale, cfg); return;
    case 2 * 64 + 16: launch_xmx_tuned<T, N, 2, 16>(q, src, dst, n_rows, scale, cfg); return;
    case 4 * 64 + 16: launch_xmx_tuned<T, N, 4, 16>(q, src, dst, n_rows, scale, cfg); return;
    case 8 * 64 + 16: launch_xmx_tuned<T, N, 8, 16>(q, src, dst, n_rows, scale, cfg); return;
    case 1 * 64 + 32: launch_xmx_tuned<T, N, 1, 32>(q, src, dst, n_rows, scale, cfg); return;
    case 2 * 64 + 32: launch_xmx_tuned<T, N, 2, 32>(q, src, dst, n_rows, scale, cfg); return;
    case 4 * 64 + 32: launch_xmx_tuned<T, N, 4, 32>(q, src, dst, n_rows, scale, cfg); return;
    case 8 * 64 + 32: launch_xmx_tuned<T, N, 8, 32>(q, src, dst, n_rows, scale, cfg); return;
    default: throw std::invalid_argument("fwht tuned: bad tm/sg combo");
    }
}

template <typename T>
void launch_xmx_tuned_n(sycl::queue &q, const T *src, T *dst, int n,
                        int64_t n_rows, float scale,
                        const FwhtXmxConfig &cfg) {
    switch (n) {
    case 16: launch_xmx_tuned_ts<T, 16>(q, src, dst, n_rows, scale, cfg); return;
    case 32: launch_xmx_tuned_ts<T, 32>(q, src, dst, n_rows, scale, cfg); return;
    case 64: launch_xmx_tuned_ts<T, 64>(q, src, dst, n_rows, scale, cfg); return;
    case 128: launch_xmx_tuned_ts<T, 128>(q, src, dst, n_rows, scale, cfg); return;
    case 256: launch_xmx_tuned_ts<T, 256>(q, src, dst, n_rows, scale, cfg); return;
    default:
        throw std::invalid_argument("fwht: n must be 16/32/64/128/256, got " +
                                    std::to_string(n));
    }
}

} // namespace

const char *fwht_impl_name(FwhtImpl impl) {
    switch (impl) {
    case FwhtImpl::Auto: return "auto";
    case FwhtImpl::Slm: return "slm";
    case FwhtImpl::Xmx: return "xmx";
    case FwhtImpl::Esimd: return "esimd";
    }
    return "?";
}

bool fwht_xmx_ok(const sycl::device &dev) {
    if (!dev.has(sycl::aspect::ext_intel_matrix)) return false;
    try {
        using namespace sycl::ext::oneapi::experimental;
        auto combos = dev.get_info<info::device::matrix_combinations>();
        for (const auto &c : combos) {
            const bool n16 = c.nsize == 16 && c.ksize == 16;
            const bool m8 = c.msize == 8 || c.max_msize == 8 ||
                            (c.msize == 0 && c.max_msize >= 8);
            if (n16 && m8) return true;
        }
    } catch (...) {
    }
    return false;
}

// Row-count regime for the tuner: decode calls are small (~KV heads + a few
// tokens), prefill is large (heads × context). Splitting the default by rows
// recovers 20-90% at decode and 2-17% at prefill vs the old fixed table (tuned
// only at rows=4096). Threshold: < 1024 rows is decode-ish, else prefill.
bool fwht_rows_small(int64_t rows) { return rows < 1024; }

FwhtXmxConfig fwht_default_xmx_config(FwhtElem elem, int n, int64_t rows) {
    // Full-lattice scans on BMG-G31, 2026-08-20 (rows=28 decode, rows=2048
    // prefill) + the original rows=4096 lattice. Two regimes per (elem, width);
    // the rows=28 picks favor wide work-groups (wg_n 4-16, wg_m=1) because a
    // few rows can't fill the device with narrow WGs; at 2048 the device has
    // rows to spare, so moderate wg_n with wg_m=1 wins. Every pick is present
    // and legal in the scans for both row counts of its elem.
    const bool small = fwht_rows_small(rows);
    switch (elem) {
    case FwhtElem::F16:
        switch (n) {
        case 16: return small ? FwhtXmxConfig{4, 32, 1, 1, 8, 1}  // 8.9us
                              : FwhtXmxConfig{4, 32, 1, 1, 0, 1}; // 8.4us
        case 32: return small ? FwhtXmxConfig{2, 32, 2, 2, 0, 1}  // 8.5us
                              : FwhtXmxConfig{4, 16, 2, 1, 8, 0}; // 8.4us
        case 64: return small ? FwhtXmxConfig{2, 32, 4, 1, 0, 0}  // 8.3us
                              : FwhtXmxConfig{4, 32, 2, 1, 0, 0}; // 10.1us
        case 128: return small ? FwhtXmxConfig{4, 32, 8, 1, 0, 0}  // 8.3us
                               : FwhtXmxConfig{4, 32, 2, 1, 0, 1}; // 11.9us
        case 256: return small ? FwhtXmxConfig{2, 32, 8, 1, 0, 1}  // 8.3us
                               : FwhtXmxConfig{2, 32, 2, 1, 0, 0}; // 15.4us
        default: break;
        }
        break;
    case FwhtElem::BF16:
        switch (n) {
        case 16: return small ? FwhtXmxConfig{1, 16, 1, 1, 8, 1}  // 8.9us
                              : FwhtXmxConfig{1, 32, 1, 2, 0, 0}; // 11.2us
        case 32: return small ? FwhtXmxConfig{2, 32, 2, 1, 0, 0}  // 8.4us
                              : FwhtXmxConfig{2, 32, 1, 1, 0, 0}; // 8.5us
        case 64: return small ? FwhtXmxConfig{1, 16, 4, 1, 0, 0}  // 8.4us
                              : FwhtXmxConfig{4, 16, 4, 1, 8, 1}; // 10.2us
        case 128: return small ? FwhtXmxConfig{2, 16, 8, 1, 0, 1}  // 10.1us
                               : FwhtXmxConfig{4, 32, 4, 1, 8, 0}; // 12.0us
        case 256: return small ? FwhtXmxConfig{2, 32, 16, 1, 8, 1}  // 8.3us
                               : FwhtXmxConfig{8, 32, 8, 1, 0, 0}; // 15.4us
        default: break;
        }
        break;
    default: break;
    }
    return {8, 16, 0, 1, 0, 0}; // legacy hardcoded config (wg_n=0 → n/16)
}

bool fwht_xmx_config_ok(const sycl::device &dev, const FwhtXmxConfig &cfg,
                        int n) {
    if (!fwht_supported(n) || !fwht_xmx_ok(dev)) return false;
    if (cfg.tm != 1 && cfg.tm != 2 && cfg.tm != 4 && cfg.tm != 8) return false;
    if (cfg.sg != 16 && cfg.sg != 32) return false;
    bool sg_ok = false;
    for (const auto s : dev.get_info<sycl::info::device::sub_group_sizes>()) {
        if (static_cast<int>(s) == cfg.sg) sg_ok = true;
    }
    if (!sg_ok) return false;
    const int wg_n = cfg.wg_n > 0 ? cfg.wg_n : n / 16;
    if (wg_n < 1 || wg_n * 16 > n) return false;
    if (cfg.wg_m < 1) return false;
    // DPAS ldm must stay 16B-aligned: 8 leaf elems (or 4 fp32). Wipe pads
    // that break alignment — they JIT fine but compute garbage.
    if (cfg.slm_pad < 0 || cfg.slm_pad > 16 || cfg.slm_pad % 8 != 0)
        return false;
    if (cfg.prefetch != 0 && cfg.prefetch != 1) return false;
    // Measured on BMG-G31 (full-lattice scan 2026-08-20): tm=1 with sg=32
    // and more than one subgroup per WG launches but computes garbage
    // (m=1 fragment layout across >1 subgroups). Wipe it.
    if (cfg.tm == 1 && cfg.sg == 32 && wg_n > 1) return false;
    const size_t local = static_cast<size_t>(wg_n) * cfg.sg;
    if (local > dev.get_info<sycl::info::device::max_work_group_size>())
        return false;
    // prefetch register buffer bound (FwhtXmxTunedKernel::kMaxPre)
    if (cfg.prefetch &&
        (cfg.tm * n + static_cast<int>(local) - 1) / static_cast<int>(local) >
            8)
        return false;
    // SLM budget: fp32 tile + leaf tile + H(16) atom (leaf is 2 bytes)
    const size_t slm =
        static_cast<size_t>(cfg.tm) * (n + cfg.slm_pad) * (4 + 2) + 256 * 2;
    if (slm > dev.get_info<sycl::info::device::local_mem_size>()) return false;
    return true;
}

bool fwht_device_xmx_tuned(sycl::queue &q, const void *src, void *dst, int n,
                           int64_t n_rows, float scale, FwhtElem elem,
                           const FwhtXmxConfig &cfg, bool wait) {
    if (!fwht_xmx_config_ok(q.get_device(), cfg, n)) return false;
    if (n_rows <= 0) return true;
    switch (elem) {
    case FwhtElem::F16:
        launch_xmx_tuned_n<sycl::half>(q, static_cast<const sycl::half *>(src),
                                       static_cast<sycl::half *>(dst), n,
                                       n_rows, scale, cfg);
        break;
    case FwhtElem::BF16:
        launch_xmx_tuned_n<sycl::ext::oneapi::bfloat16>(
            q, static_cast<const sycl::ext::oneapi::bfloat16 *>(src),
            static_cast<sycl::ext::oneapi::bfloat16 *>(dst), n, n_rows, scale,
            cfg);
        break;
    default:
        return false; // F32 keeps the existing hardcoded XMX path
    }
    if (wait) q.wait_and_throw();
    return true;
}

void fwht_host(float *rows, int n, int64_t n_rows, float scale) {
    if (!fwht_supported(n)) {
        throw std::invalid_argument("fwht_host: unsupported n");
    }
    for (int64_t r = 0; r < n_rows; ++r) {
        fwht_host_row(rows + r * n, n, scale);
    }
}

void fwht_device_typed(sycl::queue &q, const void *src, void *dst, int n,
                       int64_t n_rows, float scale, FwhtElem elem,
                       FwhtImpl impl, bool wait) {
    if (!fwht_supported(n)) {
        throw std::invalid_argument(
            "fwht_device: n must be 16/32/64/128/256, got " + std::to_string(n));
    }
    if (n_rows <= 0) {
        if (wait) q.wait_and_throw();
        return;
    }
    if (impl == FwhtImpl::Esimd) {
        if (elem == FwhtElem::F32)
            throw std::invalid_argument(
                "fwht esimd impl serves half-precision buffers only");
        if (!fwht_device_esimd_half(q, src, dst, n, n_rows, scale, elem,
                                    wait))
            throw std::invalid_argument(
                "fwht esimd impl needs aligned buffers and n in "
                "{16,32,64,128,256}");
        return;
    }
    FwhtImpl use = impl;
    if (use == FwhtImpl::Auto) {
        // GRF-resident ESIMD path: half-precision, wins from prefill rows up
        // (bench-fwht-esimd, B70 2026-08-22: bf16 2-3.8x XMX at rows 2k-37k;
        // fp16 same family, gated "bit-exact OR subnormal-store-only").
        // Exceptions: tiny widths (n=16 launch-bound), and n=128 past ~16k
        // rows where the XMX tile path re-takes the lead.
        if ((elem == FwhtElem::BF16 || elem == FwhtElem::F16) && n >= 32 &&
            n_rows >= 1024 && (n >= 256 || n_rows <= 16384) &&
            fwht_esimd_ok(src, dst, n)) {
            use = FwhtImpl::Esimd;
        } else {
            use = fwht_xmx_ok(q.get_device()) ? FwhtImpl::Xmx : FwhtImpl::Slm;
        }
    }
    if (use == FwhtImpl::Esimd) {
        if (fwht_device_esimd_half(q, src, dst, n, n_rows, scale, elem, wait))
            return;
        use = fwht_xmx_ok(q.get_device()) ? FwhtImpl::Xmx : FwhtImpl::Slm;
    }
    if (use == FwhtImpl::Xmx && !fwht_xmx_ok(q.get_device())) {
        throw std::runtime_error(
            "fwht XMX requested but no 8x16x16 DPAS tile is advertised");
    }
    switch (elem) {
    case FwhtElem::F32:
        if (use == FwhtImpl::Xmx)
            launch_xmx_n<float>(q, static_cast<const float *>(src),
                                static_cast<float *>(dst), n, n_rows, scale);
        else
            launch_slm_n<float>(q, static_cast<const float *>(src),
                                static_cast<float *>(dst), n, n_rows, scale);
        break;
    case FwhtElem::F16:
        if (use == FwhtImpl::Xmx) {
            const FwhtXmxConfig cfg = fwht_default_xmx_config(elem, n, n_rows);
            if (fwht_xmx_config_ok(q.get_device(), cfg, n)) {
                fwht_device_xmx_tuned(q, src, dst, n, n_rows, scale, elem, cfg,
                                      wait);
            } else { // winner not legal on this device — legacy hardcoded
                launch_xmx_n<sycl::half>(
                    q, static_cast<const sycl::half *>(src),
                    static_cast<sycl::half *>(dst), n, n_rows, scale);
            }
        } else {
            launch_slm_n<sycl::half>(q, static_cast<const sycl::half *>(src),
                                     static_cast<sycl::half *>(dst), n, n_rows,
                                     scale);
        }
        break;
    case FwhtElem::BF16:
        if (use == FwhtImpl::Xmx) {
            const FwhtXmxConfig cfg = fwht_default_xmx_config(elem, n, n_rows);
            if (fwht_xmx_config_ok(q.get_device(), cfg, n)) {
                fwht_device_xmx_tuned(q, src, dst, n, n_rows, scale, elem, cfg,
                                      wait);
            } else { // winner not legal on this device — legacy hardcoded
                launch_xmx_n<sycl::ext::oneapi::bfloat16>(
                    q, static_cast<const sycl::ext::oneapi::bfloat16 *>(src),
                    static_cast<sycl::ext::oneapi::bfloat16 *>(dst), n, n_rows,
                    scale);
            }
        } else {
            launch_slm_n<sycl::ext::oneapi::bfloat16>(
                q, static_cast<const sycl::ext::oneapi::bfloat16 *>(src),
                static_cast<sycl::ext::oneapi::bfloat16 *>(dst), n, n_rows,
                scale);
        }
        break;
    }
    if (wait) q.wait_and_throw();
}

void fwht_device(sycl::queue &q, const float *src, float *dst, int n,
                 int64_t n_rows, float scale, FwhtImpl impl, bool wait) {
    fwht_device_typed(q, src, dst, n, n_rows, scale, FwhtElem::F32, impl, wait);
}

} // namespace afi
