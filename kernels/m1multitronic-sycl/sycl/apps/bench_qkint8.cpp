// afi-bench-qkint8 — Phase 0 INT8 DPAS tile probe
// (docs/fused-dpas-int8-plan.md). Standalone go/no-go measurement:
//
//   (a) prints the live B70's matrix_combinations for int8 and fp16,
//   (b) microbenches joint_matrix<int8_t> stage+load+MAD+store vs the same
//       shape in fp16 DPAS at K=32 (one DPAS K-tile) and K=128 (head dim),
//       including the packed-int8 SLM staging cost, in three variants:
//         fp16  — fp16 SLM staging, fp16 MAD, fp32 store (baseline)
//         int8f — fp16 in, per-token absmax quant in-kernel, packed int8
//                 SLM staging, s8 MAD, s32 store (the Phase 1 kernel shape)
//         int8p — pre-quantized/packed int8 in, byte-copy SLM staging,
//                 s8 MAD, s32 store (isolates DPAS from the quant cost)
//   (c) gates the s32 accumulate against a host s32 reference (exact integer
//       match required) and auto-detects the int8 B-operand memory layout
//       the load lowering expects (col_major plain vs ext_intel_packed
//       VNNI4 vs row_major).
//
//   afi-bench-qkint8 [-d dev] [--rows 16384] [--iters 50]
//                    [--sg 16 [--sg 32]] [--bmode auto|colmajor|packed|packed16|rowmajor]
//
// CSV: kind,k,sg,rows,status,ms,tops,max_diff
//   kind: fp16 | int8f | int8p ; status: ok | wrong
//   max_diff: |device s32 - host s32| on the first two row blocks.

#include "afi/device.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace matrix = sycl::ext::oneapi::experimental::matrix;

namespace {

constexpr int kTileN = 16;   // DPAS N, B70 int8 + fp16
constexpr int kDpasK8 = 32;  // int8 DPAS accumulate depth
constexpr int kDpasK16 = 16; // fp16 DPAS accumulate depth
constexpr int kTm = 8;       // M per tile (max M for both formats)

// B-operand staging modes for int8. The 8-bit B tile DPAS consumes is
// VNNI4-packed in registers; which memory layout joint_matrix_load expects
// is implementation-defined, so the check phase probes each candidate.
enum BMode { BM_COLMAJOR = 0, BM_PACKED = 1, BM_PACKED_S16 = 2, BM_ROWMAJOR = 3 };

const char *bmode_name(int m) {
    switch (m) {
    case BM_COLMAJOR: return "colmajor";
    case BM_PACKED: return "packed";       // ext_intel_packed, stride 4*N
    case BM_PACKED_S16: return "packed16"; // ext_intel_packed, stride N
    default: return "rowmajor";
    }
}

// Host/device-shared staging: logical int8 B (K x N, b[k*N+n]) -> mode layout.
size_t bstage_index(int mode, int k, int n, int K) {
    switch (mode) {
    case BM_COLMAJOR: return static_cast<size_t>(n) * K + k;
    case BM_PACKED:
    case BM_PACKED_S16:
        return (static_cast<size_t>(k / 4) * kTileN + n) * 4 + (k % 4);
    default: return static_cast<size_t>(k) * kTileN + n;
    }
}

int bstage_stride(int mode, int K) {
    switch (mode) {
    case BM_COLMAJOR: return K;
    case BM_PACKED: return 4 * kTileN;
    case BM_PACKED_S16: return kTileN;
    default: return kTileN;
    }
}

template <int BMODE> struct BMat;
template <> struct BMat<BM_COLMAJOR> {
    using type = matrix::joint_matrix<sycl::sub_group, int8_t, matrix::use::b,
                                      kDpasK8, kTileN, matrix::layout::col_major>;
};
template <> struct BMat<BM_PACKED_S16> {
    using type = matrix::joint_matrix<sycl::sub_group, int8_t, matrix::use::b,
                                      kDpasK8, kTileN,
                                      matrix::layout::ext_intel_packed>;
};
template <> struct BMat<BM_PACKED> : BMat<BM_PACKED_S16> {};
template <> struct BMat<BM_ROWMAJOR> {
    using type = matrix::joint_matrix<sycl::sub_group, int8_t, matrix::use::b,
                                      kDpasK8, kTileN, matrix::layout::row_major>;
};

using Q8Ptr = sycl::multi_ptr<int8_t, sycl::access::address_space::local_space,
                              sycl::access::decorated::no>;

template <typename T>
auto gptr(T *p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space,
                                    sycl::access::decorated::no>(p);
}

// One s8 K-chunk MAD: A TMx32 row-major + B per mode -> s32 accumulate.
template <int BMODE, typename APtrT>
inline void mad_chunk(sycl::sub_group sg,
                      matrix::joint_matrix<sycl::sub_group, int32_t,
                                           matrix::use::accumulator, kTm,
                                           kTileN> &C,
                      APtrT a_ptr, Q8Ptr b_ptr, int dk, int chunk) {
    matrix::joint_matrix<sycl::sub_group, int8_t, matrix::use::a, kTm,
                         kDpasK8, matrix::layout::row_major>
        A;
    typename BMat<BMODE>::type B;
    matrix::joint_matrix_load(sg, A, a_ptr + chunk, dk);
    matrix::joint_matrix_load(sg, B, b_ptr + bstage_index(BMODE, chunk, 0, dk),
                              bstage_stride(BMODE, dk));
    matrix::joint_matrix_mad(sg, C, A, B, C);
}

// ---------------------------------------------------------------------------
// (c) correctness kernel: one WG, one 8x32x16 s8 MAD straight from global.
template <int SG, int BMODE>
class Int8DpasCheckKernel {
public:
    Int8DpasCheckKernel(const int8_t *a, const int8_t *b, int32_t *c)
        : a_(a), b_(b), c_(c) {}

    [[sycl::reqd_sub_group_size(SG)]]
    void operator()(sycl::nd_item<1> it) const {
        auto sg = it.get_sub_group();
        matrix::joint_matrix<sycl::sub_group, int8_t, matrix::use::a, kTm,
                             kDpasK8, matrix::layout::row_major>
            A;
        typename BMat<BMODE>::type B;
        matrix::joint_matrix<sycl::sub_group, int32_t,
                             matrix::use::accumulator, kTm, kTileN>
            C;
        matrix::joint_matrix_load(sg, A, gptr(a_), kDpasK8);
        matrix::joint_matrix_load(sg, B, gptr(b_),
                                  bstage_stride(BMODE, kDpasK8));
        matrix::joint_matrix_fill(sg, C, 0);
        matrix::joint_matrix_mad(sg, C, A, B, C);
        matrix::joint_matrix_store(sg, C, gptr(c_), kTileN,
                                   matrix::layout::row_major);
    }

private:
    const int8_t *a_;
    const int8_t *b_;
    int32_t *c_;
};

// ---------------------------------------------------------------------------
// fp16 baseline: stage the fp16 K-tile to SLM once per WG, then per Q
// row-block (wg_m of them): stage fp16 Q, fp16 MAD (K/16 chunks), fp32 store.
template <int SG, int DK>
class Fp16DpasBenchKernel {
public:
    Fp16DpasBenchKernel(const sycl::half *q, const sycl::half *k, float *o,
                        int wg_m,
                        sycl::local_accessor<sycl::half, 1> qs,
                        sycl::local_accessor<sycl::half, 1> ks)
        : q_(q), k_(k), o_(o), wg_m_(wg_m), qs_(qs), ks_(ks) {}

    [[sycl::reqd_sub_group_size(SG)]]
    void operator()(sycl::nd_item<1> it) const {
        const int lid = static_cast<int>(it.get_local_id(0));
        const int64_t row0 =
            static_cast<int64_t>(it.get_group(0)) * (kTm * wg_m_);
        for (int e = lid; e < DK * kTileN; e += SG)
            ks_[e] = k_[e];
        sycl::group_barrier(it.get_group());

        auto sg = it.get_sub_group();
        auto qs_ptr = qs_.template get_multi_ptr<sycl::access::decorated::no>();
        auto ks_ptr = ks_.template get_multi_ptr<sycl::access::decorated::no>();
        for (int t = 0; t < wg_m_; ++t) {
            const int64_t rb = row0 + t * kTm;
            for (int e = lid; e < kTm * DK; e += SG)
                qs_[e] = q_[rb * DK + e];
            sycl::group_barrier(it.get_group());
            matrix::joint_matrix<sycl::sub_group, float,
                                 matrix::use::accumulator, kTm, kTileN>
                C;
            matrix::joint_matrix_fill(sg, C, 0.0f);
            for (int chunk = 0; chunk < DK; chunk += kDpasK16) {
                matrix::joint_matrix<sycl::sub_group, sycl::half,
                                     matrix::use::a, kTm, kDpasK16,
                                     matrix::layout::row_major>
                    A;
                matrix::joint_matrix<sycl::sub_group, sycl::half,
                                     matrix::use::b, kDpasK16, kTileN,
                                     matrix::layout::row_major>
                    B;
                matrix::joint_matrix_load(sg, A, qs_ptr + chunk, DK);
                matrix::joint_matrix_load(sg, B, ks_ptr + chunk * kTileN,
                                          kTileN);
                matrix::joint_matrix_mad(sg, C, A, B, C);
            }
            matrix::joint_matrix_store(sg, C, gptr(o_ + rb * kTileN), kTileN,
                                       matrix::layout::row_major);
            sycl::group_barrier(it.get_group());
        }
    }

private:
    const sycl::half *q_;
    const sycl::half *k_;
    float *o_;
    int wg_m_;
    sycl::local_accessor<sycl::half, 1> qs_;
    sycl::local_accessor<sycl::half, 1> ks_;
};

// ---------------------------------------------------------------------------
// int8 fused: stage the fp16 K-tile once per WG, absmax + quant + packed
// staging for K once (reused across the wg_m Q row-blocks); per Q row-block:
// stage fp16 Q, per-token absmax, quant (multiply by the precomputed
// 127/absmax reciprocal, Georganas pattern), s8 MAD (K/32 chunks), s32 store.
// This is the Phase 1 kernel's inner shape.
template <int SG, int DK, int BMODE>
class Int8FusedDpasBenchKernel {
public:
    Int8FusedDpasBenchKernel(const sycl::half *q, const sycl::half *k,
                             int32_t *o, int wg_m,
                             sycl::local_accessor<sycl::half, 1> qs,
                             sycl::local_accessor<sycl::half, 1> ks,
                             sycl::local_accessor<int8_t, 1> q8,
                             sycl::local_accessor<int8_t, 1> k8,
                             sycl::local_accessor<float, 1> scales)
        : q_(q), k_(k), o_(o), wg_m_(wg_m), qs_(qs), ks_(ks), q8_(q8),
          k8_(k8), scales_(scales) {}

    [[sycl::reqd_sub_group_size(SG)]]
    void operator()(sycl::nd_item<1> it) const {
        const int lid = static_cast<int>(it.get_local_id(0));
        const int64_t row0 =
            static_cast<int64_t>(it.get_group(0)) * (kTm * wg_m_);
        float *q_inv = scales_
                           .template get_multi_ptr<sycl::access::decorated::no>()
                           .get();                     // [kTm]
        float *k_inv = q_inv + kTm;                    // [kTileN]

        for (int e = lid; e < DK * kTileN; e += SG)
            ks_[e] = k_[e];
        sycl::group_barrier(it.get_group());
        auto sg = it.get_sub_group();
        // Cooperative absmax: lane-strided partial + subgroup shuffle
        // reduction (the Phase 1 pattern), not a serial per-row loop.
        for (int n = 0; n < kTileN; ++n) {
            float m = 0.0f;
            for (int c = lid; c < DK; c += SG)
                m = sycl::fmax(
                    m, sycl::fabs(static_cast<float>(ks_[c * kTileN + n])));
            m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
            if (lid == 0) k_inv[n] = 127.0f / (m > 0.0f ? m : 1.0f);
        }
        sycl::group_barrier(it.get_group());
        for (int e = lid; e < DK * kTileN; e += SG) {
            const int c = e / kTileN, n = e % kTileN;
            k8_[bstage_index(BMODE, c, n, DK)] = static_cast<int8_t>(
                sycl::round(static_cast<float>(ks_[e]) * k_inv[n]));
        }

        auto q8_ptr = q8_.template get_multi_ptr<sycl::access::decorated::no>();
        auto k8_ptr = k8_.template get_multi_ptr<sycl::access::decorated::no>();
        for (int t = 0; t < wg_m_; ++t) {
            const int64_t rb = row0 + t * kTm;
            sycl::group_barrier(it.get_group());
            for (int e = lid; e < kTm * DK; e += SG)
                qs_[e] = q_[rb * DK + e];
            sycl::group_barrier(it.get_group());
            for (int r = 0; r < kTm; ++r) {
                float m = 0.0f;
                for (int c = lid; c < DK; c += SG)
                    m = sycl::fmax(
                        m, sycl::fabs(static_cast<float>(qs_[r * DK + c])));
                m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
                if (lid == 0) q_inv[r] = 127.0f / (m > 0.0f ? m : 1.0f);
            }
            sycl::group_barrier(it.get_group());
            for (int e = lid; e < kTm * DK; e += SG) {
                const int r = e / DK;
                q8_[e] = static_cast<int8_t>(
                    sycl::round(static_cast<float>(qs_[e]) * q_inv[r]));
            }
            sycl::group_barrier(it.get_group());

            matrix::joint_matrix<sycl::sub_group, int32_t,
                                 matrix::use::accumulator, kTm, kTileN>
                C;
            matrix::joint_matrix_fill(sg, C, 0);
            for (int chunk = 0; chunk < DK; chunk += kDpasK8)
                mad_chunk<BMODE>(sg, C, q8_ptr, k8_ptr, DK, chunk);
            matrix::joint_matrix_store(sg, C, gptr(o_ + rb * kTileN), kTileN,
                                       matrix::layout::row_major);
        }
    }

private:
    const sycl::half *q_;
    const sycl::half *k_;
    int32_t *o_;
    int wg_m_;
    sycl::local_accessor<sycl::half, 1> qs_;
    sycl::local_accessor<sycl::half, 1> ks_;
    sycl::local_accessor<int8_t, 1> q8_;
    sycl::local_accessor<int8_t, 1> k8_;
    sycl::local_accessor<float, 1> scales_;
};

// ---------------------------------------------------------------------------
// int8 prepacked: byte-copy SLM staging of already-quantized data (K-tile
// staged once, reused across wg_m Q row-blocks), s8 MAD, s32 store.
// Isolates achievable DPAS throughput from the quant cost.
template <int SG, int DK, int BMODE>
class Int8PrepackedDpasBenchKernel {
public:
    Int8PrepackedDpasBenchKernel(const int8_t *q8, const int8_t *k8, int32_t *o,
                                 int wg_m,
                                 sycl::local_accessor<int8_t, 1> q8s,
                                 sycl::local_accessor<int8_t, 1> k8s)
        : q_(q8), k_(k8), o_(o), wg_m_(wg_m), q8s_(q8s), k8s_(k8s) {}

    [[sycl::reqd_sub_group_size(SG)]]
    void operator()(sycl::nd_item<1> it) const {
        const int lid = static_cast<int>(it.get_local_id(0));
        const int64_t row0 =
            static_cast<int64_t>(it.get_group(0)) * (kTm * wg_m_);
        for (int e = lid; e < DK * kTileN; e += SG)
            k8s_[e] = k_[e];
        sycl::group_barrier(it.get_group());

        auto sg = it.get_sub_group();
        auto q8_ptr = q8s_.template get_multi_ptr<sycl::access::decorated::no>();
        auto k8_ptr = k8s_.template get_multi_ptr<sycl::access::decorated::no>();
        for (int t = 0; t < wg_m_; ++t) {
            const int64_t rb = row0 + t * kTm;
            for (int e = lid; e < kTm * DK; e += SG)
                q8s_[e] = q_[rb * DK + e];
            sycl::group_barrier(it.get_group());
            matrix::joint_matrix<sycl::sub_group, int32_t,
                                 matrix::use::accumulator, kTm, kTileN>
                C;
            matrix::joint_matrix_fill(sg, C, 0);
            for (int chunk = 0; chunk < DK; chunk += kDpasK8)
                mad_chunk<BMODE>(sg, C, q8_ptr, k8_ptr, DK, chunk);
            matrix::joint_matrix_store(sg, C, gptr(o_ + rb * kTileN), kTileN,
                                       matrix::layout::row_major);
            sycl::group_barrier(it.get_group());
        }
    }

private:
    const int8_t *q_;
    const int8_t *k_;
    int32_t *o_;
    int wg_m_;
    sycl::local_accessor<int8_t, 1> q8s_;
    sycl::local_accessor<int8_t, 1> k8s_;
};

// ---------------------------------------------------------------------------
// Host reference: exact s32 dot products over dk.
void host_s32(const int8_t *q8, const int8_t *b_log, int64_t rows, int dk,
              int32_t *ref) {
    for (int64_t r = 0; r < rows; ++r)
        for (int n = 0; n < kTileN; ++n) {
            int32_t acc = 0;
            for (int c = 0; c < dk; ++c)
                acc += static_cast<int32_t>(q8[r * dk + c]) *
                       static_cast<int32_t>(b_log[c * kTileN + n]);
            ref[r * kTileN + n] = acc;
        }
}

struct CheckResult {
    bool ok = false;
    int max_diff = -1;
};

template <int SG, int BMODE>
CheckResult run_check(sycl::queue &q) {
    CheckResult res;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dist(-127, 127);
    std::vector<int8_t> h_a(kTm * kDpasK8), h_blog(kDpasK8 * kTileN),
        h_bs(kDpasK8 * kTileN);
    for (auto &v : h_a) v = static_cast<int8_t>(dist(rng));
    for (auto &v : h_blog) v = static_cast<int8_t>(dist(rng));
    for (int c = 0; c < kDpasK8; ++c)
        for (int n = 0; n < kTileN; ++n)
            h_bs[bstage_index(BMODE, c, n, kDpasK8)] = h_blog[c * kTileN + n];

    int8_t *a = sycl::malloc_device<int8_t>(h_a.size(), q);
    int8_t *b = sycl::malloc_device<int8_t>(h_bs.size(), q);
    int32_t *c = sycl::malloc_device<int32_t>(kTm * kTileN, q);
    q.memcpy(a, h_a.data(), h_a.size()).wait();
    q.memcpy(b, h_bs.data(), h_bs.size()).wait();
    q.parallel_for(sycl::nd_range<1>({SG}, {SG}),
                   Int8DpasCheckKernel<SG, BMODE>(a, b, c));
    std::vector<int32_t> h_c(kTm * kTileN);
    q.memcpy(h_c.data(), c, h_c.size() * sizeof(int32_t)).wait();
    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);

    std::vector<int32_t> ref(kTm * kTileN);
    host_s32(h_a.data(), h_blog.data(), kTm, kDpasK8, ref.data());
    res.ok = true;
    res.max_diff = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const int d = std::abs(h_c[i] - ref[i]);
        res.max_diff = std::max(res.max_diff, d);
        if (d != 0) res.ok = false;
    }
    return res;
}

CheckResult check_mode(sycl::queue &q, int sg, int mode) {
    if (sg == 16) {
        switch (mode) {
        case BM_COLMAJOR: return run_check<16, BM_COLMAJOR>(q);
        case BM_PACKED: return run_check<16, BM_PACKED>(q);
        case BM_PACKED_S16: return run_check<16, BM_PACKED_S16>(q);
        default: return run_check<16, BM_ROWMAJOR>(q);
        }
    }
    switch (mode) {
    case BM_COLMAJOR: return run_check<32, BM_COLMAJOR>(q);
    case BM_PACKED: return run_check<32, BM_PACKED>(q);
    case BM_PACKED_S16: return run_check<32, BM_PACKED_S16>(q);
    default: return run_check<32, BM_ROWMAJOR>(q);
    }
}

// ---------------------------------------------------------------------------
template <typename Launch>
double time_ms(sycl::queue &q, int iters, Launch &&launch) {
    launch(); // warm (JIT paid here, not in the loop)
    q.wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) launch();
    q.wait();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

// Runs fp16 + int8f + int8p for one (SG, DK, BMODE) and prints CSV + ratios.
template <int SG, int DK, int BMODE>
void run_bench_set(sycl::queue &q, const sycl::half *q16, const sycl::half *k16,
                   const int8_t *q8d, const int8_t *k8d, float *of, int32_t *oi,
                   int64_t rows, int iters, int wg_m,
                   const std::vector<int32_t> &h_ref) {
    const size_t groups = static_cast<size_t>(rows / (kTm * wg_m));
    auto tops = [&](double ms) {
        return (2.0 * rows * kTileN * DK) / (ms * 1e-3) / 1e12;
    };
    auto check_head = [&](const char *kind, double ms) {
        std::vector<int32_t> h_oi(2 * kTm * kTileN);
        q.memcpy(h_oi.data(), oi, h_oi.size() * sizeof(int32_t)).wait();
        double dmax = 0;
        for (size_t i = 0; i < h_oi.size(); ++i)
            dmax = std::max(dmax,
                            static_cast<double>(std::abs(h_oi[i] - h_ref[i])));
        std::cout << kind << "," << DK << "," << SG << "," << rows << ","
                  << (dmax == 0 ? "ok" : "wrong") << "," << ms << ","
                  << tops(ms) << "," << dmax << "\n";
    };

    const double ms16 = time_ms(q, iters, [&] {
        q.submit([&](sycl::handler &h) {
            sycl::local_accessor<sycl::half, 1> qs(sycl::range<1>(kTm * DK), h);
            sycl::local_accessor<sycl::half, 1> ks(sycl::range<1>(DK * kTileN),
                                                   h);
            h.parallel_for(sycl::nd_range<1>({groups * SG}, {SG}),
                           Fp16DpasBenchKernel<SG, DK>(q16, k16, of, wg_m, qs,
                                                       ks));
        });
    });
    std::cout << "fp16," << DK << "," << SG << "," << rows << ",ok," << ms16
              << "," << tops(ms16) << ",\n";

    const double ms8f = time_ms(q, iters, [&] {
        q.submit([&](sycl::handler &h) {
            sycl::local_accessor<sycl::half, 1> qs(sycl::range<1>(kTm * DK), h);
            sycl::local_accessor<sycl::half, 1> ks(sycl::range<1>(DK * kTileN),
                                                   h);
            sycl::local_accessor<int8_t, 1> q8(sycl::range<1>(kTm * DK), h);
            sycl::local_accessor<int8_t, 1> k8(sycl::range<1>(DK * kTileN), h);
            sycl::local_accessor<float, 1> sc(sycl::range<1>(kTm + kTileN), h);
            h.parallel_for(sycl::nd_range<1>({groups * SG}, {SG}),
                           Int8FusedDpasBenchKernel<SG, DK, BMODE>(
                               q16, k16, oi, wg_m, qs, ks, q8, k8, sc));
        });
    });
    check_head("int8f", ms8f);

    const double ms8p = time_ms(q, iters, [&] {
        q.submit([&](sycl::handler &h) {
            sycl::local_accessor<int8_t, 1> q8(sycl::range<1>(kTm * DK), h);
            sycl::local_accessor<int8_t, 1> k8(sycl::range<1>(DK * kTileN), h);
            h.parallel_for(sycl::nd_range<1>({groups * SG}, {SG}),
                           Int8PrepackedDpasBenchKernel<SG, DK, BMODE>(
                               q8d, k8d, oi, wg_m, q8, k8));
        });
    });
    check_head("int8p", ms8p);

    std::cout << "# ratio k=" << DK << " sg=" << SG
              << ": int8f/fp16=" << (tops(ms8f) / tops(ms16))
              << " int8p/fp16=" << (tops(ms8p) / tops(ms16)) << "\n";
}

void run_bench_dispatch(sycl::queue &q, int sg, int dk, int bmode,
                        const sycl::half *q16, const sycl::half *k16,
                        const int8_t *q8d, const int8_t *k8d, float *of,
                        int32_t *oi, int64_t rows, int iters, int wg_m,
                        const std::vector<int32_t> &h_ref) {
#define AFI_RUN(SG_, DK_, BM_)                                                 \
    run_bench_set<SG_, DK_, BM_>(q, q16, k16, q8d, k8d, of, oi, rows, iters,   \
                                 wg_m, h_ref)
    if (sg == 16 && dk == 32) {
        switch (bmode) {
        case BM_COLMAJOR: return AFI_RUN(16, 32, BM_COLMAJOR);
        case BM_PACKED: return AFI_RUN(16, 32, BM_PACKED);
        case BM_PACKED_S16: return AFI_RUN(16, 32, BM_PACKED_S16);
        default: return AFI_RUN(16, 32, BM_ROWMAJOR);
        }
    }
    if (sg == 16 && dk == 128) {
        switch (bmode) {
        case BM_COLMAJOR: return AFI_RUN(16, 128, BM_COLMAJOR);
        case BM_PACKED: return AFI_RUN(16, 128, BM_PACKED);
        case BM_PACKED_S16: return AFI_RUN(16, 128, BM_PACKED_S16);
        default: return AFI_RUN(16, 128, BM_ROWMAJOR);
        }
    }
    if (sg == 32 && dk == 32) {
        switch (bmode) {
        case BM_COLMAJOR: return AFI_RUN(32, 32, BM_COLMAJOR);
        case BM_PACKED: return AFI_RUN(32, 32, BM_PACKED);
        case BM_PACKED_S16: return AFI_RUN(32, 32, BM_PACKED_S16);
        default: return AFI_RUN(32, 32, BM_ROWMAJOR);
        }
    }
    switch (bmode) {
    case BM_COLMAJOR: return AFI_RUN(32, 128, BM_COLMAJOR);
    case BM_PACKED: return AFI_RUN(32, 128, BM_PACKED);
    case BM_PACKED_S16: return AFI_RUN(32, 128, BM_PACKED_S16);
    default: return AFI_RUN(32, 128, BM_ROWMAJOR);
    }
#undef AFI_RUN
}

void print_combo_lines(const afi::DeviceInfo &info) {
    std::cout << "# matrix_combinations (int8 + fp16, subgroup scope):\n";
    for (const auto &c : info.matrix_combos) {
        const bool is8 = c.atype == "sint8" || c.atype == "uint8";
        const bool is16 = c.atype == "fp16";
        if ((!is8 && !is16) || c.max_m == 0) continue;
        std::cout << "#   " << c.atype << " x " << c.btype << " -> " << c.ctype
                  << "/" << c.dtype << "  M(max)=" << c.max_m << " N=" << c.n
                  << " K=" << c.k << "\n";
    }
}

} // namespace

int main(int argc, char **argv) {
    int dev_index = 0;
    int64_t rows = 16384;
    int iters = 50;
    int wg_m = 4; // Q row-blocks per WG (K-tile staging amortization)
    std::vector<int> sgs;
    std::string bmode_arg = "auto";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-d" || a == "--device") && i + 1 < argc) {
            dev_index = std::stoi(argv[++i]);
        } else if (a == "--rows" && i + 1 < argc) {
            rows = std::stoll(argv[++i]);
        } else if (a == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (a == "--wgm" && i + 1 < argc) {
            wg_m = std::stoi(argv[++i]);
        } else if (a == "--sg" && i + 1 < argc) {
            sgs.push_back(std::stoi(argv[++i]));
        } else if (a == "--bmode" && i + 1 < argc) {
            bmode_arg = argv[++i];
        } else {
            std::cerr << "bench-qkint8: unknown arg " << a << "\n";
            return 2;
        }
    }
    if (sgs.empty()) sgs = {16, 32};
    if (wg_m < 1 || rows % (kTm * wg_m) != 0) {
        std::cerr << "bench-qkint8: --rows must be a multiple of tm*wgm ("
                  << kTm << "*" << wg_m << ")\n";
        return 2;
    }

    try {
        auto dev = afi::pick_dgpu(dev_index);
        sycl::queue q(dev);
        const auto info = afi::inspect(dev, dev_index);
        std::cout << "# device: " << info.name << "\n";
        std::cout << "# ext_intel_matrix: "
                  << (info.ext_intel_matrix ? "yes" : "NO")
                  << "  matrix_query: " << info.matrix_query_note << "\n";
        print_combo_lines(info);

        // (c) B-layout probe + exact s32 correctness gate.
        std::vector<int> modes;
        if (bmode_arg == "auto")
            modes = {BM_COLMAJOR, BM_PACKED, BM_PACKED_S16, BM_ROWMAJOR};
        else if (bmode_arg == "colmajor") modes = {BM_COLMAJOR};
        else if (bmode_arg == "packed") modes = {BM_PACKED};
        else if (bmode_arg == "packed16") modes = {BM_PACKED_S16};
        else modes = {BM_ROWMAJOR};

        int bmode = -1;
        for (int sg : sgs) {
            for (int m : modes) {
                const auto r = check_mode(q, sg, m);
                std::cout << "# check sg=" << sg
                          << " bmode=" << bmode_name(m) << " : "
                          << (r.ok ? "EXACT MATCH" : "MISMATCH")
                          << " (max_diff=" << r.max_diff << ")\n";
                if (r.ok && bmode < 0) bmode = m;
            }
        }
        if (bmode < 0) {
            std::cout << "# VERDICT: no int8 B layout gave an exact s32 match;"
                         " joint_matrix<int8_t> is not usable as-is — Phase 1"
                         " would need the ESIMD xmx::dpas lane. STOP.\n";
            return 1;
        }
        std::cout << "# using bmode=" << bmode_name(bmode) << "\n";
        std::cout << "kind,k,sg,rows,status,ms,tops,max_diff\n";

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int dk : {32, 128}) {
            const size_t q_elems = static_cast<size_t>(rows) * dk;
            std::vector<sycl::half> h_q(q_elems), h_k(dk * kTileN);
            for (auto &v : h_q) v = sycl::half(dist(rng));
            for (auto &v : h_k) v = sycl::half(dist(rng));

            // Host-side int8 replicate of the fused path (same formula:
            // per-row absmax for Q, per-column absmax for K, multiply by the
            // precomputed 127/absmax reciprocal, round to nearest) for the
            // exact s32 gate, plus the prepacked inputs.
            std::vector<int8_t> h_q8(q_elems), h_k8log(dk * kTileN),
                h_k8s(dk * kTileN);
            for (int64_t r = 0; r < rows; ++r) {
                float m = 0.0f;
                for (int c = 0; c < dk; ++c)
                    m = std::max(m,
                                 std::fabs(static_cast<float>(h_q[r * dk + c])));
                const float inv = 127.0f / (m > 0.0f ? m : 1.0f);
                for (int c = 0; c < dk; ++c)
                    h_q8[r * dk + c] = static_cast<int8_t>(std::round(
                        static_cast<float>(h_q[r * dk + c]) * inv));
            }
            for (int n = 0; n < kTileN; ++n) {
                float m = 0.0f;
                for (int c = 0; c < dk; ++c)
                    m = std::max(
                        m, std::fabs(static_cast<float>(h_k[c * kTileN + n])));
                const float inv = 127.0f / (m > 0.0f ? m : 1.0f);
                for (int c = 0; c < dk; ++c)
                    h_k8log[c * kTileN + n] = static_cast<int8_t>(std::round(
                        static_cast<float>(h_k[c * kTileN + n]) * inv));
            }
            for (int c = 0; c < dk; ++c)
                for (int n = 0; n < kTileN; ++n)
                    h_k8s[bstage_index(bmode, c, n, dk)] =
                        h_k8log[c * kTileN + n];
            std::vector<int32_t> h_ref(rows * kTileN);
            host_s32(h_q8.data(), h_k8log.data(), rows, dk, h_ref.data());

            sycl::half *q16 = sycl::malloc_device<sycl::half>(q_elems, q);
            sycl::half *k16 = sycl::malloc_device<sycl::half>(dk * kTileN, q);
            int8_t *q8d = sycl::malloc_device<int8_t>(q_elems, q);
            int8_t *k8d = sycl::malloc_device<int8_t>(dk * kTileN, q);
            float *of = sycl::malloc_device<float>(rows * kTileN, q);
            int32_t *oi = sycl::malloc_device<int32_t>(rows * kTileN, q);
            q.memcpy(q16, h_q.data(), q_elems * sizeof(sycl::half)).wait();
            q.memcpy(k16, h_k.data(), dk * kTileN * sizeof(sycl::half)).wait();
            q.memcpy(q8d, h_q8.data(), q_elems).wait();
            q.memcpy(k8d, h_k8s.data(), dk * kTileN).wait();

            for (int sg : sgs)
                run_bench_dispatch(q, sg, dk, bmode, q16, k16, q8d, k8d, of, oi,
                                   rows, iters, wg_m, h_ref);

            sycl::free(q16, q);
            sycl::free(k16, q);
            sycl::free(q8d, q);
            sycl::free(k8d, q);
            sycl::free(of, q);
            sycl::free(oi, q);
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "bench-qkint8: " << e.what() << "\n";
        return 1;
    }
}
