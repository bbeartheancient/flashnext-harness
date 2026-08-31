// QkInt8RotKernel — Phase 1 of docs/fused-dpas-int8-plan.md.
//
// One launch: FWHT-rotate Q block (once, head-resident) + per-K-tile
// rotate + per-token absmax INT8 quant + s32 DPAS QK^T + rank-1 rescale.
// Rotation is fp32-compute with the H.16 DPAS atom for the 16-wide stages
// (same network as FwhtXmxTunedKernel) and fp32 SLM tail butterflies;
// quant is symmetric per-token INT8 (inverse scales applied only on the
// fp32 score after the s32 accumulate — no /n, no ternary, nothing folded
// into the int8 values).
//
// Work split (multi-subgroup, the Phase 0 lesson): a WG owns QB = TM*wg_q
// query rows; its wg_k subgroups each privately own one 16-key tile
// (stage -> rotate -> absmax -> quant -> MAD -> rescale with only
// sub_group barriers in the K loop). Grid is (Q blocks) x (K-tile
// ranges): kt>0 splits the K range across WGs for decode-width
// parallelism at the price of re-rotating the Q block per K range.

#include "afi/qkint8.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <vector>

namespace afi {
namespace {

namespace matrix = sycl::ext::oneapi::experimental::matrix;

// Same leaf rule as fwht.cpp: the DPAS leaf for fp16/bf16 input is the
// input type itself (fp32 compute happens in the accumulator).
template <typename T>
using leaf_t = std::conditional_t<std::is_same_v<T, float>, sycl::half, T>;

constexpr int kTileN = 16;  // DPAS N (keys per tile)
constexpr int kDpasK8 = 32; // int8 DPAS accumulate depth

// Sylvester H(16)_{i,j} — same table as fwht.cpp / python/afi/fwht.py.
constexpr int8_t sylvester_pm1(int i, int j) {
    int x = i & j;
    int bits = 0;
    for (int b = 0; b < 8; ++b)
        bits += (x >> b) & 1;
    return (bits & 1) ? static_cast<int8_t>(-1) : static_cast<int8_t>(1);
}

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
    for (int i = 0; i < n; ++i) y[i] *= scale;
}

template <typename T, int D, int TM, int SG>
class QkInt8RotKernel {
public:
    using leaf = leaf_t<T>;

    QkInt8RotKernel(const T *q, const T *k, float *out, int64_t rq, int64_t rk,
                    int wg_k, int wg_q, int kt, int ldm,
                    sycl::local_accessor<float, 1> qfp,
                    sycl::local_accessor<leaf, 1> qleaf,
                    sycl::local_accessor<int8_t, 1> q8,
                    sycl::local_accessor<float, 1> kfp,
                    sycl::local_accessor<leaf, 1> kleaf,
                    sycl::local_accessor<int8_t, 1> k8,
                    sycl::local_accessor<int32_t, 1> sc,
                    sycl::local_accessor<float, 1> scales,
                    sycl::local_accessor<leaf, 1> h_slm)
        : q_(q), k_(k), out_(out), rq_(rq), rk_(rk),
          wg_k_(wg_k), wg_q_(wg_q), kt_(kt), ldm_(ldm), qfp_(qfp),
          qleaf_(qleaf), q8_(q8), kfp_(kfp), kleaf_(kleaf), k8_(k8), sc_(sc),
          scales_(scales), h_slm_(h_slm) {}

    [[sycl::reqd_sub_group_size(SG)]]
    void operator()(sycl::nd_item<2> it) const {
        const int lid = static_cast<int>(it.get_local_linear_id());
        auto sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int sid = static_cast<int>(sg.get_group_linear_id());
        const int nthreads = wg_k_ * SG;
        const int qb = TM * wg_q_;
        const int64_t row0 = static_cast<int64_t>(it.get_group(0)) * qb;
        const int num_tiles = static_cast<int>((rk_ + kTileN - 1) / kTileN);
        const int tpw = kt_ > 0 ? kt_ : num_tiles;
        const int t0 = static_cast<int>(it.get_group(1)) * tpw;
        const int t1 = sycl::min(t0 + tpw, num_tiles);

        if (sid == 0 && lane < 16) {
            for (int j = 0; j < 16; ++j)
                h_slm_[lane * 16 + j] =
                    leaf(static_cast<float>(sylvester_pm1(lane, j)));
        }

        // ---- Q block: stage, rotate, absmax, quant (WG-cooperative) ----
        for (int e = lid; e < qb * D; e += nthreads) {
            const int r = e / D, c = e % D;
            const int64_t row = row0 + r;
            const leaf v = row < rq_ ? leaf(q_[row * D + c]) : leaf(0);
            qfp_[r * ldm_ + c] = static_cast<float>(v);
            qleaf_[r * ldm_ + c] = v;
        }
        sycl::group_barrier(it.get_group());

        matrix::joint_matrix<sycl::sub_group, leaf, matrix::use::b, 16, 16,
                             matrix::layout::row_major>
            B16;
        matrix::joint_matrix_load(
            sg, B16,
            h_slm_.template get_multi_ptr<sycl::access::decorated::no>(), 16);
        for (int chunk = sid * 16; chunk < D; chunk += wg_k_ * 16) {
            for (int rt = 0; rt < qb; rt += TM) {
                matrix::joint_matrix<sycl::sub_group, leaf, matrix::use::a, TM,
                                     16, matrix::layout::row_major>
                    A;
                matrix::joint_matrix<sycl::sub_group, float,
                                     matrix::use::accumulator, TM, 16>
                    C;
                matrix::joint_matrix_load(
                    sg, A,
                    qleaf_.template get_multi_ptr<sycl::access::decorated::no>() +
                        rt * ldm_ + chunk,
                    ldm_);
                matrix::joint_matrix_fill(sg, C, 0.0f);
                matrix::joint_matrix_mad(sg, C, A, B16, C);
                matrix::joint_matrix_store(
                    sg, C,
                    qfp_.template get_multi_ptr<sycl::access::decorated::no>() +
                        rt * ldm_ + chunk,
                    ldm_, matrix::layout::row_major);
            }
        }
        sycl::group_barrier(it.get_group());
        for (int h = 16; h < D; h *= 2) {
            const int stride = h * 2;
            for (int col = lid; col < D; col += nthreads) {
                if ((col % stride) < h) {
                    for (int r = 0; r < qb; ++r) {
                        float *rowp = &qfp_[r * ldm_];
                        const float a = rowp[col];
                        const float b = rowp[col + h];
                        rowp[col] = a + b;
                        rowp[col + h] = a - b;
                    }
                }
            }
            sycl::group_barrier(it.get_group());
        }

        float *q_inv = scales_
                           .template get_multi_ptr<sycl::access::decorated::no>()
                           .get();                     // [qb]
        float *k_inv = q_inv + qb;                      // [wg_k*16]
        for (int r = lid; r < qb; r += nthreads) {
            float m = 0.0f;
            for (int c = 0; c < D; ++c)
                m = sycl::fmax(m, sycl::fabs(qfp_[r * ldm_ + c]));
            q_inv[r] = 127.0f / (m > 0.0f ? m : 1.0f);
        }
        sycl::group_barrier(it.get_group());
        for (int e = lid; e < qb * D; e += nthreads) {
            const int r = e / D, c = e % D;
            q8_[e] = static_cast<int8_t>(
                sycl::round(qfp_[r * ldm_ + c] * q_inv[r]));
        }
        sycl::group_barrier(it.get_group());

        // ---- K loop: each subgroup privately owns 16-key tiles ----
        const int koff = sid * 16;
        for (int tile = t0 + sid; tile < t1; tile += wg_k_) {
            const int64_t col0 = static_cast<int64_t>(tile) * kTileN;
            for (int e = lane; e < 16 * D; e += SG) {
                const int r = e / D, c = e % D;
                const int64_t col = col0 + r;
                const leaf v = col < rk_ ? leaf(k_[col * D + c]) : leaf(0);
                kfp_[koff * ldm_ + r * ldm_ + c] = static_cast<float>(v);
                kleaf_[koff * ldm_ + r * ldm_ + c] = v;
            }
            sycl::group_barrier(sg);
            for (int chunk = 0; chunk < D; chunk += 16) {
                for (int rt = 0; rt < 16; rt += TM) {
                    matrix::joint_matrix<sycl::sub_group, leaf, matrix::use::a,
                                         TM, 16, matrix::layout::row_major>
                        A;
                    matrix::joint_matrix<sycl::sub_group, float,
                                         matrix::use::accumulator, TM, 16>
                        C;
                    matrix::joint_matrix_load(
                        sg, A,
                        kleaf_.template get_multi_ptr<
                            sycl::access::decorated::no>() +
                            koff * ldm_ + rt * ldm_ + chunk,
                        ldm_);
                    matrix::joint_matrix_fill(sg, C, 0.0f);
                    matrix::joint_matrix_mad(sg, C, A, B16, C);
                    matrix::joint_matrix_store(
                        sg, C,
                        kfp_.template get_multi_ptr<
                            sycl::access::decorated::no>() +
                            koff * ldm_ + rt * ldm_ + chunk,
                        ldm_, matrix::layout::row_major);
                }
            }
            sycl::group_barrier(sg);
            for (int h = 16; h < D; h *= 2) {
                const int stride = h * 2;
                for (int col = lane; col < D; col += SG) {
                    if ((col % stride) < h) {
                        for (int r = 0; r < 16; ++r) {
                            float *rowp = &kfp_[koff * ldm_ + r * ldm_];
                            const float a = rowp[col];
                            const float b = rowp[col + h];
                            rowp[col] = a + b;
                            rowp[col + h] = a - b;
                        }
                    }
                }
                sycl::group_barrier(sg);
            }
            for (int r = lane; r < 16; r += SG) {
                float m = 0.0f;
                for (int c = 0; c < D; ++c)
                    m = sycl::fmax(
                        m, sycl::fabs(kfp_[koff * ldm_ + r * ldm_ + c]));
                k_inv[koff + r] = 127.0f / (m > 0.0f ? m : 1.0f);
            }
            sycl::group_barrier(sg);
            for (int e = lane; e < 16 * D; e += SG) {
                const int r = e / D, c = e % D;
                k8_[koff * D + e] = static_cast<int8_t>(sycl::round(
                    kfp_[koff * ldm_ + r * ldm_ + c] * k_inv[koff + r]));
            }
            sycl::group_barrier(sg);

            auto k8_ptr =
                k8_.template get_multi_ptr<sycl::access::decorated::no>() +
                koff * D;
            for (int rt = 0; rt < qb; rt += TM) {
                matrix::joint_matrix<sycl::sub_group, int32_t,
                                     matrix::use::accumulator, TM, 16>
                    C;
                matrix::joint_matrix_fill(sg, C, 0);
                for (int chunk = 0; chunk < D; chunk += kDpasK8) {
                    matrix::joint_matrix<sycl::sub_group, int8_t,
                                         matrix::use::a, TM, kDpasK8,
                                         matrix::layout::row_major>
                        A;
                    matrix::joint_matrix<sycl::sub_group, int8_t,
                                         matrix::use::b, kDpasK8, 16,
                                         matrix::layout::col_major>
                        B;
                    matrix::joint_matrix_load(
                        sg, A,
                        q8_.template get_multi_ptr<
                            sycl::access::decorated::no>() +
                            rt * D + chunk,
                        D);
                    matrix::joint_matrix_load(sg, B, k8_ptr + chunk, D);
                    matrix::joint_matrix_mad(sg, C, A, B, C);
                }
                matrix::joint_matrix_store(
                    sg, C,
                    sc_.template get_multi_ptr<sycl::access::decorated::no>() +
                        koff * qb + rt * 16,
                    16, matrix::layout::row_major);
            }
            sycl::group_barrier(sg);
            for (int e = lane; e < qb * 16; e += SG) {
                const int r = e / 16, n = e % 16;
                const int64_t row = row0 + r, col = col0 + n;
                if (row < rq_ && col < rk_) {
                    out_[row * rk_ + col] =
                        static_cast<float>(sc_[koff * qb + e]) /
                        (q_inv[r] * k_inv[koff + n]);
                }
            }
            sycl::group_barrier(sg);
        }
    }

private:
    const T *q_;
    const T *k_;
    float *out_;
    int64_t rq_;
    int64_t rk_;
    int wg_k_;
    int wg_q_;
    int kt_;
    int ldm_;
    sycl::local_accessor<float, 1> qfp_;
    sycl::local_accessor<leaf, 1> qleaf_;
    sycl::local_accessor<int8_t, 1> q8_;
    sycl::local_accessor<float, 1> kfp_;
    sycl::local_accessor<leaf, 1> kleaf_;
    sycl::local_accessor<int8_t, 1> k8_;
    sycl::local_accessor<int32_t, 1> sc_;
    sycl::local_accessor<float, 1> scales_;
    sycl::local_accessor<leaf, 1> h_slm_;
};

int ceil_div(int64_t a, int64_t b) { return static_cast<int>((a + b - 1) / b); }

template <typename T, int D, int TM, int SG>
void launch(sycl::queue &q, const T *qs, const T *ks, float *out, int64_t rq,
            int64_t rk, const QkInt8Config &cfg) {
    const int qb = TM * cfg.wg_q;
    const int ldm = D + cfg.pad;
    const int64_t tiles = (rk + kTileN - 1) / kTileN;
    const int tpw = cfg.kt > 0 ? cfg.kt : static_cast<int>(tiles);
    const size_t gx = static_cast<size_t>(ceil_div(rq, qb));
    const size_t gy = static_cast<size_t>(ceil_div(tiles, tpw));
    const size_t local = static_cast<size_t>(cfg.wg_k) * SG;
    q.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> qfp(sycl::range<1>(qb * ldm), h);
        sycl::local_accessor<leaf_t<T>, 1> qleaf(sycl::range<1>(qb * ldm), h);
        sycl::local_accessor<int8_t, 1> q8(sycl::range<1>(qb * D), h);
        sycl::local_accessor<float, 1> kfp(
            sycl::range<1>(cfg.wg_k * 16 * ldm), h);
        sycl::local_accessor<leaf_t<T>, 1> kleaf(
            sycl::range<1>(cfg.wg_k * 16 * ldm), h);
        sycl::local_accessor<int8_t, 1> k8(sycl::range<1>(cfg.wg_k * 16 * D),
                                           h);
        sycl::local_accessor<int32_t, 1> sc(sycl::range<1>(cfg.wg_k * qb * 16),
                                            h);
        sycl::local_accessor<float, 1> scales(
            sycl::range<1>(qb + cfg.wg_k * 16), h);
        sycl::local_accessor<leaf_t<T>, 1> h_slm(sycl::range<1>(256), h);
        h.parallel_for(
            sycl::nd_range<2>({gx, gy * local}, {1, local}),
            QkInt8RotKernel<T, D, TM, SG>(qs, ks, out, rq, rk, cfg.wg_k,
                                          cfg.wg_q, cfg.kt, ldm, qfp, qleaf,
                                          q8, kfp, kleaf, k8, sc, scales,
                                          h_slm));
    });
}

template <typename T, int D>
void launch_ts(sycl::queue &q, const T *qs, const T *ks, float *out,
               int64_t rq, int64_t rk, const QkInt8Config &cfg) {
    const int key = cfg.tm * 64 + cfg.sg;
    switch (key) {
    case 4 * 64 + 16: launch<T, D, 4, 16>(q, qs, ks, out, rq, rk, cfg); return;
    case 8 * 64 + 16: launch<T, D, 8, 16>(q, qs, ks, out, rq, rk, cfg); return;
    case 4 * 64 + 32: launch<T, D, 4, 32>(q, qs, ks, out, rq, rk, cfg); return;
    case 8 * 64 + 32: launch<T, D, 8, 32>(q, qs, ks, out, rq, rk, cfg); return;
    default: throw std::invalid_argument("qkint8: bad tm/sg combo");
    }
}

template <typename T>
void launch_d(sycl::queue &q, const T *qs, const T *ks, float *out, int d,
              int64_t rq, int64_t rk, const QkInt8Config &cfg) {
    switch (d) {
    case 128: launch_ts<T, 128>(q, qs, ks, out, rq, rk, cfg); return;
    case 256: launch_ts<T, 256>(q, qs, ks, out, rq, rk, cfg); return;
    default:
        throw std::invalid_argument("qkint8: d must be 128 or 256, got " +
                                    std::to_string(d));
    }
}

} // namespace

bool qkint8_config_ok(const sycl::device &dev, const QkInt8Config &cfg,
                      int d) {
    if (d != 128 && d != 256) return false;
    if (!fwht_xmx_ok(dev)) return false;
    if (cfg.tm != 4 && cfg.tm != 8) return false;
    if (cfg.sg != 16 && cfg.sg != 32) return false;
    bool sg_ok = false;
    for (const auto s : dev.get_info<sycl::info::device::sub_group_sizes>()) {
        if (static_cast<int>(s) == cfg.sg) sg_ok = true;
    }
    if (!sg_ok) return false;
    if (cfg.wg_k < 1 || cfg.wg_k > 8) return false;
    if (cfg.wg_q < 1 || cfg.wg_q > 4) return false;
    // DPAS ldm must stay 16B-aligned (the docs/xmx-tuner.md lesson).
    if (cfg.pad != 0 && cfg.pad != 8) return false;
    if (cfg.kt < 0) return false;
    if (static_cast<size_t>(cfg.wg_k) * cfg.sg >
        dev.get_info<sycl::info::device::max_work_group_size>())
        return false;
    // SLM arena (bytes): fp32 + leaf rotate tiles for Q and per-sg K, int8
    // stages, s32 score tiles, scales, H16 atom.
    const size_t qb = static_cast<size_t>(cfg.tm) * cfg.wg_q;
    const size_t ld = static_cast<size_t>(d) + cfg.pad;
    const size_t slm = qb * ld * 6 + qb * d +
                       static_cast<size_t>(cfg.wg_k) * 16 * ld * 6 +
                       static_cast<size_t>(cfg.wg_k) * 16 * d +
                       static_cast<size_t>(cfg.wg_k) * qb * 16 * 4 +
                       (qb + cfg.wg_k * 16) * 4 + 512;
    if (slm > dev.get_info<sycl::info::device::local_mem_size>()) return false;
    return true;
}

bool qkint8_device(sycl::queue &q, const void *q_src, const void *k_src,
                   float *out, int d, int64_t rq, int64_t rk, FwhtElem elem,
                   const QkInt8Config &cfg, bool wait) {
    if (!qkint8_config_ok(q.get_device(), cfg, d)) return false;
    if (rq <= 0 || rk <= 0) return true;
    switch (elem) {
    case FwhtElem::F16:
        launch_d<sycl::half>(q, static_cast<const sycl::half *>(q_src),
                             static_cast<const sycl::half *>(k_src), out, d,
                             rq, rk, cfg);
        break;
    case FwhtElem::BF16:
        launch_d<sycl::ext::oneapi::bfloat16>(
            q, static_cast<const sycl::ext::oneapi::bfloat16 *>(q_src),
            static_cast<const sycl::ext::oneapi::bfloat16 *>(k_src), out, d,
            rq, rk, cfg);
        break;
    default:
        return false;
    }
    if (wait) q.wait_and_throw();
    return true;
}

void qkint8_host(const float *q, const float *k, float *out, int d,
                 int64_t rq, int64_t rk) {
    // Unnormalized Sylvester rotate: the 1/sqrt(d) factor cancels exactly
    // under per-token symmetric quant (absmax(x*s) = s*absmax(x)), so the
    // device stages the leaf inputs unscaled and stays leaf-exact.
    std::vector<float> qr(q, q + rq * d), kr(k, k + rk * d);
    for (int64_t r = 0; r < rq; ++r) fwht_host_row(&qr[r * d], d, 1.0f);
    for (int64_t r = 0; r < rk; ++r) fwht_host_row(&kr[r * d], d, 1.0f);
    std::vector<int8_t> q8(rq * d), k8(rk * d);
    std::vector<float> q_inv(rq), k_inv(rk);
    for (int64_t r = 0; r < rq; ++r) {
        float m = 0.0f;
        for (int c = 0; c < d; ++c) m = std::max(m, std::fabs(qr[r * d + c]));
        q_inv[r] = 127.0f / (m > 0.0f ? m : 1.0f);
        for (int c = 0; c < d; ++c)
            q8[r * d + c] =
                static_cast<int8_t>(std::lround(qr[r * d + c] * q_inv[r]));
    }
    for (int64_t r = 0; r < rk; ++r) {
        float m = 0.0f;
        for (int c = 0; c < d; ++c) m = std::max(m, std::fabs(kr[r * d + c]));
        k_inv[r] = 127.0f / (m > 0.0f ? m : 1.0f);
        for (int c = 0; c < d; ++c)
            k8[r * d + c] =
                static_cast<int8_t>(std::lround(kr[r * d + c] * k_inv[r]));
    }
    for (int64_t i = 0; i < rq; ++i) {
        for (int64_t j = 0; j < rk; ++j) {
            int32_t acc = 0;
            for (int c = 0; c < d; ++c)
                acc += static_cast<int32_t>(q8[i * d + c]) *
                       static_cast<int32_t>(k8[j * d + c]);
            out[i * rk + j] = static_cast<float>(acc) / (q_inv[i] * k_inv[j]);
        }
    }
}

void qkint8_host_fp32(const float *q, const float *k, float *out, int d,
                      int64_t rq, int64_t rk) {
    std::vector<float> qr(q, q + rq * d), kr(k, k + rk * d);
    for (int64_t r = 0; r < rq; ++r) fwht_host_row(&qr[r * d], d, 1.0f);
    for (int64_t r = 0; r < rk; ++r) fwht_host_row(&kr[r * d], d, 1.0f);
    for (int64_t i = 0; i < rq; ++i) {
        for (int64_t j = 0; j < rk; ++j) {
            float acc = 0.0f;
            for (int c = 0; c < d; ++c)
                acc += qr[i * d + c] * kr[j * d + c];
            out[i * rk + j] = acc;
        }
    }
}

} // namespace afi
