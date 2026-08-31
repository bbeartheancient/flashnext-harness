// afi-bench-dpas-esimd — Phase 0b ESIMD xmx::dpas s8 probe
// (docs/fused-dpas-int8-plan.md, "Parameter update + Phase 0b").
//
// Measures achievable TOPS of an ESIMD s8xs8->s32 GEMM-like loop with the
// levers oneDNN/XeTLA own and the joint_matrix lane lacked:
//   - lsc_load_2d with hardware VNNI transform for B (no SLM round-trip),
//   - A-fragment reuse across NT N-tiles per k-chunk,
//   - B-tile reuse across MB m-blocks per thread (register blocking),
//   - optional one-k-chunk-ahead lsc_prefetch_2d.
//
// One ESIMD thread computes MB x (8 x NT*16) output (RC=8 = B70 s8 max M;
// exec size N=16; K=32 per dpas call) over the full depth DK. Grid =
// (rows/(8*MB)) * (cols/(NT*16)) threads, mapped column-group major so
// co-resident threads share streamed B tiles in cache.
//
// NOTE (Phase 0b landmine): feeding dpas from a FLAT load of host-prepacked
// VNNI bytes computes garbage in some codegen contexts on intel/llvm v7 /
// BMG-G31, while the identical bytes through the hardware-transformed load
// (Transformed=true) are bit-exact everywhere (decoded register layout +
// equivalence proofs: afi-dpas-vnni-layout). All B traffic therefore goes
// through transformed loads from row-major surfaces.
//
// CSV: dk,sg,nt,mb,pf,rows,status,ms,tops,max_diff
//
// afi-bench-dpas-esimd [-d dev] [--rows 16384] [--iters 50] [--cols 512]
//   [--dk 128 ...] [--sg 16 ...] [--nt 4 8] [--mb 1 2 4] [--pf 0|1]

#include "afi/device.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;

namespace {

constexpr int kRc = 8;     // DPAS repeat count = tile M (B70 s8 max)
constexpr int kKc = 32;    // DPAS accumulate depth per call (s8)
constexpr int kExecN = 16; // DPAS execution size (BMG-G31)

template <int SG, int DK, int NT, int MB, bool PF>
struct DpasGemmName {
    // Unique kernel-name tag per instantiation.
};

// One thread owns MB consecutive 8-row strips x NT col-tiles over depth DK.
// Every B tile is loaded once per (chunk, n-tile) and consumed by all MB
// A fragments — the Georganas rectangular-reuse pattern.
template <int SG, int DK, int NT, int MB, bool PF>
double run_config(sycl::queue &q, const int8_t *ad, const int8_t *bd,
                  int32_t *cd, int64_t rows, int cols, int iters) {
    constexpr int kChunks = DK / kKc;
    const int64_t m_blocks = rows / (kRc * MB);
    const int64_t n_groups = cols / (NT * kExecN);
    const int64_t threads = m_blocks * n_groups;

    auto launch = [&] {
        q.parallel_for<DpasGemmName<SG, DK, NT, MB, PF>>(
            sycl::nd_range<1>({size_t(threads)}, {size_t(SG)}),
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                const int64_t tid =
                    static_cast<int64_t>(it.get_global_id(0));
                // Column-group-major map: co-resident threads share B tiles.
                const int64_t mb = tid % m_blocks;
                const int64_t ng = tid / m_blocks;
                const int64_t row0 = mb * kRc * MB;
                const int64_t col0 = ng * NT * kExecN;

                // A fragments [MB][chunks], each [8 x 32] row-major dwords.
                esimd::simd<int8_t, kRc * kKc> afrag[MB][kChunks];
#pragma unroll
                for (int j = 0; j < MB; ++j)
#pragma unroll
                    for (int kc = 0; kc < kChunks; ++kc)
                        afrag[j][kc] =
                            xesimd::lsc_load_2d<int8_t, kKc, kRc>(
                                ad, unsigned(DK - 1), unsigned(rows - 1),
                                unsigned(DK - 1), int(kc * kKc),
                                int(row0 + j * kRc));

                esimd::simd<int32_t, kRc * kExecN> acc[MB][NT];
#pragma unroll
                for (int j = 0; j < MB; ++j)
#pragma unroll
                    for (int t = 0; t < NT; ++t)
                        acc[j][t] = 0;

                for (int kc = 0; kc < kChunks; ++kc) {
                    if constexpr (PF) {
                        if (kc + 1 < kChunks) {
                            xesimd::lsc_prefetch_2d<
                                int8_t, kExecN * (NT > 4 ? 4 : NT), kKc, 1,
                                xesimd::cache_hint::cached,
                                xesimd::cache_hint::cached>(
                                bd, unsigned(cols - 1), unsigned(DK - 1),
                                unsigned(cols - 1), int(col0),
                                int((kc + 1) * kKc));
                            xesimd::lsc_prefetch_2d<
                                int8_t, kKc, kRc * MB, 1,
                                xesimd::cache_hint::cached,
                                xesimd::cache_hint::cached>(
                                ad, unsigned(DK - 1), unsigned(rows - 1),
                                unsigned(DK - 1), int((kc + 1) * kKc),
                                int(row0));
                        }
                    }
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        // Hardware VNNI transform during load (the proven
                        // feed); direct init — see landmine note above.
                        const esimd::simd<int8_t, kKc * kExecN> bt =
                            xesimd::lsc_load_2d<int8_t, kExecN, kKc, 1,
                                                false, true>(
                                bd, unsigned(cols - 1), unsigned(DK - 1),
                                unsigned(cols - 1), int(col0) + t * kExecN,
                                int(kc * kKc));
#pragma unroll
                        for (int j = 0; j < MB; ++j)
                            acc[j][t] =
                                esimd::xmx::dpas<8, kRc, int32_t, int32_t,
                                                 int8_t, int8_t>(acc[j][t],
                                                                 bt,
                                                                 afrag[j][kc]);
                    }
                }

#pragma unroll
                for (int j = 0; j < MB; ++j)
#pragma unroll
                    for (int t = 0; t < NT; ++t)
                        xesimd::lsc_store_2d<int32_t, kExecN, kRc>(
                            cd, unsigned(cols * sizeof(int32_t) - 1),
                            unsigned(rows - 1),
                            unsigned(cols * sizeof(int32_t) - 1),
                            int(col0) + t * kExecN, int(row0 + j * kRc),
                            acc[j][t]);
            });
    };

    launch(); // JIT + warm
    q.wait_and_throw();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        launch();
    q.wait_and_throw();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

void host_s32(const int8_t *a, const int8_t *b_rowmajor, int64_t m, int dk,
              int cols, int32_t *ref) {
    for (int64_t r = 0; r < m; ++r)
        for (int n = 0; n < cols; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < dk; ++k)
                acc += int32_t(a[r * dk + k]) *
                       int32_t(b_rowmajor[size_t(k) * cols + n]);
            ref[r * cols + n] = acc;
        }
}

template <int SG, int DK, int NT, int MB, bool PF>
void gate_and_time(sycl::queue &q, const int8_t *ad, const int8_t *bd,
                   int32_t *cd, const std::vector<int32_t> &ref, int64_t rows,
                   int cols, int iters) {
    // Correctness: one launch, compare the first two m-strips exactly.
    run_config<SG, DK, NT, MB, PF>(q, ad, bd, cd, rows, cols, 1);
    std::vector<int32_t> hc(std::min<int64_t>(rows, 2 * kRc * MB) * cols);
    q.memcpy(hc.data(), cd, hc.size() * sizeof(int32_t)).wait();
    long long max_diff = 0;
    size_t first_bad = SIZE_MAX;
    // ref covers min(rows, 2*kRc*4) rows (>= what this MB config wrote).
    for (size_t i = 0; i < hc.size(); ++i) {
        long long diff = (long long)hc[i] - (long long)ref[i];
        if (diff < 0) diff = -diff;
        if (diff > max_diff) max_diff = diff;
        if (diff != 0 && first_bad == SIZE_MAX) {
            first_bad = i;
            std::cerr << "# first mismatch idx=" << i
                      << " (r=" << i / cols << " c=" << i % cols
                      << ") dev=" << hc[i] << " ref=" << ref[i] << "\n";
        }
    }
    const bool ok = max_diff == 0;

    double ms = 0.0, tops = 0.0;
    if (ok) {
        ms = run_config<SG, DK, NT, MB, PF>(q, ad, bd, cd, rows, cols, iters);
        tops = (2.0 * double(rows) * cols * DK) / (ms * 1e-3) / 1e12;
    }
    std::cout << DK << "," << SG << "," << NT << "," << MB << ","
              << (PF ? 1 : 0) << "," << rows << "," << (ok ? "ok" : "wrong")
              << "," << ms << "," << tops << "," << max_diff << "\n";
}

template <int SG, int DK, int NT, int MB>
void dispatch_pf(sycl::queue &q, const int8_t *ad, const int8_t *bd,
                 int32_t *cd, const std::vector<int32_t> &ref, int64_t rows,
                 int cols, int iters, int pf) {
    if (pf == 0 || pf == -1)
        gate_and_time<SG, DK, NT, MB, false>(q, ad, bd, cd, ref, rows, cols,
                                             iters);
    if (pf == 1 || pf == -1)
        gate_and_time<SG, DK, NT, MB, true>(q, ad, bd, cd, ref, rows, cols,
                                            iters);
}

template <int SG, int DK, int NT>
void dispatch_mb(sycl::queue &q, const int8_t *ad, const int8_t *bd,
                 int32_t *cd, const std::vector<int32_t> &ref, int64_t rows,
                 int cols, int iters, int mb, int pf) {
    if (mb != 1 && mb != 2 && mb != 4 && mb != -1)
        throw std::invalid_argument("mb must be 1, 2 or 4");
    if (rows % (kRc * (mb == -1 ? 1 : mb)) != 0)
        return; // shape not divisible for this mb
    if (mb == 1 || mb == -1)
        dispatch_pf<SG, DK, NT, 1>(q, ad, bd, cd, ref, rows, cols, iters, pf);
    if (mb == 2 || mb == -1)
        dispatch_pf<SG, DK, NT, 2>(q, ad, bd, cd, ref, rows, cols, iters, pf);
    if (mb == 4 || mb == -1)
        dispatch_pf<SG, DK, NT, 4>(q, ad, bd, cd, ref, rows, cols, iters, pf);
}

template <int SG, int DK>
void dispatch_nt(sycl::queue &q, const int8_t *ad, const int8_t *bd,
                 int32_t *cd, const std::vector<int32_t> &ref, int64_t rows,
                 int cols, int iters, int nt, int mb, int pf) {
    if (nt != 4 && nt != 8 && nt != -1)
        throw std::invalid_argument("nt must be 4 or 8");
    if (nt == 4 || nt == -1)
        dispatch_mb<SG, DK, 4>(q, ad, bd, cd, ref, rows, cols, iters, mb, pf);
    if (nt == 8 || nt == -1)
        dispatch_mb<SG, DK, 8>(q, ad, bd, cd, ref, rows, cols, iters, mb, pf);
}

template <int SG>
void dispatch_dk(sycl::queue &q, const int8_t *ad, const int8_t *bd,
                 int32_t *cd, const std::vector<int32_t> &ref, int64_t rows,
                 int cols, int iters, int dk, int nt, int mb, int pf) {
    if (dk == 128 || dk == -1)
        dispatch_nt<SG, 128>(q, ad, bd, cd, ref, rows, cols, iters,
                             dk == -1 ? -1 : nt, mb, pf);
    if (dk == 512 || dk == -1)
        dispatch_nt<SG, 512>(q, ad, bd, cd, ref, rows, cols, iters,
                             dk == -1 ? -1 : nt, mb, pf);
}

} // namespace

int main(int argc, char **argv) {
    int dev_index = 0;
    int64_t rows = 16384;
    int iters = 50;
    int cols = 512;
    std::vector<int> sgs, dks;
    int nt = -1;   // both
    int mb = -1;   // all
    int pf = -1;   // both
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-d" || a == "--device") && i + 1 < argc)
            dev_index = std::stoi(argv[++i]);
        else if (a == "--rows" && i + 1 < argc)
            rows = std::stoll(argv[++i]);
        else if (a == "--iters" && i + 1 < argc)
            iters = std::stoi(argv[++i]);
        else if (a == "--cols" && i + 1 < argc)
            cols = std::stoi(argv[++i]);
        else if (a == "--dk" && i + 1 < argc)
            dks.push_back(std::stoi(argv[++i]));
        else if (a == "--sg" && i + 1 < argc)
            sgs.push_back(std::stoi(argv[++i]));
        else if (a == "--nt" && i + 1 < argc)
            nt = std::stoi(argv[++i]);
        else if (a == "--mb" && i + 1 < argc)
            mb = std::stoi(argv[++i]);
        else if (a == "--pf" && i + 1 < argc)
            pf = std::stoi(argv[++i]);
        else {
            std::cerr << "bench-dpas-esimd: unknown arg " << a << "\n";
            return 2;
        }
    }
    if (sgs.empty()) sgs = {16, 32};
    if (dks.empty()) dks = {128, 512};
    if (rows % (kRc * 4) != 0 || cols % (kExecN * 4) != 0 ||
        (cols % 128) != 0) {
        std::cerr << "bench-dpas-esimd: rows multiple of 32, cols multiple"
                     " of 128\n";
        return 2;
    }

    try {
        auto dev = afi::pick_dgpu(dev_index);
        sycl::queue q(dev, [](sycl::exception_list els) {
            for (const auto &e : els) {
                try { std::rethrow_exception(e); }
                catch (const std::exception &ex) {
                    std::cerr << "# ASYNC DEVICE ERROR: " << ex.what()
                              << "\n";
                }
            }
        });
        const auto info = afi::inspect(dev, dev_index);
        std::cout << "# device: " << info.name << "\n";

        constexpr int kDkMax = 512;
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(-127, 127);
        std::vector<int8_t> ha(size_t(rows) * kDkMax),
            hb(size_t(kDkMax) * cols);
        for (auto &v : ha) v = int8_t(dist(rng));
        for (auto &v : hb) v = int8_t(dist(rng));

        int8_t *ad = sycl::malloc_device<int8_t>(ha.size(), q);
        int8_t *bd = sycl::malloc_device<int8_t>(hb.size(), q);
        int32_t *cd = sycl::malloc_device<int32_t>(size_t(rows) * cols, q);
        q.memcpy(ad, ha.data(), ha.size()).wait();
        q.memcpy(bd, hb.data(), hb.size()).wait();

        std::cout << "dk,sg,nt,mb,pf,rows,status,ms,tops,max_diff\n";
        for (int sg : sgs) {
            for (int dk : dks) {
                std::vector<int32_t> ref(
                    std::min<int64_t>(rows, 2 * kRc * 4) * cols);
                host_s32(ha.data(), hb.data(),
                         std::min<int64_t>(rows, 2 * kRc * 4), dk, cols,
                         ref.data());
                if (sg == 16)
                    dispatch_dk<16>(q, ad, bd, cd, ref, rows, cols, iters, dk,
                                    nt, mb, pf);
                else
                    dispatch_dk<32>(q, ad, bd, cd, ref, rows, cols, iters, dk,
                                    nt, mb, pf);
            }
        }

        sycl::free(ad, q);
        sycl::free(bd, q);
        sycl::free(cd, q);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "bench-dpas-esimd: " << e.what() << "\n";
        return 1;
    }
}
