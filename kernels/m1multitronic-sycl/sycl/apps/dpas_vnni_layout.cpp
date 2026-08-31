// afi-dpas-vnni-layout — Phase 0b helper: decode the register byte order the
// hardware VNNI-transformed lsc_load_2d produces for one [K=32 x N=16] s8
// tile, so the host can pre-pack tiles a flat load can feed to dpas.
//
// int8 identity fills are ambiguous past 256 (mod-256 wrap), so two tiles are
// dumped: one filled with the row index (k), one with the column index (n).
// CSV: pos,k_at_pos,n_at_pos
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <random>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;

namespace {

struct DumpName {};
struct DumpName2 {};
struct EquivName {};
struct EquivName2 {};
struct MiniGemmName {};

// Bench-replica: DK=128 (4 chunks) x COLS=64 (4 tiles), rows=8, one thread
// per (m-block, col-group) exactly like afi-bench-dpas-esimd's prepack path.
constexpr int MDK = 128, MCOLS = 64, MROWS = 128;

void mini_gemm(sycl::queue &q, const int8_t *ad, const int8_t *bpack,
               int32_t *cd) {
    q.parallel_for<MiniGemmName>(
        sycl::nd_range<1>({16}, {16}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int64_t tid = static_cast<int64_t>(it.get_global_id(0));
            constexpr int kChunks = MDK / 32;
            const int64_t mb = tid % (MROWS / 8);
            const int64_t ng = tid / (MROWS / 8);
            esimd::simd<int8_t, 8 * 32> afrag[kChunks];
#pragma unroll
            for (int kc = 0; kc < kChunks; ++kc)
                afrag[kc] = xesimd::lsc_load_2d<int8_t, 32, 8>(
                    ad, unsigned(MDK - 1), unsigned(MROWS - 1),
                    unsigned(MDK - 1), int(kc * 32), int(mb * 8));
            esimd::simd<int32_t, 8 * 16> acc[4];
#pragma unroll
            for (int t = 0; t < 4; ++t)
                acc[t] = 0;
#pragma unroll
            for (int kc = 0; kc < kChunks; ++kc) {
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    auto bt =
                        xesimd::lsc_load_2d<int8_t, 32, 16>(
                            bpack + (size_t(kc) * (MCOLS / 16) + t) * 512,
                            31u, 15u, 31u, 0, 0);
                    acc[t] = esimd::xmx::dpas<8, 8, int32_t, int32_t,
                                              int8_t, int8_t>(acc[t], bt,
                                                              afrag[kc]);
                }
            }
#pragma unroll
            for (int t = 0; t < 4; ++t)
                xesimd::lsc_store_2d<int32_t, 16, 8>(
                    cd, unsigned(MCOLS * 4 - 1), unsigned(MROWS - 1),
                    unsigned(MCOLS * 4 - 1), int(ng * 64) + t * 16,
                    int(mb * 8), acc[t]);
        })
        .wait();
}

void dump_tile(sycl::queue &q, const std::vector<int8_t> &hin,
               std::vector<int8_t> &hout, bool flat) {
    auto *in = sycl::malloc_device<int8_t>(hin.size(), q);
    auto *out = sycl::malloc_device<int8_t>(512, q);
    q.memcpy(in, hin.data(), hin.size()).wait();
    if (!flat) {
        q.parallel_for<DumpName>(sycl::range<1>(1),
                                 [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
                                     auto v = xesimd::lsc_load_2d<
                                         int8_t, 16, 32, 1, false, true>(
                                         in, 15u, 31u, 15u, 0, 0);
#pragma unroll
                                     for (int i = 0; i < 8; ++i)
                                         xesimd::lsc_block_store<
                                             int8_t, 64,
                                             xesimd::lsc_data_size::u8>(
                                             out + i * 64,
                                             v.select<64, 1>(i * 64));
                                 })
            .wait();
    } else {
        q.parallel_for<DumpName2>(sycl::range<1>(1),
                                  [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
                                      auto v = xesimd::lsc_load_2d<
                                          int8_t, 32, 16, 1, false, false>(
                                          in, 31u, 15u, 31u, 0, 0);
#pragma unroll
                                      for (int i = 0; i < 8; ++i)
                                          xesimd::lsc_block_store<
                                              int8_t, 64,
                                              xesimd::lsc_data_size::u8>(
                                              out + i * 64,
                                              v.select<64, 1>(i * 64));
                                  })
            .wait();
    }
    hout.resize(512);
    q.memcpy(hout.data(), out, 512).wait();
    sycl::free(in, q);
    sycl::free(out, q);
}

} // namespace

int main() {
    try {
        sycl::queue q{sycl::device(sycl::gpu_selector_v)};
        std::vector<int8_t> hk(32 * 16), hn(32 * 16);
        for (int k = 0; k < 32; ++k)
            for (int n = 0; n < 16; ++n) {
                hk[k * 16 + n] = int8_t(k); // unique: 0..31
                hn[k * 16 + n] = int8_t(n); // unique: 0..15
            }
        std::vector<int8_t> ok_, on_, fk_, fn_;
        dump_tile(q, hk, ok_, false);
        dump_tile(q, hn, on_, false);
        dump_tile(q, hk, fk_, true);
        dump_tile(q, hn, fn_, true);
        std::cout << "# pos,vnni_k,vnni_n,flat_k,flat_n\n";
        for (int p = 0; p < 512; ++p)
            std::cout << p << "," << int(ok_[p]) << "," << int(on_[p]) << ","
                      << int(fk_[p]) << "," << int(fn_[p]) << "\n";

        // Equivalence check: random tile — transformed load vs flat load of
        // the host-packed permutation. Bitwise compare of the two GRF views.
        std::mt19937 rng(9);
        std::uniform_int_distribution<int> d(-128, 127);
        std::vector<int8_t> tile(512), packed(512);
        for (auto &x : tile) x = int8_t(d(rng));
        for (int k = 0; k < 32; ++k)
            for (int n = 0; n < 16; ++n)
                packed[(k >> 2) * 64 + (n >> 3) * 32 + (n & 7) * 4 + (k & 3)] =
                    tile[k * 16 + n];
        auto *tin = sycl::malloc_device<int8_t>(512, q);
        auto *pin = sycl::malloc_device<int8_t>(512, q);
        q.memcpy(tin, tile.data(), 512).wait();
        q.memcpy(pin, packed.data(), 512).wait();
        auto *ta = sycl::malloc_device<int8_t>(512, q);
        auto *tb = sycl::malloc_device<int8_t>(512, q);
        q.parallel_for<EquivName>(sycl::range<1>(1),
                                  [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
                                      auto vt = xesimd::lsc_load_2d<
                                          int8_t, 16, 32, 1, false, true>(
                                          tin, 15u, 31u, 15u, 0, 0);
                                      auto vp =
                                          xesimd::lsc_load_2d<int8_t, 32, 16>(
                                              pin, 31u, 15u, 31u, 0, 0);
#pragma unroll
                                      for (int i = 0; i < 8; ++i) {
                                          xesimd::lsc_block_store<
                                              int8_t, 64,
                                              xesimd::lsc_data_size::u8>(
                                              ta + i * 64,
                                              vt.select<64, 1>(i * 64));
                                          xesimd::lsc_block_store<
                                              int8_t, 64,
                                              xesimd::lsc_data_size::u8>(
                                              tb + i * 64,
                                              vp.select<64, 1>(i * 64));
                                      }
                                  })
            .wait();
        std::vector<int8_t> ha(512), hb(512);
        q.memcpy(ha.data(), ta, 512).wait();
        q.memcpy(hb.data(), tb, 512).wait();
        int mismatches = 0;
        for (int i = 0; i < 512; ++i)
            if (ha[i] != hb[i]) {
                if (mismatches < 5)
                    std::cout << "# equiv mismatch at " << i << ": "
                              << int(ha[i]) << " vs " << int(hb[i]) << "\n";
                ++mismatches;
            }
        std::cout << "# equivalence: "
                  << (mismatches == 0 ? "EXACT" : "DIFFER")
                  << " (mismatches=" << mismatches << ")\n";

        // dpas-level check: identical A/B inputs through both feeds must
        // produce bit-identical s32 accumulators.
        std::vector<int8_t> av(8 * 32);
        for (auto &x : av) x = int8_t(d(rng));
        auto *ain = sycl::malloc_device<int8_t>(av.size(), q);
        auto *ca = sycl::malloc_device<int32_t>(128, q);
        auto *cb = sycl::malloc_device<int32_t>(128, q);
        q.memcpy(ain, av.data(), av.size()).wait();
        q.parallel_for<EquivName2>(sycl::range<1>(1),
                                  [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
                                      auto aa =
                                          xesimd::lsc_load_2d<int8_t, 32, 8>(
                                              ain, 31u, 7u, 31u, 0, 0);
                                      auto vt = xesimd::lsc_load_2d<
                                          int8_t, 16, 32, 1, false, true>(
                                          tin, 15u, 31u, 15u, 0, 0);
                                      auto vp =
                                          xesimd::lsc_load_2d<int8_t, 32, 16>(
                                              pin, 31u, 15u, 31u, 0, 0);
                                      auto rt = esimd::xmx::dpas<
                                          8, 8, int32_t, int32_t, int8_t,
                                          int8_t>(esimd::simd<int32_t, 128>(0),
                                                  vt, aa);
                                      auto rp = esimd::xmx::dpas<
                                          8, 8, int32_t, int32_t, int8_t,
                                          int8_t>(esimd::simd<int32_t, 128>(0),
                                                  vp, aa);
#pragma unroll
                                      for (int i = 0; i < 4; ++i) {
                                          xesimd::lsc_block_store<
                                              int32_t, 32>(
                                              ca + i * 32,
                                              rt.select<32, 1>(i * 32));
                                          xesimd::lsc_block_store<
                                              int32_t, 32>(
                                              cb + i * 32,
                                              rp.select<32, 1>(i * 32));
                                      }
                                  })
            .wait();
        std::vector<int32_t> ra(128), rb(128);
        q.memcpy(ra.data(), ca, 512).wait();
        q.memcpy(rb.data(), cb, 512).wait();
        int dmis = 0;
        for (int i = 0; i < 128; ++i)
            if (ra[i] != rb[i]) {
                if (dmis < 5)
                    std::cout << "# dpas mismatch at " << i << ": " << ra[i]
                              << " vs " << rb[i] << "\n";
                ++dmis;
            }
        // Host reference for the same single-tile product.
        long long hmax = 0;
        for (int m = 0; m < 8; ++m)
            for (int n = 0; n < 16; ++n) {
                int32_t acc = 0;
                for (int k = 0; k < 32; ++k)
                    acc += int32_t(av[m * 32 + k]) *
                           int32_t(tile[k * 16 + n]);
                const int want = ra[m * 16 + n];
                const long long dd =
                    std::abs((long long)want - (long long)acc);
                if (dd > hmax) hmax = dd;
            }
        std::cout << "# dpas feeds: "
                  << (dmis == 0 ? "IDENTICAL" : "DIFFER")
                  << " (mismatches=" << dmis
                  << "); vnni-vs-host max_diff=" << hmax << "\n";
        sycl::free(ain, q); sycl::free(ca, q); sycl::free(cb, q);

        // Bench-replica multi-chunk/multi-tile check (prepack feed).
        {
            constexpr int RC = 8, KC = 32;
            std::vector<int8_t> ma(size_t(MROWS) * MDK),
                mb_(size_t(MDK) * MCOLS);
            for (auto &x : ma) x = int8_t(d(rng));
            for (auto &x : mb_) x = int8_t(d(rng));
            const int chunks = MDK / KC, tiles = MCOLS / 16;
            std::vector<int8_t> mp(size_t(chunks) * tiles * (KC * 16), 0);
            for (int kc = 0; kc < chunks; ++kc)
                for (int t = 0; t < tiles; ++t)
                    for (int kk = 0; kk < KC; ++kk)
                        for (int n = 0; n < 16; ++n)
                            mp[(size_t(kc) * tiles + t) * (KC * 16) +
                               (kk >> 2) * 64 + (n >> 3) * 32 + (n & 7) * 4 +
                               (kk & 3)] =
                                mb_[(size_t(kc) * KC + kk) * MCOLS +
                                    (t * 16 + n)];
            auto *mad = sycl::malloc_device<int8_t>(ma.size(), q);
            auto *mbd = sycl::malloc_device<int8_t>(mp.size(), q);
            auto *mcd = sycl::malloc_device<int32_t>(size_t(MROWS) * MCOLS, q);
            q.memcpy(mad, ma.data(), ma.size()).wait();
            q.memcpy(mbd, mp.data(), mp.size()).wait();
            // threads = (MROWS/8) * (MCOLS/(4*16)) = 1 * 1 = 1... use 2
            // col-groups? MCOLS=64 with NT=4 -> one group. Launch 1 thread.
            mini_gemm(q, mad, mbd, mcd);
            std::vector<int32_t> mc(size_t(MROWS) * MCOLS);
            q.memcpy(mc.data(), mcd, mc.size() * 4).wait();
            long long mmax = 0;
            size_t badidx = SIZE_MAX;
            for (int r = 0; r < MROWS; ++r)
                for (int c = 0; c < MCOLS; ++c) {
                    int32_t acc = 0;
                    for (int k = 0; k < MDK; ++k)
                        acc += int32_t(ma[r * MDK + k]) *
                               int32_t(mb_[k * MCOLS + c]);
                    const long long dd =
                        std::abs((long long)mc[r * MCOLS + c] - (long long)acc);
                    if (dd > mmax) { mmax = dd; if (badidx == SIZE_MAX) badidx = size_t(r) * MCOLS + c; }
                }
            std::cout << "# mini prepack gemm: "
                      << (mmax == 0 ? "EXACT" : "WRONG")
                      << " (max_diff=" << mmax << " first_bad=" << badidx
                      << ")\n";
            sycl::free(mad, q); sycl::free(mbd, q); sycl::free(mcd, q);
        }
        sycl::free(tin, q); sycl::free(pin, q);
        sycl::free(ta, q); sycl::free(tb, q);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "dpas-vnni-layout: " << e.what() << "\n";
        return 1;
    }
}
